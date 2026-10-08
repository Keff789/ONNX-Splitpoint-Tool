#!/usr/bin/env python3
"""Integrated YOLOv7 Native three-stage performance canary.

One logical Native invocation is executed with three concurrent internal stages:

P1  image read + letterbox + Hailo-8 part 1
P2  boundary handoff + TensorRT part 2 + raw-head D2H
P3  exact sparse NumPy decode + class-aware NMS + inverse letterbox + records

The C++ runtime owns P1/P2 and a dedicated postprocessing worker thread. The
third thread calls the contract-bound NumPy implementation through a zero-copy
ctypes callback on pinned host output buffers. The frozen quality oracle,
cryptographic payload hashes and contract checks run only in a separate
postflight pass outside every performance timing window.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fast_decode_runtime import (
    ExactSparseYoloV7Decoder,
    canonical_oracle_dynamic,
    complete_fast_dynamic,
)

HERE = Path(__file__).resolve().parent
CPP_DIR = HERE / "cpp"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_oracle_sources(
    tool_root: Path, processor_contract: Mapping[str, Any],
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Use the normal frozen-contract verifier, retaining original source pins.

    The verifier already binds historical decoder versions to their original
    arithmetic. A newer source file can implement that same sealed contract;
    byte equality with the archived source is reported separately below.
    """
    sys.path.insert(0, str(tool_root))
    import onnx_splitpoint_tool.native_detection_postprocess as ndp
    from onnx_splitpoint_tool.runners.harness import yolo

    verified = ndp.verify_frozen_postprocess_contract(processor_contract)
    if verified != dict(processor_contract):
        raise RuntimeError("frozen_oracle_contract_projection_mismatch")
    current = ndp._implementation_artifacts()
    source_checks: list[dict[str, Any]] = []
    for role, artifact in verified["implementation_artifacts"].items():
        path = tool_root / str(artifact["relative_path"])
        actual_sha = sha256_file(path) if path.is_file() else ""
        expected_sha = str(artifact["sha256"])
        passed = bool(path.is_file() and actual_sha == current[role]["sha256"])
        source_checks.append({
            "role": role,
            "path": str(path),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "byte_identical": actual_sha == expected_sha,
            "verification": "verify_frozen_postprocess_contract",
            "passed": passed,
        })
    if not all(row["passed"] for row in source_checks):
        raise RuntimeError("frozen_oracle_source_mismatch")
    return ndp, yolo, source_checks


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(values: Sequence[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": statistics.fmean(data) if data else None,
        "median": percentile(data, 0.5),
        "p05": percentile(data, 0.05),
        "p95": percentile(data, 0.95),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def run_logged(
    command: list[str], log: Path, *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    log.write_text(
        "$ "
        + " ".join(command)
        + f"\n# elapsed_s={elapsed:.6f} rc={process.returncode}\n\n"
        + (process.stdout or ""),
        encoding="utf-8",
    )
    return process


def build_cpp(work: Path) -> Path:
    source = work / "source"
    build = work / "build"
    source.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CPP_DIR / "three_stage_canary.cpp", source / "three_stage_canary.cpp")
    shutil.copy2(CPP_DIR / "CMakeLists.txt", source / "CMakeLists.txt")
    configured = run_logged(
        ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
        work / "cmake_configure.log",
        cwd=work,
    )
    if configured.returncode != 0:
        raise RuntimeError("cmake_configure_failed")
    compiled = run_logged(
        ["cmake", "--build", str(build), "-j"],
        work / "cmake_build.log",
        cwd=work,
    )
    if compiled.returncode != 0:
        raise RuntimeError("cmake_build_failed")
    library = build / "libonnx_splitpoint_three_stage_canary.so"
    if not library.is_file():
        raise RuntimeError("three_stage_shared_library_missing")
    return library.resolve()


def choose_exact_artifact(
    spec: Mapping[str, Any]
) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates: list[Path] = []
    staged = str(spec.get("staged_file") or "")
    if staged:
        candidates.append(HERE / staged)
    preferred = str(spec.get("preferred_path") or "")
    if preferred:
        candidates.append(Path(preferred).expanduser())
    expected_sha = str(spec.get("sha256") or "")
    expected_size = int(spec.get("size_bytes") or 0)
    selected: Path | None = None
    audit: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except Exception:
            path = candidate
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        row: dict[str, Any] = {"path": key, "exists": path.is_file()}
        if path.is_file():
            actual_sha = sha256_file(path)
            actual_size = path.stat().st_size
            row.update(
                {
                    "sha256": actual_sha,
                    "size_bytes": actual_size,
                    "sha256_match": actual_sha == expected_sha,
                    "size_match": (not expected_size) or actual_size == expected_size,
                }
            )
            if row["sha256_match"] and row["size_match"] and selected is None:
                selected = path
                row["selected"] = True
        audit.append(row)
    return selected, audit


class ThreeStageConfigV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("hef_path", ctypes.c_char_p),
        ("engine_path", ctypes.c_char_p),
        ("images_dir", ctypes.c_char_p),
        ("out_json", ctypes.c_char_p),
        ("device_id", ctypes.c_char_p),
        ("frames", ctypes.c_int32),
        ("warmup", ctypes.c_int32),
        ("p1_queue_depth", ctypes.c_int32),
        ("post_queue_depth", ctypes.c_int32),
        ("letterbox_pad_value", ctypes.c_int32),
    ]


class ThreeStageConfigV2(ctypes.Structure):
    _fields_ = [
        ("base", ThreeStageConfigV1),
        ("prepared_input_path", ctypes.c_char_p),
        ("runtime_instance_id", ctypes.c_char_p),
        ("input_width", ctypes.c_int32), ("input_height", ctypes.c_int32),
        ("original_width", ctypes.c_int32), ("original_height", ctypes.c_int32),
        ("duration_s", ctypes.c_double),
        ("output_name8", ctypes.c_char_p), ("output_name16", ctypes.c_char_p),
        ("output_name32", ctypes.c_char_p), ("boundary_name", ctypes.c_char_p),
        ("trt_input_name", ctypes.c_char_p),
        ("boundary_height", ctypes.c_int32), ("boundary_width", ctypes.c_int32),
        ("boundary_channels", ctypes.c_int32),
        ("persist_sample", ctypes.c_int32),
        ("input_name", ctypes.c_char_p),
    ]


POST_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.c_void_p,
)


class CallbackContext:
    def __init__(
        self,
        *,
        decoder: ExactSparseYoloV7Decoder,
        processor_contract: Mapping[str, Any],
        yolo_module: Any,
        ndp_module: Any,
        harness: Any,
        corpus_items: list[dict[str, Any]],
        expected_rows: Mapping[int, Mapping[str, Any]],
        verification_mode: bool,
    ) -> None:
        self.decoder = decoder
        self.processor_contract = processor_contract
        self.yolo = yolo_module
        self.ndp = ndp_module
        self.harness = harness
        self.corpus_items = corpus_items
        self.expected_rows = expected_rows
        self.verification_mode = verification_mode
        self.errors: list[dict[str, Any]] = []
        self.first_measured_records: dict[int, list[dict[str, Any]]] = {}
        self.measured_seen_counts: dict[int, int] = {}
        self.first_candidate_counts: dict[int, list[dict[str, int]]] = {}
        self.verification_rows: list[dict[str, Any]] = []

    @staticmethod
    def _array(
        pointer: ctypes.POINTER(ctypes.c_float),
        elements: int,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        expected = int(np.prod(shape))
        if int(elements) != expected:
            raise RuntimeError(
                f"raw_head_element_count_mismatch:{elements}:{expected}:{shape}"
            )
        return np.ctypeslib.as_array(pointer, shape=(expected,)).reshape(shape)

    @staticmethod
    def _buffer_sha256(array: np.ndarray) -> str:
        if not array.flags.c_contiguous:
            raise RuntimeError("raw_head_not_contiguous")
        return hashlib.sha256(memoryview(array).cast("B")).hexdigest()

    def callback(
        self,
        measured: int,
        sequence: int,
        image_index: int,
        original_width: int,
        original_height: int,
        head8_pointer: ctypes.POINTER(ctypes.c_float),
        head8_elements: int,
        head16_pointer: ctypes.POINTER(ctypes.c_float),
        head16_elements: int,
        head32_pointer: ctypes.POINTER(ctypes.c_float),
        head32_elements: int,
        detection_count_pointer: ctypes.POINTER(ctypes.c_int32),
        _user_data: int,
    ) -> int:
        try:
            if not (0 <= int(image_index) < len(self.corpus_items)):
                raise RuntimeError(f"image_index_out_of_range:{image_index}")
            item = self.corpus_items[int(image_index)]
            if int(original_width) != int(item["width"]) or int(original_height) != int(
                item["height"]
            ):
                raise RuntimeError(
                    "image_geometry_mismatch:"
                    f"{image_index}:{original_width}x{original_height}:"
                    f"{item['width']}x{item['height']}"
                )

            outputs = {
                "yolov7_stride_8": self._array(
                    head8_pointer, int(head8_elements), (1, 3, 80, 80, 85)
                ),
                "yolov7_stride_16": self._array(
                    head16_pointer, int(head16_elements), (1, 3, 40, 40, 85)
                ),
                "yolov7_stride_32": self._array(
                    head32_pointer, int(head32_elements), (1, 3, 20, 20, 85)
                ),
            }
            records, candidate_counts = complete_fast_dynamic(
                self.decoder,
                outputs,
                original_wh=(int(original_width), int(original_height)),
                yolo_module=self.yolo,
                ndp_module=self.ndp,
                processor_contract=self.processor_contract,
            )
            detection_count_pointer[0] = len(records)

            if int(measured):
                self.measured_seen_counts[int(image_index)] = (
                    self.measured_seen_counts.get(int(image_index), 0) + 1
                )
                if int(image_index) not in self.first_measured_records:
                    self.first_measured_records[int(image_index)] = list(records)
                    self.first_candidate_counts[int(image_index)] = candidate_counts

            if self.verification_mode:
                expected = self.expected_rows[int(item["image_id"])]
                raw_expected_by_grid = {
                    int(record["shape"][2]): str(record["sha256"])
                    for record in expected.get("raw_outputs") or []
                }
                raw_actual = {
                    80: self._buffer_sha256(outputs["yolov7_stride_8"]),
                    40: self._buffer_sha256(outputs["yolov7_stride_16"]),
                    20: self._buffer_sha256(outputs["yolov7_stride_32"]),
                }
                raw_exact = raw_actual == raw_expected_by_grid
                oracle_records = canonical_oracle_dynamic(
                    self.harness,
                    outputs,
                    original_wh=(int(original_width), int(original_height)),
                    ndp_module=self.ndp,
                    processor_contract=self.processor_contract,
                )
                expected_records = list(expected.get("oracle_detections") or [])
                oracle_fast_exact = oracle_records == records
                prior_oracle_exact = oracle_records == expected_records
                final_exact = bool(raw_exact and oracle_fast_exact and prior_oracle_exact)
                self.verification_rows.append(
                    {
                        "sequence": int(sequence),
                        "image_index": int(image_index),
                        "image_id": int(item["image_id"]),
                        "raw_head_sha256_exact": raw_exact,
                        "raw_head_actual_sha256_by_grid": raw_actual,
                        "raw_head_expected_sha256_by_grid": raw_expected_by_grid,
                        "oracle_fast_exact": oracle_fast_exact,
                        "current_oracle_equals_prior_oracle": prior_oracle_exact,
                        "detection_count": len(records),
                        "exact": final_exact,
                    }
                )
                if not final_exact:
                    return 2
            return 0
        except Exception as error:  # ctypes callbacks must never leak exceptions.
            detection_count_pointer[0] = 0
            self.errors.append(
                {
                    "measured": bool(measured),
                    "sequence": int(sequence),
                    "image_index": int(image_index),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            return 1


class NativeLibrary:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.library = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        self.library.onnx_splitpoint_three_stage_config_size_v1.argtypes = []
        self.library.onnx_splitpoint_three_stage_config_size_v1.restype = ctypes.c_size_t
        self.library.onnx_splitpoint_run_three_stage_v1.argtypes = [
            ctypes.POINTER(ThreeStageConfigV1),
            POST_CALLBACK,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
        ]
        self.library.onnx_splitpoint_run_three_stage_v1.restype = ctypes.c_int
        native_size = int(
            self.library.onnx_splitpoint_three_stage_config_size_v1()
        )
        python_size = ctypes.sizeof(ThreeStageConfigV1)
        if native_size != python_size:
            raise RuntimeError(
                f"ctypes_config_size_mismatch:native={native_size}:python={python_size}"
            )

    def run(
        self,
        *,
        hef: Path,
        engine: Path,
        images_dir: Path,
        out_json: Path,
        context: CallbackContext,
        frames: int,
        warmup: int,
        p1_queue_depth: int,
        post_queue_depth: int,
        letterbox_pad_value: int,
        device_id: str = "",
    ) -> dict[str, Any]:
        # Keep encoded byte strings alive for the complete native call.
        strings = {
            "hef": os.fsencode(hef),
            "engine": os.fsencode(engine),
            "images": os.fsencode(images_dir),
            "out": os.fsencode(out_json),
            "device": device_id.encode("utf-8"),
        }
        config = ThreeStageConfigV1(
            abi_version=1,
            struct_size=ctypes.sizeof(ThreeStageConfigV1),
            hef_path=strings["hef"],
            engine_path=strings["engine"],
            images_dir=strings["images"],
            out_json=strings["out"],
            device_id=strings["device"],
            frames=int(frames),
            warmup=int(warmup),
            p1_queue_depth=int(p1_queue_depth),
            post_queue_depth=int(post_queue_depth),
            letterbox_pad_value=int(letterbox_pad_value),
        )
        callback = POST_CALLBACK(context.callback)
        error_buffer = ctypes.create_string_buffer(8192)
        started = time.time()
        return_code = int(
            self.library.onnx_splitpoint_run_three_stage_v1(
                ctypes.byref(config),
                callback,
                None,
                error_buffer,
                ctypes.sizeof(error_buffer),
            )
        )
        elapsed = time.time() - started
        error = error_buffer.value.decode("utf-8", errors="replace")
        runtime_report = (
            json.loads(out_json.read_text(encoding="utf-8"))
            if out_json.is_file()
            else {}
        )
        return {
            "return_code": return_code,
            "error": error,
            "elapsed_s": elapsed,
            "runtime_report_path": str(out_json),
            "runtime_report": runtime_report,
            "callback_errors": list(context.errors),
            "first_measured_records": dict(context.first_measured_records),
            "first_candidate_counts": dict(context.first_candidate_counts),
            "measured_seen_counts": dict(context.measured_seen_counts),
            "verification_rows": list(context.verification_rows),
        }


    def run_campaign(
        self, config: Mapping[str, Any], *, out_json: Path,
        context: "CampaignCallbackContext", runtime_instance_id: str,
        persist_sample: bool = True,
    ) -> dict[str, Any]:
        native = self.library.onnx_splitpoint_run_three_stage_v2
        native.argtypes = [ctypes.POINTER(ThreeStageConfigV2), POST_CALLBACK,
                           ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_size_t]
        native.restype = ctypes.c_int
        size = self.library.onnx_splitpoint_three_stage_config_size_v2
        size.argtypes, size.restype = [], ctypes.c_size_t
        if int(size()) != ctypes.sizeof(ThreeStageConfigV2):
            raise RuntimeError("campaign_ctypes_config_size_mismatch")
        boundary = config["boundary_contract"]
        prepared = config["prepared_input"]
        outputs = config["outputs"]
        # The structure owns its bytes for the entire call; no historical
        # expected-artifact lookup participates in the campaign path.
        base = ThreeStageConfigV1(
            abi_version=2, struct_size=ctypes.sizeof(ThreeStageConfigV2),
            hef_path=os.fsencode(config["hef"]["path"]),
            engine_path=os.fsencode(config["engine"]["path"]), images_dir=b"",
            out_json=os.fsencode(out_json), device_id=str(config.get("device_id") or "").encode(),
            frames=config["frames"], warmup=config["warmup"],
            p1_queue_depth=config["p1_queue_depth"], post_queue_depth=config["post_queue_depth"],
            letterbox_pad_value=114,
        )
        value = ThreeStageConfigV2(
            base=base, prepared_input_path=os.fsencode(prepared["path"]),
            runtime_instance_id=runtime_instance_id.encode(),
            input_width=prepared["shape"][1], input_height=prepared["shape"][0],
            original_width=config["original_wh"][0], original_height=config["original_wh"][1],
            duration_s=float(config.get("duration_s", 0.0)),
            output_name8=outputs[0]["name"].encode(), output_name16=outputs[1]["name"].encode(),
            output_name32=outputs[2]["name"].encode(), boundary_name=boundary["name"].encode(),
            trt_input_name=boundary["trt_input"]["name"].encode(),
            boundary_height=boundary["shape"][0], boundary_width=boundary["shape"][1],
            boundary_channels=boundary["shape"][2], persist_sample=int(persist_sample),
            input_name=prepared["name"].encode(),
        )
        callback = POST_CALLBACK(context.callback)
        error = ctypes.create_string_buffer(8192)
        code = int(native(ctypes.byref(value), callback, None, error, ctypes.sizeof(error)))
        report = json.loads(out_json.read_text()) if out_json.is_file() else {}
        if code or context.errors or report.get("ok") is not True:
            raise RuntimeError("campaign_runtime_failed:" + json.dumps({
                "return_code": code, "error": error.value.decode(errors="replace"),
                "callback_errors": context.errors, "runtime_report": str(out_json)}))
        return report


class CampaignCallbackContext:
    """Existing measured decoder, separate warmup state and untimed sentinel."""

    def __init__(self, config: Mapping[str, Any], sentinel: Mapping[str, Any]) -> None:
        from onnx_splitpoint_tool.native_three_stage import FastDetectionCompletionRuntime
        self.outputs = list(config["outputs"])
        self.original_wh = list(config["original_wh"])
        self.warmup = FastDetectionCompletionRuntime(config["completion_contract"])
        self.measured = FastDetectionCompletionRuntime(config["completion_contract"])
        self.measured.bind_sentinel_context(sentinel)
        self.errors: list[dict[str, Any]] = []

    def callback(self, measured, sequence, image_index, width, height,
                 head8, count8, head16, count16, head32, count32, detections, _userdata):
        try:
            if measured == -1:
                self.measured.snapshot_completed_source()
                return 0
            if measured not in (0, 1) or [width, height] != self.original_wh or image_index != 0:
                raise RuntimeError("campaign_callback_context_mismatch")
            runtime = self.measured if measured else self.warmup
            if sequence != runtime.completed_count:
                raise RuntimeError("campaign_callback_sequence_mismatch")
            arrays = {}
            for spec, ptr, count in zip(self.outputs, (head8, head16, head32), (count8, count16, count32)):
                if count != math.prod(spec["shape"]):
                    raise RuntimeError("campaign_callback_output_size_mismatch")
                arrays[spec["name"]] = np.ctypeslib.as_array(ptr, shape=(count,)).reshape(spec["shape"])
            result = runtime.process(arrays)
            detections[0] = int(result["detection_count"])
            return 0
        except Exception as error:
            self.errors.append({"measured": measured, "sequence": sequence, "error": str(error)})
            return 1


def _campaign_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    from onnx_splitpoint_tool.native_detection_postprocess import verify_detection_completion_execution_contract
    value = dict(config)
    contract = verify_detection_completion_execution_contract(value["completion_contract"])
    if contract != value["completion_contract"] or contract["processor_contract"] != value["processor_contract"]:
        raise RuntimeError("campaign_completion_contract_mismatch")
    if contract["model_family"] != "yolov7" or contract["completion_mode"] != "raw_head_frozen_decode_nms":
        raise RuntimeError("campaign_completion_mode_unsupported")
    for name in ("frames", "warmup", "repetitions", "p1_queue_depth", "post_queue_depth"):
        if type(value.get(name)) is not int or value[name] < (0 if name == "warmup" else 1):
            raise RuntimeError("campaign_budget_invalid:" + name)
    duration = value.get("duration_s", 0.0)
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        raise RuntimeError("campaign_duration_invalid")
    if duration == 0 and ((value["frames"], value["warmup"]) != (1000, 100) or value["repetitions"] not in (1, 3)):
        raise RuntimeError("campaign_performance_budget_mismatch")
    if duration and (duration != 60.0 or value["warmup"] != 0 or value["repetitions"] != 1):
        raise RuntimeError("campaign_energy_budget_mismatch")
    prepared = value["prepared_input"]
    if (not isinstance(prepared.get("name"), str) or not prepared["name"]
            or prepared.get("shape") != [640, 640, 3] or prepared.get("dtype") != "uint8"
            or prepared.get("format") not in {"raw", "raw_bin", "binary"}):
        raise RuntimeError("campaign_prepared_input_contract_mismatch")
    if value["original_wh"] != contract["processor_contract"]["original_wh"] or contract["processor_contract"]["input_hw"] != [640, 640]:
        raise RuntimeError("campaign_input_geometry_mismatch")
    signature = contract["source_endpoint"]["tensor_signature"]["tensors"]
    expected = [{"name": item["name"], "shape": item["shape"], "dtype": item["dtype"]} for item in signature]
    if value["outputs"] != expected or any(item["shape"] != [1, 3, grid, grid, 85] or item["dtype"] != "float32"
                                            for item, grid in zip(expected, (80, 40, 20))) or len(expected) != 3:
        raise RuntimeError("campaign_output_contract_mismatch")
    boundary = value["boundary_contract"]
    shape = boundary.get("shape")
    trt = boundary.get("trt_input", {})
    if (not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
            or boundary.get("dtype") != "uint8" or not boundary.get("name")
            or trt.get("dtype") != "uint8" or not trt.get("name")
            or trt.get("shape") != [1, shape[2], shape[0], shape[1]]):
        raise RuntimeError("campaign_boundary_contract_mismatch")
    return value


def _campaign_artifact(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"]).resolve(strict=True)
    if not path.is_file() or not spec.get("sha256") or sha256_file(path) != spec["sha256"]:
        raise RuntimeError("campaign_artifact_sha256_mismatch:" + str(path))
    if type(spec.get("size_bytes")) is not int or path.stat().st_size != spec["size_bytes"]:
        raise RuntimeError("campaign_artifact_size_mismatch:" + str(path))
    return {"path": str(path), "sha256": spec["sha256"], "size_bytes": path.stat().st_size}


def _campaign_repetition(config, runtime, context, *, index, runtime_id, repetition_id, out, energy=False):
    completed = runtime["completed_detection"]
    count = completed["completed_frames"]
    start, end, drain = (runtime[key] for key in ("measurement_start_ns", "measurement_end_ns", "drain_end_ns"))
    duration = float(config.get("duration_s", 0))
    if (type(count) is not int or count <= 0 or (not duration and count != config["frames"])
            or runtime["raw_model_outputs"]["completed_frames"] != count
            or runtime["measured_tasks_requested"] != count
            or context.measured.completed_count != count or context.warmup.completed_count != config["warmup"]
            or runtime["warmup_result"] != {"requested": config["warmup"], "raw_completed": config["warmup"],
                                           "completed": config["warmup"], "callback_failures": 0}
            or not 0 < start < end <= drain or not start < runtime["raw_end_ns"] <= end
            or runtime["workers_ready_before_start"] != 3 or runtime["preprocessing_included"] is not False
            or runtime["warmup_fully_drained_before_measurement"] is not True
            or runtime["runtime_instance_id"] != runtime_id
            or runtime["runtime_artifacts"] != {"hef": config["hef"]["path"], "engine": config["engine"]["path"]}
            or runtime["measurement_boundary"] != "prepared_input_before_p1_to_last_completed_task"):
        raise RuntimeError("campaign_runtime_evidence_mismatch")
    elapsed_ms = (end - start) / 1e6
    if (not math.isclose(completed["makespan_ms"], elapsed_ms, rel_tol=1e-8, abs_tol=1e-6)
            or not math.isclose(completed["throughput_fps"], count * 1000 / elapsed_ms, rel_tol=1e-8)
            or (duration and elapsed_ms < duration * 1000)):
        raise RuntimeError("campaign_runtime_time_mismatch")
    boundary = runtime["last_boundary"]
    trt = runtime["trt_input"]
    expected_boundary = config["boundary_contract"]
    if ({key: boundary[key] for key in ("name", "shape", "dtype")} != {key: expected_boundary[key] for key in ("name", "shape", "dtype")}
            or trt != expected_boundary["trt_input"]
            or {key: runtime["prepared_input"][key] for key in ("path", "shape", "dtype")} !=
               {key: config["prepared_input"][key] for key in ("path", "shape", "dtype")}
            or (config["prepared_input"].get("name") and runtime["prepared_input"]["name"] != config["prepared_input"]["name"])):
        raise RuntimeError("campaign_actual_runtime_interface_mismatch")
    specs = runtime["last_raw_outputs"]
    if [{key: spec[key] for key in ("name", "shape", "dtype")} for spec in specs] != config["outputs"]:
        raise RuntimeError("campaign_actual_runtime_outputs_mismatch")
    output_specs = {}
    boundary_spec = None
    arrays = {}
    for spec in ([] if energy else [boundary, *specs]):
        raw = Path(spec["path"])
        if raw.resolve().parent != out.resolve() or raw.stat().st_size != math.prod(spec["shape"]) * np.dtype(spec["dtype"]).itemsize:
            raise RuntimeError("campaign_measured_sample_size_or_path_mismatch")
        array = np.fromfile(raw, dtype=spec["dtype"]).reshape(spec["shape"])
        if not np.isfinite(array).all():
            raise RuntimeError("campaign_measured_sample_nonfinite")
        if spec is boundary:
            # The engine owns memory_nhwc_to_nchw. Its input receives the
            # unchanged HWC storage through memcpy, viewed in the TRT shape.
            array = array.reshape(trt["shape"])
        path = raw.with_suffix(".npy")
        np.save(path, array, allow_pickle=False)
        item = {**spec, "shape": list(array.shape), "path": str(path), "format": "npy", "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        if spec is boundary:
            boundary_spec = item
        else:
            arrays[spec["name"]] = array
            output_specs[spec["name"]] = item
    from onnx_splitpoint_tool.native_three_stage import verify_fast_completion_attestation
    attestation = context.measured.attestation(completed_work_units=count)
    verify_fast_completion_attestation(attestation, execution_contract=config["completion_contract"], outputs=None if energy else arrays)
    return {
        "ok": True, "frames": count, "completed_work_units": count, "produced_frames": count,
        "warmup_completed_frames": context.warmup.completed_count,
        "makespan_ms": elapsed_ms, "fps_makespan": count * 1000 / elapsed_ms,
        "measurement_start_ns": start, "measurement_end_ns": end, "drain_end_ns": drain,
        "runtime_instance_id": runtime_id, "repetition_id": repetition_id,
        "process_local_repetition_index": index, "process_id": os.getpid(),
        "completion_attestation": attestation, "output_arrays": output_specs,
        "boundary_array": boundary_spec,
        "runtime_boundary": {"output_name": boundary["name"], "output_shape": boundary["shape"],
            "output_dtype": boundary["dtype"], "output_frame_bytes": math.prod(boundary["shape"]),
            "trt_input_name": trt["name"], "trt_input_shape": trt["shape"], "trt_input_dtype": trt["dtype"],
            "input_name": runtime["prepared_input"]["name"], "input_shape": config["prepared_input"]["shape"],
            "input_dtype": "uint8"},
        "measurement_boundary": "workers_ready_to_last_completed_task_frame",
        "warmup_contract": "fully_drained_before_worker_start", "pipeline_drained": True, "preprocessing_included": False,
        "runtime_report": str(out / "runtime.json"), "request_latency": runtime["request_latency"],
    }


def run_campaign(config: Mapping[str, Any], *, out_root: Path) -> dict[str, Any]:
    """Bound performance campaign; preparation, checks and dumps are untimed."""
    value = _campaign_configuration(config)
    if value.get("duration_s", 0):
        raise RuntimeError("campaign_energy_requires_attested_replay_entry")
    out = Path(out_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    bound = {name: _campaign_artifact(value[name]) for name in ("hef", "engine", "prepared_input", "input_image")}
    if bound["prepared_input"]["size_bytes"] != math.prod(value["prepared_input"]["shape"]):
        raise RuntimeError("campaign_prepared_input_size_mismatch")
    for name in bound:
        value[name] = {**value[name], **bound[name]}
    tool_root = Path(config.get("tool_root") or HERE.parents[2])
    _verified_oracle_sources(tool_root, value["processor_contract"])
    if value.get("native_executable"):
        existing_library = _campaign_artifact(value["native_executable"])
        library_path = Path(existing_library["path"])
    elif value.get("build", True):
        existing_library = None
        library_path = build_cpp(out / "cpp_runtime")
    else:
        raise RuntimeError("campaign_no_build_requires_bound_native_executable")
    artifacts = {name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                 for name, path in {"native_executable": library_path, "generated_cpp": CPP_DIR / "three_stage_canary.cpp",
                                    "cmake": CPP_DIR / "CMakeLists.txt"}.items()}
    if config.get("expected_library_sha256") and artifacts["native_executable"]["sha256"] != config["expected_library_sha256"]:
        raise RuntimeError("campaign_expected_library_sha256_mismatch")
    library = NativeLibrary(library_path)
    report = {"ok": False, "config": dict(config), "schema_version": 1, "mode": "bound_prepared_input_concurrent_three_stage",
              "completion_contract": value["completion_contract"], "runtime_artifacts": artifacts,
              "artifacts": bound, "repetitions": [], "status": "running"}
    report_path = out / "campaign_report.json"
    for index in range(1, value["repetitions"] + 1):
        rep_dir = out / f"rep_{index:02d}"
        rep_dir.mkdir()
        runtime_id, repetition_id = uuid.uuid4().hex, uuid.uuid4().hex
        context = CampaignCallbackContext(value, {"runtime_instance_id": runtime_id, "repetition_id": repetition_id,
            "process_local_repetition_index": index, "input_image_sha256": value["input_image"]["sha256"]})
        print(f"[three-stage campaign] repetition {index}/{value['repetitions']} warmup=100 tasks=1000", flush=True)
        runtime = library.run_campaign(value, out_json=rep_dir / "runtime.json", context=context, runtime_instance_id=runtime_id)
        rep = _campaign_repetition(value, runtime, context, index=index, runtime_id=runtime_id,
                                  repetition_id=repetition_id, out=rep_dir)
        for name in bound:
            if _campaign_artifact(value[name]) != bound[name]:
                raise RuntimeError("campaign_artifact_changed_during_measurement:" + name)
        if existing_library is not None and _campaign_artifact(value["native_executable"]) != existing_library:
            raise RuntimeError("campaign_native_executable_changed_during_measurement")
        report["repetitions"].append(rep)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[three-stage campaign] repetition {index}/{value['repetitions']} completed={rep['completed_work_units']}", flush=True)
    report["status"] = "completed"
    report["ok"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report



def run_energy_campaign(config: Mapping[str, Any], *, out_root: Path,
                        verified_workload_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the normal preflight binding using its already built library.

    The Parent has checked nonce, freshness and command identity with
    load_split_energy_workload_binding. Repeat its stat guard here, without
    model hashing, model discovery, runtime compilation or tensor dump I/O.
    The existing untimed completion sentinel remains required.
    """
    from onnx_splitpoint_tool.native_command_contract import verify_split_energy_artifact_stats
    binding = verified_workload_binding
    okay, reason = verify_split_energy_artifact_stats(binding)
    if not okay:
        raise RuntimeError("campaign_energy_preflight:" + reason)
    value = _campaign_configuration(config)
    artifacts = binding["artifacts"]
    options = binding["runtime_options"]
    prepared = binding["prepared_input_contract"]
    boundary = binding["runtime_boundary_evidence"]
    if (value.get("energy_workload_only") is not True or float(value["duration_s"]) != 60.0
            or binding.get("backend") != "hailo8_to_trt" or binding.get("workload_supported") is not True
            or value["identity"] != {"model_id": binding["model"], "case_id": binding["case"],
                "setup_id": binding["setup_id"], "precision": binding["precision"],
                "eval_run_id": binding.get("eval_run_id"), "source_run_id": binding.get("source_run_id")}
            or value.get("quality_binding_sha256") != binding.get("native_split_quality_binding_sha256")
            or value["completion_contract"] != options["completion_execution_contract"]
            or value["p1_queue_depth"] != options["queue_depth"]
            or value["post_queue_depth"] != options["post_queue_depth"]
            or value.get("device_id", "") != str(options.get("device_id") or "")
            or value["input_image"]["path"] != binding["input_image"]
            or value["input_image"]["sha256"] != binding["input_image_sha256"]
            or value["prepared_input"]["shape"] != prepared["shape"]
            or value["prepared_input"]["dtype"] != prepared["dtype"]
            or value["prepared_input"]["name"] != prepared["name"]
            or value["boundary_contract"] != {"name": boundary["output_name"], "shape": boundary["output_shape"],
                "dtype": boundary["output_dtype"], "trt_input": {"name": boundary["trt_input_name"],
                    "shape": boundary["trt_input_shape"], "dtype": boundary["trt_input_dtype"]}}):
        raise RuntimeError("campaign_energy_workload_binding_mismatch")
    for role in ("hef", "engine", "prepared_input"):
        if any(value[role].get(key) != artifacts[role].get(key) for key in ("path", "sha256", "size_bytes")):
            raise RuntimeError("campaign_energy_artifact_binding_mismatch:" + role)
    library_spec = artifacts["native_executable"]
    stat_paths = {str(Path(item["path"]).resolve()) for item in binding["verified_files"]}
    for role in ("hef", "engine", "prepared_input", "native_executable"):
        if str(Path(artifacts[role]["path"]).resolve()) not in stat_paths:
            raise RuntimeError("campaign_energy_artifact_stat_missing:" + role)
    out = Path(out_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime_id, repetition_id = uuid.uuid4().hex, uuid.uuid4().hex
    context = CampaignCallbackContext(value, {"runtime_instance_id": runtime_id, "repetition_id": repetition_id,
        "process_local_repetition_index": 1, "input_image_sha256": value["input_image"]["sha256"]})
    library = NativeLibrary(Path(library_spec["path"]))
    runtime = library.run_campaign(value, out_json=out / "runtime.json", context=context,
                                   runtime_instance_id=runtime_id, persist_sample=False)
    rep = _campaign_repetition(value, runtime, context, index=1, runtime_id=runtime_id,
                              repetition_id=repetition_id, out=out, energy=True)
    return {"ok": True, "config": dict(config), "schema_version": 1,
            "mode": "bound_prepared_input_concurrent_three_stage", "status": "completed",
            "completion_contract": value["completion_contract"], "repetitions": [rep],
            "runtime_artifacts": {"native_executable": dict(library_spec)}}


def exact_measurement_result_check(
    result: Mapping[str, Any],
    corpus_items: list[dict[str, Any]],
    expected_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    first_records = result.get("first_measured_records") or {}
    rows: list[dict[str, Any]] = []
    exact_count = 0
    for item in corpus_items:
        image_index = int(item["index"])
        image_id = int(item["image_id"])
        actual = first_records.get(image_index)
        expected = list(expected_rows[image_id].get("oracle_detections") or [])
        exact = actual == expected
        exact_count += int(exact)
        rows.append(
            {
                "image_index": image_index,
                "image_id": image_id,
                "seen_count": int(
                    (result.get("measured_seen_counts") or {}).get(image_index, 0)
                ),
                "exact": exact,
                "actual_detection_count": len(actual or []),
                "expected_detection_count": len(expected),
            }
        )
    return {
        "exact_images": exact_count,
        "requested_images": len(corpus_items),
        "all_exact": exact_count == len(corpus_items),
        "rows": rows,
    }


def extract_runtime_metrics(runtime: Mapping[str, Any]) -> dict[str, Any]:
    raw = runtime.get("raw_model_outputs") or {}
    completed = (
        runtime.get("completed_detection")
        or runtime.get("device_completed_detection")
        or {}
    )
    stages = runtime.get("stage_metrics") or {}
    post = ((stages.get("postprocessing") or {}).get(
        "fast_decode_nms_inverse_letterbox_records"
    ) or {})
    return {
        "request_latency": runtime.get("request_latency") or {},
        "raw_fps": float(raw.get("throughput_fps") or 0.0),
        "completed_fps": float(completed.get("throughput_fps") or 0.0),
        "completed_raw_ratio": float(completed.get("raw_ratio") or 0.0),
        "p1_stage_mean_ms": float(
            (((stages.get("P1") or {}).get("stage") or {}).get("mean_ms") or 0.0)
        ),
        "p2_stage_mean_ms": float(
            (((stages.get("P2") or {}).get("stage") or {}).get("mean_ms") or 0.0)
        ),
        "post_mean_ms": float(post.get("mean_ms") or 0.0),
        "post_median_ms": float(post.get("median_ms") or 0.0),
        "post_p95_ms": float(post.get("p95_ms") or 0.0),
        "post_max_ms": float(post.get("max_ms") or 0.0),
        "theoretical_raw_fps": float(
            runtime.get("theoretical_raw_fps") or 0.0
        ),
        "theoretical_three_stage_fps": float(
            runtime.get("theoretical_three_stage_fps") or 0.0
        ),
        "callback_failures": int(runtime.get("callback_failures") or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-config")
    parser.add_argument("--tool-root", default="/home/nx/ONNX-Splitpoint-Tool")
    parser.add_argument(
        "--corpus", default=str(HERE / "staged_inputs/corpus_manifest.json")
    )
    parser.add_argument(
        "--expected", default=str(HERE / "expected_artifacts.json")
    )
    parser.add_argument(
        "--reference-report",
        default=str(HERE / "reference/multi_image_fast_decode_canary_report.json"),
    )
    parser.add_argument(
        "--out-root", default="/home/nx/native_yolov7_three_stage_results"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--p1-queue-depth", type=int, default=3)
    parser.add_argument("--post-queue-depth", type=int, default=4)
    parser.add_argument("--device-id", default="")
    parser.add_argument("--expected-corpus-count", type=int, default=32)
    args = parser.parse_args()

    if args.campaign_config:
        config = json.loads(Path(args.campaign_config).read_text())
        run_campaign(config, out_root=Path(args.out_root))
        return 0

    tool_root = Path(args.tool_root).expanduser().resolve()
    corpus_path = Path(args.corpus).expanduser().resolve()
    expected_path = Path(args.expected).expanduser().resolve()
    reference_report_path = Path(args.reference_report).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = out_root / f"yolov7_three_stage_v1_{stamp}"
    out.mkdir(parents=True, exist_ok=False)

    report: dict[str, Any] = {
        "schema": "onnx-splitpoint/yolov7-native-three-stage-canary",
        "schema_version": 1,
        "status": "RUNNING",
        "scope": "current_v2782_b066_three_stage_zero_copy_fast_numpy",
        "host": {
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
            "process_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else [],
        },
        "output_dir": str(out),
        "quality_oracle_outside_performance_timing": True,
        "performance_hotloop_contains_crypto_hashing": False,
    }

    exit_code = 2
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        reference_report = json.loads(
            reference_report_path.read_text(encoding="utf-8")
        )
        corpus_items = corpus.get("items")
        expected_corpus_count = int(args.expected_corpus_count)
        if expected_corpus_count < 1:
            raise RuntimeError("expected_corpus_count_must_be_positive")
        if not isinstance(corpus_items, list) or len(corpus_items) != expected_corpus_count:
            raise RuntimeError(
                f"exact_corpus_count_required:{expected_corpus_count}:"
                f"observed={len(corpus_items) if isinstance(corpus_items, list) else 'invalid'}"
            )
        reference_rows = list(reference_report.get("rows") or [])
        expected_rows = {int(row["image_id"]): row for row in reference_rows}
        if len(expected_rows) != expected_corpus_count:
            raise RuntimeError(
                f"reference_report_row_count_required:{expected_corpus_count}:"
                f"observed={len(expected_rows)}"
            )
        corpus_ids = [int(item["image_id"]) for item in corpus_items]
        reference_ids = [int(row["image_id"]) for row in reference_rows]
        if corpus_ids != reference_ids:
            raise RuntimeError(
                "corpus_reference_order_mismatch:"
                f"corpus={corpus_ids}:reference={reference_ids}"
            )

        reference_metadata_checks: list[dict[str, Any]] = []
        for role, spec in (expected.get("reference_metadata") or {}).items():
            relative = str((spec or {}).get("file") or "")
            expected_sha = str((spec or {}).get("sha256") or "")
            path = (HERE / relative).resolve()
            actual_sha = sha256_file(path) if path.is_file() else ""
            passed = bool(path.is_file() and actual_sha == expected_sha)
            reference_metadata_checks.append(
                {
                    "role": str(role),
                    "path": str(path),
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                    "passed": passed,
                }
            )
        report["reference_metadata_checks"] = reference_metadata_checks
        if not all(row["passed"] for row in reference_metadata_checks):
            raise RuntimeError("frozen_reference_metadata_mismatch")

        current_spec = expected["reference_metadata"]["current_result"]
        current_result_path = (HERE / str(current_spec["file"])).resolve()
        if sha256_file(current_result_path) != str(current_spec["sha256"]):
            raise RuntimeError("current_result_reference_sha256_mismatch")
        current_result = json.loads(current_result_path.read_text(encoding="utf-8"))
        execution_contract = current_result.get("completion_execution_contract")
        if not isinstance(execution_contract, Mapping):
            raise RuntimeError("completion_execution_contract_missing")
        processor_contract = execution_contract.get("processor_contract")
        if not isinstance(processor_contract, Mapping):
            raise RuntimeError("processor_contract_missing")

        ndp, yolo, source_checks = _verified_oracle_sources(
            tool_root, processor_contract,
        )
        report["source_checks"] = source_checks
        report["source_ok"] = all(row["passed"] for row in source_checks)

        runtime_artifacts = expected["runtime_artifacts"]
        hef, hef_audit = choose_exact_artifact(runtime_artifacts["hef"])
        engine, engine_audit = choose_exact_artifact(runtime_artifacts["engine"])
        report["artifact_recovery"] = {
            "hef": hef_audit,
            "engine": engine_audit,
        }
        if hef is None or engine is None:
            raise RuntimeError("exact_hef_or_engine_missing")
        before_hashes = {"hef": sha256_file(hef), "engine": sha256_file(engine)}

        for item in corpus_items:
            image = (corpus_path.parent / str(item["staged_file"])).resolve()
            if not image.is_file():
                raise RuntimeError(f"staged_image_missing:{image}")
            if sha256_file(image) != str(item["staged_sha256"]):
                raise RuntimeError(f"staged_image_sha256_mismatch:{image}")
        images_dir = corpus_path.parent / "images"

        library_path = build_cpp(out / "cpp_build")
        report["native_library"] = {
            "path": str(library_path),
            "sha256": sha256_file(library_path),
            "source_sha256": sha256_file(CPP_DIR / "three_stage_canary.cpp"),
        }
        native = NativeLibrary(library_path)

        decoder = ExactSparseYoloV7Decoder(processor_contract)
        harness = yolo.YoloHarness(
            conf_thresh=float(processor_contract["confidence_threshold"]),
            iou_thresh=float(processor_contract["iou_threshold"]),
            max_det=int(processor_contract["max_detections"]),
            multiscale_activation_mode=processor_contract.get(
                "multiscale_activation_mode"
            ),
            model_id=str(processor_contract.get("model_id") or ""),
            multiscale_decoder_contract=processor_contract.get(
                "model_bound_decoder_contract"
            ),
        )

        repetitions: list[dict[str, Any]] = []
        for repetition in range(int(args.repetitions)):
            repetition_dir = out / "repetitions" / f"rep_{repetition + 1:02d}"
            repetition_dir.mkdir(parents=True, exist_ok=True)
            context = CallbackContext(
                decoder=decoder,
                processor_contract=processor_contract,
                yolo_module=yolo,
                ndp_module=ndp,
                harness=harness,
                corpus_items=corpus_items,
                expected_rows=expected_rows,
                verification_mode=False,
            )
            run_result = native.run(
                hef=hef,
                engine=engine,
                images_dir=images_dir,
                out_json=repetition_dir / "native_three_stage_runtime.json",
                context=context,
                frames=int(args.frames),
                warmup=int(args.warmup),
                p1_queue_depth=int(args.p1_queue_depth),
                post_queue_depth=int(args.post_queue_depth),
                letterbox_pad_value=114,
                device_id=str(args.device_id),
            )
            parity = exact_measurement_result_check(
                run_result, corpus_items, expected_rows
            )
            metrics = extract_runtime_metrics(run_result["runtime_report"])
            compact = {
                "repetition": repetition + 1,
                "return_code": run_result["return_code"],
                "error": run_result["error"],
                "elapsed_s": run_result["elapsed_s"],
                "runtime_report_path": run_result["runtime_report_path"],
                "runtime_ok": bool(
                    (run_result["runtime_report"] or {}).get("ok")
                ),
                "metrics": metrics,
                "measurement_result_parity": parity,
                "callback_errors": run_result["callback_errors"],
                "measured_seen_counts": run_result["measured_seen_counts"],
                "first_candidate_counts": run_result["first_candidate_counts"],
            }
            (repetition_dir / "repetition_summary.json").write_text(
                json.dumps(compact, indent=2), encoding="utf-8"
            )
            repetitions.append(compact)

        verification_dir = out / "postflight_quality_oracle"
        verification_dir.mkdir(parents=True, exist_ok=True)
        verification_context = CallbackContext(
            decoder=decoder,
            processor_contract=processor_contract,
            yolo_module=yolo,
            ndp_module=ndp,
            harness=harness,
            corpus_items=corpus_items,
            expected_rows=expected_rows,
            verification_mode=True,
        )
        verification_run = native.run(
            hef=hef,
            engine=engine,
            images_dir=images_dir,
            out_json=verification_dir / "verification_runtime_not_performance.json",
            context=verification_context,
            frames=expected_corpus_count,
            warmup=0,
            p1_queue_depth=int(args.p1_queue_depth),
            post_queue_depth=int(args.post_queue_depth),
            letterbox_pad_value=114,
            device_id=str(args.device_id),
        )
        verification_rows = verification_run["verification_rows"]
        verification_exact = sum(
            1 for row in verification_rows if row.get("exact")
        )
        postflight = {
            "performance_interpretation": "forbidden_verification_only",
            "return_code": verification_run["return_code"],
            "error": verification_run["error"],
            "callback_errors": verification_run["callback_errors"],
            "requested_images": expected_corpus_count,
            "verified_images": len(verification_rows),
            "exact_images": verification_exact,
            "all_exact": verification_exact == expected_corpus_count and len(verification_rows) == expected_corpus_count,
            "rows": verification_rows,
        }
        (verification_dir / "postflight_quality_oracle_report.json").write_text(
            json.dumps(postflight, indent=2), encoding="utf-8"
        )

        after_hashes = {"hef": sha256_file(hef), "engine": sha256_file(engine)}
        source_mutated = before_hashes != after_hashes
        raw_fps_values = [row["metrics"]["raw_fps"] for row in repetitions]
        completed_fps_values = [
            row["metrics"]["completed_fps"] for row in repetitions
        ]
        ratio_values = [
            row["metrics"]["completed_raw_ratio"] for row in repetitions
        ]
        post_p95_values = [row["metrics"]["post_p95_ms"] for row in repetitions]
        post_max_values = [row["metrics"]["post_max_ms"] for row in repetitions]
        runtime_complete = all(
            row["return_code"] == 0
            and row["runtime_ok"]
            and row["metrics"]["callback_failures"] == 0
            for row in repetitions
        )
        measurement_parity = all(
            row["measurement_result_parity"]["all_exact"]
            for row in repetitions
        )
        raw_median = float(statistics.median(raw_fps_values))
        completed_median = float(statistics.median(completed_fps_values))
        ratio_median = float(statistics.median(ratio_values))
        post_p95_guard = float(max(post_p95_values))
        post_max_guard = float(max(post_max_values))
        target_met = bool(
            runtime_complete
            and measurement_parity
            and postflight["all_exact"]
            and raw_median >= 90.0
            and completed_median >= 90.0
            and ratio_median >= 0.90
            and post_p95_guard <= 10.0
            and post_max_guard <= 20.0
        )

        if source_mutated:
            status = "FAIL_SOURCE_OR_RUNTIME_ARTIFACT_MUTATED"
        elif not runtime_complete:
            status = "FAIL_THREE_STAGE_RUNTIME"
        elif not measurement_parity:
            status = "FAIL_MEASUREMENT_RESULT_PARITY"
        elif not postflight["all_exact"]:
            status = "FAIL_POSTFLIGHT_QUALITY_ORACLE"
        elif target_met:
            status = "PASS_THREE_STAGE_TARGET_MET"
        else:
            status = "PASS_THREE_STAGE_EXACT_SPEED_OPEN"

        aggregate = {
            "repetitions_requested": int(args.repetitions),
            "repetitions_valid": sum(
                1
                for row in repetitions
                if row["return_code"] == 0 and row["runtime_ok"]
            ),
            "raw_fps": summarize(raw_fps_values),
            "completed_detection_fps": summarize(
                completed_fps_values
            ),
            "completed_raw_ratio": summarize(ratio_values),
            "post_p95_ms_across_repetitions": summarize(post_p95_values),
            "post_max_ms_across_repetitions": summarize(post_max_values),
            "raw_fps_median": raw_median,
            "completed_fps_median": completed_median,
            "completed_raw_ratio_median": ratio_median,
            "post_p95_guard_ms": post_p95_guard,
            "post_max_guard_ms": post_max_guard,
        }
        report.update(
            {
                "status": status,
                "corpus": corpus,
                "runtime_artifacts": {
                    "hef": str(hef),
                    "hef_sha256": after_hashes["hef"],
                    "engine": str(engine),
                    "engine_sha256": after_hashes["engine"],
                    "source_mutated": source_mutated,
                },
                "measurement_contract": {
                    "repetitions": int(args.repetitions),
                    "frames_per_repetition": int(args.frames),
                    "warmup_frames_fully_drained": int(args.warmup),
                    "p1_queue_depth": int(args.p1_queue_depth),
                    "post_queue_depth": int(args.post_queue_depth),
                    "images_cycled": expected_corpus_count,
                    "P1": "image_read_letterbox_hailo8_part1",
                    "P2": "boundary_handoff_tensorrt_part2_raw_heads_d2h",
                    "raw_endpoint_semantics": "observed_inside_the_coupled_three_stage_pipeline_with_queue_backpressure_reported_separately",
                    "raw_stage_theoretical_semantics": "1000_over_max_mean_P1_stage_mean_P2_stage",
                    "postprocessing": "exact_sparse_numpy_decode_classaware_nms_inverse_letterbox_canonical_records",
                    "postprocess_location": "orin_nx_host_cpu_numpy_inprocess_callback",
                    "completed_endpoint": "completed_detection",
                    "quality_oracle": f"separate_{expected_corpus_count}_image_postflight_outside_timing",
                },
                "repetitions": repetitions,
                "postflight_quality_oracle": postflight,
                "aggregate": aggregate,
                "decision": {
                    "raw_fps_min": 90.0,
                    "completed_detection_fps_min": 90.0,
                    "completed_raw_ratio_min": 0.90,
                    "post_p95_guard_ms": 10.0,
                    "post_max_guard_ms": 20.0,
                    "target_met": target_met,
                    "next_step": (
                        "integrate_contract_bound_three_stage_native_runner_once_in_v2_79"
                        if status == "PASS_THREE_STAGE_TARGET_MET"
                        else "inspect_stage_and_queue_metrics_before_any_product_integration"
                    ),
                },
            }
        )
        exit_code = 0 if status.startswith("PASS") else 2
    except Exception as error:
        report["status"] = "FAIL_CANARY_INFRASTRUCTURE"
        report["error"] = f"{type(error).__name__}: {error}"
        exit_code = 2

    report_path = out / "three_stage_canary_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    aggregate = report.get("aggregate") or {}
    repetitions = report.get("repetitions") or []
    lines = [
        "# YOLOv7 Native Three-Stage Canary",
        "",
        f"- Status: **{report.get('status')}**",
        "- Topology: **P1 → P2 → Postprocessing**",
        "- Quality oracle: **outside all performance timing windows**",
        f"- Raw endpoint median: **{float(aggregate.get('raw_fps_median') or 0.0):.3f} FPS**",
        f"- Device-completed detection median: **{float(aggregate.get('completed_fps_median') or 0.0):.3f} FPS**",
        f"- Completed/raw ratio: **{float(aggregate.get('completed_raw_ratio_median') or 0.0):.4f}**",
        f"- Postprocessing P95 guard: **{float(aggregate.get('post_p95_guard_ms') or 0.0):.3f} ms**",
        f"- Postflight oracle parity: **{int((report.get('postflight_quality_oracle') or {}).get('exact_images') or 0)}/{int((report.get('postflight_quality_oracle') or {}).get('requested_images') or 0)} exact**",
        "",
        "| Rep | P1 mean ms | P2 mean ms | Post mean ms | Post P95 ms | Raw FPS observed | Raw FPS stage-theoretical | Completed FPS | Ratio | corpus result parity |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in repetitions:
        metrics = row.get("metrics") or {}
        parity = row.get("measurement_result_parity") or {}
        lines.append(
            f"| {row.get('repetition')} | {float(metrics.get('p1_stage_mean_ms') or 0.0):.3f} | "
            f"{float(metrics.get('p2_stage_mean_ms') or 0.0):.3f} | "
            f"{float(metrics.get('post_mean_ms') or 0.0):.3f} | "
            f"{float(metrics.get('post_p95_ms') or 0.0):.3f} | "
            f"{float(metrics.get('raw_fps') or 0.0):.3f} | "
            f"{float(metrics.get('theoretical_raw_fps') or 0.0):.3f} | "
            f"{float(metrics.get('completed_fps') or 0.0):.3f} | "
            f"{float(metrics.get('completed_raw_ratio') or 0.0):.4f} | "
            f"{int(parity.get('exact_images') or 0)}/{int(parity.get('requested_images') or 0)} |"
        )
    lines += [
        "",
        "The performance callback performs no SHA-256 hashing, JSON evidence generation or slow-oracle execution.",
        "A separate postflight over the configured sentinel corpus replays the same runtime artifacts and requires exact raw-head hashes, exact current-oracle parity and equality to the prior validated oracle results.",
        "No HEF or TensorRT engine is built, no B500 or energy run is executed, and the installed tool is not modified.",
    ]
    markdown_path = out / "three_stage_canary_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": report.get("status"),
                "report": str(report_path),
                "markdown": str(markdown_path),
                "raw_fps_median": aggregate.get("raw_fps_median"),
                "completed_fps_median": aggregate.get("completed_fps_median"),
                "completed_raw_ratio_median": aggregate.get(
                    "completed_raw_ratio_median"
                ),
                "post_p95_guard_ms": aggregate.get("post_p95_guard_ms"),
                "postflight_exact": (
                    report.get("postflight_quality_oracle") or {}
                ).get("exact_images"),
                "next": (report.get("decision") or {}).get("next_step", ""),
            },
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
