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

        sys.path.insert(0, str(tool_root))
        import onnx_splitpoint_tool.native_detection_postprocess as ndp
        from onnx_splitpoint_tool.runners.harness import yolo

        source_checks: list[dict[str, Any]] = []
        source_ok = True
        for artifact in (
            processor_contract.get("implementation_artifacts") or {}
        ).values():
            if not isinstance(artifact, Mapping):
                continue
            relative = str(artifact.get("relative_path") or "")
            path = tool_root / relative
            expected_sha = str(artifact.get("sha256") or "")
            actual_sha = sha256_file(path) if path.is_file() else ""
            passed = bool(path.is_file() and actual_sha == expected_sha)
            source_ok = source_ok and passed
            source_checks.append(
                {
                    "path": str(path),
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                    "passed": passed,
                }
            )
        report["source_checks"] = source_checks
        report["source_ok"] = source_ok
        if not source_ok:
            raise RuntimeError("frozen_oracle_source_mismatch")

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
