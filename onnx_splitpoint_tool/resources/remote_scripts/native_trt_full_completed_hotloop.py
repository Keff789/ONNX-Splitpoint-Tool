#!/usr/bin/env python3
"""Measure TensorRT Native-Full through the completed detection endpoint.

The prepared runtime tensor and quality-sealed engine are loaded before the
measured loop.  Every work unit includes either frozen raw-head decode/NMS or
frozen Direct-BN6 filtering/NMS/inverse-letterbox normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from onnx_splitpoint_tool.native_detection_postprocess import (  # noqa: E402
    FrozenDecodedNmsPostprocessor,
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_normalized_detection_endpoint_attestation,
    persist_completed_result_artifact as _persist_shared_completed_result_artifact,
    verify_frozen_decoded_nms_normalization_contract,
    verify_frozen_postprocess_contract,
    verify_letterbox_geometry_contract,
)
from native_hailo10_trt_e2e_from_benchmarkset import NativeTRT  # noqa: E402


INPUT_SCHEMA = "onnx-splitpoint/native-full-input-dump"
PREFLIGHT_SCHEMA = "onnx-splitpoint/energy-preflight-attestation"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _strict_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return text if len(text) == 64 and all(c in "0123456789abcdef" for c in text) else ""


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _completed_detection_artifact(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    embedded = result.get("completed_result_artifact")
    if isinstance(embedded, Mapping) and embedded:
        return dict(embedded)
    raw_detections = result.get("detections")
    if not isinstance(raw_detections, list):
        raise RuntimeError("completed result detections are unavailable")
    detections: list[dict[str, Any]] = []
    for raw in raw_detections:
        if not isinstance(raw, Mapping):
            raise RuntimeError("completed result detection is invalid")
        try:
            class_value = float(raw["class_id"])
            detection = {
                "class_id": int(class_value),
                "score": float(raw["score"]),
                "x1": float(raw["x1"]),
                "y1": float(raw["y1"]),
                "x2": float(raw["x2"]),
                "y2": float(raw["y2"]),
            }
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("completed result detection is invalid") from exc
        if (
            isinstance(raw.get("class_id"), bool)
            or class_value != detection["class_id"]
            or any(
                not math.isfinite(float(detection[field]))
                for field in ("score", "x1", "y1", "x2", "y2")
            )
            or not 0.0 <= detection["score"] <= 1.0
            or detection["x2"] < detection["x1"]
            or detection["y2"] < detection["y1"]
        ):
            raise RuntimeError("completed result detection is invalid")
        detections.append(detection)
    detections.sort(key=lambda value: (
        -float(value["score"]),
        int(value["class_id"]),
        float(value["x1"]),
        float(value["y1"]),
        float(value["x2"]),
        float(value["y2"]),
    ))
    return {
        "schema": "onnx-splitpoint/frozen-completed-detection-result-artifact",
        "schema_version": 1,
        "record_schema": "xyxy_score_class_id_v1",
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": "score_desc_class_id_asc_xyxy_lexicographic_v1",
        "detections": detections,
    }


def _persist_completed_result_artifact(
    artifact: Mapping[str, Any],
    expected_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    if str(expected_sha256 or "").strip().lower() != (
        _canonical_json_sha256(dict(artifact))
    ):
        raise RuntimeError(
            "completed result artifact canonical SHA-256 mismatch"
        )
    evidence = _persist_shared_completed_result_artifact(
        artifact,
        expected_sha256=expected_sha256,
        output_path=output_path,
    )
    return {
        "saved": True,
        "path": str(evidence["completed_task_result_artifact_path"]),
        "file_sha256": str(
            evidence["completed_task_result_artifact_file_sha256"]
        ),
    }


def _fail(report: Path, status: str, **extra: Any) -> int:
    payload = {"ok": False, "status": status, **extra}
    _write_report(report, payload)
    print(json.dumps(payload), file=sys.stderr, flush=True)
    return 5


def _verify_preflight(
    path: Path,
    *,
    nonce: str,
    source_contract_sha256: str,
    expected_artifacts: Mapping[str, str],
) -> bool:
    """Verify the small, fresh attestation without hashing heavy artifacts."""
    try:
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            return False
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            return False
        sealed = dict(raw)
        declared = _strict_sha256(sealed.pop("attestation_sha256", ""))
        now_ns = time.time_ns()
        artifacts = dict(raw.get("verified_artifact_sha256") or {})
        return bool(
            raw.get("schema") == PREFLIGHT_SCHEMA
            and int(raw.get("schema_version") or 0) == 1
            and raw.get("ok") is True
            and str(raw.get("nonce") or "") == str(nonce)
            and str(raw.get("command_contract_sha256") or "")
            == str(source_contract_sha256)
            and str(raw.get("artifact_verification_status") or "") == "pass"
            and str(raw.get("host") or "") == socket.gethostname()
            and int(raw.get("created_at_unix_ns") or 0)
            <= now_ns
            <= int(raw.get("expires_at_unix_ns") or 0)
            and declared
            and _canonical_json_sha256(sealed) == declared
            and all(
                str(artifacts.get(name) or "") == digest
                for name, digest in expected_artifacts.items()
            )
        )
    except Exception:
        return False


def _load_input_manifest(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_tensor_sha256: str,
    preflight_verified: bool,
) -> tuple[dict[str, Any], Path, np.ndarray]:
    if not path.is_file():
        raise RuntimeError("native_full_input_manifest_missing")
    if (
        not preflight_verified
        and _sha256_file(path) != expected_manifest_sha256
    ):
        raise RuntimeError("native_full_input_manifest_sha256_mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != INPUT_SCHEMA
        or int(payload.get("schema_version") or 0) not in {1, 2}
        or str(payload.get("case") or "") != "full"
    ):
        raise RuntimeError("native_full_input_manifest_contract_invalid")
    runtime_file = path.parent / Path(
        str(payload.get("runtime_input_file") or "")
    ).name
    declared_sha = _strict_sha256(payload.get("runtime_input_sha256"))
    if (
        not runtime_file.is_file()
        or not declared_sha
        or declared_sha != expected_tensor_sha256
        or (
            not preflight_verified
            and _sha256_file(runtime_file) != expected_tensor_sha256
        )
    ):
        raise RuntimeError("native_full_runtime_input_sha256_mismatch")
    shape_raw = payload.get("runtime_input_shape")
    if (
        not isinstance(shape_raw, list)
        or not shape_raw
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape_raw
        )
    ):
        raise RuntimeError("native_full_runtime_input_shape_invalid")
    shape = tuple(int(value) for value in shape_raw)
    try:
        dtype = np.dtype(str(payload.get("runtime_input_dtype") or ""))
    except Exception as exc:
        raise RuntimeError("native_full_runtime_input_dtype_invalid") from exc
    expected_bytes = int(math.prod(shape) * dtype.itemsize)
    if (
        int(payload.get("runtime_input_bytes") or 0) != expected_bytes
        or int(runtime_file.stat().st_size) != expected_bytes
    ):
        raise RuntimeError("native_full_runtime_input_byte_count_mismatch")
    tensor = np.fromfile(runtime_file, dtype=dtype)
    if int(tensor.size) != int(math.prod(shape)):
        raise RuntimeError("native_full_runtime_input_element_count_mismatch")
    return dict(payload), runtime_file, np.ascontiguousarray(tensor.reshape(shape))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--frozen-postprocess-contract-json", default="")
    parser.add_argument(
        "--frozen-decoded-nms-normalization-contract-json", default="",
    )
    parser.add_argument("--source-endpoint-contract-hash", default="")
    parser.add_argument("--quality-first-producer-identity-sha256", default="")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-engine-sha256", required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--expected-runtime-input-sha256", required=True)
    parser.add_argument("--source-contract-sha256", default="")
    parser.add_argument("--preflight-attestation", default="")
    parser.add_argument("--preflight-nonce", default="")
    ns = parser.parse_args()

    report = Path(ns.json_out).expanduser().resolve()
    runner = Path(__file__).resolve()
    engine = Path(ns.engine).expanduser().resolve()
    input_manifest = Path(ns.input_manifest).expanduser().resolve()
    expected = {
        "runner": _strict_sha256(ns.expected_runner_sha256),
        "engine": _strict_sha256(ns.expected_engine_sha256),
        "input_manifest": _strict_sha256(ns.expected_input_manifest_sha256),
        "runtime_input": _strict_sha256(ns.expected_runtime_input_sha256),
    }
    if not all(expected.values()):
        return _fail(report, "expected_artifact_sha256_invalid")
    preflight_path = str(ns.preflight_attestation or "").strip()
    preflight_verified = False
    if preflight_path:
        source_sha = _strict_sha256(ns.source_contract_sha256)
        if not source_sha or not str(ns.preflight_nonce or "").strip():
            return _fail(report, "energy_preflight_identity_missing")
        preflight_verified = _verify_preflight(
            Path(preflight_path).expanduser(),
            nonce=str(ns.preflight_nonce),
            source_contract_sha256=source_sha,
            expected_artifacts={
                "hotloop_runner": expected["runner"],
                "engine": expected["engine"],
                "input_manifest": expected["input_manifest"],
                "runtime_input_tensor": expected["runtime_input"],
            },
        )
        if not preflight_verified:
            return _fail(report, "energy_preflight_attestation_invalid")
    else:
        if (
            not runner.is_file()
            or _sha256_file(runner) != expected["runner"]
            or not engine.is_file()
            or _sha256_file(engine) != expected["engine"]
        ):
            return _fail(report, "tensorrt_completed_hotloop_artifact_mismatch")
    if not engine.is_file():
        return _fail(report, "tensorrt_engine_missing")

    runtime: NativeTRT | None = None
    try:
        payload, runtime_file, tensor = _load_input_manifest(
            input_manifest,
            expected_manifest_sha256=expected["input_manifest"],
            expected_tensor_sha256=expected["runtime_input"],
            preflight_verified=preflight_verified,
        )
        raw_contract_json = str(
            ns.frozen_postprocess_contract_json or ""
        ).strip()
        direct_contract_json = str(
            ns.frozen_decoded_nms_normalization_contract_json or ""
        ).strip()
        if bool(raw_contract_json) == bool(direct_contract_json):
            raise RuntimeError(
                "completed_hotloop_exactly_one_completion_contract_required"
            )
        completion_kind = "raw_host_tail"
        if raw_contract_json:
            completion_contract = verify_frozen_postprocess_contract(
                json.loads(raw_contract_json)
            )
        else:
            completion_kind = "direct_bn6_normalization"
            completion_contract = (
                verify_frozen_decoded_nms_normalization_contract(
                    json.loads(direct_contract_json)
                )
            )
            manifest_geometry = verify_letterbox_geometry_contract(
                payload.get("letterbox_geometry_contract")
            )
            if (
                int(payload.get("schema_version") or 0) != 2
                or str(
                    payload.get(
                        "letterbox_geometry_contract_sha256"
                    ) or ""
                )
                != str(
                    manifest_geometry["geometry_contract_sha256"]
                )
                or dict(manifest_geometry)
                != dict(
                    completion_contract[
                        "letterbox_geometry_contract"
                    ]
                )
                or list(payload.get("original_image_wh") or [])
                != list(completion_contract["original_wh"])
            ):
                raise RuntimeError(
                    "direct_bn6_input_geometry_binding_invalid"
                )
        original_wh = [
            int(value)
            for value in list(
                completion_contract.get("original_wh") or []
            )
        ]
        runtime = NativeTRT(engine)
        if len(runtime.inputs) != 1:
            raise RuntimeError(
                "tensorrt_full_completed_hotloop_requires_one_input"
            )
        input_name = str(payload.get("runtime_input_name") or "")
        if input_name != str(runtime.inputs[0]):
            raise RuntimeError(
                "tensorrt_full_runtime_input_name_mismatch"
            )
        if (
            tuple(tensor.shape) != tuple(runtime.shapes[input_name])
            or np.dtype(tensor.dtype) != np.dtype(runtime.dtypes[input_name])
        ):
            raise RuntimeError(
                "tensorrt_full_runtime_input_shape_dtype_mismatch"
            )
        runtime.prepare_inputs({input_name: tensor})

        # The untimed probe binds the completion implementation to the exact
        # physical engine output structure.
        structural_outputs = runtime.run_prepared()
        if completion_kind == "raw_host_tail":
            verify_frozen_postprocess_contract(
                completion_contract, outputs=structural_outputs,
            )
            processor: (
                FrozenDetectionPostprocessor
                | FrozenDecodedNmsPostprocessor
            ) = FrozenDetectionPostprocessor(completion_contract)
        else:
            if (
                dict(
                    completion_contract[
                        "source_output_tensor_signature"
                    ]
                )
                != {
                    "tensor_count": len(structural_outputs),
                    "tensors": [
                        {
                            "index": int(index),
                            "name": str(name),
                            "rank": int(np.asarray(value).ndim),
                            "shape": [
                                int(dim)
                                for dim in np.asarray(value).shape
                            ],
                            "dtype": str(np.asarray(value).dtype),
                        }
                        for index, (name, value)
                        in enumerate(structural_outputs.items())
                    ],
                }
            ):
                raise RuntimeError(
                    "direct_bn6_runtime_output_signature_mismatch"
                )
            processor = FrozenDecodedNmsPostprocessor(
                completion_contract
            )

        warmup = max(0, int(ns.warmup))
        frames = max(1, int(ns.frames))
        duration_s = max(0.0, float(ns.duration_s or 0.0))
        for _ in range(warmup):
            outputs = runtime.run_prepared()
            processor.process(outputs, original_wh=original_wh)

        timings_ms: list[float] = []
        started = time.perf_counter()
        while (
            len(timings_ms) < frames
            or time.perf_counter() - started < duration_s
        ):
            iteration_started = time.perf_counter()
            outputs = runtime.run_prepared()
            completion_result = processor.process(
                outputs, original_wh=original_wh,
            )
            timings_ms.append(
                (time.perf_counter() - iteration_started) * 1000.0
            )
        makespan_s = max(0.0, time.perf_counter() - started)
        completed = len(timings_ms)
        postprocess_completed = int(processor.completed_count) - warmup
        duration_ok = bool(
            duration_s <= 0.0 or makespan_s >= duration_s
        )
        count_ok = bool(
            completed >= frames
            and postprocess_completed == completed
        )
        completed_result_artifact = _completed_detection_artifact(
            completion_result
        )
        completed_result_artifact_sha256 = str(
            completion_result.get("completed_result_artifact_sha256") or ""
        ).strip().lower()
        if not completed_result_artifact_sha256:
            completed_result_artifact_sha256 = _canonical_json_sha256(
                completed_result_artifact
            )
        completed_result_persistence = _persist_completed_result_artifact(
            completed_result_artifact,
            completed_result_artifact_sha256,
            report.with_name(
                f"{report.stem}.completed_task_result_artifact.json"
            ),
        )
        completed_result_artifact_saved = bool(
            completed_result_persistence["saved"]
        )
        count_ok = bool(count_ok and completed_result_artifact_saved)
        if count_ok and completion_kind == "raw_host_tail":
            completed_attestation = (
                build_completed_detection_endpoint_attestation(
                    completion_contract,
                    completion_result,
                    completed_frames=completed,
                    postprocess_completed_frames=postprocess_completed,
                    source_endpoint_contract_hash=str(
                        ns.source_endpoint_contract_hash or ""
                    ),
                )
            )
        elif count_ok:
            completed_attestation = (
                build_normalized_detection_endpoint_attestation(
                    completion_contract,
                    completion_result,
                    completed_frames=completed,
                    postprocess_completed_frames=postprocess_completed,
                )
            )
        else:
            completed_attestation = {}
        ok = bool(count_ok and duration_ok and completed_attestation)
        mean_ms = (
            float(sum(timings_ms) / completed) if completed else None
        )
        result = {
            "ok": ok,
            "status": "ok" if ok else "completed_work_units_mismatch",
            "benchmark_kind": "tensorrt_full_completed_task_hotloop",
            "e2e_scope": "full_task_pipeline",
            "measurement_concurrency": 1,
            "measurement_control": (
                "minimum_frames_and_duration"
                if duration_s > 0.0 else "exact_frames"
            ),
            "requested_work_units": frames,
            "completed_frames": completed,
            "completed_work_units": completed,
            "completed_work_units_source": (
                "tensorrt_sync_output_plus_frozen_completion_success_counter"
            ),
            "completed_work_units_status": (
                "exact_runtime_counter" if ok else "count_mismatch"
            ),
            "warmup_count": warmup,
            "requested_duration_s": duration_s,
            "measured_duration_s": makespan_s,
            "minimum_duration_satisfied": duration_ok,
            "fps_makespan": (
                float(completed / makespan_s)
                if makespan_s > 0.0 else None
            ),
            "latency_mean_ms": mean_ms,
            "latency_p50_ms": _percentile(timings_ms, 50.0),
            "latency_p95_ms": _percentile(timings_ms, 95.0),
            "latency_semantics": (
                (
                    "prepared_input_h2d_engine_d2h_sync_decode_class_aware_nms"
                    if completion_kind == "raw_host_tail"
                    else (
                        "prepared_input_h2d_engine_d2h_sync_bn6_filter_"
                        "class_aware_nms_inverse_letterbox"
                    )
                )
            ),
            "engine": str(engine),
            "engine_sha256": expected["engine"],
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": expected["input_manifest"],
            "runtime_input_file": str(runtime_file),
            "runtime_input_sha256": expected["runtime_input"],
            "runtime_input_name": input_name,
            "runtime_input_shape": [int(value) for value in tensor.shape],
            "runtime_input_dtype": str(tensor.dtype),
            "runtime_input_bytes": int(tensor.nbytes),
            "runtime_input_binding_verified": True,
            "preprocessing_timed": False,
            "disk_io_timed": False,
            "host_postprocess_frozen": (
                completion_kind == "raw_host_tail"
            ),
            "normalization_frozen": (
                completion_kind == "direct_bn6_normalization"
            ),
            "postprocess_included": True,
            "postprocess_completed_frames": postprocess_completed,
            "postprocess_completion_verified": bool(
                postprocess_completed == completed
            ),
            "frozen_host_postprocess_contract": (
                completion_contract
                if completion_kind == "raw_host_tail" else {}
            ),
            "frozen_host_postprocess_contract_sha256": str(
                completion_contract.get("contract_sha256") or ""
                if completion_kind == "raw_host_tail" else ""
            ),
            "frozen_host_postprocess_result": (
                completion_result
                if completion_kind == "raw_host_tail" else {}
            ),
            "frozen_decoded_nms_normalization_contract": (
                completion_contract
                if completion_kind == "direct_bn6_normalization" else {}
            ),
            "frozen_decoded_nms_normalization_contract_sha256": str(
                completion_contract.get("contract_sha256") or ""
                if completion_kind == "direct_bn6_normalization" else ""
            ),
            "frozen_decoded_nms_normalization_result": (
                completion_result
                if completion_kind == "direct_bn6_normalization" else {}
            ),
            "completed_task_stage": str(
                completed_attestation.get("stage") or ""
            ),
            "completed_task_contract_family": (
                "decoded_nms"
                if completed_attestation.get("attested") is True else ""
            ),
            "completed_task_endpoint_contract_hash": str(
                completed_attestation.get("endpoint_contract_hash") or ""
            ),
            "completed_task_output_endpoint_id": str(
                completed_attestation.get("output_endpoint_id") or ""
            ),
            "completed_task_comparison_endpoint_contract": (
                completed_attestation.get(
                    "completed_task_comparison_endpoint_contract"
                )
                if isinstance(
                    completed_attestation.get(
                        "completed_task_comparison_endpoint_contract"
                    ),
                    dict,
                )
                else {}
            ),
            "completed_task_comparison_endpoint_contract_hash": str(
                completed_attestation.get(
                    "completed_task_comparison_endpoint_contract_hash"
                ) or ""
            ),
            "completed_task_comparison_output_endpoint_id": str(
                completed_attestation.get(
                    "completed_task_comparison_output_endpoint_id"
                ) or ""
            ),
            "completed_task_completion_mode": str(
                completed_attestation.get(
                    "completed_task_completion_mode"
                ) or ""
            ),
            "completed_task_result_artifact_saved": (
                completed_result_artifact_saved
            ),
            "completed_task_result_artifact": completed_result_artifact,
            "completed_task_result_artifact_sha256": (
                completed_result_artifact_sha256
            ),
            "completed_task_result_artifact_path": str(
                completed_result_persistence["path"]
            ),
            "completed_task_result_artifact_file_sha256": str(
                completed_result_persistence["file_sha256"]
            ),
            "completed_task_endpoint_attestation": completed_attestation,
            "quality_first_producer_identity_sha256": str(
                ns.quality_first_producer_identity_sha256 or ""
            ),
            "source_contract_sha256": str(
                ns.source_contract_sha256 or ""
            ),
            "preflight_verified": preflight_verified,
        }
        _write_report(report, result)
        print(json.dumps(result), flush=True)
        return 0 if ok else 5
    except Exception as exc:
        return _fail(
            report,
            "tensorrt_full_completed_hotloop_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
