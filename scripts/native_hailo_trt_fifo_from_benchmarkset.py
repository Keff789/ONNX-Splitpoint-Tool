#!/usr/bin/env python3
"""Build/run a native HailoRT -> TensorRT FIFO fastpath for one BenchmarkSet split.

This is a deliberately small, generic C++ runner generator.  It does not know
about YOLOv7 specifically; it consumes a split_manifest/BenchmarkSet directory,
a Hailo Part1 HEF and a native TensorRT Part2 engine and measures the paper-style
pipeline:

  Thread A: image preprocess -> Hailo Part1 -> FIFO
  Thread B: FIFO -> TensorRT Part2

The first target is Hailo-8 -> TensorRT b066, but all paths are discovered from
BenchmarkSet artifacts so the same runner can be used for other Hailo->TRT splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import shutil
import site
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


def _activate_process_local_extra_sites() -> list[str]:
    """Expose existing accelerator bindings without modifying an environment.

    Hailo-8 detection is a mixed HailoRT/TensorRT process.  TensorRT belongs to
    the Jetson system interpreter while Hailo's bindings remain in the existing
    hailo_py environment.  The launcher passes those directories explicitly and
    this process adds them before importing either runtime.
    """
    activated: list[str] = []
    for raw in str(os.environ.get("SPLITPOINT_EXTRA_SITES") or "").split(
        os.pathsep
    ):
        path = str(raw or "").strip()
        if not path or path in activated:
            continue
        if not Path(path).is_dir():
            raise RuntimeError(
                f"splitpoint_extra_site_missing:{path}"
            )
        site.addsitedir(path)
        activated.append(path)
    return activated


_PROCESS_LOCAL_EXTRA_SITES = _activate_process_local_extra_sites()

import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool import __version__ as TOOL_VERSION
from onnx_splitpoint_tool.native_command_contract import (
    hailo8_preprocess_binding,
    load_split_energy_workload_binding,
)
from onnx_splitpoint_tool.native_split_quality import (
    bind_quality_to_native_split,
    canonical_native_split_backend,
    canonical_json_sha256,
    native_split_quality_selection_duplicates,
    validate_native_split_quality_binding,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    load_manifest_outputs,
    runtime_output_contract,
)
from onnx_splitpoint_tool.native_three_stage import (
    ADAPTER_CLASSIFICATION_LOGITS,
    ADAPTER_YOLO11_DFL16,
    ADAPTER_YOLO26_DECODED,
    ADAPTER_YOLOV7_SPARSE,
    FastDetectionCompletionRuntime,
    NativeThreeStageError,
    adapter_id_for_contract_family,
    infer_p2_output_contract_family,
    project_three_stage_endpoints,
)

from onnx_splitpoint_tool.native_detection_postprocess import (
    DetectionCompletionRuntime,
    build_detection_completion_execution_contract,
    persist_detection_completion_execution_artifacts,
    verify_detection_completion_execution_attestation,
    verify_detection_completion_execution_contract,
)
from onnx_splitpoint_tool.runners._types import RunCfg
from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend

try:
    import native_hailo10_trt_e2e_from_benchmarkset as _native_trt_module
    NativeTRT = _native_trt_module.NativeTRT
except Exception as exc:  # pragma: no cover - hardware dependency surface
    _native_trt_module = None  # type: ignore[assignment]
    NativeTRT = None  # type: ignore[assignment]
    _NATIVE_TRT_IMPORT_ERROR = exc
else:
    _NATIVE_TRT_IMPORT_ERROR = None


_REPLAY_CONTRACT_SCHEMA = "onnx-splitpoint/native-command-contract"
_CONCURRENT_LOG_PREFIX = f"[native-fifo][v{TOOL_VERSION}][concurrent]"


def _cache_verify_only() -> bool:
    return str(
        os.environ.get("ONNX_SPLITPOINT_ARTIFACT_POLICY") or ""
    ).strip().lower() == "cache_verify_only"

_REPETITION_MEDIAN_KEYS = (
    'preprocess_ms', 'p1_ms', 'handoff_ms', 'p2_run_ms', 'p1_thread_ms',
    'p2_thread_ms', 'single_latency_model_ms', 'paper_equivalent_cycle_ms',
    'paper_equivalent_fps', 'makespan_ms', 'fps_makespan',
)


def _require_hailo8_detection_runtime() -> dict[str, Any]:
    """Re-attest the mixed runtime before resolving artifacts or hardware."""
    import ctypes
    import ctypes.util
    import importlib

    modules: dict[str, str] = {}
    for name in ("tensorrt", "hailo_platform", "numpy", "PIL"):
        module = importlib.import_module(name)
        modules[name] = str(getattr(module, "__file__", "") or "")
    cudart = ctypes.util.find_library("cudart") or "libcudart.so"
    ctypes.CDLL(cudart)
    if NativeTRT is None or _native_trt_module is None:
        raise RuntimeError(
            "native TensorRT runtime is unavailable: "
            f"{_NATIVE_TRT_IMPORT_ERROR!r}"
        )
    consumer_source = Path(
        str(getattr(_native_trt_module, "__file__", "") or "")
    ).resolve()
    if not consumer_source.is_file():
        raise RuntimeError(
            "native_trt_consumer_source_missing"
        )
    return {
        "status": "ready",
        "runtime_mode": (
            "system_tensorrt_with_process_local_hailo_sites"
        ),
        "site_policy": "site.addsitedir_after_system_defaults",
        "python_executable": str(sys.executable),
        "resolved_python_executable": str(
            Path(sys.executable).resolve()
        ),
        "process_local_extra_sites": list(
            _PROCESS_LOCAL_EXTRA_SITES
        ),
        "modules": modules,
        "cudart": cudart,
        "native_trt_consumer_source": str(consumer_source),
        "native_trt_consumer_source_sha256": _sha256_file(
            consumer_source
        ),
        "source_closure_ok": True,
    }


def _fresh_hailo_repetition_identity(
    repetition_index: int, repetition_out: Path,
) -> tuple[str, str]:
    """Return independent logical-repeat and runtime-process identities."""
    execution_nonce = uuid.uuid4().hex
    invocation_token = hashlib.sha256(
        (
            f"hailo8-native-process:{os.getpid()}:{time.time_ns()}:"
            f"{execution_nonce}:{int(repetition_index)}:"
            f"{Path(repetition_out).resolve()}"
        ).encode('utf-8')
    ).hexdigest()
    return f'hailo8:{execution_nonce}', f'fresh_process:{invocation_token}'


def _fresh_hailo_runtime_identity(
    repetition_index: int,
) -> tuple[str, str]:
    """Identify a fresh runtime set without claiming a fresh OS process."""
    execution_nonce = uuid.uuid4().hex
    invocation_token = hashlib.sha256(
        (
            f"hailo8-python-runtime:{os.getpid()}:{time.time_ns()}:"
            f"{execution_nonce}:{int(repetition_index)}"
        ).encode("utf-8")
    ).hexdigest()
    return (
        f"hailo8-runtime:{execution_nonce}",
        f"fresh_runtime:{invocation_token}",
    )


def _median_ci95(values: list[float]) -> tuple[float, float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float('nan'), float('nan'), float('nan')
    med = float(statistics.median(vals))
    if len(vals) == 1:
        return med, med, med
    rng = random.Random(0x261F)
    boot = sorted(
        float(statistics.median(rng.choices(vals, k=len(vals))))
        for _ in range(20000)
    )
    return med, boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]


def _aggregate_repetition_payloads(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError('at least one repetition result is required')
    out = dict(runs[-1])
    for key in _REPETITION_MEDIAN_KEYS:
        vals = [float(row[key]) for row in runs if row.get(key) is not None and math.isfinite(float(row[key]))]
        if vals:
            out[key] = float(statistics.median(vals))
    fps_values = [float(row['fps_makespan']) for row in runs if row.get('fps_makespan') is not None and math.isfinite(float(row['fps_makespan']))]
    fps_median, fps_low, fps_high = _median_ci95(fps_values)
    records=[]
    for i,row in enumerate(runs):
        attestation = row.get('completed_task_endpoint_attestation') or {}
        identity = attestation.get('sentinel_identity') if isinstance(attestation, Mapping) else None
        if isinstance(identity, Mapping) and 'repetition_id' in identity:
            if any(identity.get(key) != value for key, value in {
                'process_local_repetition_index': i + 1,
                'repetition_id': row.get('repetition_id'),
                'runtime_instance_id': row.get('runtime_instance_id'),
                'completed_work_unit_index': row.get('completed_work_units'),
            }.items()):
                raise RuntimeError('fast_oracle_repetition_identity_mismatch')
        record=dict(row, repetition_index=i + 1)
        completed=int(record.get('completed_work_units') or record.get('completed_frames') or record.get('frames') or 0)
        record.update({'frames': completed, 'completed_frames': completed, 'completed_work_units': completed, 'ok': True, 'status': 'ok'})
        records.append(record)
    runtime_instance_ids = [str(row.get('runtime_instance_id') or '') for row in records]
    independence_verified = bool(
        runtime_instance_ids
        and all(runtime_instance_ids)
        and len(set(runtime_instance_ids)) == len(runtime_instance_ids)
        and all(int(row.get('completed_work_units') or 0) > 0 for row in records)
        and len({int(row.get('completed_work_units') or 0) for row in records}) == 1
    )
    scopes = {
        str(row.get("repetition_runtime_scope") or "")
        for row in records
    }
    runtime_scope = (
        scopes.pop() if len(scopes) == 1
        else "mixed_or_missing_runtime_scope"
    )
    out.update({
        'repetitions_requested': len(runs),
        'repetitions_completed': len(runs),
        'repetition_count_requested': len(runs),
        'repetition_count_attempted': len(runs),
        'repetition_count_valid': len(runs),
        'repetition_status': 'complete',
        'repetition_aggregation': 'median_never_best_of',
        'fps_makespan': fps_median,
        'fps_makespan_median': fps_median,
        'fps_makespan_ci95_low': fps_low,
        'fps_makespan_ci95_high': fps_high,
        'fps_median': fps_median,
        'fps_ci95_low': fps_low,
        'fps_ci95_high': fps_high,
        'fps_makespan_ci95_method': 'deterministic_percentile_bootstrap_of_repetition_medians_20000',
        'fps_ci95_method': 'deterministic_percentile_bootstrap_of_repetition_medians_20000',
        'fps_ci95_level': 0.95,
        'aggregate_total_completed_frames': int(sum(int(row.get('completed_frames') or row.get('frames') or 0) for row in runs)),
        'repetition_evidence': records,
        'repetition_records': records,
        'repetition_runtime_scope': runtime_scope,
        'repetition_independence_verified': independence_verified,
        'repetition_runtime_instance_ids': runtime_instance_ids,
    })
    selected_attestation = out.get('completed_task_endpoint_attestation') or {}
    if isinstance(selected_attestation, Mapping) and isinstance(selected_attestation.get('sentinel_identity'), Mapping):
        selected = selected_attestation['sentinel_identity']
        out.update({
            'semantic_evidence_selection': 'last_completed_repetition_never_best_of',
            'semantic_evidence_repetition_index': selected.get('process_local_repetition_index'),
            'semantic_evidence_repetition_id': selected.get('repetition_id'),
            'semantic_evidence_runtime_instance_id': selected.get('runtime_instance_id'),
        })
    return out


def _bind_repetition_records_to_workload(
    payload: dict[str, Any], workload_contract_sha256: str,
) -> None:
    """Bind every repetition mirror to the final sealed command contract."""

    for records_key in ('repetition_records', 'repetition_evidence'):
        records = payload.get(records_key)
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    record['workload_contract_sha256'] = (
                        workload_contract_sha256
                    )


def _sha256_file(path: str | Path | None) -> str:
    """Return a bare SHA-256 digest without accepting missing artefacts."""
    if not path:
        return ""
    source = Path(path).expanduser()
    if not source.is_file():
        return ""
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hailo8_input_geometry(prepared: Any) -> tuple[str, tuple[int, ...], int, int, int, bool]:
    handle = prepared.handle
    shapes = dict(
        getattr(handle, "runtime_input_shapes", None)
        or getattr(handle, "input_shapes", None)
        or {}
    )
    if len(list(prepared.input_names or [])) != 1:
        raise RuntimeError("Hailo-8 split detection requires exactly one Part1 input")
    name = str(prepared.input_names[0])
    shape = tuple(int(value) for value in shapes.get(name, ()))
    if len(shape) not in {3, 4} or any(value <= 0 for value in shape):
        raise RuntimeError(f"Hailo-8 runtime input shape is unavailable: {shape}")
    if len(shape) == 4 and shape[0] != 1:
        raise RuntimeError("Hailo-8 split detection supports batch size one only")
    inner = shape[1:] if len(shape) == 4 else shape
    if inner[-1] in {1, 3, 4}:
        height, width, channels = inner
        channels_first = False
    elif inner[0] in {1, 3, 4}:
        channels, height, width = inner
        channels_first = True
    else:
        raise RuntimeError(f"Hailo-8 image tensor layout is ambiguous: {shape}")
    return name, shape, height, width, channels, channels_first


def _hailo8_python_input(
    prepared: Any,
    image_path: Path,
    *,
    quantized: bool,
    preprocess_mode: str,
    letterbox_pad_value: int,
    prepared_input_rgb: str | Path = "",
    expected_prepared_input_sha256: str = "",
) -> dict[str, np.ndarray]:
    """Load the shared RGB bytes, or prepare a standalone input, before timing.

    A dual invocation consumes the raw endpoint's actual OpenCV-prepared bytes.
    It must never decode or resize the JPEG independently in the Python endpoint.
    Runtime layout and uint8/float conversion are applied only after that binding.
    """
    name, shape, height, width, channels, channels_first = _hailo8_input_geometry(prepared)
    if prepared_input_rgb:
        if channels != 3:
            raise RuntimeError("shared_prepared_rgb_requires_three_channels")
        source = Path(prepared_input_rgb).expanduser().resolve()
        expected_size = height * width * channels
        if source.stat().st_size != expected_size:
            raise RuntimeError("shared_prepared_rgb_byte_count_mismatch")
        raw = source.read_bytes()
        if len(raw) != expected_size:
            raise RuntimeError("shared_prepared_rgb_byte_count_mismatch")
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_prepared_input_sha256 and actual_sha256 != expected_prepared_input_sha256:
            raise RuntimeError("shared_prepared_rgb_sha256_mismatch")
        array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)
    else:
        if expected_prepared_input_sha256:
            raise RuntimeError("shared_prepared_rgb_missing")
        if Image is None:
            raise RuntimeError("Pillow is required for Hailo-8 detection input")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            if preprocess_mode == "letterbox":
                scale = min(float(width) / float(image.width), float(height) / float(image.height))
                resized_width = max(1, int(round(image.width * scale)))
                resized_height = max(1, int(round(image.height * scale)))
                resized = image.resize((resized_width, resized_height))
                canvas = Image.new("RGB", (width, height), (int(letterbox_pad_value),) * 3)
                canvas.paste(resized, ((width - resized_width) // 2, (height - resized_height) // 2))
                array = np.asarray(canvas)
            else:
                array = np.asarray(image.resize((width, height)))
        if channels == 1:
            array = array[..., :1]
    if channels_first:
        array = np.transpose(array, (2, 0, 1))
    if len(shape) == 4:
        array = array[None, ...]
    result = array.astype(np.uint8, copy=False) if quantized else array.astype(np.float32) / np.float32(255.0)
    result = np.ascontiguousarray(result)
    if tuple(result.shape) != shape:
        raise RuntimeError(f"Hailo-8 prepared input shape drift: {result.shape} != {shape}")
    return {name: result}


def _hailo8_reuse_python_input(prepared: Any, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bind the already materialized tensor to a fresh repetition's runtime."""
    name, shape, *_ = _hailo8_input_geometry(prepared)
    if list(inputs) != [name] or tuple(inputs[name].shape) != shape:
        raise RuntimeError("shared_prepared_runtime_input_drift")
    return dict(inputs)


def _hailo8_python_boundary(
    outputs: Mapping[str, Any],
    trt: Any,
    *,
    expected_name: str = "",
    expected_shape: list[int] | None = None,
    expected_dtype: str = "",
) -> tuple[str, np.ndarray, dict[str, Any]]:
    """Bind one VStreams output to the sole Part2 input without guessing."""
    if len(outputs) != 1:
        raise RuntimeError(
            "Hailo-8 split detection requires exactly one Part1 output"
        )
    if len(list(getattr(trt, "inputs", []) or [])) != 1:
        raise RuntimeError(
            "Hailo-8 split detection requires exactly one TensorRT input"
        )
    target = str(trt.inputs[0])
    candidates: list[tuple[str, np.ndarray]] = []
    for name, value in outputs.items():
        array = np.asarray(value)
        if int(array.size) == int(np.prod(trt.shapes[target])):
            candidates.append((str(name), array))
    if expected_name:
        exact = [
            item for item in candidates if item[0] == str(expected_name)
        ]
        if len(exact) != 1:
            raise RuntimeError(
                "native_split_quality_runtime_boundary_name_mismatch"
            )
        candidates = exact
    if len(candidates) != 1:
        raise RuntimeError(
            "Hailo-8 Part1 output to TensorRT input mapping is ambiguous"
        )
    source_name, raw = candidates[0]
    if expected_shape and tuple(raw.shape) != tuple(expected_shape):
        raise RuntimeError(
            "native_split_quality_runtime_boundary_shape_mismatch"
        )
    if expected_dtype and str(raw.dtype).lower() != str(expected_dtype).lower():
        raise RuntimeError(
            "native_split_quality_runtime_boundary_dtype_mismatch"
        )
    target_shape = tuple(int(value) for value in trt.shapes[target])
    target_dtype = np.dtype(trt.dtypes[target])
    array = np.asarray(raw)
    if tuple(array.shape) != target_shape:
        array = array.reshape(target_shape)
    if array.dtype != target_dtype:
        array = array.astype(target_dtype, copy=False)
    # This is the single owned accelerator-to-FIFO boundary materialization.
    owned = np.array(array, dtype=target_dtype, order="C", copy=True)
    return target, owned, {
        "status": "exact_runtime_boundary_verified",
        "runtime_name": source_name,
        "shape": [int(value) for value in raw.shape],
        "dtype": str(raw.dtype),
        "element_count": int(raw.size),
        "trt_input_name": target,
        "trt_input_shape": [int(value) for value in target_shape],
        "boundary_copy_count": 1,
    }


def _hailo8_detection_geometry(
    inputs: Mapping[str, np.ndarray],
    image_path: Path,
) -> tuple[list[int], list[int]]:
    if Image is None:
        raise RuntimeError("Pillow is required for detection geometry")
    if len(inputs) != 1:
        raise RuntimeError("detection geometry requires exactly one input")
    shape = tuple(int(value) for value in next(iter(inputs.values())).shape)
    inner = shape[1:] if len(shape) == 4 and shape[0] == 1 else shape
    candidates: list[tuple[int, int]] = []
    if len(inner) == 3:
        if inner[-1] in {1, 3, 4}:
            candidates.append((inner[0], inner[1]))
        if inner[0] in {1, 3, 4}:
            candidates.append((inner[1], inner[2]))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            f"detection completion input geometry is ambiguous: {shape}"
        )
    with Image.open(image_path) as source:
        original_wh = [int(source.width), int(source.height)]
    return [int(candidates[0][0]), int(candidates[0][1])], original_wh


def _hailo8_detection_completion_contract(
    *,
    benchmark_set: Path,
    model_id: str,
    outputs: Mapping[str, Any],
    inputs: Mapping[str, np.ndarray],
    image_path: Path,
    preprocess_mode: str,
    letterbox_pad_value: int,
) -> dict[str, Any]:
    input_hw, original_wh = _hailo8_detection_geometry(
        inputs, image_path,
    )
    declaration = load_authoritative_output_contract(
        benchmark_set,
        backend="tensorrt",
        model_id=str(model_id),
        variant="full",
        task="detection",
    )
    source_endpoint = runtime_output_contract(
        "detection",
        dict(outputs),
        raw_fallback=False,
        declared_contract=declaration,
    )
    contract = build_detection_completion_execution_contract(
        model_id=str(model_id),
        outputs=dict(outputs),
        input_hw=input_hw,
        original_wh=original_wh,
        source_endpoint_contract=source_endpoint,
        preprocess={
            "mode": str(preprocess_mode),
            "rgb": True,
            "pad_value": int(letterbox_pad_value),
        },
    )
    return verify_detection_completion_execution_contract(contract)


def _hailo8_completion_fields(
    completion_runtime: Any,
    *,
    completed_work_units: int,
) -> dict[str, Any]:
    if int(getattr(completion_runtime, "completed_count", -1)) != int(
        completed_work_units
    ):
        raise RuntimeError(
            "detection completion count does not match completed work units"
        )
    raw = completion_runtime.attestation(
        completed_work_units=int(completed_work_units)
    )
    if not isinstance(raw, Mapping):
        raise RuntimeError("detection completion attestation missing")
    attestation = (
        verify_detection_completion_execution_attestation(
            raw,
            execution_contract=completion_runtime.execution_contract,
            expected_observation_relation="same_hotloop_sentinel",
        )
        if isinstance(completion_runtime, DetectionCompletionRuntime)
        else dict(raw)
    )
    observation_relation = str(
        attestation.get("observation_relation") or ""
    )
    if (
        attestation.get("attested") is not True
        or attestation.get("status") != "passed"
        or observation_relation not in {
            "same_hotloop_sentinel",
            "postflight_oracle_sentinel",
        }
        or attestation.get("exact_result_claim_bound") is not True
        or int(attestation.get("completed_work_units") or 0)
        != int(completed_work_units)
        or int(attestation.get("completion_count") or 0)
        != int(completed_work_units)
    ):
        raise RuntimeError("detection completion attestation invalid")
    completed_endpoint = dict(
        attestation.get("completed_endpoint_contract") or {}
    )
    comparison_endpoint = dict(
        attestation.get("comparison_endpoint_contract") or {}
    )
    execution_contract = getattr(
        completion_runtime, "execution_contract", None
    )
    source_endpoint = (
        dict(execution_contract.get("source_endpoint") or {})
        if isinstance(execution_contract, Mapping) else {}
    )
    return {
        **({
            "completion_sentinel_identity": dict(attestation["sentinel_identity"]),
            "semantic_evidence_repetition_index": attestation["sentinel_identity"].get("process_local_repetition_index"),
            "semantic_evidence_repetition_id": attestation["sentinel_identity"].get("repetition_id"),
            "semantic_evidence_runtime_instance_id": attestation["sentinel_identity"].get("runtime_instance_id"),
        } if isinstance(attestation.get("sentinel_identity"), Mapping) else {}),
        **({
            "endpoint_contract_hash": str(source_endpoint["endpoint_contract_hash"]),
            "accelerator_endpoint_contract_hash": str(source_endpoint["endpoint_contract_hash"]),
            "output_endpoint_id": str(source_endpoint["output_endpoint_id"]),
            "physical_output_endpoint_id": str(source_endpoint["output_endpoint_id"]),
        } if source_endpoint else {}),
        "completed_work_units": int(completed_work_units),
        "completed_frames": int(completed_work_units),
        "postprocess_included": True,
        "postprocess_completed_frames": int(completed_work_units),
        "postprocess_completion_verified": True,
        **({
            "completion_execution_contract": dict(execution_contract),
        } if isinstance(execution_contract, Mapping) else {}),
        "completion_execution_attestation": dict(attestation),
        "completion_execution_contract_sha256": str(
            attestation.get("execution_contract_sha256") or ""
        ),
        "completion_observation_relation": observation_relation,
        "completion_exact_result_claim_bound": True,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_completion_mode": (
            "native_three_stage_fast_oracle_outside_timing"
            if observation_relation == "postflight_oracle_sentinel"
            else "detection_completion_execution_v1"
        ),
        "quality_oracle_location": (
            "outside_performance_timing"
            if observation_relation == "postflight_oracle_sentinel"
            else "same_hotloop"
        ),
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation": dict(attestation),
        "completed_task_endpoint_attestation_status": "passed",
        "completed_task_endpoint_contract": completed_endpoint,
        "completed_task_endpoint_contract_hash": str(
            completed_endpoint.get("endpoint_contract_hash") or ""
        ),
        "completed_task_output_endpoint_id": str(
            completed_endpoint.get("output_endpoint_id") or ""
        ),
        "comparison_endpoint_contract": comparison_endpoint,
        "comparison_endpoint_contract_hash": str(
            comparison_endpoint.get("endpoint_contract_hash") or ""
        ),
        "completed_task_comparison_endpoint_contract": comparison_endpoint,
        "completed_task_comparison_endpoint_contract_hash": str(
            comparison_endpoint.get("endpoint_contract_hash") or ""
        ),
        "completed_task_comparison_output_endpoint_id": str(
            comparison_endpoint.get("output_endpoint_id") or ""
        ),
        "completion_artifact_sha256": str(
            attestation.get("artifact_sha256") or ""
        ),
        "completion_schema_sha256": str(
            attestation.get("schema_sha256") or ""
        ),
        "completion_content_sha256": str(
            attestation.get("content_sha256") or ""
        ),
        "completion_invocation_sha256": str(
            attestation.get("invocation_sha256") or ""
        ),
        "completion_relation_sha256": str(
            attestation.get("relation_sha256") or ""
        ),
    }


def _hailo8_python_fifo_run(
    backend: Any,
    prepared: Any,
    inputs: Mapping[str, np.ndarray],
    trt: Any,
    *,
    frames: int,
    warmup: int,
    queue_depth: int,
    duration_s: float,
    completion_runtime: Any,
    warmup_completion_runtime: Any | None,
    expected_boundary_name: str = "",
    expected_boundary_shape: list[int] | None = None,
    expected_boundary_dtype: str = "",
) -> dict[str, Any]:
    """Measure VStreams P1 -> TRT P2 -> Completed Detection in one FIFO."""
    if completion_runtime is None:
        raise RuntimeError(
            "detection completion runtime missing from measured hotloop"
        )
    if int(warmup) > 0 and warmup_completion_runtime is None:
        raise RuntimeError(
            "detection completion runtime missing from warmup hotloop"
        )
    for _ in range(max(0, int(warmup))):
        hailo_outputs = backend.run(prepared, dict(inputs)).outputs
        name, boundary, _meta = _hailo8_python_boundary(
            hailo_outputs,
            trt,
            expected_name=expected_boundary_name,
            expected_shape=expected_boundary_shape,
            expected_dtype=expected_boundary_dtype,
        )
        trt_outputs = trt.run({name: boundary})
        warmup_completion_runtime.process(trt_outputs)

    fifo: queue.Queue[Any] = queue.Queue(
        maxsize=max(1, int(queue_depth))
    )
    sentinel = object()
    errors: list[str] = []
    p1_times: list[float] = []
    handoff_times: list[float] = []
    p2_times: list[float] = []
    completion_times: list[float] = []
    boundary_evidence: dict[str, Any] = {}
    counters = {"produced": 0, "completed": 0}
    ready = threading.Barrier(3)
    start_event = threading.Event()
    cancel_event = threading.Event()
    measurement = {"start": 0.0, "last_completion": 0.0}

    def put_payload(payload: Any) -> bool:
        while not cancel_event.is_set():
            try:
                fifo.put(payload, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def producer() -> None:
        ready.wait()
        start_event.wait()
        produced = 0
        duration_mode = float(duration_s or 0.0) > 0.0
        try:
            while not cancel_event.is_set() and (
                (
                    duration_mode
                    and time.perf_counter() - measurement["start"]
                    < float(duration_s)
                )
                or (not duration_mode and produced < int(frames))
            ):
                t0 = time.perf_counter()
                hailo_outputs = backend.run(
                    prepared, dict(inputs)
                ).outputs
                t1 = time.perf_counter()
                name, boundary, evidence = _hailo8_python_boundary(
                    hailo_outputs,
                    trt,
                    expected_name=expected_boundary_name,
                    expected_shape=expected_boundary_shape,
                    expected_dtype=expected_boundary_dtype,
                )
                t2 = time.perf_counter()
                if boundary_evidence and evidence != boundary_evidence:
                    raise RuntimeError(
                        "Hailo-8 runtime boundary evidence drift"
                    )
                boundary_evidence.update(evidence)
                if not put_payload((name, boundary)):
                    break
                p1_times.append((t1 - t0) * 1000.0)
                handoff_times.append((t2 - t1) * 1000.0)
                produced += 1
                counters["produced"] = produced
        except Exception as exc:
            errors.append(f"producer: {exc!r}")
            cancel_event.set()
        finally:
            if cancel_event.is_set():
                try:
                    fifo.put_nowait(sentinel)
                except queue.Full:
                    pass
            else:
                fifo.put(sentinel)

    def consumer() -> None:
        ready.wait()
        start_event.wait()
        try:
            while True:
                try:
                    item = fifo.get(timeout=0.05)
                except queue.Empty:
                    if cancel_event.is_set():
                        break
                    continue
                if item is sentinel:
                    break
                name, boundary = item
                t0 = time.perf_counter()
                trt_outputs = trt.run({name: boundary})
                t1 = time.perf_counter()
                completion_runtime.process(trt_outputs)
                t2 = time.perf_counter()
                p2_times.append((t1 - t0) * 1000.0)
                completion_times.append((t2 - t1) * 1000.0)
                counters["completed"] += 1
                # The scientific completion boundary advances only after
                # decode/normalization, NMS, coordinate projection, record
                # materialization and all completion hashes succeeded.
                measurement["last_completion"] = t2
        except Exception as exc:
            errors.append(f"consumer: {exc!r}")
            cancel_event.set()

    producer_thread = threading.Thread(target=producer, daemon=True)
    consumer_thread = threading.Thread(target=consumer, daemon=True)
    producer_thread.start()
    consumer_thread.start()
    ready.wait()
    start = time.perf_counter()
    measurement["start"] = start
    start_event.set()
    producer_thread.join()
    consumer_thread.join()
    if errors:
        raise RuntimeError("; ".join(errors))
    completed = int(counters["completed"])
    if (
        completed <= 0
        or completed != int(counters["produced"])
        or measurement["last_completion"] <= 0.0
    ):
        raise RuntimeError(
            "Hailo-8 FIFO did not fully drain to Completed Detection"
        )

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    makespan_ms = (
        float(measurement["last_completion"]) - start
    ) * 1000.0
    p1_ms = mean(p1_times)
    handoff_ms = mean(handoff_times)
    p2_ms = mean(p2_times)
    completion_ms = mean(completion_times)
    p1_thread_ms = p1_ms + handoff_ms
    p2_thread_ms = p2_ms + completion_ms
    cycle_ms = max(p1_thread_ms, p2_thread_ms)
    result = {
        "ok": True,
        "mode": "hailo8_python_vstreams_trt_completed_detection_fifo",
        "producer_impl": "hailo8_python_vstreams_fifo",
        "frames": completed,
        "completed_frames": completed,
        "completed_work_units": completed,
        "requested_frames": int(frames),
        "duration_s": float(duration_s or 0.0),
        "warmup": int(warmup),
        "queue_depth": int(queue_depth),
        "produced_frames": int(counters["produced"]),
        "p1_ms": p1_ms,
        "handoff_ms": handoff_ms,
        "p2_run_ms": p2_ms,
        "completion_tail_ms": completion_ms,
        "postprocess_ms": completion_ms,
        "p1_thread_ms": p1_thread_ms,
        "p2_thread_ms": p2_thread_ms,
        "paper_equivalent_cycle_ms": cycle_ms,
        "paper_equivalent_fps": (
            1000.0 / cycle_ms if cycle_ms > 0.0 else 0.0
        ),
        "makespan_ms": makespan_ms,
        "fps_makespan": (
            float(completed) / (makespan_ms / 1000.0)
            if makespan_ms > 0.0 else 0.0
        ),
        "measurement_boundary": (
            "workers_ready_to_last_completed_task_frame"
        ),
        "last_completion_source": (
            "same_hotloop_completed_task_sentinel"
        ),
        "completed_work_units_source": (
            "same_hotloop_detection_completion_counter"
        ),
        "completed_work_units_status": "exact_runtime_counter",
        "warmup_contract": "fully_drained_before_worker_start",
        "boundary_copy_count": 1,
        "boundary_copy_total": completed,
        "boundary_copy_policy": (
            "one_owned_contiguous_typed_copy_before_fifo_enqueue"
        ),
        "fifo_payload_ownership": "owned_no_alias_to_vstreams_output",
        "strict_quality_boundary": boundary_evidence,
    }
    result.update(_hailo8_completion_fields(
        completion_runtime,
        completed_work_units=completed,
    ))
    return result


def _open_hailo8_python_runtime(
    *,
    hef: Path,
    engine: Path,
    work: Path,
    args: argparse.Namespace,
    runtime_label: str,
) -> tuple[HailoBackend, Any, Any]:
    """Open one fresh Hailo-8 VStreams and TensorRT runtime pair."""
    if NativeTRT is None:
        raise RuntimeError(
            "native TensorRT runtime is unavailable: "
            f"{_NATIVE_TRT_IMPORT_ERROR!r}"
        )
    quantized = str(args.hailo_format) != "float32"
    options = {
        "hw_arch": "hailo8",
        "hef_path": str(hef),
        "compile_backend": "local",
        "runtime_api": "vstreams",
        "quantized_inputs": quantized,
        "quantized_outputs": quantized,
        "persistent_activation": True,
        "copy_outputs": True,
        "device_id": str(args.device_id or ""),
    }
    backend = HailoBackend(strict=True, **options)
    prepared = backend.prepare(
        RunCfg(model_path=hef, options=options),
        work / "python_vstreams" / runtime_label,
    )
    try:
        trt = NativeTRT(engine)
    except Exception:
        backend.cleanup(prepared)
        raise
    return backend, prepared, trt


def _close_hailo8_python_runtime(
    backend: Any,
    prepared: Any,
    trt: Any,
) -> None:
    try:
        backend.cleanup(prepared)
    finally:
        trt.close()


def _dump_hailo8_python_semantic_evidence(
    *,
    work: Path,
    inputs: Mapping[str, np.ndarray],
    boundary: np.ndarray,
    boundary_meta: Mapping[str, Any],
    trt_outputs: Mapping[str, Any],
    image: Path,
    args: argparse.Namespace,
    output_scope: str = "separate_bound_probe_inference",
    completion_attestation: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Persist already available tensors outside the measurement interval."""
    result: dict[str, str] = {}
    if args.dump_outputs:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir else work / "native_fifo_outputs"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, (name, value) in enumerate(trt_outputs.items()):
            array = np.ascontiguousarray(np.asarray(value))
            output_file = output_dir / f"output_{index:03d}.bin"
            output_file.write_bytes(array.tobytes(order="C"))
            rows.append({
                "index": index,
                "name": str(name),
                "dtype": str(array.dtype),
                "shape": [int(x) for x in array.shape],
                "file": output_file.name,
            })
        manifest = output_dir / "native_fifo_output_manifest.json"
        output_payload = {
            "schema": "onnx-splitpoint/native-output-dump",
            "schema_version": 4,
            "dump_inference_scope": output_scope,
            "input_image": str(image),
            "outputs": rows,
        }
        if completion_attestation is not None:
            identity = dict(completion_attestation.get("sentinel_identity") or {})
            output_payload.update({
                "completion_sentinel_identity": identity,
                "completion_attestation_sha256": completion_attestation.get("attestation_sha256"),
                "input_image_sha256": identity.get("input_image_sha256"),
            })
        manifest.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
        _annotate_output_contract(manifest, Path(args.benchmark_set))
        _seal_manifest_payload_files(manifest)
        result["native_fifo_output_manifest"] = str(manifest)
    if args.dump_boundary:
        boundary_dir = (
            Path(args.boundary_dir).expanduser().resolve()
            if args.boundary_dir else work / "native_fifo_boundary"
        )
        boundary_dir.mkdir(parents=True, exist_ok=True)
        boundary_file = boundary_dir / "boundary.bin"
        boundary_file.write_bytes(
            np.ascontiguousarray(boundary).tobytes(order="C")
        )
        input_name, input_value = next(iter(inputs.items()))
        input_file = boundary_dir / "prepared_input.bin"
        input_file.write_bytes(
            np.ascontiguousarray(input_value).tobytes(order="C")
        )
        manifest = boundary_dir / "native_fifo_boundary_manifest.json"
        manifest.write_text(json.dumps({
            "schema": "onnx-splitpoint/native-boundary-dump",
            "schema_version": 2,
            "seq": -1,
            "image": str(image),
            "dump_inference_scope": "separate_bound_probe_inference_before_measurement",
            "file": boundary_file.name,
            "dtype": str(boundary.dtype),
            "shape": [int(x) for x in boundary.shape],
            "nbytes": int(boundary.nbytes),
            "trt_input_name": str(
                boundary_meta.get("trt_input_name") or ""
            ),
            "trt_input_dtype": str(boundary.dtype),
            "trt_input_bytes": int(boundary.nbytes),
            "boundary_shape_source": "tensorrt_input_binding",
            "input_dump": input_file.name,
            "input_name": str(input_name),
            "input_shape": [int(x) for x in input_value.shape],
            "preprocess": {
                "task": "detection",
                "mode": str(args.preprocess_mode_effective),
                "pad_value": (
                    int(args.letterbox_pad_value)
                    if args.preprocess_mode_effective == "letterbox" else 0
                ),
                "rgb": True,
            },
        }, indent=2), encoding="utf-8")
        _annotate_boundary_manifest(
            str(manifest), image, str(args.precision),
        )
        _seal_manifest_payload_files(manifest)
        result["native_fifo_boundary_manifest"] = str(manifest)
    return result


def _run_hailo8_python_detection(
    *,
    bs: Path,
    case: str,
    hef: Path,
    engine: Path,
    image: Path,
    work: Path,
    args: argparse.Namespace,
    quality_binding: Mapping[str, Any] | None,
    expected_boundary_name: str = "",
    expected_boundary_shape: list[int] | None = None,
    expected_boundary_dtype: str = "",
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Run fresh Python VStreams/TRT/Completion runtimes per repetition."""
    model_id = str(args.model_id or _benchmark_model_id(bs))
    quantized = str(args.hailo_format) != "float32"
    probe_backend, probe_prepared, probe_trt = (
        _open_hailo8_python_runtime(
            hef=hef, engine=engine, work=work, args=args,
            runtime_label="contract_probe",
        )
    )
    try:
        inputs = _hailo8_python_input(
            probe_prepared,
            image,
            quantized=quantized,
            preprocess_mode=str(args.preprocess_mode_effective),
            letterbox_pad_value=int(args.letterbox_pad_value),
            prepared_input_rgb=str(getattr(args, "prepared_input_rgb", "") or ""),
            expected_prepared_input_sha256=str(getattr(args, "expected_prepared_input_sha256", "") or ""),
        )
        hailo_outputs = probe_backend.run(
            probe_prepared, dict(inputs)
        ).outputs
        trt_name, probe_boundary, boundary_meta = (
            _hailo8_python_boundary(
                hailo_outputs,
                probe_trt,
                expected_name=expected_boundary_name,
                expected_shape=expected_boundary_shape,
                expected_dtype=expected_boundary_dtype,
            )
        )
        probe_outputs = probe_trt.run({trt_name: probe_boundary})
        completion_contract = _hailo8_detection_completion_contract(
            benchmark_set=bs,
            model_id=model_id,
            outputs=probe_outputs,
            inputs=inputs,
            image_path=image,
            preprocess_mode=str(args.preprocess_mode_effective),
            letterbox_pad_value=int(args.letterbox_pad_value),
        )
        # A contract probe is not the measured sentinel. Fast mode only saves
        # measured outputs; legacy mode keeps the probe at a separate path. The
        # input/boundary probe remains explicitly labelled in both modes.
        probe_args = argparse.Namespace(**vars(args))
        if str(getattr(args, "completion_runtime_mode", "fast_oracle_outside_timing")) == "fast_oracle_outside_timing":
            probe_args.dump_outputs = False
        elif args.dump_outputs:
            probe_output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else work / "native_fifo_outputs"
            probe_args.output_dir = str(probe_output_root / "contract_probe")
        probe_evidence = _dump_hailo8_python_semantic_evidence(
            work=work,
            inputs=inputs,
            boundary=probe_boundary,
            boundary_meta=boundary_meta,
            trt_outputs=probe_outputs,
            image=image,
            args=probe_args,
        )
    finally:
        _close_hailo8_python_runtime(
            probe_backend, probe_prepared, probe_trt,
        )
    prepared_name, prepared_value = next(iter(inputs.items()))
    prepared_input = (
        work / "contract_artifacts"
        / "hailo8_prepared_input_python.bin"
    )
    prepared_input.parent.mkdir(parents=True, exist_ok=True)
    prepared_input.write_bytes(
        np.ascontiguousarray(prepared_value).tobytes(order="C")
    )
    repetition_payloads: list[dict[str, Any]] = []
    input_image_sha256 = _sha256_file(image)
    for repetition_index in range(int(args.repetitions)):
        repetition_id, runtime_instance_id = _fresh_hailo_runtime_identity(repetition_index + 1)
        semantic_evidence = {
            key: value for key, value in probe_evidence.items()
            if key != "native_fifo_output_manifest"
        }
        backend, prepared, trt = _open_hailo8_python_runtime(
            hef=hef, engine=engine, work=work, args=args,
            runtime_label=f"repetition_{repetition_index + 1:03d}",
        )
        try:
            repeat_inputs = _hailo8_reuse_python_input(prepared, inputs)
            runtime_mode = str(
                getattr(
                    args,
                    "completion_runtime_mode",
                    "fast_oracle_outside_timing",
                )
            )
            runtime_type = (
                FastDetectionCompletionRuntime
                if runtime_mode == "fast_oracle_outside_timing"
                else DetectionCompletionRuntime
            )
            completion_runtime = runtime_type(completion_contract)
            if isinstance(completion_runtime, FastDetectionCompletionRuntime):
                completion_runtime.bind_sentinel_context({
                    "repetition_id": repetition_id,
                    "runtime_instance_id": runtime_instance_id,
                    "process_local_repetition_index": repetition_index + 1,
                    "input_image_sha256": input_image_sha256,
                })
            warmup_runtime = (
                runtime_type(completion_contract)
                if int(args.warmup) > 0 else None
            )
            row = _hailo8_python_fifo_run(
                backend,
                prepared,
                repeat_inputs,
                trt,
                frames=int(args.frames),
                warmup=int(args.warmup),
                queue_depth=int(args.queue_depth),
                duration_s=float(args.duration_s or 0.0),
                completion_runtime=completion_runtime,
                warmup_completion_runtime=warmup_runtime,
                expected_boundary_name=expected_boundary_name,
                expected_boundary_shape=expected_boundary_shape,
                expected_boundary_dtype=expected_boundary_dtype,
            )
            if (
                isinstance(completion_runtime, FastDetectionCompletionRuntime)
                and args.dump_outputs
            ):
                output_args = argparse.Namespace(**vars(args))
                output_args.dump_boundary = False
                if repetition_index != int(args.repetitions) - 1:
                    output_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else work / "native_fifo_outputs"
                    output_args.output_dir = str(output_root / f"repetition_{repetition_index + 1:03d}")
                semantic_evidence.update(_dump_hailo8_python_semantic_evidence(
                    work=work, inputs=repeat_inputs, boundary=probe_boundary,
                    boundary_meta=boundary_meta,
                    trt_outputs=completion_runtime.last_source_outputs,
                    image=image, args=output_args,
                    output_scope="last_measured_completion_source_outputs",
                    completion_attestation=row["completed_task_endpoint_attestation"],
                ))
            elif args.dump_outputs:
                # Legacy strict mode has per-frame result hashes but no owned
                # final source snapshot. Its probe must not claim measured scope.
                semantic_evidence["native_fifo_output_manifest"] = probe_evidence["native_fifo_output_manifest"]
        finally:
            _close_hailo8_python_runtime(backend, prepared, trt)
        row.update({
            "runtime_instance_id": runtime_instance_id,
            "repetition_id": repetition_id,
            "process_local_repetition_index": repetition_index + 1,
            "repetition_runtime_scope": (
                "fresh_hailo_vstreams_trt_completion_runtime_per_repetition"
            ),
            "completion_runtime_mode": str(
                getattr(
                    args,
                    "completion_runtime_mode",
                    "fast_oracle_outside_timing",
                )
            ),
            "prepared_input_shape": [
                int(x) for x in prepared_value.shape
            ],
            "prepared_input_dtype": str(prepared_value.dtype),
            "prepared_input_name": str(prepared_name),
            "hailo_runtime_output_count": 1,
            "hailo_runtime_output_name": str(
                boundary_meta.get("runtime_name") or ""
            ),
            "hailo_runtime_output_frame_bytes": int(
                probe_boundary.nbytes
            ),
            "native_split_quality_runtime_boundary_verified": True,
            "trt_input_dtype": str(probe_boundary.dtype),
            "trt_input_bytes": int(probe_boundary.nbytes),
            "hef": str(hef),
            "engine": str(engine),
            "task": "detection",
            "hailo_format": str(args.hailo_format),
            "preprocess_mode_effective": str(
                args.preprocess_mode_effective
            ),
            "prepared_feed_contract": (
                "exact_prepared_feed_loaded_outside_counted_loop"
            ),
            **semantic_evidence,
        })
        repetition_payloads.append(row)
    payload = _aggregate_repetition_payloads(repetition_payloads)
    return payload, prepared_input, completion_contract

CPP_SOURCE = r'''
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static inline double ms_since(const Clock::time_point &a, const Clock::time_point &b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static inline void check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(e);
        throw std::runtime_error(oss.str());
    }
}

struct Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
    }
};
static Logger g_logger;

static size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: return 8;
        case nvinfer1::DataType::kBF16: return 2;
#endif
        default: throw std::runtime_error("unsupported TensorRT dtype");
    }
}

static std::string dtype_name(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kUINT8: return "uint8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL: return "bool";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kINT64: return "int64";
        case nvinfer1::DataType::kBF16: return "bf16";
#endif
        default: return "unknown";
    }
}

static size_t volume(const nvinfer1::Dims &d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        if (d.d[i] < 0) throw std::runtime_error("dynamic TensorRT dims are not supported by native FIFO smoke yet");
        v *= static_cast<size_t>(d.d[i]);
    }
    return v;
}

struct Stats {
    std::vector<double> xs;
    void add(double v) { xs.push_back(v); }
    double mean() const { return xs.empty() ? 0.0 : std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size(); }
    double min() const { return xs.empty() ? 0.0 : *std::min_element(xs.begin(), xs.end()); }
    double max() const { return xs.empty() ? 0.0 : *std::max_element(xs.begin(), xs.end()); }
};

template<typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t cap) : cap_(cap) {}
    void push(T v) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_full_.wait(lk, [&]{ return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push(std::move(v));
        cv_not_empty_.notify_one();
    }
    bool pop(T &out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_empty_.wait(lk, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        cv_not_full_.notify_one();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }
private:
    size_t cap_;
    std::mutex mu_;
    std::condition_variable cv_not_empty_, cv_not_full_;
    std::queue<T> q_;
    bool closed_ = false;
};

struct Options {
    std::string hef;
    std::string engine;
    std::string image;
    std::string out_json = "native_fifo_results.json";
    std::string device_id;
    std::string hailo_format = "uint8"; // uint8|auto
    int frames = 100;
    int warmup = 10;
    double duration_s = 0.0; // measured workload duration; 0 means frame-count mode
    int queue_depth = 3;
    bool copy_outputs = true;
    bool dump_outputs = false;
    bool dump_boundary = false;
    bool reuse_preprocessed_input = false;
    std::string task;
    std::string preprocess_mode = "auto";
    int letterbox_pad_value = 0;
    std::string output_dir;
    std::string boundary_dir;
    std::string prepared_input_rgb;
    std::string prepared_input_out;
    int expected_output_count = 0;
    size_t expected_output_bytes = 0;
    std::string expected_output_name;
};

static Options parse_args(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return argv[++i]; };
        if (a == "--hef") o.hef = next();
        else if (a == "--engine") o.engine = next();
        else if (a == "--image" || a == "--images") o.image = next();
        else if (a == "--out") o.out_json = next();
        else if (a == "--device-id") o.device_id = next();
        else if (a == "--hailo-format") o.hailo_format = next();
        else if (a == "--frames") o.frames = std::stoi(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--duration-s" || a == "--duration") o.duration_s = std::stod(next());
        else if (a == "--queue-depth") o.queue_depth = std::stoi(next());
        else if (a == "--copy-outputs") o.copy_outputs = std::stoi(next()) != 0;
        else if (a == "--dump-outputs") o.dump_outputs = std::stoi(next()) != 0;
        else if (a == "--dump-boundary") o.dump_boundary = std::stoi(next()) != 0;
        else if (a == "--reuse-preprocessed-input") o.reuse_preprocessed_input = std::stoi(next()) != 0;
        else if (a == "--task") o.task = next();
        else if (a == "--preprocess-mode") o.preprocess_mode = next();
        else if (a == "--letterbox-pad-value" || a == "--letterbox-pad") o.letterbox_pad_value = std::stoi(next());
        else if (a == "--output-dir") o.output_dir = next();
        else if (a == "--boundary-dir") o.boundary_dir = next();
        else if (a == "--prepared-input-rgb") o.prepared_input_rgb = next();
        else if (a == "--prepared-input-out") o.prepared_input_out = next();
        else if (a == "--expected-output-count") o.expected_output_count = std::stoi(next());
        else if (a == "--expected-output-bytes") o.expected_output_bytes = static_cast<size_t>(std::stoull(next()));
        else if (a == "--expected-output-name") o.expected_output_name = next();
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: split_native_hailo_trt_fifo --hef part1.hef --engine part2.engine --image img-or-dir [--frames N --warmup N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (o.hef.empty() || o.engine.empty() || o.image.empty()) throw std::runtime_error("--hef, --engine and --image are required");
    if (o.frames <= 0 && o.duration_s <= 0.0) throw std::runtime_error("--frames must be >0 unless --duration-s is set");
    if (o.warmup < 0) throw std::runtime_error("--warmup must be >=0");
    if (o.queue_depth < 1) throw std::runtime_error("--queue-depth must be >=1");
    if (o.task != "classification" && o.task != "detection") throw std::runtime_error("--task must be classification or detection");
    if (o.preprocess_mode == "auto") o.preprocess_mode = (o.task == "detection") ? "letterbox" : "resize";
    if (o.preprocess_mode != "resize" && o.preprocess_mode != "letterbox") throw std::runtime_error("--preprocess-mode must be auto, resize, or letterbox");
    return o;
}

static std::vector<std::string> list_images(const std::string &path) {
    std::vector<std::string> xs;
    fs::path p(path);
    if (fs::is_directory(p)) {
        for (auto &e : fs::directory_iterator(p)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") xs.push_back(e.path().string());
        }
        std::sort(xs.begin(), xs.end());
    } else {
        xs.push_back(path);
    }
    if (xs.empty()) throw std::runtime_error("no images found: " + path);
    return xs;
}

static cv::Mat letterbox_rgb_uint8(const cv::Mat &bgr, int w, int h, int pad_value) {
    if (bgr.empty()) throw std::runtime_error("empty image");
    double scale = std::min(w / static_cast<double>(bgr.cols), h / static_cast<double>(bgr.rows));
    int nw = std::max(1, static_cast<int>(std::round(bgr.cols * scale)));
    int nh = std::max(1, static_cast<int>(std::round(bgr.rows * scale)));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(nw, nh));
    cv::Mat canvas(h, w, bgr.type(), cv::Scalar(pad_value, pad_value, pad_value));
    int left = (w - nw) / 2;
    int top = (h - nh) / 2;
    resized.copyTo(canvas(cv::Rect(left, top, nw, nh)));
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

static cv::Mat resize_rgb_uint8(const cv::Mat &bgr, int w, int h) {
    if (bgr.empty()) throw std::runtime_error("empty image");
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(w, h));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

static cv::Mat preprocess_rgb_uint8(const cv::Mat &bgr, int w, int h, const Options &opt) {
    if (opt.preprocess_mode == "letterbox") {
        return letterbox_rgb_uint8(bgr, w, h, opt.letterbox_pad_value);
    }
    if (opt.preprocess_mode == "resize") {
        return resize_rgb_uint8(bgr, w, h);
    }
    throw std::runtime_error("unresolved preprocess mode");
}

class HailoStage {
public:
    explicit HailoStage(const Options &o) {
        // HailoRT treats an explicit empty device vector as device_count=0.
        // For auto-discovery we must call the no-argument overload instead.
        hailort::Expected<std::unique_ptr<hailort::VDevice>> vdev_exp =
            o.device_id.empty() ? hailort::VDevice::create() : hailort::VDevice::create(std::vector<std::string>{o.device_id});
        if (!vdev_exp) {
            std::ostringstream oss;
            oss << "VDevice::create failed status=" << static_cast<int>(vdev_exp.status())
                << " device_id='" << o.device_id << "'";
            throw std::runtime_error(oss.str());
        }
        vdev_ = std::move(vdev_exp.value());
        auto hef_exp = hailort::Hef::create(o.hef);
        if (!hef_exp) throw std::runtime_error("Hef::create failed: " + o.hef);
        auto cfg_exp = hef_exp->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!cfg_exp) throw std::runtime_error("create_configure_params failed");
        auto ng_exp = vdev_->configure(hef_exp.value(), cfg_exp.value());
        if (!ng_exp) throw std::runtime_error("VDevice configure failed");
        if (ng_exp->size() != 1) throw std::runtime_error("expected one Hailo network group");
        network_group_ = std::move(ng_exp->front());
        hailo_format_type_t fmt = HAILO_FORMAT_TYPE_AUTO;
        if (o.hailo_format == "uint8" || o.hailo_format == "raw_uint8") fmt = HAILO_FORMAT_TYPE_UINT8;
        else if (o.hailo_format == "float32" || o.hailo_format == "dequant_float32") fmt = HAILO_FORMAT_TYPE_FLOAT32;
        auto vs_exp = hailort::VStreamsBuilder::create_vstreams(*network_group_, true, fmt);
        if (!vs_exp) throw std::runtime_error("create_vstreams failed");
        vstreams_ = std::move(vs_exp.value());
        if (vstreams_.first.size() != 1 || vstreams_.second.size() != 1) throw std::runtime_error("native split requires exactly one Hailo input and one output vstream");
        auto in_info = vstreams_.first.front().get_info();
        input_h_ = in_info.shape.height;
        input_w_ = in_info.shape.width;
        input_features_ = in_info.shape.features;
        input_frame_size_ = vstreams_.first.front().get_frame_size();
        output_frame_size_ = vstreams_.second.front().get_frame_size();
        output_name_ = vstreams_.second.front().get_info().name;
        if (o.expected_output_count > 0 && o.expected_output_count != static_cast<int>(vstreams_.second.size())) throw std::runtime_error("Quality boundary output count mismatch");
        if (o.expected_output_bytes > 0 && o.expected_output_bytes != output_frame_size_) throw std::runtime_error("Quality boundary output frame byte-size mismatch");
        if (!o.expected_output_name.empty() && o.expected_output_name != output_name_) throw std::runtime_error("Quality boundary output name mismatch");
        quality_boundary_verified_ = o.expected_output_count == 1 && o.expected_output_bytes == output_frame_size_ && !o.expected_output_name.empty() && o.expected_output_name == output_name_;
        std::cerr << "[native-fifo][hailo] input_shape=" << input_w_ << "x" << input_h_ << "x" << input_features_
                  << " input_frame_size=" << input_frame_size_ << " output_frame_size=" << output_frame_size_ << std::endl;
    }
    int input_w() const { return input_w_; }
    int input_h() const { return input_h_; }
    size_t output_frame_size() const { return output_frame_size_; }
    const std::string &output_name() const { return output_name_; }
    bool quality_boundary_verified() const { return quality_boundary_verified_; }

    void infer(const cv::Mat &rgb_uint8, std::vector<uint8_t> &out) {
        std::vector<uint8_t> input_bytes;
        std::vector<float> input_f32;
        const size_t uint8_size = static_cast<size_t>(rgb_uint8.total() * rgb_uint8.elemSize());
        if (input_frame_size_ == uint8_size) {
            const uint8_t *src = rgb_uint8.ptr<uint8_t>(0);
            auto status = vstreams_.first.front().write(hailort::MemoryView(const_cast<uint8_t*>(src), uint8_size));
            if (status != HAILO_SUCCESS) throw std::runtime_error("Hailo input write failed");
        } else if (input_frame_size_ == uint8_size * sizeof(float)) {
            input_f32.resize(uint8_size);
            const uint8_t *src = rgb_uint8.ptr<uint8_t>(0);
            for (size_t i = 0; i < uint8_size; ++i) input_f32[i] = static_cast<float>(src[i]) / 255.0f;
            auto status = vstreams_.first.front().write(hailort::MemoryView(input_f32.data(), input_f32.size() * sizeof(float)));
            if (status != HAILO_SUCCESS) throw std::runtime_error("Hailo input write(float32) failed");
        } else {
            std::ostringstream oss; oss << "unsupported Hailo input_frame_size=" << input_frame_size_ << " image_bytes=" << uint8_size;
            throw std::runtime_error(oss.str());
        }
        out.resize(output_frame_size_);
        auto status = vstreams_.second.front().read(hailort::MemoryView(out.data(), out.size()));
        if (status != HAILO_SUCCESS) throw std::runtime_error("Hailo output read failed");
    }
private:
    std::unique_ptr<hailort::VDevice> vdev_;
    std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
    std::pair<std::vector<hailort::InputVStream>, std::vector<hailort::OutputVStream>> vstreams_;
    int input_w_ = 0, input_h_ = 0, input_features_ = 0;
    size_t input_frame_size_ = 0, output_frame_size_ = 0;
    std::string output_name_;
    bool quality_boundary_verified_ = false;
};

struct TensorBinding {
    std::string name;
    bool is_input = false;
    nvinfer1::DataType dtype;
    nvinfer1::Dims dims;
    size_t elem_count = 0;
    size_t bytes = 0;
    void *dev = nullptr;
    void *host = nullptr;
};

class TRTStage {
public:
    explicit TRTStage(const std::string &engine_path, bool copy_outputs) : copy_outputs_(copy_outputs) {
        initLibNvInferPlugins(&g_logger, "");
        std::ifstream f(engine_path, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open engine: " + engine_path);
        f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0, std::ios::beg);
        std::vector<char> data(n); f.read(data.data(), n);
        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed");
        check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        const int nb = engine_->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            TensorBinding b;
            b.name = engine_->getIOTensorName(i);
            b.is_input = engine_->getTensorIOMode(b.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
            b.dtype = engine_->getTensorDataType(b.name.c_str());
            b.dims = engine_->getTensorShape(b.name.c_str());
            b.elem_count = volume(b.dims);
            b.bytes = b.elem_count * dtype_size(b.dtype);
            check_cuda(cudaMalloc(&b.dev, b.bytes), "cudaMalloc");
            check_cuda(cudaMallocHost(&b.host, b.bytes), "cudaMallocHost");
            if (!context_->setTensorAddress(b.name.c_str(), b.dev)) throw std::runtime_error("setTensorAddress failed for " + b.name);
            if (b.is_input) input_index_ = bindings_.size();
            bindings_.push_back(b);
        }
        if (input_index_ < 0) throw std::runtime_error("TRT engine has no input");
        std::cerr << "[native-fifo][trt] input=" << input().name << " dtype=" << dtype_name(input().dtype) << " bytes=" << input().bytes << std::endl;
    }
    ~TRTStage() {
        for (auto &b : bindings_) { if (b.dev) cudaFree(b.dev); if (b.host) cudaFreeHost(b.host); }
        if (stream_) cudaStreamDestroy(stream_);
    }
    const TensorBinding &input() const { return bindings_.at(static_cast<size_t>(input_index_)); }

    void copy_input_from_boundary(const std::vector<uint8_t> &boundary) {
        TensorBinding &in = bindings_.at(static_cast<size_t>(input_index_));
        if (in.dtype == nvinfer1::DataType::kUINT8 || in.dtype == nvinfer1::DataType::kINT8) {
            if (boundary.size() != in.bytes) {
                std::ostringstream oss; oss << "boundary byte size " << boundary.size() << " does not match TRT input bytes " << in.bytes;
                throw std::runtime_error(oss.str());
            }
            std::memcpy(in.host, boundary.data(), in.bytes);
        } else if (in.dtype == nvinfer1::DataType::kFLOAT) {
            if (boundary.size() == in.bytes) {
                // Hailo VStream can be requested as FLOAT32. In that case the
                // boundary buffer is already a raw float32 tensor and must be
                // copied byte-for-byte into the TensorRT input binding.
                std::memcpy(in.host, boundary.data(), in.bytes);
            } else if (boundary.size() == in.elem_count) {
                // Legacy fast-path for raw uint8 boundary into a float input.
                // This is diagnostic only and is not semantically equivalent
                // to a real dequantized Hailo output.
                float *dst = static_cast<float*>(in.host);
                for (size_t i = 0; i < boundary.size(); ++i) dst[i] = static_cast<float>(boundary[i]);
            } else {
                std::ostringstream oss; oss << "boundary byte size " << boundary.size()
                    << " does not match float TRT input bytes " << in.bytes
                    << " or elems " << in.elem_count;
                throw std::runtime_error(oss.str());
            }
        } else if (in.dtype == nvinfer1::DataType::kHALF && boundary.size() == in.bytes) {
            std::memcpy(in.host, boundary.data(), in.bytes);
        } else {
            throw std::runtime_error("unsupported TRT input dtype for boundary: " + dtype_name(in.dtype));
        }
    }

    void run() {
        TensorBinding &in = bindings_.at(static_cast<size_t>(input_index_));
        check_cuda(cudaMemcpyAsync(in.dev, in.host, in.bytes, cudaMemcpyHostToDevice, stream_), "H2D");
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
        if (copy_outputs_) {
            for (auto &b : bindings_) {
                if (b.is_input) continue;
                check_cuda(cudaMemcpyAsync(b.host, b.dev, b.bytes, cudaMemcpyDeviceToHost, stream_), "D2H");
            }
        }
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

    static std::string safe_name(std::string x) {
        for (char &c : x) {
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')) c = '_';
        }
        return x.empty() ? std::string("tensor") : x;
    }

    std::string dump_outputs(const std::string &dir) {
        if (dir.empty()) return std::string();
        fs::create_directories(dir);
        std::string manifest = (fs::path(dir) / "native_fifo_outputs_manifest.json").string();
        std::ofstream js(manifest);
        js << "{\n";
        js << "  \"schema\": \"onnx-splitpoint/native-fifo-output-dump\",\n";
        js << "  \"schema_version\": 1,\n";
        js << "  \"outputs\": [\n";
        bool first = true;
        for (auto &b : bindings_) {
            if (b.is_input || !b.host || b.bytes == 0) continue;
            std::string fname = safe_name(b.name) + ".bin";
            fs::path outp = fs::path(dir) / fname;
            std::ofstream f(outp, std::ios::binary);
            f.write(static_cast<const char*>(b.host), static_cast<std::streamsize>(b.bytes));
            if (!first) js << ",\n";
            first = false;
            js << "    {\"name\": \"" << b.name << "\", \"file\": \"" << fname << "\", \"dtype\": \"" << dtype_name(b.dtype) << "\", \"bytes\": " << b.bytes << ", \"shape\": [";
            for (int i = 0; i < b.dims.nbDims; ++i) { if (i) js << ", "; js << b.dims.d[i]; }
            js << "]}";
        }
        js << "\n  ]\n}\n";
        return manifest;
    }
private:
    struct RuntimeDel { void operator()(nvinfer1::IRuntime *p) const { delete p; } };
    struct EngineDel { void operator()(nvinfer1::ICudaEngine *p) const { delete p; } };
    struct ContextDel { void operator()(nvinfer1::IExecutionContext *p) const { delete p; } };
    std::unique_ptr<nvinfer1::IRuntime, RuntimeDel> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDel> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, ContextDel> context_;
    std::vector<TensorBinding> bindings_;
    int input_index_ = -1;
    cudaStream_t stream_ = nullptr;
    bool copy_outputs_ = true;
};

struct Slot {
    int seq = 0;
    std::string image_path;
    std::vector<uint8_t> boundary;
    double pre_ms = 0.0;
    double p1_ms = 0.0;
};

int main(int argc, char **argv) {
    try {
        Options opt = parse_args(argc, argv);
        auto images = list_images(opt.image);
        HailoStage hailo(opt);
        TRTStage trt(opt.engine, opt.copy_outputs);
        cv::Mat prepared_rgb;
        std::string prepared_image_path;
        double prepared_feed_setup_ms = 0.0;
        if (!opt.prepared_input_rgb.empty()) {
            auto prep0 = Clock::now();
            prepared_image_path = images.front();
            const size_t expected_bytes = static_cast<size_t>(hailo.input_w()) * static_cast<size_t>(hailo.input_h()) * 3U;
            std::ifstream prepared_file(opt.prepared_input_rgb, std::ios::binary | std::ios::ate);
            if (!prepared_file) throw std::runtime_error("failed to open --prepared-input-rgb");
            const auto prepared_size = static_cast<size_t>(prepared_file.tellg());
            if (prepared_size != expected_bytes) throw std::runtime_error("prepared RGB input size mismatch");
            prepared_file.seekg(0, std::ios::beg);
            prepared_rgb = cv::Mat(hailo.input_h(), hailo.input_w(), CV_8UC3);
            prepared_file.read(reinterpret_cast<char*>(prepared_rgb.data), static_cast<std::streamsize>(expected_bytes));
            if (!prepared_file) throw std::runtime_error("failed to read prepared RGB input");
            auto prep1 = Clock::now();
            prepared_feed_setup_ms = ms_since(prep0, prep1);
        } else if (opt.reuse_preprocessed_input || !opt.prepared_input_out.empty()) {
            if (images.size() != 1) throw std::runtime_error("--reuse-preprocessed-input requires one exact image file");
            auto prep0 = Clock::now();
            prepared_image_path = images.front();
            cv::Mat prepared_bgr = cv::imread(prepared_image_path);
            if (prepared_bgr.empty()) throw std::runtime_error("failed to read prepared input image");
            prepared_rgb = preprocess_rgb_uint8(prepared_bgr, hailo.input_w(), hailo.input_h(), opt);
            if (!prepared_rgb.isContinuous()) prepared_rgb = prepared_rgb.clone();
            auto prep1 = Clock::now();
            prepared_feed_setup_ms = ms_since(prep0, prep1);
        }
        if (!opt.prepared_input_out.empty()) {
            fs::path prepared_out(opt.prepared_input_out);
            if (!prepared_out.parent_path().empty()) fs::create_directories(prepared_out.parent_path());
            std::ofstream prepared_stream(prepared_out, std::ios::binary);
            prepared_stream.write(reinterpret_cast<const char*>(prepared_rgb.data), static_cast<std::streamsize>(prepared_rgb.total() * prepared_rgb.elemSize()));
            if (!prepared_stream) throw std::runtime_error("failed to write prepared input contract artifact");
        }
        const bool duration_mode = opt.duration_s > 0.0;
        // Stabilize both stages and fully drain every warm-up item before the
        // measured FIFO workers are even created.  This prevents old warm-up
        // work from leaking into the first measured makespan interval.
        std::vector<uint8_t> warmup_boundary(hailo.output_frame_size());
        for (int seq = 0; seq < opt.warmup; ++seq) {
            cv::Mat rgb;
            if (opt.reuse_preprocessed_input || !opt.prepared_input_rgb.empty()) {
                rgb = prepared_rgb;
            } else {
                const std::string &path = images[static_cast<size_t>(seq) % images.size()];
                cv::Mat bgr = cv::imread(path);
                if (bgr.empty()) throw std::runtime_error("failed to read warm-up image");
                rgb = preprocess_rgb_uint8(bgr, hailo.input_w(), hailo.input_h(), opt);
            }
            hailo.infer(rgb, warmup_boundary);
            trt.copy_input_from_boundary(warmup_boundary);
            trt.run();
        }

        const int total = duration_mode ? std::numeric_limits<int>::max() : opt.frames;
        std::atomic<int> measured_frames{0};
        std::vector<Slot> slots(static_cast<size_t>(opt.queue_depth));
        for (auto &s : slots) s.boundary.resize(hailo.output_frame_size());
        BlockingQueue<int> free_q(static_cast<size_t>(opt.queue_depth));
        BlockingQueue<int> filled_q(static_cast<size_t>(opt.queue_depth));
        for (int i = 0; i < opt.queue_depth; ++i) free_q.push(i);
        Stats pre_s, p1_s, handoff_s, p2run_s, p1thread_s, p2thread_s, latency_s;
        Clock::time_point meas_start, meas_end;
        std::mutex start_mu;
        std::condition_variable start_cv;
        int workers_ready = 0;
        bool measurement_released = false;
        auto wait_for_measurement_start = [&]() {
            std::unique_lock<std::mutex> lk(start_mu);
            ++workers_ready;
            start_cv.notify_all();
            start_cv.wait(lk, [&]() { return measurement_released; });
        };

        auto p1_thread = [&]() {
            wait_for_measurement_start();
            for (int seq = 0; seq < total; ++seq) {
                if (duration_mode) {
                    double elapsed_s = std::chrono::duration<double>(Clock::now() - meas_start).count();
                    if (elapsed_s >= opt.duration_s) break;
                }
                int idx;
                if (!free_q.pop(idx)) break;
                if (duration_mode) {
                    double elapsed_s = std::chrono::duration<double>(Clock::now() - meas_start).count();
                    if (elapsed_s >= opt.duration_s) {
                        free_q.push(idx);
                        break;
                    }
                }
                Slot &slot = slots[static_cast<size_t>(idx)];
                // Keep the historical absolute sequence/image mapping even
                // though warm-up is now a separate, fully drained phase.
                slot.seq = opt.warmup + seq;
                auto t0 = Clock::now();
                cv::Mat rgb;
                if (opt.reuse_preprocessed_input || !opt.prepared_input_rgb.empty()) {
                    slot.image_path = prepared_image_path;
                    rgb = prepared_rgb;
                } else {
                    slot.image_path = images[static_cast<size_t>(slot.seq) % images.size()];
                    cv::Mat bgr = cv::imread(slot.image_path);
                    if (bgr.empty()) throw std::runtime_error("failed to read image");
                    rgb = preprocess_rgb_uint8(bgr, hailo.input_w(), hailo.input_h(), opt);
                }
                auto t1 = Clock::now();
                hailo.infer(rgb, slot.boundary);
                auto t2 = Clock::now();
                slot.pre_ms = ms_since(t0,t1);
                slot.p1_ms = ms_since(t1,t2);
                filled_q.push(idx);
            }
            filled_q.close();
        };

        auto p2_thread = [&]() {
            wait_for_measurement_start();
            int idx;
            while (filled_q.pop(idx)) {
                Slot &slot = slots[static_cast<size_t>(idx)];
                auto t0 = Clock::now();
                trt.copy_input_from_boundary(slot.boundary);
                auto t1 = Clock::now();
                trt.run();
                auto t2 = Clock::now();
                measured_frames.fetch_add(1);
                pre_s.add(slot.pre_ms);
                p1_s.add(slot.p1_ms);
                handoff_s.add(ms_since(t0,t1));
                p2run_s.add(ms_since(t1,t2));
                p1thread_s.add(slot.pre_ms + slot.p1_ms);
                p2thread_s.add(ms_since(t0,t2));
                latency_s.add(slot.pre_ms + slot.p1_ms + ms_since(t0,t2));
                meas_end = t2;
                free_q.push(idx);
            }
        };

        std::thread a(p1_thread);
        std::thread b(p2_thread);
        {
            std::unique_lock<std::mutex> lk(start_mu);
            start_cv.wait(lk, [&]() { return workers_ready == 2; });
            meas_start = Clock::now();
            measurement_released = true;
        }
        start_cv.notify_all();
        a.join();
        b.join();
        double makespan_ms = ms_since(meas_start, meas_end);
        int measured_count = measured_frames.load();
        if (measured_count <= 0 || makespan_ms <= 0.0) throw std::runtime_error("no measured frames completed");
        double fps_makespan = 1000.0 * static_cast<double>(measured_count) / makespan_ms;
        double paper_cycle = std::max(p1thread_s.mean(), p2thread_s.mean());
        double paper_fps = paper_cycle > 0 ? 1000.0 / paper_cycle : 0.0;

        // Semantic evidence is deliberately produced by one separate bound
        // inference after the performance window.  No measured frame performs
        // an input/boundary dump copy, and the output dump corresponds exactly
        // to this same boundary inference.
        std::vector<uint8_t> last_boundary_dump;
        std::vector<uint8_t> last_input_dump;
        int last_boundary_seq = opt.warmup + measured_count - 1;
        std::string last_boundary_image;
        if (opt.dump_boundary || opt.dump_outputs) {
            cv::Mat dump_rgb;
            if (opt.reuse_preprocessed_input || !opt.prepared_input_rgb.empty()) {
                last_boundary_image = prepared_image_path;
                dump_rgb = prepared_rgb;
            } else {
                last_boundary_image = images[static_cast<size_t>(last_boundary_seq) % images.size()];
                cv::Mat dump_bgr = cv::imread(last_boundary_image);
                if (dump_bgr.empty()) throw std::runtime_error("failed to read dump image");
                dump_rgb = preprocess_rgb_uint8(dump_bgr, hailo.input_w(), hailo.input_h(), opt);
            }
            hailo.infer(dump_rgb, last_boundary_dump);
            trt.copy_input_from_boundary(last_boundary_dump);
            trt.run();
            if (opt.dump_boundary) {
                const size_t rgb_bytes = static_cast<size_t>(dump_rgb.total() * dump_rgb.elemSize());
                const uint8_t *rgb_src = dump_rgb.ptr<uint8_t>(0);
                last_input_dump.assign(rgb_src, rgb_src + rgb_bytes);
            }
        }
        std::string boundary_manifest;
        if (opt.dump_boundary && !last_boundary_dump.empty()) {
            fs::path bdir = opt.boundary_dir.empty() ? (fs::path(opt.out_json).parent_path() / "native_fifo_boundary") : fs::path(opt.boundary_dir);
            fs::create_directories(bdir);
            std::string boundary_dtype = (opt.hailo_format == "float32" || opt.hailo_format == "dequant_float32") ? "float32" : "uint8";
            fs::path bin = bdir / ((boundary_dtype == "float32") ? "boundary_float32.bin" : "boundary_uint8.bin");
            std::ofstream bout(bin, std::ios::binary);
            bout.write(reinterpret_cast<const char*>(last_boundary_dump.data()), static_cast<std::streamsize>(last_boundary_dump.size()));
            bout.close();
            fs::path input_bin;
            if (!last_input_dump.empty()) {
                input_bin = bdir / "input_rgb_uint8.bin";
                std::ofstream ib(input_bin, std::ios::binary);
                ib.write(reinterpret_cast<const char*>(last_input_dump.data()), static_cast<std::streamsize>(last_input_dump.size()));
                ib.close();
            }
            fs::path man = bdir / "native_fifo_boundary_manifest.json";
            std::ofstream mj(man);
            mj << "{\n";
            mj << "  \"schema\": \"onnx-splitpoint/native-boundary-dump\",\n";
            mj << "  \"schema_version\": 2,\n";
            mj << "  \"dtype\": \"" << boundary_dtype << "\",\n";
            mj << "  \"nbytes\": " << last_boundary_dump.size() << ",\n";
            mj << "  \"seq\": " << last_boundary_seq << ",\n";
            mj << "  \"trt_input_name\": \"" << trt.input().name << "\",\n";
            mj << "  \"trt_input_dtype\": \"" << dtype_name(trt.input().dtype) << "\",\n";
            mj << "  \"trt_input_bytes\": " << trt.input().bytes << ",\n";
            mj << "  \"shape\": [";
            for (int i = 0; i < trt.input().dims.nbDims; ++i) { if (i) mj << ", "; mj << trt.input().dims.d[i]; }
            mj << "],\n";
            mj << "  \"boundary_shape_source\": \"tensorrt_input_binding\",\n";
            mj << "  \"dump_inference_scope\": \"separate_bound_inference_after_measurement\",\n";
            if (!last_boundary_image.empty()) mj << "  \"image\": \"" << last_boundary_image << "\",\n";
            mj << "  \"file\": \"" << bin.string() << "\",\n";
            if (!input_bin.empty()) mj << "  \"input_dump\": \"" << input_bin.string() << "\",\n";
            if (!input_bin.empty()) mj << "  \"input_shape_hwc\": [" << hailo.input_h() << ", " << hailo.input_w() << ", 3],\n";
            mj << "  \"preprocess\": {\"task\": \"" << opt.task << "\", \"mode\": \"" << opt.preprocess_mode << "_rgb_uint8\", \"pad_value\": " << ((opt.preprocess_mode == "letterbox") ? opt.letterbox_pad_value : 0) << ", \"rgb\": true, \"ort_model_scale\": \"norm\"}\n";
            mj << "}\n";
            mj.close();
            boundary_manifest = man.string();
            std::cout << "[native-fifo] boundary dump: " << boundary_manifest << "\n";
        }
        std::string output_manifest;
        if (opt.dump_outputs) {
            std::string odir = opt.output_dir.empty() ? (fs::path(opt.out_json).parent_path() / "native_fifo_outputs").string() : opt.output_dir;
            output_manifest = trt.dump_outputs(odir);
            std::cout << "[native-fifo] output dump: " << output_manifest << "\n";
        }

        std::cout << "[native-fifo] frames=" << measured_count << " warmup=" << opt.warmup << " duration_s=" << opt.duration_s << " queue_depth=" << opt.queue_depth << "\n";
        std::cout << "[native-fifo] pre=" << pre_s.mean() << " ms p1=" << p1_s.mean() << " ms handoff=" << handoff_s.mean() << " ms p2_run=" << p2run_s.mean() << " ms\n";
        std::cout << "[native-fifo] p1_thread=" << p1thread_s.mean() << " ms p2_thread=" << p2thread_s.mean() << " ms paper_fps=" << paper_fps << " fps_makespan=" << fps_makespan << "\n";

        std::ofstream out(opt.out_json);
        out << std::fixed << std::setprecision(6);
        out << "{\n";
        out << "  \"ok\": true,\n";
        out << "  \"mode\": \"native_hailort_tensorrt_fifo\",\n";
        out << "  \"hef\": \"" << opt.hef << "\",\n";
        out << "  \"engine\": \"" << opt.engine << "\",\n";
        out << "  \"frames\": " << measured_count << ",\n";
        out << "  \"completed_frames\": " << measured_count << ",\n";
        out << "  \"requested_frames\": " << opt.frames << ",\n";
        out << "  \"duration_s\": " << opt.duration_s << ",\n";
        out << "  \"warmup\": " << opt.warmup << ",\n";
        out << "  \"queue_depth\": " << opt.queue_depth << ",\n";
        out << "  \"hailo_format\": \"" << opt.hailo_format << "\",\n";
        out << "  \"task\": \"" << opt.task << "\",\n";
        out << "  \"preprocess_mode_effective\": \"" << opt.preprocess_mode << "\",\n";
        out << "  \"preprocess_pad_value_effective\": " << ((opt.preprocess_mode == "letterbox") ? opt.letterbox_pad_value : 0) << ",\n";
        out << "  \"letterbox_pad_value\": " << opt.letterbox_pad_value << ",\n";
        out << "  \"reuse_preprocessed_input\": " << (opt.reuse_preprocessed_input ? "true" : "false") << ",\n";
        out << "  \"prepared_feed_contract\": \"" << (opt.reuse_preprocessed_input ? "prepared_feed_preprocess_outside_counted_loop" : "per_frame_preprocess") << "\",\n";
        out << "  \"prepared_feed_setup_ms\": " << prepared_feed_setup_ms << ",\n";
        out << "  \"measurement_boundary\": \"workers_ready_to_last_completed_trt_frame\",\n";
        out << "  \"warmup_contract\": \"fully_drained_before_worker_start\",\n";
        out << "  \"dump_inference_scope\": \"separate_bound_inference_after_measurement\",\n";
        out << "  \"trt_host_memory_policy\": \"cuda_pinned_all_bindings\",\n";
        out << "  \"trt_output_materialization_policy\": \"synchronized_d2h_pinned_buffer_no_post_copy\",\n";
        out << "  \"trt_copy_outputs\": " << (opt.copy_outputs ? "true" : "false") << ",\n";
        out << "  \"hailo_runtime_output_count\": 1,\n";
        out << "  \"hailo_runtime_output_name\": \"" << hailo.output_name() << "\",\n";
        out << "  \"hailo_runtime_output_frame_bytes\": " << hailo.output_frame_size() << ",\n";
        out << "  \"native_split_quality_runtime_boundary_verified\": " << (hailo.quality_boundary_verified() ? "true" : "false") << ",\n";
        out << "  \"prepared_input_rgb\": \"" << opt.prepared_input_rgb << "\",\n";
        out << "  \"prepared_input_out\": \"" << opt.prepared_input_out << "\",\n";
        out << "  \"prepared_input_shape\": [" << hailo.input_h() << ", " << hailo.input_w() << ", 3],\n";
        out << "  \"trt_input_dtype\": \"" << dtype_name(trt.input().dtype) << "\",\n";
        out << "  \"trt_input_bytes\": " << trt.input().bytes << ",\n";
        out << "  \"preprocess_ms\": " << pre_s.mean() << ",\n";
        out << "  \"p1_ms\": " << p1_s.mean() << ",\n";
        out << "  \"handoff_ms\": " << handoff_s.mean() << ",\n";
        out << "  \"trt_input_copy_ms\": " << handoff_s.mean() << ",\n";
        out << "  \"p2_run_ms\": " << p2run_s.mean() << ",\n";
        out << "  \"p1_thread_ms\": " << p1thread_s.mean() << ",\n";
        out << "  \"p2_thread_ms\": " << p2thread_s.mean() << ",\n";
        out << "  \"single_latency_model_ms\": " << latency_s.mean() << ",\n";
        out << "  \"paper_equivalent_cycle_ms\": " << paper_cycle << ",\n";
        out << "  \"paper_equivalent_fps\": " << paper_fps << ",\n";
        out << "  \"makespan_ms\": " << makespan_ms << ",\n";
        out << "  \"fps_makespan\": " << fps_makespan;
        if (!output_manifest.empty()) out << ",\n  \"native_fifo_output_manifest\": \"" << output_manifest << "\"";
        if (!boundary_manifest.empty()) out << ",\n  \"native_fifo_boundary_manifest\": \"" << boundary_manifest << "\"";
        out << "\n}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "[native-fifo][error] " << e.what() << std::endl;
        return 2;
    }
}
'''

CMAKE_TXT = r'''
cmake_minimum_required(VERSION 3.16)
project(split_native_hailo_trt_fifo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(HailoRT REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/src/tensorrt/include /usr/local/TensorRT/include)
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/TensorRT/lib)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/TensorRT/lib)
find_library(CUDART_LIB cudart HINTS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu /usr/lib)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h HINTS /usr/local/cuda/include /usr/include)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVINFER_PLUGIN_LIB OR NOT CUDART_LIB OR NOT CUDA_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find TensorRT/CUDA runtime libraries. TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR} NVINFER_LIB=${NVINFER_LIB} NVINFER_PLUGIN_LIB=${NVINFER_PLUGIN_LIB} CUDART_LIB=${CUDART_LIB} CUDA_INCLUDE_DIR=${CUDA_INCLUDE_DIR}")
endif()

add_executable(split_native_hailo_trt_fifo main.cpp)
target_include_directories(split_native_hailo_trt_fifo PRIVATE ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(split_native_hailo_trt_fifo PRIVATE HailoRT::libhailort Threads::Threads ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} ${CUDART_LIB} ${OpenCV_LIBS})
target_compile_options(split_native_hailo_trt_fifo PRIVATE -O3 -Wno-deprecated-declarations)
'''

def _load_json(p: Path) -> Any:
    with p.open('r', encoding='utf-8') as f:
        return json.load(f, object_pairs_hook=_no_duplicate_json_keys)


def _no_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def _load_native_split_quality_binding(
    path: str, *, args: argparse.Namespace, case: str, task: str,
) -> dict[str, Any] | None:
    """Locally re-hash and consume the exact Quality-FIRST Hailo-8 binding."""
    if not str(path or "").strip():
        return None
    binding_path = Path(path).expanduser().resolve()
    raw = _load_json(binding_path)
    expected = {
        "backend": "hailo8_to_trt",
        "model": str(args.model_id or ""),
        "case": case,
        "setup_id": str(args.setup_id or ""),
        "task": task,
        "precision": str(args.precision or ""),
    }
    binding, status = validate_native_split_quality_binding(
        raw, expected_identity=expected, verification_mode="local",
    )
    if binding is None:
        raise RuntimeError(f"native_split_quality_local_validation_failed:{status}")
    if str(binding.get("eval_run_id") or "") != str(args.eval_run_id or ""):
        raise RuntimeError("native_split_quality_eval_run_id_mismatch")
    if canonical_native_split_backend(
        binding.get("source_run_id"), args.setup_id,
    ) != canonical_native_split_backend(args.source_run_id, args.setup_id):
        raise RuntimeError("native_split_quality_source_run_id_mismatch")
    _native_split_selection_evidence(binding)
    return binding


def _native_split_selection_evidence(
    binding: Mapping[str, Any],
) -> dict[str, str]:
    """Return Central selection evidence or the explicit cache-only replay.

    Cache verification is diagnostic and deliberately has no new Central
    Quality result.  Its sealed replay is therefore accepted only while the
    process-wide artifact fence is active and only when it proves that no
    compiler was dispatched.  Normal runs retain the mandatory Central receipt.
    """
    try:
        return native_split_quality_selection_duplicates(binding)
    except ValueError as exc:
        replay = binding.get("cache_verify_replay")
        policy = str(
            os.environ.get("ONNX_SPLITPOINT_ARTIFACT_POLICY") or ""
        ).strip().lower()
        source_sha = str(
            replay.get("source_binding_sha256") if isinstance(replay, Mapping)
            else ""
        ).strip().lower()
        if not (
            policy == "cache_verify_only"
            and isinstance(replay, Mapping)
            and replay.get("artifact_policy") == "cache_verify_only"
            and replay.get("compiler_dispatched") is False
            and str(replay.get("local_validation_status") or "").startswith(
                "local_files_rehashed_"
            )
        ):
            raise RuntimeError(
                f"native_split_quality_central_selection_invalid:{exc}"
            ) from exc
        replay_sha = canonical_json_sha256(replay)
        return {
            "native_split_quality_cache_verify_source_binding_sha256": (
                source_sha
            ),
            "native_split_quality_cache_verify_replay_sha256": replay_sha,
        }


def _binding_artifact_path(binding: Mapping[str, Any], name: str) -> Path:
    row = (binding.get("artifacts") or {}).get(name)
    if not isinstance(row, Mapping):
        raise RuntimeError(f"native_split_quality_binding_artifact_missing:{name}")
    return Path(str(row.get("path") or "")).resolve()


def _seal_split_consumer_attestation(
    *, binding: Mapping[str, Any], command: Mapping[str, Any], args: argparse.Namespace,
    case: str, task: str,
) -> dict[str, Any]:
    local_proof = dict(binding.get("local_artifact_verification") or {})
    artifacts = command.get("artifacts") if isinstance(command.get("artifacts"), Mapping) else {}
    payload: dict[str, Any] = {
        "schema": "onnx-splitpoint/native-split-quality-consumer-attestation",
        "schema_version": 1,
        "status": (
            "diagnostic_metadata_only"
            if _cache_verify_only()
            else "local_files_rehashed_and_exact_command_join_verified"
        ),
        "binding_sha256": str(binding.get("binding_sha256") or ""),
        "command_contract_sha256": str(command.get("contract_sha256") or ""),
        "eval_run_id": str(binding.get("eval_run_id") or ""),
        "source_run_id": canonical_native_split_backend(
            binding.get("source_run_id"), args.setup_id,
        ),
        "backend": "hailo8_to_trt",
        "model_id": str(args.model_id or ""),
        "case_id": str(case),
        "setup_id": str(args.setup_id or ""),
        "task": str(task),
        "precision": str(args.precision),
        "local_artifact_verification_sha256": canonical_json_sha256(local_proof),
        "semantic_output_manifest_sha256": str((artifacts.get("semantic_output_manifest") or {}).get("sha256") or ""),
        "semantic_boundary_manifest_sha256": str((artifacts.get("semantic_boundary_manifest") or {}).get("sha256") or ""),
        **_native_split_selection_evidence(binding),
    }
    payload["attestation_sha256"] = canonical_json_sha256(payload)
    return payload


def _bind_quality_for_execution(
    *, native_row: Mapping[str, Any], quality_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    """Bind strictly for claimable runs, semantically for cache diagnostics.

    The cache Canary has already re-hashed the complete artifact set locally
    and executes from a unique run directory.  An exact command/receipt join is
    still recorded, but metadata drift in that redundant proof must not turn a
    successful diagnostic execution red.  Identity and current-run fields stay
    hard requirements.
    """

    joined, status = bind_quality_to_native_split(
        native_row=native_row,
        quality_binding=quality_binding,
    )
    if joined is not None:
        return dict(joined), str(status), ""
    if not _cache_verify_only():
        raise RuntimeError(
            "native_split_quality_consumer_join_failed:" + str(status)
        )
    expected = {
        "backend": "hailo8_to_trt",
        "model": str(native_row.get("model_id") or ""),
        "case": str(native_row.get("case_id") or ""),
        "setup_id": str(native_row.get("setup_id") or ""),
        "task": str(native_row.get("task") or ""),
        "precision": str(native_row.get("precision") or ""),
    }
    verified, semantic_status = validate_native_split_quality_binding(
        quality_binding,
        expected_identity=expected,
        verification_mode="portable",
    )
    if verified is None:
        raise RuntimeError(
            "native_split_quality_semantic_consumer_join_failed:"
            + str(semantic_status)
        )
    expected_eval = str(verified.get("eval_run_id") or "")
    actual_eval = str(native_row.get("eval_run_id") or "")
    expected_source = canonical_native_split_backend(
        verified.get("source_run_id"), expected["setup_id"],
    )
    actual_source = canonical_native_split_backend(
        native_row.get("source_run_id"), expected["setup_id"],
    )
    if not expected_eval or actual_eval != expected_eval:
        raise RuntimeError("native_split_quality_eval_run_id_mismatch")
    if not expected_source or actual_source != expected_source:
        raise RuntimeError("native_split_quality_source_run_id_mismatch")
    return (
        dict(verified),
        "cache_verify_semantic_binding_consumed",
        "cache_verify_diagnostic_exact_consumer_join_failed:" + str(status),
    )


def _find_hef(bs: Path, case: str, hw_arch: str) -> Path:
    cands = [
        bs / case / 'hailo' / hw_arch / 'part1' / 'compiled.hef',
        bs / case / 'hailo' / hw_arch / 'part1' / f'{case}_part1.hef',
        bs / case / 'hailo' / hw_arch / 'part1' / f'{hw_arch}.hef',
    ]
    cands.extend(sorted((bs / case).glob(f'hailo/{hw_arch}/part1/**/*.hef')))
    cands.extend(sorted((bs / case).glob(f'**/{hw_arch}/part1/**/*.hef')))
    cands.extend(sorted((bs / case).glob('**/part1/**/*.hef')))
    for p in cands:
        if '.hailo-generations' in p.parts:
            continue  # A backup generation is not the active HEF.
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f'No Hailo Part1 HEF found for case={case} hw_arch={hw_arch} under {bs/case}')


def _find_engine(bs: Path, case: str, precision: str) -> Path:
    name = 'part2_uint8_cast_fp16.engine' if precision == 'uint8_cast_fp16' else ('part2_uint8_dequant_fp16.engine' if precision == 'uint8_dequant_fp16' else f'part2_{precision}.engine')
    cands = [
        bs / 'native_trt' / case / 'part2' / precision / name,
        bs / case / 'native_trt' / 'part2' / precision / name,
    ]
    cands.extend(sorted(bs.glob(f'native_trt/{case}/part2/{precision}/*.engine')))
    for p in cands:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f'No native TensorRT Part2 engine found for case={case} precision={precision}. Build it first with scripts/native_trt_from_benchmarkset.py')


def _engine_boundary_contract(engine: Path) -> dict[str, Any]:
    """Read the build-time boundary bridge that the existing engine embeds."""
    meta_path = engine.parent / "native_trt_meta.json"
    try:
        meta = _load_json(meta_path) if meta_path.is_file() else {}
    except Exception:
        meta = {}
    bridge = meta.get("uint8_cast_bridge") if isinstance(meta, dict) else {}
    bridge = bridge if isinstance(bridge, dict) else {}
    layout = bridge.get("boundary_layout")
    layout = layout if isinstance(layout, dict) else {}
    requested = str(layout.get("requested") or "").strip()
    effective = str(layout.get("effective") or requested or "as_input").strip()
    return {
        "metadata_path": str(meta_path.resolve()) if meta_path.is_file() else "",
        "metadata_sha256": _sha256_file(meta_path),
        "boundary_layout_requested": requested or effective,
        "boundary_layout_effective": effective,
        "dequant_scale": bridge.get("scale"),
        "dequant_zero_point": bridge.get("zero_point"),
        "bridge_schema": str(bridge.get("schema") or ""),
    }


def _native_command_contract(
    *,
    bs: Path,
    case: str,
    args: argparse.Namespace,
    hef: Path,
    engine: Path,
    image: Path,
    work: Path,
    executable: Path,
    prepared_input: Path | None = None,
    prepared_input_shape: list[int] | None = None,
    prepared_input_name: str = "",
    prepared_input_dtype: str = "uint8",
    quality_binding: Mapping[str, Any] | None = None,
    semantic_payload: Mapping[str, Any] | None = None,
    completion_execution_contract: Mapping[str, Any] | None = None,
    producer_impl: str = "hailo8_cpp_vstreams_fifo",
) -> dict[str, Any]:
    """Archive every runtime-relevant input needed for a fail-closed replay."""
    boundary = _engine_boundary_contract(engine)
    contract: dict[str, Any] = {
        "schema": _REPLAY_CONTRACT_SCHEMA,
        "schema_version": 1,
        "backend": "hailo8_to_trt",
        "model": str(args.model_id or bs.parent.name),
        "setup_id": str(args.setup_id or "orin_nx_hailo8_01"),
        "comparison_backend": "hailo8",
        "runner": "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "python_executable": str(sys.executable),
        "interpreter_identity": {
            "executable": str(sys.executable),
            "resolved_executable": str(Path(sys.executable).resolve()),
            "executable_sha256": _sha256_file(sys.executable),
            "prefix": str(sys.prefix),
            "base_prefix": str(getattr(sys, "base_prefix", "")),
            "version": str(sys.version),
            "runtime_mode": (
                "system_tensorrt_with_process_local_hailo_sites"
                if producer_impl == "hailo8_python_vstreams_fifo"
                else "native_cpp_launcher"
            ),
            "process_local_extra_sites": list(
                _PROCESS_LOCAL_EXTRA_SITES
            ),
        },
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "benchmark_set": str(bs),
        "case": str(case),
        "hw_arch": str(args.hw_arch),
        "precision": str(args.precision),
        "input_image": str(image),
        "input_image_source": "exact_file",
        "input_image_sha256": _sha256_file(image),
        "hef": str(hef),
        "hef_sha256": _sha256_file(hef),
        "engine": str(engine),
        "engine_sha256": _sha256_file(engine),
        "native_executable": str(executable),
        "native_executable_sha256": _sha256_file(executable),
        "artifacts": {
            "python_executable": {"path": str(sys.executable), "sha256": _sha256_file(sys.executable)},
            "hef": {"path": str(hef), "sha256": _sha256_file(hef)},
            "engine": {"path": str(engine), "sha256": _sha256_file(engine)},
            "native_executable": {"path": str(executable), "sha256": _sha256_file(executable)},
            "generated_cpp": {"path": str(work / "main.cpp"), "sha256": _sha256_file(work / "main.cpp")},
            "cmake": {"path": str(work / "CMakeLists.txt"), "sha256": _sha256_file(work / "CMakeLists.txt")},
        },
        "prepared_input_contract": {
            "format": "raw_runtime_tensor",
            "name": str(prepared_input_name or ""),
            "shape": [int(x) for x in (prepared_input_shape or [])],
            "dtype": str(prepared_input_dtype or ""),
            "layout": (
                "NHWC"
                if len(prepared_input_shape or []) == 4
                and int((prepared_input_shape or [0])[-1]) in {1, 3, 4}
                else "NCHW"
                if len(prepared_input_shape or []) == 4
                else "HWC"
                if len(prepared_input_shape or []) == 3
                and int((prepared_input_shape or [0])[-1]) in {1, 3, 4}
                else "CHW"
            ),
            "task": str(args.task),
            "preprocess": (
                f"pillow_{args.preprocess_mode_effective}_rgb_"
                f"{prepared_input_dtype}"
                if producer_impl == "hailo8_python_vstreams_fifo"
                and not str(getattr(args, "prepared_input_rgb", "") or "")
                else f"opencv_{args.preprocess_mode_effective}_rgb_uint8"
            ),
            "shared_rgb_source": str(getattr(args, "prepared_input_rgb", "") or ""),
            "shared_rgb_source_sha256": str(getattr(args, "expected_prepared_input_sha256", "") or ""),
            "preprocess_mode_requested": str(args.preprocess_mode),
            "preprocess_mode_effective": str(args.preprocess_mode_effective),
            "letterbox_pad_value_requested": int(args.letterbox_pad_value),
            "letterbox_pad_value_effective": (
                int(args.letterbox_pad_value)
                if str(args.preprocess_mode_effective) == "letterbox" else 0
            ),
            "letterbox_pad_value": (
                int(args.letterbox_pad_value)
                if str(args.preprocess_mode_effective) == "letterbox" else 0
            ),
            "pad_value_effective": (
                int(args.letterbox_pad_value)
                if str(args.preprocess_mode_effective) == "letterbox" else 0
            ),
            "source_image": str(image),
            "source_image_sha256": _sha256_file(image),
        } if prepared_input is not None and prepared_input.is_file() else {},
        "measurement_endpoint": (
            "completed_task"
            if producer_impl == "hailo8_python_vstreams_fifo"
            else "raw_model_outputs"
            if str(args.task) == "detection"
            else "model_outputs"
        ),
        "runtime_options": {
            "measurement_endpoint": (
                "completed_task"
                if producer_impl == "hailo8_python_vstreams_fifo"
                else "raw_model_outputs"
                if str(args.task) == "detection"
                else "model_outputs"
            ),
            "frames": int(args.frames),
            "duration_s": float(args.duration_s or 0.0),
            "warmup": int(args.warmup),
            "repetitions": int(args.repetitions),
            "queue_depth": int(args.queue_depth),
            "hailo_format": str(args.hailo_format),
            "task": str(args.task),
            "preprocess_mode_requested": str(args.preprocess_mode),
            "preprocess_mode_effective": str(args.preprocess_mode_effective),
            "letterbox_pad_value_requested": int(args.letterbox_pad_value),
            "letterbox_pad_value_effective": (
                int(args.letterbox_pad_value)
                if str(args.preprocess_mode_effective) == "letterbox" else 0
            ),
            "letterbox_pad_value": (
                int(args.letterbox_pad_value)
                if str(args.preprocess_mode_effective) == "letterbox" else 0
            ),
            "copy_outputs": bool(args.copy_outputs),
            "dump_outputs": bool(args.dump_outputs),
            "dump_boundary": bool(args.dump_boundary),
            "device_id": str(args.device_id or ""),
            "build": bool(args.build),
            "energy_prepared_feed_capable": True,
            "prepared_feed_policy": "exact_prepared_feed_reused_for_every_work_unit",
            "producer_impl": str(producer_impl),
            "completion_runtime_mode": str(getattr(args, "completion_runtime_mode", "legacy_attested_hotloop")),
            "completion_execution_contract": (
                dict(completion_execution_contract)
                if completion_execution_contract is not None else None
            ),
            "completion_execution_contract_sha256": str(
                (completion_execution_contract or {}).get(
                    "contract_sha256"
                ) or ""
            ),
        },
        "boundary_contract": boundary,
    }
    if producer_impl == "hailo8_python_vstreams_fifo":
        mixed_runtime = getattr(args, "mixed_runtime_probe", None)
        mixed_runtime = (
            dict(mixed_runtime)
            if isinstance(mixed_runtime, Mapping) else {}
        )
        consumer_source = Path(
            str(mixed_runtime.get("native_trt_consumer_source") or "")
        )
        contract["mixed_runtime_contract"] = mixed_runtime
        contract["runtime_options"][
            "process_local_extra_sites"
        ] = list(_PROCESS_LOCAL_EXTRA_SITES)
        contract["runtime_options"][
            "mixed_runtime_site_policy"
        ] = "site.addsitedir_after_system_defaults"
        contract["artifacts"]["native_trt_consumer_source"] = {
            "path": str(consumer_source),
            "sha256": (
                _sha256_file(consumer_source)
                if consumer_source.is_file() else ""
            ),
        }
    if quality_binding is not None:
        binding = dict(quality_binding)
        selection = dict(binding.get("preselection") or {})
        quality_source_run_id = canonical_native_split_backend(
            binding.get("source_run_id"), args.setup_id,
        )
        contract.update({
            "eval_run_id": str(binding.get("eval_run_id") or ""),
            "source_run_id": quality_source_run_id,
            "native_split_quality_binding": binding,
            "native_split_quality_binding_sha256": str(binding.get("binding_sha256") or ""),
            "native_split_quality_eval_run_id": str(binding.get("eval_run_id") or ""),
            "native_split_quality_source_run_id": quality_source_run_id,
            **_native_split_selection_evidence(binding),
            "native_split_quality_local_verification": dict(binding.get("local_artifact_verification") or {}),
            "quality_preselection": selection,
            "quality_preselection_sha256": str(selection.get("selection_sha256") or ""),
            "quality_boundary_contract": dict(binding.get("boundary_contract") or {}),
            "quality_boundary_contract_sha256": str(binding.get("boundary_contract_sha256") or ""),
        })
        role_map = {
            "part1_runtime": "hef",
            "boundary_metadata": "boundary_metadata",
            "source_part2_onnx": "source_part2_onnx",
            "build_part2_onnx": "build_part2_onnx",
            "engine": "engine",
            "native_trt_meta": "native_trt_meta",
            "engine_build_receipt": "engine_build_receipt",
            "trtexec": "trtexec",
        }
        binding_artifacts = binding.get("artifacts") or {}
        for source_name, contract_name in role_map.items():
            row = binding_artifacts.get(source_name)
            if isinstance(row, Mapping):
                contract["artifacts"][contract_name] = dict(row)
    if semantic_payload is not None:
        strict_boundary = dict(
            semantic_payload.get("strict_quality_boundary") or {}
        )
        contract["runtime_boundary_evidence"] = {
            "status": "exact_runtime_boundary_verified"
            if semantic_payload.get("native_split_quality_runtime_boundary_verified") is True
            else "unverified",
            "output_count": semantic_payload.get("hailo_runtime_output_count"),
            "output_name": semantic_payload.get("hailo_runtime_output_name"),
            "output_frame_bytes": semantic_payload.get("hailo_runtime_output_frame_bytes"),
            "output_shape": [
                int(value)
                for value in list(strict_boundary.get("shape") or [])
            ],
            "output_dtype": str(strict_boundary.get("dtype") or ""),
            "trt_input_name": str(
                strict_boundary.get("trt_input_name") or ""
            ),
            "trt_input_shape": [
                int(value)
                for value in list(
                    strict_boundary.get("trt_input_shape") or []
                )
            ],
        }
        for payload_name, artifact_name in (
            ("native_fifo_output_manifest", "semantic_output_manifest"),
            ("native_fifo_boundary_manifest", "semantic_boundary_manifest"),
        ):
            manifest = Path(str(semantic_payload.get(payload_name) or ""))
            if manifest.is_file():
                contract["artifacts"][artifact_name] = {
                    "path": str(manifest.resolve()),
                    "sha256": _sha256_file(manifest),
                    "size_bytes": int(manifest.stat().st_size),
                }
    if prepared_input is not None and prepared_input.is_file():
        contract["artifacts"]["prepared_input"] = {
            "path": str(prepared_input), "sha256": _sha256_file(prepared_input),
        }
        contract["runtime_options"]["prepared_input_bound"] = True
    contract["complete"] = bool(
        contract["runner_sha256"]
        and contract["input_image_sha256"]
        and contract["hef_sha256"]
        and contract["engine_sha256"]
        and contract["native_executable_sha256"]
        and contract["artifacts"]["generated_cpp"]["sha256"]
        and contract["artifacts"]["cmake"]["sha256"]
        and bool(contract.get("prepared_input_contract"))
        and bool((contract.get("artifacts") or {}).get("prepared_input", {}).get("sha256"))
        and hailo8_preprocess_binding(
            contract["runtime_options"], contract["prepared_input_contract"],
        )
        and boundary.get("boundary_layout_effective")
        and (
            str(args.task) != "detection"
            or producer_impl != "hailo8_python_vstreams_fifo"
            or (
                completion_execution_contract is not None
                and len(str(
                    completion_execution_contract.get(
                        "contract_sha256"
                    ) or ""
                )) == 64
                and contract.get("mixed_runtime_contract", {}).get(
                    "status"
                ) == "ready"
                and contract.get("mixed_runtime_contract", {}).get(
                    "source_closure_ok"
                ) is True
                and len(str(
                    contract.get("artifacts", {}).get(
                        "native_trt_consumer_source", {}
                    ).get("sha256") or ""
                )) == 64
            )
        )
    )
    contract["contract_sha256"] = _stable_json_sha256(contract)
    return contract


def _verify_replay_expectations(
    args: argparse.Namespace,
    *,
    hef: Path,
    engine: Path,
    image: Path,
    executable: Path,
    boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail before inference when a staged replay no longer matches evidence."""
    checks = {
        "runner_sha256": (_sha256_file(Path(__file__).resolve()), str(args.expected_runner_sha256 or "")),
        "input_image_sha256": (_sha256_file(image), str(args.expected_image_sha256 or "")),
        "hef_sha256": (_sha256_file(hef), str(args.expected_hef_sha256 or "")),
        "engine_sha256": (_sha256_file(engine), str(args.expected_engine_sha256 or "")),
        "native_executable_sha256": (
            _sha256_file(executable), str(args.expected_executable_sha256 or "")
        ),
        "boundary_layout": (
            str(boundary.get("boundary_layout_effective") or ""),
            str(args.expected_boundary_layout or ""),
        ),
    }
    mismatches = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in checks.items()
        if expected and actual.lower() != expected.lower()
    }
    result = {
        "requested": any(expected for _actual, expected in checks.values()),
        "ok": not mismatches,
        "checks": {
            name: {"actual": actual, "expected": expected, "ok": (not expected or actual.lower() == expected.lower())}
            for name, (actual, expected) in checks.items()
        },
        "mismatches": mismatches,
        "source_contract_sha256": str(args.source_contract_sha256 or ""),
    }
    if mismatches:
        raise RuntimeError("native replay contract mismatch: " + json.dumps(mismatches, sort_keys=True))
    return result


def _energy_workload_only(args: argparse.Namespace) -> int:
    """Run the already-built C++ FIFO executable from a fresh preflight seal.

    This branch deliberately executes before normal discovery, generated-source
    writes, builds and SHA-256 replay checks.  Only duration/output paths may be
    supplied by the measurement plan; every runtime-relevant choice comes from
    the successful command contract embedded in the attestation.
    """
    binding, reason = load_split_energy_workload_binding(
        args.energy_preflight_attestation,
        expected_nonce=str(args.energy_preflight_nonce or ""),
        expected_command_contract_sha256=str(args.source_contract_sha256 or ""),
        expected_backend="hailo8_to_trt",
        max_age_s=float(args.energy_preflight_max_age_s),
    )
    if binding is None:
        print(f"split_energy_attestation_rejected:{reason}", file=sys.stderr)
        return 6
    options = dict(binding.get("runtime_options") or {})
    artifacts = dict(binding.get("artifacts") or {})
    prepared_contract = dict(binding.get("prepared_input_contract") or {})
    errors: list[str] = []
    if int(args.warmup) != 0 or int(options.get("warmup") if options.get("warmup") is not None else -1) != 0:
        errors.append("split_energy_warmup_must_be_zero")
    if bool(args.build):
        errors.append("split_energy_build_must_be_disabled")
    if bool(args.dump_outputs) or bool(args.dump_boundary):
        errors.append("split_energy_dumps_must_be_disabled")
    if bool(options.get("dump_outputs")) or bool(options.get("dump_boundary")):
        errors.append("split_energy_attested_dumps_not_disabled")
    if float(args.duration_s or 0.0) <= 0.0:
        errors.append("split_energy_duration_must_be_positive")
    task = str(options.get("task") or "")
    runtime_boundary = dict(
        binding.get("runtime_boundary_evidence") or {}
    )
    requested_mode = str(options.get("preprocess_mode_requested") or "")
    effective_mode = str(options.get("preprocess_mode_effective") or "")
    if task not in {"classification", "detection"}:
        errors.append("split_energy_attested_task_missing")
    if task == "detection":
        completion_contract = options.get(
            "completion_execution_contract"
        )
        try:
            verified_completion_contract = (
                verify_detection_completion_execution_contract(
                    completion_contract
                )
            )
        except Exception:
            verified_completion_contract = None
            errors.append(
                "split_energy_detection_completion_contract_invalid"
            )
        if (
            str(options.get("producer_impl") or "")
            != "hailo8_python_vstreams_fifo"
        ):
            errors.append(
                "split_energy_detection_python_vstreams_producer_missing"
            )
        if (
            runtime_boundary.get("status")
            != "exact_runtime_boundary_verified"
            or int(runtime_boundary.get("output_count") or 0) != 1
            or not str(runtime_boundary.get("output_name") or "")
            or not list(runtime_boundary.get("output_shape") or [])
            or not str(runtime_boundary.get("output_dtype") or "")
        ):
            errors.append(
                "split_energy_exact_runtime_boundary_attestation_missing"
            )
    else:
        verified_completion_contract = None
    if requested_mode not in {"auto", "resize", "letterbox"}:
        errors.append("split_energy_attested_preprocess_request_missing")
    if effective_mode not in {"resize", "letterbox"}:
        errors.append("split_energy_attested_preprocess_effective_missing")
    if requested_mode == "auto" and effective_mode != (
        "letterbox" if task == "detection" else "resize"
    ):
        errors.append("split_energy_attested_auto_preprocess_resolution_mismatch")
    if not hailo8_preprocess_binding(options, prepared_contract):
        errors.append("split_energy_prepared_preprocess_contract_mismatch")
    identity_checks = {
        "benchmark_set": (str(args.benchmark_set), str(binding.get("benchmark_set") or "")),
        "case": (str(args.case), str(binding.get("case") or "")),
        "precision": (str(args.precision), str(binding.get("precision") or "")),
        "image": (str(Path(args.image).expanduser().resolve()), str(Path(str(binding.get("input_image") or "")).expanduser().resolve())),
        "hw_arch": (str(args.hw_arch), str(binding.get("hw_arch") or "hailo8")),
        "queue_depth": (str(int(args.queue_depth)), str(int(options.get("queue_depth") or 0))),
        "hailo_format": (str(args.hailo_format), str(options.get("hailo_format") or "")),
        "task": (str(args.task), task),
        "preprocess_mode": (str(args.preprocess_mode), requested_mode),
        "copy_outputs": (str(bool(args.copy_outputs)), str(bool(options.get("copy_outputs")))),
        "letterbox_pad_value": (
            str(int(args.letterbox_pad_value)),
            str(int(options.get("letterbox_pad_value_requested")))
            if options.get("letterbox_pad_value_requested") is not None else "-1",
        ),
    }
    for label, (actual, expected) in identity_checks.items():
        if actual != expected:
            errors.append(f"split_energy_cli_{label}_mismatch")
    if errors:
        print(";".join(errors), file=sys.stderr)
        return 7

    def artifact_path(name: str) -> str:
        row = artifacts.get(name)
        return str(row.get("path") or "") if isinstance(row, Mapping) else ""

    executable = artifact_path("native_executable")
    hef = artifact_path("hef")
    engine = artifact_path("engine")
    prepared_input = artifact_path("prepared_input")
    image = str(binding.get("input_image") or "")
    if not executable or not hef or not engine or not image or not prepared_input:
        print("split_energy_attested_execution_path_missing", file=sys.stderr)
        return 8
    out_json = Path(args.result_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if task == "detection":
        try:
            shape = tuple(
                int(value)
                for value in list(prepared_contract.get("shape") or [])
            )
            dtype = np.dtype(
                str(prepared_contract.get("dtype") or "")
            )
            if not shape or any(value <= 0 for value in shape):
                raise RuntimeError("prepared input shape invalid")
            raw = Path(prepared_input).read_bytes()
            expected_bytes = int(np.prod(shape)) * int(dtype.itemsize)
            if len(raw) != expected_bytes:
                raise RuntimeError(
                    "prepared input byte count does not match contract"
                )
            prepared_array = np.frombuffer(
                raw, dtype=dtype
            ).reshape(shape).copy()
            backend, prepared, trt = _open_hailo8_python_runtime(
                hef=Path(hef),
                engine=Path(engine),
                work=out_json.parent,
                args=args,
                runtime_label="energy_workload",
            )
            try:
                actual_names = list(prepared.input_names or [])
                expected_name = str(
                    prepared_contract.get("name") or ""
                )
                if len(actual_names) != 1:
                    raise RuntimeError(
                        "energy workload requires one Hailo input"
                    )
                if expected_name and actual_names[0] != expected_name:
                    raise RuntimeError(
                        "prepared input name does not match runtime"
                    )
                input_name = expected_name or str(actual_names[0])
                completion_mode = str(options.get("completion_runtime_mode") or "strict")
                if completion_mode not in ("strict", "legacy_attested_hotloop", "fast_oracle_outside_timing"):
                    raise RuntimeError("split_energy_completion_runtime_mode_unsupported")
                completion_runtime = (
                    FastDetectionCompletionRuntime(verified_completion_contract)
                    if completion_mode == "fast_oracle_outside_timing"
                    else DetectionCompletionRuntime(verified_completion_contract)
                )
                payload = _hailo8_python_fifo_run(
                    backend,
                    prepared,
                    {input_name: prepared_array},
                    trt,
                    frames=max(1, int(args.frames)),
                    warmup=0,
                    queue_depth=int(
                        options.get("queue_depth") or 1
                    ),
                    duration_s=float(args.duration_s),
                    completion_runtime=completion_runtime,
                    warmup_completion_runtime=None,
                    expected_boundary_name=str(
                        runtime_boundary.get("output_name") or ""
                    ),
                    expected_boundary_shape=[
                        int(value)
                        for value in list(
                            runtime_boundary.get("output_shape") or []
                        )
                    ],
                    expected_boundary_dtype=str(
                        runtime_boundary.get("output_dtype") or ""
                    ),
                )
            finally:
                _close_hailo8_python_runtime(
                    backend, prepared, trt,
                )
            count = int(payload.get("completed_work_units") or 0)
            if count <= 0:
                raise RuntimeError(
                    "exact completed detection count missing"
                )
            payload.update({
                "completed_frames": count,
                "warmup": 0,
                "energy_workload_only": True,
                "energy_preflight_status": reason,
                "source_contract_sha256": str(
                    args.source_contract_sha256 or ""
                ),
                "prepared_feed_contract": (
                    "exact_prepared_feed_loaded_outside_counted_loop"
                ),
                "prepared_input_source": (
                    "preflight_verified_contract_artifact"
                ),
            })
            payload.update({
                "measurement_endpoint": "completed_task",
                "energy_preflight_nonce": str(args.energy_preflight_nonce),
                "energy_completion_observation_scope": "fresh_energy_invocation",
                "energy_completion_window_source": "collector_command_marker_window",
                **{key: binding.get(key) for key in (
                    "model", "case", "setup_id", "comparison_backend", "precision",
                ) if binding.get(key) not in (None, "")},
            })
            out_json.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            # One bounded publication after the measured FIFO and oracle have
            # finished. The collector captures this invocation's stdout inside
            # its own command-marker window; no evidence work enters the loop.
            if completion_mode == "fast_oracle_outside_timing":
                print("__SPLITPOINT_ENERGY_COMPLETION__=" + json.dumps(payload, separators=(",", ":")))
        except Exception as exc:
            print(
                "split_energy_completed_detection_failed:"
                f"{type(exc).__name__}:{exc}",
                file=sys.stderr,
            )
            return 9
        print(f"__SPLITPOINT_WORK_UNITS__={count}")
        print(
            "__SPLITPOINT_WORK_UNITS_SOURCE__="
            "completed_work_units"
        )
        print("__SPLITPOINT_WORK_UNITS_EXACT__=1")
        return 0
    cmd = [
        executable,
        "--hef", hef,
        "--engine", engine,
        "--image", image,
        "--out", str(out_json),
        "--frames", str(max(1, int(args.frames))),
        "--duration-s", str(float(args.duration_s)),
        "--warmup", "0",
        "--queue-depth", str(int(options.get("queue_depth") or 1)),
        "--hailo-format", str(options.get("hailo_format") or "uint8"),
        "--task", task,
        "--preprocess-mode", effective_mode,
        "--copy-outputs", "1" if options.get("copy_outputs") else "0",
        "--dump-outputs", "0",
        "--dump-boundary", "0",
        "--letterbox-pad-value", str(int(options.get("letterbox_pad_value_requested"))),
        "--output-dir", str(out_json.parent / "unused_outputs"),
        "--boundary-dir", str(out_json.parent / "unused_boundary"),
        "--reuse-preprocessed-input", "1",
        "--prepared-input-rgb", prepared_input,
    ]
    device_id = str(options.get("device_id") or "").strip()
    if device_id:
        cmd += ["--device-id", device_id[4:] if device_id.startswith("pci/") else device_id]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print(f"split_energy_native_executable_rc={completed.returncode}", file=sys.stderr)
        return int(completed.returncode)
    try:
        payload = _load_json(out_json)
    except Exception as exc:
        print(f"split_energy_result_unreadable:{type(exc).__name__}", file=sys.stderr)
        return 9
    count = int(payload.get("completed_frames") or 0) if isinstance(payload, Mapping) else 0
    if count <= 0:
        print("split_energy_exact_completed_frames_missing", file=sys.stderr)
        return 10
    if isinstance(payload, dict):
        payload.update({
            "completed_frames": count,
            "warmup": 0,
            "energy_workload_only": True,
            "energy_preflight_status": reason,
            "source_contract_sha256": str(args.source_contract_sha256 or ""),
            "prepared_feed_contract": "exact_prepared_feed_loaded_outside_counted_loop",
            "prepared_input_source": "preflight_verified_contract_artifact",
        })
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"__SPLITPOINT_WORK_UNITS__={count}")
    print("__SPLITPOINT_WORK_UNITS_SOURCE__=completed_frames")
    print("__SPLITPOINT_WORK_UNITS_EXACT__=1")
    return 0



_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}

def _image_from_manifest(manifest: Path) -> Path | None:
    try:
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        samples = payload.get('samples') if isinstance(payload, dict) else None
        if not isinstance(samples, list):
            return None
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            rel = str(sample.get('image') or sample.get('file') or '').strip()
            if not rel:
                continue
            candidate = (manifest.parent / rel).resolve()
            if candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES:
                return candidate
    except Exception:
        return None
    return None

def _first_validation_image(root: Path) -> Path | None:
    if not root.exists():
        return None
    # Prefer the run-mode subset manifest because it records the exact sample
    # selected for the evaluation contract, including nested ImageNet classes.
    for manifest in sorted(root.glob('**/manifest.json')):
        image = _image_from_manifest(manifest)
        if image is not None:
            return image
    images = sorted(
        p.resolve() for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    return images[0] if images else None


def _default_image(bs: Path) -> Path:
    """Return one exact validation image from the materialised run-mode subset.

    ImageNet subsets are class-stratified and therefore nested below
    ``images/<wnid>/``.  Passing their parent directory to the C++ runner (which
    intentionally scans only direct children) resulted in ``no images found``.
    Resolving the first manifest sample also makes Native and Full-ONNX
    self-reference provenance deterministic.
    """
    resources = bs / 'resources'
    exact = resources / 'test_image_coco.png'
    if exact.is_file():
        return exact.resolve()
    selected = _first_validation_image(resources / 'validation')
    if selected is not None:
        return selected
    selected = _first_validation_image(bs)
    if selected is not None:
        return selected
    raise FileNotFoundError('No default validation image found. Pass --image explicitly.')



def _resolve_image_for_seq(image_path: Path, seq: int | None) -> str:
    """Mirror the C++ native runner image ordering for provenance.

    If a directory was passed, C++ uses directory_iterator and then lexicographic sort.
    If a file was passed, that exact file is used for every frame.
    """
    try:
        p = Path(image_path).expanduser().resolve()
        if p.is_file():
            return str(p)
        if p.is_dir():
            imgs=[]
            for child in p.iterdir():
                if child.is_file() and child.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                    imgs.append(str(child.resolve()))
            imgs=sorted(imgs)
            if imgs:
                idx = int(seq or 0) % len(imgs)
                return imgs[idx]
    except Exception:
        pass
    return ''



def _benchmark_task(bs: Path) -> str:
    try:
        payload = _load_json(bs / 'benchmark_set.json')
        return str(payload.get('benchmark_task') or payload.get('task') or payload.get('model_task') or '').strip().lower()
    except Exception:
        return ''


def _benchmark_model_id(bs: Path) -> str:
    payload = _load_json(bs / 'benchmark_set.json')
    if isinstance(payload, Mapping):
        return str(payload.get('model_id') or payload.get('model_name') or '').strip()
    return ''


def _resolve_preprocess_mode(task: str, requested: str) -> str:
    task_value = str(task or '').strip().lower()
    mode = str(requested or 'auto').strip().lower()
    if task_value not in {'classification', 'detection'}:
        raise RuntimeError('benchmark task must be classification or detection')
    if mode == 'auto':
        return 'letterbox' if task_value == 'detection' else 'resize'
    if mode not in {'resize', 'letterbox'}:
        raise RuntimeError('preprocess mode must be auto, resize, or letterbox')
    return mode

def _annotate_output_contract(manifest_path: str | Path, bs: Path) -> None:
    try:
        mp = Path(manifest_path).expanduser()
        if not mp.is_file():
            return
        payload = json.loads(mp.read_text(encoding='utf-8'))
        task = _benchmark_task(bs)
        runtime_outputs = load_manifest_outputs(mp, payload)
        declaration = load_authoritative_output_contract(
            bs, backend='tensorrt', model_id=_benchmark_model_id(bs),
            variant='full', task=task,
        )
        meta = runtime_output_contract(
            task, runtime_outputs, raw_fallback=False,
            declared_contract=declaration,
        )
        payload['authoritative_output_contract_resolution'] = declaration
        payload.update(meta)
        payload['schema_version'] = max(4, int(payload.get('schema_version') or 1))
        mp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except Exception:
        return


def _annotate_boundary_manifest(boundary_manifest: str | None, image_path: Path, precision: str) -> None:
    if not boundary_manifest:
        return
    try:
        mp = Path(str(boundary_manifest)).expanduser()
        if not mp.is_file():
            return
        mj = json.loads(mp.read_text(encoding='utf-8'))
        seq = mj.get('seq')
        img = _resolve_image_for_seq(image_path, int(seq) if seq is not None else None)
        mj['input_image'] = img
        mj['provenance'] = {
            'image': img,
            'image_source': 'exact_file' if Path(image_path).expanduser().is_file() else 'directory_seq_mod',
            'seq': seq,
            'precision': precision,
        }
        mp.write_text(json.dumps(mj, indent=2), encoding='utf-8')
    except Exception:
        pass

def _seal_manifest_payload_files(manifest_path: Any) -> None:
    """Hash every binary referenced by a semantic/boundary manifest."""
    path=Path(str(manifest_path or '')).expanduser()
    if not path.is_file():
        return
    payload=_load_json(path)
    if not isinstance(payload,dict):
        raise RuntimeError(f'native_semantic_manifest_invalid:{path}')
    sealed=[]
    def _seal_file(value: Any, role: str) -> dict[str, Any]:
        declared_path=Path(str(value or ''))
        file_path=declared_path
        if not file_path.is_absolute(): file_path=path.parent/file_path
        file_path=file_path.resolve()
        if not file_path.is_file() or file_path.stat().st_size <= 0:
            raise RuntimeError(f'native_semantic_payload_missing:{role}:{file_path}')
        try:
            portable_path=str(file_path.relative_to(path.parent.resolve()))
        except ValueError:
            portable_path=str(file_path)
        row={'role':role,'path':portable_path,'sha256':_sha256_file(file_path),'size_bytes':int(file_path.stat().st_size)}
        sealed.append(row)
        return row
    for collection_name in ('outputs','tensors'):
        rows=payload.get(collection_name)
        if isinstance(rows,list):
            for index,row in enumerate(rows):
                if not isinstance(row,dict): continue
                value=row.get('file') or row.get('path')
                if value:
                    identity=_seal_file(value,f'{collection_name}[{index}]')
                    row['sha256']=identity['sha256']; row['size_bytes']=identity['size_bytes']
    for key in ('file','input_dump','selected_input_dump'):
        if payload.get(key):
            identity=_seal_file(payload[key],key)
            payload[f'{key}_sha256']=identity['sha256']; payload[f'{key}_size_bytes']=identity['size_bytes']
    if not sealed:
        raise RuntimeError(f'native_semantic_manifest_has_no_payload_files:{path}')
    payload['payload_artifacts']=sealed
    payload['payload_artifacts_sha256']=_stable_json_sha256(sealed)
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')


def _endpoint_result_path(work: Path, endpoint: str) -> Path:
    return work / "endpoints" / endpoint / "native_fifo_results.json"


def _endpoint_config_path(work: Path, endpoint: str) -> Path:
    return work / "endpoints" / endpoint / "native_fifo_config.json"


def _endpoint_work_dir(work: Path, endpoint: str) -> Path:
    return work / "endpoints" / endpoint


def _manifest_payload_identities(manifest_path: str | Path | None) -> list[dict[str, Any]]:
    """Return stable tensor payload identities independent of absolute paths."""
    if not manifest_path:
        return []
    path = Path(str(manifest_path)).expanduser()
    if not path.is_file():
        return []
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        return []
    rows = payload.get("outputs") or payload.get("tensors") or []
    if not isinstance(rows, list):
        return []
    identities: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            continue
        value = str(raw.get("file") or raw.get("path") or "")
        file_path = Path(value).expanduser()
        if value and not file_path.is_absolute():
            file_path = path.parent / file_path
        identities.append({
            "index": int(raw.get("index") if raw.get("index") is not None else index),
            "name": str(raw.get("name") or raw.get("tensor") or ""),
            "dtype": str(raw.get("dtype") or ""),
            "shape": [int(x) for x in list(raw.get("shape") or [])],
            "sha256": _sha256_file(file_path) if file_path.is_file() else "",
            "size_bytes": int(file_path.stat().st_size) if file_path.is_file() else 0,
        })
    return sorted(identities, key=lambda row: (row["index"], row["name"]))


def _boundary_payload_identity(manifest_path: str | Path | None) -> dict[str, Any]:
    if not manifest_path:
        return {}
    path = Path(str(manifest_path)).expanduser()
    if not path.is_file():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        return {}
    value = str(payload.get("file") or "")
    file_path = Path(value).expanduser()
    if value and not file_path.is_absolute():
        file_path = path.parent / file_path
    input_value = str(payload.get("input_dump") or "")
    input_path = Path(input_value).expanduser()
    if input_value and not input_path.is_absolute():
        input_path = path.parent / input_path
    return {
        "dtype": str(payload.get("dtype") or ""),
        "shape": [int(x) for x in list(payload.get("shape") or [])],
        "nbytes": int(payload.get("nbytes") or (file_path.stat().st_size if file_path.is_file() else 0)),
        "sha256": _sha256_file(file_path) if file_path.is_file() else "",
        "input_sha256": _sha256_file(input_path) if input_path.is_file() else "",
    }


def _dual_endpoint_relation(
    raw_payload: Mapping[str, Any], completed_payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_outputs = _manifest_payload_identities(
        raw_payload.get("native_fifo_output_manifest")
    )
    completed_outputs = _manifest_payload_identities(
        completed_payload.get("native_fifo_output_manifest")
    )
    raw_boundary = _boundary_payload_identity(
        raw_payload.get("native_fifo_boundary_manifest")
    )
    completed_boundary = _boundary_payload_identity(
        completed_payload.get("native_fifo_boundary_manifest")
    )
    output_exact = bool(raw_outputs and raw_outputs == completed_outputs)
    boundary_exact = bool(
        raw_boundary
        and completed_boundary
        and raw_boundary.get("sha256")
        and raw_boundary.get("sha256") == completed_boundary.get("sha256")
        and raw_boundary.get("dtype") == completed_boundary.get("dtype")
        and raw_boundary.get("shape") == completed_boundary.get("shape")
    )
    input_exact = bool(
        raw_boundary.get("input_sha256")
        and completed_boundary.get("input_sha256")
        and raw_boundary.get("input_sha256") == completed_boundary.get("input_sha256")
    )
    same_hef = bool(
        _sha256_file(raw_payload.get("hef"))
        and _sha256_file(raw_payload.get("hef"))
        == _sha256_file(completed_payload.get("hef"))
    )
    same_engine = bool(
        _sha256_file(raw_payload.get("engine"))
        and _sha256_file(raw_payload.get("engine"))
        == _sha256_file(completed_payload.get("engine"))
    )
    same_image = bool(
        str(raw_payload.get("input_image_sha256") or "")
        and str(raw_payload.get("input_image_sha256") or "")
        == str(completed_payload.get("input_image_sha256") or "")
    )
    verified = bool(
        output_exact and boundary_exact and input_exact
        and same_hef and same_engine and same_image
    )
    payload = {
        "schema": "onnx-splitpoint/native-dual-endpoint-relation",
        "schema_version": 1,
        "status": (
            "exact_shared_artifacts_input_boundary_and_raw_outputs"
            if verified else "endpoint_payload_or_artifact_parity_failed"
        ),
        "verified": verified,
        "same_hef_sha256": same_hef,
        "same_engine_sha256": same_engine,
        "same_input_image_sha256": same_image,
        "prepared_input_exact": input_exact,
        "boundary_exact": boundary_exact,
        "raw_outputs_exact": output_exact,
        "raw_output_tensors": raw_outputs,
        "completed_output_tensors": completed_outputs,
        "raw_boundary": raw_boundary,
        "completed_boundary": completed_boundary,
    }
    payload["relation_sha256"] = _stable_json_sha256(payload)
    return payload


def _dual_child_command(
    *, args: argparse.Namespace, endpoint: str, work: Path,
    result_json: Path, config_json: Path,
) -> list[str]:
    endpoint_work = _endpoint_work_dir(work, endpoint)
    output_dir = endpoint_work / "native_fifo_outputs"
    boundary_dir = endpoint_work / "native_fifo_boundary"
    cmd = [
        str(sys.executable), str(Path(__file__).resolve()),
        "--benchmark-set", str(args.benchmark_set),
        "--case", str(args.case),
        "--hw-arch", str(args.hw_arch),
        "--precision", str(args.precision),
        "--image", str(args.image),
        "--frames", str(int(args.frames)),
        "--warmup", str(int(args.warmup)),
        "--repetitions", str(int(args.repetitions)),
        "--duration-s", str(float(args.duration_s or 0.0)),
        "--queue-depth", str(int(args.queue_depth)),
        "--hailo-format", str(args.hailo_format),
        "--task", "detection",
        "--preprocess-mode", str(args.preprocess_mode),
        "--letterbox-pad-value", str(int(args.letterbox_pad_value)),
        "--work-dir", str(endpoint_work),
        "--result-json", str(result_json),
        "--config-json", str(config_json),
        "--output-dir", str(output_dir),
        "--boundary-dir", str(boundary_dir),
        "--detection-endpoints", endpoint,
        "--completion-runtime-mode", str(
            getattr(
                args,
                "completion_runtime_mode",
                "fast_oracle_outside_timing",
            )
        ),
        "--raw-preprocess-scope", str(args.raw_preprocess_scope),
        "--dump-outputs", "--dump-boundary",
        "--copy-outputs",
        "--setup-id", str(args.setup_id or ""),
        "--eval-run-id", str(args.eval_run_id or ""),
        "--source-run-id", str(args.source_run_id or "hailo8_to_trt"),
        "--model-id", str(args.model_id or ""),
    ]
    if not bool(args.build):
        cmd.append("--no-build")
    if not bool(args.run):
        cmd.append("--no-run")
    if args.device_id:
        cmd += ["--device-id", str(args.device_id)]
    if args.native_split_quality_binding:
        cmd += [
            "--native-split-quality-binding",
            str(args.native_split_quality_binding),
        ]
    for option, value in (
        ("--expected-runner-sha256", args.expected_runner_sha256),
        ("--expected-image-sha256", args.expected_image_sha256),
        ("--expected-hef-sha256", args.expected_hef_sha256),
        ("--expected-engine-sha256", args.expected_engine_sha256),
        ("--expected-boundary-layout", args.expected_boundary_layout),
        ("--source-contract-sha256", args.source_contract_sha256),
        ("--prepared-input-rgb", getattr(args, "prepared_input_rgb", "")),
        ("--expected-prepared-input-sha256", getattr(args, "expected_prepared_input_sha256", "")),
    ):
        if value:
            cmd += [option, str(value)]
    # The archived Detection executable identity belongs to the historical
    # Python Completed-Task endpoint.  A C++ raw endpoint has its own freshly
    # sealed executable identity and must not inherit that ambiguous hash.
    if endpoint == "completed_task" and args.expected_executable_sha256:
        cmd += [
            "--expected-executable-sha256",
            str(args.expected_executable_sha256),
        ]
    return cmd




def _v2791_use_concurrent_three_stage(args: argparse.Namespace) -> bool:
    """Admit only the hardware-validated YOLOv7 Hailo-8 contract."""
    model_token = str(getattr(args, "model_id", "") or "").strip().lower()
    return bool(
        str(getattr(args, "task", "")) == "detection"
        and str(getattr(args, "detection_endpoints", "")) == "dual"
        and str(getattr(args, "completion_runtime_mode", ""))
        == "fast_oracle_outside_timing"
        and str(getattr(args, "hw_arch", "")) == "hailo8"
        and str(getattr(args, "case", "")) == "b066"
        and str(getattr(args, "precision", "")) == "uint8_dequant_fp16"
        and model_token in {"", "yolov7", "yolov7_paper"}
        and bool(str(getattr(args, "native_split_quality_binding", "") or ""))
        and bool(getattr(args, "run", True))
    )


def _v2791_concurrent_three_stage_command(
    *, args: argparse.Namespace, work: Path, result_json: Path,
) -> list[str]:
    helper = Path(__file__).resolve().with_name(
        "native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
    )
    if not helper.is_file():
        raise FileNotFoundError(f"concurrent_three_stage_helper_missing:{helper}")
    binding = str(getattr(args, "native_split_quality_binding", "") or "")
    if not binding:
        raise RuntimeError("concurrent_three_stage_requires_native_split_quality_binding")
    cmd = [
        str(sys.executable), str(helper),
        "--benchmark-set", str(args.benchmark_set),
        "--case", str(args.case),
        "--hw-arch", str(args.hw_arch),
        "--precision", str(args.precision),
        "--image", str(args.image),
        "--frames", str(int(args.frames)),
        "--warmup", str(int(args.warmup)),
        "--repetitions", str(int(args.repetitions)),
        "--queue-depth", str(int(args.queue_depth)),
        "--post-queue-depth", str(int(getattr(args, "post_queue_depth", 4))),
        "--duration-s", str(float(args.duration_s or 0.0)),
        "--setup-id", str(args.setup_id or ""),
        "--eval-run-id", str(args.eval_run_id or ""),
        "--source-run-id", str(args.source_run_id or "hailo8_to_trt"),
        "--model-id", str(args.model_id or "yolov7_paper"),
        "--native-split-quality-binding", binding,
        "--work-dir", str(work / "concurrent_three_stage"),
        "--result-json", str(result_json),
        "--timeout-s", str(float(getattr(args, "concurrent_timeout_s", 3600.0))),
    ]
    for option, value in (
        ("--expected-image-sha256", args.expected_image_sha256),
        ("--expected-hef-sha256", args.expected_hef_sha256),
        ("--expected-engine-sha256", args.expected_engine_sha256),
        ("--expected-boundary-layout", args.expected_boundary_layout),
    ):
        if value:
            cmd += [option, str(value)]
    return cmd


def _v2791_oracle_parity_passed(payload: Mapping[str, Any]) -> bool:
    parity = payload.get("oracle_parity")
    return bool(
        str(payload.get("quality_oracle_status") or "").strip().lower() == "passed"
        and isinstance(parity, Mapping)
        and str(parity.get("status") or "").strip().lower() == "passed"
    )


def _v2791_concurrent_benchmark_row(
    *, bs: Path, case: str, args: argparse.Namespace,
    result_path: Path, payload: Mapping[str, Any],
) -> dict[str, Any]:
    p2_fps = payload.get("p2_output_fps")
    completed_fps = payload.get("completed_detection_fps")
    oracle_passed = _v2791_oracle_parity_passed(payload)
    identity = payload.get("identity") if isinstance(payload.get("identity"), Mapping) else {}
    return {
        "model_id": str(args.model_id or _benchmark_model_id(bs)),
        "case_id": case,
        "backend": f"{args.hw_arch}_to_tensorrt",
        "run_id": f"{args.hw_arch}_to_trt_native_fifo",
        "variant": "composed",
        "primary_variant": "composed",
        "stage1_provider": args.hw_arch,
        "stage2_provider": "tensorrt",
        "native_fifo_enabled": True,
        "native_fifo_mode": payload.get("mode"),
        "native_fifo_precision": args.precision,
        "performance_endpoint": "p2_output",
        "primary_performance_endpoint": "p2_output",
        "application_performance_endpoint": "completed_detection",
        "energy_performance_endpoint": "completed_detection",
        "throughput_primary_metric": "p2_output_makespan_fps",
        "throughput_primary_fps": p2_fps,
        "pipeline_fps_selected": p2_fps,
        "p2_output_fps": p2_fps,
        "completed_detection_fps": completed_fps,
        "application_throughput_fps": completed_fps,
        "completed_to_p2_ratio": payload.get("completed_to_p2_ratio"),
        "p2_output_contract_family": payload.get("p2_output_contract_family"),
        "postprocess_adapter_id": payload.get("postprocess_adapter_id"),
        "postprocess_location": payload.get("postprocess_location"),
        "quality_oracle_location": payload.get("quality_oracle_location"),
        "quality_oracle_status": payload.get("quality_oracle_status"),
        "endpoint_execution_policy": payload.get("endpoint_execution_policy"),
        "three_stage_concurrency_directly_measured": payload.get(
            "three_stage_concurrency_directly_measured"
        ),
        "three_stage_hardware_integration_status": payload.get(
            "three_stage_hardware_integration_status"
        ),
        "directly_measured": payload.get("directly_measured"),
        "endpoint_relation_verified": payload.get("endpoint_relation_verified"),
        "stage_timings": payload.get("stage_timings"),
        "oracle_parity": payload.get("oracle_parity"),
        "phases": payload.get("phases"),
        "postprocess_included": True,
        "postprocess_completion_verified": oracle_passed,
        "runtime_ok": payload.get("ok") is True,
        "validation_ok": oracle_passed,
        "native_fifo_results_json": str(result_path),
        "native_split_quality_binding": identity.get("binding"),
        "native_split_quality_binding_sha256": identity.get("binding_sha256"),
        "setup_id": str(args.setup_id),
        "eval_run_id": str(args.eval_run_id),
        "source_run_id": canonical_native_split_backend(
            args.source_run_id, args.setup_id,
        ),
    }


def _run_v2791_concurrent_three_stage(
    *, bs: Path, case: str, work: Path, args: argparse.Namespace,
) -> int:
    result_path = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json else work / "native_fifo_results.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _v2791_concurrent_three_stage_command(
        args=args, work=work, result_json=result_path,
    )
    print(_CONCURRENT_LOG_PREFIX + " " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(work))
    if not result_path.is_file():
        raise RuntimeError(
            f"concurrent_three_stage_result_missing_after_rc:{proc.returncode}:"
            f"{result_path}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    benchmark_results = bs / f"benchmark_results_native_fifo_{case}.json"
    row = _v2791_concurrent_benchmark_row(
        bs=bs, case=case, args=args,
        result_path=result_path, payload=payload,
    )
    benchmark_results.write_text(json.dumps({
        "schema": "onnx-splitpoint/native-fifo-benchmark-results",
        "schema_version": 3,
        "endpoint_model": "one_native_deployment_concurrent_three_stage_two_endpoints",
        "results": [row],
    }, indent=2), encoding="utf-8")
    print(_CONCURRENT_LOG_PREFIX + " results:", result_path)
    print(_CONCURRENT_LOG_PREFIX + " benchmark results row:", benchmark_results)
    if proc.returncode != 0:
        return int(proc.returncode)
    if payload.get("ok") is not True:
        return 1
    if not _v2791_oracle_parity_passed(payload):
        print("concurrent_three_stage_oracle_parity_failed", file=sys.stderr)
        return 9
    if payload.get("three_stage_concurrency_directly_measured") is not True:
        print("concurrent_three_stage_direct_measurement_missing", file=sys.stderr)
        return 9
    if payload.get("endpoint_relation_verified") is not True:
        print("concurrent_three_stage_endpoint_relation_failed", file=sys.stderr)
        return 9
    return 0

def _v279_three_stage_projection(
    *,
    payload: Mapping[str, Any],
    raw_payload: Mapping[str, Any],
    completed_payload: Mapping[str, Any],
    model_id: str,
) -> dict[str, Any]:
    """Project directly measured dual observations onto v2.79 endpoint names.

    The v2.78 child directories remain readable as compatibility artifacts.
    This projection never claims that the two isolated child phases were one
    concurrent run; ``endpoint_execution_policy`` remains explicit.
    """
    try:
        contract_family = infer_p2_output_contract_family(
            completed_payload, model_id=model_id, task="detection",
        )
    except NativeThreeStageError:
        # Archived v2.78 dual-endpoint payloads and narrow unit fixtures may
        # predate the explicit physical P2 attestation.  A known, sealed model
        # family can be projected for compatibility, but unknown models remain
        # fail-closed.
        token = str(model_id or "").lower().replace("-", "").replace("_", "")
        if "yolov7" in token or "yolo7" in token:
            contract_family = "yolov7_anchor_multiscale_raw"
        elif "yolo11" in token:
            contract_family = "yolo11_regcls_dfl16_raw"
        elif "yolo26" in token:
            contract_family = "yolo26_decoded_nms"
        else:
            raise
    adapter_id = adapter_id_for_contract_family(contract_family)
    raw_fps = raw_payload.get("fps_median", raw_payload.get("fps_makespan"))
    completed_fps = completed_payload.get(
        "fps_median", completed_payload.get("fps_makespan")
    )
    projected = project_three_stage_endpoints(
        payload,
        p2_output_fps=raw_fps,
        completed_detection_fps=completed_fps,
        p2_output_contract_family=contract_family,
        postprocess_adapter_id=adapter_id,
        postprocess_location=str(
            completed_payload.get("postprocess_location")
            or "host_cpu_python_numpy"
        ),
        p1_samples_ms=[
            raw_payload.get("p1_thread_ms", raw_payload.get("p1_ms"))
        ],
        p2_samples_ms=[
            raw_payload.get("p2_thread_ms", raw_payload.get("p2_run_ms"))
        ],
        post_samples_ms=[completed_payload.get("completion_tail_ms")],
        directly_measured=True,
    )
    projected.update({
        "endpoint_execution_policy": (
            "isolated_sequential_fresh_runtime_phases"
        ),
        "three_stage_concurrency_directly_measured": False,
        "three_stage_hardware_integration_status": (
            "endpoint_contract_integrated; concurrent_three_stage_smoke_required"
        ),
    })
    return projected

def _dual_endpoint_benchmark_row(
    *, bs: Path, case: str, args: argparse.Namespace,
    combined_path: Path, raw_payload: Mapping[str, Any],
    completed_payload: Mapping[str, Any], relation: Mapping[str, Any],
) -> dict[str, Any]:
    raw_fps = raw_payload.get("fps_median", raw_payload.get("fps_makespan"))
    completed_fps = completed_payload.get(
        "fps_median", completed_payload.get("fps_makespan")
    )
    model_id = str(args.model_id or _benchmark_model_id(bs))
    row = {
        "model_id": model_id,
        "case_id": case,
        "backend": f"{args.hw_arch}_to_tensorrt",
        "run_id": f"{args.hw_arch}_to_trt_native_fifo",
        "variant": "composed",
        "primary_variant": "composed",
        "stage1_provider": args.hw_arch,
        "stage2_provider": "tensorrt",
        "native_fifo_enabled": True,
        "native_fifo_mode": "native_hailort_tensorrt_dual_endpoint_fifo",
        "native_fifo_precision": args.precision,
        "performance_endpoint": "p2_output",
        "primary_performance_endpoint": "p2_output",
        "application_performance_endpoint": "completed_detection",
        "energy_performance_endpoint": "completed_detection",
        "throughput_primary_metric": "p2_output_makespan_fps",
        "legacy_performance_endpoint": "raw_model_outputs",
        "legacy_application_performance_endpoint": "completed_task",
        "throughput_primary_fps": raw_fps,
        "pipeline_fps_selected": raw_fps,
        "native_fifo_fps_makespan": raw_payload.get("fps_makespan"),
        "fps_median": raw_payload.get("fps_median", raw_fps),
        "fps_ci95_low": raw_payload.get("fps_ci95_low"),
        "fps_ci95_high": raw_payload.get("fps_ci95_high"),
        "raw_model_outputs_fps_makespan": raw_payload.get("fps_makespan"),
        "raw_model_outputs_fps_median": raw_payload.get("fps_median", raw_fps),
        "raw_model_outputs_paper_equivalent_fps": raw_payload.get("paper_equivalent_fps"),
        "raw_model_outputs_preprocess_ms": raw_payload.get("preprocess_ms"),
        "raw_model_outputs_p1_ms": raw_payload.get("p1_ms"),
        "raw_model_outputs_handoff_ms": raw_payload.get("handoff_ms"),
        "raw_model_outputs_p2_run_ms": raw_payload.get("p2_run_ms"),
        "raw_model_outputs_p1_thread_ms": raw_payload.get("p1_thread_ms"),
        "raw_model_outputs_p2_thread_ms": raw_payload.get("p2_thread_ms"),
        "raw_model_outputs_measurement_boundary": raw_payload.get("measurement_boundary"),
        "raw_model_outputs_prepared_feed_contract": raw_payload.get("prepared_feed_contract"),
        "completed_task_fps_makespan": completed_payload.get("fps_makespan"),
        "completed_task_fps_median": completed_payload.get("fps_median", completed_fps),
        "completed_task_fps_ci95_low": completed_payload.get("fps_ci95_low"),
        "completed_task_fps_ci95_high": completed_payload.get("fps_ci95_high"),
        "completed_task_p1_ms": completed_payload.get("p1_ms"),
        "completed_task_handoff_ms": completed_payload.get("handoff_ms"),
        "completed_task_p2_run_ms": completed_payload.get("p2_run_ms"),
        "completed_task_completion_tail_ms": completed_payload.get("completion_tail_ms"),
        "completed_task_p1_thread_ms": completed_payload.get("p1_thread_ms"),
        "completed_task_p2_thread_ms": completed_payload.get("p2_thread_ms"),
        "completed_task_measurement_boundary": completed_payload.get("measurement_boundary"),
        "completed_work_units": completed_payload.get("completed_work_units"),
        "postprocess_included": True,
        "postprocess_completion_verified": True,
        "runtime_ok": True,
        "validation_ok": None,
        "native_fifo_results_json": str(combined_path),
        "native_fifo_raw_model_outputs_results_json": str(
            _endpoint_result_path(combined_path.parent, "raw_model_outputs")
        ),
        "native_fifo_completed_task_results_json": str(
            _endpoint_result_path(combined_path.parent, "completed_task")
        ),
        "native_fifo_output_manifest": completed_payload.get("native_fifo_output_manifest"),
        "native_fifo_boundary_manifest": completed_payload.get("native_fifo_boundary_manifest"),
        "raw_model_outputs_output_manifest": raw_payload.get("native_fifo_output_manifest"),
        "raw_model_outputs_boundary_manifest": raw_payload.get("native_fifo_boundary_manifest"),
        "dual_endpoint_relation": dict(relation),
        "dual_endpoint_relation_verified": relation.get("verified") is True,
        "setup_id": str(args.setup_id),
        "eval_run_id": str(args.eval_run_id),
        "source_run_id": canonical_native_split_backend(
            args.source_run_id, args.setup_id,
        ),
        "native_command_contract": completed_payload.get("native_command_contract"),
        "native_command_contract_sha256": completed_payload.get("native_command_contract_sha256"),
        "workload_contract_sha256": completed_payload.get("workload_contract_sha256"),
        "raw_model_outputs_command_contract": raw_payload.get("native_command_contract"),
        "raw_model_outputs_command_contract_sha256": raw_payload.get("native_command_contract_sha256"),
        "completed_task_command_contract": completed_payload.get("native_command_contract"),
        "completed_task_command_contract_sha256": completed_payload.get("native_command_contract_sha256"),
        "native_split_quality_binding": completed_payload.get("native_split_quality_binding"),
        "native_split_quality_binding_sha256": completed_payload.get("native_split_quality_binding_sha256"),
        "native_split_quality_eval_run_id": completed_payload.get("native_split_quality_eval_run_id"),
        "native_split_quality_source_run_id": completed_payload.get("native_split_quality_source_run_id"),
        "source_request_sha256": completed_payload.get("source_request_sha256"),
        "native_split_quality_source_request_sha256": completed_payload.get("native_split_quality_source_request_sha256"),
        "native_split_quality_central_result_sha256": completed_payload.get("native_split_quality_central_result_sha256"),
        "native_split_quality_selection_sha256": completed_payload.get("native_split_quality_selection_sha256"),
        "native_split_quality_consumer_attestation": completed_payload.get("native_split_quality_consumer_attestation"),
        "native_split_quality_consumer_status": completed_payload.get("native_split_quality_consumer_status"),
        "completed_task_endpoint_contract": completed_payload.get("completed_task_endpoint_contract"),
        "completed_task_comparison_endpoint_contract": completed_payload.get("completed_task_comparison_endpoint_contract"),
        "completion_execution_attestation": completed_payload.get("completion_execution_attestation"),
    }
    return _v279_three_stage_projection(
        payload=row,
        raw_payload=raw_payload,
        completed_payload=completed_payload,
        model_id=model_id,
    )


def _pin_raw_prepared_feed(path: Path, expected_sha256: str = "") -> str:
    digest = _sha256_file(path)
    if not digest or (expected_sha256 and digest != expected_sha256):
        raise RuntimeError("shared_prepared_raw_repetition_feed_changed")
    return digest


def _shared_rgb_from_raw_endpoint(
    payload: Mapping[str, Any], *, args: argparse.Namespace,
) -> dict[str, Any]:
    """Bind completed-task input to the actual, sealed raw endpoint feed."""
    contract = dict(payload.get("native_command_contract") or {})
    declared_hash = contract.pop("contract_sha256", "")
    if not declared_hash or declared_hash != _stable_json_sha256(contract) or contract.get("complete") is not True:
        raise RuntimeError("shared_prepared_raw_contract_invalid")
    prepared = dict(contract.get("prepared_input_contract") or {})
    artifact = dict((contract.get("artifacts") or {}).get("prepared_input") or {})
    path = Path(str(artifact.get("path") or "")).expanduser().resolve()
    shape = [int(x) for x in prepared.get("shape") or []]
    if prepared.get("dtype") != "uint8" or prepared.get("layout") != "HWC" or len(shape) != 3 or shape[-1] != 3 or any(x <= 0 for x in shape):
        raise RuntimeError("shared_prepared_raw_rgb_contract_invalid")
    digest = _sha256_file(path)
    if not digest or digest != artifact.get("sha256") or path.stat().st_size != math.prod(shape):
        raise RuntimeError("shared_prepared_raw_artifact_mismatch")
    image_hash = _sha256_file(args.image)
    if not image_hash or image_hash != contract.get("input_image_sha256") or image_hash != prepared.get("source_image_sha256"):
        raise RuntimeError("shared_prepared_raw_image_mismatch")
    options = dict(contract.get("runtime_options") or {})
    mode = str(getattr(args, "preprocess_mode_effective", "") or _resolve_preprocess_mode(str(args.task if hasattr(args, "task") else "detection"), str(args.preprocess_mode)))
    if not hailo8_preprocess_binding(options, prepared) or prepared.get("preprocess_mode_effective") != mode or int(prepared.get("letterbox_pad_value_requested", -1)) != int(args.letterbox_pad_value):
        raise RuntimeError("shared_prepared_raw_preprocess_mismatch")
    if _boundary_payload_identity(payload.get("native_fifo_boundary_manifest")).get("input_sha256") != digest:
        raise RuntimeError("shared_prepared_raw_consumed_bytes_mismatch")
    return {"path": str(path), "sha256": digest, "shape": shape}


def _run_detection_dual_endpoint(
    *, bs: Path, case: str, work: Path, args: argparse.Namespace,
) -> int:
    """Run one Native invocation with two isolated endpoint phases.

    The C++ raw phase and the existing correctness-first Completed-Task phase
    are separate child processes.  This prevents Decode/NMS backpressure from
    throttling the raw model-output makespan while preserving the exact quality
    and completion contracts of the second phase.
    """
    canonical_out = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json else work / "native_fifo_results.json"
    )
    canonical_out.parent.mkdir(parents=True, exist_ok=True)
    raw_result = _endpoint_result_path(work, "raw_model_outputs")
    completed_result = _endpoint_result_path(work, "completed_task")
    for path in (raw_result, completed_result):
        path.parent.mkdir(parents=True, exist_ok=True)
    phases: list[dict[str, Any]] = []
    # Run raw first so a completed-task failure never erases the recovered
    # hardware-fast observation.  The combined result is still fail-closed.
    for endpoint, result_path in (
        ("raw_model_outputs", raw_result),
        ("completed_task", completed_result),
    ):
        child_args = argparse.Namespace(**vars(args))
        if endpoint == "completed_task":
            try:
                shared = _shared_rgb_from_raw_endpoint(_load_json(raw_result), args=args)
                child_args.prepared_input_rgb = str(shared["path"])
                child_args.expected_prepared_input_sha256 = str(shared["sha256"])
            except Exception as exc:
                failure = {
                    "schema": "onnx-splitpoint/native-dual-endpoint-result",
                    "schema_version": 1,
                    "ok": False,
                    "mode": "native_hailort_tensorrt_dual_endpoint_fifo",
                    "dual_endpoint_enabled": True,
                    "failure_reason": "shared_prepared_input_binding_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "phases": phases,
                    "endpoint_result_paths": {"raw_model_outputs": str(raw_result)},
                    "completed_task_execution": "not_started_invalid_shared_input",
                }
                canonical_out.write_text(json.dumps(failure, indent=2), encoding="utf-8")
                return 8
        command = _dual_child_command(
            args=child_args,
            endpoint=endpoint,
            work=work,
            result_json=result_path,
            config_json=_endpoint_config_path(work, endpoint),
        )
        print(f"[native-fifo][dual] {endpoint}:", " ".join(command))
        started = time.perf_counter()
        completed = subprocess.run(command, check=False)
        phases.append({
            "endpoint": endpoint,
            "command": command,
            "returncode": int(completed.returncode),
            "elapsed_s": float(time.perf_counter() - started),
            "result_json": str(result_path),
            "result_sha256": _sha256_file(result_path),
        })
        if completed.returncode != 0 or not result_path.is_file():
            failure = {
                "schema": "onnx-splitpoint/native-dual-endpoint-result",
                "schema_version": 1,
                "ok": False,
                "mode": "native_hailort_tensorrt_dual_endpoint_fifo",
                "dual_endpoint_enabled": True,
                "failure_reason": f"{endpoint}_phase_failed",
                "phases": phases,
            }
            canonical_out.write_text(json.dumps(failure, indent=2), encoding="utf-8")
            return int(completed.returncode or 8)
    raw_payload = _load_json(raw_result)
    completed_payload = _load_json(completed_result)
    if not isinstance(raw_payload, Mapping) or raw_payload.get("ok") is not True:
        raise RuntimeError("dual_endpoint_raw_payload_invalid")
    if not isinstance(completed_payload, Mapping) or completed_payload.get("ok") is not True:
        raise RuntimeError("dual_endpoint_completed_payload_invalid")
    relation = _dual_endpoint_relation(raw_payload, completed_payload)
    combined = dict(completed_payload)
    combined.update({
        "schema": "onnx-splitpoint/native-dual-endpoint-result",
        "schema_version": 1,
        "ok": relation.get("verified") is True,
        # Keep the legacy top-level Completed-Task fields intact for energy and
        # quality consumers; primary performance is explicit below.
        "mode": "native_hailort_tensorrt_dual_endpoint_fifo",
        "dual_endpoint_enabled": True,
        "performance_endpoint": "p2_output",
        "primary_performance_endpoint": "p2_output",
        "application_performance_endpoint": "completed_detection",
        "energy_performance_endpoint": "completed_detection",
        "throughput_primary_metric": "p2_output_makespan_fps",
        "legacy_performance_endpoint": "raw_model_outputs",
        "legacy_application_performance_endpoint": "completed_task",
        "throughput_primary_fps": raw_payload.get("fps_median", raw_payload.get("fps_makespan")),
        "pipeline_fps_selected": raw_payload.get("fps_median", raw_payload.get("fps_makespan")),
        "application_throughput_metric": "completed_detection_makespan_fps",
        "application_throughput_fps": completed_payload.get("fps_median", completed_payload.get("fps_makespan")),
        "endpoint_execution_policy": "isolated_sequential_fresh_runtime_phases",
        "endpoint_order": ["raw_model_outputs", "completed_task"],
        "endpoint_results": {
            "raw_model_outputs": dict(raw_payload),
            "completed_task": dict(completed_payload),
        },
        "endpoint_result_files": {
            "raw_model_outputs": str(raw_result),
            "completed_task": str(completed_result),
        },
        "endpoint_relation": relation,
        "endpoint_relation_verified": relation.get("verified") is True,
        "raw_model_outputs_fps_makespan": raw_payload.get("fps_makespan"),
        "raw_model_outputs_fps_median": raw_payload.get("fps_median", raw_payload.get("fps_makespan")),
        "raw_model_outputs_fps_ci95_low": raw_payload.get("fps_ci95_low"),
        "raw_model_outputs_fps_ci95_high": raw_payload.get("fps_ci95_high"),
        "raw_model_outputs_paper_equivalent_fps": raw_payload.get("paper_equivalent_fps"),
        "raw_model_outputs_preprocess_ms": raw_payload.get("preprocess_ms"),
        "raw_model_outputs_p1_ms": raw_payload.get("p1_ms"),
        "raw_model_outputs_handoff_ms": raw_payload.get("handoff_ms"),
        "raw_model_outputs_p2_run_ms": raw_payload.get("p2_run_ms"),
        "raw_model_outputs_p1_thread_ms": raw_payload.get("p1_thread_ms"),
        "raw_model_outputs_p2_thread_ms": raw_payload.get("p2_thread_ms"),
        "raw_model_outputs_measurement_boundary": raw_payload.get("measurement_boundary"),
        "completed_task_fps_makespan": completed_payload.get("fps_makespan"),
        "completed_task_fps_median": completed_payload.get("fps_median", completed_payload.get("fps_makespan")),
        "completed_task_fps_ci95_low": completed_payload.get("fps_ci95_low"),
        "completed_task_fps_ci95_high": completed_payload.get("fps_ci95_high"),
        "completed_task_p1_ms": completed_payload.get("p1_ms"),
        "completed_task_handoff_ms": completed_payload.get("handoff_ms"),
        "completed_task_p2_run_ms": completed_payload.get("p2_run_ms"),
        "completed_task_completion_tail_ms": completed_payload.get("completion_tail_ms"),
        "completed_task_p1_thread_ms": completed_payload.get("p1_thread_ms"),
        "completed_task_p2_thread_ms": completed_payload.get("p2_thread_ms"),
        "completed_task_measurement_boundary": completed_payload.get("measurement_boundary"),
        "raw_model_outputs_energy_eligible": False,
        "completed_task_energy_eligible": True,
        "phases": phases,
    })
    combined = _v279_three_stage_projection(
        payload=combined,
        raw_payload=raw_payload,
        completed_payload=completed_payload,
        model_id=str(args.model_id or _benchmark_model_id(bs)),
    )
    canonical_out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    row = _dual_endpoint_benchmark_row(
        bs=bs,
        case=case,
        args=args,
        combined_path=canonical_out,
        raw_payload=raw_payload,
        completed_payload=completed_payload,
        relation=relation,
    )
    benchmark_results = bs / f"benchmark_results_native_fifo_{case}.json"
    benchmark_results.write_text(json.dumps({
        "schema": "onnx-splitpoint/native-fifo-benchmark-results",
        "schema_version": 2,
        "endpoint_model": "one_native_deployment_two_isolated_endpoint_observations",
        "results": [row],
    }, indent=2), encoding="utf-8")
    print("[native-fifo][dual] results:", canonical_out)
    print("[native-fifo][dual] benchmark results row:", benchmark_results)
    if relation.get("verified") is not True:
        print("dual_endpoint_payload_parity_failed", file=sys.stderr)
        return 9
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description='Build/run native HailoRT->TensorRT FIFO fastpath for a BenchmarkSet case.')
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True, help='Case id such as b066')
    ap.add_argument('--hw-arch', default='hailo8')
    ap.add_argument('--precision', default='uint8_cast_fp16', choices=['fp16', 'uint8_cast_fp16', 'uint8_dequant_fp16', 'float32_layout_fp16'], help='Native TRT part2 engine precision/tag to use.')
    ap.add_argument('--image', default='', help='Image file or directory. Defaults to validation dir/test image in BenchmarkSet.')
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--repetitions', type=int, default=1, help='Independent native process measurements; reported as median with a 95%% repetition-level CI (never best-of).')
    ap.add_argument('--duration-s', type=float, default=0.0, help='Run measured workload for this many seconds instead of a fixed frame count. Used for duration-based energy windows.')
    ap.add_argument('--queue-depth', type=int, default=3)
    ap.add_argument('--hailo-format', default='uint8', choices=['uint8', 'float32', 'auto'])
    ap.add_argument('--task', default='', choices=['', 'classification', 'detection'], help='Benchmark task. If omitted it is read from benchmark_set.json.')
    ap.add_argument('--detection-endpoints', default='dual', choices=['dual', 'p2_output', 'completed_detection', 'raw_model_outputs', 'completed_task'], help='Detection execution contract. dual exposes v2.79 p2_output and completed_detection observations while retaining legacy child aliases.')
    ap.add_argument('--completion-runtime-mode', default='fast_oracle_outside_timing', choices=['fast_oracle_outside_timing', 'legacy_attested_hotloop'], help='Postprocess evidence policy for the completed_detection phase. The v2.79 default executes task work in timing and runs the frozen Quality oracle once in postflight.')
    ap.add_argument('--raw-preprocess-scope', default='shared_prepared_input', choices=['paper_image', 'shared_prepared_input'], help='Raw endpoint input scope. paper_image includes per-frame image read/preprocess like the paper; shared_prepared_input reuses one prepared feed for direct endpoint overhead comparisons.')
    ap.add_argument('--prepared-input-rgb', default='', help='Exact HWC RGB uint8 payload shared by native endpoint phases, loaded outside timing.')
    ap.add_argument('--expected-prepared-input-sha256', default='', help='Verify the shared RGB payload against the producer contract before inference.')
    ap.add_argument('--preprocess-mode', default='auto', choices=['auto', 'resize', 'letterbox'], help='Image preprocessing; auto resolves to resize for classification and letterbox for detection.')
    ap.add_argument('--device-id', default='')
    ap.add_argument('--copy-outputs', action='store_true', default=True)
    ap.add_argument('--no-copy-outputs', dest='copy_outputs', action='store_false')
    ap.add_argument('--dump-outputs', action='store_true', default=False, help='Dump last TensorRT output tensors as raw .bin files plus manifest for generic validation.')
    ap.add_argument('--dump-boundary', action='store_true', default=False, help='Run one separate bound inference after timing and dump its boundary/input payload as .bin plus manifest.')
    ap.add_argument('--letterbox-pad-value', type=int, default=0, help='RGB letterbox padding value used by the native Hailo input preprocessor. 0 preserves historical native behavior; 114 matches the YOLO generic harness.')
    ap.add_argument('--output-dir', default='', help='Output tensor dump directory. Defaults to <work-dir>/native_fifo_outputs.')
    ap.add_argument('--boundary-dir', default='', help='Boundary dump directory. Defaults to <work-dir>/native_fifo_boundary.')
    ap.add_argument('--build', action='store_true', default=True)
    ap.add_argument('--no-build', dest='build', action='store_false')
    ap.add_argument('--run', action='store_true', default=True)
    ap.add_argument('--no-run', dest='run', action='store_false')
    ap.add_argument('--work-dir', default='', help='Output work dir. Defaults to <BS>/native_pipeline/<case>/hailo_to_trt/<precision>.')
    ap.add_argument('--result-json', default='', help='Fresh result JSON path. Used by hash-bound A/B replays to prevent stale-result reuse.')
    ap.add_argument('--config-json', default='', help='Fresh resolved-command contract path. Defaults to <work-dir>/native_fifo_config.json.')
    ap.add_argument('--expected-runner-sha256', default='', help='Fail closed unless the executing runner matches this archived SHA-256.')
    ap.add_argument('--expected-image-sha256', default='', help='Fail closed unless the exact input image matches this archived SHA-256.')
    ap.add_argument('--expected-hef-sha256', default='', help='Fail closed unless the selected HEF matches this archived SHA-256.')
    ap.add_argument('--expected-engine-sha256', default='', help='Fail closed unless the selected TensorRT engine matches this archived SHA-256.')
    ap.add_argument('--expected-executable-sha256', default='', help='Fail closed unless the prebuilt native FIFO executable matches this archived SHA-256.')
    ap.add_argument('--expected-boundary-layout', default='', help='Fail closed unless the existing engine embeds this archived boundary layout.')
    ap.add_argument('--source-contract-sha256', default='', help='Archived successful command-contract identity recorded in replay evidence.')
    ap.add_argument('--setup-id', default=os.environ.get('ONNX_SPLITPOINT_SETUP_ID', ''))
    ap.add_argument('--eval-run-id', default='')
    ap.add_argument('--source-run-id', default='hailo8_to_trt')
    ap.add_argument('--model-id', default='')
    ap.add_argument('--native-split-quality-binding', default='', help='Quality-FIRST split binding; locally re-hashed and consumed with no engine discovery or rebuild.')
    ap.add_argument('--energy-workload-only', action='store_true', help='Run only a preflight-attested, warmup-free measured hotloop. No build/hash/dump operations are permitted.')
    ap.add_argument('--energy-preflight-attestation', default='', help='Fresh nonce-bound split-energy preflight attestation JSON.')
    ap.add_argument('--energy-preflight-nonce', default='', help='Fresh collector repeat nonce expected in the preflight attestation.')
    ap.add_argument('--energy-preflight-max-age-s', type=float, default=300.0, help='Maximum accepted age of the preflight attestation.')
    args = ap.parse_args()
    if args.detection_endpoints == "p2_output":
        args.detection_endpoints = "raw_model_outputs"
    elif args.detection_endpoints == "completed_detection":
        args.detection_endpoints = "completed_task"
    if int(args.repetitions) < 1:
        ap.error('--repetitions must be >= 1')

    if args.energy_workload_only:
        if int(args.repetitions) != 1:
            print('split_energy_repetitions_must_be_one', file=sys.stderr)
            return 6
        if not args.energy_preflight_attestation or not args.energy_preflight_nonce:
            print('split_energy_preflight_attestation_or_nonce_missing', file=sys.stderr)
            return 6
        if not args.result_json:
            print('split_energy_fresh_result_json_missing', file=sys.stderr)
            return 6
        return _energy_workload_only(args)

    if not bool(args.copy_outputs):
        print('native_claim_run_requires_copy_outputs', file=sys.stderr)
        return 6

    bs = Path(args.benchmark_set).expanduser().resolve()
    args.task = str(args.task or _benchmark_task(bs)).strip().lower()
    args.preprocess_mode_effective = _resolve_preprocess_mode(args.task, args.preprocess_mode)
    args.mixed_runtime_probe = (
        _require_hailo8_detection_runtime()
        if args.task == "detection"
        and args.detection_endpoints in {"dual", "completed_task"}
        else {}
    )
    case = args.case if args.case.startswith('b') else f'b{int(args.case):03d}'
    expected_boundary_name = ""
    expected_boundary_shape: list[int] = []
    expected_boundary_dtype = ""
    expected_boundary_bytes = 0
    quality_binding = _load_native_split_quality_binding(
        args.native_split_quality_binding, args=args, case=case, task=args.task,
    )
    if quality_binding is not None:
        selection = quality_binding['preselection']
        args.precision = str(selection['precision'])
        args.hailo_format = str(selection['hailo_format'])
        args.preprocess_mode = str(selection['preprocess_mode'])
        args.preprocess_mode_effective = str(selection['preprocess_mode'])
        args.letterbox_pad_value = int(selection['letterbox_pad_value'])
        hef = _binding_artifact_path(quality_binding, 'part1_runtime')
        engine = _binding_artifact_path(quality_binding, 'engine')
        boundary_metadata = _load_json(_binding_artifact_path(quality_binding, 'boundary_metadata'))
        boundary_tensor = boundary_metadata.get('boundary_tensor') if isinstance(boundary_metadata, Mapping) else None
        if not isinstance(boundary_tensor, Mapping):
            raise RuntimeError('native_split_quality_boundary_metadata_tensor_missing')
        expected_boundary_name = str(boundary_tensor.get('runtime_name') or boundary_tensor.get('name') or '')
        expected_boundary_shape = [int(x) for x in list(boundary_tensor.get('shape') or [])]
        expected_boundary_dtype = str(boundary_tensor.get('dtype') or '').lower()
        dtype_bytes = {'uint8': 1, 'float32': 4}.get(expected_boundary_dtype, 0)
        expected_boundary_bytes = dtype_bytes
        for dim in expected_boundary_shape:
            expected_boundary_bytes *= dim
        if not expected_boundary_name or expected_boundary_bytes <= 0:
            raise RuntimeError('native_split_quality_boundary_runtime_identity_invalid')
    else:
        hef = _find_hef(bs, case, args.hw_arch)
        engine = _find_engine(bs, case, args.precision)
    image = Path(args.image).expanduser().resolve() if args.image else _default_image(bs)
    if image.is_dir():
        selected = _first_validation_image(image)
        if selected is None:
            raise FileNotFoundError(f'No image found below explicit directory: {image}')
        image = selected
    # Freeze the exact source image before a dual invocation forks its two
    # isolated phases.  Both children therefore receive one identical file,
    # not two independent directory/default resolutions.
    args.image = str(image)
    work = Path(args.work_dir).expanduser().resolve() if args.work_dir else bs / 'native_pipeline' / case / 'hailo_to_trt' / args.precision
    work.mkdir(parents=True, exist_ok=True)
    boundary_contract = _engine_boundary_contract(engine)
    (work / 'main.cpp').write_text(CPP_SOURCE, encoding='utf-8')
    (work / 'CMakeLists.txt').write_text(CMAKE_TXT, encoding='utf-8')
    cfg = {
        'benchmark_set': str(bs),
        'case': case,
        'hw_arch': args.hw_arch,
        'hef': str(hef),
        'engine': str(engine),
        'image': str(image),
        'precision': args.precision,
        'hailo_format': args.hailo_format,
        'task': args.task,
        'detection_endpoints': str(args.detection_endpoints),
        'raw_preprocess_scope': str(args.raw_preprocess_scope),
        'preprocess_mode_requested': args.preprocess_mode,
        'preprocess_mode_effective': args.preprocess_mode_effective,
        'frames': args.frames,
        'duration_s': float(args.duration_s or 0.0),
        'warmup': args.warmup,
        'repetitions': int(args.repetitions),
        'queue_depth': args.queue_depth,
        'copy_outputs': bool(args.copy_outputs),
        'work_dir': str(work),
        'dump_outputs': bool(args.dump_outputs),
        'dump_boundary': bool(args.dump_boundary),
        'letterbox_pad_value': int(args.letterbox_pad_value),
        'output_dir': str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else str(work / 'native_fifo_outputs'),
        'boundary_dir': str(Path(args.boundary_dir).expanduser().resolve()) if args.boundary_dir else str(work / 'native_fifo_boundary'),
        'native_command_contract_status': 'pending_native_executable',
    }
    config_json = Path(args.config_json).expanduser().resolve() if args.config_json else work / 'native_fifo_config.json'
    config_json.parent.mkdir(parents=True, exist_ok=True)
    config_json.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('[native-fifo] work_dir:', work)
    print('[native-fifo] hef:', hef)
    print('[native-fifo] engine:', engine)
    print('[native-fifo] image:', image)

    if _v2791_use_concurrent_three_stage(args):
        return _run_v2791_concurrent_three_stage(
            bs=bs, case=case, work=work, args=args,
        )

    if args.task == "detection" and args.detection_endpoints == "dual":
        return _run_detection_dual_endpoint(
            bs=bs, case=case, work=work, args=args,
        )

    python_detection = (
        args.task == "detection"
        and args.detection_endpoints == "completed_task"
    )
    if args.build and not python_detection:
        build_dir = work / 'build'
        build_dir.mkdir(exist_ok=True)
        cmd1 = ['cmake', '-S', str(work), '-B', str(build_dir), '-DCMAKE_BUILD_TYPE=Release']
        cmd2 = ['cmake', '--build', str(build_dir), '-j']
        print('[native-fifo] cmake:', ' '.join(cmd1))
        r1 = subprocess.run(cmd1, cwd=str(work))
        if r1.returncode != 0:
            return r1.returncode
        print('[native-fifo] build:', ' '.join(cmd2))
        r2 = subprocess.run(cmd2, cwd=str(work))
        if r2.returncode != 0:
            return r2.returncode

    exe = (
        Path(sys.executable).resolve()
        if python_detection
        else work / 'build' / 'split_native_hailo_trt_fifo'
    )
    if not exe.is_file():
        raise FileNotFoundError(f'Native FIFO executable not found: {exe}')
    replay_verification = _verify_replay_expectations(
        args, hef=hef, engine=engine, image=image, executable=exe,
        boundary=boundary_contract,
    )
    command_contract = _native_command_contract(
        bs=bs, case=case, args=args, hef=hef, engine=engine, image=image,
        work=work, executable=exe, quality_binding=quality_binding,
        producer_impl=(
            "hailo8_python_vstreams_fifo"
            if python_detection else "hailo8_cpp_vstreams_fifo"
        ),
    )
    cfg.update({
        'native_command_contract_status': 'pending_prepared_input_artifact',
        'native_command_contract': command_contract,
        'replay_verification': replay_verification,
    })
    config_json.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    if args.run:
        if python_detection:
            out_json = (
                Path(args.result_json).expanduser().resolve()
                if args.result_json else work / "native_fifo_results.json"
            )
            out_json.parent.mkdir(parents=True, exist_ok=True)
            payload, prepared_input_path, completion_contract = (
                _run_hailo8_python_detection(
                    bs=bs,
                    case=case,
                    hef=hef,
                    engine=engine,
                    image=image,
                    work=work,
                    args=args,
                    quality_binding=quality_binding,
                    expected_boundary_name=expected_boundary_name,
                    expected_boundary_shape=expected_boundary_shape,
                    expected_boundary_dtype=expected_boundary_dtype,
                )
            )
            command_contract = _native_command_contract(
                bs=bs,
                case=case,
                args=args,
                hef=hef,
                engine=engine,
                image=image,
                work=work,
                executable=exe,
                prepared_input=prepared_input_path,
                prepared_input_shape=[
                    int(x)
                    for x in list(
                        payload.get("prepared_input_shape") or []
                    )
                ],
                prepared_input_name=str(
                    payload.get("prepared_input_name") or ""
                ),
                prepared_input_dtype=str(
                    payload.get("prepared_input_dtype") or ""
                ),
                quality_binding=quality_binding,
                semantic_payload=payload,
                completion_execution_contract=completion_contract,
                producer_impl="hailo8_python_vstreams_fifo",
            )
            if command_contract.get("complete") is not True:
                raise RuntimeError(
                    "completed-detection command contract incomplete"
                )
            if quality_binding is not None and (
                payload.get(
                    "native_split_quality_runtime_boundary_verified"
                ) is not True
                or int(payload.get("hailo_runtime_output_count") or 0)
                != 1
                or str(payload.get("hailo_runtime_output_name") or "")
                != expected_boundary_name
                or int(
                    payload.get("hailo_runtime_output_frame_bytes") or 0
                ) != expected_boundary_bytes
            ):
                raise RuntimeError(
                    "native_split_quality_runtime_boundary_evidence_mismatch"
                )
            payload.update({
                "benchmark_set": str(bs),
                "case": case,
                "hw_arch": args.hw_arch,
                "precision": args.precision,
                "measurement_endpoint": "completed_task",
                "performance_endpoint": "completed_task",
                "primary_performance_endpoint": "completed_task",
                "application_performance_endpoint": "completed_task",
                "energy_performance_endpoint": "completed_task",
                "endpoint_energy_eligible": True,
                "input_image": str(image),
                "input_image_source": "exact_file",
                "input_image_sha256": _sha256_file(image),
                "native_command_contract": command_contract,
                "native_command_contract_sha256": str(
                    command_contract.get("contract_sha256") or ""
                ),
                "workload_contract_sha256": str(
                    command_contract.get("contract_sha256") or ""
                ),
                "replay_verification": replay_verification,
                "resolved_config_json": str(config_json),
                "completion_execution_contract": completion_contract,
            })
            if quality_binding is not None:
                payload.update({
                    "setup_id": str(args.setup_id),
                    "eval_run_id": str(args.eval_run_id),
                    "source_run_id": canonical_native_split_backend(
                        args.source_run_id, args.setup_id,
                    ),
                    "native_split_quality_binding": quality_binding,
                    "native_split_quality_binding_sha256": str(
                        quality_binding.get("binding_sha256") or ""
                    ),
                    "native_split_quality_eval_run_id": str(
                        quality_binding.get("eval_run_id") or ""
                    ),
                    "native_split_quality_source_run_id": (
                        canonical_native_split_backend(
                            quality_binding.get("source_run_id"),
                            args.setup_id,
                        )
                    ),
                    **_native_split_selection_evidence(
                        quality_binding
                    ),
                })
                joined, join_status, join_diagnostic = (
                    _bind_quality_for_execution(
                        native_row={
                        **payload,
                        "backend": "hailo8_to_trt",
                        "model_id": str(args.model_id),
                        "case_id": case,
                        "task": "detection",
                        "precision": str(args.precision),
                        "setup_id": str(args.setup_id),
                        "comparison_backend": "hailo8",
                        },
                        quality_binding=quality_binding,
                    )
                )
                if join_diagnostic:
                    payload.setdefault(
                        "cache_verify_diagnostics", []
                    ).append(join_diagnostic)
                payload[
                    "native_split_quality_consumer_attestation"
                ] = _seal_split_consumer_attestation(
                    binding=joined,
                    command=command_contract,
                    args=args,
                    case=case,
                    task="detection",
                )
                payload[
                    "native_split_quality_consumer_status"
                ] = join_status
            workload_sha = str(
                command_contract.get("contract_sha256") or ""
            )
            _bind_repetition_records_to_workload(payload, workload_sha)
            cfg.update({
                "native_command_contract_status": (
                    "complete_completed_detection"
                ),
                "native_command_contract": command_contract,
            })
            config_json.write_text(
                json.dumps(cfg, indent=2), encoding="utf-8"
            )
            payload.update(
                persist_detection_completion_execution_artifacts(
                    payload,
                    output_path=out_json.with_name(
                        f"{out_json.stem}.completed_task_result_artifact.json"
                    ),
                )
            )
            out_json.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            result_row = {
                "model_id": str(
                    args.model_id or _benchmark_model_id(bs)
                ),
                "case_id": case,
                "backend": f"{args.hw_arch}_to_tensorrt",
                "run_id": f"{args.hw_arch}_to_trt_native_fifo",
                "variant": "composed",
                "primary_variant": "composed",
                "stage1_provider": args.hw_arch,
                "stage2_provider": "tensorrt",
                "native_fifo_enabled": True,
                "native_fifo_mode": payload.get("mode"),
                "native_fifo_precision": args.precision,
                "performance_endpoint": "completed_task",
                "primary_performance_endpoint": "completed_task",
                "application_performance_endpoint": "completed_task",
                "energy_performance_endpoint": "completed_task",
                "native_fifo_fps_makespan": payload.get(
                    "fps_makespan"
                ),
                "throughput_primary_fps": payload.get(
                    "fps_makespan"
                ),
                "throughput_primary_metric": (
                    "completed_detection_fifo_makespan_fps"
                ),
                "runtime_ok": True,
                "validation_ok": None,
                "native_fifo_results_json": str(out_json),
                "native_command_contract": command_contract,
                "native_command_contract_sha256": workload_sha,
                "workload_contract_sha256": workload_sha,
                "completion_execution_attestation": payload.get(
                    "completion_execution_attestation"
                ),
                "completed_task_result_artifact_saved": payload.get(
                    "completed_task_result_artifact_saved"
                ),
                "completed_task_result_artifact": payload.get(
                    "completed_task_result_artifact"
                ),
                "completed_task_result_artifact_sha256": payload.get(
                    "completed_task_result_artifact_sha256"
                ),
                "completed_task_result_artifact_path": payload.get(
                    "completed_task_result_artifact_path"
                ),
                "completed_task_result_artifact_file_sha256": payload.get(
                    "completed_task_result_artifact_file_sha256"
                ),
                "completed_task_endpoint_contract": payload.get(
                    "completed_task_endpoint_contract"
                ),
                "completed_task_comparison_endpoint_contract": (
                    payload.get(
                        "completed_task_comparison_endpoint_contract"
                    )
                ),
                "completed_work_units": payload.get(
                    "completed_work_units"
                ),
                "postprocess_included": True,
                "postprocess_completion_verified": True,
                "native_split_quality_consumer_attestation": (
                    payload.get(
                        "native_split_quality_consumer_attestation"
                    )
                ),
                "native_split_quality_consumer_status": payload.get(
                    "native_split_quality_consumer_status"
                ),
            }
            benchmark_results = (
                bs / f"benchmark_results_native_fifo_{case}.json"
            )
            benchmark_results.write_text(json.dumps({
                "schema": (
                    "onnx-splitpoint/native-fifo-benchmark-results"
                ),
                "schema_version": 1,
                "results": [result_row],
            }, indent=2), encoding="utf-8")
            print("[native-fifo] results:", out_json)
            print(
                "[native-fifo] benchmark results row:",
                benchmark_results,
            )
            return 0
        prepared_input_path = work / 'contract_artifacts' / 'hailo8_prepared_input_rgb_uint8.bin'
        out_json = Path(args.result_json).expanduser().resolve() if args.result_json else work / 'native_fifo_results.json'
        out_json.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(exe),
            '--hef', str(hef),
            '--engine', str(engine),
            '--image', str(image),
            '--out', str(out_json),
            '--frames', str(args.frames),
            '--warmup', str(args.warmup),
            '--queue-depth', str(args.queue_depth),
            '--hailo-format', str(args.hailo_format),
            '--task', str(args.task),
            '--preprocess-mode', str(args.preprocess_mode_effective),
            '--copy-outputs', '1' if args.copy_outputs else '0',
            '--dump-outputs', '1' if args.dump_outputs else '0',
            '--dump-boundary', '1' if args.dump_boundary else '0',
            '--letterbox-pad-value', str(int(args.letterbox_pad_value)),
            '--output-dir', str(Path(args.output_dir).expanduser().resolve() if args.output_dir else work / 'native_fifo_outputs'),
            '--boundary-dir', str(Path(args.boundary_dir).expanduser().resolve() if args.boundary_dir else work / 'native_fifo_boundary'),
            '--prepared-input-out', str(prepared_input_path),
            '--reuse-preprocessed-input', (
                '1'
                if args.task != 'detection'
                or args.raw_preprocess_scope == 'shared_prepared_input'
                else '0'
            ),
        ]
        if args.prepared_input_rgb:
            shared_path = Path(args.prepared_input_rgb).expanduser().resolve()
            shared_sha = _sha256_file(shared_path)
            if not shared_sha or (args.expected_prepared_input_sha256 and shared_sha != args.expected_prepared_input_sha256):
                raise RuntimeError("shared_prepared_rgb_sha256_mismatch")
            cmd += ['--prepared-input-rgb', str(shared_path)]
        if quality_binding is not None:
            cmd += [
                '--expected-output-count', '1',
                '--expected-output-name', expected_boundary_name,
                '--expected-output-bytes', str(expected_boundary_bytes),
            ]
        if args.duration_s and float(args.duration_s) > 0:
            cmd += ['--duration-s', str(float(args.duration_s))]
        if args.device_id:
            device_id = str(args.device_id).strip()
            if device_id.startswith('pci/'):
                device_id = device_id[4:]
            cmd += ['--device-id', device_id]
        repetition_payloads: list[dict[str, Any]] = []
        repetition_dir = work / 'repetitions'
        if int(args.repetitions) > 1:
            repetition_dir.mkdir(parents=True, exist_ok=True)
        r: subprocess.CompletedProcess[Any] | None = None
        failed_result_path: Path | None = None
        raw_feed_sha256 = ""
        raw_feed_error = ""
        for repetition_index in range(int(args.repetitions)):
            is_last = repetition_index + 1 == int(args.repetitions)
            repetition_out = (
                out_json if int(args.repetitions) == 1
                else repetition_dir / f'repetition_{repetition_index + 1:03d}.json'
            )
            run_cmd = list(cmd)
            if repetition_index > 0 and args.raw_preprocess_scope == 'shared_prepared_input':
                try:
                    _pin_raw_prepared_feed(prepared_input_path, raw_feed_sha256)
                except RuntimeError as exc:
                    failed_result_path = repetition_out
                    raw_feed_error = str(exc)
                    break
                # The first raw repetition materialized this exact feed.  Later
                # fresh C++ processes load it, without decoding/resizing again.
                if '--prepared-input-rgb' not in run_cmd:
                    run_cmd += ['--prepared-input-rgb', str(prepared_input_path)]
            run_cmd[run_cmd.index('--out') + 1] = str(repetition_out)
            # The semantic dump is one separate bound inference after the final
            # repetition only.  It never contaminates a measured interval.
            run_cmd[run_cmd.index('--dump-outputs') + 1] = '1' if (args.dump_outputs and is_last) else '0'
            run_cmd[run_cmd.index('--dump-boundary') + 1] = '1' if (args.dump_boundary and is_last) else '0'
            print(f'[native-fifo] repetition {repetition_index + 1}/{args.repetitions}:', ' '.join(run_cmd))
            r = subprocess.run(run_cmd, cwd=str(work))
            if r.returncode != 0 or not repetition_out.is_file():
                failed_result_path = repetition_out
                break
            row = _load_json(repetition_out)
            if not isinstance(row, dict) or not row.get('ok'):
                failed_result_path = repetition_out
                break
            if args.raw_preprocess_scope == 'shared_prepared_input':
                try:
                    raw_feed_sha256 = _pin_raw_prepared_feed(prepared_input_path, raw_feed_sha256)
                except RuntimeError as exc:
                    failed_result_path = repetition_out
                    raw_feed_error = str(exc)
                    break
                row['prepared_input_sha256'] = raw_feed_sha256
            # The workflow may launch this runner more than once with
            # ``--repetitions 1``.  The old index+path hash then repeated even
            # though a genuinely fresh process performed the measurement.
            repetition_id, runtime_instance_id = _fresh_hailo_repetition_identity(
                repetition_index + 1, repetition_out,
            )
            row['runtime_instance_id'] = runtime_instance_id
            row['repetition_id'] = repetition_id
            row['process_local_repetition_index'] = repetition_index + 1
            row['repetition_runtime_scope'] = 'fresh_process_per_repetition'
            repetition_payloads.append(row)
        if len(repetition_payloads) == int(args.repetitions):
            payload = _aggregate_repetition_payloads(repetition_payloads)
            out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print('[native-fifo] results:', out_json)
        if len(repetition_payloads) != int(args.repetitions):
            err = {
                'ok': False,
                'mode': 'native_hailort_tensorrt_fifo',
                'returncode': 8 if raw_feed_error else int(r.returncode if r is not None else 1),
                'error': raw_feed_error or 'native FIFO repetition failed before a complete aggregate could be written',
                'hint': 'Check the terminal output above. For Hailo auto-discovery, v59aq uses VDevice::create() when --device-id is empty.',
                'hef': str(hef),
                'engine': str(engine),
                'case': case,
                'hw_arch': args.hw_arch,
                'precision': args.precision,
                'repetitions_requested': int(args.repetitions),
                'repetitions_completed': len(repetition_payloads),
                'repetition_count_requested': int(args.repetitions),
                'repetition_count_attempted': min(int(args.repetitions), len(repetition_payloads) + 1),
                'repetition_count_valid': len(repetition_payloads),
                'repetition_status': 'failed',
                'failed_result_path': str(failed_result_path or ''),
                'repetition_evidence': [dict(row, repetition_index=i + 1) for i, row in enumerate(repetition_payloads)],
                'repetition_records': [dict(row, repetition_index=i + 1) for i, row in enumerate(repetition_payloads)],
                'native_command_contract': command_contract,
                'replay_verification': replay_verification,
            }
            out_json.write_text(json.dumps(err, indent=2), encoding='utf-8')
            if raw_feed_error:
                return 8
        if out_json.is_file():
            try:
                payload = _load_json(out_json)
                if isinstance(payload, dict):
                    command_contract = _native_command_contract(
                        bs=bs, case=case, args=args, hef=hef, engine=engine,
                        image=image, work=work, executable=exe,
                        prepared_input=prepared_input_path,
                        prepared_input_shape=[int(x) for x in list(payload.get('prepared_input_shape') or [])],
                        quality_binding=quality_binding,
                        semantic_payload=payload,
                    )
                    if command_contract.get('complete') is not True:
                        raise RuntimeError('native command contract incomplete after prepared input generation')
                    if quality_binding is not None and (
                        payload.get('native_split_quality_runtime_boundary_verified') is not True
                        or int(payload.get('hailo_runtime_output_count') or 0) != 1
                        or str(payload.get('hailo_runtime_output_name') or '') != expected_boundary_name
                        or int(payload.get('hailo_runtime_output_frame_bytes') or 0) != expected_boundary_bytes
                    ):
                        raise RuntimeError('native_split_quality_runtime_boundary_evidence_mismatch')
                    cfg.update({
                        'native_command_contract_status': 'complete_after_prepared_input',
                        'native_command_contract': command_contract,
                    })
                    config_json.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
                    payload.update({
                        'benchmark_set': str(bs),
                        'case': case,
                        'hw_arch': args.hw_arch,
                        'precision': args.precision,
                        'measurement_endpoint': (
                            'raw_model_outputs'
                            if args.task == 'detection'
                            else 'model_outputs'
                        ),
                        'performance_endpoint': (
                            'raw_model_outputs'
                            if args.task == 'detection'
                            else 'model_outputs'
                        ),
                        'primary_performance_endpoint': (
                            'raw_model_outputs'
                            if args.task == 'detection'
                            else 'model_outputs'
                        ),
                        'application_performance_endpoint': (
                            'completed_task'
                            if args.task == 'detection'
                            else 'model_outputs'
                        ),
                        'energy_performance_endpoint': (
                            'completed_task'
                            if args.task == 'detection'
                            else 'model_outputs'
                        ),
                        'endpoint_energy_eligible': args.task != 'detection',
                        'input_image': str(image),
                        'input_image_source': 'exact_file',
                        'input_image_sha256': _sha256_file(image),
                        'native_command_contract': command_contract,
                        'replay_verification': replay_verification,
                        'resolved_config_json': str(config_json),
                        'native_command_contract_sha256': str(command_contract.get('contract_sha256') or ''),
                    })
                    if quality_binding is not None:
                        payload.update({
                            'setup_id': str(args.setup_id),
                            'eval_run_id': str(args.eval_run_id),
                            'source_run_id': canonical_native_split_backend(
                                args.source_run_id, args.setup_id,
                            ),
                            'native_split_quality_binding': quality_binding,
                            'native_split_quality_binding_sha256': str(quality_binding.get('binding_sha256') or ''),
                            'native_split_quality_eval_run_id': str(quality_binding.get('eval_run_id') or ''),
                            'native_split_quality_source_run_id': canonical_native_split_backend(
                                quality_binding.get('source_run_id'), args.setup_id,
                            ),
                            **_native_split_selection_evidence(quality_binding),
                        })
                        joined, join_status, join_diagnostic = (
                            _bind_quality_for_execution(
                                native_row={
                                **payload,
                                'backend': 'hailo8_to_trt',
                                'model_id': str(args.model_id),
                                'case_id': case,
                                'task': str(args.task),
                                'precision': str(args.precision),
                                'setup_id': str(args.setup_id),
                                'comparison_backend': 'hailo8',
                                },
                                quality_binding=quality_binding,
                            )
                        )
                        if join_diagnostic:
                            payload.setdefault(
                                'cache_verify_diagnostics', []
                            ).append(join_diagnostic)
                        payload['native_split_quality_consumer_attestation'] = _seal_split_consumer_attestation(
                            binding=joined, command=command_contract, args=args,
                            case=case, task=args.task,
                        )
                        payload['native_split_quality_consumer_status'] = join_status
                    workload_contract_sha256 = str(command_contract.get('contract_sha256') or '')
                    payload['workload_contract_sha256'] = workload_contract_sha256
                    _bind_repetition_records_to_workload(
                        payload, workload_contract_sha256,
                    )
                    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                print(json.dumps(payload, indent=2)[:4000])
                try:
                    # Annotate both manifests with the exact image used by the
                    # separate post-measurement dump inference.  Its sequence is
                    # bound to the historical last-measured image index, so old
                    # directory contracts remain reproducible without copying
                    # dump payloads inside the performance loop.
                    mp = payload.get('native_fifo_output_manifest') if isinstance(payload, dict) else None
                    if mp and Path(str(mp)).is_file():
                        _seal_manifest_payload_files(mp)
                        mj = json.loads(Path(str(mp)).read_text(encoding='utf-8'))
                        seq = payload.get('frames') or payload.get('measured_frames')
                        # prefer boundary manifest seq if present below; otherwise fall back to last frame.
                        bman = payload.get('native_fifo_boundary_manifest')
                        if bman and Path(str(bman)).is_file():
                            try:
                                bjs = json.loads(Path(str(bman)).read_text(encoding='utf-8'))
                                seq = bjs.get('seq', seq)
                            except Exception:
                                pass
                        img = _resolve_image_for_seq(image, int(seq) if seq is not None else None)
                        mj['input_image'] = img
                        mj['provenance'] = {'image': img, 'image_source': 'exact_file' if image.is_file() else 'directory_seq_mod', 'seq': seq}
                        Path(str(mp)).write_text(json.dumps(mj, indent=2), encoding='utf-8')
                        _annotate_output_contract(str(mp), bs)
                        endpoint_manifest = _load_json(Path(str(mp)))
                        if isinstance(endpoint_manifest, Mapping):
                            payload.update({
                                'task': str(endpoint_manifest.get('task') or ''),
                                'output_format': str(endpoint_manifest.get('output_format') or ''),
                                'contract_family': str(endpoint_manifest.get('contract_family') or ''),
                                'stage': str(endpoint_manifest.get('stage') or ''),
                                'contract_source': str(endpoint_manifest.get('contract_source') or ''),
                                'endpoint_contract_complete': endpoint_manifest.get('endpoint_contract_complete') is True,
                                'endpoint_contract_hash': str(endpoint_manifest.get('endpoint_contract_hash') or ''),
                                'tensor_signature': endpoint_manifest.get('tensor_signature') if isinstance(endpoint_manifest.get('tensor_signature'), Mapping) else {},
                                'output_endpoint_attestation': endpoint_manifest.get('output_endpoint_attestation') if isinstance(endpoint_manifest.get('output_endpoint_attestation'), Mapping) else {},
                            })
                            out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                    if isinstance(payload, dict):
                        _annotate_boundary_manifest(payload.get('native_fifo_boundary_manifest'), image, args.precision)
                        if payload.get('native_fifo_boundary_manifest'):
                            _seal_manifest_payload_files(payload.get('native_fifo_boundary_manifest'))
                except Exception:
                    pass
                # Manifests are post-processed above, so seal their final bytes
                # only now.  No semantic artifact may change after this point.
                if isinstance(payload, dict) and payload.get('ok'):
                    command_contract = _native_command_contract(
                        bs=bs, case=case, args=args, hef=hef, engine=engine,
                        image=image, work=work, executable=exe,
                        prepared_input=prepared_input_path,
                        prepared_input_shape=[int(x) for x in list(payload.get('prepared_input_shape') or [])],
                        quality_binding=quality_binding,
                        semantic_payload=payload,
                    )
                    payload['native_command_contract'] = command_contract
                    payload['native_command_contract_sha256'] = str(command_contract.get('contract_sha256') or '')
                    payload['workload_contract_sha256'] = str(command_contract.get('contract_sha256') or '')
                    # The semantic manifests above can change the final sealed
                    # command hash.  Re-bind every measured repetition to this
                    # final command, not to the pre-dump intermediate contract.
                    _bind_repetition_records_to_workload(
                        payload, payload['workload_contract_sha256'],
                    )
                    if quality_binding is not None:
                        joined, join_status, join_diagnostic = (
                            _bind_quality_for_execution(
                                native_row={
                                **payload,
                                'backend': 'hailo8_to_trt',
                                'model_id': str(args.model_id),
                                'case_id': case,
                                'task': str(args.task),
                                'precision': str(args.precision),
                                'setup_id': str(args.setup_id),
                                'comparison_backend': 'hailo8',
                                },
                                quality_binding=quality_binding,
                            )
                        )
                        if join_diagnostic:
                            payload.setdefault(
                                'cache_verify_diagnostics', []
                            ).append(join_diagnostic)
                        payload['native_split_quality_consumer_attestation'] = _seal_split_consumer_attestation(
                            binding=joined, command=command_contract, args=args,
                            case=case, task=args.task,
                        )
                        payload['native_split_quality_consumer_status'] = join_status
                    cfg.update({
                        'native_command_contract_status': 'complete_after_semantic_dump',
                        'native_command_contract': command_contract,
                    })
                    config_json.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
                    out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                if isinstance(payload, dict) and payload.get('ok'):
                    model_id = str(args.model_id or bs.name)
                    try:
                        bjs = _load_json(bs / 'benchmark_set.json')
                        model_id = str(bjs.get('model_name') or bjs.get('model_id') or model_id)
                    except Exception:
                        pass
                    row = {
                        'model_id': model_id,
                        'case_id': case,
                        'backend': f'{args.hw_arch}_to_tensorrt',
                        'run_id': f'{args.hw_arch}_to_trt_native_fifo',
                        'variant': 'composed',
                        'primary_variant': 'composed',
                        'stage1_provider': args.hw_arch,
                        'stage2_provider': 'tensorrt',
                        'native_fifo_enabled': True,
                        'native_fifo_mode': payload.get('mode'),
                        'native_fifo_precision': args.precision,
                        'native_fifo_boundary_mode': 'raw_uint8_hailo' if args.precision == 'uint8_cast_fp16' else ('hailort_float32_layout' if args.precision == 'float32_layout_fp16' else args.hailo_format),
                        'native_fifo_preprocess_ms': payload.get('preprocess_ms'),
                        'native_fifo_p1_ms': payload.get('p1_ms'),
                        'native_fifo_handoff_ms': payload.get('handoff_ms'),
                        'native_fifo_p2_run_ms': payload.get('p2_run_ms'),
                        'native_fifo_p1_thread_ms': payload.get('p1_thread_ms'),
                        'native_fifo_p2_thread_ms': payload.get('p2_thread_ms'),
                        'native_fifo_paper_equivalent_cycle_ms': payload.get('paper_equivalent_cycle_ms'),
                        'native_fifo_paper_equivalent_fps': payload.get('paper_equivalent_fps'),
                        'native_fifo_fps_makespan': payload.get('fps_makespan'),
                        'part1_latency_ms': payload.get('p1_ms'),
                        'part2_latency_ms': payload.get('p2_run_ms'),
                        'transfer_latency_ms': payload.get('handoff_ms'),
                        'preprocess_latency_ms': payload.get('preprocess_ms'),
                        'split_latency_e2e_ms': payload.get('single_latency_model_ms'),
                        'composed_latency_ms': payload.get('single_latency_model_ms'),
                        'pipeline_cycle_selected_ms': payload.get('paper_equivalent_cycle_ms'),
                        'pipeline_fps_selected': payload.get('fps_makespan') or payload.get('paper_equivalent_fps'),
                        'throughput_primary_fps': payload.get('fps_makespan') or payload.get('paper_equivalent_fps'),
                        'performance_endpoint': (
                            'raw_model_outputs'
                            if args.task == 'detection'
                            else 'model_outputs'
                        ),
                        'primary_performance_endpoint': (
                            'raw_model_outputs'
                            if args.task == 'detection'
                            else 'model_outputs'
                        ),
                        'energy_performance_endpoint': (
                            'completed_task'
                            if args.task == 'detection'
                            else 'model_outputs'
                        ),
                        'throughput_primary_metric': (
                            'raw_model_outputs_fifo_makespan_fps'
                            if args.task == 'detection'
                            else 'native_fifo_makespan_fps'
                        ),
                        'postprocess_included': False,
                        'postprocess_completion_verified': False,
                        'trt_input_dtype': payload.get('trt_input_dtype'),
                        'trt_input_bytes': payload.get('trt_input_bytes'),
                        'runtime_ok': True,
                        'validation_ok': None,
                        'native_fifo_results_json': str(out_json),
                        'native_fifo_output_manifest': payload.get('native_fifo_output_manifest'),
                        'native_fifo_boundary_manifest': payload.get('native_fifo_boundary_manifest'),
                        'setup_id': str(args.setup_id),
                        'eval_run_id': str(args.eval_run_id),
                        'source_run_id': canonical_native_split_backend(
                            args.source_run_id, args.setup_id,
                        ),
                        'native_command_contract': payload.get('native_command_contract'),
                        'native_command_contract_sha256': payload.get('native_command_contract_sha256'),
                        'workload_contract_sha256': payload.get('workload_contract_sha256'),
                        'native_split_quality_binding': payload.get('native_split_quality_binding'),
                        'native_split_quality_binding_sha256': payload.get('native_split_quality_binding_sha256'),
                        'native_split_quality_eval_run_id': payload.get('native_split_quality_eval_run_id'),
                        'native_split_quality_source_run_id': payload.get('native_split_quality_source_run_id'),
                        'source_request_sha256': payload.get('source_request_sha256'),
                        'native_split_quality_source_request_sha256': payload.get('native_split_quality_source_request_sha256'),
                        'native_split_quality_central_result_sha256': payload.get('native_split_quality_central_result_sha256'),
                        'native_split_quality_selection_sha256': payload.get('native_split_quality_selection_sha256'),
                        'native_split_quality_consumer_attestation': payload.get('native_split_quality_consumer_attestation'),
                        'native_split_quality_consumer_status': payload.get('native_split_quality_consumer_status'),
                    }
                    br = bs / f'benchmark_results_native_fifo_{case}.json'
                    br.write_text(json.dumps({'schema': 'onnx-splitpoint/native-fifo-benchmark-results', 'schema_version': 1, 'results': [row]}, indent=2), encoding='utf-8')
                    print('[native-fifo] benchmark results row:', br)
            except Exception as exc:
                if quality_binding is not None:
                    raise
        return int(r.returncode if r is not None else 1)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
