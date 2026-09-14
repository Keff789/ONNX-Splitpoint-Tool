#!/usr/bin/env python3
"""Verify the semantic terminal result of the compiler-free cache canary.

The canary is an operational diagnostic, not a scientific performance claim.
Its PASS decision therefore comes from one run-local producer result and the
semantic artifact/runtime identity that was actually exercised.  Historical
hashes, receipts, command seals and recursively projected collector rows remain
useful diagnostics, but are deliberately not terminal success conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping


EXPECTED_IDENTITY = (
    "native_split",
    "hailo8_to_trt",
    "resnet50",
    "b052",
    "orin_nx_hailo8_01",
    "hailo8",
)
EXPECTED_VARIANT = "v27510_resnet50_b052_hailo8_smoke_cache_verify"
EXPECTED_VARIANT_NAMESPACE = "v000_" + EXPECTED_VARIANT

_VARIANT_ROOT = (
    Path("native_producers/variants")
    / EXPECTED_VARIANT_NAMESPACE
    / "hailo8"
)
_BINDING_SET_REL = (
    _VARIANT_ROOT
    / "cache_verify_native_split/orin_nx_hailo8_01"
    / "native_split_quality_binding_set.json"
)
_RUNNER_REL = _VARIANT_ROOT / "analysis_tables/native_fifo_eval_runner.json"
_RESULT_REL = (
    _VARIANT_ROOT
    / "resnet50/benchmark_set/native_pipeline/b052/hailo_to_trt"
    / "float32_layout_fp16/native_fifo_results.json"
)
_SUMMARY_REL = Path("reports/native_producer_summary.json")

_EXPECTED_PRESELECTION: dict[str, Any] = {
    "model_id": "resnet50",
    "model_family": "resnet",
    "case_id": "b052",
    "setup_id": "orin_nx_hailo8_01",
    "backend": "hailo8_to_trt",
    "stage2_backend": "native_tensorrt",
    "task": "classification",
    "precision": "float32_layout_fp16",
    "hailo_format": "float32",
    "boundary_dtype": "float32",
    "boundary_layout": "memory_nhwc_to_nchw",
    "boundary_transform": "layout_only",
    "quantization_policy": "none",
    "preprocess_mode": "resize",
    "letterbox_pad_value": 0,
    "boundary_tensor_name": "add_6",
    "boundary_tensor_shape": [28, 28, 512],
    "boundary_tensor_dtype": "float32",
    "canonical_part2_shape": [1, 512, 28, 28],
}
_SEMANTIC_ARTIFACT_ROLES = {
    "part1_runtime",
    "boundary_metadata",
    "source_part2_onnx",
    "build_part2_onnx",
    "engine",
    "native_trt_meta",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def _read_object(path: Path, errors: list[str], *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except FileNotFoundError:
        errors.append(f"{label}_missing")
        return {}
    except Exception as exc:
        errors.append(
            f"{label}_invalid_json:{type(exc).__name__}:{exc}"
        )
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}_not_an_object")
        return {}
    return value


def _read_diagnostic_object(
    run_dir: Path, relative: Path,
) -> tuple[dict[str, Any], str]:
    """Read optional diagnostics without following links outside the run."""

    if relative.is_absolute() or ".." in relative.parts:
        return {}, "unsafe_path"
    cursor = run_dir
    try:
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("symlink_component")
        path = run_dir / relative
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(run_dir):
            raise ValueError("outside_run_dir")
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError("not_regular_file")
    except FileNotFoundError:
        return {}, "missing"
    except Exception as exc:
        return {}, f"unsafe_path:{type(exc).__name__}:{exc}"
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except Exception as exc:
        return {}, f"invalid_json:{type(exc).__name__}:{exc}"
    return (value, "ok") if isinstance(value, dict) else ({}, "not_an_object")


def _required_regular_path(
    run_dir: Path,
    relative: Path,
    *,
    label: str,
    errors: list[str],
) -> Path | None:
    """Resolve one fixed run-local file without following symlink components."""

    if relative.is_absolute() or ".." in relative.parts:
        errors.append(f"{label}_path_not_run_local")
        return None
    lexical = run_dir.joinpath(relative).absolute()
    cursor = run_dir
    try:
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("symlink_component")
        resolved = lexical.resolve(strict=True)
        if not resolved.is_relative_to(run_dir):
            raise ValueError("outside_run_dir")
        mode = resolved.stat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError("not_regular_file")
    except Exception as exc:
        errors.append(f"{label}_path_invalid:{type(exc).__name__}:{exc}")
        return None
    return resolved


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _zero_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and parsed == 0.0


def _fresh_process_runtime_id(value: Any) -> str:
    token = str(value or "").strip().lower()
    prefix = "fresh_process:"
    suffix = token[len(prefix):] if token.startswith(prefix) else ""
    if len(suffix) != 64 or any(
        character not in "0123456789abcdef" for character in suffix
    ):
        return ""
    return token


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_mismatches(
    value: Mapping[str, Any], expected: Mapping[str, Any], *, prefix: str,
) -> list[str]:
    def differs(observed: Any, wanted: Any) -> bool:
        if (
            isinstance(wanted, (int, float))
            and not isinstance(wanted, bool)
            and isinstance(observed, bool)
        ):
            return True
        return observed != wanted

    return [
        f"{prefix}.{field}:{value.get(field)!r}!={wanted!r}"
        for field, wanted in expected.items()
        if differs(value.get(field), wanted)
    ]


def _verify_manifest(
    manifest: Mapping[str, Any], *, run_dir: Path, errors: list[str],
) -> dict[str, Any]:
    if (
        manifest.get("schema")
        != "onnx-splitpoint/evaluation-run-manifest"
        or manifest.get("schema_version") != 1
        or manifest.get("run_id") != run_dir.name
    ):
        errors.append("manifest_identity_invalid")

    snapshot = manifest.get("profile_start_snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    attestation = snapshot.get("cache_verify_attestation")
    attestation = attestation if isinstance(attestation, Mapping) else {}
    if (
        attestation.get("status") != "verified"
        or attestation.get("mode") != "cache_verify_only"
        or attestation.get("compiler_dispatch_allowed") is not False
    ):
        errors.append("manifest_cache_verify_or_compiler_fence_invalid")

    actual = attestation.get("actual_plan")
    actual = actual if isinstance(actual, Mapping) else {}
    no_build_expected = {
        "hailo_build_mode": "cache_verify_only",
        "hailo_force_build": False,
        "deepx_build_mode": "cache_verify_only",
        "deepx_force_build": False,
        "native_build_missing_engines": False,
        "native_force_rebuild_variants": [],
        "deepx_prefetch_enabled": False,
        "generic_runtime_enabled": False,
    }
    errors.extend(_semantic_mismatches(
        actual, no_build_expected, prefix="manifest.actual_plan",
    ))
    execution_expected = {
        "models": ["resnet50"],
        "native_backends": ["hailo8"],
        "native_full_backends": [],
        "native_energy_enabled": False,
        "native_frames": 10,
        "native_warmup": 2,
        "native_repetitions": 1,
    }
    errors.extend(_semantic_mismatches(
        actual, execution_expected, prefix="manifest.actual_plan",
    ))
    native_rows = actual.get("native_rows")
    if (
        not isinstance(native_rows, list)
        or len(native_rows) != 1
        or not isinstance(native_rows[0], Mapping)
    ):
        errors.append("manifest_native_row_not_exactly_one")
    else:
        errors.extend(_semantic_mismatches(
            native_rows[0],
            {
                "model_id": "resnet50",
                "case_id": "b052",
                "backend": "hailo8",
                "setup_id": "orin_nx_hailo8_01",
                "variant_id": EXPECTED_VARIANT,
            },
            prefix="manifest.native_row",
        ))
    return dict(attestation)


def _semantic_binding_projection(
    binding: Mapping[str, Any],
    *,
    run_dir: Path,
    label: str,
    errors: list[str],
) -> dict[str, Any]:
    if (
        binding.get("schema")
        != "onnx-splitpoint/native-split-quality-binding"
        or binding.get("schema_version") != 1
        or binding.get("eval_run_id") != run_dir.name
        or binding.get("source_run_id") != "hailo8_to_trt"
        or binding.get("quality_completed") is not True
    ):
        errors.append(f"{label}_envelope_or_current_run_invalid")

    preselection = binding.get("preselection")
    preselection = preselection if isinstance(preselection, Mapping) else {}
    errors.extend(_semantic_mismatches(
        preselection, _EXPECTED_PRESELECTION,
        prefix=f"{label}.preselection",
    ))

    boundary = binding.get("boundary_contract")
    boundary = boundary if isinstance(boundary, Mapping) else {}
    errors.extend(_semantic_mismatches(
        boundary,
        {
            "precision": "float32_layout_fp16",
            "boundary_layout": "memory_nhwc_to_nchw",
            "boundary_transform": "layout_only",
            "boundary_tensor_name": "add_6",
            "boundary_tensor_dtype": "float32",
            "boundary_tensor_shape": [28, 28, 512],
            "dequant_scale": None,
            "dequant_zero_point": None,
        },
        prefix=f"{label}.boundary_contract",
    ))

    metadata = binding.get("boundary_metadata_payload")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    tensor = metadata.get("boundary_tensor")
    tensor = tensor if isinstance(tensor, Mapping) else {}
    errors.extend(_semantic_mismatches(
        metadata,
        {
            "model_id": "resnet50",
            "case_id": "b052",
            "setup_id": "orin_nx_hailo8_01",
            "backend": "hailo8_to_trt",
            "source_run_id": "hailo8_to_trt",
            "boundary_layout": "memory_nhwc_to_nchw",
            "boundary_transform": "layout_only",
            "boundary_tensor_count": 1,
        },
        prefix=f"{label}.boundary_metadata",
    ))
    errors.extend(_semantic_mismatches(
        tensor,
        {
            "name": "add_6",
            "runtime_name": "resnet50_part1_b52/conv24",
            "shape": [28, 28, 512],
            "dtype": "float32",
            "canonical_part2_shape": [1, 512, 28, 28],
        },
        prefix=f"{label}.boundary_tensor",
    ))

    trt_meta = binding.get("native_trt_meta_payload")
    trt_meta = trt_meta if isinstance(trt_meta, Mapping) else {}
    inputs = trt_meta.get("inputs")
    outputs = trt_meta.get("outputs")
    semantic_trt_ok = bool(
        trt_meta.get("schema") == "onnx-splitpoint/native-trt-meta"
        and trt_meta.get("schema_version") == 1
        and trt_meta.get("case") == "b052"
        and trt_meta.get("precision") == "float32_layout_fp16"
        and isinstance(inputs, list)
        and len(inputs) == 1
        and isinstance(inputs[0], Mapping)
        and inputs[0].get("name") == "add_6"
        and inputs[0].get("shape") == [1, 512, 28, 28]
        and inputs[0].get("elem_type") == "FLOAT"
        and isinstance(outputs, list)
        and len(outputs) == 1
        and isinstance(outputs[0], Mapping)
        and outputs[0].get("name") == "logits"
        and outputs[0].get("shape") == [1, 1000]
        and outputs[0].get("elem_type") == "FLOAT"
    )
    if not semantic_trt_ok:
        errors.append(f"{label}_tensorrt_semantic_signature_invalid")

    artifacts = binding.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    if not _SEMANTIC_ARTIFACT_ROLES <= set(artifacts):
        errors.append(f"{label}_semantic_artifact_roles_missing")
    else:
        for role in sorted(_SEMANTIC_ARTIFACT_ROLES):
            row = artifacts.get(role)
            if not isinstance(row, Mapping) or not str(
                row.get("path") or ""
            ).strip():
                errors.append(f"{label}_semantic_artifact_path_missing:{role}")

    replay = binding.get("cache_verify_replay")
    replay = replay if isinstance(replay, Mapping) else {}
    if (
        replay.get("artifact_policy") != "cache_verify_only"
        or replay.get("compiler_dispatched") is not False
    ):
        errors.append(f"{label}_cache_replay_or_compiler_fence_invalid")

    return {
        field: preselection.get(field)
        for field in _EXPECTED_PRESELECTION
    }


def _verify_binding_set(
    payload: Mapping[str, Any], *, run_dir: Path, errors: list[str],
) -> dict[str, Any]:
    if (
        payload.get("schema")
        != "onnx-splitpoint/native-split-quality-binding-set"
        or payload.get("schema_version") != 2
        or payload.get("mode") != "cache_verify_only"
        or payload.get("eval_run_id") != run_dir.name
        or payload.get("setup_id") != "orin_nx_hailo8_01"
    ):
        errors.append("cache_binding_set_semantic_envelope_invalid")

    attestation = payload.get("cache_verify_attestation")
    attestation = attestation if isinstance(attestation, Mapping) else {}
    if (
        attestation.get("status") != "verified"
        or attestation.get("artifact_policy") != "cache_verify_only"
        or attestation.get("compiler_dispatch_allowed") is not False
        or attestation.get("compiler_dispatched") is not False
        or attestation.get("eval_run_id") != run_dir.name
        or attestation.get("model_id") != "resnet50"
        or attestation.get("case_id") != "b052"
        or attestation.get("setup_id") != "orin_nx_hailo8_01"
        or attestation.get("backend") != "hailo8_to_trt"
    ):
        errors.append("cache_binding_attestation_or_compiler_fence_invalid")

    bindings = payload.get("bindings_by_model_case_backend")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    if set(bindings) != {"resnet50|b052|hailo8_to_trt"}:
        errors.append("cache_binding_identity_not_exact")
        return {}
    binding = bindings.get("resnet50|b052|hailo8_to_trt")
    if not isinstance(binding, Mapping):
        errors.append("cache_binding_not_an_object")
        return {}
    _semantic_binding_projection(
        binding, run_dir=run_dir, label="cache_binding", errors=errors,
    )
    return dict(binding)


def _verify_runner(
    runner: Mapping[str, Any], *, run_dir: Path, errors: list[str],
) -> dict[str, Any]:
    rows = runner.get("rows")
    if (
        runner.get("ok") is not True
        or runner.get("orchestration_status") != "ok"
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
    ):
        errors.append("native_runner_envelope_or_row_count_invalid")
        return {}
    row = rows[0]
    errors.extend(_semantic_mismatches(
        row,
        {
            "model": "resnet50",
            "case_id": "b052",
            "setup_id": "orin_nx_hailo8_01",
            "comparison_backend": "hailo8",
            "status": "ok",
            "result_ok": True,
            "returncode": 0,
            "timed_out": False,
            "child_result_fresh": True,
            "eval_run_id": run_dir.name,
            "source_run_id": "hailo8_to_trt",
        },
        prefix="native_runner.row",
    ))
    if _finite_positive(row.get("fps_makespan")) is None:
        errors.append("native_runner_fps_not_finite_positive")
    return dict(row)


def _verify_repetition_row(
    record: Mapping[str, Any],
    *,
    expected_runtime_id: str,
    label: str,
    errors: list[str],
) -> float | None:
    errors.extend(_semantic_mismatches(
        record,
        {
            "ok": True,
            "status": "ok",
            "mode": "native_hailort_tensorrt_fifo",
            "frames": 10,
            "completed_frames": 10,
            "requested_frames": 10,
            "completed_work_units": 10,
            "warmup": 2,
            "queue_depth": 2,
            "task": "classification",
            "runtime_instance_id": expected_runtime_id,
            "process_local_repetition_index": 1,
            "repetition_index": 1,
            "repetition_runtime_scope": "fresh_process_per_repetition",
        },
        prefix=label,
    ))
    if not _zero_number(record.get("duration_s")):
        errors.append(f"{label}.duration_s_not_zero")
    if not str(record.get("repetition_id") or "").strip():
        errors.append(f"{label}.repetition_id_missing")
    fps = _finite_positive(record.get("fps_makespan"))
    if fps is None:
        errors.append(f"{label}.fps_not_finite_positive")
    return fps


def _verify_result(
    result: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    run_dir: Path,
    errors: list[str],
) -> None:
    errors.extend(_semantic_mismatches(
        result,
        {
            "ok": True,
            # Classification writers use ``ok`` as the canonical top-level
            # success flag; when another writer mirrors ``status``, it must
            # agree instead of becoming a second mandatory field.
            **({"status": "ok"} if "status" in result else {}),
            "mode": "native_hailort_tensorrt_fifo",
            "frames": 10,
            "completed_frames": 10,
            "requested_frames": 10,
            "warmup": 2,
            "queue_depth": 2,
            "task": "classification",
            "precision": "float32_layout_fp16",
            "hw_arch": "hailo8",
            "case": "b052",
            "setup_id": "orin_nx_hailo8_01",
            "eval_run_id": run_dir.name,
            "source_run_id": "hailo8_to_trt",
            "repetitions_requested": 1,
            "repetitions_completed": 1,
            "repetition_status": "complete",
            "repetition_runtime_scope": "fresh_process_per_repetition",
            "repetition_independence_verified": True,
            "native_split_quality_runtime_boundary_verified": True,
            "hailo_runtime_output_count": 1,
            "hailo_runtime_output_name": "resnet50_part1_b52/conv24",
            "hailo_runtime_output_frame_bytes": 1_605_632,
        },
        prefix="native_result",
    ))
    if not _zero_number(result.get("duration_s")):
        errors.append("native_result.duration_s_not_zero")
    result_fps = _finite_positive(result.get("fps_makespan"))
    if result_fps is None:
        errors.append("native_result_fps_not_finite_positive")

    runtime_id = _fresh_process_runtime_id(result.get("runtime_instance_id"))
    if not runtime_id:
        errors.append("native_result_current_runtime_identity_invalid")

    records = result.get("repetition_records")
    if (
        not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], Mapping)
    ):
        errors.append("native_result_repetition_record_count_invalid")
    else:
        _verify_repetition_row(
            records[0], expected_runtime_id=runtime_id,
            label="native_result.repetition_record", errors=errors,
        )

    embedded = result.get("native_split_quality_binding")
    if not isinstance(embedded, Mapping):
        errors.append("native_result_semantic_binding_missing")
    else:
        materialized_projection = _semantic_binding_projection(
            binding, run_dir=run_dir, label="materialized_binding",
            errors=errors,
        )
        embedded_projection = _semantic_binding_projection(
            embedded, run_dir=run_dir, label="native_result.binding",
            errors=errors,
        )
        if embedded_projection != materialized_projection:
            errors.append("native_result_semantic_binding_projection_mismatch")


def _diagnostics(
    *,
    binding_set: Mapping[str, Any],
    binding: Mapping[str, Any],
    runner_row: Mapping[str, Any],
    result: Mapping[str, Any],
    result_path: Path | None,
    summary: Mapping[str, Any],
    summary_status: str,
) -> dict[str, Any]:
    attestation = binding_set.get("cache_verify_attestation")
    attestation = attestation if isinstance(attestation, Mapping) else {}
    artifacts = binding.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    result_actual_sha = ""
    result_size = 0
    if result_path is not None:
        try:
            result_actual_sha = _sha256_file(result_path)
            result_size = result_path.stat().st_size
        except OSError:
            pass
    declared_result_sha = str(
        runner_row.get("native_fifo_result_sha256") or ""
    ).strip().lower()
    command = result.get("native_command_contract")
    command = command if isinstance(command, Mapping) else {}
    return {
        "collector_summary": {
            "read_status": summary_status,
            "row_count": summary.get("row_count"),
            "ok_count": summary.get("ok_count"),
            "evidence_status": summary.get("evidence_status"),
            "authoritative": False,
        },
        "cache_selection": {
            "selected_cache_root": attestation.get("selected_cache_root"),
            "source_binding_sha256": attestation.get(
                "source_binding_sha256"
            ),
            "artifact_set_sha256": attestation.get(
                "source_binding_artifact_set_sha256"
            ),
            "authoritative": False,
        },
        "hash_receipt_command": {
            "binding_set_sha256": binding_set.get("binding_set_sha256"),
            "binding_sha256": binding.get("binding_sha256"),
            "artifact_sha256_by_role": {
                str(role): row.get("sha256")
                for role, row in artifacts.items()
                if isinstance(row, Mapping)
            },
            "engine_build_receipt_present": isinstance(
                binding.get("engine_build_receipt"), Mapping
            ),
            "command_contract_present": bool(command),
            "command_contract_sha256": command.get("contract_sha256"),
            "consumer_attestation_present": isinstance(
                result.get("native_split_quality_consumer_attestation"),
                Mapping,
            ),
            "authoritative": False,
        },
        "native_result_file": {
            "path": str(result_path or ""),
            "actual_sha256": result_actual_sha,
            "declared_sha256": declared_result_sha,
            "sha256_match": bool(
                result_actual_sha
                and declared_result_sha
                and result_actual_sha == declared_result_sha
            ),
            "actual_size_bytes": result_size,
            "declared_size_bytes": runner_row.get(
                "native_fifo_result_size_bytes"
            ),
            "authoritative": False,
        },
        "redundant_execution_mirrors": {
            "result_fps_makespan": result.get("fps_makespan"),
            "runner_fps_makespan": runner_row.get("fps_makespan"),
            "repetition_evidence_count": len(
                result.get("repetition_evidence")
            ) if isinstance(result.get("repetition_evidence"), list) else None,
            "repetition_count_requested": result.get(
                "repetition_count_requested"
            ),
            "repetition_count_attempted": result.get(
                "repetition_count_attempted"
            ),
            "repetition_count_valid": result.get(
                "repetition_count_valid"
            ),
            "authoritative": False,
        },
    }


def verify_cache_canary(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    errors: list[str] = []

    manifest_path = _required_regular_path(
        run_dir, Path("run_manifest.json"), label="manifest", errors=errors,
    )
    binding_path = _required_regular_path(
        run_dir, _BINDING_SET_REL, label="cache_binding_set", errors=errors,
    )
    runner_path = _required_regular_path(
        run_dir, _RUNNER_REL, label="native_runner", errors=errors,
    )
    result_path = _required_regular_path(
        run_dir, _RESULT_REL, label="native_result", errors=errors,
    )

    manifest = (
        _read_object(manifest_path, errors, label="manifest")
        if manifest_path is not None else {}
    )
    binding_set = (
        _read_object(binding_path, errors, label="cache_binding_set")
        if binding_path is not None else {}
    )
    runner = (
        _read_object(runner_path, errors, label="native_runner")
        if runner_path is not None else {}
    )
    result = (
        _read_object(result_path, errors, label="native_result")
        if result_path is not None else {}
    )

    _verify_manifest(manifest, run_dir=run_dir, errors=errors)
    binding = _verify_binding_set(
        binding_set, run_dir=run_dir, errors=errors,
    )
    runner_row = _verify_runner(runner, run_dir=run_dir, errors=errors)
    _verify_result(
        result, binding=binding, run_dir=run_dir, errors=errors,
    )

    summary, summary_status = _read_diagnostic_object(
        run_dir, _SUMMARY_REL,
    )
    diagnostics = _diagnostics(
        binding_set=binding_set,
        binding=binding,
        runner_row=runner_row,
        result=result,
        result_path=result_path,
        summary=summary,
        summary_status=summary_status,
    )
    return {
        "ok": not errors,
        "status": (
            "semantic_cache_canary_pass"
            if not errors else "cache_canary_failed"
        ),
        "run_dir": str(run_dir),
        "expected_identity": list(EXPECTED_IDENTITY),
        "expected_variant": EXPECTED_VARIANT,
        "authoritative_sources": {
            "manifest": str(manifest_path or ""),
            "binding_set": str(binding_path or ""),
            "native_runner": str(runner_path or ""),
            "native_result": str(result_path or ""),
        },
        "diagnostics": diagnostics,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = verify_cache_canary(Path(args.run_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
