#!/usr/bin/env python3
"""Fail-closed v2.79.8 YOLO11l six-path runtime-identity gate.

The v2.79.6 gate accepted a timing row plus a separately valid build receipt.
That did not prove that the measured Full row was produced by the named
accelerator.  This verifier instead requires three fresh Native-Full producer
rows, two fresh composed rows, and exactly one hash-bound read-only import.
Generic reciprocal Full timings and technical blocks are never terminal PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROFILE_ID = "yolo11l_v2798_r8b_full_b067_gate"
MODEL_ID = "yolo11l"
MODEL_SHA256 = "f0fcdf56a4ac24d87ec30c627170492ccad9db80486ec5694df6de65c1b3d147"
RELEASE_VERSION = "2.79.8"
BUILD_ID = "v2.79.8-yolo11-gate-profile-schema-closure"
VERDICT_SCHEMA = "onnx-splitpoint/yolo11-r8b-terminal-gate/v2"
RECOVERY_SCHEMA = "onnx-splitpoint/yolo11-r8b-recovery-manifest"
ARTIFACT_INDEX_SCHEMA = "onnx-splitpoint/artifact-index"
RUNTIME_IDENTITY_CONTRACT = "backend_bound_native_full_v2"
MAX_JSON_BYTES = 64 * 1024 * 1024
RUN_PROFILE_IDS = {
    "hailo8", "hailo8_to_trt", "hailo10", "hailo10_to_tensorrt",
    "deepx_m1_full", "deepx_m1_to_tensorrt",
}
SETUPS = {
    "orin_nx_hailo8_01": {"hailo8"},
    "orin_nx_hailo10_01": {"hailo10", "hailo10h"},
    "orin_nx_deepx_m1_01": {"deepx", "deepx_m1"},
}
FRESH_REQUIRED = [
    "hailo8_full", "hailo10h_full", "deepx_full",
    "hailo10h_b067_composed", "deepx_b067_composed",
]


def _load_sibling(filename: str, module_name: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"sibling_import_failed:{filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_sibling(
    "verify_v2796_yolo11_r8b_gate.py", "_v2798_gate_v2796_base",
)
recovery_tool = _load_sibling(
    "prepare_v2798_yolo11_r8b_recovery.py", "_v2798_gate_recovery",
)

# Reuse only the hardened no-follow readers and release/run identity checks.
# The v2.79.8 classifications below do not call the v2.79.6 row classifier.
base.PROFILE_ID = PROFILE_ID
base.RELEASE_VERSION = RELEASE_VERSION
base.BUILD_ID = BUILD_ID
base.RUN_PROFILE_IDS = set(RUN_PROFILE_IDS)
base.SETUPS = {
    "orin_nx_hailo8_01": "hailo8",
    "orin_nx_hailo10_01": "hailo10",
    "orin_nx_deepx_m1_01": "deepx_m1",
}


class GateError(ValueError):
    pass


def _need(value: Any, reason: str) -> None:
    if not value:
        raise GateError(reason)


def _mapping(value: Any, reason: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), reason)
    return dict(value)


def _sha(value: Any, reason: str) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    _need(re.fullmatch(r"[0-9a-f]{64}", token) is not None, reason)
    return token


def _positive(value: Any, reason: str) -> float:
    _need(
        isinstance(value, (int, float)) and not isinstance(value, bool), reason,
    )
    number = float(value)
    _need(math.isfinite(number) and number > 0.0, reason)
    return number


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_completed_detection_payload(
    payload: Mapping[str, Any], *, label: str,
) -> None:
    _need(
        payload.get("schema")
        == "onnx-splitpoint/frozen-completed-detection-result-artifact",
        f"{label}_schema",
    )
    _need(payload.get("schema_version") == 1, f"{label}_version")
    _need(payload.get("record_schema") == "xyxy_score_class_id_v1", f"{label}_record_schema")
    _need(payload.get("coordinate_space") == "original_image_xyxy_pixels", f"{label}_coordinate_space")
    _need(
        payload.get("sort_policy")
        == "score_desc_class_id_asc_xyxy_lexicographic_v1",
        f"{label}_sort_policy",
    )
    detections = payload.get("detections")
    _need(isinstance(detections, list), f"{label}_detections")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(detections):
        row = _mapping(raw, f"{label}_detection_{index}")
        _need(
            set(row) == {"class_id", "score", "x1", "y1", "x2", "y2"},
            f"{label}_detection_fields_{index}",
        )
        _need(type(row["class_id"]) is int and row["class_id"] >= 0, f"{label}_class_{index}")
        for field in ("score", "x1", "y1", "x2", "y2"):
            value = row[field]
            _need(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"{label}_{field}_{index}",
            )
        _need(0.0 <= float(row["score"]) <= 1.0, f"{label}_score_range_{index}")
        _need(float(row["x2"]) >= float(row["x1"]), f"{label}_x_order_{index}")
        _need(float(row["y2"]) >= float(row["y1"]), f"{label}_y_order_{index}")
        normalized.append(row)
    expected = sorted(
        normalized,
        key=lambda row: (
            -float(row["score"]), int(row["class_id"]),
            float(row["x1"]), float(row["y1"]),
            float(row["x2"]), float(row["y2"]),
        ),
    )
    _need(normalized == expected, f"{label}_canonical_sort")


def _external_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": base._file_sha(path),
        "size_bytes": path.stat().st_size,
    }


def _safe_external_file(path: Path, label: str) -> Path:
    token = os.fspath(path)
    _need(os.path.isabs(token), f"{label}_not_absolute")
    _need(os.path.normpath(token) == token, f"{label}_not_canonical")
    return base._safe_file(path.parent, path.name, label)


def _provider_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _validate_profile(run: Path, expected_profile: Path) -> dict[str, Any]:
    expected = base._yaml(
        expected_profile.parent, expected_profile.name, "expected_profile",
    )
    source = base._yaml(run, "profile_source.yaml", "profile_source")
    resolved = base._yaml(run, "profile.yaml", "profile")
    _need(isinstance(expected, Mapping), "expected_profile_not_mapping")
    _need(source == expected, "profile_source_mismatch")
    for label, raw in (("source", source), ("resolved", resolved)):
        profile = _mapping(raw, f"{label}_profile")
        _need(profile.get("name") == PROFILE_ID, f"{label}_profile_id")
        workflow = _mapping(profile.get("workflow"), f"{label}_workflow")
        _need(workflow.get("execution_mode") == "generate_and_run", f"{label}_execution_mode")
        _need(workflow.get("only_model") == MODEL_ID, f"{label}_model_scope")
        _need(workflow.get("max_models") == 1, f"{label}_model_count")
        profiles = profile.get("run_profiles")
        _need(isinstance(profiles, list), f"{label}_run_profiles")
        profile_map = {
            str(row.get("id") or ""): dict(row)
            for row in profiles if isinstance(row, Mapping)
        }
        _need(set(profile_map) == RUN_PROFILE_IDS, f"{label}_run_profile_set")
        _need(
            profile_map["hailo10_to_tensorrt"].get("hardware_setup_id")
            == "orin_nx_hailo10_01",
            f"{label}_hailo10_run_binding",
        )
        _need(
            profile_map["hailo8_to_trt"].get("hardware_setup_id")
            == "orin_nx_hailo8_01",
            f"{label}_hailo8_run_binding",
        )
        _need(
            profile_map["deepx_m1_to_tensorrt"].get("hardware_setup_id")
            == "orin_nx_deepx_m1_01",
            f"{label}_deepx_run_binding",
        )
        native = _mapping(profile.get("native_producers"), f"{label}_native")
        _need(native.get("enabled") is True, f"{label}_native_disabled")
        _need(list(native.get("backends") or []) == ["hailo8", "hailo10h", "deepx"], f"{label}_native_backends")
        _need(native.get("frames") == 32, f"{label}_native_frames")
        _need(native.get("warmup") == 4, f"{label}_native_warmup")
        _need(native.get("repetitions") == 1, f"{label}_native_repetitions")
        _need(native.get("inflight") == 1, f"{label}_native_inflight")
        _need(native.get("dump_outputs") is True, f"{label}_native_dump_outputs")
        full = _mapping(native.get("full_baselines"), f"{label}_full_baselines")
        _need(full.get("enabled") is True, f"{label}_full_baselines_disabled")
        _need(list(full.get("backends") or []) == ["hailo8", "hailo10h", "deepx"], f"{label}_full_backends")
        policy = _mapping(profile.get("v2798_yolo11_recovery"), f"{label}_recovery")
        _need(policy.get("schema") == "onnx-splitpoint/yolo11-r8b-recovery-policy", f"{label}_recovery_schema")
        _need(policy.get("schema_version") == 1, f"{label}_recovery_version")
        _need(list(policy.get("imported_terminal_paths") or []) == ["hailo8_b067_composed"], f"{label}_recovery_import_scope")
        _need(list(policy.get("fresh_terminal_paths") or []) == FRESH_REQUIRED, f"{label}_recovery_fresh_scope")
        _need(policy.get("forbid_generic_full_success") is True, f"{label}_generic_full_not_forbidden")
        hailo = _mapping(profile.get("hailo_build"), f"{label}_hailo")
        _need(hailo.get("build_full") is True and hailo.get("build_part1") is True, f"{label}_hailo_build_scope")
        _need(hailo.get("cold_build_timeout_s") == 0, f"{label}_hailo_timeout")
        forced = _mapping(
            _mapping(profile.get("selection_policy"), f"{label}_selection").get("forced_cases"),
            f"{label}_forced_cases",
        )
        _need(list(forced.get(MODEL_ID) or []) == ["b067"], f"{label}_forced_b067")
    return dict(resolved)


def _validate_run_identity_alias_aware(run: Path) -> dict[str, Any]:
    """Run the inherited identity proof after an explicit alias-set check."""
    matrix = _mapping(
        base._json(run, "hardware_matrix.json", "hardware_matrix"),
        "hardware_matrix",
    )
    rows = matrix.get("hardware_targets") or matrix.get("targets")
    _need(isinstance(rows, list), "hardware_targets_not_list")
    actual: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("enabled") is False:
            continue
        setup_id = str(raw.get("id") or raw.get("setup_id") or "")
        provider = _provider_token(
            raw.get("accelerator") or raw.get("provider"),
        )
        actual[setup_id] = provider
    _need(set(actual) == set(SETUPS), "hardware_setup_set_mismatch")
    for setup_id, allowed in SETUPS.items():
        _need(
            actual[setup_id] in allowed,
            f"hardware_provider_mismatch:{setup_id}:{actual[setup_id]}",
        )

    # The inherited checker also validates every session/build/model field but
    # historically accepted one scalar provider per setup.  Feed it the
    # already-validated physical tokens so hailo10 and hailo10h remain exact
    # aliases while no additional token is admitted.
    saved = dict(base.SETUPS)
    try:
        base.SETUPS = dict(actual)
        return base._validate_run_identity(run)
    finally:
        base.SETUPS = saved


def _validate_artifact_index(
    run: Path, release_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw = _mapping(base._json(run, "artifact_index.json", "artifact_index"), "artifact_index")
    _need(raw.get("schema") == ARTIFACT_INDEX_SCHEMA, "artifact_index_schema")
    _need(type(raw.get("schema_version")) is int and raw.get("schema_version") == 2, "artifact_index_requires_exact_schema_v2")
    closure, indexed = base._verify_terminal_artifact_index(
        run, run_identity=release_identity,
    )
    _need(
        "reports/native_producer_combined_summary.json" in indexed,
        "artifact_index_native_summary_missing",
    )
    return closure, indexed


def _validate_recovery_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _safe_external_file(path, "recovery_manifest")
    _need(path.stat().st_size <= MAX_JSON_BYTES, "recovery_manifest_too_large")
    try:
        supplied = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError("recovery_manifest_invalid_json") from exc
    supplied = _mapping(supplied, "recovery_manifest_not_mapping")
    expected_top = {
        "schema": RECOVERY_SCHEMA,
        "schema_version": 1,
        "target_version": RELEASE_VERSION,
        "target_build_id": BUILD_ID,
        "source_version": "2.79.6",
        "source_build_id": "v2.79.6-remaining-changes-yolo11-admission-closure",
        "source_access_mode": "read_only_hash_bound_no_copy",
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "imported_terminal_count": 1,
    }
    for key, expected in expected_top.items():
        _need(supplied.get(key) == expected, f"recovery_manifest_identity:{key}")
    _need(list(supplied.get("fresh_required") or []) == FRESH_REQUIRED, "recovery_manifest_fresh_scope")
    imports = _mapping(supplied.get("imports"), "recovery_manifest_imports")
    _need(set(imports) == {"hailo8_b067_composed"}, "recovery_manifest_import_scope")
    imported = _mapping(imports["hailo8_b067_composed"], "recovery_manifest_hailo8_import")
    _need(imported.get("terminal_eligible") is True, "recovery_import_not_terminal")
    _need(imported.get("evidence_mode") == "read_only_import", "recovery_import_mode")
    _need(imported.get("source_case_id") == "b067", "recovery_import_case")
    _sha(imported.get("source_row_sha256"), "recovery_import_row_sha")
    diagnostic = _mapping(supplied.get("diagnostic_only"), "recovery_manifest_diagnostic")
    _need(set(diagnostic) == {"hailo10h_full_generic"}, "recovery_diagnostic_scope")
    _need(
        _mapping(diagnostic["hailo10h_full_generic"], "recovery_h10_diagnostic").get("terminal_eligible") is False,
        "recovery_h10_generic_terminal_eligible",
    )
    source_text = str(supplied.get("source_run_dir") or "")
    _need(os.path.isabs(source_text) and os.path.normpath(source_text) == source_text, "recovery_source_not_canonical")
    source = Path(source_text)
    recomputed = recovery_tool.build_manifest(source)
    supplied_cmp = dict(supplied)
    recomputed_cmp = dict(recomputed)
    supplied_cmp.pop("created_at", None)
    recomputed_cmp.pop("created_at", None)
    _need(supplied_cmp == recomputed_cmp, "recovery_manifest_current_source_hash_mismatch")
    receipt = _external_identity(path)
    receipt["payload_sha256"] = _canonical_sha(supplied)
    return supplied, receipt


def _runtime_target(
    run: Path, *, setup_id: str, providers: set[str], run_id: str,
) -> dict[str, Any]:
    relative = Path("models") / MODEL_ID / "benchmark_results" / f"remote_benchmark_status_{setup_id}.json"
    path = base._safe_file(run, relative, f"remote_status_{setup_id}")
    payload = _mapping(base._json(run, relative, f"remote_status_{setup_id}"), f"remote_status_mapping_{setup_id}")
    _need(str(payload.get("status") or "").lower() == "ok", f"remote_status_not_ok:{setup_id}")
    gates = _mapping(payload.get("execution_gates"), f"remote_gates:{setup_id}")
    target = _mapping(gates.get("hardware_target"), f"remote_target:{setup_id}")
    _need(target.get("id") == setup_id, f"remote_setup_id:{setup_id}")
    runtime = _mapping(target.get("runtime"), f"remote_runtime:{setup_id}")
    actual = {
        _provider_token(target.get("accelerator")),
        _provider_token(target.get("provider")),
        _provider_token(runtime.get("provider")),
    }
    _need(bool(actual & providers), f"remote_provider:{setup_id}")
    run_ids = {str(item) for item in list(gates.get("hardware_run_ids") or [])}
    _need(run_id in run_ids, f"remote_run_id:{setup_id}:{run_id}")
    canonical = {str(item) for item in list(payload.get("canonical_run_ids_with_rows") or [])}
    _need(f"{run_id}_auto" in canonical, f"remote_canonical_result:{setup_id}:{run_id}")
    return {
        "status": base._artifact_identity(path, run),
        "setup_id": setup_id,
        "providers": sorted(providers),
        "run_id": run_id,
    }


def _runtime_log(
    run: Path, *, setup_id: str, required_markers: Sequence[str],
) -> dict[str, Any]:
    relative = Path("models") / MODEL_ID / "benchmark_results" / "remote_diagnostics" / setup_id / "logs" / "stdout.txt"
    path = base._safe_file(run, relative, f"runtime_log_{setup_id}")
    text = path.read_text(encoding="utf-8", errors="strict")
    for marker in required_markers:
        _need(marker in text, f"runtime_log_marker_missing:{setup_id}:{marker}")
    return base._artifact_identity(path, run)


def _indexed_json_candidates(
    run: Path, indexed: Mapping[str, Mapping[str, Any]], *, basename: str,
) -> list[tuple[Path, dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for logical in sorted(indexed):
        if Path(logical).name != basename:
            continue
        path = base._safe_file(run, logical, f"indexed_json:{basename}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            candidates.append((path, dict(payload)))
    return candidates


def _hailo_native_runtime_report(
    run: Path, indexed: Mapping[str, Mapping[str, Any]],
    *, spec: "NativeSpec", row: Mapping[str, Any], arch: str,
) -> dict[str, Any]:
    basename = Path(str(row.get("report") or "")).name
    _need(basename == "native_full_hailo_report.json", f"native_hailo_report_name:{spec.key}")
    expected_api = "vstreams" if arch == "hailo8" else "infer_model"
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path, payload in _indexed_json_candidates(
        run, indexed, basename=basename,
    ):
        if (
            payload.get("backend") == spec.backend
            and payload.get("model") == MODEL_ID
            and payload.get("setup_id") == spec.setup_id
            and _provider_token(payload.get("hw_arch")) == arch
            and payload.get("runtime_api") == expected_api
        ):
            matches.append((path, payload))
    _need(len(matches) == 1, f"native_hailo_runtime_report_count:{spec.key}:{len(matches)}")
    path, report = matches[0]
    throughput = _mapping(report.get("throughput"), f"native_hailo_report_throughput:{spec.key}")
    _need(throughput.get("requested_frames") == 32, f"native_hailo_report_requested:{spec.key}")
    _need(throughput.get("completed_frames") == 32, f"native_hailo_report_completed:{spec.key}")
    _need(throughput.get("completed_work_units") == 32, f"native_hailo_report_units:{spec.key}")
    _need(throughput.get("completed_work_units_status") == "exact_runtime_counter", f"native_hailo_report_counter:{spec.key}")
    _need(
        throughput.get("completed_work_units_source")
        == row.get("completed_work_units_source"),
        f"native_hailo_report_counter_source:{spec.key}",
    )
    _need(report.get("copy_outputs") is True, f"native_hailo_report_copy_outputs:{spec.key}")
    _need(report.get("claim_copy_outputs_verified") is True, f"native_hailo_report_copy_verification:{spec.key}")
    _need(report.get("completed_task_endpoint_attested") is True, f"native_hailo_report_endpoint:{spec.key}")
    _need(str(report.get("completed_task_endpoint_attestation_status") or "").lower() == "passed", f"native_hailo_report_endpoint_status:{spec.key}")
    _need(
        _sha(report.get("completed_task_result_artifact_file_sha256"), f"native_hailo_report_completed_sha:{spec.key}")
        == _sha(row.get("completed_task_result_artifact_file_sha256"), f"native_hailo_row_completed_sha:{spec.key}"),
        f"native_hailo_report_completed_mismatch:{spec.key}",
    )
    _need(
        _mapping(
            report.get("completed_task_result_artifact"),
            f"native_hailo_report_completed_payload:{spec.key}",
        )
        == _mapping(
            row.get("completed_task_result_artifact"),
            f"native_hailo_row_completed_payload:{spec.key}",
        ),
        f"native_hailo_report_completed_payload_mismatch:{spec.key}",
    )
    report_fps = _positive(throughput.get("fps"), f"native_hailo_report_fps:{spec.key}")
    row_fps = _positive(row.get("fps_makespan"), f"native_hailo_row_fps:{spec.key}")
    elapsed_s = _positive(throughput.get("elapsed_s"), f"native_hailo_report_elapsed:{spec.key}")
    _need(
        math.isclose(32.0 / elapsed_s, report_fps, rel_tol=1e-9, abs_tol=1e-9),
        f"native_hailo_report_elapsed_fps_mismatch:{spec.key}",
    )
    _need(
        math.isclose(report_fps, row_fps, rel_tol=1e-12, abs_tol=1e-12),
        f"native_hailo_report_fps_mismatch:{spec.key}",
    )
    report_attestation = _mapping(
        report.get("completed_task_endpoint_attestation"),
        f"native_hailo_report_attestation:{spec.key}",
    )
    _need(
        _sha(report_attestation.get("endpoint_contract_hash"), f"native_hailo_report_endpoint_hash:{spec.key}")
        == _sha(row.get("completed_task_endpoint_contract_hash"), f"native_hailo_row_endpoint_hash:{spec.key}"),
        f"native_hailo_report_endpoint_hash_mismatch:{spec.key}",
    )
    _need(
        report_attestation.get("output_endpoint_id")
        == row.get("completed_task_output_endpoint_id"),
        f"native_hailo_report_endpoint_id_mismatch:{spec.key}",
    )
    return base._artifact_identity(path, run)


def _deepx_native_runtime_result(
    run: Path, indexed: Mapping[str, Mapping[str, Any]],
    *, spec: "NativeSpec", row: Mapping[str, Any],
) -> dict[str, Any]:
    basename = Path(str(row.get("result_source") or row.get("report") or "")).name
    _need(basename.endswith(".json"), "native_deepx_result_name")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for logical in sorted(indexed):
        if Path(logical).name != basename:
            continue
        path = base._safe_file(run, logical, f"indexed_deepx_json:{basename}")
        payload = base._json(run, logical, f"indexed_deepx_json:{basename}")
        if isinstance(payload, list):
            # Production BenchmarkSet output is a top-level row list.
            raw_rows = payload
        elif isinstance(payload, Mapping):
            nested_rows = payload.get("rows") or payload.get("results")
            raw_rows = nested_rows if isinstance(nested_rows, list) else [payload]
        else:
            continue
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                continue
            candidate = dict(raw)
            if (
                candidate.get("run_id") == spec.run_id
                and _provider_token(candidate.get("provider")) == "deepx_m1"
                and candidate.get("runtime_ok") is True
            ):
                matches.append((path, candidate))
    _need(len(matches) == 1, f"native_deepx_runtime_result_count:{len(matches)}")
    path, result = matches[0]
    _need(result.get("case_id") == "full", "native_deepx_result_case")
    _need(result.get("variant") == "full", "native_deepx_result_variant")
    _need(_provider_token(result.get("backend")) in {"deepx", "deepx_m1"}, "native_deepx_result_backend")
    _need(result.get("performance_benchmark_source") == "dx_engine_prepared_feed", "native_deepx_result_source")
    # The benchmark-suite result deliberately keeps the authoritative runtime,
    # endpoint, and completed-task evidence in this nested object.  The outer
    # row contains convenience projections only; accepting top-level-only
    # evidence would let a generic timing row borrow an unrelated DeepX run.
    prepared = _mapping(
        result.get("deepx_prepared_feed_benchmark"),
        "native_deepx_result_prepared_feed",
    )
    _need(prepared.get("enabled") is True, "native_deepx_prepared_enabled")
    _need(prepared.get("status") == "ok", "native_deepx_prepared_status")
    _need(
        prepared.get("benchmark_kind") == "prepared_feed_dx_engine",
        "native_deepx_prepared_kind",
    )
    _need(prepared.get("run_count") == 32, "native_deepx_prepared_run_count")
    _need(prepared.get("requested_frames") == 32, "native_deepx_prepared_requested")
    _need(prepared.get("completed_frames") == 32, "native_deepx_prepared_frames")
    _need(prepared.get("completed_work_units") == 32, "native_deepx_prepared_units")
    _need(
        prepared.get("completed_work_units_status") == "exact_runtime_counter",
        "native_deepx_prepared_counter",
    )
    _need(
        prepared.get("completed_work_units_source")
        == "dx_engine_prepared_feed_frozen_completion_timed_loop",
        "native_deepx_prepared_counter_source",
    )
    _need(prepared.get("warmup_count") == 4, "native_deepx_prepared_warmup")
    summary_prepared = _mapping(
        row.get("deepx_prepared_feed_benchmark"),
        "native_deepx_summary_prepared_feed",
    )
    expected_summary_prepared = dict(prepared)
    # The Native-Full collector adds this single canonical alias while
    # preserving every raw prepared-feed field byte-for-value.  It is not
    # present in the BenchmarkSet template's nested output.
    expected_summary_prepared["performance_benchmark_source"] = (
        "dx_engine_prepared_feed"
    )
    _need(
        expected_summary_prepared == summary_prepared,
        "native_deepx_prepared_summary_mismatch",
    )
    result_fps = _positive(prepared.get("fps_makespan"), "native_deepx_result_fps")
    row_fps = _positive(row.get("fps_makespan"), "native_deepx_row_fps")
    makespan_s = _positive(prepared.get("makespan_s"), "native_deepx_result_makespan")
    _need(
        math.isclose(32.0 / makespan_s, result_fps, rel_tol=1e-9, abs_tol=1e-9),
        "native_deepx_result_makespan_fps_mismatch",
    )
    _need(
        math.isclose(result_fps, row_fps, rel_tol=1e-12, abs_tol=1e-12),
        "native_deepx_result_fps_mismatch",
    )
    for field in ("completed_frames", "completed_work_units"):
        _need(
            result.get(field) == prepared.get(field) == row.get(field),
            f"native_deepx_result_projection_mismatch:{field}",
        )
    _need(
        result.get("completed_work_units_status")
        == prepared.get("completed_work_units_status")
        == row.get("completed_work_units_status"),
        "native_deepx_result_projection_mismatch:completed_work_units_status",
    )
    _need(
        result.get("completed_work_units_source")
        == prepared.get("completed_work_units_source")
        == row.get("completed_work_units_source"),
        "native_deepx_result_projection_mismatch:completed_work_units_source",
    )
    _need(
        math.isclose(
            _positive(result.get("fps_makespan"), "native_deepx_outer_fps"),
            result_fps, rel_tol=1e-12, abs_tol=1e-12,
        ),
        "native_deepx_outer_fps_mismatch",
    )
    outer_makespan = _positive(
        result.get("measured_makespan_s"), "native_deepx_outer_makespan",
    )
    summary_makespan = _positive(
        row.get("measured_makespan_s"), "native_deepx_summary_makespan",
    )
    _need(
        math.isclose(outer_makespan, makespan_s, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(summary_makespan, makespan_s, rel_tol=1e-12, abs_tol=1e-12),
        "native_deepx_makespan_projection_mismatch",
    )
    result_attestation = _mapping(
        prepared.get("completed_task_endpoint_attestation"),
        "native_deepx_result_attestation",
    )
    _need(result_attestation.get("attested") is True, "native_deepx_result_endpoint_unattested")
    _need(
        str(result_attestation.get("status") or "").lower() == "passed",
        "native_deepx_result_endpoint_status",
    )
    _need(
        result_attestation
        == _mapping(
            row.get("completed_task_endpoint_attestation"),
            "native_deepx_summary_attestation",
        ),
        "native_deepx_result_attestation_mismatch",
    )
    _need(
        _sha(result_attestation.get("endpoint_contract_hash"), "native_deepx_result_endpoint_hash")
        == _sha(row.get("completed_task_endpoint_contract_hash"), "native_deepx_row_endpoint_hash"),
        "native_deepx_result_endpoint_hash_mismatch",
    )
    _need(
        result_attestation.get("output_endpoint_id")
        == row.get("completed_task_output_endpoint_id"),
        "native_deepx_result_endpoint_id_mismatch",
    )
    _need(
        _mapping(
            prepared.get("completed_task_result_artifact"),
            "native_deepx_result_completed_payload",
        )
        == _mapping(
            row.get("completed_task_result_artifact"),
            "native_deepx_row_completed_payload",
        ),
        "native_deepx_result_completed_payload_mismatch",
    )
    _need(
        _sha(prepared.get("completed_task_result_artifact_file_sha256"), "native_deepx_result_completed_sha")
        == _sha(row.get("completed_task_result_artifact_file_sha256"), "native_deepx_row_completed_sha"),
        "native_deepx_result_completed_sha_mismatch",
    )
    return base._artifact_identity(path, run)


@dataclass(frozen=True)
class NativeSpec:
    key: str
    backend: str
    setup_id: str
    comparison: str
    run_id: str
    producer_impl: str
    kind: str


NATIVE_SPECS = (
    NativeSpec("hailo8_full", "native_full_hailo8", "orin_nx_hailo8_01", "hailo8", "hailo8", "hailo8_vstreams_full", "hailo8"),
    NativeSpec("hailo10h_full", "native_full_hailo10h", "orin_nx_hailo10_01", "hailo10h", "hailo10", "hailo10_infermodel_async_full", "hailo10h"),
    NativeSpec("deepx_full", "native_full_deepx", "orin_nx_deepx_m1_01", "deepx", "deepx_m1_full", "native_full_deepx_native_full_suite", "deepx"),
)


def _native_rows(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    relative = Path("reports/native_producer_combined_summary.json")
    path = base._safe_file(run, relative, "native_summary")
    payload = _mapping(base._json(run, relative, "native_summary"), "native_summary")
    _need(payload.get("schema") == "onnx-splitpoint/native-producer-combined-summary", "native_summary_schema")
    _need(type(payload.get("schema_version")) is int and payload.get("schema_version") >= 7, "native_summary_version")
    rows = payload.get("rows")
    _need(isinstance(rows, list), "native_summary_rows")
    return payload, [dict(row) for row in rows if isinstance(row, Mapping)], path


def _validate_repetition(row: Mapping[str, Any], spec: NativeSpec) -> None:
    _need(row.get("repetition_count_requested") == 1, f"native_repetition_requested:{spec.key}")
    _need(row.get("repetition_count_attempted") == 1, f"native_repetition_attempted:{spec.key}")
    _need(row.get("repetition_count_valid") == 1, f"native_repetition_valid:{spec.key}")
    _need(row.get("repetition_status") == "complete", f"native_repetition_status:{spec.key}")
    _need(row.get("repetition_independence_verified") is True, f"native_repetition_independence:{spec.key}")
    _need(row.get("repetition_runtime_scope") == "fresh_process_per_repetition", f"native_repetition_scope:{spec.key}")
    records = row.get("repetition_records")
    _need(isinstance(records, list) and len(records) == 1, f"native_repetition_records:{spec.key}")
    record = _mapping(records[0], f"native_repetition_record:{spec.key}")
    _need(record.get("ok") is True and record.get("status") == "ok", f"native_repetition_failed:{spec.key}")
    _need(type(record.get("returncode")) is int and record.get("returncode") == 0, f"native_repetition_returncode:{spec.key}")
    _need(record.get("timed_out") is False, f"native_repetition_timeout:{spec.key}")
    _need(
        re.fullmatch(
            r"fresh_process:[0-9a-f]{64}",
            str(record.get("runtime_instance_id") or ""),
        ) is not None,
        f"native_runtime_instance:{spec.key}",
    )
    _need(record.get("completed_work_units") == 32, f"native_repetition_completed_units:{spec.key}")
    repetition_fps = _positive(record.get("fps_makespan"), f"native_repetition_fps:{spec.key}")
    aggregate_fps = _positive(row.get("fps_makespan"), f"native_aggregate_fps:{spec.key}")
    _need(
        math.isclose(
            repetition_fps, aggregate_fps,
            rel_tol=1e-12, abs_tol=1e-12,
        ),
        f"native_repetition_fps_mismatch:{spec.key}",
    )


def _validate_completion(
    row: Mapping[str, Any], spec: NativeSpec,
    *, indexed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _need(row.get("frames") == 32, f"native_frames:{spec.key}")
    _need(row.get("completed_frames") == 32, f"native_completed_frames:{spec.key}")
    _need(row.get("completed_work_units") == 32, f"native_completed_units:{spec.key}")
    _need(row.get("completed_work_units_status") == "exact_runtime_counter", f"native_completion_counter:{spec.key}")
    _positive(row.get("fps_makespan"), f"native_makespan_fps:{spec.key}")
    _need(row.get("measurement_concurrency") == 1, f"native_measurement_concurrency:{spec.key}")
    _need(
        row.get("full_backend_throughput_source") != "full_latency_fps",
        f"native_generic_full_latency_source_forbidden:{spec.key}",
    )
    if spec.kind.startswith("hailo"):
        _need(row.get("latency_mean_ms") is None, f"native_reciprocal_latency_forbidden:{spec.key}")
        _need(row.get("latency_p50_ms") is None, f"native_p50_latency_forbidden:{spec.key}")
        _need(row.get("latency_p95_ms") is None, f"native_p95_latency_forbidden:{spec.key}")
        _need(
            row.get("latency_semantics")
            == "not_measured_async_or_streaming_throughput",
            f"native_latency_semantics:{spec.key}",
        )
        _positive(row.get("completion_interval_mean_ms"), f"native_completion_interval:{spec.key}")
        _need(row.get("configured_inflight") == 1, f"native_configured_inflight:{spec.key}")
    _need(row.get("e2e_scope") == "full_task_pipeline", f"native_e2e_scope:{spec.key}")
    _need(row.get("postprocess_included") is True, f"native_postprocess_missing:{spec.key}")
    _need(row.get("postprocess_completion_verified") is True, f"native_postprocess_unverified:{spec.key}")
    _need(row.get("completed_task_endpoint_attested") is True, f"native_endpoint_unattested:{spec.key}")
    _need(str(row.get("completed_task_endpoint_attestation_status") or "").lower() == "passed", f"native_endpoint_status:{spec.key}")
    attestation = _mapping(row.get("completed_task_endpoint_attestation"), f"native_endpoint_attestation:{spec.key}")
    _need(attestation.get("attested") is True, f"native_endpoint_attestation_flag:{spec.key}")
    _need(str(attestation.get("status") or "").lower() == "passed", f"native_endpoint_attestation_state:{spec.key}")
    _sha(row.get("completed_task_endpoint_contract_hash"), f"native_endpoint_hash:{spec.key}")
    _need(str(row.get("completed_task_output_endpoint_id") or ""), f"native_endpoint_id:{spec.key}")
    _need(row.get("completed_task_result_artifact_saved") is True, f"native_completed_artifact_unsaved:{spec.key}")
    completed = _sha(row.get("completed_task_result_artifact_sha256"), f"native_completed_artifact_sha:{spec.key}")
    completed_payload = _mapping(
        row.get("completed_task_result_artifact"),
        f"native_completed_artifact_payload:{spec.key}",
    )
    _validate_completed_detection_payload(
        completed_payload, label=f"native_completed_artifact:{spec.key}",
    )
    _need(
        _canonical_sha(completed_payload) == completed,
        f"native_completed_artifact_payload_mismatch:{spec.key}",
    )
    file_sha = _sha(row.get("completed_task_result_artifact_file_sha256"), f"native_completed_file_sha:{spec.key}")
    _need(file_sha == completed, f"native_completed_artifact_file_mismatch:{spec.key}")
    sealed_field = (
        "frozen_decoded_nms_normalization_result"
        if row.get("normalization_frozen") is True
        else "frozen_host_postprocess_result"
    )
    sealed = _mapping(row.get(sealed_field), f"native_sealed_result:{spec.key}")
    _need(
        _mapping(
            sealed.get("completed_result_artifact"),
            f"native_sealed_completed_artifact:{spec.key}",
        ) == completed_payload,
        f"native_sealed_completed_artifact_mismatch:{spec.key}",
    )
    _need(
        _sha(
            sealed.get("completed_result_artifact_sha256"),
            f"native_sealed_completed_sha:{spec.key}",
        ) == completed,
        f"native_sealed_completed_sha_mismatch:{spec.key}",
    )
    _need(sealed.get("detection_count") == len(completed_payload["detections"]), f"native_sealed_detection_count:{spec.key}")
    _need(sealed.get("detections") == completed_payload["detections"], f"native_sealed_detections:{spec.key}")
    indexed_matches = [
        dict(identity) for identity in indexed.values()
        if str(identity.get("sha256") or "") == file_sha
    ]
    _need(indexed_matches, f"native_completed_artifact_not_indexed:{spec.key}")
    _need(str(row.get("completed_task_result_artifact_verification_status") or "").lower() in {"verified", "verified_exact", "passed"}, f"native_completed_artifact_status:{spec.key}")
    command = _mapping(row.get("full_command_contract"), f"native_full_command:{spec.key}")
    declared_command_sha = _sha(
        row.get("full_command_contract_sha256"),
        f"native_full_command_sha:{spec.key}",
    )
    command_without_hash = dict(command)
    embedded_command_sha = command_without_hash.pop("contract_sha256", None)
    _need(embedded_command_sha == declared_command_sha, f"native_full_command_embedded_sha:{spec.key}")
    _need(_canonical_sha(command_without_hash) == declared_command_sha, f"native_full_command_payload_sha:{spec.key}")
    _need(command.get("runner") == "scripts/native_full_baseline_eval_runner.py", f"native_full_command_runner:{spec.key}")
    _need(command.get("schema") == "onnx-splitpoint/native-full-command-contract", f"native_full_command_schema:{spec.key}")
    _need(command.get("schema_version") == 1, f"native_full_command_version:{spec.key}")
    runner_path = Path(__file__).resolve().with_name(
        "native_full_baseline_eval_runner.py"
    )
    _need(
        _sha(command.get("runner_sha256"), f"native_full_command_runner_sha:{spec.key}")
        == base._file_sha(runner_path),
        f"native_full_command_runner_bytes:{spec.key}",
    )
    _need(command.get("backend") == spec.backend, f"native_full_command_backend:{spec.key}")
    _need(command.get("backend_arg") == spec.kind, f"native_full_command_backend_arg:{spec.key}")
    _need(command.get("model") == MODEL_ID and command.get("case") == "full", f"native_full_command_variant:{spec.key}")
    _need(command.get("setup_id") == spec.setup_id, f"native_full_command_setup:{spec.key}")
    _need(command.get("comparison_backend") == spec.comparison, f"native_full_command_comparison:{spec.key}")
    options = _mapping(command.get("runtime_options"), f"native_full_command_options:{spec.key}")
    for field, expected in {
        "frames": 32,
        "warmup": 4,
        "performance_repetitions": 1,
        "inflight": 1,
        "dump_outputs": True,
    }.items():
        _need(options.get(field) == expected, f"native_full_command_option:{spec.key}:{field}")
    artifacts = _mapping(command.get("artifacts"), f"native_full_command_artifacts:{spec.key}")
    performance_report = _mapping(
        artifacts.get("performance_report"),
        f"native_full_command_performance_report:{spec.key}",
    )
    _need(str(performance_report.get("path") or ""), f"native_full_command_report_path:{spec.key}")
    _sha(performance_report.get("sha256"), f"native_full_command_report_sha:{spec.key}")
    _need(command.get("complete") is True, f"native_full_command_incomplete:{spec.key}")
    return {
        "completed_result_sha256": file_sha,
        "indexed_current_byte_matches": indexed_matches,
        "full_command_contract_sha256": declared_command_sha,
        "performance_report": dict(performance_report),
    }


def _deepx_current_artifact(
    run: Path, status_rel: Path, *, expected_source: str, full: bool,
) -> dict[str, Any]:
    status_path = base._safe_file(run, status_rel, "deepx_status")
    status = _mapping(base._json(run, status_rel, "deepx_status"), "deepx_status")
    output_path: Path | None = None
    if full:
        _need(status.get("schema") == "onnx-splitpoint/deepx-artifact-status", "deepx_full_status_schema")
        _need(status.get("status") == "ok", "deepx_full_status")
        contracts = [
            dict(item) for item in list(status.get("contracts") or [])
            if isinstance(item, Mapping) and item.get("variant") == "full"
        ]
        _need(len(contracts) == 1, "deepx_full_contract_count")
        contract = contracts[0]
        _need(_sha(contract.get("source_onnx_sha256"), "deepx_full_source") == expected_source, "deepx_full_source_mismatch")
        endpoint = _mapping(contract.get("endpoint_semantic_attestation"), "deepx_full_endpoint")
        _need(endpoint.get("pass") is True, "deepx_full_endpoint_unattested")
        identity = _mapping(status.get("current_full_artifact_identity"), "deepx_full_identity")
        _need(identity.get("status") == "verified", "deepx_full_identity_status")
        _need(identity.get("mode") == "v2796_current_bytes_sha256", "deepx_full_identity_mode")
        dxnn_rel = Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite" / "deepx" / "deepx_m1" / "full" / "model.dxnn"
    else:
        _need(status.get("ok") is True and status.get("status") in {"ok", "ready", "success"}, "deepx_b067_status")
        _need(_sha(status.get("source_onnx_sha256"), "deepx_b067_source") == expected_source, "deepx_b067_source_mismatch")
        contract = _mapping(status.get("cache_contract"), "deepx_b067_cache_contract")
        _need(contract.get("case_id") == "b067", "deepx_b067_case")
        case_root = Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite" / "b067"
        dxnn_rel = case_root / str(status.get("dxnn_path") or "")
        output_rel = case_root / str(status.get("output_contract") or "")
        output_path = base._safe_file(run, output_rel, "deepx_b067_output_contract")
        output = _mapping(base._json(run, output_rel, "deepx_b067_output_contract"), "deepx_b067_output_contract")
        _need(output.get("case_id") == "b067", "deepx_b067_output_case")
        _need(_sha(output.get("source_onnx_sha256"), "deepx_b067_output_source") == expected_source, "deepx_b067_output_source_mismatch")
        endpoint = _mapping(output.get("endpoint_semantic_attestation"), "deepx_b067_endpoint")
        _need(endpoint.get("pass") is True, "deepx_b067_endpoint_unattested")
        _need(_sha(status.get("output_contract_sha256"), "deepx_b067_output_hash") == base._file_sha(output_path), "deepx_b067_output_hash_mismatch")
    dxnn = base._safe_file(run, dxnn_rel, "deepx_dxnn")
    _need(dxnn.stat().st_size > 0, "deepx_dxnn_empty")
    dxnn_sha = base._file_sha(dxnn)
    if full:
        for key in ("artifact_sha256", "suite_artifact_sha256"):
            _need(_sha(identity.get(key), f"deepx_full_identity_{key}") == dxnn_sha, f"deepx_full_identity_mismatch:{key}")
        _need(identity.get("artifact_size_bytes") == dxnn.stat().st_size, "deepx_full_artifact_size")
        _need(identity.get("suite_artifact_size_bytes") == dxnn.stat().st_size, "deepx_full_suite_size")
        _need(_sha(contract.get("suite_artifact_sha256"), "deepx_full_contract_artifact") == dxnn_sha, "deepx_full_contract_artifact_mismatch")
    else:
        _need(_sha(status.get("dxnn_sha256"), "deepx_b067_dxnn_sha") == dxnn_sha, "deepx_b067_dxnn_hash_mismatch")
        _need(status.get("dxnn_size_bytes") == dxnn.stat().st_size, "deepx_b067_dxnn_size")
        _need(_sha(output.get("artifact_sha256"), "deepx_b067_output_artifact") == dxnn_sha, "deepx_b067_output_artifact_mismatch")
        _need(output.get("artifact_size_bytes") == dxnn.stat().st_size, "deepx_b067_output_size")
    evidence = {
        "status": base._artifact_identity(status_path, run),
        "artifact": base._artifact_identity(dxnn, run),
        "source_onnx_sha256": expected_source,
        "endpoint_attested": True,
    }
    if output_path is not None:
        evidence["output_contract"] = base._artifact_identity(output_path, run)
    return evidence


def _native_full(
    run: Path, rows: Sequence[Mapping[str, Any]], spec: NativeSpec,
    *, source_bindings: Mapping[str, Any],
    indexed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    matched = [
        dict(row) for row in rows
        if row.get("backend") == spec.backend
        and row.get("model") == MODEL_ID
        and row.get("case") == "full"
        and row.get("setup_id") == spec.setup_id
        and row.get("comparison_backend") == spec.comparison
    ]
    _need(len(matched) == 1, f"native_row_count:{spec.key}:{len(matched)}")
    row = matched[0]
    _need(row.get("run_id") == spec.run_id, f"native_run_id:{spec.key}")
    _need(row.get("producer_impl") == spec.producer_impl, f"native_producer_impl:{spec.key}")
    _need(row.get("execution_mode") == "native_full_baseline", f"native_execution_mode:{spec.key}")
    _need(row.get("ok") is True and row.get("status") == "ok", f"native_status:{spec.key}")
    _need(type(row.get("returncode")) is int and row.get("returncode") == 0, f"native_returncode:{spec.key}")
    _need(row.get("timed_out") is False, f"native_timeout:{spec.key}")
    _need(row.get("failure_reason") in {None, ""}, f"native_failure_reason:{spec.key}")
    _validate_repetition(row, spec)
    completion_evidence = _validate_completion(row, spec, indexed=indexed)
    full_source = str(source_bindings["full_sha256"])
    target_run_id = spec.run_id
    providers = SETUPS[spec.setup_id]
    target = _runtime_target(
        run, setup_id=spec.setup_id, providers=providers, run_id=target_run_id,
    )
    if spec.kind.startswith("hailo"):
        arch = spec.kind
        receipt_rel = Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite" / "hailo" / ("hailo8" if arch == "hailo8" else "hailo10") / "full" / "hailo_hef_build_receipt.json"
        artifact = base._hailo_success(
            run, receipt_rel, arch, full=True,
            expected_source_sha256=full_source,
        )
        probe = _mapping(row.get("runtime_python_probe"), f"native_runtime_probe:{spec.key}")
        _need(str(probe.get("selected") or ""), f"native_runtime_python:{spec.key}")
        _need(_provider_token(probe.get("hw_arch")) == arch, f"native_runtime_probe_arch:{spec.key}")
        overlay = _mapping(row.get("hailo_output_contract_overlay"), f"native_overlay:{spec.key}")
        _need(overlay.get("resolution_status") == "attested", f"native_overlay_unattested:{spec.key}")
        _sha(overlay.get("sha256"), f"native_overlay_sha:{spec.key}")
        _need(
            row.get("hailo_hef_build_receipt_status")
            == "hailo_hef_build_receipt_verified_exact",
            f"native_hailo_receipt_status:{spec.key}",
        )
        _need(row.get("failure_reason") != "hailo_full_output_contract_overlay_invalid", f"native_overlay_invalid:{spec.key}")
        expected_counter = (
            "synchronous_vstreams_frozen_postprocess_success_counter"
            if arch == "hailo8"
            else "hailo_infermodel_frozen_postprocess_success_callback_counter"
        )
        _need(
            row.get("completed_work_units_source") == expected_counter,
            f"native_callback_completion_source:{spec.key}",
        )
        _need(row.get("claim_copy_outputs_verified") is True, f"native_copy_outputs_unverified:{spec.key}")
        _need(row.get("copy_outputs") is True, f"native_copy_outputs_disabled:{spec.key}")
        _need(row.get("host_postprocess_frozen") is True, f"native_frozen_postprocess_missing:{spec.key}")
        _sha(row.get("frozen_host_postprocess_contract_sha256"), f"native_frozen_postprocess_sha:{spec.key}")
        native_runtime = _hailo_native_runtime_report(
            run, indexed, spec=spec, row=row, arch=arch,
        )
    else:
        artifact = _deepx_current_artifact(
            run,
            Path("models") / MODEL_ID / "benchmark_set" / "deepx" / "deepx_artifact_status.json",
            expected_source=full_source, full=True,
        )
        _need(row.get("performance_benchmark_source") == "dx_engine_prepared_feed", "native_deepx_performance_source")
        _need(row.get("outer_makespan_verified") is True, "native_deepx_outer_makespan")
        _need(row.get("completed_work_units_source") not in {None, ""}, "native_deepx_counter_source")
        prepared = _mapping(row.get("deepx_prepared_feed_benchmark"), "native_deepx_prepared_feed")
        _need(prepared.get("completed_work_units_status") == "exact_runtime_counter", "native_deepx_prepared_counter")
        _need(prepared.get("completed_work_units") == 32, "native_deepx_prepared_units")
        _need(str(row.get("dxnn_path") or ""), "native_deepx_dxnn_path")
        native_runtime = _deepx_native_runtime_result(
            run, indexed, spec=spec, row=row,
        )
    command_report = _mapping(
        completion_evidence.get("performance_report"),
        f"native_command_report:{spec.key}",
    )
    _need(
        _sha(command_report.get("sha256"), f"native_command_report_sha:{spec.key}")
        == _sha(native_runtime.get("sha256"), f"native_runtime_report_sha:{spec.key}"),
        f"native_command_runtime_report_hash_mismatch:{spec.key}",
    )
    _need(
        Path(str(command_report.get("path") or "")).name
        == Path(str(native_runtime.get("relative_path") or "")).name,
        f"native_command_runtime_report_path_mismatch:{spec.key}",
    )
    return {
        "backend": spec.backend,
        "setup_id": spec.setup_id,
        "run_id": spec.run_id,
        "variant": "full",
        "case_id": "full",
        "evidence_mode": "fresh_native_full",
        "terminal_status": "success",
        "technical_classification": "fresh_native_full_runtime_measured_and_endpoint_bound",
        "producer_impl": spec.producer_impl,
        "fps_makespan": float(row["fps_makespan"]),
        "runtime_target": target,
        "native_runtime_report": native_runtime,
        "artifact": artifact,
        "completion_evidence": completion_evidence,
        "runtime_attestation": {
            "status": "passed",
            "provider_bound": True,
            "hardware_bound": True,
            "variant_bound": True,
            "current_artifact_bytes_bound": True,
            "makespan_bound": True,
            "completion_bound": True,
            "endpoint_bound": True,
        },
    }


@dataclass(frozen=True)
class ComposedSpec:
    key: str
    backend: str
    setup_id: str
    run_id: str
    filename: str
    stage1: set[str]
    kind: str


COMPOSED_SPECS = (
    ComposedSpec("hailo10h_b067_composed", "hailo10_to_tensorrt", "orin_nx_hailo10_01", "hailo10_to_tensorrt", "benchmark_results_hailo10_to_tensorrt_auto.json", {"hailo10", "hailo10h"}, "hailo10h"),
    ComposedSpec("deepx_b067_composed", "deepx_m1_to_tensorrt", "orin_nx_deepx_m1_01", "deepx_m1_to_tensorrt", "benchmark_results_deepx_m1_to_tensorrt_auto.json", {"deepx", "deepx_m1"}, "deepx"),
)


def _bind_normalized_composed_source(
    run: Path, spec: ComposedSpec, row: Mapping[str, Any],
) -> tuple[str, Path, dict[str, Any]]:
    """Bind immutable raw generator output to its unique normalized row."""

    source_row_sha = _canonical_sha(row)
    normalized_rel = (
        Path("models") / MODEL_ID / "benchmark_results"
        / "normalized_results.json"
    )
    normalized_path = base._safe_file(
        run, normalized_rel, f"composed_normalized_results:{spec.key}",
    )
    normalized_payload = base._json(
        run, normalized_rel, f"composed_normalized_results:{spec.key}",
    )
    if isinstance(normalized_payload, Mapping):
        normalized_raw = (
            normalized_payload.get("results")
            or normalized_payload.get("rows")
            or normalized_payload.get("records")
        )
    else:
        normalized_raw = normalized_payload
    _need(isinstance(normalized_raw, list), f"composed_normalized_rows:{spec.key}")
    normalized_matches = [
        dict(item) for item in normalized_raw
        if isinstance(item, Mapping)
        and str(item.get("run_id") or "") == spec.run_id
        and str(item.get("case_id") or "").lower() == "b067"
        and str(item.get("source_row_sha256") or "").lower()
        == source_row_sha
    ]
    _need(
        len(normalized_matches) == 1,
        f"composed_normalized_source_binding:{spec.key}:{len(normalized_matches)}",
    )
    return source_row_sha, normalized_path, normalized_matches[0]


def _composed(
    run: Path, spec: ComposedSpec, *, source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    rows = base._rows(run, spec.filename)
    _need(len(rows) == 1, f"composed_row_count:{spec.key}:{len(rows)}")
    row = rows[0]
    _need(str(row.get("case_id") or "").lower() == "b067", f"composed_case:{spec.key}")
    _need(row.get("run_id") == spec.run_id, f"composed_run_id:{spec.key}")
    _need(_provider_token(row.get("provider")) == "tensorrt", f"composed_provider:{spec.key}")
    _need(_provider_token(row.get("full_provider")) == "tensorrt", f"composed_full_provider:{spec.key}")
    _need(_provider_token(row.get("stage1_provider")) in spec.stage1, f"composed_stage1:{spec.key}")
    _need(_provider_token(row.get("stage2_provider")) == "tensorrt", f"composed_stage2:{spec.key}")
    tokens = _mapping(row.get("backend_tokens"), f"composed_tokens:{spec.key}")
    _need(_provider_token(tokens.get("stage1")) in spec.stage1, f"composed_stage1_token:{spec.key}")
    _need(_provider_token(tokens.get("stage2")) == "tensorrt", f"composed_stage2_token:{spec.key}")
    _need(base._technical_success(row, "composed"), f"composed_not_measured:{spec.key}")
    throughput = _mapping(row.get("throughput"), f"composed_throughput:{spec.key}")
    _need(throughput.get("mode") == "measured_streaming", f"composed_throughput_mode:{spec.key}")
    makespan = _positive(throughput.get("fps_makespan"), f"composed_makespan:{spec.key}")
    # Raw BenchmarkSet result files are immutable generator output and do not
    # self-declare this digest.  Recompute it from the exact indexed row, then
    # bind the unique b067 projection in normalized_results.json to it.
    source_row_sha, normalized_path, _normalized_row = (
        _bind_normalized_composed_source(run, spec, row)
    )
    deployment = _mapping(row.get("deployment_contract"), f"composed_deployment:{spec.key}")
    trt = _mapping(deployment.get("native_trt_io_contracts"), f"composed_trt_contracts:{spec.key}")
    _need(_mapping(trt.get("part2"), f"composed_trt_part2:{spec.key}").get("backend") == "native_tensorrt", f"composed_trt_backend:{spec.key}")
    part1_sha = str(source_bindings["part1_sha256"])
    if spec.kind == "hailo10h":
        hailo = _mapping(deployment.get("hailo_io_contracts"), f"composed_hailo_contracts:{spec.key}")
        _need(_mapping(hailo.get("part1"), f"composed_hailo_part1:{spec.key}").get("backend") == "hailo", f"composed_hailo_backend:{spec.key}")
        artifact = base._hailo_success(
            run,
            Path("models/yolo11l/benchmark_set/legacy_suite/b067/hailo/hailo10/part1/hailo_hef_build_receipt.json"),
            "hailo10h", full=False, expected_source_sha256=part1_sha,
        )
        log = _runtime_log(
            run, setup_id=spec.setup_id,
            required_markers=["[hailo][runtime] hw_arch=hailo10h api=infer_model", "streaming throughput: fps(makespan)="],
        )
    else:
        deepx_contracts = _mapping(deployment.get("deepx_io_contracts"), f"composed_deepx_contracts:{spec.key}")
        _need(_mapping(deepx_contracts.get("part1"), f"composed_deepx_part1:{spec.key}").get("backend") in {"deepx", "deepx_m1"}, f"composed_deepx_backend:{spec.key}")
        artifact = _deepx_current_artifact(
            run,
            Path("models/yolo11l/benchmark_set/legacy_suite/b067/deepx/deepx_m1/part1/deepx_part1_artifact_status.json"),
            expected_source=part1_sha, full=False,
        )
        _need(row.get("runtime_ok") is True, "composed_deepx_runtime_ok")
        _need(type(row.get("returncode")) is int and row.get("returncode") == 0, "composed_deepx_returncode")
        log = _runtime_log(
            run, setup_id=spec.setup_id,
            required_markers=["deepx_m1", "fps(makespan)="],
        )
    target = _runtime_target(
        run, setup_id=spec.setup_id,
        providers=SETUPS[spec.setup_id], run_id=spec.run_id,
    )
    result_path = base._safe_file(
        run, Path("models") / MODEL_ID / "benchmark_results" / spec.filename,
        f"composed_result:{spec.key}",
    )
    return {
        "backend": spec.backend,
        "setup_id": spec.setup_id,
        "run_id": spec.run_id,
        "variant": "composed",
        "case_id": "b067",
        "source_case_id": "b067",
        "source_row_sha256": source_row_sha,
        "evidence_mode": "fresh_composed",
        "terminal_status": "success",
        "technical_classification": "fresh_composed_runtime_measured_and_backend_bound",
        "fps_makespan": makespan,
        "result_file": base._artifact_identity(result_path, run),
        "normalized_results": base._artifact_identity(normalized_path, run),
        "runtime_target": target,
        "runtime_log": log,
        "artifact": artifact,
        "runtime_attestation": {
            "status": "passed",
            "provider_bound": True,
            "hardware_bound": True,
            "variant_bound": True,
            "current_artifact_bytes_bound": True,
            "makespan_bound": True,
        },
    }


def _imported_path(manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    imported = _mapping(
        _mapping(manifest.get("imports"), "recovery_imports").get("hailo8_b067_composed"),
        "recovery_hailo8_import",
    )
    contract = _mapping(imported.get("runtime_contract"), "recovery_hailo8_contract")
    _need(contract.get("run_id") == "hailo8_to_trt", "recovery_hailo8_run_id")
    _need(contract.get("backend") == "hailo8_to_tensorrt", "recovery_hailo8_backend")
    _need(contract.get("stage1") == "hailo8" and contract.get("stage2") == "native_tensorrt", "recovery_hailo8_stages")
    _need(contract.get("variant") == "composed", "recovery_hailo8_variant")
    _need(contract.get("throughput_mode") == "measured_streaming", "recovery_hailo8_throughput")
    _positive(contract.get("fps_makespan"), "recovery_hailo8_fps")
    target = _mapping(imported.get("runtime_target"), "recovery_hailo8_target")
    _need(target.get("setup_id") == "orin_nx_hailo8_01", "recovery_hailo8_setup")
    _need(target.get("provider") == "hailo8", "recovery_hailo8_provider")
    _need(target.get("run_id") == "hailo8_to_trt", "recovery_hailo8_target_run")
    return {
        "backend": "hailo8_to_tensorrt",
        "setup_id": "orin_nx_hailo8_01",
        "run_id": "hailo8_to_trt",
        "variant": "composed",
        "case_id": "b067",
        "source_case_id": "b067",
        "source_row_sha256": _sha(imported.get("source_row_sha256"), "recovery_hailo8_row_sha"),
        "evidence_mode": "read_only_import",
        "terminal_status": "success",
        "technical_classification": "read_only_hash_bound_v2796_runtime_import",
        "fps_makespan": float(contract["fps_makespan"]),
        "recovery_manifest_sha256": str(receipt["sha256"]),
        "source_run_id": str(manifest.get("source_run_id") or ""),
        "source_result_sha256": _sha(
            _mapping(imported.get("result"), "recovery_hailo8_result").get("sha256"),
            "recovery_hailo8_result_sha",
        ),
        "runtime_attestation": {
            "status": "passed",
            "provider_bound": True,
            "hardware_bound": True,
            "variant_bound": True,
            "current_artifact_bytes_bound": True,
            "makespan_bound": True,
        },
    }


def verify(
    *, gate_output: Path, expected_profile: Path,
    recovery_manifest: Path, workflow_rc: int,
) -> dict[str, Any]:
    _need(type(workflow_rc) is int and workflow_rc in {0, 1}, f"workflow_rc_not_allowed:{workflow_rc!r}")
    run = base._find_run(gate_output)
    _validate_profile(run, expected_profile)
    release_identity = _validate_run_identity_alias_aware(run)
    source_manifest = base._release_source_identity(expected_profile)
    artifact_closure, indexed = _validate_artifact_index(run, release_identity)
    source_bindings = base._source_bindings(run)
    recovery, recovery_receipt = _validate_recovery_manifest(recovery_manifest)
    native_summary, native_rows, native_summary_path = _native_rows(run)
    paths: dict[str, Any] = {
        spec.key: _native_full(
            run, native_rows, spec, source_bindings=source_bindings,
            indexed=indexed,
        )
        for spec in NATIVE_SPECS
    }
    paths.update({
        spec.key: _composed(
            run, spec, source_bindings=source_bindings,
        )
        for spec in COMPOSED_SPECS
    })
    paths["hailo8_b067_composed"] = _imported_path(
        recovery, recovery_receipt,
    )
    expected_keys = {
        "hailo8_full", "hailo10h_full", "deepx_full",
        "hailo8_b067_composed", "hailo10h_b067_composed",
        "deepx_b067_composed",
    }
    _need(set(paths) == expected_keys, "terminal_scope_incomplete")
    _need(all(path["terminal_status"] == "success" for path in paths.values()), "terminal_success_incomplete")
    _need(workflow_rc == 0, "workflow_rc_nonzero_for_success_only_gate")
    current_evidence = {
        "source_bindings": source_bindings,
        "native_summary": base._artifact_identity(native_summary_path, run),
        "fresh_paths": {
            key: value for key, value in paths.items()
            if value.get("evidence_mode") != "read_only_import"
        },
    }
    base._require_projected_identities_indexed(current_evidence, indexed=indexed)
    return {
        "schema": VERDICT_SCHEMA,
        "schema_version": 2,
        "status": "PASS",
        "ok": True,
        "valid_terminal": True,
        "scope_complete": True,
        "runtime_identity_contract": RUNTIME_IDENTITY_CONTRACT,
        "profile_id": PROFILE_ID,
        "version": RELEASE_VERSION,
        "build_id": BUILD_ID,
        "workflow_version": BUILD_ID,
        "release_identity": release_identity,
        "source_manifest_sha256": source_manifest["sha256"],
        "source_manifest": source_manifest,
        "artifact_index": artifact_closure["artifact_index"],
        "artifact_index_closure": artifact_closure,
        "recovery_manifest": recovery_receipt,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "run_dir": str(run),
        "workflow_rc": workflow_rc,
        "quality_decision_used_for_terminal": False,
        "required_path_count": 6,
        "fresh_native_full_count": 3,
        "fresh_composed_count": 2,
        "imported_terminal_count": 1,
        "success_count": 6,
        "blocked_count": 0,
        "source_bindings": source_bindings,
        "native_summary": {
            "schema": str(native_summary.get("schema") or ""),
            "schema_version": native_summary.get("schema_version"),
            "artifact": base._artifact_identity(native_summary_path, run),
        },
        "paths": paths,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-output", required=True, type=Path)
    parser.add_argument("--workflow-rc", required=True, type=int)
    parser.add_argument("--recovery-manifest", required=True, type=Path)
    parser.add_argument(
        "--expected-profile", type=Path,
        default=Path(__file__).resolve().parents[1] / "profiles" / "yolo11l_v2798_r8b_full_b067_gate.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = args.gate_output.expanduser().resolve(strict=True)
        profile = args.expected_profile.expanduser().resolve(strict=True)
        recovery = args.recovery_manifest.expanduser().resolve(strict=True)
        verdict = verify(
            gate_output=output, expected_profile=profile,
            recovery_manifest=recovery, workflow_rc=args.workflow_rc,
        )
        rc = 0
    except Exception as exc:
        verdict = {
            "schema": VERDICT_SCHEMA,
            "schema_version": 2,
            "status": "FAIL",
            "ok": False,
            "valid_terminal": False,
            "quality_decision_used_for_terminal": False,
            "error": f"{type(exc).__name__}:{exc}",
        }
        rc = 2
    print(json.dumps(verdict, indent=2, sort_keys=True, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

