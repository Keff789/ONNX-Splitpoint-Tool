#!/usr/bin/env python3
"""Read-only, fail-closed verifier for the v2.79.6 YOLO11l R8B gate.

The gate is deliberately technical.  Each of the three Full paths and each
current b067 composed path must end either in a receipt-bound runtime success
or in a precise technical block.  Accuracy/quality fields are archived as an
annotation but never select the terminal state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROFILE_ID = "yolo11l_v2796_r8b_full_b067_gate"
MODEL_ID = "yolo11l"
MODEL_SHA256 = "f0fcdf56a4ac24d87ec30c627170492ccad9db80486ec5694df6de65c1b3d147"
RELEASE_VERSION = "2.79.6"
BUILD_ID = "v2.79.6-remaining-changes-yolo11-admission-closure"
VERDICT_SCHEMA = "onnx-splitpoint/yolo11-r8b-terminal-gate/v1"
ATTEMPT_SCHEMA = "onnx-splitpoint/hailo-build-attempt-receipt"
HEF_RECEIPT_SCHEMA = "onnx-splitpoint/hailo-hef-build-receipt/v2"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_YAML_BYTES = 8 * 1024 * 1024
DOCUMENTED_UNLIMITED_TOKENS = [0, "off", "none", "unlimited", "disabled"]
ALLOWED_WORKFLOW_RCS = {0, 1}
ALLOWED_RESULT_BLOCK_CLASSES = {
    "backend_failed",
    "compiler_failed",
    "deepx_compile_failed",
    "deepx_compiler_failed",
    "deepx_runtime_failed",
    "execution_failed",
    "hailo_compiler_failed",
    "hailo_compiler_timeout",
    "hailo_runtime_failed",
    "invalid_runtime_contract",
    "native_runtime_failed",
    "remote_runtime_failed",
    "runtime_failed",
    "unsupported_backend_contract",
}
ALLOWED_HAILO_ATTEMPT_BLOCK_CLASSES = {
    "builder_exception",
    "hailo_dfc_cuda_cudnn_failure",
    "hailo_dfc_import_failed",
    "hard_timeout",
    "idle_timeout",
    "invalid_build_contract",
    "invalid_calibration_set",
    "invalid_preprocessing_contract",
    "launch_error",
    "missing_structured_result",
    "sdk_unavailable",
    "timeout",
    "unsupported_splitpoint",
}
ALLOWED_DEEPX_BLOCK_STATES = {
    "artifact_identity_failed",
    "build_failed",
    "cache_contract_preparation_failed",
    "cache_miss_blocked",
    "cache_publish_failed",
    "calibration_dir_missing",
    "calibration_manifest_contract_invalid",
    "compile_failed",
    "compiler_identity_unresolved",
    "compiler_not_ready",
    "failed",
    "model_missing",
    "runtime_failed",
    "task_contract_missing",
    "unsupported",
}
GENERIC_BLOCK_REASONS = {
    "backend failed",
    "compile failed",
    "compiler failed",
    "error",
    "execution failed",
    "failed",
    "runtime failed",
    "timeout",
    "unsupported",
}
RUN_PROFILE_IDS = {
    "hailo8",
    "hailo8_to_trt",
    "hailo10",
    "hailo10_to_tensorrt",
    "deepx_m1_full",
    "deepx_m1_to_tensorrt",
}
SETUPS = {
    "orin_nx_hailo8_01": "hailo8",
    "orin_nx_hailo10_01": "hailo10",
    "orin_nx_deepx_m1_01": "deepx_m1",
}
YOLO11_RAW_END_NODES = [
    "/model.23/cv2.0/cv2.0.2/Conv",
    "/model.23/cv3.0/cv3.0.2/Conv",
    "/model.23/cv2.1/cv2.1.2/Conv",
    "/model.23/cv3.1/cv3.1.2/Conv",
    "/model.23/cv2.2/cv2.2.2/Conv",
    "/model.23/cv3.2/cv3.2.2/Conv",
]


class GateError(ValueError):
    pass


def _need(value: Any, reason: str) -> None:
    if not value:
        raise GateError(reason)


def _sha(value: Any, label: str) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[len("sha256:"):]
    _need(
        len(token) == 64
        and all(character in "0123456789abcdef" for character in token),
        f"{label}_invalid_sha256",
    )
    return token


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), f"{label}_not_mapping")
    return dict(value)


def _open_dir_nofollow(path: Path, label: str) -> None:
    token = os.fspath(path)
    _need(os.path.isabs(token), f"{label}_not_absolute")
    _need(os.path.normpath(token) == token, f"{label}_not_canonical")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for part in Path(token).parts[1:]:
            _need(part not in {"", ".", ".."}, f"{label}_unsafe")
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        _need(stat.S_ISDIR(os.fstat(fd).st_mode), f"{label}_not_directory")
    except OSError as exc:
        raise GateError(f"{label}_unsafe_or_missing:{type(exc).__name__}") from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _safe_file(root: Path, relative: str | Path, label: str) -> Path:
    rel = Path(relative)
    _need(not rel.is_absolute(), f"{label}_absolute")
    _need(all(part not in {"", ".", ".."} for part in rel.parts), f"{label}_unsafe")
    _open_dir_nofollow(root, f"{label}_root")
    current = root
    try:
        for index, part in enumerate(rel.parts):
            current = current / part
            info = current.lstat()
            _need(not stat.S_ISLNK(info.st_mode), f"{label}_symlink_component")
            if index + 1 < len(rel.parts):
                _need(stat.S_ISDIR(info.st_mode), f"{label}_ancestor_not_directory")
            else:
                _need(stat.S_ISREG(info.st_mode), f"{label}_not_regular")
    except GateError:
        raise
    except OSError as exc:
        raise GateError(f"{label}_missing:{type(exc).__name__}") from exc
    return current


def _safe_relative(value: Any, label: str) -> Path:
    token = str(value or "").strip().replace("\\", "/")
    rel = Path(token)
    _need(token and not rel.is_absolute(), f"{label}_absolute_or_empty")
    _need(all(part not in {"", ".", ".."} for part in rel.parts), f"{label}_unsafe")
    return rel


def _json(root: Path, relative: str | Path, label: str) -> Any:
    path = _safe_file(root, relative, label)
    _need(path.stat().st_size <= MAX_JSON_BYTES, f"{label}_too_large")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{label}_invalid_json") from exc


def _yaml(root: Path, relative: str | Path, label: str) -> Any:
    path = _safe_file(root, relative, label)
    _need(path.stat().st_size <= MAX_YAML_BYTES, f"{label}_too_large")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise GateError(f"{label}_invalid_yaml") from exc


def _find_run(root: Path) -> Path:
    _open_dir_nofollow(root, "gate_output")
    if (root / "run_manifest.json").is_file():
        return root
    candidates: list[Path] = []
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
            child = root / entry.name
            try:
                _safe_file(child, "run_manifest.json", "candidate_manifest")
            except GateError:
                continue
            candidates.append(child)
    _need(len(candidates) == 1, f"expected_one_run_dir_found_{len(candidates)}")
    return candidates[0]


def _validate_profile(run: Path, expected_profile: Path) -> dict[str, Any]:
    expected = _yaml(expected_profile.parent, expected_profile.name, "expected_profile")
    source = _yaml(run, "profile_source.yaml", "profile_source")
    resolved = _yaml(run, "profile.yaml", "profile")
    _need(isinstance(expected, Mapping), "expected_profile_not_mapping")
    _need(source == expected, "profile_source_mismatch")
    for label, raw in (("source", source), ("resolved", resolved)):
        profile = _mapping(raw, f"{label}_profile")
        _need(profile.get("name") == PROFILE_ID, f"{label}_profile_id")
        workflow = _mapping(profile.get("workflow"), f"{label}_workflow")
        _need(workflow.get("execution_mode") == "generate_and_run", f"{label}_execution_mode")
        _need(workflow.get("only_model") == MODEL_ID, f"{label}_only_model")
        _need(workflow.get("max_models") == 1, f"{label}_max_models")
        hailo = _mapping(profile.get("hailo_build"), f"{label}_hailo")
        _need(hailo.get("build_full") is True, f"{label}_full_disabled")
        _need(hailo.get("build_part1") is True, f"{label}_part1_disabled")
        _need(hailo.get("build_part2") is False, f"{label}_part2_enabled")
        _need(hailo.get("cold_build_timeout_s") == 0, f"{label}_cold_timeout_not_unlimited")
        _need(hailo.get("immutable_attempt_receipts") is True, f"{label}_receipts_mutable")
        _need(hailo.get("terminal_attempt_selection") == "last_attempt_even_on_failure", f"{label}_terminal_selection")
        _need(list(hailo.get("hard_timeout_disable_tokens") or []) == DOCUMENTED_UNLIMITED_TOKENS, f"{label}_timeout_tokens")
        _need(set(hailo.get("targets") or []) == {"hailo8", "hailo10"}, f"{label}_hailo_targets")
        profiles = profile.get("run_profiles")
        _need(isinstance(profiles, list), f"{label}_run_profiles")
        _need({str(row.get("id") or "") for row in profiles if isinstance(row, Mapping)} == RUN_PROFILE_IDS, f"{label}_run_profile_set")
        forced = _mapping(profile.get("selection_policy"), f"{label}_selection").get("forced_cases")
        forced = _mapping(forced, f"{label}_forced_cases")
        _need(list(forced.get(MODEL_ID) or []) == ["b067"], f"{label}_forced_b067")
        suite = _mapping(profile.get("model_suite"), f"{label}_model_suite")
        primary = suite.get("primary")
        _need(isinstance(primary, list) and len(primary) == 1, f"{label}_model_count")
        model = _mapping(primary[0], f"{label}_model")
        _need(model.get("id") == MODEL_ID, f"{label}_model_id")
        _need(_sha(model.get("model_sha256"), f"{label}_model") == MODEL_SHA256, f"{label}_model_sha")
        _need(_mapping(profile.get("native_producers"), f"{label}_native").get("enabled") is False, f"{label}_native_enabled")
        _need(_mapping(profile.get("energy"), f"{label}_energy").get("enabled") is False, f"{label}_energy_enabled")
        _need(_mapping(profile.get("ranking_validation"), f"{label}_ranking").get("enabled") is False, f"{label}_ranking_enabled")
    return dict(resolved)


def _validate_run_identity(run: Path) -> dict[str, Any]:
    manifest = _mapping(_json(run, "run_manifest.json", "run_manifest"), "run_manifest")
    _need(manifest.get("profile_id") == PROFILE_ID, "run_manifest_profile")
    _need(str(manifest.get("run_id") or "") == run.name, "run_manifest_run_id")
    _need(manifest.get("tool_version") == RELEASE_VERSION, "run_manifest_tool_version")
    _need(
        manifest.get("current_tool_version") == RELEASE_VERSION,
        "run_manifest_current_tool_version",
    )
    if "workflow_version" in manifest:
        _need(
            manifest.get("workflow_version") == BUILD_ID,
            "run_manifest_workflow_version",
        )
    _need(
        manifest.get("current_workflow_version") == BUILD_ID,
        "run_manifest_current_workflow_version",
    )
    current_build = _mapping(
        manifest.get("current_tool_build"), "run_manifest_current_tool_build"
    )
    _need(
        current_build.get("package_version") == RELEASE_VERSION,
        "run_manifest_current_build_package_version",
    )
    _need(
        current_build.get("build_id") == BUILD_ID,
        "run_manifest_current_build_id",
    )
    current_session_id = str(manifest.get("current_session_id") or "")
    _need(current_session_id, "run_manifest_current_session_id")
    sessions = manifest.get("execution_sessions")
    _need(isinstance(sessions, list), "run_manifest_execution_sessions")
    current_sessions = [
        dict(row)
        for row in sessions
        if isinstance(row, Mapping)
        and str(row.get("session_id") or "") == current_session_id
    ]
    _need(len(current_sessions) == 1, "run_manifest_current_session_count")
    current_session = current_sessions[0]
    _need(
        current_session.get("tool_version") == RELEASE_VERSION,
        "run_manifest_session_tool_version",
    )
    _need(
        current_session.get("workflow_version") == BUILD_ID,
        "run_manifest_session_workflow_version",
    )
    session_build = _mapping(
        current_session.get("tool_build"), "run_manifest_session_tool_build"
    )
    _need(
        session_build.get("package_version") == RELEASE_VERSION,
        "run_manifest_session_build_package_version",
    )
    _need(
        session_build.get("build_id") == BUILD_ID,
        "run_manifest_session_build_id",
    )
    model = _mapping(_json(run, f"models/{MODEL_ID}/model_manifest.json", "model_manifest"), "model_manifest")
    _need(model.get("model_id") == MODEL_ID, "model_manifest_id")
    observed = model.get("observed_model_sha256") or model.get("model_sha256") or _mapping(model.get("file"), "model_file").get("sha256")
    _need(_sha(observed, "observed_model") == MODEL_SHA256, "observed_model_sha_mismatch")
    matrix = _mapping(_json(run, "hardware_matrix.json", "hardware_matrix"), "hardware_matrix")
    rows = matrix.get("hardware_targets") or matrix.get("targets")
    _need(isinstance(rows, list), "hardware_targets_not_list")
    actual: dict[str, str] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or raw.get("enabled") is False:
            continue
        setup_id = str(raw.get("id") or raw.get("setup_id") or "")
        accelerator = str(raw.get("accelerator") or raw.get("provider") or "").lower().replace("-", "_")
        actual[setup_id] = accelerator
    _need(set(actual) == set(SETUPS), "hardware_setup_set_mismatch")
    for setup, expected in SETUPS.items():
        _need(actual[setup] in {expected, expected.replace("_m1", "")}, f"hardware_provider_mismatch:{setup}")
    return {
        "version": RELEASE_VERSION,
        "build_id": BUILD_ID,
        "workflow_version": BUILD_ID,
        "tool_version": str(manifest["tool_version"]),
        "current_tool_version": str(manifest["current_tool_version"]),
        "current_workflow_version": str(manifest["current_workflow_version"]),
        "current_session_id": current_session_id,
        "current_tool_build": {
            "package_version": str(current_build["package_version"]),
            "build_id": str(current_build["build_id"]),
        },
    }


def _source_bindings(run: Path) -> dict[str, Any]:
    """Bind portable Full evidence and the current in-run b067 Part-1 bytes.

    Full is intentionally portable: the retained run's model-manifest observed
    SHA must equal the release/profile pin.  The launcher separately hashes the
    installed external model before execution.  The retained verifier never
    depends on that host-only absolute path.  Part-1 is an in-run artifact and
    is therefore hashed directly here.
    """

    model_manifest_rel = Path("models") / MODEL_ID / "model_manifest.json"
    model_manifest_path = _safe_file(
        run, model_manifest_rel, "source_binding_model_manifest"
    )
    model_manifest = _mapping(
        _json(run, model_manifest_rel, "source_binding_model_manifest"),
        "source_binding_model_manifest",
    )
    observed = (
        model_manifest.get("observed_model_sha256")
        or model_manifest.get("model_sha256")
        or _mapping(model_manifest.get("file"), "source_binding_model_file").get(
            "sha256"
        )
    )
    _need(
        _sha(observed, "source_binding_full_model") == MODEL_SHA256,
        "source_binding_full_model_sha_mismatch",
    )

    case_root = Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite" / "b067"
    split_rel = case_root / "split_manifest.json"
    split_path = _safe_file(run, split_rel, "b067_split_manifest")
    split = _mapping(_json(run, split_rel, "b067_split_manifest"), "b067_split_manifest")
    boundary = split.get("boundary", split.get("boundary_index"))
    _need(type(boundary) is int and int(boundary) == 67, "b067_boundary_mismatch")
    _need(
        str(split.get("full_model") or "").replace("\\", "/")
        == f"../models/{MODEL_ID}.onnx",
        "b067_full_model_binding",
    )
    part1_rel = _safe_relative(
        split.get("part1") or split.get("part1_model") or split.get("part1_path"),
        "b067_part1",
    )
    _need(len(part1_rel.parts) == 1, "b067_part1_not_case_local")
    part1_path = _safe_file(run, case_root / part1_rel, "b067_part1")
    _need(part1_path.stat().st_size > 0, "b067_part1_empty")
    part1_sha = _file_sha(part1_path)
    return {
        "full_sha256": MODEL_SHA256,
        "full_binding_mode": "portable_profile_pin_plus_model_manifest_observed_sha256",
        "model_manifest": _artifact_identity(model_manifest_path, run),
        "part1_sha256": part1_sha,
        "part1": _artifact_identity(part1_path, run),
        "split_manifest": _artifact_identity(split_path, run),
    }


def _rows(run: Path, filename: str) -> list[dict[str, Any]]:
    relative = Path("models") / MODEL_ID / "benchmark_results" / filename
    try:
        raw = _json(run, relative, filename)
    except GateError as exc:
        if "_missing:" in str(exc):
            return []
        raise
    if isinstance(raw, list):
        return [dict(row) for row in raw if isinstance(row, Mapping)]
    if isinstance(raw, Mapping):
        for key in ("results", "rows", "records"):
            value = raw.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
    raise GateError(f"{filename}_invalid_rows")


def _case_row(run: Path, filename: str, case_id: str) -> dict[str, Any] | None:
    rows = _rows(run, filename)
    matched = [
        row for row in rows
        if str(row.get("case_id") or row.get("case") or "").strip().lower() == case_id
    ]
    _need(len(matched) <= 1, f"duplicate_result_row:{filename}:{case_id}")
    _need(
        len(rows) == len(matched),
        f"unexpected_result_rows:{filename}:{len(rows) - len(matched)}",
    )
    return matched[0] if matched else None


def _timing(row: Mapping[str, Any], variant: str) -> float:
    timings = row.get("timings") if isinstance(row.get("timings"), Mapping) else {}
    value = timings.get(variant) if isinstance(timings.get(variant), Mapping) else {}
    candidate = value.get("mean_ms")
    if candidate is None:
        candidate = row.get(f"{variant}_mean_ms", row.get("latency_mean_ms"))
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return 0.0
    number = float(candidate)
    return number if math.isfinite(number) and number > 0.0 else 0.0


def _technical_success(row: Mapping[str, Any] | None, variant: str) -> bool:
    if not row:
        return False
    status = row.get("variant_status") if isinstance(row.get("variant_status"), Mapping) else {}
    measured_raw = row.get("measured_variants")
    measured = (
        {str(value) for value in measured_raw}
        if isinstance(measured_raw, list)
        else set()
    )
    state = str(status.get(variant) or "").lower()
    ok = state in {"ok", "success", "measured", "passed"}
    raw_rc = row.get("returncode")
    explicit_runtime_marker = (
        row.get("runtime_ok") is True
        and type(raw_rc) is int
        and raw_rc == 0
    )
    measured_marker = isinstance(measured_raw, list) and variant in measured
    ok = ok and (explicit_runtime_marker or measured_marker)
    if row.get("runtime_ok") is False:
        ok = False
    if raw_rc is not None and raw_rc != "" and (
        type(raw_rc) is not int or raw_rc != 0
    ):
        ok = False
    return bool(ok and _timing(row, variant))


def _quality_annotation(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"present": False}
    return {
        "present": True,
        "final_pass": row.get("final_pass"),
        "task_quality_pass": row.get("task_quality_pass"),
        "quality_gate_status": row.get("quality_gate_status"),
        "used_for_terminal_decision": False,
    }


def _technical_row_block(row: Mapping[str, Any] | None, variant: str) -> dict[str, Any] | None:
    if not row or _technical_success(row, variant):
        return None
    status = row.get("variant_status") if isinstance(row.get("variant_status"), Mapping) else {}
    state = str(status.get(variant) or row.get("status") or "").lower()
    reasons = [
        str(row.get(name) or "").strip()
        for name in (
            "error_detail", "failure_reason", "status_detail", "error",
            "unsupported_reason", "error_class",
        )
    ]
    reason = next((value for value in reasons if value), "")
    error_class = str(
        row.get("error_class")
        or row.get("failure_kind")
        or row.get("timeout_kind")
        or ""
    ).strip().lower().replace("-", "_")
    lower = reason.lower().strip()
    _need(
        error_class in ALLOWED_RESULT_BLOCK_CLASSES,
        f"result_block_class_not_allowed:{error_class or 'missing'}",
    )
    _need(
        lower
        and lower not in GENERIC_BLOCK_REASONS
        and lower != error_class.replace("_", " ")
        and not lower.startswith("quality")
        and "accuracy_gate" not in lower,
        "result_block_reason_not_precise",
    )
    raw_rc = row.get("returncode")
    _need(
        raw_rc in {None, ""} or type(raw_rc) is int,
        "result_block_returncode_invalid",
    )
    nonzero_rc = type(raw_rc) is int and raw_rc != 0
    if state not in {"ok", "success", "passed", "measured"}:
        return {
            "source": "result_row",
            "status": error_class,
            "error_class": error_class,
            "reason": reason,
            "variant_status": state,
            "returncode": raw_rc if nonzero_rc else raw_rc,
        }
    return None


def _artifact_identity(path: Path, run: Path) -> dict[str, Any]:
    info = path.stat()
    return {"relative_path": path.relative_to(run).as_posix(), "sha256": _file_sha(path), "size_bytes": info.st_size}


def _release_source_identity(expected_profile: Path) -> dict[str, Any]:
    tool_root = expected_profile.parent.parent
    path = _safe_file(tool_root, "SOURCE_MANIFEST.json", "source_manifest")
    payload = _mapping(
        _json(tool_root, "SOURCE_MANIFEST.json", "source_manifest"),
        "source_manifest",
    )
    _need(
        payload.get("schema") == "onnx-splitpoint/source-manifest-v1",
        "source_manifest_schema",
    )
    _need(
        payload.get("package_version") == RELEASE_VERSION,
        "source_manifest_package_version",
    )
    _need(
        payload.get("workflow_version") == BUILD_ID,
        "source_manifest_workflow_version",
    )
    _need(
        type(payload.get("file_count")) is int and payload.get("file_count") > 0,
        "source_manifest_file_count",
    )
    return {
        "sha256": _file_sha(path),
        "size_bytes": path.stat().st_size,
        "package_version": RELEASE_VERSION,
        "workflow_version": BUILD_ID,
        "verification_contract": "wrapper_installed_scope_verified_before_hardware",
    }


def _verify_terminal_artifact_index(
    run: Path,
    *,
    run_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify the workflow's committed terminal closure without mutating it."""

    index_path = _safe_file(run, "artifact_index.json", "artifact_index")
    payload = _mapping(
        _json(run, "artifact_index.json", "artifact_index"), "artifact_index"
    )
    _need(
        payload.get("schema") == "onnx-splitpoint/artifact-index",
        "artifact_index_schema",
    )
    _need(
        type(payload.get("schema_version")) is int
        and int(payload.get("schema_version")) >= 2,
        "artifact_index_schema_version",
    )
    _need(payload.get("run_id") == run.name, "artifact_index_run_id")
    rows = payload.get("artifacts")
    _need(isinstance(rows, list), "artifact_index_rows")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"artifact_index_row_{index}")
        relative = _safe_relative(row.get("path"), f"artifact_index_path_{index}")
        logical = relative.as_posix()
        _need(logical != "artifact_index.json", "artifact_index_self_hash_forbidden")
        _need(logical not in indexed, f"artifact_index_duplicate_path:{logical}")
        path = _safe_file(run, relative, f"artifact_index_file_{index}")
        recorded_sha = _sha(row.get("sha256"), f"artifact_index_sha_{index}")
        recorded_size = row.get("size_bytes")
        _need(
            type(recorded_size) is int and recorded_size == path.stat().st_size,
            f"artifact_index_size_mismatch:{logical}",
        )
        _need(
            recorded_sha == _file_sha(path),
            f"artifact_index_sha_mismatch:{logical}",
        )
        indexed[logical] = {
            "relative_path": logical,
            "sha256": recorded_sha,
            "size_bytes": recorded_size,
        }

    session_id = str(run_identity.get("current_session_id") or "")
    terminal = _mapping(
        payload.get("terminal_closure"), "artifact_index_terminal_closure"
    )
    _need(
        terminal.get("schema")
        == "onnx-splitpoint/artifact-index-terminal-closure-binding",
        "artifact_index_terminal_schema",
    )
    _need(terminal.get("schema_version") == 1, "artifact_index_terminal_version")
    _need(terminal.get("status") == "pass", "artifact_index_terminal_status")
    _need(terminal.get("run_id") == run.name, "artifact_index_terminal_run_id")
    _need(
        str(terminal.get("session_id") or "") == session_id,
        "artifact_index_terminal_session_id",
    )
    _need(
        terminal.get("report_path") == "reports/artifact_index_closure.json",
        "artifact_index_terminal_report_path",
    )
    _need(
        terminal.get("self_hash_excluded") is True,
        "artifact_index_terminal_self_hash_contract",
    )
    _need(
        terminal.get("verification_error_count") == 0,
        "artifact_index_terminal_verification_errors",
    )

    closure_path = _safe_file(
        run,
        "reports/artifact_index_closure.json",
        "artifact_index_closure_report",
    )
    closure = _mapping(
        _json(
            run,
            "reports/artifact_index_closure.json",
            "artifact_index_closure_report",
        ),
        "artifact_index_closure_report",
    )
    _need(
        closure.get("schema")
        == "onnx-splitpoint/artifact-index-terminal-closure",
        "artifact_index_closure_schema",
    )
    _need(closure.get("schema_version") == 1, "artifact_index_closure_version")
    _need(closure.get("status") == "pass", "artifact_index_closure_status")
    _need(closure.get("run_id") == run.name, "artifact_index_closure_run_id")
    _need(
        str(closure.get("session_id") or "") == session_id,
        "artifact_index_closure_session_id",
    )
    manifest = _mapping(
        _json(run, "run_manifest.json", "artifact_index_run_manifest"),
        "artifact_index_run_manifest",
    )
    _need(
        closure.get("workflow_status") == manifest.get("status"),
        "artifact_index_closure_workflow_status",
    )
    _need(
        closure.get("artifact_index_path") == "artifact_index.json",
        "artifact_index_closure_index_path",
    )
    _need(
        closure.get("self_hash_excluded") is True,
        "artifact_index_closure_self_hash_contract",
    )
    _need(
        closure.get("verification_error_count") == 0,
        "artifact_index_closure_verification_errors",
    )
    _need(
        terminal.get("artifact_record_count") == len(rows)
        and closure.get("artifact_record_count") == len(rows),
        "artifact_index_artifact_record_count",
    )
    required_count = terminal.get("required_coverage_path_count")
    _need(
        type(required_count) is int
        and required_count > 0
        and closure.get("required_coverage_path_count") == required_count,
        "artifact_index_required_coverage_count",
    )

    required = {
        "run_manifest.json",
        "profile_source.yaml",
        "profile.yaml",
        "hardware_matrix.json",
        f"models/{MODEL_ID}/model_manifest.json",
        f"models/{MODEL_ID}/benchmark_set/legacy_suite/b067/split_manifest.json",
        "reports/artifact_index_closure.json",
    }
    missing = sorted(required - set(indexed))
    _need(not missing, "artifact_index_required_paths_missing:" + ",".join(missing))
    _need(
        indexed["reports/artifact_index_closure.json"]["sha256"]
        == _file_sha(closure_path),
        "artifact_index_closure_row_hash",
    )
    return (
        {
            "artifact_index": _artifact_identity(index_path, run),
            "closure_report": _artifact_identity(closure_path, run),
            "status": "pass",
            "session_id": session_id,
            "artifact_record_count": len(rows),
            "required_coverage_path_count": required_count,
        },
        indexed,
    )


def _require_projected_identities_indexed(
    value: Any,
    *,
    indexed: Mapping[str, Mapping[str, Any]],
) -> None:
    if isinstance(value, Mapping):
        if {"relative_path", "sha256", "size_bytes"}.issubset(value):
            logical = str(value.get("relative_path") or "")
            expected = indexed.get(logical)
            _need(expected is not None, f"gate_evidence_unindexed:{logical}")
            _need(
                str(expected.get("sha256") or "") == str(value.get("sha256") or "")
                and expected.get("size_bytes") == value.get("size_bytes"),
                f"gate_evidence_index_mismatch:{logical}",
            )
        for child in value.values():
            _require_projected_identities_indexed(child, indexed=indexed)
    elif isinstance(value, list):
        for child in value:
            _require_projected_identities_indexed(child, indexed=indexed)


def _hailo_success(
    run: Path,
    receipt_rel: Path,
    arch: str,
    *,
    full: bool,
    expected_source_sha256: str,
) -> dict[str, Any]:
    receipt_path = _safe_file(run, receipt_rel, f"{arch}_hef_receipt")
    receipt = _mapping(_json(run, receipt_rel, f"{arch}_hef_receipt"), f"{arch}_hef_receipt")
    _need(receipt.get("schema") == HEF_RECEIPT_SCHEMA, f"{arch}_hef_receipt_schema")
    actual_arch = str(receipt.get("hw_arch") or "").lower().replace("-", "")
    _need(actual_arch == arch, f"{arch}_hef_receipt_arch")
    source_sha = _sha(receipt.get("source_onnx_sha256"), f"{arch}_source")
    compiler_sha = _sha(receipt.get("compiler_onnx_sha256"), f"{arch}_compiler")
    hef_sha = _sha(receipt.get("hef_sha256"), f"{arch}_hef")
    _sha(receipt.get("cache_key"), f"{arch}_cache_key")
    cache = _mapping(receipt.get("cache_payload"), f"{arch}_cache_payload")
    _need(str(cache.get("hw_arch") or "").lower().replace("-", "") == arch, f"{arch}_cache_arch")
    _need(source_sha == expected_source_sha256, f"{arch}_source_artifact_mismatch")
    _need(
        _sha(cache.get("model_sha256"), f"{arch}_cache_model") == compiler_sha,
        f"{arch}_cache_compiler_mismatch",
    )
    if full:
        _need(source_sha == MODEL_SHA256, f"{arch}_full_source_model_mismatch")
        nodes = list(cache.get("end_nodes") or receipt.get("end_nodes") or [])
        _need(nodes == YOLO11_RAW_END_NODES, f"{arch}_full_raw_end_nodes")
    candidates = [receipt_rel.parent / "compiled.hef", receipt_rel.parent / "model.hef"]
    hef = None
    for candidate in candidates:
        try:
            hef = _safe_file(run, candidate, f"{arch}_hef_file")
            break
        except GateError as exc:
            if "_missing:" not in str(exc):
                raise
    _need(hef is not None, f"{arch}_hef_file_missing")
    _need(_file_sha(hef) == hef_sha, f"{arch}_hef_file_hash")
    _need(int(receipt.get("hef_size_bytes") or 0) == hef.stat().st_size > 0, f"{arch}_hef_file_size")
    return {
        "receipt": _artifact_identity(receipt_path, run),
        "artifact": _artifact_identity(hef, run),
        "source_onnx_sha256": source_sha,
        "compiler_onnx_sha256": compiler_sha,
        "hw_arch": arch,
    }


def _hailo_block(
    run: Path,
    receipt_dir: Path,
    arch: str,
    *,
    full: bool,
    expected_source_sha256: str,
) -> dict[str, Any] | None:
    pointer_rel = receipt_dir / "hailo_attempt_receipts" / "terminal_attempt.json"
    try:
        pointer_path = _safe_file(run, pointer_rel, f"{arch}_terminal_pointer")
        pointer = _mapping(_json(run, pointer_rel, f"{arch}_terminal_pointer"), f"{arch}_terminal_pointer")
    except GateError as exc:
        if "_missing:" in str(exc):
            return None
        raise
    _need(pointer.get("schema") == ATTEMPT_SCHEMA, f"{arch}_attempt_schema")
    _need(pointer.get("schema_version") == 2, f"{arch}_attempt_schema_version")
    _need(pointer.get("terminal") is True, f"{arch}_attempt_not_terminal")
    semantic = str(pointer.get("semantic_status") or "").lower()
    if semantic not in {"failed", "timeout", "unsupported"}:
        return None
    _need(str(pointer.get("hw_arch") or "").lower().replace("-", "") == arch, f"{arch}_attempt_arch")
    source_sha = _sha(pointer.get("source_onnx_sha256"), f"{arch}_attempt_source")
    _need(source_sha == expected_source_sha256, f"{arch}_attempt_source_mismatch")
    immutable_name = Path(str(pointer.get("immutable_receipt") or "")).name
    _need(immutable_name.startswith("attempt_") and immutable_name.endswith(".json"), f"{arch}_immutable_name")
    immutable_rel = pointer_rel.parent / immutable_name
    immutable_path = _safe_file(run, immutable_rel, f"{arch}_immutable_receipt")
    _need(_file_sha(immutable_path) == _sha(pointer.get("immutable_receipt_sha256"), f"{arch}_immutable"), f"{arch}_immutable_hash")
    immutable = _mapping(_json(run, immutable_rel, f"{arch}_immutable_receipt"), f"{arch}_immutable_receipt")
    for field in (
        "schema", "schema_version", "attempt_id", "terminal", "semantic_status",
        "hw_arch", "source_onnx_sha256", "ended_at_epoch_s", "compiler_phase",
        "last_active_stage", "error_class", "failure_kind", "timeout_kind",
        "unsupported_reason", "error", "end_nodes",
    ):
        _need(immutable.get(field) == pointer.get(field), f"{arch}_terminal_pointer_mismatch:{field}")
    _need(str(pointer.get("compiler_phase") or pointer.get("last_active_stage") or "").strip(), f"{arch}_attempt_phase_missing")
    detail = str(pointer.get("error") or pointer.get("error_class") or pointer.get("unsupported_reason") or pointer.get("failure_kind") or "").strip()
    block_class = str(
        pointer.get("error_class")
        or pointer.get("failure_kind")
        or pointer.get("timeout_kind")
        or ""
    ).strip().lower().replace("-", "_")
    _need(
        block_class in ALLOWED_HAILO_ATTEMPT_BLOCK_CLASSES,
        f"{arch}_attempt_block_class_not_allowed:{block_class or 'missing'}",
    )
    lower_detail = detail.lower().strip()
    _need(
        lower_detail
        and lower_detail not in GENERIC_BLOCK_REASONS
        and lower_detail != block_class.replace("_", " "),
        f"{arch}_attempt_error_not_precise",
    )
    if full:
        _need(list(pointer.get("end_nodes") or []) == YOLO11_RAW_END_NODES, f"{arch}_blocked_full_raw_end_nodes")
    return {
        "source": "immutable_terminal_attempt",
        "semantic_status": semantic,
        "error_class": block_class,
        "compiler_phase": str(pointer.get("compiler_phase") or ""),
        "reason": detail,
        "source_onnx_sha256": source_sha,
        "pointer": _artifact_identity(pointer_path, run),
        "immutable_receipt": _artifact_identity(immutable_path, run),
    }


def _deepx_success(
    run: Path,
    status_rel: Path,
    *,
    full: bool,
    expected_source_sha256: str,
) -> dict[str, Any]:
    status_path = _safe_file(run, status_rel, "deepx_status")
    status = _mapping(_json(run, status_rel, "deepx_status"), "deepx_status")
    output_path: Path | None = None
    if full:
        _need(status.get("schema") == "onnx-splitpoint/deepx-artifact-status", "deepx_full_schema")
        _need(str(status.get("status") or "") == "ok", "deepx_full_status")
        contracts = [dict(row) for row in list(status.get("contracts") or []) if isinstance(row, Mapping) and row.get("variant") == "full"]
        _need(len(contracts) == 1, "deepx_full_contract_count")
        contract = contracts[0]
        _need(_sha(contract.get("source_onnx_sha256"), "deepx_full_source") == expected_source_sha256, "deepx_full_source_mismatch")
        _sha(contract.get("build_onnx_sha256"), "deepx_full_build")
        attestation = _mapping(contract.get("endpoint_semantic_attestation"), "deepx_full_endpoint_attestation")
        _need(attestation.get("pass") is True, "deepx_full_endpoint_unattested")
        dxnn_rel = Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite" / "deepx" / "deepx_m1" / "full" / "model.dxnn"
        identity = _mapping(
            status.get("current_full_artifact_identity"),
            "deepx_full_current_artifact_identity",
        )
        _need(identity.get("status") == "verified", "deepx_full_identity_unverified")
        _need(
            identity.get("mode") == "v2796_current_bytes_sha256",
            "deepx_full_identity_mode",
        )
    else:
        _need(status.get("ok") is True and str(status.get("status") or "") in {"ok", "ready", "success"}, "deepx_b067_status")
        _need(str(status.get("build_status") or "").startswith("ready"), "deepx_b067_build_status")
        _need(str(status.get("cache_key") or "").strip(), "deepx_b067_cache_key")
        contract = _mapping(status.get("cache_contract"), "deepx_b067_cache_contract")
        _need(contract.get("case_id") == "b067", "deepx_b067_contract_case")
        _need(
            _sha(status.get("source_onnx_sha256"), "deepx_b067_source")
            == expected_source_sha256,
            "deepx_b067_source_mismatch",
        )
        _need(str(contract.get("compiler_identity") or "").strip(), "deepx_b067_compiler_identity")
        _sha(contract.get("calibration_manifest_identity"), "deepx_b067_calibration")
        case_root = (
            Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite" / "b067"
        )
        output_rel = case_root / str(status.get("output_contract") or "")
        output_path = _safe_file(run, output_rel, "deepx_b067_output_contract")
        output = _mapping(
            _json(run, output_rel, "deepx_b067_output_contract"),
            "deepx_b067_output_contract",
        )
        _need(output.get("case_id") == "b067", "deepx_b067_output_contract_case")
        _need(
            _sha(output.get("source_onnx_sha256"), "deepx_b067_output_source")
            == expected_source_sha256,
            "deepx_b067_output_source_mismatch",
        )
        dxnn_rel = case_root / str(status.get("dxnn_path") or "")
    dxnn = _safe_file(run, dxnn_rel, "deepx_dxnn")
    _need(dxnn.stat().st_size > 0, "deepx_dxnn_empty")
    dxnn_sha = _file_sha(dxnn)
    if full:
        _need(
            _sha(contract.get("artifact_sha256"), "deepx_full_contract_artifact")
            == dxnn_sha,
            "deepx_full_contract_artifact_mismatch",
        )
        _need(
            _sha(contract.get("suite_artifact_sha256"), "deepx_full_suite_artifact")
            == dxnn_sha,
            "deepx_full_suite_artifact_mismatch",
        )
        _need(
            _sha(identity.get("source_onnx_sha256"), "deepx_full_identity_source")
            == expected_source_sha256,
            "deepx_full_identity_source_mismatch",
        )
        _need(
            _sha(identity.get("artifact_sha256"), "deepx_full_identity_artifact")
            == dxnn_sha,
            "deepx_full_identity_artifact_mismatch",
        )
        _need(
            _sha(identity.get("suite_artifact_sha256"), "deepx_full_identity_suite")
            == dxnn_sha,
            "deepx_full_identity_suite_mismatch",
        )
        _need(
            int(identity.get("suite_artifact_size_bytes") or 0) == dxnn.stat().st_size,
            "deepx_full_identity_size_mismatch",
        )
    else:
        _need(
            _sha(status.get("dxnn_sha256"), "deepx_b067_status_artifact")
            == dxnn_sha,
            "deepx_b067_status_artifact_mismatch",
        )
        _need(
            int(status.get("dxnn_size_bytes") or 0) == dxnn.stat().st_size,
            "deepx_b067_status_size_mismatch",
        )
        _need(
            _sha(output.get("artifact_sha256"), "deepx_b067_output_artifact")
            == dxnn_sha,
            "deepx_b067_output_artifact_mismatch",
        )
        _need(
            int(output.get("artifact_size_bytes") or 0) == dxnn.stat().st_size,
            "deepx_b067_output_size_mismatch",
        )
        _need(
            _sha(status.get("output_contract_sha256"), "deepx_b067_output_contract_file")
            == _file_sha(output_path),
            "deepx_b067_output_contract_file_mismatch",
        )
    evidence = {
        "status": _artifact_identity(status_path, run),
        "artifact": _artifact_identity(dxnn, run),
        "source_onnx_sha256": expected_source_sha256,
    }
    if output_path is not None:
        evidence["output_contract"] = _artifact_identity(output_path, run)
    return evidence


def _deepx_block(
    run: Path,
    status_rel: Path,
    *,
    expected_source_sha256: str,
) -> dict[str, Any] | None:
    try:
        status_path = _safe_file(run, status_rel, "deepx_block_status")
        status = _mapping(_json(run, status_rel, "deepx_block_status"), "deepx_block_status")
    except GateError as exc:
        if "_missing:" in str(exc):
            return None
        raise
    state = str(status.get("status") or status.get("build_status") or "").strip().lower()
    reasons = [str(status.get(key) or "").strip() for key in ("error_class", "error", "message", "reason")]
    queue = status.get("queue") if isinstance(status.get("queue"), list) else []
    for raw in reversed(queue):
        if isinstance(raw, Mapping):
            state = str(raw.get("status") or state).strip().lower()
            reasons.extend(str(raw.get(key) or "").strip() for key in ("error_class", "error", "message", "reason"))
    reason = next((value for value in reasons if value), "")
    if state in {"", "ok", "ready", "ready_reused", "ready_built", "not_selected", "skipped"}:
        return None
    _need(state in ALLOWED_DEEPX_BLOCK_STATES, f"deepx_block_state_not_allowed:{state}")
    _need(
        _sha(status.get("source_onnx_sha256"), "deepx_block_source")
        == expected_source_sha256,
        "deepx_block_source_mismatch",
    )
    lower = reason.lower().strip()
    _need(
        lower
        and lower not in GENERIC_BLOCK_REASONS
        and "quality" not in lower,
        "deepx_block_reason_not_precise",
    )
    return {
        "source": "deepx_artifact_status",
        "status": state,
        "reason": reason,
        "source_onnx_sha256": expected_source_sha256,
        "evidence": _artifact_identity(status_path, run),
    }


@dataclass(frozen=True)
class PathSpec:
    key: str
    backend: str
    variant: str
    result_file: str
    receipt: str
    run_id: str
    arch: str = ""


SPECS: Sequence[PathSpec] = (
    PathSpec("hailo8_full", "hailo8", "full", "benchmark_results_hailo8_auto.json", "models/yolo11l/benchmark_set/legacy_suite/hailo/hailo8/full/hailo_hef_build_receipt.json", "hailo8", "hailo8"),
    PathSpec("hailo10h_full", "hailo10h", "full", "benchmark_results_hailo10_auto.json", "models/yolo11l/benchmark_set/legacy_suite/hailo/hailo10/full/hailo_hef_build_receipt.json", "hailo10", "hailo10h"),
    PathSpec("deepx_full", "deepx_m1", "full", "benchmark_results_deepx_m1_full_auto.json", "models/yolo11l/benchmark_set/deepx/deepx_artifact_status.json", "deepx_m1_full"),
    PathSpec("hailo8_b067_composed", "hailo8", "composed", "benchmark_results_hailo8_to_trt_auto.json", "models/yolo11l/benchmark_set/legacy_suite/b067/hailo/hailo8/part1/hailo_hef_build_receipt.json", "hailo8_to_trt", "hailo8"),
    PathSpec("hailo10h_b067_composed", "hailo10h", "composed", "benchmark_results_hailo10_to_trt_auto.json", "models/yolo11l/benchmark_set/legacy_suite/b067/hailo/hailo10/part1/hailo_hef_build_receipt.json", "hailo10_to_tensorrt", "hailo10h"),
    PathSpec("deepx_b067_composed", "deepx_m1", "composed", "benchmark_results_deepx_m1_to_tensorrt_auto.json", "models/yolo11l/benchmark_set/legacy_suite/b067/deepx/deepx_m1/part1/deepx_part1_artifact_status.json", "deepx_m1_to_tensorrt"),
)


def _classify(
    run: Path,
    spec: PathSpec,
    *,
    source_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    case_id = "full" if spec.variant == "full" else "b067"
    row = _case_row(run, spec.result_file, case_id)
    result_rel = Path("models") / MODEL_ID / "benchmark_results" / spec.result_file
    result_identity = None
    try:
        result_identity = _artifact_identity(
            _safe_file(run, result_rel, f"result_file_{spec.key}"), run
        )
    except GateError as exc:
        if "_missing:" not in str(exc):
            raise
    if row is not None:
        _need(str(row.get("run_id") or "") == spec.run_id, f"result_run_id_mismatch:{spec.key}")
    quality = _quality_annotation(row)
    full = spec.variant == "full"
    expected_source_sha = str(
        source_bindings["full_sha256" if full else "part1_sha256"]
    )
    receipt_rel = Path(spec.receipt)
    if _technical_success(row, spec.variant):
        evidence = (
            _hailo_success(
                run,
                receipt_rel,
                spec.arch,
                full=full,
                expected_source_sha256=expected_source_sha,
            )
            if spec.arch
            else _deepx_success(
                run,
                receipt_rel,
                full=full,
                expected_source_sha256=expected_source_sha,
            )
        )
        return {
            "backend": spec.backend,
            "variant": spec.variant,
            "case_id": case_id,
            "terminal_status": "success",
            "technical_classification": "runtime_measured_receipt_bound",
            "result_file": result_identity,
            "evidence": evidence,
            "quality_annotation": quality,
        }
    block = None
    if spec.arch:
        block = _hailo_block(
            run,
            receipt_rel.parent,
            spec.arch,
            full=full,
            expected_source_sha256=expected_source_sha,
        )
    else:
        block = _deepx_block(
            run,
            receipt_rel,
            expected_source_sha256=expected_source_sha,
        )
    if block is None:
        block = _technical_row_block(row, spec.variant)
    _need(block is not None, f"unclassified_terminal:{spec.key}")
    return {
        "backend": spec.backend,
        "variant": spec.variant,
        "case_id": case_id,
        "terminal_status": "blocked",
        "technical_classification": str(block.get("semantic_status") or block.get("status") or "technical_block"),
        "result_file": result_identity,
        "evidence": block,
        "quality_annotation": quality,
    }


def verify(*, gate_output: Path, expected_profile: Path, workflow_rc: int) -> dict[str, Any]:
    _need(
        not isinstance(workflow_rc, bool)
        and isinstance(workflow_rc, int)
        and workflow_rc in ALLOWED_WORKFLOW_RCS,
        f"workflow_rc_not_allowed:{workflow_rc!r}",
    )
    run = _find_run(gate_output)
    _validate_profile(run, expected_profile)
    release_identity = _validate_run_identity(run)
    source_manifest = _release_source_identity(expected_profile)
    artifact_closure, indexed = _verify_terminal_artifact_index(
        run,
        run_identity=release_identity,
    )
    source_bindings = _source_bindings(run)
    paths = {
        spec.key: _classify(
            run,
            spec,
            source_bindings=source_bindings,
        )
        for spec in SPECS
    }
    _need(set(paths) == {spec.key for spec in SPECS}, "terminal_scope_incomplete")
    success_count = sum(row["terminal_status"] == "success" for row in paths.values())
    blocked_count = sum(row["terminal_status"] == "blocked" for row in paths.values())
    _need(success_count + blocked_count == len(SPECS), "terminal_count_incomplete")
    _need(
        workflow_rc == 0 or blocked_count > 0,
        "workflow_rc_1_without_exact_technical_block",
    )
    _require_projected_identities_indexed(
        {"source_bindings": source_bindings, "paths": paths},
        indexed=indexed,
    )
    return {
        "schema": VERDICT_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "ok": True,
        "valid_terminal": True,
        "profile_id": PROFILE_ID,
        "version": RELEASE_VERSION,
        "build_id": BUILD_ID,
        "workflow_version": BUILD_ID,
        "release_identity": release_identity,
        "source_manifest_sha256": source_manifest["sha256"],
        "source_manifest": source_manifest,
        "artifact_index": artifact_closure["artifact_index"],
        "artifact_index_closure": artifact_closure,
        "model_id": MODEL_ID,
        "model_sha256": MODEL_SHA256,
        "run_dir": str(run),
        "workflow_rc": int(workflow_rc),
        "workflow_rc_policy": {
            "allowed": sorted(ALLOWED_WORKFLOW_RCS),
            "rc_0": "accepted_after_six_exact_terminal_classifications",
            "rc_1": "accepted_only_when_at_least_one_exact_technical_block_is_bound",
            "other": "rejected",
        },
        "quality_decision_used_for_terminal": False,
        "required_path_count": len(SPECS),
        "success_count": success_count,
        "blocked_count": blocked_count,
        "scope_complete": True,
        "source_bindings": source_bindings,
        "paths": paths,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-output", required=True, type=Path)
    parser.add_argument("--workflow-rc", required=True, type=int)
    parser.add_argument(
        "--expected-profile",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "profiles" / "yolo11l_v2796_r8b_full_b067_gate.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = args.gate_output.expanduser().resolve(strict=True)
        profile = args.expected_profile.expanduser().resolve(strict=True)
        verdict = verify(gate_output=output, expected_profile=profile, workflow_rc=args.workflow_rc)
        rc = 0
    except Exception as exc:
        verdict = {
            "schema": VERDICT_SCHEMA,
            "schema_version": 1,
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
