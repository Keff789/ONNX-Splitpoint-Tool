#!/usr/bin/env python3
"""Read-only verifier for the v2.78.3 YOLO11l Hailo Gate-A output.

The Evaluation Workflow deliberately reports ``partial`` (and therefore exits
with code 1) when ``workflow.stop_after`` is reached.  This verifier is the
fail-closed bridge between that generic workflow result and the narrower Gate-A
decision: a non-zero workflow status is accepted only when the archived run,
stage checkpoints and both receipt-bound Part-1 HEFs prove that the requested
stop was reached after a real common anchor was materialised.

The verifier never writes to the supplied Gate-A directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.benchmark.services import (  # noqa: E402
    _revalidate_hailo_feasibility_anchor_v2783,
    _validate_hailo_feasibility_resume_state_v2783,
    normalize_hailo_feasibility_control,
)
from onnx_splitpoint_tool.build_evidence import (  # noqa: E402
    FileObservation,
    _read_regular_nofollow,
)


PROFILE_ID = "yolo11l_v2783_hailo8_first_b5_gate_a"
MODEL_ID = "yolo11l"
RECEIPT_SCHEMA = "onnx-splitpoint/hailo8-first-feasibility-receipt/v1"
RUN_MANIFEST_SCHEMA = "onnx-splitpoint/evaluation-run-manifest"
BENCHMARK_CONTRACT_SCHEMA = "onnx-splitpoint/benchmark-set-contract"
EXPECTED_STOP_AFTER = "build_backend_artifacts"
EXPECTED_EXECUTION_MODE = "generate_benchmarksets"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_YAML_BYTES = 4 * 1024 * 1024
EXPECTED_ROOT_STAGES = ("resolve_profile", "campaign_preflight")
EXPECTED_MODEL_STAGES = (
    "resolve_model",
    "check_validation_assets",
    "prepare_model",
    "analyze_model",
    "select_split_candidates",
    "prepare_full_baselines",
    "generate_benchmark_set",
    "build_backend_artifacts",
)
FORBIDDEN_MODEL_STAGES = (
    "run_benchmarks",
    "validate_outputs",
    "hardware_smoke",
)
FORBIDDEN_ROOT_STAGES = (
    "evaluate_quality",
    "aggregate_results",
    "run_native_producers",
    "generate_report",
)


class GateVerificationError(ValueError):
    """One fail-closed Gate-A admission check failed."""


def _need(value: Any, reason: str) -> None:
    if not value:
        raise GateVerificationError(reason)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: Any, *, label: str) -> str:
    token = str(value or "").strip().lower().removeprefix("sha256:")
    _need(
        len(token) == 64 and all(character in "0123456789abcdef" for character in token),
        f"{label}_invalid_sha256",
    )
    return token


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    """Open every absolute path component without following a symlink."""

    token = os.fspath(path)
    _need(os.path.isabs(token), f"{label}_not_absolute")
    _need(os.path.normpath(token) == token, f"{label}_not_canonical")
    parts = Path(token).parts[1:]
    _need(all(part not in {"", ".", ".."} for part in parts), f"{label}_unsafe")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for part in parts:
            child = os.open(part, flags, dir_fd=fd)
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child)
                raise GateVerificationError(f"{label}_component_not_directory")
            os.close(fd)
            fd = child
        result = fd
        fd = -1
        return result
    except OSError as exc:
        raise GateVerificationError(
            f"{label}_unsafe_or_missing:{type(exc).__name__}"
        ) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _safe_relative(value: str, *, label: str) -> Path:
    token = str(value or "")
    relative = Path(token)
    _need(token != "", f"{label}_empty")
    _need(not relative.is_absolute(), f"{label}_absolute")
    _need(
        all(part not in {"", ".", ".."} for part in relative.parts),
        f"{label}_unsafe",
    )
    return relative


def _read_regular_under(
    root: Path,
    relative: str | Path,
    *,
    limit: int,
    label: str,
) -> bytes:
    return _observe_regular_under(
        root,
        relative,
        limit=limit,
        label=label,
    )[0]


def _observe_regular_under(
    root: Path,
    relative: str | Path,
    *,
    limit: int,
    label: str,
) -> tuple[bytes, FileObservation]:
    rel = _safe_relative(os.fspath(relative), label=label)
    try:
        observation = _read_regular_nofollow(
            root / rel,
            label=label,
            collect=True,
            size_limit=limit,
        )
    except Exception as exc:
        raise GateVerificationError(
            f"{label}_unsafe_or_missing:{type(exc).__name__}"
        ) from exc
    _need(observation.data is not None, f"{label}_read_failed")
    return observation.data, observation


def _read_json_under(root: Path, relative: str | Path, *, label: str) -> Any:
    raw = _read_regular_under(root, relative, limit=MAX_JSON_BYTES, label=label)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateVerificationError(f"{label}_invalid_json") from exc


def _read_yaml_under(root: Path, relative: str | Path, *, label: str) -> Any:
    raw = _read_regular_under(root, relative, limit=MAX_YAML_BYTES, label=label)
    try:
        return yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise GateVerificationError(f"{label}_invalid_yaml") from exc


def _read_external_yaml(path: Path, *, label: str) -> Mapping[str, Any]:
    _need(path.name == os.fspath(path.relative_to(path.parent)), f"{label}_unsafe_name")
    raw = _read_regular_under(
        path.parent,
        path.name,
        limit=MAX_YAML_BYTES,
        label=label,
    )
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise GateVerificationError(f"{label}_invalid_yaml") from exc
    _need(isinstance(payload, Mapping), f"{label}_not_mapping")
    return dict(payload)


def _find_single_run_dir(gate_output: Path) -> Path:
    root_fd = _open_directory_nofollow(gate_output, label="gate_output")
    os.close(root_fd)
    candidates: list[Path] = []
    with os.scandir(gate_output) as entries:
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
            child = gate_output / entry.name
            try:
                _read_regular_under(
                    child,
                    "run_manifest.json",
                    limit=MAX_JSON_BYTES,
                    label="candidate_run_manifest",
                )
            except GateVerificationError:
                continue
            candidates.append(child)
    _need(len(candidates) == 1, f"expected_one_run_dir_found_{len(candidates)}")
    return candidates[0]


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), f"{label}_not_mapping")
    return dict(value)


def _validate_profile(
    *,
    run_dir: Path,
    expected_profile: Mapping[str, Any],
) -> dict[str, Any]:
    source = _read_yaml_under(
        run_dir, "profile_source.yaml", label="profile_source"
    )
    resolved = _read_yaml_under(run_dir, "profile.yaml", label="profile")
    _need(isinstance(source, Mapping), "profile_source_not_mapping")
    _need(isinstance(resolved, Mapping), "profile_not_mapping")
    _need(dict(source) == dict(expected_profile), "profile_source_mismatch")
    for label, payload in (("source", source), ("resolved", resolved)):
        _need(str(payload.get("name") or "") == PROFILE_ID, f"{label}_profile_id")
        workflow = _mapping(payload.get("workflow"), label=f"{label}_workflow")
        _need(
            workflow.get("stop_after") == EXPECTED_STOP_AFTER,
            f"{label}_stop_after_mismatch",
        )
        _need(
            workflow.get("execution_mode") == EXPECTED_EXECUTION_MODE,
            f"{label}_execution_mode_mismatch",
        )
        _need(workflow.get("only_model") == MODEL_ID, f"{label}_only_model")
        _need(workflow.get("max_models") == 1, f"{label}_max_models")
        hailo = _mapping(payload.get("hailo_build"), label=f"{label}_hailo_build")
        _need(hailo.get("build_full") is False, f"{label}_hailo_full_enabled")
        _need(hailo.get("build_part1") is True, f"{label}_hailo_part1_disabled")
        _need(hailo.get("build_part2") is False, f"{label}_hailo_part2_enabled")
        _need(
            list(hailo.get("targets") or []) == ["hailo8", "hailo10"],
            f"{label}_hailo_targets",
        )
        control = _mapping(
            hailo.get("feasibility_control"), label=f"{label}_feasibility_control"
        )
        _need(control.get("enabled") is True, f"{label}_gate_disabled")
        _need(
            control.get("mode") == "hailo8_first_common_anchor",
            f"{label}_gate_mode",
        )
    return dict(resolved)


def _validate_stage_result(
    *,
    run_dir: Path,
    manifest_row: Mapping[str, Any],
    stage: str,
    model_id: str | None,
    allowed_statuses: Sequence[str] = ("ok",),
    allowed_states: Sequence[str] = ("completed",),
) -> dict[str, Any]:
    expected_rel = (
        Path("models") / model_id / "stages" / stage / "stage_result.json"
        if model_id
        else Path("stages") / stage / "stage_result.json"
    )
    _need(
        str(manifest_row.get("stage_result_path") or "")
        == expected_rel.as_posix(),
        f"stage_result_path_mismatch:{model_id or 'root'}:{stage}",
    )
    result = _read_json_under(
        run_dir,
        expected_rel,
        label=f"stage_result_{model_id or 'root'}_{stage}",
    )
    result = _mapping(result, label=f"stage_result_{model_id or 'root'}_{stage}")
    _need(result.get("stage") == stage, f"stage_identity_mismatch:{stage}")
    observed_model = str(result.get("model_id") or "")
    _need(observed_model == str(model_id or ""), f"stage_model_mismatch:{stage}")
    _need(
        result.get("status") in set(allowed_statuses),
        f"stage_not_allowed_status:{stage}:{result.get('status')}",
    )
    _need(
        result.get("state") in set(allowed_states),
        f"stage_not_allowed_state:{stage}:{result.get('state')}",
    )
    _need(result.get("complete") is True, f"stage_incomplete:{stage}")
    projected = dict(manifest_row)
    projected.pop("stage_result_path", None)
    _need(projected == result, f"stage_manifest_checkpoint_mismatch:{stage}")
    return result


def _manifest_header(
    *,
    run_dir: Path,
    workflow_rc: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _read_json_under(run_dir, "run_manifest.json", label="run_manifest")
    manifest = _mapping(manifest, label="run_manifest")
    _need(manifest.get("schema") == RUN_MANIFEST_SCHEMA, "run_manifest_schema")
    _need(manifest.get("schema_version") == 1, "run_manifest_schema_version")
    _need(manifest.get("profile_id") == PROFILE_ID, "run_manifest_profile_id")
    _need(manifest.get("run_id") == run_dir.name, "run_manifest_run_id")
    _need(
        os.path.normpath(str(manifest.get("run_dir") or "")) == os.fspath(run_dir),
        "run_manifest_run_dir",
    )
    options = _mapping(manifest.get("options"), label="run_manifest_options")
    _need(options.get("stop_after") == EXPECTED_STOP_AFTER, "manifest_stop_after")
    _need(
        options.get("execution_mode") == EXPECTED_EXECUTION_MODE,
        "manifest_execution_mode",
    )
    _need(workflow_rc in {0, 1}, f"unexpected_workflow_rc:{workflow_rc}")

    root_stages = _mapping(manifest.get("root_stages"), label="root_stages")
    models = _mapping(manifest.get("models"), label="models")
    _need(set(models) == {MODEL_ID}, "run_manifest_model_set")
    model = _mapping(models.get(MODEL_ID), label="model_manifest")
    _need(model.get("model_id") == MODEL_ID, "model_manifest_identity")
    model_stages = _mapping(model.get("stages"), label="model_stages")

    return manifest, root_stages, model_stages


def _validate_manifest_and_stages(
    *,
    run_dir: Path,
    workflow_rc: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, root_stages, model_stages = _manifest_header(
        run_dir=run_dir,
        workflow_rc=workflow_rc,
    )
    # An accepted stop_after run has exactly the root and model prefix through
    # build_backend_artifacts.  This prevents an unrelated failure/later stage
    # from being relabelled as the expected partial workflow result.
    _need(set(root_stages) == set(EXPECTED_ROOT_STAGES), "root_stage_set_mismatch")
    _need(set(model_stages) == set(EXPECTED_MODEL_STAGES), "model_stage_set_mismatch")
    _need(
        not any(stage in root_stages for stage in FORBIDDEN_ROOT_STAGES),
        "forbidden_root_stage_present",
    )
    _need(
        not any(stage in model_stages for stage in FORBIDDEN_MODEL_STAGES),
        "forbidden_model_stage_present",
    )
    for stage in EXPECTED_ROOT_STAGES:
        _validate_stage_result(
            run_dir=run_dir,
            manifest_row=_mapping(root_stages[stage], label=f"root_stage_{stage}"),
            stage=stage,
            model_id=None,
        )
    stage_results: dict[str, Any] = {}
    for stage in EXPECTED_MODEL_STAGES:
        stage_results[stage] = _validate_stage_result(
            run_dir=run_dir,
            manifest_row=_mapping(model_stages[stage], label=f"model_stage_{stage}"),
            stage=stage,
            model_id=MODEL_ID,
        )
    return manifest, stage_results


def _validate_budget_manifest_and_stages(
    *,
    run_dir: Path,
    workflow_rc: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, root_stages, model_stages = _manifest_header(
        run_dir=run_dir,
        workflow_rc=workflow_rc,
    )
    _need(workflow_rc == 1, "budget_terminal_requires_workflow_rc1")
    _need(
        manifest.get("status") in {"failed", "partial"},
        "budget_manifest_status",
    )
    _need(set(root_stages) == set(EXPECTED_ROOT_STAGES), "budget_root_stage_set")
    budget_model_stages = EXPECTED_MODEL_STAGES[:-1]
    _need(set(model_stages) == set(budget_model_stages), "budget_model_stage_set")
    for stage in EXPECTED_ROOT_STAGES:
        _validate_stage_result(
            run_dir=run_dir,
            manifest_row=_mapping(root_stages[stage], label=f"budget_root_{stage}"),
            stage=stage,
            model_id=None,
        )
    stage_results: dict[str, Any] = {}
    for stage in budget_model_stages[:-1]:
        stage_results[stage] = _validate_stage_result(
            run_dir=run_dir,
            manifest_row=_mapping(model_stages[stage], label=f"budget_model_{stage}"),
            stage=stage,
            model_id=MODEL_ID,
        )
    generate = "generate_benchmark_set"
    stage_results[generate] = _validate_stage_result(
        run_dir=run_dir,
        manifest_row=_mapping(model_stages[generate], label="budget_generate"),
        stage=generate,
        model_id=MODEL_ID,
        allowed_statuses=("failed", "partial"),
        allowed_states=("failed", "completed"),
    )
    return manifest, stage_results


def _state_without_revalidation_marker(value: Mapping[str, Any]) -> dict[str, Any]:
    body = json.loads(json.dumps(dict(value), allow_nan=False))
    body.pop("resume_anchor_revalidation", None)
    return body


def _validate_receipt(
    *,
    run_dir: Path,
    resolved_profile: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_rel = Path("models") / MODEL_ID / "benchmark_set" / "hailo_feasibility_receipt.json"
    receipt = _read_json_under(run_dir, receipt_rel, label="hailo_feasibility_receipt")
    receipt = _mapping(receipt, label="hailo_feasibility_receipt")
    _need(receipt.get("schema") == RECEIPT_SCHEMA, "receipt_schema")
    _need(receipt.get("schema_version") == 1, "receipt_schema_version")
    _need(receipt.get("model_id") == MODEL_ID, "receipt_model_id")
    _need(receipt.get("profile_id") == PROFILE_ID, "receipt_profile_id")
    _need(receipt.get("run_id") == run_dir.name, "receipt_run_id")
    _need(receipt.get("fallback_allowed") is False, "receipt_fallback_allowed")
    _need(not str(receipt.get("validation_error") or ""), "receipt_validation_error")
    outcome = str(receipt.get("outcome") or "")
    _need(
        outcome in {"ANCHOR_FOUND", "CANARY_BUDGET_EXHAUSTED"},
        f"receipt_nonterminal_outcome:{outcome or 'missing'}",
    )
    state = _mapping(receipt.get("state"), label="receipt_state")
    _need(state.get("outcome") == outcome, "receipt_state_outcome")
    order = receipt.get("candidate_order")
    _need(
        isinstance(order, list)
        and bool(order)
        and all(type(value) is int and value >= 0 for value in order),
        "receipt_candidate_order",
    )
    order_sha = _canonical_sha256(order)
    _need(receipt.get("candidate_order_sha256") == order_sha, "receipt_order_hash")
    _need(state.get("candidate_order") == order, "receipt_state_order")
    _need(state.get("candidate_order_sha256") == order_sha, "state_order_hash")

    expected_hailo = _mapping(
        resolved_profile.get("hailo_build"), label="resolved_hailo_build"
    )
    expected_control = normalize_hailo_feasibility_control(
        _mapping(
            expected_hailo.get("feasibility_control"),
            label="resolved_feasibility_control",
        )
    )
    expected_backend = str(expected_hailo.get("backend") or "").strip()
    _need(
        str(state.get("backend") or "").strip() == expected_backend,
        "receipt_state_backend_mismatch",
    )
    _digest(
        state.get("full_source_onnx_sha256"),
        label="receipt_state_full_source_onnx",
    )
    # build_run_plan() sorts the physical target axis.  Dispatch remains
    # Hailo-8-first through primary_target; this persisted list is the
    # run-plan/manifest order, not compiler dispatch order.
    state_targets = state.get("targets")
    _need(
        state_targets == ["hailo10", "hailo8"],
        "receipt_state_target_order_mismatch",
    )
    validated = _validate_hailo_feasibility_resume_state_v2783(
        state,
        control=expected_control,
        candidate_order=order,
        targets=state_targets,
        backend=expected_backend,
        full_source_onnx_sha256=str(state.get("full_source_onnx_sha256") or ""),
    )
    _need(validated.get("outcome") == outcome, "validated_state_outcome")
    if outcome == "ANCHOR_FOUND":
        _need(receipt.get("stop_workflow") is False, "anchor_receipt_stops_workflow")
    else:
        _need(receipt.get("stop_workflow") is True, "budget_receipt_did_not_stop")
    return receipt, state


def _validate_anchor_material(
    *,
    run_dir: Path,
    state: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    boundary = state.get("anchor_boundary")
    _need(type(boundary) is int, "anchor_boundary_invalid")
    boundary = int(boundary)
    suite_rel = Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite"
    suite_dir = run_dir / suite_rel
    suite_fd = _open_directory_nofollow(suite_dir, label="legacy_suite")
    os.close(suite_fd)
    suite_payload = _read_json_under(
        run_dir, suite_rel / "benchmark_set.json", label="legacy_benchmark_set"
    )
    suite_payload = _mapping(suite_payload, label="legacy_benchmark_set")
    cases_raw = suite_payload.get("cases") or suite_payload.get("accepted_cases") or []
    _need(isinstance(cases_raw, list), "legacy_cases_not_list")
    cases = [dict(row) for row in cases_raw if isinstance(row, Mapping)]
    matching = [
        row
        for row in cases
        if type(row.get("boundary")) is int and int(row["boundary"]) == boundary
    ]
    _need(len(cases) == 1 and len(matching) == 1, "anchor_case_set_mismatch")

    generation = _read_json_under(
        run_dir, suite_rel / "generation_state.json", label="generation_state"
    )
    generation = _mapping(generation, label="generation_state")
    completed = {
        int(value)
        for value in list(generation.get("completed_boundaries") or [])
        if type(value) is int
    }
    accepted = {
        int(value)
        for value in list(generation.get("accepted_boundaries") or [])
        if type(value) is int
    }
    _need(boundary in completed and boundary in accepted, "anchor_not_completed_accepted")
    persisted_state = generation.get("hailo_feasibility_state")
    _need(isinstance(persisted_state, Mapping), "generation_gate_state_missing")
    _need(
        _state_without_revalidation_marker(persisted_state)
        == _state_without_revalidation_marker(state),
        "receipt_generation_state_mismatch",
    )
    _need(
        _revalidate_hailo_feasibility_anchor_v2783(
            state,
            out_dir=suite_dir,
            cases=cases,
            completed_boundaries=completed,
            accepted_boundaries=accepted,
            targets=["hailo10", "hailo8"],
        ),
        "anchor_material_revalidation_failed",
    )

    formal = _read_json_under(
        run_dir,
        Path("models") / MODEL_ID / "benchmark_set" / "benchmark_set.json",
        label="formal_benchmark_contract",
    )
    formal = _mapping(formal, label="formal_benchmark_contract")
    _need(formal.get("schema") == BENCHMARK_CONTRACT_SCHEMA, "formal_contract_schema")
    _need(formal.get("profile_id") == PROFILE_ID, "formal_contract_profile")
    _need(formal.get("run_id") == run_dir.name, "formal_contract_run")
    _need(formal.get("model_id") == MODEL_ID, "formal_contract_model")
    _need(formal.get("status") == "ok", "formal_contract_status")
    _need(formal.get("materialized") is True, "formal_contract_not_materialized")
    _need(formal.get("hailo_feasibility_outcome") == "ANCHOR_FOUND", "formal_contract_outcome")
    _need(formal.get("fallback_allowed") is False, "formal_contract_fallback")
    formal_cases = formal.get("cases")
    _need(isinstance(formal_cases, list) and formal_cases == cases, "formal_case_binding")

    case = matching[0]
    folder = _safe_relative(
        str(case.get("case_dir") or case.get("folder") or ""),
        label="anchor_case_folder",
    )
    manifest_name = _safe_relative(
        str(case.get("manifest") or "split_manifest.json"),
        label="anchor_split_manifest_name",
    )
    _need(len(manifest_name.parts) == 1, "anchor_manifest_not_case_local")
    split = _read_json_under(
        run_dir,
        suite_rel / folder / manifest_name,
        label="anchor_split_manifest",
    )
    split = _mapping(split, label="anchor_split_manifest")
    _need(int(split.get("boundary", -1)) == boundary, "split_manifest_boundary")
    hailo = _mapping(split.get("hailo"), label="split_manifest_hailo")
    hefs = _mapping(hailo.get("hefs"), label="split_manifest_hefs")
    _need(set(hefs) == {"hailo8", "hailo10"}, "split_manifest_target_set")
    evidence: dict[str, Any] = {}
    inodes: set[tuple[int, int]] = set()
    observed_paths: set[str] = set()
    for target, expected_arch in (("hailo8", "hailo8"), ("hailo10", "hailo10h")):
        meta = _mapping(hefs.get(target), label=f"split_manifest_{target}")
        part1 = _safe_relative(str(meta.get("part1") or ""), label=f"{target}_part1")
        expected_rel = Path("hailo") / target / "part1" / "compiled.hef"
        _need(part1 == expected_rel, f"{target}_part1_path")
        hef_rel = suite_rel / folder / part1
        hef_bytes, info = _observe_regular_under(
            run_dir,
            hef_rel,
            limit=64 * 1024 * 1024,
            label=f"{target}_hef",
        )
        inode = (info.device, info.inode)
        _need(inode not in inodes, "anchor_hefs_share_inode")
        inodes.add(inode)
        observed_paths.add(hef_rel.as_posix())
        receipt_rel = hef_rel.parent / "hailo_hef_build_receipt.json"
        build_receipt = _read_json_under(
            run_dir, receipt_rel, label=f"{target}_build_receipt"
        )
        build_receipt = _mapping(build_receipt, label=f"{target}_build_receipt")
        _need(
            str(build_receipt.get("hw_arch") or "").lower().replace("-", "")
            == expected_arch,
            f"{target}_receipt_arch",
        )
        actual_sha = hashlib.sha256(hef_bytes).hexdigest()
        _need(build_receipt.get("hef_sha256") == actual_sha, f"{target}_hef_hash")
        _need(
            build_receipt.get("hef_size_bytes") == info.size_bytes,
            f"{target}_hef_size",
        )
        evidence[target] = {
            "hef_relative_path": hef_rel.as_posix(),
            "hef_sha256": actual_sha,
            "hef_size_bytes": info.size_bytes,
            "receipt_relative_path": receipt_rel.as_posix(),
            "hw_arch": str(build_receipt.get("hw_arch") or ""),
            "cache_key": str(build_receipt.get("cache_key") or ""),
        }

    # The accepted anchor itself may contain Part-2 ONNX files, but exactly the
    # two declared Part-1 HEFs are admitted.  No Full/Part-2 HEF can hide next
    # to them.
    case_dir = suite_dir / folder
    found_hefs: set[str] = set()
    for directory, subdirs, files in os.walk(case_dir, followlinks=False):
        for name in list(subdirs) + list(files):
            _need(not os.path.islink(os.path.join(directory, name)), "anchor_case_symlink")
        for name in files:
            if name != "compiled.hef":
                continue
            path = Path(directory) / name
            found_hefs.add(path.relative_to(run_dir).as_posix())
    _need(found_hefs == observed_paths, "unexpected_anchor_hef_set")
    return boundary, evidence


def _validate_budget_material(
    *,
    run_dir: Path,
    state: Mapping[str, Any],
) -> None:
    """Bind a fail-closed exhaustion receipt to its non-materialized suite."""

    _need(
        state.get("exhaustion_reason")
        in {"candidate_pool_exhausted", "wall_time_budget_exhausted"},
        "budget_exhaustion_reason",
    )
    order = [int(value) for value in list(state.get("candidate_order") or [])]
    candidate_rows = state.get("candidates")
    _need(
        isinstance(candidate_rows, list) and bool(candidate_rows),
        "budget_candidate_ledger_empty",
    )
    candidate_boundaries: list[int] = []
    for index, raw in enumerate(candidate_rows):
        row = _mapping(raw, label=f"budget_candidate_{index}")
        boundary = row.get("boundary")
        _need(
            type(boundary) is int and int(boundary) in set(order),
            "budget_candidate_outside_order",
        )
        _need(row.get("anchor") is False, "budget_candidate_claims_anchor")
        _need(
            isinstance(row.get("phases"), list),
            "budget_candidate_phases_invalid",
        )
        target_outcomes = _mapping(
            row.get("target_outcomes"),
            label=f"budget_candidate_{index}_target_outcomes",
        )
        _need(
            set(target_outcomes) == {"hailo8", "hailo10"},
            "budget_candidate_target_set",
        )
        _need(
            not all(value == "ARTIFACT_PASS" for value in target_outcomes.values()),
            "budget_candidate_is_hidden_anchor",
        )
        candidate_boundaries.append(int(boundary))
    _need(
        len(candidate_boundaries) == len(set(candidate_boundaries)),
        "budget_candidate_boundary_duplicate",
    )
    attempts = _mapping(
        state.get("attempts_by_boundary"), label="budget_attempts_by_boundary"
    )
    _need(
        set(attempts) == {str(value) for value in candidate_boundaries},
        "budget_attempt_ledger_mismatch",
    )

    suite_rel = Path("models") / MODEL_ID / "benchmark_set" / "legacy_suite"
    suite_payload = _mapping(
        _read_json_under(
            run_dir,
            suite_rel / "benchmark_set.json",
            label="budget_legacy_benchmark_set",
        ),
        label="budget_legacy_benchmark_set",
    )
    _need(
        "cases" in suite_payload or "accepted_cases" in suite_payload,
        "budget_legacy_case_field_missing",
    )
    _need(
        list(suite_payload.get("cases") or []) == []
        and list(suite_payload.get("accepted_cases") or []) == [],
        "budget_legacy_has_accepted_case",
    )

    generation = _mapping(
        _read_json_under(
            run_dir,
            suite_rel / "generation_state.json",
            label="budget_generation_state",
        ),
        label="budget_generation_state",
    )
    _need(generation.get("status") == "partial", "budget_generation_status")
    _need(
        list(generation.get("accepted_boundaries") or []) == [],
        "budget_generation_has_accepted_boundary",
    )
    completed = {
        int(value)
        for value in list(generation.get("completed_boundaries") or [])
        if type(value) is int
    }
    _need(
        completed == set(candidate_boundaries),
        "budget_completed_boundary_mismatch",
    )
    persisted_state = generation.get("hailo_feasibility_state")
    _need(isinstance(persisted_state, Mapping), "budget_generation_gate_state_missing")
    _need(
        _state_without_revalidation_marker(persisted_state)
        == _state_without_revalidation_marker(state),
        "budget_receipt_generation_state_mismatch",
    )

    full_source = _read_regular_under(
        run_dir,
        suite_rel / "models" / f"{MODEL_ID}.onnx",
        limit=2 * 1024 * 1024 * 1024,
        label="budget_full_source_onnx",
    )
    _need(
        hashlib.sha256(full_source).hexdigest()
        == _digest(
            state.get("full_source_onnx_sha256"),
            label="budget_full_source_onnx",
        ),
        "budget_full_source_onnx_mismatch",
    )

    formal = _mapping(
        _read_json_under(
            run_dir,
            Path("models") / MODEL_ID / "benchmark_set" / "benchmark_set.json",
            label="budget_formal_benchmark_contract",
        ),
        label="budget_formal_benchmark_contract",
    )
    _need(formal.get("schema") == BENCHMARK_CONTRACT_SCHEMA, "budget_contract_schema")
    _need(formal.get("profile_id") == PROFILE_ID, "budget_contract_profile")
    _need(formal.get("run_id") == run_dir.name, "budget_contract_run")
    _need(formal.get("model_id") == MODEL_ID, "budget_contract_model")
    _need(formal.get("status") == "failed", "budget_contract_status")
    _need(formal.get("materialized") is False, "budget_contract_materialized")
    _need(
        formal.get("hailo_feasibility_outcome") == "CANARY_BUDGET_EXHAUSTED",
        "budget_contract_outcome",
    )
    _need(formal.get("fallback_allowed") is False, "budget_contract_fallback")
    _need(list(formal.get("cases") or []) == [], "budget_contract_has_cases")


def verify_gate_a_output(
    *,
    gate_output: Path,
    workflow_rc: int,
    expected_profile_path: Path,
) -> dict[str, Any]:
    gate_output = Path(os.path.abspath(os.path.normpath(os.fspath(gate_output))))
    expected_profile_path = Path(
        os.path.abspath(os.path.normpath(os.fspath(expected_profile_path)))
    )
    verdict: dict[str, Any] = {
        "schema": "onnx-splitpoint/v2783-yolo11-gate-a-output-verdict/v1",
        "schema_version": 1,
        "status": "INVALID",
        "ok": False,
        "valid_terminal": False,
        "outcome": "INVALID",
        "anchor_boundary": None,
        "expected_stop_after": EXPECTED_STOP_AFTER,
        "expected_stop_after_partial": False,
        "workflow_rc": int(workflow_rc),
        "gate_output": str(gate_output),
        "run_dir": "",
        "profile_id": PROFILE_ID,
        "model_id": MODEL_ID,
        "full_source_onnx_sha256": "",
        "artifacts": {},
        "errors": [],
    }
    try:
        expected_profile = _read_external_yaml(
            expected_profile_path, label="expected_profile"
        )
        run_dir = _find_single_run_dir(gate_output)
        verdict["run_dir"] = str(run_dir)
        resolved_profile = _validate_profile(
            run_dir=run_dir, expected_profile=expected_profile
        )
        receipt, state = _validate_receipt(
            run_dir=run_dir, resolved_profile=resolved_profile
        )
        outcome = str(receipt.get("outcome") or "")
        verdict["outcome"] = outcome
        full_source_sha256 = _digest(
            state.get("full_source_onnx_sha256"),
            label="receipt_state_full_source_onnx",
        )
        verdict["full_source_onnx_sha256"] = full_source_sha256
        receipt_relative = (
            Path("models")
            / MODEL_ID
            / "benchmark_set"
            / "hailo_feasibility_receipt.json"
        ).as_posix()
        if outcome == "CANARY_BUDGET_EXHAUSTED":
            manifest, stage_results = _validate_budget_manifest_and_stages(
                run_dir=run_dir, workflow_rc=int(workflow_rc)
            )
            _need(
                manifest.get("technical_status") in {"failed", "partial"},
                "budget_technical_status",
            )
            _need(
                manifest.get("quality_decision") == "not_evaluated",
                "budget_quality_was_evaluated",
            )
            generate_artifacts = set(
                stage_results["generate_benchmark_set"].get("artifacts") or []
            )
            _need(
                receipt_relative in generate_artifacts,
                "budget_receipt_not_bound_to_generation_stage",
            )
            _validate_budget_material(run_dir=run_dir, state=state)
            verdict.update(
                {
                    "status": "BLOCKED",
                    "valid_terminal": True,
                    "errors": ["canary_budget_exhausted"],
                }
            )
            return verdict

        manifest, stage_results = _validate_manifest_and_stages(
            run_dir=run_dir, workflow_rc=int(workflow_rc)
        )
        _need(manifest.get("status") == "partial", "run_manifest_not_partial")
        _need(manifest.get("technical_status") == "partial", "technical_status_not_partial")
        _need(
            manifest.get("quality_decision") == "not_evaluated",
            "quality_was_evaluated_in_gate_a",
        )
        generate_artifacts = set(stage_results["generate_benchmark_set"].get("artifacts") or [])
        _need(receipt_relative in generate_artifacts, "receipt_not_bound_to_generation_stage")
        boundary, artifacts = _validate_anchor_material(run_dir=run_dir, state=state)
        verdict.update(
            {
                "status": "PASS",
                "ok": True,
                "valid_terminal": True,
                "anchor_boundary": boundary,
                "full_source_onnx_sha256": full_source_sha256,
                "expected_stop_after_partial": True,
                "artifacts": artifacts,
                "errors": [],
            }
        )
        return verdict
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        verdict["errors"] = [reason]
        return verdict


def _shell_value(value: Any) -> str:
    return str(value).replace("\n", " ").replace("\r", " ")


def _print_shell(verdict: Mapping[str, Any]) -> None:
    ok = verdict.get("ok") is True
    outcome = str(verdict.get("outcome") or "INVALID")
    boundary = verdict.get("anchor_boundary")
    errors = list(verdict.get("errors") or [])
    print(f"GATE_A_VERIFICATION={'PASS' if ok else verdict.get('status', 'INVALID')}")
    print(f"GATE_A_OUTCOME={_shell_value(outcome)}")
    print(f"ANCHOR_FOUND={'YES' if ok else 'NO'}")
    if ok and type(boundary) is int:
        print(f"ANCHOR_BOUNDARY=b{int(boundary)}")
        print(
            "FULL_SOURCE_ONNX_SHA256="
            f"{_shell_value(verdict.get('full_source_onnx_sha256') or '')}"
        )
    print(
        "CANARY_BUDGET_EXHAUSTED="
        + (
            "YES"
            if outcome == "CANARY_BUDGET_EXHAUSTED"
            else "NO" if outcome == "ANCHOR_FOUND" else "UNKNOWN"
        )
    )
    print(
        "EXPECTED_STOP_AFTER_PARTIAL="
        + ("PASS" if verdict.get("expected_stop_after_partial") is True else "NO")
    )
    print(f"WORKFLOW_RC={int(verdict.get('workflow_rc') or 0)}")
    print(f"B5_BLOCKED={'NO' if ok else 'YES'}")
    print("B5_LAUNCHED=NO")
    print("RUNTIME_RUN=NO")
    print("NATIVE_RUN=NO")
    print("ENERGY_RUN=NO")
    print(f"GATE_A_OUTPUT={_shell_value(verdict.get('gate_output') or '')}")
    print(f"GATE_A_RUN_DIR={_shell_value(verdict.get('run_dir') or '')}")
    if errors:
        print(f"GATE_A_ERROR={_shell_value(errors[0])}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-output", required=True, type=Path)
    parser.add_argument("--workflow-rc", required=True, type=int)
    parser.add_argument(
        "--expected-profile",
        type=Path,
        default=ROOT / "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml",
    )
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    ns = parser.parse_args(argv)
    verdict = verify_gate_a_output(
        gate_output=ns.gate_output,
        workflow_rc=ns.workflow_rc,
        expected_profile_path=ns.expected_profile,
    )
    if ns.format == "shell":
        _print_shell(verdict)
    else:
        print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    if verdict.get("ok") is True:
        return 0
    if verdict.get("valid_terminal") is True:
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
