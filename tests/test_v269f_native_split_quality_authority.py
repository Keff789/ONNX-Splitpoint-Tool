from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_split_quality_authority import (
    CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
    apply_native_split_quality_authority,
    canonical_native_split_backend,
    is_native_split_backend,
    resolve_native_split_quality_authority,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy


ROOT = Path(__file__).resolve().parents[1]
FULL_SNAPSHOT_SHA = "a" * 64
SELECTION_SHA = "b" * 64


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("hailo8_to_trt", "hailo8_to_trt"),
        ("hailo8_to_tensorrt", "hailo8_to_trt"),
        ("hailo10_to_trt", "hailo10h_to_trt"),
        ("hailo10h_to_trt", "hailo10h_to_trt"),
        ("hailo10_to_tensorrt", "hailo10h_to_trt"),
        ("hailo10h_to_tensorrt", "hailo10h_to_trt"),
        ("deepx_to_trt", "deepx_to_trt"),
        ("deepx_to_tensorrt", "deepx_to_trt"),
        ("deepx_m1_to_trt", "deepx_to_trt"),
        ("deepx_m1_to_tensorrt", "deepx_to_trt"),
    ],
)
def test_every_supported_split_alias_is_authoritatively_managed(
    alias: str, canonical: str,
) -> None:
    assert is_native_split_backend(alias) is True
    assert canonical_native_split_backend(alias) == canonical


@pytest.mark.parametrize(
    ("script_name", "payload"),
    [
        (
            "native_producer_validate_visualize.py",
            '{"schema":"central","results":[],"results":[]}',
        ),
        (
            "native_producer_validate_visualize.py",
            '{"schema":"central","results":[],"results":[{}]}',
        ),
        (
            "native_producer_final_report.py",
            '{"schema":"validation","rows":[],"rows":[]}',
        ),
        (
            "native_producer_final_report.py",
            '{"schema":"validation","rows":[],"rows":[{}]}',
        ),
    ],
)
def test_claim_critical_summary_load_rejects_every_duplicate_key(
    tmp_path: Path, script_name: str, payload: str,
) -> None:
    module = _load_script(script_name)
    path = tmp_path / script_name.replace('.py', '.json')
    path.write_text(payload, encoding='utf-8')
    assert module._load_strict_json_object(path) is None


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v269f_authority_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_current_context(run: Path, *, selection_sha: str = SELECTION_SHA) -> None:
    (run / "reports").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "current_workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "tool_version": "2.69.6",
        "current_tool_version": "2.69.6",
        "execution_sessions": [{
            "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
            "tool_version": "2.69.6",
        }],
        "profile_start_snapshot": {
            "snapshot_sha256": "sha256:" + FULL_SNAPSHOT_SHA,
            "requested_selection": {"snapshot_sha256": SELECTION_SHA},
            "resolved_selection": {"snapshot_sha256": selection_sha},
        },
    }
    stage = {
        "schema": "onnx-splitpoint/native-producer-stage",
        "schema_version": 3,
        "run_id": run.name,
        "workflow_version": CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW,
        "tool_version": "2.69.6",
        "profile_start_snapshot_sha256": FULL_SNAPSHOT_SHA,
        "profile_selection_snapshot_sha256": SELECTION_SHA,
        "native_split_quality_first": {"required": True},
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "reports" / "native_producer_stage.json").write_text(
        json.dumps(stage), encoding="utf-8",
    )


def _authority(run: Path) -> dict[str, object]:
    return resolve_native_split_quality_authority(
        run_manifest_path=run / "run_manifest.json",
        stage_path=run / "reports" / "native_producer_stage.json",
    )


def _legacy_row_and_result(policy: AccuracyGatePolicy):
    endpoint = "1" * 64
    provenance = {
        "source_request_sha256": "2" * 64,
        "model_sha256": "3" * 64,
        "validation_dataset_sha256": "4" * 64,
        "validation_dataset_image_ids_sha256": "5" * 64,
        "validation_dataset_ground_truth_sha256": "6" * 64,
        "quality_contract_sha256": "7" * 64,
        "preprocessing_contract_sha256": "8" * 64,
        "policy_sha256": policy.sha256(),
    }
    row = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "setup_id": "hailo8_setup",
        "task": "classification",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint,
        "runtime_precision_identity": "float32_layout_fp16",
    }
    identity = {
        "schema_version": 4,
        "identity_valid": True,
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "b052",
        "source_run_id": "hailo8_to_trt",
        "setup_id": "hailo8_setup",
        "variant": "composed",
        "endpoint_contract_hash": endpoint,
        "runtime_precision_identity": "float32_layout_fp16",
        **provenance,
    }
    result = {
        "status": "completed",
        "technical_status": "completed",
        "decision": "pass",
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "b052",
        "source_run_id": "hailo8_to_trt",
        "source_setup_id": "hailo8_setup",
        "variant": "composed",
        "endpoint_contract_hash": endpoint,
        "runtime_precision_identity": "float32_layout_fp16",
        "request_identity": identity,
        "primary": {"metric": "top1_accuracy", "delta": 0.0},
        **provenance,
    }
    return row, result


def test_current_authority_binds_real_snapshot_layout(tmp_path: Path) -> None:
    run = tmp_path / "run-current"
    _write_current_context(run)
    authority = _authority(run)
    assert authority["valid"] is True
    assert authority["mode"] == "required"
    assert authority["profile_start_snapshot_sha256"] == FULL_SNAPSHOT_SHA
    assert authority["profile_selection_snapshot_sha256"] == SELECTION_SHA


def test_selection_snapshot_copies_cannot_disagree(tmp_path: Path) -> None:
    run = tmp_path / "run-drift"
    _write_current_context(run, selection_sha="c" * 64)
    authority = _authority(run)
    assert authority["valid"] is False
    assert authority["native_split_quality_required"] is True
    assert "run_manifest_selection_snapshot_sha256_mismatch" in authority["errors"]


def test_marker_strip_cannot_downgrade_current_validator(tmp_path: Path) -> None:
    run = tmp_path / "run-strip"
    _write_current_context(run)
    validator = _load_script("native_producer_validate_visualize.py")
    policy = AccuracyGatePolicy()
    row, result = _legacy_row_and_result(policy)

    # This is the exploit shape: both otherwise-consistent rows advertise only
    # the historical command/result fields and omit every QF marker/binding.
    validator._bind_central_quality_evidence(
        row, [result], policy, split_quality_authority=_authority(run),
    )

    assert row["central_quality_evidence_verified"] is False
    assert row["precision_quality_verified"] is False
    assert row["native_split_quality_required"] is True
    assert row["central_quality_binding_status"] == (
        "native_split_quality_binding_invalid_or_missing"
    )


def test_unknown_context_is_fail_closed_and_legacy_is_diagnostic_only(
    tmp_path: Path,
) -> None:
    unknown = resolve_native_split_quality_authority(
        run_manifest_path=tmp_path / "missing-manifest.json",
        stage_path=tmp_path / "missing-stage.json",
    )
    assert unknown["valid"] is False
    assert unknown["native_split_quality_required"] is True

    legacy_run = tmp_path / "legacy-run"
    legacy_run.mkdir()
    (legacy_run / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "run_id": legacy_run.name,
        "workflow_version": "v2.69e-smoke-quality-start-snapshot-repair",
        "tool_version": "2.69.5",
    }), encoding="utf-8")
    legacy = resolve_native_split_quality_authority(
        run_manifest_path=legacy_run / "run_manifest.json",
        stage_path=legacy_run / "reports" / "native_producer_stage.json",
    )
    assert legacy["valid"] is True
    assert legacy["mode"] == "legacy"
    row = {"backend": "hailo8_to_trt"}
    apply_native_split_quality_authority(row, legacy)
    assert row["native_split_quality_legacy_status"] == "historical_diagnostic_only"
    assert row["execution_role"] == "legacy_manual_diagnostic"
    assert row["performance_claims_emitted"] is False

    validator = _load_script("native_producer_validate_visualize.py")
    policy = AccuracyGatePolicy()
    passing_row, passing_result = _legacy_row_and_result(policy)
    apply_native_split_quality_authority(passing_row, legacy)
    validator._bind_central_quality_evidence(
        passing_row, [passing_result], policy,
        split_quality_authority=legacy,
    )
    assert passing_row["central_quality_evidence_verified"] is True
    passing_row.update({
        "claim_ok": True,
        "ok": True,
        "eligible_for_ranking": True,
        "performance_claim_eligible": True,
        "energy_claim_eligible": True,
        "pareto_eligible": True,
        "thesis_claim_eligible": True,
    })
    validator._enforce_historical_split_diagnostic_only(passing_row)
    for field in (
        "claim_ok", "ok", "eligible_for_ranking",
        "performance_claim_eligible", "energy_claim_eligible",
        "pareto_eligible", "thesis_claim_eligible",
        "ranking_eligible", "performance_eligible", "energy_eligible",
        "thesis_valid",
    ):
        assert passing_row[field] is False
    assert passing_row["status"] == "historical_diagnostic_only"
    final = _load_script("native_producer_final_report.py")
    final._apply_comparison_claim_gates([passing_row])
    assert passing_row["performance_claim_eligible"] is False
    assert "native_split_legacy_historical_diagnostic_only" in passing_row[
        "performance_claim_exclusion_reasons"
    ]


def test_final_report_discovers_nested_report_authority_and_blocks_stripped_row(
    tmp_path: Path,
) -> None:
    run = tmp_path / "nested-run"
    _write_current_context(run)
    final = _load_script("native_producer_final_report.py")
    nested = run / "reports" / "native_producers" / "attached"
    nested.mkdir(parents=True)
    authority = final._native_split_authority(nested)
    assert authority["valid"] is True
    row = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "precision": "float32_layout_fp16",
    }
    final._apply_native_split_authority_to_rows([row], authority)
    assert row["native_split_quality_required"] is True
    assert row["native_split_semantic_binding_valid"] is False
    assert row["native_split_final_portable_binding_valid"] is False


def test_current_direct_without_binding_set_stops_before_remote_run(
    tmp_path: Path, monkeypatch,
) -> None:
    run = tmp_path / "direct-current"
    _write_current_context(run)
    # Reach the central-binding preflight with a genuinely runnable exact
    # selection.  Missing BenchmarkSets are now rejected even earlier and are
    # a different fail-closed condition than the one exercised here.
    benchmark_set = run / "models" / "resnet50" / "benchmark_set"
    case = benchmark_set / "b052"
    case.mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"task": "classification"}),
        encoding="utf-8",
    )
    (case / "split_manifest.json").write_text("{}", encoding="utf-8")
    # Exercise checkpoint creation too: remove the pre-created stage so the
    # updater must derive current policy from the immutable run manifest.
    (run / "reports" / "native_producer_stage.json").unlink()
    updater = _load_script("update_evalset_native_producers.py")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        updater, "_run",
        lambda command, **_kwargs: calls.append(command) or {"rc": 0},
    )
    stage = updater._run_native_producers(run, {
        "backends": ["hailo8"],
        "remotes": {"hailo8": {"ssh": "should-not-run", "setup_id": "hailo8_setup"}},
        "build_missing_engines": True,
        "native_force_rebuild_engines": True,
    }, timeout=1)

    assert calls == []
    assert stage["status"] == "failed"
    assert stage["native_split_quality_first"]["required"] is True
    assert stage["native_split_quality_first"]["build_or_force_flags_forwarded"] is False
    assert "hailo8" in stage["preflight_errors_by_backend"]
    checkpoint = json.loads(
        (run / "reports" / "native_producer_stage.json").read_text(encoding="utf-8")
    )
    assert checkpoint["workflow_version"] == CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW
    assert checkpoint["native_split_quality_first"]["required"] is True


def test_managed_callers_forward_central_summary_and_checkpoint_before_gates() -> None:
    updater = (ROOT / "scripts" / "update_evalset_native_producers.py").read_text(
        encoding="utf-8",
    )
    variants = (ROOT / "scripts" / "run_evalrun_native_producer_variants.py").read_text(
        encoding="utf-8",
    )
    assert '"--central-quality-summary"' in updater
    assert '"--central-quality-summary"' in variants
    assert "_write_json(checkpoint_path, stage)" in updater
    assert "_write_json(canonical_stage_path, stage)" in variants
