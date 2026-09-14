from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.campaign import (
    EVALUATED_MATRIX_CLAIM_SCOPE,
    RANKING_GENERALIZATION_CLAIM_SCOPE,
    build_campaign_readiness,
    resolve_campaign_claim_scope,
)
from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.gui.run_mode_editor import _SECTIONS
from onnx_splitpoint_tool.run_modes import (
    RUN_MODE_SCHEMA_VERSION,
    apply_run_mode,
    default_run_modes_config,
    validate_run_modes_config,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from onnx_splitpoint_tool.workflow.scientific_reporting import _build_report_payload


def _checks(report: dict) -> dict[str, dict]:
    return {
        str(row.get("id")): dict(row)
        for row in list(report.get("checks") or [])
        if isinstance(row, dict)
    }


def _schema_profile(*, scope: object = "__missing__", ranking_enabled: bool = False) -> dict:
    campaign: dict[str, object] = {
        "mode": "final",
        "enforcement": "strict",
    }
    if scope != "__missing__":
        campaign["claim_scope"] = scope
    return {
        "name": "claim-scope-schema-fixture",
        "selection_policy": {},
        "model_suite": {"primary": [{"id": "model-a"}], "reserve": []},
        "run_profiles": [{"id": "cpu", "type": "same_backend_reference"}],
        "validation": {},
        "reporting": {},
        "campaign": campaign,
        "ranking_validation": {"enabled": ranking_enabled},
    }


def _readiness_profile(
    *,
    scope: object = EVALUATED_MATRIX_CLAIM_SCOPE,
    ranking_enabled: bool = False,
    mode: str = "final",
) -> dict:
    campaign: dict[str, object] = {
        "id": "claim-scope-readiness-fixture",
        "mode": mode,
        "enforcement": "strict" if mode == "final" else "warn",
        "frozen_before_final_campaign": False,
    }
    if scope != "__missing__":
        campaign["claim_scope"] = scope
    return {
        "name": "claim-scope-readiness-fixture",
        "campaign": campaign,
        "quality_gate": {"frozen_before_final_campaign": False},
        "model_suite": {
            "primary": [
                {
                    "id": "model-a",
                    "enabled": True,
                    "evaluation_role": "development",
                    "family_id": "family-a",
                    "generalization_scope": "development",
                    "validation_tier": "final",
                    "path": "does-not-exist.onnx",
                    "candidate_universe": {"mode": "declared_shortlist"},
                }
            ],
            "reserve": [],
        },
        "ranking_validation": {"enabled": ranking_enabled},
        "native_producers": {"enabled": False},
        "measurement_campaign": {"system_power": {}},
    }


def _report_payload(tmp_path: Path, *, scope: str, ranking_enabled: bool) -> dict:
    profile = {
        "campaign": {"claim_scope": scope},
        "quality_gate": {
            "frozen_before_final_campaign": True,
            "dataset_tier": "final",
        },
        "ranking_validation": {"enabled": ranking_enabled},
        "model_suite": {
            "primary": [
                {
                    "id": "model-a",
                    "evaluation_role": "development",
                    "validation_tier": "final",
                }
            ],
            "reserve": [],
        },
    }
    return _build_report_payload(
        source_kind="benchmark_set",
        source_root=tmp_path,
        profile_id="claim-scope-report-fixture",
        tool_version="2.75.27",
        workflow_version="benchmark_suite",
        profile=profile,
        policy=AccuracyGatePolicy.from_mapping(profile),
        rows=[],
        predictions={},
        model_facts=[],
    )


@pytest.mark.parametrize(
    "scope,ranking_enabled",
    [
        (EVALUATED_MATRIX_CLAIM_SCOPE, False),
        (EVALUATED_MATRIX_CLAIM_SCOPE, True),
        (RANKING_GENERALIZATION_CLAIM_SCOPE, True),
        # The schema may accept this combination so readiness can explain the
        # missing prerequisite; it must never become final-ready at runtime.
        (RANKING_GENERALIZATION_CLAIM_SCOPE, False),
    ],
)
def test_schema_accepts_only_the_two_claim_scope_values(
    scope: str, ranking_enabled: bool
) -> None:
    validated = validate_evaluation_profile_payload(
        _schema_profile(scope=scope, ranking_enabled=ranking_enabled)
    )
    assert validated["campaign"]["claim_scope"] == scope


def test_schema_keeps_legacy_profiles_without_claim_scope_loadable() -> None:
    validated = validate_evaluation_profile_payload(_schema_profile())
    assert "claim_scope" not in validated["campaign"]


@pytest.mark.parametrize(
    "bad_scope",
    ["matrix", "evaluated-matrix", "EVALUATED_MATRIX", "", None, 7, []],
)
def test_schema_rejects_unknown_or_non_string_claim_scope(bad_scope: object) -> None:
    with pytest.raises(ValueError, match="claim_scope"):
        validate_evaluation_profile_payload(_schema_profile(scope=bad_scope))


def test_legacy_scope_resolution_is_intent_preserving_and_auditable() -> None:
    matrix = resolve_campaign_claim_scope(
        _readiness_profile(scope="__missing__", ranking_enabled=False)
    )
    assert matrix == {
        "scope": EVALUATED_MATRIX_CLAIM_SCOPE,
        "valid": True,
        "explicit": False,
        "source": "legacy_evaluated_matrix_intent",
        "raw": "",
    }

    ranking = resolve_campaign_claim_scope(
        _readiness_profile(scope="__missing__", ranking_enabled=True)
    )
    assert ranking["scope"] == RANKING_GENERALIZATION_CLAIM_SCOPE
    assert ranking["explicit"] is False
    assert ranking["source"] == "legacy_ranking_or_holdout_intent"

    holdout = _readiness_profile(scope="__missing__", ranking_enabled=False)
    holdout["model_suite"]["primary"][0]["evaluation_role"] = "holdout"
    resolved_holdout = resolve_campaign_claim_scope(holdout)
    assert resolved_holdout["scope"] == RANKING_GENERALIZATION_CLAIM_SCOPE
    assert resolved_holdout["source"] == "legacy_ranking_or_holdout_intent"


@pytest.mark.parametrize(
    "bad_scope",
    ["matrix", "evaluated-matrix", "EVALUATED_MATRIX", "", None, 7],
)
def test_direct_readiness_is_fail_closed_for_present_invalid_scope(
    bad_scope: object,
) -> None:
    report = build_campaign_readiness(_readiness_profile(scope=bad_scope))
    checks = _checks(report)
    assert report["claim_scope_valid"] is False
    assert report["status"] == "blocked"
    assert report["ready"] is False
    assert checks["claim_scope_valid"]["status"] == "fail"
    # Invalid input must not escape strict gates by looking like matrix scope.
    assert checks["holdout_models_present"]["status"] != "not_applicable"


def test_final_evaluated_matrix_marks_generalization_gates_not_applicable_but_keeps_core_strict() -> None:
    report = build_campaign_readiness(
        _readiness_profile(
            scope=EVALUATED_MATRIX_CLAIM_SCOPE,
            ranking_enabled=False,
        )
    )
    checks = _checks(report)

    assert report["claim_scope"] == EVALUATED_MATRIX_CLAIM_SCOPE
    assert report["ranking_generalization_required"] is False
    assert report["claim_scope_explicit"] is True
    for check_id in (
        "campaign_freeze_artifact",
        "holdout_models_present",
        "holdout_adapter_sources_frozen",
        "holdout_registry",
        "holdout_registry_integrity",
        "ranking_validation_enabled",
        "ranking_model_bundle_integrity",
        "stage_time_model",
        "native_handover_model",
    ):
        assert checks[check_id]["status"] == "not_applicable", check_id

    # The narrower scientific claim does not turn Final into a permissive mode.
    for check_id in (
        "profile_frozen",
        "dataset_classification_calibration",
        "dataset_classification_validation",
        "dataset_detection_calibration",
        "dataset_detection_validation",
        "pipeline_contract_manifest",
        "model_identity_model-a",
        "full_system_power_scope",
        "energy_command_window",
        "energy_repetitions",
        "energy_confidence_interval",
        "energy_run_order",
        "energy_calibration_manifest",
    ):
        assert checks[check_id]["status"] == "fail", check_id
    assert report["status"] == "blocked"
    assert report["required_failure_count"] > 0
    assert report["deferred_final_requirement_count"] == 0


def test_matrix_ranking_diagnostics_do_not_reenable_generalization_gates() -> None:
    report = build_campaign_readiness(
        _readiness_profile(
            scope=EVALUATED_MATRIX_CLAIM_SCOPE,
            ranking_enabled=True,
        )
    )
    checks = _checks(report)
    assert report["ranking_generalization_required"] is False
    assert checks["ranking_validation_enabled"]["status"] == "not_applicable"
    assert checks["holdout_models_present"]["status"] == "not_applicable"
    assert checks["ranking_model_bundle_integrity"]["status"] == "not_applicable"


def test_final_ranking_generalization_requires_enabled_ranking_and_holdout_contract() -> None:
    enabled = build_campaign_readiness(
        _readiness_profile(
            scope=RANKING_GENERALIZATION_CLAIM_SCOPE,
            ranking_enabled=True,
        )
    )
    enabled_checks = _checks(enabled)
    assert enabled["ranking_generalization_required"] is True
    assert enabled_checks["ranking_validation_enabled"]["status"] == "pass"
    assert enabled_checks["holdout_models_present"]["status"] == "fail"
    assert enabled_checks["holdout_registry"]["status"] == "fail"
    assert enabled_checks["ranking_model_bundle_integrity"]["status"] == "fail"
    assert enabled["status"] == "blocked"

    disabled = build_campaign_readiness(
        _readiness_profile(
            scope=RANKING_GENERALIZATION_CLAIM_SCOPE,
            ranking_enabled=False,
        )
    )
    disabled_checks = _checks(disabled)
    assert disabled_checks["ranking_validation_enabled"]["status"] == "fail"
    assert disabled["status"] == "blocked"


def test_development_scope_uses_deferred_only_for_applicable_final_gates() -> None:
    matrix = build_campaign_readiness(
        _readiness_profile(
            scope=EVALUATED_MATRIX_CLAIM_SCOPE,
            ranking_enabled=False,
            mode="development",
        )
    )
    matrix_checks = _checks(matrix)
    assert matrix_checks["profile_frozen"]["status"] == "deferred"
    assert matrix_checks["holdout_models_present"]["status"] == "not_applicable"
    assert matrix_checks["ranking_model_bundle_integrity"]["status"] == "not_applicable"

    strict = build_campaign_readiness(
        _readiness_profile(
            scope=RANKING_GENERALIZATION_CLAIM_SCOPE,
            ranking_enabled=True,
            mode="development",
        )
    )
    strict_checks = _checks(strict)
    assert strict_checks["holdout_models_present"]["status"] == "deferred"
    assert strict_checks["ranking_model_bundle_integrity"]["status"] == "deferred"


def test_run_mode_final_defaults_to_matrix_without_hidden_generalization() -> None:
    config = default_run_modes_config()
    assert RUN_MODE_SCHEMA_VERSION == 13
    for mode in config["modes"].values():
        assert mode["campaign"]["claim_scope"] == EVALUATED_MATRIX_CLAIM_SCOPE

    final = config["modes"]["final"]
    assert final["campaign"]["mode"] == "development"
    assert final["campaign"]["enforcement"] == "warn"
    assert final["ranking"]["enabled"] is True
    assert final["campaign"]["require_fitted_stage_time"] is False
    assert final["campaign"]["require_native_handover_model"] is False
    assert final["campaign"]["require_campaign_freeze"] is False
    assert final["campaign"]["require_prediction_freeze_approval"] is False
    assert final["holdout"]["prediction_freeze_enabled"] is False
    assert final["holdout"]["require_complete_candidate_universe"] is False
    assert final["holdout"]["require_frozen_predictions"] is False
    assert final["holdout"]["require_unseen_attestation"] is False


def test_run_mode_validation_rejects_unknown_scope() -> None:
    config = default_run_modes_config()
    config["modes"]["final"]["campaign"]["claim_scope"] = "matrix"
    with pytest.raises(ValueError, match="claim_scope"):
        validate_run_modes_config(config)


def test_run_mode_editor_exposes_exact_claim_scope_choices() -> None:
    specs = {
        spec[1]: spec
        for _section, section_specs in _SECTIONS
        for spec in section_specs
    }
    claim_scope = specs["campaign.claim_scope"]
    assert claim_scope[2] == "choice"
    assert claim_scope[3] == EVALUATED_MATRIX_CLAIM_SCOPE
    assert claim_scope[4] == (
        EVALUATED_MATRIX_CLAIM_SCOPE,
        RANKING_GENERALIZATION_CLAIM_SCOPE,
    )


def test_apply_run_mode_preserves_explicit_strict_opt_in() -> None:
    profile = {
        "name": "strict-opt-in",
        "campaign": {"claim_scope": RANKING_GENERALIZATION_CLAIM_SCOPE},
        "execution_preset": {"id": "final", "native": True, "energy": True},
    }
    resolved, _audit = apply_run_mode(profile, config=default_run_modes_config())
    assert resolved["campaign"]["claim_scope"] == RANKING_GENERALIZATION_CLAIM_SCOPE


def test_main_reporter_has_no_false_holdout_blocker_for_evaluated_matrix(
    tmp_path: Path,
) -> None:
    payload = _report_payload(
        tmp_path,
        scope=EVALUATED_MATRIX_CLAIM_SCOPE,
        ranking_enabled=False,
    )
    open_ids = {str(row.get("id")) for row in payload["open_items"]}
    assert payload["summary"]["ranking"]["status"] == "disabled"
    assert "independent_holdout_missing" not in open_ids
    assert "candidate_universe_incomplete" not in open_ids
    assert "predictions_not_frozen" not in open_ids
    assert "native_handover_model_unconfigured" not in open_ids
    assert "ranking_method_inputs_missing" not in open_ids
    assert payload["claim_scope"] == EVALUATED_MATRIX_CLAIM_SCOPE


def test_main_reporter_keeps_missing_holdout_blocker_for_strict_generalization(
    tmp_path: Path,
) -> None:
    payload = _report_payload(
        tmp_path,
        scope=RANKING_GENERALIZATION_CLAIM_SCOPE,
        ranking_enabled=True,
    )
    open_ids = {str(row.get("id")) for row in payload["open_items"]}
    assert "independent_holdout_missing" in open_ids
    assert payload["claim_scope"] == RANKING_GENERALIZATION_CLAIM_SCOPE


def test_generated_reporter_has_no_false_holdout_blocker_for_matrix(
    tmp_path: Path,
) -> None:
    write_benchmark_suite_script(tmp_path)
    reporter = tmp_path / "scientific_reporter_v60.py"
    spec = importlib.util.spec_from_file_location(
        "scientific_reporter_v27527_claim_scope", reporter
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    plan = {
        "model_id": "model-a",
        "evaluation_profile": {"profile_id": "matrix-generated-report"},
        "campaign": {"claim_scope": EVALUATED_MATRIX_CLAIM_SCOPE},
        "quality_gate": {
            "frozen_before_final_campaign": True,
            "dataset_tier": "final",
        },
        "ranking_validation": {"enabled": False},
        "model_suite": {
            "primary": [
                {
                    "id": "model-a",
                    "evaluation_role": "development",
                    "validation_tier": "final",
                }
            ],
            "reserve": [],
        },
    }
    payload = module.write_scientific_report(tmp_path, {}, plan)
    open_ids = {str(row.get("id")) for row in payload["open_items"]}
    assert "ranking_holdout_incomplete" not in open_ids
    assert "native_handover_model_unconfigured" not in open_ids
    assert payload["summary"]["ranking"]["status"] == "disabled"
    assert payload["claim_scope"] == EVALUATED_MATRIX_CLAIM_SCOPE
