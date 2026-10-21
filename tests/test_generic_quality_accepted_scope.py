"""Selective quality scope against the real immutable scope projector."""
import json

import pytest

from onnx_splitpoint_tool.workflow.required_run_scope import (
    RequiredRunScopeError, build_global_required_run_scope, seal_required_run_scope,
)
from onnx_splitpoint_tool.workflow.runner import _seal_model_scope_from_accepted_cases_v27919
from tests.test_generic_quality_accounting import projected


def setup_scope(root, *, ordinary=False):
    profile, plan = projected()
    run = plan["runs"][0]
    model = "yolo26s"
    run_ids = [run["id"]]
    if ordinary:
        run_ids.append("ort_tensorrt")
    global_scope = build_global_required_run_scope(
        profile_id="synthetic-selective-quality", model_entries=[{"id": model, "task": "detection"}],
        effective_plan={"logical_run_profiles": run_ids, "effective_generic_run_ids": run_ids,
                        "setup_groups": {"hailo10_setup": run_ids}},
        hardware_targets=[{"id": run["hardware_setup_id"], "accelerator": "hailo10"}],
        run_profiles=profile["run_profiles"], created_at="before-generation",
    )
    seal_required_run_scope(root / "required_run_scope.json", global_scope)
    (root / "profile.yaml").write_text(json.dumps(profile))
    return profile, plan, dict(run_dir=root, model_id=model, benchmark_plan_source=plan,
                              benchmark_set_contract={"cases": [{"case_id": "b021"}, {"case_id": "b039"}]})


def test_profile_authorized_quality_subset_keeps_full_accepted_catalogue(tmp_path):
    _, _, arguments = setup_scope(tmp_path)
    scope, _, _, accepted = _seal_model_scope_from_accepted_cases_v27919(**arguments)
    assert scope["accepted_case_ids"] == ["b021", "b039"]
    assert {row["case_id"] for row in accepted["cases"]} == {"b021", "b039"}
    assert {row["case_id"] for row in scope["identities"]} == {"b039"}
    # A normal explicit reentry verifies the immutable original scope.
    assert _seal_model_scope_from_accepted_cases_v27919(**arguments)[0]["scope_sha256"] == scope["scope_sha256"]


@pytest.mark.parametrize("fault", ["profile_missing", "no_authority", "plan_drift", "case_missing"])
def test_selective_quality_cannot_silently_shrink_or_replace_authority(tmp_path, fault):
    profile, plan, arguments = setup_scope(tmp_path)
    if fault == "profile_missing":
        (tmp_path / "profile.yaml").unlink()
    elif fault == "no_authority":
        profile["run_profiles"] = []
        (tmp_path / "profile.yaml").write_text(json.dumps(profile))
    elif fault == "plan_drift":
        plan["runs"][0]["generic_quality_completion_sources"]["b039"]["path"] = "/different/config.json"
    else:
        arguments["benchmark_set_contract"] = {"cases": [{"case_id": "b021"}]}
    with pytest.raises(RequiredRunScopeError):
        _seal_model_scope_from_accepted_cases_v27919(**arguments)
    assert not (tmp_path / "models/yolo26s/benchmark_set/required_run_scope.json").exists()


def test_quality_subset_does_not_shrink_other_requested_split_paths(tmp_path):
    _, _, arguments = setup_scope(tmp_path, ordinary=True)
    scope, _, _, _ = _seal_model_scope_from_accepted_cases_v27919(**arguments)
    ordinary = [row for row in scope["identities"] if row["run_id"] == "ort_tensorrt" and row["variant"] == "split"]
    assert {row["case_id"] for row in ordinary} == {"b021", "b039"}


@pytest.mark.parametrize("single_catalogue", [True, False])
def test_missing_profile_authority_rejects_even_when_case_counts_agree(tmp_path, single_catalogue):
    profile, plan, arguments = setup_scope(tmp_path)
    profile["run_profiles"] = []
    (tmp_path / "profile.yaml").write_text(json.dumps(profile))
    if single_catalogue:
        arguments["benchmark_set_contract"] = {"cases": [{"case_id": "b039"}]}
    else:
        plan["runs"][0]["case_ids"] = ["b021", "b039"]
    with pytest.raises(RequiredRunScopeError, match="quality_authorization_missing"):
        _seal_model_scope_from_accepted_cases_v27919(**arguments)
    assert not (tmp_path / "models/yolo26s/benchmark_set/required_run_scope.json").exists()
