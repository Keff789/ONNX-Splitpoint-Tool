"""Bounded profile selection and coordinator accounting; hardware is mocked."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance as acceptance
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config


@pytest.fixture
def acceptance_request(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    native = source / "native.json"
    generic = source / "generic.json"
    identity = {"model_id": "resnet50", "setup_id": "orin_nx_hailo8_01", "case_id": "b067"}
    native.write_text(json.dumps({**identity, "case_id": "historical-job", "case": "b067"}))
    generic.write_text(json.dumps({"rows": [{"identity": {**identity, "comparison_backend": "hailo8"}}]}))
    cell = {**identity, "native_backend": "hailo8_to_trt", "runtime_variants": ["Native", "Generic Completed"],
            "native_source_plan": acceptance.file_binding(native), "generic_source_plan": acceptance.file_binding(generic)}
    profile = {"name": "bounded_acceptance", "model_suite": {"primary": [{"id": "resnet50"}]},
        "hardware": {"selected_setups": ["orin_nx_hailo8_01"]},
        "native_producers": {"frames": 1000, "warmup": 100, "repetitions": 3,
            "energy": {"duration_s": 60, "physical_scope": "FS", "window_label": "command",
                "task_budget": {"enabled": True, "max_retries": 2, "max_transport_failures": 20}}},
        "hardware_acceptance": {"enabled": True, "schema": acceptance.SCHEMA,
            "source_report_root": str(source), "source_primary_run": str(source), "cells": [cell]}}
    options = WorkflowOptions("profile.yaml", str(tmp_path / "new"), execution_mode="generate_and_run")
    return profile, options


def test_single_split_and_normal_budgets_survive_mode_resolution(acceptance_request):
    profile, _ = acceptance_request
    mode = default_run_modes_config()["modes"]["standard"]
    profile["execution_preset"] = {"id": "standard", "follow_tool_config": False, "snapshot": mode}
    resolved, _ = apply_run_mode(profile)
    assert resolved["hardware_acceptance"] == profile["hardware_acceptance"]
    config = acceptance.validate_request(profile)
    assert config["cells"][0]["runtime_variants"] == ["native", "generic_completed"]
    assert config["cells"][0]["backend"] == "hailo8_to_trt"


@pytest.mark.parametrize("field,value", [("frames", 1), ("warmup", 0), ("repetitions", 1)])
def test_hidden_budget_shortening_is_rejected(acceptance_request, field, value):
    profile, options = acceptance_request
    profile["native_producers"][field] = value
    with pytest.raises(ValueError, match="regular"):
        acceptance.validate_request(profile, options)


def test_functional_recipe_is_explicit_and_keeps_task_window_and_retry_limits(acceptance_request, tmp_path):
    profile, options = acceptance_request
    profile["hardware_acceptance"]["mode"] = "functional"
    profile["native_producers"]["repetitions"] = 1
    with pytest.raises(ValueError, match="one explicitly assigned runtime"):
        acceptance.validate_request(profile, options)
    profile["hardware_acceptance"]["cells"][0]["runtime_variants"] = ["generic_completed"]
    config = acceptance.validate_request(profile, options)
    context = {"budget_file": str(tmp_path / "functional_budget.json"),
        "source_ids": {"orin_nx_hailo8_01": "192.0.2.20"}, "environment": {}}
    acceptance._initialize_budget(context, config, acceptance._bound_cell(config["cells"][0], "generic_completed"), "generic_completed")
    task = next(iter(json.loads(Path(context["budget_file"]).read_text())["tasks"].values()))
    assert task["limits"] == {"max_chains": 3, "max_retries": 2, "max_transport_failures": 20}
    assert not task["chains"]
    assert any("keine neuen wissenschaftlichen" in line for line in acceptance.summary_lines(profile))
    for field, value in [("frames", 1), ("warmup", 0), ("repetitions", 3)]:
        changed = copy.deepcopy(profile)
        changed["native_producers"][field] = value
        with pytest.raises(ValueError, match="regular"):
            acceptance.validate_request(changed, options)


def test_functional_pass_requires_oracle_import_and_no_scientific_claim(tmp_path):
    proof = tmp_path / "fresh.json"
    proof.write_text("{}")
    context = {"run_dir": str(tmp_path), "expected_repetitions": 1, "acceptance_mode": "functional"}
    result = {"status": "completed", "fresh_hardware_executed": True, "new_capture_count": 1,
        "valid_repeats": {"performance": 1, "energy": 1}, "evidence_paths": [str(proof)],
        "acceptance_mode": "functional", "evaluation_role": "functional_acceptance",
        "functional_acceptance": True, "diagnostic_only": True,
        "scientific_cohort_complete": False, "scientific_claim_eligible": False,
        "scientific_statistics_published": False, "output_validation": {"status": "passed"},
        "import_projection": {"normal_report_verified": True}}
    assert acceptance._accepted_result(result, context)["status"] == "ok"
    for field, value in [("scientific_claim_eligible", True), ("scientific_statistics_published", True),
                         ("output_validation", {"status": "failed"}),
                         ("import_projection", {"normal_report_verified": False})]:
        changed = {**result, field: value}
        with pytest.raises(ValueError, match="functional PASS"):
            acceptance._accepted_result(changed, context)


@pytest.mark.parametrize("field,value", [("resume", True), ("dry_run", True), ("skip_benchmarks", True),
    ("no_remote", True), ("stop_after", "prepare_model"), ("force_stage", ["run_native_producers"])])
def test_shortcut_cannot_replace_fresh_hardware(acceptance_request, field, value):
    profile, options = acceptance_request
    setattr(options, field, value)
    with pytest.raises(ValueError, match="shortcut"):
        acceptance.validate_request(profile, options)


def test_duplicate_cell_cannot_expand_to_second_split(acceptance_request):
    profile, _ = acceptance_request
    cell = copy.deepcopy(profile["hardware_acceptance"]["cells"][0])
    cell["case_id"] = "b068"
    profile["hardware_acceptance"]["cells"].append(cell)
    with pytest.raises(ValueError, match="one selected split"):
        acceptance.validate_request(profile)


def test_source_identity_and_sibling_hash_are_verified(acceptance_request):
    profile, _ = acceptance_request
    cell = acceptance.validate_request(profile)["cells"][0]
    assert acceptance._bound_cell(cell, "native")["backend"] == "hailo8_to_trt"
    binding = cell["generic_source_plan"]
    cell["generic_source_plan"], cell["generic_source_plan_sha256"] = binding["path"], binding["sha256"]
    Path(binding["path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        acceptance._bound_cell(cell, "generic_completed")
    # The independently supported Native path remains selected and valid.
    acceptance._bound_cell(cell, "native")


def test_native_and_generic_have_exact_distinct_budget_tasks(acceptance_request, tmp_path):
    profile, _ = acceptance_request
    config = acceptance.validate_request(profile)
    context = {"budget_file": str(tmp_path / "new" / "energy_task_budget.json"),
        "source_ids": {"orin_nx_hailo8_01": "192.0.2.20"}, "environment": {}, "cancel_event": threading.Event()}
    for variant in config["cells"][0]["runtime_variants"]:
        acceptance._initialize_budget(context, config, acceptance._bound_cell(config["cells"][0], variant), variant)
    tasks = json.loads(Path(context["budget_file"]).read_text())["tasks"]
    assert set(tasks) == {
        '["hailo8_to_trt","resnet50","b067","orin_nx_hailo8_01"]',
        '["generic_completed:hailo8","resnet50","b067","orin_nx_hailo8_01"]'}
    assert all(not task["chains"] and task["limits"]["max_chains"] == 9 for task in tasks.values())


def test_no_cache_only_pass_or_partial_repeats(tmp_path):
    proof = tmp_path / "fresh.json"
    proof.write_text("{}")
    context = {"run_dir": str(tmp_path), "expected_repetitions": 3}
    result = {"status": "completed", "fresh_hardware_executed": True, "new_capture_count": 3,
              "valid_repeats": {"performance": 3, "energy": 3}, "evidence_paths": [str(proof)]}
    assert acceptance._accepted_result(result, context)["status"] == "ok"
    result["new_capture_count"] = 0
    with pytest.raises(ValueError, match="new DUT"):
        acceptance._accepted_result(result, context)
    result["new_capture_count"] = 1
    with pytest.raises(ValueError, match="new DUT"):
        acceptance._accepted_result(result, context)
    result["new_capture_count"] = 3
    result["valid_repeats"]["energy"] = 2
    with pytest.raises(ValueError, match="all regular"):
        acceptance._accepted_result(result, context)


def test_failure_keeps_actual_collector_attempts(acceptance_request, tmp_path):
    profile, _ = acceptance_request
    config = acceptance.validate_request(profile)
    cell = acceptance._bound_cell(config["cells"][0], "native")
    context = {"budget_file": str(tmp_path / "energy_task_budget.json"),
        "source_ids": {"orin_nx_hailo8_01": "192.0.2.20"}, "environment": {}, "cancel_event": threading.Event()}
    acceptance._initialize_budget(context, config, cell, "native")
    path = Path(context["budget_file"])
    budget = json.loads(path.read_text())
    next(iter(budget["tasks"].values()))["chains"] = [{"collector_started": True, "valid": False,
        "run_directory": str(tmp_path / "capture1")}]
    path.write_text(json.dumps(budget))
    failure = acceptance._failure_evidence(cell, "native", context, invoked=True, reason="test failure")
    assert failure["fresh_hardware_executed"] is True
    assert failure["new_capture_count"] == 1
    assert failure["failed_attempts"] == {"performance": None, "energy": 1}


def test_independent_variant_runs_when_one_source_is_blocked(acceptance_request, tmp_path, monkeypatch):
    profile, options = acceptance_request
    profile["hardware_acceptance"]["cells"][0]["generic_source_plan"]["sha256"] = "0" * 64
    root = tmp_path / "new"
    root.mkdir()
    events = []
    runner = SimpleNamespace(profile_payload=profile, options=options, run_dir=root, run_id="new",
        manifest={}, stage_results=[], warnings=[], report_paths=[], outputs={},
        manifest_path=root / "run_manifest.json", artifact_index_path=root / "artifact_index.json",
        _cancel_event=threading.Event(), _process_registry=None, _remote_process_registry=SimpleNamespace(cancelled=False),
        log=lambda value: None, progress=lambda *args: None, job_event=events.append,
        _save_manifest=lambda status: None, _write_workflow_control=lambda: None,
        _register_artifacts=lambda *args, **kwargs: None)
    class Broker:
        def tick(self): pass
        def close(self): pass
    def resources(runner, config, context):
        context.update(environment={}, source_ids={"orin_nx_hailo8_01": "192.0.2.20"})
        return Broker()
    monkeypatch.setattr(acceptance, "_resource_context", resources)
    calls = []
    def adapter(variant):
        def prepare(cell, context):
            calls.append((variant, "prepare"))
            return {"cell": cell}
        def run(prepared, context):
            calls.append((variant, "run"))
            path = root / "fixture_fresh_hardware_evidence.json"
            path.write_text("{}")
            return {"status": "passed", "fresh_hardware_executed": True, "new_capture_count": 3,
                    "valid_repeats": {"performance": 3, "energy": 3}, "evidence_paths": [str(path)]}
        return prepare, run
    monkeypatch.setattr(acceptance, "_adapter", adapter)
    result = acceptance.run_acceptance(runner)
    assert calls == [("native", "prepare"), ("native", "run")]
    assert result.status == "partial"
    report = json.loads((root / "reports/hardware_acceptance_results.json").read_text())
    assert report["new_capture_count"] == 3
    assert report["rows"][0]["status"] == "ok"
    assert report["rows"][1]["status"] == "blocked"
    assert report["source_campaign_resumed"] is False
    assert report["new_dataset_quality_started"] is False
    assert len(events) == 4


@pytest.mark.parametrize("functional", [False, True])
def test_normal_runner_lifecycle_uses_acceptance_without_legacy_quality_or_sweep(acceptance_request, tmp_path, monkeypatch, functional):
    """Exercise run locks, profile copying, cleanup and sealing with fake DUTs."""
    import yaml
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot, workflow_options_from_profile_snapshot
    minimal, _ = acceptance_request
    repo = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load((repo / "onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml").read_text())
    profile.update(name="acceptance_lifecycle_fixture", hardware_acceptance=minimal["hardware_acceptance"])
    profile["model_suite"]["primary"] = [row for row in profile["model_suite"]["primary"] if row["id"] == "resnet50"]
    profile["model_suite"]["reserve"] = []
    profile["native_producers"] = copy.deepcopy(minimal["native_producers"])
    repetitions = 1 if functional else 3
    if functional:
        profile["hardware_acceptance"]["mode"] = "functional"
        profile["hardware_acceptance"]["cells"][0]["runtime_variants"] = ["generic_completed"]
        profile["native_producers"]["repetitions"] = repetitions
    profile["native_producers"].update(enabled=True, backends=["hailo8"])
    profile["hardware"] = minimal["hardware"]
    mode = default_run_modes_config()["modes"]["standard"]
    profile["execution_preset"] = {"id": "standard", "follow_tool_config": False, "snapshot": mode,
                                    "overrides": {"native_enabled": True, "energy_enabled": True,
                                        "native_performance": {"frames": 1000, "warmup": 100, "repetitions": repetitions}}}
    profile["workflow"] = {"execution_mode": "generate_and_run", "skip_runtime_benchmarks": False}
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    _, snapshot = load_runtime_profile_snapshot(str(path))
    options = workflow_options_from_profile_snapshot(profile_request=str(path), out_root=str(tmp_path / "outputs"), start_snapshot=snapshot)
    class Broker:
        def tick(self): pass
        def close(self): pass
    def resources(runner, config, context):
        context.update(environment={}, source_ids={"orin_nx_hailo8_01": "192.0.2.20"})
        return Broker()
    monkeypatch.setattr(acceptance, "_resource_context", resources)
    def adapter(variant):
        def prepare(cell, context):
            assert context["expected_repetitions"] == repetitions
            assert context["requested_repetitions"] == repetitions
            assert context["acceptance_mode"] == ("functional" if functional else "scientific")
            return {"identity": cell}
        def execute(prepared, context):
            proof = Path(context["run_dir"]) / "reports" / (variant + "_fixture.json")
            proof.write_text("{}")
            return {"status": "completed", "fresh_hardware_executed": True, "new_capture_count": repetitions,
                    "valid_repeats": {"performance": repetitions, "energy": repetitions}, "evidence_paths": [str(proof)],
                    "acceptance_mode": "functional" if functional else "scientific",
                    "evaluation_role": "functional_acceptance", "functional_acceptance": functional,
                    "diagnostic_only": functional, "scientific_cohort_complete": False,
                    "scientific_claim_eligible": False, "scientific_statistics_published": False,
                    "output_validation": {"status": "passed"}, "import_projection": {"normal_report_verified": True}}
        return prepare, execute
    monkeypatch.setattr(acceptance, "_adapter", adapter)
    def legacy_forbidden(*args, **kwargs):
        pytest.fail("acceptance cannot start legacy model/quality/full-sweep stages")
    monkeypatch.setattr(EvaluationWorkflowRunner, "_run_model_pipeline_with_cache_preflight", legacy_forbidden)
    monkeypatch.setattr(EvaluationWorkflowRunner, "_stage_evaluate_quality", legacy_forbidden)
    result = EvaluationWorkflowRunner(options).run()
    assert result.status == "ok"
    root = Path(result.run_dir)
    closure = json.loads((root / "reports/artifact_index_closure.json").read_text())
    assert closure["status"] == "pass"
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["hardware_acceptance"]["runtime_path_count"] == (1 if functional else 2)
    assert manifest["workflow_stages"] == ["hardware_acceptance"]
