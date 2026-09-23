"""AP08 configuration/selection contracts using real resolvers, without Tk/HW.

The pure selection/capability test below proves the existing layer contract;
the existing global single-input filter is separately asserted, not hidden.
"""
from copy import deepcopy
import json

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.gui.panels.panel_evaluation_workflow import _profile_summary_payload
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.execution_binding import (
    _parallel_powercalc_workers,
    _parallel_remote_max_setups,
    _parallel_remote_max_uploads,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _native_selection_contract_runs_v270e,
    _native_split_case_support_v270e,
)
from onnx_splitpoint_tool.workflow.start_snapshot import (
    StartSnapshotConsistencyError,
    build_profile_start_snapshot,
    profile_selection_view,
    start_snapshot_matches_preview,
    validate_profile_start_snapshot,
)


LEGACY = {"native_release_mode": "global_barrier", "setup_queue_mode": "model_barrier"}
PARALLEL = {"native_release_mode": "per_case", "setup_queue_mode": "per_setup"}


def _source(modes=None):
    config = default_run_modes_config()
    source = {
        "name": "ap08_configuration",
        "selection_policy": {
            "max_accepted_cases_per_model": 20,
            "preferred_shortlist": 60,
            "selection_strategy": "stratified_windows",
            "min_gap": 1,
            "candidate_search_pool": "auto",
            "require_single_part2_input": False,
        },
        "model_suite": {"primary": [{"id": "resnet50", "task": "classification", "enabled": True}]},
        "run_profiles": [{"id": "ort_cpu", "type": "same_backend_reference", "full": "cpu", "stage1": "cpu", "stage2": "cpu", "enabled": True}],
        "execution_preset": {
            "id": "smoke", "follow_tool_config": False,
            "snapshot": deepcopy(config["modes"]["smoke"]),
            "overrides": {"native_enabled": False, "energy_enabled": False},
        },
    }
    if modes is not None:
        source["workflow_execution"] = deepcopy(modes)
    return source


def _resolved(source):
    return apply_run_mode(source, config=default_run_modes_config(), follow_tool_config=False)[0]


def _snapshot(source, resolved=None):
    return build_profile_start_snapshot(
        profile_request="ap08_configuration", source_profile=source,
        resolved_profile=_resolved(source) if resolved is None else resolved,
        profile_id="ap08_configuration", profile_path="/tmp/ap08_configuration.yaml",
        profile_source="file",
    )


@pytest.mark.parametrize("modes", [None, LEGACY, PARALLEL])
def test_summary_save_reload_and_plan_show_same_scheduling_without_tk(tmp_path, modes):
    source = _source(modes)
    path = tmp_path / "configuration.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    loaded = load_evaluation_profile(path, validate=True)
    expected = modes or LEGACY
    assert profile_selection_view(loaded.raw_profile)["workflow_execution"] == expected
    assert build_effective_execution_plan(loaded.raw_profile)["workflow_execution"] == expected
    lines, visible = _profile_summary_payload(str(path))
    assert visible, lines
    assert (
        f"Native-Freigabe: {expected['native_release_mode']} · Setup-Queue: {expected['setup_queue_mode']}"
        in lines
    )
    snapshot = loaded.start_snapshot
    assert snapshot["requested_selection"]["workflow_execution"] == expected
    assert snapshot["resolved_selection"]["workflow_execution"] == expected
    assert snapshot["effective_execution_plan"]["workflow_execution"] == expected
    assert validate_profile_start_snapshot(snapshot)["consistency"]["status"] == "ok"
    assert loaded.raw_profile["selection_policy"]["max_accepted_cases_per_model"] == 20
    # A visible preview and a second normal load bind the same real resolver.
    again_lines, again = _profile_summary_payload(str(path))
    assert start_snapshot_matches_preview(visible, again, profile_request=str(path)), again_lines


@pytest.mark.parametrize("field", list(PARALLEL))
def test_each_scheduler_switch_invalidates_existing_preview(field):
    source = _source(LEGACY)
    old = _snapshot(source)
    changed = deepcopy(source)
    changed["workflow_execution"][field] = PARALLEL[field]
    new = _snapshot(changed)
    assert old["selection_fingerprint"] != new["selection_fingerprint"]
    assert old["resolved_execution_sha256"] != new["resolved_execution_sha256"]
    assert not start_snapshot_matches_preview(old, new, profile_request="ap08_configuration")


@pytest.mark.parametrize("field", list(PARALLEL))
def test_resolution_cannot_silently_replace_requested_scheduler(field):
    source = _source(PARALLEL)
    resolved = _resolved(source)
    resolved["workflow_execution"][field] = LEGACY[field]
    with pytest.raises(StartSnapshotConsistencyError, match="workflow_execution"):
        _snapshot(source, resolved)


def test_snapshot_is_immutable_when_source_or_resolved_modes_change():
    source = _source(PARALLEL)
    resolved = _resolved(source)
    snapshot = _snapshot(source, resolved)
    source["workflow_execution"]["native_release_mode"] = "global_barrier"
    resolved["workflow_execution"]["setup_queue_mode"] = "model_barrier"
    assert snapshot["resolved_profile"]["workflow_execution"] == PARALLEL
    assert snapshot["effective_execution_plan"]["workflow_execution"] == PARALLEL
    assert validate_profile_start_snapshot(snapshot)["consistency"]["status"] == "ok"


def test_parallel_knob_defaults_stay_three_setups_one_upload_one_postcalc():
    resolved = _resolved(_source())
    options = WorkflowOptions(profile="", out=".")
    assert _parallel_remote_max_setups(options, resolved) == 3
    assert _parallel_remote_max_uploads(options, resolved) == 1
    assert _parallel_powercalc_workers(options, resolved) == 1


def test_explicit_two_upload_slots_survive_normal_profile_resolution():
    source = _source(PARALLEL)
    source["workflow"] = {"max_parallel_uploads": 2}
    resolved = _resolved(source)
    assert resolved["workflow"]["max_parallel_uploads"] == 2
    assert _parallel_remote_max_uploads(WorkflowOptions(profile="", out="."), resolved) == 2
    assert _snapshot(source, resolved)["resolved_profile"]["workflow"]["max_parallel_uploads"] == 2


def test_two_upload_slots_in_frozen_mode_reach_actual_dispatch_resolver():
    source = _source(PARALLEL)
    source["execution_preset"]["snapshot"]["runtime"]["parallel"]["max_uploads"] = 2
    resolved = _resolved(source)
    options = WorkflowOptions(profile="", out=".")
    assert _parallel_remote_max_setups(options, resolved) == 3
    assert _parallel_remote_max_uploads(options, resolved) == 2
    assert _parallel_powercalc_workers(options, resolved) == 1


def test_generic_twenty_stratified_cases_and_native_subset_do_not_change_with_scheduler(tmp_path):
    candidates = [
        {"case_id": f"b{boundary:03}", "split_index": boundary, "boundary": boundary,
         "cut_bytes": boundary, "part2_input_count": 3 if boundary % 9 == 1 else 1}
        for boundary in range(1, 61)
    ]
    selected_by_mode = []
    for mode_index, modes in enumerate((LEGACY, PARALLEL)):
        runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
        runner.profile_payload = _resolved(_source(modes))
        runner.profile_id = "ap08_configuration"
        runner.run_id = f"selection_{mode_index}"
        runner.run_dir = tmp_path / runner.run_id
        selected, excluded, windows = runner._select_stratified_split_candidates(
            candidates, requested=20, min_gap=1, node_count=60,
        )
        assert len(selected) == len(windows) == 20
        assert {row["window_index"] for row in selected} == set(range(1, 21))
        assert len(excluded) == 40
        analysis = runner.run_dir / "models" / "resnet50" / "analysis"
        analysis.mkdir(parents=True)
        (analysis / "prediction.json").write_text(json.dumps({
            "schema": "onnx-splitpoint/split-prediction", "schema_version": 1,
            "artifact_id": "fixed_prediction", "model_id": "resnet50", "node_count": 60,
            "policy_excluded_candidates": [], "candidates": candidates,
        }), encoding="utf-8")
        artifacts, metrics, message, status = runner._stage_select_split_candidates(
            "resnet50", {"id": "resnet50", "family": "resnet", "task": "classification", "evaluation_role": "development"},
        )
        assert status == "ok", message
        plan = json.loads(artifacts["final_candidate_plan_json"].read_text())
        assert metrics["selection_shortfall"] == 0
        assert plan["effective_require_single_part2_input"] is False
        assert [row["case_id"] for row in plan["selected_candidates"]] == [row["case_id"] for row in selected]
        selected_by_mode.append(selected)
    assert selected_by_mode[0] == selected_by_mode[1]
    generic = selected_by_mode[0]
    assert any(row["part2_input_count"] > 1 for row in generic)
    case_ids = [row["case_id"] for row in generic]
    benchmark = tmp_path / "benchmark_set"
    for row in generic:
        case_dir = benchmark / row["case_id"]
        case_dir.mkdir(parents=True)
        (case_dir / "split_manifest.json").write_text(json.dumps({
            "part2_external_inputs": [f"input_{i}" for i in range(row["part2_input_count"])],
        }), encoding="utf-8")
    support = _native_split_case_support_v270e({"resnet50": benchmark}, {"resnet50": case_ids})
    native_ids = [row["case"] for row in support if row["native_supported"]]
    assert 0 < len(native_ids) < 20
    assert native_ids == [row["case_id"] for row in generic if row["part2_input_count"] == 1]
    assert {row["reason"] for row in support if not row["native_supported"]} == {"part2_input_count_not_one"}
    for backend in ("hailo8", "hailo10h", "deepx"):
        contracts = _native_selection_contract_runs_v270e(backend, {"resnet50": native_ids})
        assert len(contracts) == 1
        assert contracts[0]["case_map"] == {"resnet50": native_ids}
    assert len(case_ids) == 20  # Capability filtering never rewrites the Generic list.


def test_native_capability_does_not_override_explicit_generic_checkbox():
    for modes in (LEGACY, PARALLEL):
        source = _source(modes)
        source["run_profiles"] = [{"id": "hailo8_to_trt", "type": "mixed_backend", "stage1": "hailo8", "stage2": "tensorrt", "enabled": True}]
        source["execution_preset"]["overrides"]["native_enabled"] = True
        plan = build_effective_execution_plan(_resolved(source))
        assert plan["native_split_requires_single_part2_input"] is True
        assert plan["effective_require_single_part2_input"] is False
        assert plan["native_multi_input_policy"] == "supported_subset_of_selected_generic_cases"
