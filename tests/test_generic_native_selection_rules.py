"""G-N1–G-N3: normal profile, real ONNX generation, bounded local doubles."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tkinter as tk

import onnx
from onnx import TensorProto, helper
import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile, resolve_evaluation_profile, save_evaluation_profile_yaml,
)
from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks, BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService, BenchmarkGenerationRuntime,
    BenchmarkGenerationOrchestrationConfig, BenchmarkGenerationOrchestrationService,
)
from onnx_splitpoint_tool.gui import profile_editor
from onnx_splitpoint_tool.run_modes import default_run_modes_config
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _resolve_generation_candidate_scope, reconcile_candidate_plan_after_generation,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
from tests.test_v269f_variant_native_split_quality_first import _load_script

ROOT = Path(__file__).resolve().parents[1]


def _model(path):
    nodes, previous = [], "x"
    for index in range(45):
        output = f"v{index}"
        nodes.append(helper.make_node("Add" if index == 20 else "Relu",
            [previous, "x"] if index == 20 else [previous], [output], name=f"node_{index}"))
        previous = output
    model = helper.make_model(helper.make_graph(nodes, "selection",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, 4])]),
        opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save_model(model, path)
    return model


def _profile(*, single, native):
    source = yaml.safe_load((ROOT / "onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml").read_text())
    source["selection_policy"].update(max_accepted_cases_per_model=20,
        preferred_shortlist=20, selection_strategy="stratified_windows", min_gap=1,
        candidate_search_pool="auto", require_single_part2_input=single)
    source["native_producers"] = {"enabled": native, "backends": ["hailo8"]}
    source["execution_preset"] = {"id": "smoke", "follow_tool_config": False,
        "snapshot": copy.deepcopy(default_run_modes_config()["modes"]["smoke"]),
        "overrides": {"native_enabled": native, "energy_enabled": False}}
    return source


def _select(tmp_path, *, single=False, native=True, requested=20):
    source = _profile(single=single, native=native)
    source["selection_policy"]["max_accepted_cases_per_model"] = requested
    profile_path = tmp_path / "profile.yaml"
    save_evaluation_profile_yaml(profile_path, source)
    loaded = load_evaluation_profile(profile_path)
    resolution = resolve_evaluation_profile(profile_path,
        export_metadata={"model_name": "resnet50", "task": "classification"})
    assert resolution.overrides["require_single_part2_input"] is single
    assert loaded.start_snapshot["effective_execution_plan"]["effective_require_single_part2_input"] is single
    model_path = tmp_path / "selection.onnx"
    model = _model(model_path)
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile=str(profile_path), out=str(tmp_path), run_id="selection"))
    runner.run_id = "selection"
    runner.run_dir = tmp_path / "selection"
    runner.run_dir.mkdir()
    runner.profile_id = loaded.profile_id
    runner.profile_payload = loaded.raw_profile
    row = {"id": "generic", "family": "generic", "task": "classification",
        "evaluation_role": "development", "resolved_path": str(model_path)}
    artifacts, _, _, status = runner._stage_analyze_model("generic", row)
    assert status == "ok"
    prediction = json.loads(artifacts["prediction_json"].read_text())
    artifacts, metrics, message, status = runner._stage_select_split_candidates("generic", row)
    assert status == ("partial" if metrics["selection_shortfall"] else "ok")
    plan = json.loads(artifacts["final_candidate_plan_json"].read_text())
    return runner, resolution, model, prediction, plan, metrics, message


def _generate(tmp_path, model, boundaries, *, single=False, orchestrate=False, **options):
    suite = tmp_path / "suite"
    suite.mkdir()
    runtime = BenchmarkGenerationRuntime(out_dir=suite, bench_log_path=suite / "benchmark.log",
        state_path=suite / "generation_state.json", requested_cases=len(boundaries),
        ranked_candidates=boundaries, candidate_search_pool=boundaries,
        model_name="generic", model_source=str(tmp_path / "selection.onnx"), hef_full_policy="skip")
    cfg = BenchmarkGenerationExecutionConfig(runtime=runtime, target_cases=len(boundaries), gap=0,
        ranked_candidates=boundaries, candidate_search_pool=boundaries, out_dir=suite,
        base="generic", pad=3, strict_boundary=False, model=model, nodes=list(model.graph.node),
        order=list(range(len(model.graph.node))), analysis_payload={},
        full_model_src=str(tmp_path / "selection.onnx"), require_single_part2_input=single,
        **options)
    cb = BenchmarkGenerationExecutionCallbacks(log=lambda *_a, **_k: None,
        queue_put=lambda *_a: None, persist_state=lambda **_k: None,
        publish_hailo_diagnostics=lambda *_a, **_k: None, predicted_metrics_for_boundary=lambda *_a: {},
        hailo_parse_entry_for_boundary=lambda *_a: None, hailo_parse_scalar_fields=lambda *_a: {})
    if orchestrate:
        from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
        orchestration = BenchmarkGenerationOrchestrationConfig(
            runtime=runtime, execution_cfg=cfg, execution_callbacks=cb, target_cases=len(boundaries),
            preferred_shortlist_original=boundaries, ranked_candidates=boundaries, candidate_search_pool=boundaries,
            out_dir=suite, base="generic", pad=3, full_model_src=cfg.full_model_src, full_model_dst=cfg.full_model_src,
            analysis_payload={}, analysis_params_payload={}, system_spec_payload=None,
            bench_log_path=str(runtime.bench_log_path), bench_plan_runs=cfg.bench_plan_runs,
            hef_targets=cfg.hef_targets, hef_full=False, hef_part1=cfg.hef_part1, hef_part2=cfg.hef_part2,
            hef_backend="test", hef_fixup=False, hef_opt_level=1, hef_calib_dir=None,
            hef_calib_count=1, hef_calib_bs=1, hef_force=False, hef_keep=True,
            hef_wsl_distro=None, hef_wsl_venv="", hef_timeout_s=60,
            full_hef_policy="skip", full_model_preflight_policy="skip", hailo_selected=True,
            hailo_build_hef_fn=cfg.hailo_build_hef_fn,
            write_harness_script=lambda dst_dir, bench_json_name="benchmark_set.json": write_benchmark_suite_script(dst_dir, bench_json_name=bench_json_name),
            evaluation_profile_meta={"source": "evaluation_workflow"},
        )
        result = BenchmarkGenerationOrchestrationService().run(orchestration)
        assert result.ranked_candidates == result.candidate_search_pool == boundaries
        assert result.shortlist_prefiltered_boundaries == []
        chosen = [row["boundary"] for row in runtime.cases]
    else:
        chosen = BenchmarkGenerationExecutionService().execute_case_build_loop(cfg, cb)
    return chosen, runtime, suite


@pytest.mark.parametrize("single,native", [(False, False), (False, True), (True, False), (True, True)])
def test_checkbox_resolver_analysis_selection_and_real_generator_agree(tmp_path, single, native):
    runner, resolution, model, prediction, plan, _, _ = _select(tmp_path, single=single, native=native)
    counts = {row["part2_input_count"] for row in plan["selected_candidates"]}
    assert counts == ({1} if single else {1, 2})
    assert len(plan["selected_candidates"]) == 20
    assert plan["native_capability_backfills"] == []
    assert plan["native_capability_excluded_candidates"] == []
    ranked, pool, requested, exact = _resolve_generation_candidate_scope(plan, prediction,
        expected_selection_policy=runner.profile_payload["selection_policy"])
    assert exact and requested == 20 and ranked == pool
    assert ranked == [row["boundary"] for row in plan["selected_candidates"]]
    chosen, runtime, suite = _generate(tmp_path, model, pool,
        single=resolution.overrides["require_single_part2_input"])
    assert chosen == ranked
    assert [row["boundary"] for row in runtime.cases] == ranked
    actual_counts = {len(json.loads((suite / f"b{b:03d}" / "split_manifest.json").read_text())["part2_external_inputs"]) for b in ranked}
    assert actual_counts == counts
    preserved, trace = reconcile_candidate_plan_after_generation(plan, prediction=prediction,
        accepted_cases=runtime.cases, rejected_cases=[])
    assert preserved == plan and trace["backfilled_case_ids"] == []


def test_shortfall_is_honest_and_rejections_do_not_replace_or_reorder(tmp_path):
    _, _, _, prediction, plan, metrics, message = _select(tmp_path, requested=100)
    selected = plan["selected_candidates"]
    assert 0 < len(selected) < 100
    assert metrics["selection_shortfall"] == 100 - len(selected)
    assert "shortfall=" in message
    ranked, pool, requested, exact = _resolve_generation_candidate_scope(plan, prediction)
    assert exact and ranked == pool and requested == len(selected)
    accepted = [dict(row) for row in selected[1:]]
    rejected = [{**selected[0], "reason": "common_split_export_failure"}]
    result, trace = reconcile_candidate_plan_after_generation(plan, prediction=prediction,
        accepted_cases=accepted, rejected_cases=rejected)
    assert result == plan
    assert trace["removed_case_ids"] == trace["backfilled_case_ids"] == []
    assert trace["claim_eligible"] is False
    assert trace["final_case_ids"] == [row["case_id"] for row in selected]
    assert trace["accepted_case_ids"] == [row["case_id"] for row in accepted]
    with pytest.raises(ValueError):
        reconcile_candidate_plan_after_generation(plan, prediction=prediction,
            accepted_cases=list(reversed(accepted)), rejected_cases=rejected)


@pytest.mark.parametrize("guard", ["part2_precheck", "part1_static", "h8_part1_cluster"])
def test_backend_local_early_guard_preserves_generic_export_and_other_paths(tmp_path, monkeypatch, guard):
    model = _model(tmp_path / "selection.onnx")
    service = BenchmarkGenerationExecutionService
    builds = []
    def build(source, **kwargs):
        builds.append((kwargs["hw_arch"], "part1" if "part1" in kwargs["net_name"] else "part2"))
        path = Path(kwargs["outdir"]) / "compiled.hef"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local build double")
        return SimpleNamespace(ok=True, skipped=False, timed_out=False, hef_path=str(path),
            elapsed_s=0, context_count=None, detected={}, process={}, details={}, calib_info={}, error="")
    options = dict(hef_targets=["hailo8", "hailo10h"], hef_part1=True, hef_part2=True,
        hailo_build_hef_fn=build, hailo_salvage_enable=False,
        bench_plan_runs=[{"id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt"}])
    if guard == "part2_precheck":
        options["hailo_part2_precheck_fn"] = lambda _manifest: {"inspect_ok": True, "compatible": False}
    elif guard == "part1_static":
        monkeypatch.setattr(service, "_yolo26_part1_static_skip_reason", lambda *_a: "controlled static Part1 exclusion")
    else:
        from onnx_splitpoint_tool.benchmark.hailo_policy import HailoClusterSkipDecision
        monkeypatch.setattr("onnx_splitpoint_tool.benchmark.services.should_skip_from_failure_cluster",
            lambda *_a, **kw: HailoClusterSkipDecision(kw["hw_archs"] == ["hailo8"] and kw["stage"] == "part1", 2, "controlled cluster"))
    chosen, runtime, suite = _generate(tmp_path, model, [2], **options)
    assert chosen == [2] and runtime.discarded_cases == []
    manifest = json.loads((suite / "b002/split_manifest.json").read_text())
    assert len(manifest["part2_external_inputs"]) == 2
    assert (suite / "b002/generic_part1_b2.onnx").is_file()
    assert (suite / "b002/generic_part2_b2.onnx").is_file()
    exclusions = manifest["hailo"]["backend_exclusions"]
    assert exclusions and manifest["hailo"]["partial_keep"] is True
    if guard == "part2_precheck":
        assert set(builds) == {("hailo8", "part1"), ("hailo10h", "part1")}
    elif guard == "part1_static":
        assert set(builds) == {("hailo8", "part2"), ("hailo10h", "part2")}
    else:
        assert set(builds) == {("hailo8", "part2"), ("hailo10h", "part1"), ("hailo10h", "part2")}
    # Part2 availability cannot stand in for successful composed execution.
    for arch, availability in manifest["hailo"]["case_variant_availability"].items():
        if guard == "part1_static" or (guard == "h8_part1_cluster" and arch == "hailo8"):
            assert availability["part2"] is True and availability["composed"] is False


@pytest.mark.parametrize("cases", [["b003", "b001", "b002"], ["b001"], []])
def test_native_subset_keeps_generic_order_and_empty_subset_keeps_unique_full(tmp_path, monkeypatch, cases):
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path), run_id="native-subset"))
    runner.run_dir = tmp_path / "native-subset"
    runner.run_id = runner.run_dir.name
    runner.manifest = {"models": {"resnet50": {}}}
    runner.profile_payload = {"native_producers": {"enabled": True}}
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=runner.run_dir / "reports/remote_lease_journal",
    )
    suite = runner.run_dir / "models/resnet50/benchmark_set"
    suite.mkdir(parents=True)
    (suite / "benchmark_set.json").write_text(json.dumps({"cases": []}))
    for case in ["b001", "b002", "b003"]:
        directory = suite / case
        directory.mkdir()
        (directory / "split_manifest.json").write_text(json.dumps({
            "part2_external_inputs": ["one"] if case != "b001" else ["one", "two"]}))
    analysis = runner.run_dir / "models/resnet50/analysis"
    analysis.mkdir()
    (analysis / "final_candidate_plan.json").write_text(json.dumps({"selected_candidates": [{"case_id": case} for case in cases]}))
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner.benchmark_set_postcondition_v60v",
        lambda _p: {"valid": True, "selected_suite_dir": str(suite)})
    cfg = {"enabled": True, "models": ["resnet50"], "backends": ["hailo8"], "split_backends": ["hailo8"],
        "case_policy": "case_map_only", "case_map": {"resnet50": cases},
        "full_baselines": {"enabled": True, "backends_by_producer": {"hailo8": ["hailo8", "tensorrt"]}},
        "remotes": {"hailo8": {"setup_id": "h8", "ssh": "unused@test"}}}
    monkeypatch.setattr(runner, "_materialize_variant_native_remotes", lambda _cfg: cfg["remotes"])
    variants = runner._normal_native_release_variants(cfg)
    assert len(variants) == 1
    assert variants[0]["case_map"] == {"resnet50": [case for case in cases if case != "b001"]}
    assert variants[0]["split_backends"] == (["hailo8"] if any(case != "b001" for case in cases) else [])
    coordinator = _load_script("run_evalrun_native_producer_variants.py")
    leaves = coordinator._per_case_variants(runner.run_dir, cfg, variants)
    full = [leaf for leaf in leaves if leaf["release_kind"] == "full"]
    split = [leaf for leaf in leaves if leaf["release_kind"] == "split"]
    assert len(full) == 2
    assert [leaf["case_map"]["resnet50"][0] for leaf in split] == [case for case in cases if case != "b001"]
    assert json.loads((analysis / "final_candidate_plan.json").read_text())["selected_candidates"] == [{"case_id": case} for case in cases]


@pytest.mark.parametrize("single,native", [(False, False), (False, True), (True, False), (True, True)])
def test_real_editor_save_reload_keeps_checkbox_without_new_mode(tmp_path, monkeypatch, single, native):
    probe = subprocess.run([sys.executable, "-B", "-c", "import tkinter as tk; r=tk.Tk(); r.destroy()"],
        capture_output=True, text=True, timeout=6)
    if probe.returncode:
        pytest.skip("real Tk display unavailable: " + probe.stderr.strip())
    original, saved = tmp_path / "original.yaml", tmp_path / "saved.yaml"
    save_evaluation_profile_yaml(original, _profile(single=not single, native=native))
    original_bytes = original.read_bytes()
    errors = []
    monkeypatch.setattr(profile_editor.messagebox, "showerror", lambda *a, **_k: errors.append(a))
    monkeypatch.setattr(profile_editor.filedialog, "asksaveasfilename", lambda **_k: str(saved))
    root = tk.Tk()
    root.withdraw()
    editor = None
    try:
        editor = profile_editor.EvaluationProfileEditor(root, profile_var=tk.StringVar(root, str(original)))
        editor.withdraw()
        assert editor._load_profile(str(original)), errors
        editor.var_require_single_part2_input.set(single)
        editor.var_native_enabled.set(native)
        editor._save(use_after=False)
        assert saved.is_file() and not errors, errors
        assert editor._load_profile(str(saved)), errors
        assert editor.var_require_single_part2_input.get() is single
        loaded = load_evaluation_profile(saved)
        assert loaded.raw_profile["selection_policy"]["require_single_part2_input"] is single
        assert loaded.start_snapshot["effective_execution_plan"]["effective_require_single_part2_input"] is single
        assert "native_case_selection" not in loaded.raw_profile["selection_policy"]
        assert original.read_bytes() == original_bytes
    finally:
        if editor is not None:
            editor.destroy()
        root.destroy()


@pytest.mark.parametrize("drift", ["outside_declared", "reorder", "duplicate", "noncanonical"])
def test_forced_scope_membership_and_order_remain_validated(drift):
    candidates = [{"case_id": f"b{n:03d}", "boundary": n} for n in [3, 1, 2]]
    plan = {"model_id": "m", "selected_candidates": copy.deepcopy(candidates[:2]), "requested_cases": 2}
    policy = {"forced_cases": {"m": ["b003", "b001"]}, "require_single_part2_input": False}
    if drift == "outside_declared":
        plan["selected_candidates"][1] = candidates[2]
    elif drift == "reorder":
        plan["selected_candidates"].reverse()
    elif drift == "duplicate":
        policy["forced_cases"]["m"] = ["b003", "b003"]
    else:
        policy["forced_cases"]["m"] = ["unknown", "b001"]
    with pytest.raises(ValueError, match="Forced candidate scope"):
        _resolve_generation_candidate_scope(plan, {"candidates": candidates}, expected_selection_policy=policy)


def test_forced_scope_respects_explicit_single_tensor_shortfall():
    prediction = {"candidates": [{"case_id": "b003", "boundary": 3, "part2_input_count": 1}],
        "policy_excluded_candidates": [{"case_id": "b001", "boundary": 1, "part2_input_count": 2,
            "exclude_source": "requested_selection_policy", "exclude_reason": "part2_input_count_not_one"}]}
    plan = {"model_id": "m", "selected_candidates": prediction["candidates"], "requested_cases": 2}
    policy = {"forced_cases": {"m": ["b003", "b001"]}, "require_single_part2_input": True}
    assert _resolve_generation_candidate_scope(plan, prediction, expected_selection_policy=policy) == ([3], [3], 1, True)


def test_empty_supported_native_subset_without_full_skips_before_any_ssh(tmp_path, monkeypatch):
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path), run_id="no-native-jobs"))
    runner.run_dir = tmp_path / "no-native-jobs"
    runner.run_id = runner.run_dir.name
    runner.manifest = {"models": {"resnet50": {}}}
    runner.profile_payload = {"native_producers": {"enabled": True}}
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id),
        journal_dir=runner.run_dir / "reports/remote_lease_journal",
    )
    suite = runner.run_dir / "models/resnet50/benchmark_set"
    (suite / "b001").mkdir(parents=True)
    (suite / "benchmark_set.json").write_text(json.dumps({"cases": [{"folder": "b001", "boundary": 1}]}))
    (suite / "b001/split_manifest.json").write_text(json.dumps({"part2_external_inputs": ["one", "two"]}))
    cfg = {"enabled": True, "models": ["resnet50"], "backends": ["hailo8"], "split_backends": ["hailo8"],
        "full_baselines": {"enabled": False}, "energy": {"enabled": False}, "copy_benchmarksets": False}
    monkeypatch.setattr(runner, "_native_producer_config", lambda: cfg)
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.runner.benchmark_set_postcondition_v60v",
        lambda _p: {"valid": True, "selected_suite_dir": str(suite)})
    paths, _, _, status = runner._stage_run_native_producers()
    assert status == "skipped"
    stage = json.loads(paths["native_producer_stage_json"].read_text())
    assert stage["failure_reason"] == "no_supported_native_split_cases"
    assert stage["native_split_selected_case_map"] == {"resnet50": ["b001"]}
    assert stage["native_split_supported_case_map"] == {}
    assert stage["native_split_capability_exclusions"][0]["reason"] == "part2_input_count_not_one"
    assert stage["started_remote_count"] == 0 and stage["transfer_attempted"] is False


def test_normal_orchestration_does_not_globally_filter_or_promote_hailo_subset(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.hailo_backend import HailoHefBuildResult
    model = _model(tmp_path / "selection.onnx")
    calls = []
    def build(_source, **kw):
        calls.append(kw["net_name"])
        path = Path(kw["outdir"]) / "compiled.hef"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local builder double")
        return HailoHefBuildResult(ok=True, elapsed_s=0, hw_arch=kw["hw_arch"],
            net_name=kw["net_name"], hef_path=str(path))
    def forbidden_global_reselection(*_a, **_k):
        raise AssertionError("the selected Generic cohort reached an adaptive Hailo gate")
    for name in ("_probe_any_hailo_part2_compatible", "_promote_yolo26_true_hailo_part2_candidate", "_promote_yolo26_hetero_throughput_candidates"):
        monkeypatch.setattr(BenchmarkGenerationOrchestrationService, name, forbidden_global_reselection)
    runs = [{"id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt", "variants": ["composed"]},
        {"id": "tensorrt_to_hailo8", "type": "matrix", "stage1": {"type": "onnxruntime", "provider": "tensorrt"},
         "stage2": {"type": "hailo", "hw_arch": "hailo8"}, "variants": ["part2", "composed"]}]
    chosen, runtime, suite = _generate(tmp_path, model, [30, 2], orchestrate=True,
        bench_plan_runs=runs, hef_targets=["hailo8"], hef_part1=True, hef_part2=True,
        hailo_build_hef_fn=build, hailo_salvage_enable=False,
        hailo_part2_precheck_fn=lambda manifest: {"inspect_ok": True,
            "compatible": len(manifest["part2_external_inputs"]) == 1})
    try:
        assert chosen == [30, 2] and runtime.discarded_cases == []
        assert calls == ["generic_part1_b30", "generic_part2_b30", "generic_part1_b2"]
        manifest = json.loads((suite / "b002/split_manifest.json").read_text())
        assert manifest["hailo"]["backend_exclusions"][0]["reason"] == "hailo_part2_auto_filtered"
        assert manifest["hailo"]["case_variant_availability"]["hailo8"]["composed"] is False
    finally:
        runtime.close()
