from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from onnx_splitpoint_tool.benchmark import remote_run
from onnx_splitpoint_tool.workflow import execution_binding as binding
from tests.test_v27521_setup_local_trt_dispatch import SETUPS, _profile, _targets, _run_profiles
from tests.test_v269a_deepx_full_central_quality import _suite_module


def _row(case="b001", action="execute", **extra):
    return dict(model_id="yolo26m", setup_id=SETUPS["deepx"], run_id="ort_tensorrt",
                case_id=case, variant="split", action=action, **extra)


def test_scheduler_filters_completed_setups_and_second_application(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "benchmark_plan.json").write_text(json.dumps({"runs": _run_profiles()}))
    (suite / "benchmark_set.json").write_text("{}")
    results = tmp_path / "results"
    results.mkdir()
    calls = []
    monkeypatch.setattr(binding, "matrix_for_runtime", lambda _p: _targets())
    def leaf(**kwargs):
        calls.append(kwargs)
        return binding.ExecutionBindingResult({}, {"remote_result_files_copied": 1}, "ok", "controlled")
    monkeypatch.setattr(binding, "_run_remote_dispatch_once", leaf)
    rows = [_row("b001", "preserve"), _row("b002", "negative_completed"), _row("b003")]
    rows.append(dict(_row("full", "preserve"), variant="full", setup_id=SETUPS["hailo8"], run_id="hailo8"))
    kwargs = dict(run_root=tmp_path, model_id="yolo26m", options=SimpleNamespace(
        no_remote=False, benchmark_execution_backend="remote", parallel_remote_setups=False),
        profile_payload=_profile(), suite_dir=suite, benchmark_set_json=suite / "benchmark_set.json",
        result_dir=results, contains_hailo=True, gates={"selective_completion": True}, log=None,
        workflow_session_id="a" * 32, selective_completion_rows=rows)
    binding._remote_execution_if_requested(**kwargs)
    assert len(calls) == 1
    assert calls[0]["target_id"] == SETUPS["deepx"]
    tokens = shlex.split(calls[0]["runtime_override"]["add_args"])
    assert tokens[tokens.index("--run-ids") + 1] == "ort_tensorrt"
    assert "--quality-only-run-ids" not in tokens
    transported = json.loads(tokens[tokens.index("--selective-completion-rows") + 1])
    assert transported == rows[:3]
    calls.clear()
    rows[2]["action"] = "preserve"
    binding._remote_execution_if_requested(**kwargs)
    assert calls == []


def test_missing_companion_is_quality_only_even_on_former_performance_owner(tmp_path, monkeypatch):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "benchmark_plan.json").write_text(json.dumps({"runs": _run_profiles()}))
    (suite / "benchmark_set.json").write_text("{}")
    results = tmp_path / "results"
    results.mkdir()
    calls = []
    monkeypatch.setattr(binding, "matrix_for_runtime", lambda _p: _targets())
    monkeypatch.setattr(binding, "_run_remote_dispatch_once", lambda **kwargs:
                        (calls.append(kwargs) or binding.ExecutionBindingResult({}, {}, "ok", "controlled")))
    rows = [dict(_row("full", stage="quality_companion"), variant="full",
                 run_id="native_full_tensorrt", dispatch_run_id="ort_tensorrt")]
    binding._remote_execution_if_requested(
        run_root=tmp_path, model_id="yolo26m", options=SimpleNamespace(
            no_remote=False, benchmark_execution_backend="remote", parallel_remote_setups=False),
        profile_payload=_profile(), suite_dir=suite, benchmark_set_json=suite / "benchmark_set.json",
        result_dir=results, contains_hailo=True, gates={}, log=None, selective_completion_rows=rows)
    assert len(calls) == 1
    assert calls[0]["gates"]["quality_only_run_ids"] == ["ort_tensorrt"]
    assert calls[0]["gates"]["quality_companion_required"]


def test_trt_requirements_do_not_build_completed_or_negative_cases(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "benchmark_set.json").write_text(json.dumps({"cases": ["b001", "b002", "b003"]}))
    (suite / "benchmark_plan.json").write_text(json.dumps({"runs": [
        {"id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt"}]}))
    rows = [_row("b001", "preserve"), _row("b002", "negative_completed"), _row("b003")]
    requirements = remote_run._trt_preflight_run_requirements(suite, selective_completion_rows=rows)
    assert requirements["p1_cases"] == ["b003"]
    assert requirements["generic_p2_cases"] == ["b003"]
    assert not requirements["full_required"]
    rows[-1]["action"] = "preserve"
    requirements = remote_run._trt_preflight_run_requirements(suite, selective_completion_rows=rows)
    assert not any(requirements.values())


def test_selective_cache_hit_uses_compatible_existing_native_pair(tmp_path):
    from tests.test_v2803_fix2_trt_preflight import _pair_fixture
    from tests.test_v27920_remote_trt_cache_preflight import _LocalReadOnlyTransport
    fixture = _pair_fixture(tmp_path, 0)
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns)
              for path in fixture["base"].rglob("*") if path.is_file()}
    row = dict(_row("b135"), run_id=fixture["backend"], setup_id=fixture["setup"])
    args = SimpleNamespace(add_args=binding._scheduler_owned_remote_add_args(
        "", run_ids=[fixture["backend"]], selective_completion_rows=[row]))
    transport = _LocalReadOnlyTransport(fixture["base"])
    result = remote_run.probe_remote_trt_artifact_cache(
        transport=transport, suite_dir=fixture["suite"], setup_id=fixture["setup"],
        setup_accelerator="hailo8", active_run_ids=[fixture["backend"]], args=args,
        resolved_remote_base=str(fixture["base"]), builder_abi=fixture["abi"])
    assert len(result["observations"]) == 1
    assert result["observations"][0]["status"] == "HIT"
    assert before == {path: (path.stat().st_size, path.stat().st_mtime_ns)
                      for path in fixture["base"].rglob("*") if path.is_file()}


def test_generated_suite_main_filters_before_runtime_and_quality_leaves(tmp_path, monkeypatch):
    module = _suite_module()
    module.__file__ = str(tmp_path / "benchmark_suite.py")
    (tmp_path / "__BENCH_JSON__").write_text(json.dumps({"model_id": "yolo26m", "cases": [
        {"case_id": f"b{i:03d}", "case_dir": f"b{i:03d}", "boundary": i} for i in range(1, 4)]}))
    (tmp_path / "benchmark_plan.json").write_text(json.dumps({"runs": [{
        "id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt"}]}))
    calls, preparations, companions = [], [], []
    monkeypatch.setattr(module, "_provider_unavailable_reason", lambda *_a: None)
    monkeypatch.setattr(module, "_prepare_native_split_quality_for_case",
                        lambda **kw: preparations.append(kw) or {})
    monkeypatch.setattr(module, "_run_native_full_trt_quality_companion",
                        lambda **kw: companions.append(kw) or {"record_count": 1})
    monkeypatch.setattr(module, "_run_case", lambda *a, **kw:
                        calls.append((a, kw)) or {"status": "ok", "case_id": "b003"})
    monkeypatch.setattr(module, "_write_v60_scientific_report", lambda *_a, **_kw: None)
    rows = [_row("b001", "preserve"), _row("b002", "negative_completed"), _row("b003"),
            dict(_row("full", "preserve", stage="quality_companion"), variant="full")]
    argv = [module.__file__, "--plan", "benchmark_plan.json", "--no-csv", "--no-plot",
            "--selective-completion-rows", json.dumps(rows)]
    monkeypatch.setattr(module.sys, "argv", argv)
    assert module.main() == 0
    assert len(calls) == len(preparations) == 1
    assert calls[0][0][0].name == "b003"
    assert calls[0][1]["variants"] == ["part1", "part2", "composed"]
    assert companions == []
    rows[2]["action"] = "preserve"
    argv[-1] = json.dumps(rows)
    calls.clear(); preparations.clear()
    module.main()
    assert calls == preparations == companions == []


def test_completion_uses_new_transport_session_without_relabeling_eval():
    value = binding._remote_transport_run_id(run_root=Path("original-eval"), model_id="m",
        target_id="setup", gates={"selective_completion": True}, workflow_session_id="a" * 32)
    assert value == "original-eval_m_setup_completion_" + "a" * 32


def test_normal_workflow_stage_merges_attempt_and_original_sources(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import runner as workflow_module
    from onnx_splitpoint_tool.benchmark import suite_refresh
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    model = "resnet50"
    suite = tmp_path / "models" / model / "benchmark_set"
    suite.mkdir(parents=True)
    results = suite.parent / "benchmark_results"
    results.mkdir()
    session = "b" * 32
    attempt = results / "attempts" / session
    attempt.mkdir(parents=True)
    plan = {"legacy_reprojection": True, "runs": [
        {"id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt", "variants": ["full"]},
        {"id": "hailo8", "type": "hailo", "hw_arch": "hailo8", "variants": ["full"]}]}
    (suite / "benchmark_plan.json").write_text(json.dumps(plan))
    (suite / "benchmark_set.json").write_text(json.dumps({"cases": [{"case_id": "b053", "boundary": 53}]}))
    original_raw = results / "benchmark_results_ort_tensorrt_auto.json"
    original_raw.write_text('[{"original_measurement":"immutable"}]')
    old = {"model_id": model, "setup_id": "setup", "run_id": "ort_tensorrt", "case_id": "full",
           "variant": "full", "backend": "tensorrt", "tool_version": "2.91.0", "latency_ms": 4.0}
    frozen = tmp_path / "original"
    frozen_results = frozen / "models" / model / "benchmark_results"
    frozen_results.mkdir(parents=True)
    (frozen_results / "normalized_results.json").write_text(json.dumps({"results": [old], "sources": [{
        "path": str(original_raw), "tag": "ort_tensorrt_auto", "row_count": 1}]}))
    new_raw = attempt / "benchmark_results_hailo8_auto.json"
    new_raw.write_text(json.dumps([{"run_id": "hailo8", "case_id": "full", "variant": "full",
        "setup_id": "setup", "backend": "hailo8", "latency_ms": 2.0, "status": "ok"}]))
    rows = [dict(old, action="preserve"), dict(old, backend="hailo8", run_id="hailo8", action="execute")]
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir, workflow.run_id, workflow.session_id = tmp_path, tmp_path.name, session
    workflow.profile_payload = {"legacy_reprojection": True}
    workflow.manifest = {"models": {model: {"task": "classification"}}}
    workflow.log, workflow.warnings = mock.Mock(), []
    workflow.artifact_index, workflow.artifact_index_path = {"artifacts": []}, tmp_path / "artifact_index.json"
    workflow.artifact_index_path.write_text('{"artifacts":[]}')
    workflow.options = SimpleNamespace(skip_benchmarks=False, no_remote=False, dry_run=False, resume_missing=True)
    workflow._cancel_event = workflow._process_registry = workflow._remote_process_registry = None
    workflow._supplement_plan = {"generic_rows": rows, "quality_companion_rows": []}
    workflow._supplement_original_root = frozen
    transitions = []
    def refresh(*args, **kwargs):
        # Actual source mutation at the normal refresh boundary must complete
        # before the reference is scheduled, including parallel setup mode.
        (suite / "benchmark_suite.py").write_text("refreshed harness\n")
        transitions.append("refresh")
        return {"changed": True}
    monkeypatch.setattr(suite_refresh, "refresh_suite_harness", refresh)
    def schedule(*args):
        assert (suite / "benchmark_suite.py").read_text() == "refreshed harness\n"
        transitions.append("reference")
    workflow._schedule_management_cpu_reference = mock.Mock(side_effect=schedule)
    workflow._ensure_central_quality_service = mock.Mock(return_value=SimpleNamespace(pause_gate=None))
    workflow._targets = mock.Mock(return_value=[])
    workflow._execution_mode = mock.Mock(return_value="generate_and_run")
    workflow._queue_central_quality_requests = mock.Mock(return_value=0)
    monkeypatch.setattr(workflow_module, "benchmark_set_postcondition_v60v",
                        lambda _p: {"valid": True, "selected_suite_dir": str(suite)})
    monkeypatch.setattr(workflow_module, "finalize_suite_for_runtime", lambda **_kw: {"benchmark_plan": plan})
    monkeypatch.setattr(workflow_module, "materialize_backend_artifact_decisions",
                        lambda **_kw: {"status": "ok", "artifacts": {}, "metrics": {}})
    dispatches = []
    def dispatch(**kwargs):
        assert transitions == ["refresh", "reference"]
        assert kwargs["runtime_plan_finalization"]["suite_harness_refreshed"] is True
        dispatches.append(kwargs)
        return binding.ExecutionBindingResult({}, {"remote_dispatched": True}, "ok", "controlled process leaf")
    monkeypatch.setattr(workflow_module, "execute_benchmark_suite_if_requested", dispatch)
    workflow._stage_run_benchmarks(model, {"id": model, "task": "classification"})
    assert dispatches[0]["selective_completion_rows"] == rows
    normalized = json.loads((results / "normalized_results.json").read_text())
    retained = [r for r in normalized["results"] if r.get("run_id") == "ort_tensorrt"]
    assert len(retained) == 1 and retained[0]["tool_version"] == "2.91.0"
    assert any(r.get("backend") == "hailo8" for r in normalized["results"])
    completeness = json.loads((results / "selected_run_completeness.json").read_text())
    assert completeness["missing_selected_run_ids"] == []
    assert original_raw.read_text() == '[{"original_measurement":"immutable"}]'
