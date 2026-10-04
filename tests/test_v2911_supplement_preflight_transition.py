"""The normal Stage -> fresh preflight writer -> build reuse boundary."""
import copy
import json
import shlex
import pytest

from onnx_splitpoint_tool.workflow import runner as workflow
from onnx_splitpoint_tool.workflow import execution_binding as binding
from onnx_splitpoint_tool.workflow import hailo_remote_binding, benchmark_binding
from onnx_splitpoint_tool.workflow.artifacts import read_json, sha256_file, write_json
from onnx_splitpoint_tool.workflow.supplement import preserve_metadata


@pytest.fixture(params=["yolo26m", "yolo26s"])
def transition(tmp_path, monkeypatch, request):
    model = request.param
    monkeypatch.setattr(workflow, "package_build_snapshot", lambda: {"package_version": "2.91.1"})
    root = tmp_path / "same-investigation"
    options = workflow.WorkflowOptions(profile="fixture", out=str(tmp_path), run_id=root.name,
                                       resume=True, resume_missing=True,
                                       benchmark_execution_backend="local")
    runner = workflow.EvaluationWorkflowRunner(options)
    runner.run_dir, runner.run_id = root, root.name
    runner.manifest_path, runner.artifact_index_path = root / "run_manifest.json", root / "artifact_index.json"
    runner.profile_id = "fixture"
    runner.profile_payload = {"targets": ["hailo8"], "hardware_targets": [{
        "id": "setup", "enabled": True, "accelerator": "hailo8",
        "runtime": {"enabled": True, "host": "fixture", "user": "fixture"}}]}
    runner.manifest = {"run_id": root.name, "tool_version": "2.91.1", "models": {}}
    runner.artifact_index = {"artifacts": []}
    base = root / "models" / model / "benchmark_set"
    suite = base / "legacy_suite"
    write_json(base / "benchmark_set.json", {"model_id": model, "materialized": True,
        "legacy_suite_dir": str(suite.relative_to(root)), "cases": []})
    write_json(base / "benchmark_plan.json", {"runs": [{"id": "hailo8_to_trt", "stage1": "hailo8", "stage2": "tensorrt"}]})
    write_json(suite / "benchmark_set.json", {"model_id": model, "cases": [{"folder": "b001"}, {"folder": "b002"}, {"folder": "b003"}]})
    write_json(suite / "benchmark_plan.json", read_json(base / "benchmark_plan.json"))
    for module in (workflow, hailo_remote_binding, benchmark_binding):
        monkeypatch.setattr(module, "now_iso", lambda: "2026-09-25T10:00:00+02:00")
    assert runner._materialize_local_cache_preflight_inputs([{"id": model}], ["hailo8"]) == []
    artifacts = [str(path.relative_to(root)) for path in base.rglob("*") if path.is_file()]
    result = {"status": "ok", "state": "completed", "complete": True, "input_hash": "original",
              "started_at": "2026-09-25T09:00:00+02:00", "finished_at": "2026-09-25T10:00:00+02:00",
              "artifacts": artifacts}
    write_json(root / "models" / model / "stages/build_backend_artifacts/stage_result.json", result)
    runner.artifact_index["artifacts"] = [{"path": name, "size_bytes": (root / name).stat().st_size,
                                          "sha256": sha256_file(root / name)} for name in artifacts]
    write_json(runner.artifact_index_path, runner.artifact_index)
    write_json(runner.manifest_path, runner.manifest)
    original = preserve_metadata(root, "a" * 32)
    runner._supplement_original_root = runner._supplement_source_root = original
    runner._supplement_source_index = {row["path"]: copy.deepcopy(row) for row in runner.artifact_index["artifacts"]}
    rows = [dict(model_id=model, setup_id="setup", expected_setup_id="setup", run_id="hailo8_to_trt",
                 case_id=case, variant="split", action=action)
            for case, action in (("b001", "preserve"), ("b002", "execute"), ("b003", "negative_completed"))]
    runner._supplement_plan = {"generic_rows": rows, "quality_companion_rows": []}
    for module in (workflow, hailo_remote_binding, benchmark_binding):
        monkeypatch.setattr(module, "now_iso", lambda: "2026-09-29T20:00:00+02:00")
    return runner, model, original


def _preflight(runner, model):
    result = runner._run_stage(None, "artifact_cache_preflight",
                              lambda: runner._stage_artifact_cache_preflight([{"id": model}]))
    assert result.status == "ok"


def _reuse(runner, model):
    result = runner._run_stage(model, "build_backend_artifacts",
                  lambda: pytest.fail("reuse must not reach a compiler/build stage"))
    assert result.skip_reason == "resume_reused_existing_stage_result"


def test_real_fresh_writer_then_build_reuse_and_exact_dispatch(transition, monkeypatch):
    runner, model, original = transition
    before = (runner.run_dir / f"models/{model}/stages/build_backend_artifacts/stage_result.json").read_bytes()
    _preflight(runner, model)
    _reuse(runner, model)
    assert (runner.run_dir / f"models/{model}/stages/build_backend_artifacts/stage_result.json").read_bytes() == before
    calls = []
    def leaf(**kwargs):
        tokens = shlex.split(kwargs["runtime_override"]["add_args"])
        rows = json.loads(tokens[tokens.index("--selective-completion-rows") + 1])
        calls.extend((row["model_id"], row["setup_id"], row["run_id"], row["case_id"])
                     for row in rows if row["action"] == "execute")
        return binding.ExecutionBindingResult({}, {}, "ok", "controlled external leaf")
    monkeypatch.setattr(binding, "_run_remote_dispatch_once", leaf)
    runner.options.benchmark_execution_backend = "remote"
    suite = runner.run_dir / f"models/{model}/benchmark_set/legacy_suite"
    binding._remote_execution_if_requested(run_root=runner.run_dir, model_id=model,
        options=runner.options, profile_payload=runner.profile_payload, suite_dir=suite,
        benchmark_set_json=suite / "benchmark_set.json", result_dir=runner.run_dir / "result",
        contains_hailo=True, gates={"selective_completion": True}, log=None,
        workflow_session_id="b" * 32, selective_completion_rows=runner._supplement_dispatch_rows(model))
    assert calls == [(model, "setup", "hailo8_to_trt", "b002")]


def test_preflight_abort_then_second_entry_keeps_original_binding(transition):
    runner, model, original = transition
    _preflight(runner, model)
    from onnx_splitpoint_tool.workflow.supplement import prepare_source_evidence
    source, index = prepare_source_evidence(runner.run_dir)
    assert source == original
    runner._supplement_original_root = preserve_metadata(runner.run_dir, "b" * 32)
    runner._supplement_source_root = source
    runner._supplement_source_index = {row["path"]: row for row in index["artifacts"]}
    _preflight(runner, model)
    _reuse(runner, model)
    assert [r["case_id"] for r in runner._supplement_dispatch_rows(model) if r["action"] == "execute"] == ["b002"]


@pytest.mark.parametrize("damage", ["source", "boundary", "precision", "artifact", "missing"])
def test_genuine_changes_still_block_after_preflight(transition, damage):
    runner, model, original = transition
    _preflight(runner, model)
    name = "benchmark_set.json" if damage in {"source", "missing"} else "backend_artifact_decisions.json"
    path = runner.run_dir / f"models/{model}/benchmark_set/{name}"
    if damage == "missing":
        path.unlink()
    else:
        payload = read_json(path)
        payload[{"source": "model_id", "boundary": "boundary", "precision": "precision", "artifact": "artifact_sha256"}[damage]] = "changed-original-contract"
        write_json(path, payload)
    with pytest.raises(ValueError, match="supplement_source_contract_changed"):
        _reuse(runner, model)
