"""Generation reuse accepts only the existing writers' exact downstream output."""
import copy
import shutil

import pytest

from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.workflow import supplement
from onnx_splitpoint_tool.workflow.artifacts import read_json, sha256_file, write_json
from onnx_splitpoint_tool.workflow.execution_binding import finalize_suite_for_runtime
from onnx_splitpoint_tool.workflow.required_run_scope import (
    authoritative_run_descriptors,
    merge_authoritative_run_descriptors,
)


JSON_ALIASES = (
    "benchmark_set.json", "benchmark_plan.json",
    "legacy_suite/benchmark_set.json", "legacy_suite/benchmark_plan.json",
)


def _descriptor(root, raw, stage="run_benchmarks", model="yolo26m"):
    path = root / raw
    return {"path": raw, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "producer_stage": stage,
            "model_id": model}


def _generation(tmp_path, model="yolo26m"):
    root = tmp_path / "run"
    original = root / "jobs/supplement_attempts/first/original"
    base = root / "models" / model / "benchmark_set"
    suite = base / "legacy_suite"
    profile = {"quality_gate": {"statistics": {"execution_location": "central_management"}}}
    run = {"id": "hailo8_to_trt", "type": "matrix", "provider": "tensorrt",
           "stage1": {"type": "hailo", "hw_arch": "hailo8"},
           "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
           "variants": ["part1", "part2", "composed"],
           "precision": "fp16", "validation_images": "datasets/frozen.json",
           "validation_budget_authoritative": True, "validation_items_requested": 5000,
           "validation_max_images": 5000, "warmup": 10, "throughput_frames": 1000}
    plan = {"schema": "onnx-splitpoint/benchmark-plan", "model_id": model,
            "runs": [run, {"id": "ort_cpu", "provider": "cpu"}]}
    for parent in (base, suite):
        write_json(parent / "benchmark_plan.json", plan)
        write_json(parent / "benchmark_set.json", {
            "schema": "onnx-splitpoint/benchmark-set", "model_id": model,
            "source_model": "frozen-model.onnx", "cases": [{"case_id": "b003"}],
            "plan": copy.deepcopy(plan),
        })

    def finalize():
        return finalize_suite_for_runtime(
            suite_dir=suite, run_root=root, model_id=model,
            suite_payload=read_json(suite / "benchmark_set.json"),
            benchmark_plan=read_json(suite / "benchmark_plan.json"),
            profile_payload=profile, model_task="detection",
            require_existing_cpu_reference=True,
        )

    finalize()
    scope = {"schema": "onnx-splitpoint/required-run-scope", "schema_version": 3,
             "identity_mode": "physical_strict_v3", "physical_run_descriptors": [{
                 "model_id": model, "run_id": "hailo8_to_trt", "backend": "hailo8_to_trt",
                 "variant": "split", "expected_setup_id": "orin_nx_hailo8_01",
                 "measurement_endpoint": "completed_detection",
                 "quality_endpoint": "completed_detection",
             }]}
    write_json(root / "required_run_scope.json", scope)
    script = suite / "benchmark_suite.py"
    script.write_text("# archived generation harness\n", encoding="utf-8")
    raws = [str((base / name).relative_to(root)) for name in JSON_ALIASES]
    raws += [str(script.relative_to(root)), "required_run_scope.json"]
    source_index = {}
    for raw in raws:
        stage = "generate_benchmark_set" if raw.endswith(".py") else "run_benchmarks"
        source_index[raw] = _descriptor(root, raw, stage, model)
        target = original / raw
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / raw, target)
    write_json(original / "artifact_index.json", {"artifacts": list(source_index.values())})
    projected = merge_authoritative_run_descriptors(
        read_json(suite / "benchmark_plan.json"),
        authoritative_run_descriptors(scope, model_id=model),
    )
    # The ordinary selective payload writer materializes this projection;
    # runtime finalization subsequently mirrors it into all four aliases.
    old_executable = read_json(suite / "benchmark_plan.json")
    old_executable["runs"] = projected["runs"]
    write_json(suite / "benchmark_plan.json", old_executable)
    finalize()
    write_benchmark_suite_script(suite)
    current_index = {raw: _descriptor(root, raw, model=model) for raw in raws}
    return root, original, source_index, current_index, model


def _check(fixture, name):
    root, original, source_index, current_index, model = fixture
    raw = f"models/{model}/benchmark_set/{name}"
    return supplement.generation_runtime_binding(
        root, original, raw, source_index[raw], source_index, current_index,
    )


def _attempt_decision_path(runner, stage_dir):
    return (runner.run_dir / "jobs/supplement_attempts" / runner.session_id
            / "resume_decisions" / stage_dir.relative_to(runner.run_dir)
            / "resume_decision.json")


@pytest.mark.parametrize("model", ["yolo26m", "yolo26s"])
def test_exact_writer_output_reuses_all_aliases_without_mutation(tmp_path, model):
    fixture = _generation(tmp_path, model)
    root, original, source_index, _, _ = fixture
    before = {root / raw: (root / raw).read_bytes() for raw in source_index}
    before.update({original / raw: (original / raw).read_bytes() for raw in source_index})
    for name in (*JSON_ALIASES, "legacy_suite/benchmark_suite.py"):
        assert _check(fixture, name)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("field,value", [
    ("precision", "int8"), ("validation_images", "other-dataset.json"),
    ("validation_items_requested", 50), ("measurement_endpoint", "raw_head"),
    ("throughput_frames", 10), ("validation_budget_authoritative", 1),
])
def test_reindexed_measurement_contract_mutation_is_still_rejected(tmp_path, field, value):
    fixture = _generation(tmp_path)
    root, _, _, current_index, model = fixture
    raw = f"models/{model}/benchmark_set/legacy_suite/benchmark_plan.json"
    payload = read_json(root / raw)
    for alias in ("runs", "planned_runs"):
        payload[alias][0][field] = value
    write_json(root / raw, payload)
    current_index[raw] = _descriptor(root, raw)
    assert _check(fixture, "legacy_suite/benchmark_plan.json") is None


@pytest.mark.parametrize("field,value", [
    ("source_model", "other-model.onnx"), ("cases", [{"case_id": "b021"}]),
])
def test_reindexed_source_or_selection_mutation_is_still_rejected(tmp_path, field, value):
    fixture = _generation(tmp_path)
    root, _, _, current_index, model = fixture
    raw = f"models/{model}/benchmark_set/benchmark_set.json"
    payload = read_json(root / raw)
    payload[field] = value
    write_json(root / raw, payload)
    current_index[raw] = _descriptor(root, raw)
    assert _check(fixture, "benchmark_set.json") is None


@pytest.mark.parametrize("damage", [
    "source_json", "source_scope", "live_scope", "current_binding",
    "missing_original", "scope_symlink",
])
def test_original_evidence_and_live_scope_remain_bound(tmp_path, damage):
    fixture = _generation(tmp_path)
    root, original, source_index, current_index, model = fixture
    raw = f"models/{model}/benchmark_set/benchmark_plan.json"
    if damage == "current_binding":
        current_index[raw]["sha256"] = "0" * 64
    elif damage == "missing_original":
        (original / raw).unlink()
    elif damage == "scope_symlink":
        (root / "required_run_scope.json").unlink()
        (root / "required_run_scope.json").symlink_to(original / "required_run_scope.json")
    else:
        path = (original / raw if damage == "source_json" else
                (root if damage == "live_scope" else original) / "required_run_scope.json")
        payload = read_json(path)
        payload["unexpected_contract_change"] = True
        write_json(path, payload)
    try:
        result = _check(fixture, "benchmark_plan.json")
    except ValueError:
        return
    assert result is None


@pytest.mark.parametrize("damage", ["noncanonical", "current_unbound", "original_tampered"])
def test_python_refresh_requires_canonical_writer_and_existing_bindings(tmp_path, damage):
    fixture = _generation(tmp_path)
    root, original, _, current_index, model = fixture
    raw = f"models/{model}/benchmark_set/legacy_suite/benchmark_suite.py"
    if damage == "current_unbound":
        current_index[raw]["sha256"] = "0" * 64
    else:
        path = (original if damage == "original_tampered" else root) / raw
        path.write_bytes(path.read_bytes() + b"\nraise RuntimeError('unexpected execution')\n")
        if damage == "noncanonical":
            current_index[raw] = _descriptor(root, raw)
    if damage == "noncanonical":
        assert _check(fixture, "legacy_suite/benchmark_suite.py") is None
    else:
        with pytest.raises(ValueError, match="supplement_generation_binding_changed"):
            _check(fixture, "legacy_suite/benchmark_suite.py")


def test_missing_early_python_copy_does_not_relabel_historical_source(tmp_path):
    fixture = _generation(tmp_path)
    _, original, source_index, _, model = fixture
    raw = f"models/{model}/benchmark_set/legacy_suite/benchmark_suite.py"
    historical = copy.deepcopy(source_index[raw])
    (original / raw).unlink()
    assert _check(fixture, "legacy_suite/benchmark_suite.py")
    assert source_index[raw] == historical
    assert not (original / raw).exists()


@pytest.mark.parametrize("model", ["yolo26m", "yolo26s"])
def test_normal_generation_stage_reuses_original_on_second_entry(tmp_path, monkeypatch, model):
    from onnx_splitpoint_tool.workflow import runner as workflow

    root, original, source_index, current_index, _ = _generation(tmp_path, model)
    monkeypatch.setattr(workflow, "package_build_snapshot", lambda: {"package_version": "2.91.2"})
    options = workflow.WorkflowOptions(
        profile="fixture", out=str(tmp_path), run_id=root.name,
        resume=True, resume_missing=True,
    )
    runner = workflow.EvaluationWorkflowRunner(options)
    runner.run_dir, runner.run_id = root, root.name
    runner.manifest_path = root / "run_manifest.json"
    runner.artifact_index_path = root / "artifact_index.json"
    runner.profile_id, runner.profile_payload = "fixture", {}
    runner.manifest = {"run_id": root.name, "tool_version": "2.91.1", "models": {},
                       "execution_sessions": [{"session_id": "first"}]}
    runner.artifact_index = {"artifacts": list(current_index.values())}
    write_json(runner.manifest_path, runner.manifest)
    write_json(runner.artifact_index_path, runner.artifact_index)
    write_json(original / "run_manifest.json", runner.manifest)
    runner._supplement_source_root = runner._supplement_original_root = original
    runner._supplement_source_index = source_index
    runner._supplement_plan = {"generic_rows": [{"model_id": model, "action": "preserve"}],
                               "quality_companion_rows": []}
    result_path = root / f"models/{model}/stages/generate_benchmark_set/stage_result.json"
    write_json(result_path, {
        "status": "ok", "state": "completed", "complete": True,
        "input_hash": "original-generation-source",
        "started_at": "2026-09-25T10:00:00+02:00", "finished_at": "2026-09-25T10:01:00+02:00",
        "artifacts": [raw for raw in source_index if raw != "required_run_scope.json"],
    })
    original_stage_bytes = result_path.read_bytes()
    source_index_bytes = (original / "artifact_index.json").read_bytes()

    def generation_forbidden():
        pytest.fail("completed generation must never be executed again")

    for entry in range(2):
        runner.session_id = ("a" if not entry else "b") * 32
        if entry:
            runner._supplement_original_root = supplement.preserve_metadata(root, "second")
            selected, index = supplement.prepare_source_evidence(root)
            assert selected == original
            runner._supplement_source_root = selected
            runner._supplement_source_index = {row["path"]: row for row in index["artifacts"]}
        result = runner._run_stage(model, "generate_benchmark_set", generation_forbidden)
        assert result.skip_reason == "resume_reused_existing_stage_result"
        decision = read_json(_attempt_decision_path(runner, result_path.parent))
        assert decision["historical_stage_source"] == str(original)
        assert decision["previous_measurement_provenance_preserved"] is True
        assert len(decision["verified_generation_runtime_outputs"]) == 5
        assert result_path.read_bytes() == original_stage_bytes
        assert (original / "artifact_index.json").read_bytes() == source_index_bytes

    # A newer snapshot and freshly registered bytes cannot approve drift.
    raw = f"models/{model}/benchmark_set/legacy_suite/benchmark_plan.json"
    changed = read_json(root / raw)
    changed["runs"][0]["precision"] = "int8"
    write_json(root / raw, changed)
    current_index[raw] = _descriptor(root, raw, model=model)
    runner.artifact_index["artifacts"] = list(current_index.values())
    with pytest.raises(ValueError, match="supplement_source_contract_changed"):
        runner._run_stage(model, "generate_benchmark_set", generation_forbidden)
    assert result_path.read_bytes() == original_stage_bytes


@pytest.mark.parametrize("model", ["yolo26m", "yolo26s"])
@pytest.mark.parametrize("stage", ["run_benchmarks", "hardware_smoke"])
def test_interrupted_reuse_preserves_indexed_previous_decision(tmp_path, monkeypatch, model, stage):
    from onnx_splitpoint_tool.workflow import runner as workflow

    monkeypatch.setattr(workflow, "package_build_snapshot", lambda: {"package_version": "2.91.2"})
    root = tmp_path / "run"
    options = workflow.WorkflowOptions(profile="fixture", out=str(tmp_path), run_id=root.name,
                                       resume=True, resume_missing=True)
    runner = workflow.EvaluationWorkflowRunner(options)
    runner.run_dir, runner.run_id = root, root.name
    runner.manifest_path, runner.artifact_index_path = root / "run_manifest.json", root / "artifact_index.json"
    runner.profile_id, runner.profile_payload = "fixture", {}
    runner.manifest = {"run_id": root.name, "tool_version": "2.91.1", "models": {}}
    runner._supplement_plan = {"generic_rows": [{"model_id": model, "action": "preserve"}],
                               "quality_companion_rows": []}
    stage_dir = root / f"models/{model}/stages/{stage}"
    old_decision = write_json(stage_dir / "resume_decision.json", {
        "schema": "onnx-splitpoint/evaluation-stage-resume-decision",
        "stage": stage, "model_id": model, "reusable": False,
        "reason": "supplement_missing_work_or_local_projection",
    })
    raw = old_decision.relative_to(root).as_posix()
    runner.artifact_index = {"artifacts": [_descriptor(root, raw, stage, model)]}
    write_json(runner.manifest_path, runner.manifest)
    write_json(runner.artifact_index_path, runner.artifact_index)
    result_path = write_json(stage_dir / "stage_result.json", {
        "status": "partial" if stage == "run_benchmarks" else "ok",
        "state": "completed", "complete": True, "input_hash": "original",
        "started_at": "2026-09-30T04:00:00+02:00", "finished_at": "2026-09-30T05:00:00+02:00",
        "artifacts": [raw],
    })
    original_bytes = {path: path.read_bytes()
                      for path in (old_decision, result_path, runner.artifact_index_path)}
    original_index = copy.deepcopy(runner.artifact_index)

    def dispatch_forbidden():
        pytest.fail("previously completed stage must not dispatch")

    decisions = []
    # No terminal index closure occurs between the two session entries.
    for session in ("a" * 32, "b" * 32):
        runner.session_id = session
        result = runner._run_stage(model, stage, dispatch_forbidden)
        assert result.skip_reason == "resume_reused_existing_stage_result"
        path = _attempt_decision_path(runner, stage_dir)
        decisions.append(path)
        assert read_json(path)["reusable"] is True
        assert path.relative_to(root).as_posix() in result.artifacts
        assert all(path.read_bytes() == before for path, before in original_bytes.items())
        assert runner.artifact_index == original_index
    assert decisions[0] != decisions[1]
    assert all(path.is_file() for path in decisions)

    write_json(old_decision, {"tampered_historical_decision": True})
    with pytest.raises(ValueError, match="unbound_or_tampered:.*resume_decision.json"):
        runner._run_stage(model, stage, dispatch_forbidden)
    assert result_path.read_bytes() == original_bytes[result_path]
    assert runner.artifact_index == original_index
