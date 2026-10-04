"""Selective continuation through normal stage and quality queue boundaries."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import runner as workflow
from onnx_splitpoint_tool.workflow.artifacts import sha256_file
from onnx_splitpoint_tool.workflow.supplement import (
    build_supplement_plan, completed_quality, merge_preserved_results,
    preserve_metadata, result_key,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "run"
    model = "yolo26m"
    required = []
    for case in ("b001", "b002", "b003"):
        required.append(dict(model_id=model, expected_setup_id="setup", run_id="hailo8_to_trt",
                             case_id=case, variant="split", logical_identity_sha256=case))
    suite = root / "models" / model / "benchmark_set/legacy_suite"
    cases = []
    for case in ("b001", "b002", "b003"):
        write(suite / case / "split_manifest.json", {"io": {"part2_external_inputs":
              ["boundary", "images"] if case == "b003" else ["boundary"]}})
        cases.append({"folder": case})
    cases[-1]["hailo_compile"] = {"hailo8": {"part1": {"ok": False, "error": "Parsing failed. The errors found in the graph are: unsupported input"}}}
    write(suite / "benchmark_set.json", {"cases": cases})
    write(root / "effective_execution_plan.json", {"models": [model]})
    write(root / "models" / model / "benchmark_set/required_run_scope.json", {"identities": required})
    measured = {**required[0], "setup_id": "setup", "fps": 23, "tool_version": "2.91.0"}
    write(root / "reports/scientific/performance_results.json", [measured])
    request = root / "models" / model / "benchmark_results/quality_inputs/setup/results/b001/task_quality_inputs/composed_request.json"
    candidate = write(request.with_name("candidate.json"), {"predictions": [1]})
    write(request, {"model_id": model, "case_id": "b001", "candidate": {
        "path": candidate.name, "size_bytes": candidate.stat().st_size,
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()}})
    quality = {"model_id": model, "case_id": "b001", "source_request": str(request.relative_to(root)),
               "source_request_sha256": sha256_file(request), "status": "completed",
               "technical_status": "completed", "decision": "fail", "tool_version": "2.91.0"}
    write(root / "quality_management/central_quality_summary.json", {"results": [quality]})
    native = [dict(model=model, case=case, execution_mode="native_split", setup_id="setup",
                   backend="hailo8_to_trt", actual_ok=case == "b001") for case in ("b001", "b002", "b003")]
    write(root / "reports/native_expected_matrix.json", {"present_expected_rows": native})
    report = write(root / "native_producers/original_measurement.json", {"fps_makespan": 23})
    write(root / "reports/native_producer_summary.json", {"rows": [{
        **native[0], "ok": True, "report": str(report), "source_reports": [str(report)],
        "native_command_contract_sha256": "original-command", "fps_makespan": 23,
    }]})
    return root


def test_original_contract_projection_and_second_application(archive):
    plan = build_supplement_plan(archive)
    assert plan["counts"] == {"generic": {"preserve": 1, "execute": 1, "negative_completed": 1},
        "quality_companions": {}, "native": {"preserve_and_validate_locally": 1, "execute": 1, "not_supported": 1},
        "cohort": 3, "completed_quality": 1}
    kept = json.loads((archive / "reports/scientific/performance_results.json").read_text())
    old = copy.deepcopy(kept)
    new = {**plan["generic_rows"][1], "fps": 25, "tool_version": "2.91.1"}
    merged = merge_preserved_results(kept, [new])
    assert kept == old and merged[0] == old[0]
    write(archive / "reports/scientific/performance_results.json", merged)
    second = build_supplement_plan(archive)
    assert second["cohort"] == plan["cohort"]
    assert second["counts"]["generic"] == {"preserve": 2, "negative_completed": 1}
    assert second["completed_quality_count"] == 1


@pytest.mark.parametrize("damage", ["request", "candidate", "duplicate"])
def test_completed_quality_reuse_is_fail_closed(archive, damage):
    quality_path = archive / "quality_management/central_quality_summary.json"
    payload = json.loads(quality_path.read_text())
    request = archive / payload["results"][0]["source_request"]
    if damage == "request":
        request.write_text(request.read_text() + " ")
    elif damage == "candidate":
        request.with_name("candidate.json").write_text("altered")
    else:
        payload["results"].append(payload["results"][0]); write(quality_path, payload)
    with pytest.raises(ValueError, match="supplement_"):
        completed_quality(archive)


def test_history_copies_only_metadata_never_hardlinks_or_outputs(archive):
    raw = archive / "native_producers/payload.bin"
    raw.parent.mkdir(exist_ok=True); raw.write_bytes(b"keep measurement")
    history = preserve_metadata(archive, "a" * 32)
    source = archive / "reports/scientific/performance_results.json"
    target = history / source.relative_to(archive)
    assert source.read_bytes() == target.read_bytes()
    assert source.stat().st_ino != target.stat().st_ino
    assert not (history / "native_producers").exists()
    write(source, [])
    assert json.loads(target.read_text())


def make_runner(root, monkeypatch):
    monkeypatch.setattr(workflow, "package_build_snapshot", lambda: {"package_version": "2.91.1"})
    options = workflow.WorkflowOptions(profile="fixture", out=str(root.parent), resume=True,
                                       resume_missing=True, run_id=root.name)
    runner = workflow.EvaluationWorkflowRunner(options)
    runner.run_dir, runner.run_id = root, root.name
    runner.manifest_path, runner.artifact_index_path = root / "run_manifest.json", root / "artifact_index.json"
    runner.profile_id = "fixture"
    runner.profile_payload = {"quality_gate": {"statistics": {"execution_location": "central_management"}}}
    runner.manifest = {"models": {}}
    runner.artifact_index = {"artifacts": []}
    runner._supplement_plan = build_supplement_plan(root)
    runner._supplement_original_root = root / "jobs/supplement_attempts/original"
    return runner


def test_normal_quality_queue_retains_completed_fail_without_new_bootstrap(archive, monkeypatch):
    runner = make_runner(archive, monkeypatch)
    results, requests = completed_quality(archive)
    runner._preserved_central_quality_results = results
    runner._preserved_central_quality_requests = requests
    submitted = []
    runner._central_quality_coord_executor = SimpleNamespace(submit=lambda *args: submitted.append(args))
    monkeypatch.setattr(runner, "_ensure_central_quality_service", lambda: None)
    assert runner._queue_central_quality_requests("yolo26m") == 0
    assert submitted == [] and results[0]["decision"] == "fail"
    assert results[0]["tool_version"] == "2.91.0"


def test_normal_model_stages_reuse_across_patch_and_dispatch_only_missing(archive, monkeypatch):
    runner = make_runner(archive, monkeypatch)
    observed = []
    for stage in workflow.MODEL_STAGES:
        artifact = write(archive / "evidence" / (stage + ".json"), {"original": "2.91.0"})
        relative = str(artifact.relative_to(archive))
        runner.artifact_index["artifacts"].append({"path": relative, "size_bytes": artifact.stat().st_size,
                                                  "sha256": sha256_file(artifact)})
        write(archive / "models/yolo26m/stages" / stage / "stage_result.json", {
            "status": "ok", "state": "completed", "complete": True, "artifacts": [relative],
            "input_hash": "old-version", "details": {"resume_key_hash": "old-version", "original_tool_version": "2.91.0"}})
        def leaf(model_id, row, stage=stage, artifact=artifact):
            observed.append(stage)
            return {"evidence": artifact}, {}, "controlled process leaf", "ok"
        monkeypatch.setattr(runner, "_stage_" + stage, leaf)
    runner._run_model({"id": "yolo26m"})
    assert observed == ["run_benchmarks", "validate_outputs", "hardware_smoke"]
    assert "build_backend_artifacts" not in observed
    generated = json.loads((archive / "models/yolo26m/stages/generate_benchmark_set/stage_result.json").read_text())
    assert generated["details"]["original_tool_version"] == "2.91.0"
    assert generated["input_hash"] == "old-version"
    decision = json.loads((archive / "jobs/supplement_attempts" / runner.session_id /
                           "resume_decisions/models/yolo26m/stages/generate_benchmark_set/resume_decision.json").read_text())
    assert decision["previous_measurement_provenance_preserved"]
    # Re-entry after a completed addition does not invoke a measurement leaf.
    for row in runner._supplement_plan["generic_rows"]:
        if row["action"] == "execute": row["action"] = "preserve"
    observed.clear()
    runner._run_model({"id": "yolo26m"})
    assert not observed


def test_changed_archived_stage_artifact_is_not_rebuilt_silently(archive, monkeypatch):
    runner = make_runner(archive, monkeypatch)
    evidence = write(archive / "evidence.json", {"original": True})
    runner.artifact_index["artifacts"] = [{"path": "evidence.json", "size_bytes": evidence.stat().st_size,
                                          "sha256": sha256_file(evidence)}]
    write(evidence, {"modified": True})
    previous = {"state": "completed", "complete": True, "artifacts": ["evidence.json"]}
    with pytest.raises(ValueError, match="source_contract_changed"):
        runner._supplement_stage_decision({}, "yolo26m", "generate_benchmark_set", previous)
