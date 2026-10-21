"""Normal workflow Native refresh selection and honest cohort reuse accounting."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance as acceptance
from onnx_splitpoint_tool.workflow import native_refresh as refresh
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return acceptance.file_binding(path)


@pytest.fixture
def selection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows, cells = [], []
    for model, count in (("mobile", 17), ("regnet", 20), ("resnet", 19), ("y11", 3), ("y26m", 3), ("y26s", 3), ("y7", 3)):
        for case in ["full"] + [f"b{i:03d}" for i in range(count)]:
            backend = "native_full_deepx" if case == "full" else "deepx_to_trt"
            row = {"runner": "native", "setup_id": "DeepX", "model_id": model, "case_id": case,
                   "backend": backend, "kind": "full" if case == "full" else "split"}
            row["uid"] = f"native|{model}|DeepX|{backend}|{case}"
            rows.append(row)
            cells.append({"model_id": model, "setup_id": "orin_nx_deepx_m1_01", "case_id": case,
                          "backend": backend, "runtime_variants": ["native"]})
    snapshot = write(source / "cases.json", rows)
    matrix = write(tmp_path / "phase/FINITE_MATRIX.json", {"schema": refresh.MATRIX_SCHEMA,
        "recorded_before_hardware_start": "2026-10-15T09:00:00Z", "source_snapshot": snapshot, "cells": cells})
    images = write(source / "one_image.json", {"images": ["original.jpg"], "source_images": [{"local_path": "original.jpg"}]})
    for cell in cells:
        cell["numerical_images"] = images
    profile = {"name": "refresh", "model_suite": {"primary": [{"id": "regnet"}]},
        "hardware": {"selected_setups": ["orin_nx_deepx_m1_01"]},
        "native_producers": {"enabled": True, "backends": ["deepx"], "split_backends": ["deepx_to_trt"],
            "deepx_runtime_mode": "A4", "build_missing_engines": False,
            "full_baselines": {"active_producers": ["deepx"], "backends_by_producer": {"deepx": ["deepx"]}},
            "frames": 1000, "warmup": 100, "repetitions": 3,
            "energy": {"duration_s": 60, "physical_scope": "FS", "window_label": "command",
                "task_budget": {"enabled": True, "max_retries": 2, "max_transport_failures": 20}}},
        "native_refresh": {"enabled": True, "schema": refresh.SCHEMA, "source_report_root": str(source),
            "source_primary_run": str(source), "source_snapshot": snapshot, "matrix_evidence": matrix,
            "cells": [cell for cell in cells if cell["model_id"] == "regnet"]}}
    options = WorkflowOptions("profile.yaml", str(tmp_path / "output"), execution_mode="generate_and_run")
    return profile, options


def test_multiple_original_cuts_are_grouped_and_normal_resume_is_allowed(selection):
    profile, options = selection
    options.resume = True
    request = refresh.validate_request(profile, options)
    assert len(request["cells"]) == 21
    assert request["regular_budget"]["repetitions"] == 3
    assert options.resume is True
    assert "21" in refresh.summary_lines(profile)[0]


def test_acceptance_one_split_rule_is_unchanged(selection):
    profile, _ = selection
    profile["hardware_acceptance"] = {**profile.pop("native_refresh"), "schema": acceptance.SCHEMA}
    with pytest.raises(ValueError, match="one selected split"):
        acceptance.validate_request(profile)


@pytest.mark.parametrize("change", ["missing_cut", "duplicate_cut", "different_cut", "Generic", "GPU", "S1", "build", "shortened_recipe", "second_model"])
def test_scope_or_recipe_cannot_change(selection, change):
    profile, options = copy.deepcopy(selection)
    cells = profile["native_refresh"]["cells"]
    if change == "missing_cut":
        cells.pop()
    elif change == "duplicate_cut":
        cells[-1] = copy.deepcopy(cells[-2])
    elif change == "different_cut":
        cells[-1]["case_id"] = "b999"
    elif change == "Generic":
        cells[-1]["runtime_variants"] = ["generic_completed"]
    elif change == "GPU":
        cells[0]["backend"] = "native_full_tensorrt"
    elif change == "S1":
        profile["native_producers"]["deepx_runtime_mode"] = "S1"
    elif change == "build":
        profile["native_producers"]["build_missing_engines"] = True
    elif change == "shortened_recipe":
        profile["native_producers"]["frames"] = 10
    else:
        cells[-1]["model_id"] = "resnet"
    with pytest.raises(ValueError):
        refresh.validate_request(profile, options)


def test_numerical_failure_never_starts_regular_measurements(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
    images = write(tmp_path / "images.json", {"images": ["image"]})
    monkeypatch.setattr(numerical, "run_numerical_native", lambda *args: {"status": "failed", "new_capture_count": 0})
    calls = []
    result = refresh.execute_point({"numerical_images": images}, {"output": str(tmp_path)}, {}, lambda *args: calls.append(args))
    assert result["status"] == "failed" and calls == []


def test_same_job_numerical_result_is_validated_before_regular_execute(tmp_path, monkeypatch):
    write(tmp_path / "NUMERICAL_RESULT.json", {"status": "numerical_passed"})
    order = []
    monkeypatch.setattr(acceptance, "_verify_numerical_reference", lambda *args: order.append("validate") or {"status": "numerical_passed"})
    result = refresh.execute_point({}, {"output": str(tmp_path)}, {}, lambda *args: order.append("execute") or {"status": "completed"})
    assert result["status"] == "completed" and order == ["validate", "execute"]


def test_verified_reuse_reports_zero_new_calls_and_revalidates_proof(tmp_path, monkeypatch):
    prepared = {"output": str(tmp_path), "identity": {"model": "m", "setup_id": "s", "case": "full", "backend": "native_full_deepx"}}
    write(tmp_path / "PREPARED.json", prepared)
    proof = {"reference": {"path": "original", "sha256": "f" * 64}, "original_result": {"evidence_paths": ["original-raw"]}}
    monkeypatch.setattr(refresh, "_cohort_proof", lambda *args: proof)
    result = refresh.execute_point({"cohort_reference": proof["reference"]}, prepared, {}, lambda *args: pytest.fail("reuse must not execute"))
    assert result["new_capture_count"] == result["new_performance_processes"] == 0
    assert result["fresh_hardware_executed"] is False
    assert result["original_evidence_paths"] == ["original-raw"]
    assert refresh.accepted_result(result, {"run_dir": str(tmp_path)})["status"] == "ok"
    with pytest.raises(ValueError, match="physical start accounting"):
        refresh.accepted_result({**result, "new_capture_count": 3}, {"run_dir": str(tmp_path)})


def test_reuse_requires_actual_normal_raw_3plus3_not_sum_or_smoke(tmp_path):
    result = write(tmp_path / "RESULT.json", {"status": "completed", "valid_repeats": {"performance": 3, "energy": 3}})
    with pytest.raises(ValueError, match="original normal point"):
        refresh._cohort_proof({"model_id": "m", "setup_id": "s", "case_id": "full", "backend": "native_full_deepx"},
                              {"identity": {"model": "m", "setup_id": "s", "case": "full", "backend": "native_full_deepx"}}, result)


def test_refresh_uses_existing_normal_executor(selection, monkeypatch):
    profile, options = selection
    runner = SimpleNamespace(profile_payload=profile, options=options)
    monkeypatch.setattr(acceptance, "run_acceptance", lambda actual, **kwargs: (actual, kwargs))
    actual, arguments = refresh.run_refresh(runner)
    assert actual is runner and arguments["workflow_kind"] == "native_refresh"
    assert len(arguments["config"]["cells"]) == 21


@pytest.fixture
def completed_cohort(tmp_path, monkeypatch):
    """Real closed-file layout; raw/collector validators retain their own tests."""
    identity = {"model": "m", "setup_id": "orin_nx_deepx_m1_01", "case": "b001", "backend": "deepx_to_trt"}
    cell = {"model_id": "m", "setup_id": identity["setup_id"], "case_id": "b001", "backend": "deepx_to_trt"}
    run = tmp_path / "closed_normal_gui_job"
    output = run / "native" / "m__orin_nx_deepx_m1_01__b001"
    prepared = {"identity": identity, "deepx_runtime_mode": "A4", "source_contract": {"contract_sha256": "source"},
        "source_plan": "original-plan", "source_plan_sha256": "plan-hash", "source_raw_report_sha256": "raw-hash",
        "source_validation": "original-quality", "source_validation_sha256": "quality-hash",
        "target_binding": {"dut": "original-dut"}, "performance_recipe": {"processes": 3},
        "energy_recipe": {"duration_s": 60}, "full_baseline_references": [], "remote_tool_dir": "/original-tool",
        "output": str(output), "runs": [{"repeat": n, "argv": ["python", "worker", "--out-dir", f"/remote/rep{n:02d}"],
            "remote_output_root": f"/remote/rep{n:02d}", "result_json": f"/remote/rep{n:02d}/raw.json", "env": {}} for n in (1, 2, 3)]}
    write(output / "PREPARED.json", prepared)
    from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata
    profile = {"name": "ordinary_gui", "model_suite": {"primary": [{"id": "m"}]},
        "native_producers": {"enabled": True, "deepx_runtime_mode": "A4", "warmup": 100, "frames": 1000, "repetitions": 3},
        "hardware_acceptance": {"enabled": True, "cells": [{**cell, "runtime_variants": ["native"]}]}}
    metadata = public_start_snapshot_metadata(build_profile_start_snapshot(profile_request="original-profile.yaml",
        source_profile=profile, resolved_profile=profile, profile_id="ordinary_gui", profile_path="original-profile.yaml", profile_source="file"))
    write(run / "profile.yaml", profile)
    write(run / "profile_source.yaml", profile)
    write(run / "profile_start_snapshot.json", metadata)
    origin = {"kind": "gui", "job_id": "actual-job", "app_pid": 123, "window_id": 567, "widget_class": "Tk", "visible": True}
    write(run / "run_manifest.json", {"run_id": run.name, "status": "ok", "hardware_acceptance": {"source_campaign_resumed": False},
        "profile_start_snapshot": metadata, "start_origin": origin, "execution_sessions": [{"status": "ok", "start_origin": origin}]})
    write(run / "reports/artifact_index_closure.json", {"schema": "onnx-splitpoint/artifact-index-terminal-closure",
        "status": "pass", "run_id": run.name, "workflow_status": "ok"})
    write(output / "reports/native_producer_combined_summary.json", {"rows": [{**identity, "repetition_count_valid": 3,
        "native_command_contract": {"contract_sha256": "a" * 64, "runtime_options": {"deepx_runtime_mode": "A4", "deepx_max_inflight": 4}}}]})
    for repeat in (1, 2, 3):
        report = output / f"rep{repeat:02d}/download/raw.json"
        binding = write(report, {"runtime_instance_id": f"process-{repeat}", "native_command_contract": {"contract_sha256": "a" * 64, "runtime_options": {"deepx_runtime_mode": "A4", "deepx_max_inflight": 4}}})
        write(output / f"rep{repeat:02d}/VALIDATED.json", {"report": str(report), "report_sha256": binding["sha256"],
            "repeat": repeat, "runtime_instance_id": f"process-{repeat}", "contract_sha256": "a" * 64,
            "contract": {"runtime_options": {"deepx_runtime_mode": "A4", "deepx_max_inflight": 4}}})
    write(output / "performance_import.json", {"returncode": 0, "summary": acceptance.file_binding(output / "reports/native_producer_combined_summary.json"), "source_reports": [acceptance.file_binding(output / f"rep{n:02d}/download/raw.json") for n in (1, 2, 3)], "command_contract_sha256": "a" * 64})
    write(output / "DISPATCH_ATTEMPTS.json", {"attempts": [{"repeat": item["repeat"], "argv": item["argv"],
        "environment": {}, "returncode": 0, "completed_at": 1.0} for item in prepared["runs"]]})
    command_binding = write(output / "energy_view/reports/native_energy_measurements/command_contract.json", {
        "contract_sha256": "a" * 64, "runtime_options": {"deepx_runtime_mode": "A4", "deepx_max_inflight": 4}})
    energy_source = output / "energy_view/reports/native_energy_measurements/native_producer_energy_results.json"
    write(energy_source, {"rows": [{"row": {
        **identity, "successful_command_contract_sha256": "a" * 64, "duration_s": 60, "warmup": 0,
        "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
        "command_contract_file": command_binding["path"], "command_contract_file_sha256": command_binding["sha256"],
        "measurement_effective_repeats": 3}}]})
    reference = write(output / "RESULT.json", {"status": "completed", "normal_projection": True,
        "valid_repeats": {"performance": 3, "energy": 3}, "fps_median": 123, "evidence_paths": [str(output)]})
    monkeypatch.setattr(refresh.native, "_validate_raw", lambda raw, *a, **k: {"runtime_instance_id": raw["runtime_instance_id"],
        "contract_sha256": "a" * 64, "contract": {"runtime_options": {"deepx_runtime_mode": "A4", "deepx_max_inflight": 4}}})
    monkeypatch.setattr(refresh.native, "_deepx_completion_safety", lambda raw: {"device_completion_proven": True})
    energy_row = {**identity, "ok": True, "energy_repeat_valid_n": 3, "deepx_runtime_mode": "A4",
        "deepx_max_inflight": 4, "source": str(energy_source), "command_contract_source": command_binding["path"],
        "energy_calibration_verified": True, "environment_observation_status": "pass"}
    monkeypatch.setattr(refresh.native, "collect_native_energy", lambda *a, **k: [energy_row])
    return cell, prepared, reference, output, energy_row


def test_complete_actual_layout_is_revalidated_per_uid(completed_cohort):
    cell, prepared, reference, _, _ = completed_cohort
    proof = refresh._cohort_proof(cell, prepared, reference)
    assert proof["uid"] == "native|m|DeepX|deepx_to_trt|b001"
    assert len(proof["performance_reports"]) == len(set(proof["runtime_instance_ids"])) == 3


@pytest.mark.parametrize("origin", [None, {"kind": "cli", "profile_driven": True},
    {"kind": "gui", "visible": False}])
def test_ordinary_cohort_origin_is_observational(completed_cohort, origin):
    cell, prepared, reference, output, _ = completed_cohort
    path = output.parent.parent / "run_manifest.json"
    manifest = json.loads(path.read_text())
    if origin is None:
        manifest.pop("start_origin")
        manifest["execution_sessions"][0].pop("start_origin")
    else:
        manifest["start_origin"] = origin
        manifest["execution_sessions"][0]["start_origin"] = origin
    path.write_text(json.dumps(manifest))
    proof = refresh._cohort_proof(cell, prepared, reference)
    key = "GUI_origin" if origin and origin.get("kind") == "gui" else "workflow_origin"
    assert proof[key]["context"] == (origin or {})
    assert len(proof["performance_reports"]) == len(set(proof["runtime_instance_ids"])) == 3
    assert proof["valid_repeats"] == {"performance": 3, "energy": 3}
    if key == "workflow_origin":
        assert "GUI_origin" not in proof


@pytest.mark.parametrize("sessions", [[], [{"status": "running"}], [{"status": "failed"}]])
def test_cohort_requires_completed_workflow_session_without_origin(completed_cohort, sessions):
    cell, prepared, reference, output, _ = completed_cohort
    path = output.parent.parent / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("start_origin")
    manifest["execution_sessions"] = sessions
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completed ordinary workflow execution session"):
        refresh._cohort_proof(cell, prepared, reference)


def test_complete_cohort_retains_normal_technical_replacement_leaf(completed_cohort):
    cell, prepared, reference, output, _ = completed_cohort
    source = prepared["runs"][1]
    retry = refresh.native._performance_retry_item(source)
    original = output / "rep02/VALIDATED.json"
    validation = json.loads(original.read_text())
    report = output / "rep02_retry01/download/raw.json"
    report.parent.mkdir(parents=True)
    Path(validation["report"]).rename(report)
    validation["report"] = str(report)
    write(output / "rep02_retry01/VALIDATED.json", validation)
    original.unlink()
    dispatch_path = output / "DISPATCH_ATTEMPTS.json"
    dispatch = json.loads(dispatch_path.read_text())
    dispatch["attempts"][1]["returncode"] = 1
    dispatch["attempts"].append({"repeat": 2, "argv": retry["argv"], "environment": {}, "returncode": 0,
        "completed_at": 2.0, "technical_replacement": True})
    dispatch_path.write_text(json.dumps(dispatch))
    receipt_path = output / "performance_import.json"
    receipt = json.loads(receipt_path.read_text()); receipt["source_reports"][1] = acceptance.file_binding(report)
    receipt_path.write_text(json.dumps(receipt))
    proof = refresh._cohort_proof(cell, prepared, reference)
    assert "rep02_retry01" in proof["performance_reports"][1]["path"]


@pytest.mark.parametrize("without_gui_origin", [False, True], ids=["gui", "cli"])
@pytest.mark.parametrize("change", ["identity", "input", "energy_missing", "energy_S1", "energy_calibration", "same_process", "changed_raw", "wrong_duration", "wrong_command", "snapshot_changed", "stale_summary", "changed_summary", "wrong_import_sources", "energy_max1", "energy_max_missing", "energy_contract_drift"])
def test_wrong_or_incomplete_cohort_is_not_reused(completed_cohort, change, without_gui_origin):
    cell, prepared, reference, output, energy = completed_cohort
    if without_gui_origin:
        path = output.parent.parent / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest.pop("start_origin")
        manifest["execution_sessions"][0].pop("start_origin")
        path.write_text(json.dumps(manifest))
    if change == "identity":
        prepared["identity"] = {**prepared["identity"], "case": "b002"}
    elif change == "input":
        prepared["source_contract"]["input_image_sha256"] = "different-image"
    elif change == "energy_missing":
        energy["energy_repeat_valid_n"] = 2
    elif change == "energy_S1":
        energy["deepx_runtime_mode"] = "S1"
    elif change == "energy_calibration":
        energy["energy_calibration_verified"] = False
    elif change == "energy_max1":
        energy["deepx_max_inflight"] = 1
    elif change == "energy_max_missing":
        energy.pop("deepx_max_inflight")
    elif change == "energy_contract_drift":
        path = Path(energy["command_contract_source"])
        payload = json.loads(path.read_text()); payload["runtime_options"]["deepx_max_inflight"] = 1
        path.write_text(json.dumps(payload))
    elif change in {"same_process", "changed_raw"}:
        path = output / "rep02/download/raw.json"
        raw = json.loads(path.read_text());raw["runtime_instance_id"] = "process-1";path.write_text(json.dumps(raw))
        if change == "same_process":
            validation_path = output / "rep02/VALIDATED.json"
            validation = json.loads(validation_path.read_text());validation.update(report_sha256=acceptance.file_binding(path)["sha256"],runtime_instance_id="process-1")
            validation_path.write_text(json.dumps(validation))
    elif change == "snapshot_changed":
        write(output.parent.parent / "profile.yaml", {"name": "different-profile"})
    elif change in {"stale_summary", "changed_summary", "wrong_import_sources"}:
        summary_path = output / "reports/native_producer_combined_summary.json"
        receipt_path = output / "performance_import.json"
        receipt = json.loads(receipt_path.read_text())
        if change == "wrong_import_sources":
            receipt["source_reports"] = list(reversed(receipt["source_reports"]))
        else:
            summary = json.loads(summary_path.read_text())
            summary["rows"][0]["native_command_contract"]["contract_sha256"] = "b" * 64
            summary_path.write_text(json.dumps(summary))
            if change == "stale_summary":
                receipt.update(summary=acceptance.file_binding(summary_path), command_contract_sha256="b" * 64)
        receipt_path.write_text(json.dumps(receipt))
    else:
        path = output / "energy_view/reports/native_energy_measurements/native_producer_energy_results.json"
        payload = json.loads(path.read_text());payload["rows"][0]["row"]["duration_s" if change == "wrong_duration" else "successful_command_contract_sha256"] = 30 if change == "wrong_duration" else "unpaired-command"
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        refresh._cohort_proof(cell, prepared, reference)


def test_gui_origin_is_observational_and_does_not_change_resume_contract():
    from onnx_splitpoint_tool.workflow.run_control import stable_resume_options
    options = WorkflowOptions("profile.yaml", "/output")
    prior = stable_resume_options(options)
    options.gui_start_context = {"kind": "gui", "job_id": "different-resume-GUI-job"}
    assert stable_resume_options(options) == prior


@pytest.mark.parametrize("profile_driven", [False, True])
def test_cli_records_its_actual_entrypoint_without_gui_fields(monkeypatch, capsys, profile_driven):
    import os
    from onnx_splitpoint_tool.workflow import profile_options, run_evaluation
    from onnx_splitpoint_tool.workflow.run_control import stable_resume_options
    from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot

    profile = {"name": "cli_observation", "model_suite": {"primary": []}, "run_profiles": []}
    snapshot = build_profile_start_snapshot(profile_request="cli.yaml", source_profile=profile,
        resolved_profile=profile, profile_id="cli_observation", profile_path="cli.yaml", profile_source="file")
    monkeypatch.setattr(profile_options, "load_runtime_profile_snapshot", lambda request: (profile, snapshot))
    observed = {}

    class Runner:
        def __init__(self, options, **kwargs):
            observed["options"] = options

        def run(self):
            return SimpleNamespace(ok=True, status="ok", to_dict=lambda: {"ok": True, "status": "ok"})

    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", Runner)
    arguments = ["--profile", "cli.yaml", "--out", "/unused/output", "--json"]
    if profile_driven:
        arguments.insert(0, "--profile-driven")
    assert run_evaluation.main(arguments) == 0, capsys.readouterr().out
    options = observed["options"]
    assert options.gui_start_context == {"kind": "cli", "entrypoint": "onnx_splitpoint_tool.workflow.run_evaluation",
        "profile_driven": profile_driven, "pid": os.getpid()}
    assert stable_resume_options(options) == stable_resume_options(WorkflowOptions(**options.to_dict()))
    if profile_driven:
        assert options.profile_start_snapshot == snapshot
