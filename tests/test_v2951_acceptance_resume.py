"""Ordinary finite Native Resume retains verified slots and physical budgets."""
import copy
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance as coordinator
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from test_v295_full_hardware_acceptance import profile
from test_v295_numerical_product_acceptance import proof as numerical_proof


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def test_resume_is_confined_to_explicit_a4_native_selection(tmp_path):
    request = profile(tmp_path)
    options = WorkflowOptions("profile.yaml", str(tmp_path), resume=True, execution_mode="generate_and_run")
    assert len(coordinator.validate_request(request, options)["cells"]) == 14
    request["native_producers"]["deepx_runtime_mode"] = "S1"
    with pytest.raises(ValueError, match="shortcut"):
        coordinator.validate_request(request, options)


def test_same_run_prepared_identity_is_immutable_but_debt_is_reobserved(tmp_path):
    original = {"output": str(tmp_path / "point"), "source_plan_sha256": "a" * 64,
                "phase_energy_admission": {"prior_chains": 0}}
    native._publish_prepared(original, {})
    payload = copy.deepcopy(original)
    payload["phase_energy_admission"]["prior_chains"] = 2
    assert native._publish_prepared(payload, {"resume": True}) == original
    assert native._read(tmp_path / "point/PREPARED.json") == original
    payload["source_plan_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="preparation changed"):
        native._publish_prepared(payload, {"resume": True})
    missing = {"output": str(tmp_path / "not_started_point")}
    assert native._publish_prepared(missing, {"resume": True}) == missing


@pytest.fixture
def ordinary_point(tmp_path, monkeypatch):
    run = tmp_path / "phase/hardware_runs/resnet50__setup/current"
    output = run / "native/resnet50__setup__b045"
    output.mkdir(parents=True)
    identity = {"model": "resnet50", "case": "b045", "setup_id": "setup", "backend": "deepx_to_trt"}
    contract = {**identity, "contract_sha256": "a" * 64, "runtime_options": {"deepx_runtime_mode": "A4", "deepx_max_inflight": 4},
                "artifacts": {"dxnn": {"sha256": "b" * 64}, "engine": {"sha256": "c" * 64}}}
    matrix = write(tmp_path / "phase/FINITE_MATRIX.json", {"execution_limits": {
        "performance_technical_replacements_per_cell_max": 1, "performance_technical_replacements_global_max": 6}})
    prepared = {"schema": "onnx-splitpoint/hardware-acceptance-native-plan/v1", "identity": identity,
        "output": str(output), "deepx_runtime_mode": "A4", "source_contract": contract,
        "target": {"remote": {"host": "192.0.2.1", "user": "nx"}}, "target_binding": {},
        "remote_tool_dir": "/home/nx/tool", "full_baseline_references": [], "full_baseline_proofs": [],
        "phase_matrix": {"path": str(matrix), "sha256": native._sha(matrix)}, "runs": []}
    for name in ("source_plan", "source_raw_report", "source_validation"):
        path = write(tmp_path / (name + ".json"), {})
        prepared[name], prepared[name + "_sha256"] = str(path), native._sha(path)
    for repeat in (1, 2, 3):
        remote = f"/home/nx/fresh/rep{repeat:02d}"
        prepared["runs"].append({"repeat": repeat, "argv": ["/venv/bin/python", "worker.py", "--out-dir", remote],
            "env": {}, "remote_output_root": remote, "result_json": remote + "/result.json"})
    write(output / "PREPARED.json", prepared)
    monkeypatch.setattr(native, "verify_native_command_contract", lambda value, **kwargs: (value, "verified"))
    monkeypatch.setattr(native, "validate_latency", lambda value: value)
    from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
    monkeypatch.setattr(numerical, "_owned_dispatch_closure", lambda *args: {"proven": True})
    monkeypatch.setattr(numerical, "_owned_preparation_closure", lambda *args: {"proven": True}, raising=False)
    attempts = []
    def materialize(item, *, checkpoint=True):
        suffix = Path(item["remote_output_root"]).name.removeprefix(f"rep{item['repeat']:02d}")
        directory = output / (f"rep{item['repeat']:02d}" + suffix)
        from onnx_splitpoint_tool.native_deepx_async import NativeExecutionLifecycle
        lifecycle = NativeExecutionLifecycle()
        lifecycle.construction_enter()
        lifecycle.construction_return()
        for _ in range(1100):
            lifecycle.inference_enter()
            lifecycle.inference_return()
        raw = {"ok": True, "native_command_contract": contract, "completed_work_units": 1000,
            "warmup_count": 100, "postprocess_completion_verified": True, "makespan_ms": 1000., "fps_makespan": 1000.,
            "runtime_instance_id": f"runtime-{item['repeat']}{suffix}", "deepx_final_drain_complete": True,
            "deepx_execution_lifecycle": lifecycle.report(),
            "deepx_runtime_observation": {"pending": 0, "unreleased": 0},
            "request_latency": {"status": "complete", "count": 1000, "task_complete": True}}
        raw_path = write(directory / "download" / Path(item["remote_output_root"]).name / "result.json", raw)
        write(directory / "SDK_COMPLETION.json", {"device_completion_proven": True,
                                                  "owned_dispatch_closure": {"proven": True}})
        if checkpoint:
            proof = native._validate_raw(raw, identity, contract)
            proof.update(report=str(raw_path), report_sha256=native._sha(raw_path), repeat=item["repeat"],
                         runtime_instance_id=raw["runtime_instance_id"])
            write(directory / "VALIDATED.json", proof)
        return raw_path
    def retain(number):
        for item in prepared["runs"][:number]:
            materialize(item)
            attempts.append({"repeat": item["repeat"], "argv": item["argv"], "environment": {},
                             "started_at": 1., "completed_at": 2., "returncode": 0, "technical_replacement": False})
        write(output / "DISPATCH_ATTEMPTS.json", {"attempts": attempts})
    return SimpleNamespace(prepared=prepared, output=output, run=run, attempts=attempts, materialize=materialize, retain=retain)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_resume_revalidates_each_existing_valid_performance_slot(ordinary_point, count):
    ordinary_point.retain(count)
    state = native._same_run_performance_slots(ordinary_point.prepared)
    assert set(state["slots"]) == set(range(1, count + 1))
    path = Path(state["slots"][1]["report"])
    path.write_text('{"ok":false}')
    with pytest.raises(ValueError, match="changed"):
        native._same_run_performance_slots(ordinary_point.prepared)


@pytest.mark.parametrize("retained", [1, 2, 3])
def test_normal_resume_dispatches_only_missing_performance_slots(ordinary_point, monkeypatch, retained):
    point = ordinary_point
    point.retain(retained)
    write(point.output / "production-start.json", {"original": True})
    dispatched = []
    class Transport:
        def run(self, command, **kwargs):
            dispatched.append(command)
            return 0, "normal worker completed"
    monkeypatch.setattr(native, "_transport", lambda *args: Transport())
    monkeypatch.setattr(native, "_verify_current_target", lambda *args: {})
    monkeypatch.setattr(native, "ensure_current_deepx_runtime", lambda *args: None)
    monkeypatch.setattr(native, "_stage_benchmark_set", lambda *args: None)
    monkeypatch.setattr(native, "_collect_repeat", lambda transport, item, directory: point.materialize(item, checkpoint=False))
    monkeypatch.setattr(native, "physical_dut_key", lambda *args: "dut:test")
    monkeypatch.setattr(native.EvaluationRunLock, "for_resource", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(native, "bind_workflow_resource_options", lambda **kwargs: nullcontext())
    summary = write(point.output / "reports/native_producer_combined_summary.json", {"rows": []})
    monkeypatch.setattr(native, "_normal_performance_import", lambda *args: summary)
    result_path = write(point.output / "energy_result.json", {})
    import_path = write(point.output / "energy_import.json", {})
    monkeypatch.setattr(native, "_energy_capture", lambda *args: {"row": {}, "new_capture_count": 0,
        "retained_valid_repeats": 3, "valid_repeats": 3, "failed_attempts": 0,
        "start_accounting": {}, "energy_result": str(result_path), "import": str(import_path)})
    monkeypatch.setattr(native, "_reference_energy_pairs", lambda *args: {"missing_full_energy_comparisons": []})
    context = {"execute_allowed": True, "resume": True, "run_dir": str(point.run),
               "remote_process_registry": SimpleNamespace(active_count=lambda: 0)}
    result = native._run_native(point.prepared, context)
    assert len(dispatched) == result["new_performance_processes"] == 3 - retained
    assert result["retained_fresh_performance_processes"] == retained
    assert result["retained_valid_energy_repeats"] == 3 and result["new_capture_count"] == 0
    assert result["fresh_hardware_executed"] is bool(3 - retained)
    assert native._read(point.output / "production-start.json") == {"original": True}


def test_fresh_profile_cannot_reset_existing_phase_performance_starts(ordinary_point):
    point = ordinary_point
    point.retain(1)
    write(point.output / "PREPARED.json", point.prepared)
    other = copy.deepcopy(point.prepared)
    other["output"] = str(point.run.parent / "another_profile_name/native" / point.output.name)
    with pytest.raises(ValueError, match="Resume its original Run"):
        native._admit_performance_attempt(other, {})


def test_known_completed_raw_leaf_recovers_missing_validation_checkpoint(ordinary_point):
    point = ordinary_point
    point.retain(1)
    checkpoint = point.output / "rep01/VALIDATED.json"
    checkpoint.unlink()
    assert native._same_run_performance_slots(point.prepared)["slots"] == {}
    state = native._same_run_performance_slots(point.prepared, recover_completed=True)
    assert set(state["slots"]) == {1} and checkpoint.is_file()


@pytest.mark.parametrize("parent_symlink", [False, True])
def test_completed_raw_recovery_rejects_symlink_escape(ordinary_point, tmp_path, parent_symlink):
    point = ordinary_point
    point.retain(1)
    (point.output / "rep01/VALIDATED.json").unlink()
    raw = point.output / "rep01/download/rep01/result.json"
    if parent_symlink:
        original = raw.parent
        moved = tmp_path / "other_generation"
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
    else:
        moved = tmp_path / "other_generation.json"
        raw.rename(moved)
        raw.symlink_to(moved)
    with pytest.raises(ValueError, match="escapes"):
        native._same_run_performance_slots(point.prepared, recover_completed=True)


def imported_point(point):
    point.retain(3)
    slots = native._same_run_performance_slots(point.prepared)["slots"]
    reports = [Path(slots[index]["report"]) for index in sorted(slots)]
    summary = write(point.output / "reports/native_producer_combined_summary.json", {"rows": [{
        **point.prepared["identity"], "repetition_count_valid": 3,
        "native_command_contract": point.prepared["source_contract"]}]})
    receipt = {"returncode": 0, "summary": {"path": str(summary), "sha256": native._sha(summary)},
        "source_reports": [{"path": str(path), "sha256": native._sha(path)} for path in reports],
        "command_contract_sha256": point.prepared["source_contract"]["contract_sha256"]}
    write(point.output / "performance_import.json", receipt)
    return summary, receipt


@pytest.mark.parametrize("change", ["summary_bytes", "stale_s1", "other_contract", "source_report"])
def test_resume_import_rejects_stale_or_unbound_summary(ordinary_point, change):
    point = ordinary_point
    summary, receipt = imported_point(point)
    assert native.verify_performance_import(point.prepared)["command_contract_sha256"] == "a" * 64
    data = native._read(summary)
    if change == "source_report":
        receipt["source_reports"][0]["sha256"] = "f" * 64
    else:
        if change == "stale_s1":
            data["rows"][0]["native_command_contract"]["runtime_options"] = {"deepx_runtime_mode": "S1", "deepx_max_inflight": 1}
        elif change == "other_contract":
            data["rows"][0]["native_command_contract"]["contract_sha256"] = "f" * 64
        else:
            data["unbound_field"] = True
        write(summary, data)
        if change != "summary_bytes":
            receipt["summary"]["sha256"] = native._sha(summary)
    write(point.output / "performance_import.json", receipt)
    with pytest.raises(ValueError, match="binding|A4|accepted performance"):
        native.verify_performance_import(point.prepared)


def materialize_energy(point):
    root = point.output / "energy_view/reports/native_energy_measurements"
    contract = write(root / "commands/command_contract.json", point.prepared["source_contract"])
    plan = {**point.prepared["identity"], "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
        "successful_command_contract_sha256": "a" * 64, "command_contract_file": str(contract),
        "command_contract_file_sha256": native._sha(contract)}
    source = write(root / "native_producer_energy_results.json", {"rows": [{"row": plan}]})
    row = {**point.prepared["identity"], "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
        "source": str(source), "command_contract_source": str(contract), "ok": True, "energy_repeat_valid_n": 3,
        "energy_calibration_verified": True, "environment_observation_status": "pass"}
    return source, plan, row


@pytest.mark.parametrize("change", ["row_s1", "plan_s1", "different_contract"])
def test_resume_energy_must_match_a4_performance_contract(ordinary_point, change):
    point = ordinary_point
    source, plan, row = materialize_energy(point)
    native._verify_a4_energy_row(point.prepared, row, "a" * 64)
    if change == "row_s1":
        row.update(deepx_runtime_mode="S1", deepx_max_inflight=1)
    elif change == "plan_s1":
        plan.update(deepx_runtime_mode="S1", deepx_max_inflight=1)
        write(source, {"rows": [{"row": plan}]})
    else:
        plan["successful_command_contract_sha256"] = "f" * 64
        write(source, {"rows": [{"row": plan}]})
    with pytest.raises(ValueError, match="A4"):
        native._verify_a4_energy_row(point.prepared, row, "a" * 64)


def test_same_run_resume_retains_real_complete_slots_without_relabeling_new_starts(ordinary_point, monkeypatch):
    point = ordinary_point
    summary, _ = imported_point(point)
    _, _, row = materialize_energy(point)
    monkeypatch.setattr(native, "collect_native_energy", lambda *args, **kwargs: [row])
    monkeypatch.setattr(native, "_budget_counts", lambda *args: {"collector_starts": 4, "valid_repeats": 3})
    write(point.output / "energy_dispatch.json", {"start_accounting_before": {"collector_starts": 4, "valid_repeats": 3}})
    write(point.output / "PERFORMANCE_RESUME.json", {"dispatch_count_before": 3})
    context = {"resume": True, "run_dir": str(point.run), "prepared_native": point.prepared}
    result = {"performance_summary": str(summary), "new_capture_count": 0, "new_performance_processes": 0,
              "retained_valid_energy_repeats": 3, "fresh_hardware_executed": False}
    assert native.verify_same_run_resume_result(result, context)
    result["new_capture_count"] = False
    with pytest.raises(ValueError, match="relabels"):
        native.verify_same_run_resume_result(result, context)


@pytest.mark.parametrize("change", [None, "failed", "raw_drift"])
def test_numerical_resume_revalidates_and_retains_unchanged_pass(numerical_proof, change):
    cell, prepared, context, payload, old_path = numerical_proof
    prepared["output"] = str(old_path.parent)
    write(old_path.parent / "PREPARED.json", prepared)
    payload.update(fresh_hardware_executed=True, new_performance_processes=0, new_capture_count=0,
                   scientific_claim_eligible=False, scientific_cohort_complete=False, scientific_statistics_published=False,
                   evidence_paths=[str(old_path.parent / "worker.json")])
    if change == "failed":
        payload["status"] = "failed"
    path = write(old_path.parent / "NUMERICAL_RESULT.json", payload)
    before = path.read_bytes()
    context.update(run_dir=str(old_path.parent.parent.parent), resume=True, acceptance_phase="numerical",
                   prepared_native=prepared, acceptance_cell=cell)
    if change == "raw_drift":
        Path(payload["worker_report"]["path"]).write_text('{"ok":false}')
    if change:
        with pytest.raises(ValueError):
            coordinator._retained_numerical_result(cell, prepared, context)
    else:
        result = coordinator._retained_numerical_result(cell, prepared, context)
        assert coordinator._accepted_numerical_result(result, context) == result
        assert result["fresh_hardware_executed"] is False and result["new_numerical_worker_processes"] == 0
        assert result["same_run_numerical_retained"] is True and result["smoke_completed_count"] == 64
        result["mixed_completed_count"] = 31
        with pytest.raises(ValueError, match="projection"):
            coordinator._accepted_numerical_result(result, context)
    assert path.read_bytes() == before


def test_historical_performance_replacement_ceiling_does_not_block_documented_repair(ordinary_point):
    point = ordinary_point
    point.retain(1)
    point.attempts.append({"repeat": 2, "technical_replacement": True})
    write(point.output / "DISPATCH_ATTEMPTS.json", {"attempts": point.attempts})
    before = (point.output / "DISPATCH_ATTEMPTS.json").read_bytes()
    native._admit_performance_attempt(point.prepared, {}, replacement=True)
    assert (point.output / "DISPATCH_ATTEMPTS.json").read_bytes() == before


@pytest.mark.parametrize("state", ["source_stop", "task_stop", "transport_limit"])
def test_new_run_cannot_reset_prior_a4_source_or_task_stop(ordinary_point, state):
    point = ordinary_point
    source = "192.0.2.5"
    point.prepared["target_binding"]["urecs_address"] = source
    origin = point.run.parent / "earlier_gui_run"
    budget_path = origin / "energy/energy_task_budget.json"
    identity = point.prepared["identity"]
    cell = {"backend": identity["backend"], "model_id": identity["model"],
            "case_id": identity["case"], "setup_id": identity["setup_id"]}
    context = {"budget_file": str(budget_path), "environment": {}, "source_ids": {identity["setup_id"]: source}}
    config = {"regular_budget": {"repetitions": 3, "task_budget": {"max_retries": 2, "max_transport_failures": 20}}}
    coordinator._initialize_budget(context, config, cell, "native")
    budget = native._read(budget_path)
    if state == "source_stop":
        budget["sources"][source]["stop_reason"] = "campaign_source_transport_failure_limit"
    elif state == "task_stop":
        next(iter(budget["tasks"].values()))["stop_reason"] = "task_transport_failure_limit"
    else:
        budget["sources"][source]["transport_failures"] = 20
    write(budget_path, budget)
    with pytest.raises(RuntimeError, match="source/task STOP or transport limit"):
        native._native_phase_debt(point.prepared, point.run, point.prepared["phase_matrix"], None)


def test_cohort_authorization_is_exact_scope_once_and_keeps_limits(tmp_path):
    payload = profile(tmp_path)
    config = coordinator.validate_request(payload)
    config["cells"] = config["cells"][:2]
    run = tmp_path / "normal_gui_run"
    run.mkdir()
    context = {"run_dir": str(run), "budget_file": str(run / "energy/budget.json"),
        "source_ids": {"orin_nx_deepx_m1_01": "192.0.2.5"}, "environment": {},
        "finite_config": config, "acceptance_phase": "complete",
        "finite_request_binding": {"profile_id": "saved_gui_selection", "profile_hash": "original"}}
    for cell in config["cells"]:
        coordinator._initialize_budget(context, config, cell, "native")
    prepared = {"target_binding": {"urecs_address": "192.0.2.5"}}
    grant = native._authorize_energy_cohort(prepared, context)
    assert native._authorize_energy_cohort(prepared, context) == grant
    assert len(grant["row_ids"]) == 2 and grant["cohort_id"] == run.name
    root = native._read(context["budget_file"])
    assert all(row["limits"]["max_chains"] == 9 and not row["chains"] for row in root["tasks"].values())
    config["cells"] = config["cells"][:1]
    with pytest.raises(ValueError, match="different authorization"):
        native._authorize_energy_cohort(prepared, context)


def test_private_full_contract_uses_exact_bound_three_head_signature(tmp_path):
    source = tmp_path / "original"
    payloads = {"benchmark_set.json": b"{}", "models/yolov7_paper.onnx": b"onnx",
        "deepx/deepx_m1/full/model.dxnn": b"dxnn", "b044/split_manifest.json": b"{}", "timing.jpg": b"image"}
    for name, value in payloads.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    sha = lambda name: hashlib.sha256(payloads[name]).hexdigest()
    contract = {"model": "yolov7_paper", "input_case": "b044", "source_model_sha256": sha("models/yolov7_paper.onnx"),
        "input_image": str(source / "timing.jpg"), "input_image_sha256": sha("timing.jpg"),
        "artifacts": {"dxnn": {"sha256": sha("deepx/deepx_m1/full/model.dxnn")}},
        "energy_workload": {"source_endpoint_contract_hash": "a" * 64, "input_contract": {
            "model_id": "yolov7_paper", "artifact_sha256": sha("deepx/deepx_m1/full/model.dxnn"),
            "input": {"shape": [640, 640, 3]}, "outputs": [{"name": "model_outputs", "shape": None}]}}}
    original = copy.deepcopy(contract)
    tensors = [{"index": i, "name": name, "shape": [1, 3, size, size, 85], "rank": 5, "dtype": "float32"}
               for i, (name, size) in enumerate(zip(("model_outputs", "output_1", "output_2"), (80, 40, 20)))]
    attestation = {"attested": True, "status": "passed", "endpoint_contract_hash": "a" * 64,
                   "tensor_signature": {"tensor_count": 3, "tensors": tensors}}
    native._copy_full_benchmark(source, tmp_path / "fresh", contract, attestation)
    staged = native._read(tmp_path / "fresh/deepx/deepx_m1/full/output_contract.json")
    assert [item["name"] for item in staged["outputs"]] == [item["name"] for item in tensors]
    assert staged["source_output_endpoint_attestation"] == attestation and contract == original
    attestation["endpoint_contract_hash"] = "f" * 64
    with pytest.raises(ValueError, match="bound endpoint"):
        native._copy_full_benchmark(source, tmp_path / "mismatched", contract, attestation)
