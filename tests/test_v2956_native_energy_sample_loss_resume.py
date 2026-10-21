"""Normal Resume opens a finite loss-repair series without replacing slots."""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil

import pytest

from onnx_splitpoint_tool.energy import task_budget as budget
from onnx_splitpoint_tool.process_control import _proc_start_time
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.workflow import native_energy_reentry as reentry

REAL_RELEASE = budget._released_sample_loss_resources


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def loss(tmp_path, monkeypatch):
    phase = tmp_path / "phase"
    run = phase / "hardware_runs/m__setup/run"
    point = run / "native/m__setup__b008"
    view = point / "energy_view"
    measurement = view / "reports/native_energy_measurements"
    journal = run / "jobs/remote_process_leases/new"
    old = run / "jobs/remote_process_leases/old"
    row = '["deepx_to_trt","m","b008","setup"]'
    command_sha = "a" * 64
    environment = {"ONNX_SPLITPOINT_CONTROLLER_RESOURCES": "required",
        "ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID": run.name,
        "ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID": "new",
        "ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR": str(journal)}
    resources = ["controller:capture", "controller:nic", "dut:example", "source:meter"]
    config = {"run_dir": str(run), "run_id": run.name, "session_id": "new",
        "parent_pid": os.getpid(), "parent_start_ticks": _proc_start_time(os.getpid()),
        "setups": {"setup": resources}}
    write(journal / "controller.resources.json", config)
    write(old / "controller.resources.json", {**config, "session_id": "old", "parent_pid": 999999999, "parent_start_ticks": "1"})
    source = measurement / "native_producer_energy_results.partial.json"
    checkpoint = measurement / "checkpoints/native_energy/rows/row.json"
    write(source, {"rows": [{"ok": False}], "started_measurement_count": 1,
        "collector_started_repeat_count": 3, "workload_started_repeat_count": 3})
    write(checkpoint, {"state": "failed", "result": {"ok": False}})
    limits = {"max_chains": 9, "max_retries": 2, "max_transport_failures": 20}
    chains = [{"cohort_id": run.name, "logical_repeat": f"repeat:{min(i, 2)}",
        "run_directory": str(measurement / "measurements" / f"attempt_{i}"),
        "owner_pid": 999999999, "owner_start_ticks": "1", "preflight_requested": True,
        "collector_started": True, "workload_started": True, "finished": True,
        "valid": i < 2, "transport_reasons": [] if i < 2 else ["marker_dropped_samples_nonzero"],
        "source_completion_verified": True} for i in range(5)]
    root = {"campaign": True, "max_transport_failures": 20,
        "sources": {"meter": {"stop_reason": "", "transport_failures": 3}},
        "tasks": {row: {"limits": limits, "source_id": "meter", "chains": chains,
                          "stop_reason": "task_logical_retry_limit"}},
        "authorized_final_cohort": {"cohort_id": run.name, "parent_run_dir": str(run),
            "source_id": "meter", "row_ids": [row], "purpose": "final"}}
    path = run / "energy/energy_task_budget.json"
    write(path, root)
    matrix = phase / "FINITE_MATRIX.json"
    write(matrix, {})
    prepared = {"identity": {"backend": "deepx_to_trt", "model": "m", "case": "b008", "setup_id": "setup"},
        "output": str(point), "deepx_runtime_mode": "A4", "target_binding": {"urecs_address": "meter"},
        "phase_matrix": {"path": str(matrix), "sha256": native._sha(matrix)}}
    selected = measurement / "resume_attempts/resume_test"
    @contextmanager
    def released(chain, source_id, env):
        yield {"request_path": str(old / "capture.resource-request.json"), "request": {}, "reply": {},
               "resources": resources, "parent": budget._budget_parent(env)}
    monkeypatch.setattr(budget, "_released_sample_loss_resources", released)
    monkeypatch.setattr(budget, "source_completion", lambda path: {"verified": True})
    monkeypatch.setattr(budget, "_pure_sample_loss_failure", lambda *args, **kwargs: True)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    class Case:
        pass
    case = Case()
    case.__dict__.update(locals())
    return case


def authorize(case, **changes):
    kwargs = {"row_id": case.row, "command_contract_sha256": case.command_sha,
        "predecessor_result": case.source, "predecessor_checkpoint": case.checkpoint,
        "selected_view": case.selected, "environment": case.environment}
    kwargs.update(changes)
    return budget.authorize_native_resume_capture_series(case.path, **kwargs)


def test_series_keeps_original_stop_limits_and_two_accepted_slots(loss):
    original = copy.deepcopy(loss.root)
    old_files = {p: p.read_bytes() for p in (loss.source, loss.checkpoint)}
    series = authorize(loss)
    assert series["logical_repeats"] == ["repeat:2"] and series["max_starts"] == 3
    stored = json.loads(loss.path.read_text())
    task = stored["tasks"][loss.row]
    assert {key: task[key] for key in ("chains", "limits", "stop_reason")} == {
        key: original["tasks"][loss.row][key] for key in ("chains", "limits", "stop_reason")}
    assert stored["sources"] == original["sources"]
    assert all(path.read_bytes() == data for path, data in old_files.items())
    with loss.path.open("r+") as stream:
        current = budget.EnergyTaskBudget(stream, loss.limits, campaign_row=loss.row,
            source_id="meter", environment=loss.environment, cohort_id=loss.run.name)
        assert current.allowed("repeat:2")
        assert not current.allowed("repeat:0") and not current.allowed("repeat:1")
        assert current.authorized_retry_limit() is None  # collector keeps ordinary two retries
        for index in range(3):
            chain = current.reserve("repeat:2", loss.selected / f"capture_{index}", True)
            assert chain is not None
            current.collector_started(chain)
            current.finish(chain, {"status": "collector_finished", "collector_started": True,
                "valid": False, "ok": False, "command_window_request": {"errors": ["marker_dropped_samples_nonzero"]}},
                Path(chain["run_directory"]))
        assert not current.allowed("repeat:2")
        assert current.data["stop_reason"] == "task_logical_retry_limit"
        assert current.data["resume_capture_series"][-1]["stop_reason"] == "task_logical_retry_limit"


def test_callback_consumes_same_series_and_refuses_valid_or_fourth_slot(loss):
    authorize(loss)
    proof = native._native_phase_debt(loss.prepared, loss.run, loss.prepared["phase_matrix"], None,
                                     logical_repeat="repeat:2")
    assert proof["prior_and_current_chains"] == {"repeat:0": 1, "repeat:1": 1, "repeat:2": 3}
    assert proof["slot_chain_limits"]["repeat:2"] == 6
    with pytest.raises(RuntimeError, match="accepted energy slot"):
        native._native_phase_debt(loss.prepared, loss.run, loss.prepared["phase_matrix"], None, logical_repeat="repeat:0")
    root = json.loads(loss.path.read_text())
    for index in range(3):
        root["tasks"][loss.row]["chains"].append({**loss.chains[-1], "run_directory": str(loss.selected / f"capture_{index}")})
    write(loss.path, root)
    with pytest.raises(RuntimeError, match="retry/chain limit"):
        native._native_phase_debt(loss.prepared, loss.run, loss.prepared["phase_matrix"], None, logical_repeat="repeat:2")


def test_repeated_same_parent_authorization_is_idempotent(loss):
    first = authorize(loss)
    before = loss.path.read_bytes()
    assert authorize(loss) == first and loss.path.read_bytes() == before
    with pytest.raises(ValueError, match="different capture series"):
        authorize(loss, command_contract_sha256="b" * 64)


@pytest.mark.parametrize("change", ["source_stop", "unclosed", "non_loss", "foreign", "valid_missing"])
def test_non_recoverable_predecessors_cannot_authorize(loss, change):
    root = copy.deepcopy(loss.root)
    task = root["tasks"][loss.row]
    if change == "source_stop": root["sources"]["meter"]["stop_reason"] = "campaign_source_completion_unresolved"
    elif change == "unclosed": task["chains"][-1]["source_completion_verified"] = False
    elif change == "non_loss": task["chains"][-1]["transport_reasons"] = ["some_other_failure"]
    elif change == "foreign": task["chains"][-1]["run_directory"] = str(loss.run.parent / "foreign/capture")
    else: task["chains"][1]["valid"] = False
    write(loss.path, root)
    before = loss.path.read_bytes()
    with pytest.raises(ValueError): authorize(loss)
    assert loss.path.read_bytes() == before


def test_parent_mismatch_never_projects_permission(loss):
    authorize(loss)
    environment = {**loss.environment, "ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID": "other",
                   "ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR": str(loss.journal.parent / "other")}
    write(loss.journal.parent / "other/controller.resources.json", {**loss.config, "session_id": "other"})
    root = json.loads(loss.path.read_text())
    assert budget.native_resume_capture_series(root, loss.row, environment) is None
    assert budget.campaign_budget_stop_reason(root, loss.row, environment) == "task_logical_retry_limit"


def test_changed_predecessor_and_original_history_fail_closed(loss):
    authorize(loss)
    root = json.loads(loss.path.read_text())
    loss.source.write_text("{}")
    with pytest.raises(ValueError, match="predecessor changed"):
        budget.native_resume_capture_series(root, loss.row, loss.environment)


def test_selector_creates_fresh_view_without_mutating_failed_managed_row(loss, monkeypatch):
    command = loss.measurement / "plan/command.sh"
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("normal command\n")
    previous = {"rows": [{"ok": False}], "plan_payload": {"rows": [{"command_file": str(command)}]}}
    write(loss.source, previous)
    checkpoint = {"state": "failed", "result": previous["rows"][0]}
    verified = []
    monkeypatch.setattr(budget, "read_verified_campaign_repeat", lambda chain, **kwargs: verified.append(chain["logical_repeat"]))
    context = {"budget_file": str(loss.path), "run_dir": str(loss.run), "environment": loss.environment}
    original = loss.source.read_bytes(), loss.checkpoint.read_bytes()
    selected, predecessor = reentry._select_sample_loss_view(loss.prepared, context, loss.view,
        loss.source, checkpoint, loss.checkpoint, loss.command_sha, None)
    assert selected.parent == loss.measurement / "resume_attempts" and not selected.exists()
    assert predecessor == loss.source and verified == ["repeat:0", "repeat:1"]
    assert original == (loss.source.read_bytes(), loss.checkpoint.read_bytes())


def test_pending_series_continues_after_parent_exit_without_new_allowance(loss):
    original = authorize(loss)
    root = json.loads(loss.path.read_text())
    series = root["tasks"][loss.row]["resume_capture_series"][-1]
    series["parent"] = {**series["parent"], "pid": 999999997, "start_ticks": "1", "session_id": "exited"}
    for index in range(2):
        root["tasks"][loss.row]["chains"].append({**loss.chains[-1], "run_directory": str(loss.selected / f"capture_{index}")})
    write(loss.path, root)
    reentry._continue_sample_loss_view(loss.prepared, {"budget_file": str(loss.path), "environment": loss.environment},
                                       loss.selected, loss.command_sha)
    stored = json.loads(loss.path.read_text())["tasks"][loss.row]
    assert len(stored["resume_capture_series"]) == 1
    assert stored["resume_capture_series"][0]["original_chains"] == original["original_chains"]
    with loss.path.open("r+") as stream:
        current = budget.EnergyTaskBudget(stream, loss.limits, campaign_row=loss.row,
            source_id="meter", environment=loss.environment, cohort_id=loss.run.name)
        assert current.allowed("repeat:2")
        assert current._effective_limits("repeat:2")["max_retries"] == 5


def test_a_later_explicit_resume_can_open_another_bounded_series(loss):
    authorize(loss)
    root = json.loads(loss.path.read_text())
    task = root["tasks"][loss.row]
    task["resume_capture_series"][-1].update(stop_reason="task_logical_retry_limit",
        parent={**task["resume_capture_series"][-1]["parent"], "pid": 999999997, "start_ticks": "1", "session_id": "exited"})
    for index in range(3):
        task["chains"].append({**loss.chains[-1], "run_directory": str(loss.selected / f"capture_{index}")})
    write(loss.path, root)
    predecessor = loss.selected / "reports/native_energy_measurements/native_producer_energy_results.partial.json"
    checkpoint = loss.selected / "reports/native_energy_measurements/checkpoints/native_energy/rows/row.json"
    write(predecessor, {"rows": [{"ok": False}]})
    write(checkpoint, {"state": "failed"})
    next_view = loss.selected.parent / "resume_next"
    series = authorize(loss, predecessor_result=predecessor, predecessor_checkpoint=checkpoint, selected_view=next_view)
    assert series["max_starts"] == 3 and len(series["original_chains"]) == 8
    stored = json.loads(loss.path.read_text())["tasks"][loss.row]
    assert stored["limits"] == loss.limits and len(stored["resume_capture_series"]) == 2
    with loss.path.open("r+") as stream:
        current = budget.EnergyTaskBudget(stream, loss.limits, campaign_row=loss.row,
            source_id="meter", environment=loss.environment, cohort_id=loss.run.name)
        assert current.allowed("repeat:2")
        assert current._effective_limits("repeat:2") == {"max_retries": 8, "max_chains": 11, "max_transport_failures": 20}


@pytest.mark.parametrize("later", [False, True])
@pytest.mark.parametrize("mutation", ["prefix", "limits", "view", "predecessor"])
def test_corrupt_prior_series_never_changes_budget_during_parent_handoff(loss, later, mutation):
    authorize(loss)
    root = json.loads(loss.path.read_text())
    previous = root["tasks"][loss.row]["resume_capture_series"][-1]
    previous["parent"] = {**previous["parent"], "pid": 999999997, "start_ticks": "1", "session_id": "exited"}
    if later:
        previous["stop_reason"] = "task_logical_retry_limit"
        for index in range(3):
            root["tasks"][loss.row]["chains"].append({**loss.chains[-1], "run_directory": str(loss.selected / f"capture_{index}")})
    if mutation == "prefix": previous["original_chains"][0]["owner_start_ticks"] = "changed"
    elif mutation == "limits": previous["original_limits"]["max_retries"] = 3
    elif mutation == "view": previous["selected_view"] = str(loss.view / "escaped")
    else: previous["predecessor_result"]["sha256"] = "f" * 64
    write(loss.path, root)
    changes = {}
    if later:
        result = loss.selected / "reports/native_energy_measurements/native_producer_energy_results.partial.json"
        checkpoint = loss.selected / "reports/native_energy_measurements/checkpoints/native_energy/rows/row.json"
        write(result, {"ok": False}); write(checkpoint, {"state": "failed"})
        changes.update(predecessor_result=result, predecessor_checkpoint=checkpoint,
                       selected_view=loss.selected.parent / "resume_next")
    before = loss.path.read_bytes()
    with pytest.raises(ValueError): authorize(loss, **changes)
    assert loss.path.read_bytes() == before


@pytest.mark.parametrize("mutation", [None, "collector_alive", "lease_active", "quarantine", "foreign_capture"])
def test_exact_normal_release_proof_uses_original_session_and_four_resources(loss, monkeypatch, mutation):
    from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
    chain = loss.chains[-1]
    request = {"run_id": loss.run.name, "session_id": "old", "operation_id": "capture", "token": "token",
        "attempt_dir": chain["run_directory"], "purpose": "capture", "phase": "FINALIZING",
        "process_started": True, "setup_id": "setup", "owner_pid": chain["owner_pid"],
        "owner_start_ticks": chain["owner_start_ticks"], "collector_pid": 999999998, "collector_start_ticks": "1"}
    if mutation == "collector_alive": request.update(collector_pid=os.getpid(), collector_start_ticks=_proc_start_time(os.getpid()))
    if mutation == "foreign_capture": request["attempt_dir"] = str(loss.run / "foreign")
    reply = {key: request[key] for key in ("run_id", "session_id", "operation_id", "token")}
    reply.update(state="RELEASED", reason="")
    write(loss.old / "capture.resource-request.json", request)
    write(loss.old / "capture.resource-reply.json", reply)
    write(Path(chain["run_directory"]) / "collector_stdout.log.cleanup.json",
        {"owned_tree_quiescent": True, "process_started": True,
         "collector_pid": request["collector_pid"], "collector_start_ticks": request["collector_start_ticks"]})
    held = []
    quarantine = loss.run / "test_quarantine"
    if mutation == "quarantine": quarantine.touch()
    class Lease:
        previous_owner = {"state": "active" if mutation == "lease_active" else "released"}
        quarantine_path = quarantine
        def __init__(self, resource): self.resource = resource
        def acquire_for_recovery(self): held.append(self.resource)
        def release(self, **kwargs): pass
    monkeypatch.setattr(EvaluationRunLock, "for_resource", lambda resource, owner: Lease(resource))
    if mutation:
        with pytest.raises(ValueError):
            with REAL_RELEASE(chain, "meter", loss.environment): pass
    else:
        with REAL_RELEASE(chain, "meter", loss.environment) as proof:
            assert proof["request_path"] == str(loss.old / "capture.resource-request.json")
            assert proof["resources"] == sorted(loss.resources)
            assert held == sorted(loss.resources)


def managed_predecessor(loss, state):
    from onnx_splitpoint_tool.resources.remote_scripts import run_native_producer_energy_from_summary as runner
    from onnx_splitpoint_tool.workflow.checkpoints import AtomicRowJournal, write_stage_checkpoint
    command = loss.measurement / "plan/command.sh"
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text("normal command\n")
    row = {**loss.prepared["identity"], "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
        "comparison_backend": "deepx", "precision": "fp16", "duration_s": 60,
        "successful_command_contract_sha256": loss.command_sha, "command_file": str(command),
        "measurement_requested_repeats": 3, "measurement_effective_repeats": 3}
    plan = {"schema": "energy-plan", "schema_version": 1, "rows": [row], "physical_scope": "FS",
        "window_label": "command", "require_runtime_work_units": True, "require_command_window_alignment": True}
    plan_hash, identities, hashes = runner._managed_journal_contract([row], plan)
    root = loss.measurement / "checkpoints/native_energy"
    shutil.rmtree(root)  # this test fixture's skeletal checkpoint only
    journal = AtomicRowJournal.create(root, plan_hash=plan_hash, rows=[row], identities=identities, row_contract_hashes=hashes)
    valid_count = len({chain["logical_repeat"] for chain in loss.chains if chain.get("valid") is True})
    result = {"row": row, "ok": False, "run": {"rc": 1, "energy_repeat_valid_n": valid_count}}
    journal.transition(0, "running", execution={"output_dir": str(loss.measurement / "measurements/failed"),
                                               "measurement_wrapper_started": True})
    journal.transition(0, "failed", result=result)
    write(loss.source, {"plan_payload": plan, "rows": [result], "started_measurement_count": 1,
        "collector_started_repeat_count": 3, "workload_started_repeat_count": 3})
    write_stage_checkpoint(loss.measurement / "stages/native_energy/stage_result.json", stage="native_energy",
        state=state, complete=state == "failed", input_hash="b" * 64, run_root=loss.measurement)
    return root / "rows" / journal._row_filename(0, identities[0])


@pytest.mark.parametrize("state", ["running", "failed"])
def test_full_normal_selector_accepts_sealed_failed_row_and_distinct_start_counters(loss, monkeypatch, state):
    checkpoint = managed_predecessor(loss, state)
    context = {"resume": True, "finite_config": {"phase": "complete"}, "budget_file": str(loss.path),
               "run_dir": str(loss.run), "environment": loss.environment}
    verified = []
    monkeypatch.setattr(budget, "read_verified_campaign_repeat", lambda chain, **kwargs: verified.append(chain["logical_repeat"]))
    originals = {path: path.read_bytes() for path in loss.measurement.rglob("*") if path.is_file()}
    selected, previous = reentry.select_energy_view(loss.prepared, context, loss.view, loss.command_sha)
    assert selected.parent == loss.selected.parent and previous == loss.source
    assert verified == ["repeat:0", "repeat:1"]
    assert all(path.read_bytes() == data for path, data in originals.items())
    record = json.loads(loss.path.read_text())["tasks"][loss.row]["resume_capture_series"][-1]
    assert record["predecessor_checkpoint"]["path"] == str(checkpoint)


def test_normal_energy_dispatch_archives_predecessors_and_pending_resume_keeps_series(loss, monkeypatch):
    managed_predecessor(loss, "running")
    validation = loss.run / "validation.json"
    write(validation, {})
    loss.prepared.update(source_validation=str(validation), remote_tool_dir="/normal-tool",
        target={"remote": {"user": "nx", "host": "example.invalid"}}, runs=[{"env": {}}])
    write(loss.point / "PREPARED.json", loss.prepared)
    summary = loss.point / "reports/native_producer_combined_summary.json"
    write(summary, {"rows": [{**loss.prepared["identity"], "native_command_contract": {"contract_sha256": loss.command_sha}}]})
    write(loss.point / "energy_dispatch.json", {"command": ["normal-runner", "--out-dir", str(loss.measurement)], "returncode": 1})
    write(loss.point / "RESULT.json", {"status": "failed"})
    originals = {name: (loss.point / name).read_bytes() for name in ("energy_dispatch.json", "RESULT.json")}
    monkeypatch.setattr(native, "verify_performance_import", lambda *args: {"command_contract_sha256": loss.command_sha})
    monkeypatch.setattr(native, "_native_phase_debt", lambda *args: {})
    monkeypatch.setattr(native, "_budget_counts", lambda *args: {"collector_starts": 5, "valid_repeats": 2})
    monkeypatch.setattr(native, "_authorize_energy_cohort", lambda *args: {"cohort_id": loss.run.name})
    monkeypatch.setattr(budget, "read_verified_campaign_repeat", lambda *args, **kwargs: {})
    class ControlledStop(Exception): pass
    commands = []
    def dispatch(command, *args, **kwargs):
        commands.append(command)
        raise ControlledStop
    monkeypatch.setattr(native, "_run_local", dispatch)
    context = {"resume": True, "finite_config": {"phase": "complete"}, "budget_file": str(loss.path),
        "run_dir": str(loss.run), "environment": loss.environment, "python": "unused-python",
        "hardware_registry": "unused-registry", "profile": {}}
    with pytest.raises(ControlledStop): native._energy_capture(loss.prepared, summary, context)
    command = commands[0]
    selected = Path(command[command.index("--out-dir") + 1]).parent.parent
    assert "--resume-checkpoint" not in command and "--previous-results" not in command
    assert command[command.index("--campaign-max-retries") + 1] == "2"
    assert command[command.index("--campaign-budget-file") + 1] == str(loss.path)
    assert all((selected / "resume_history" / name).read_bytes() == data for name, data in originals.items())
    before = loss.path.read_bytes()
    assert reentry.select_energy_view(loss.prepared, context, loss.view, loss.command_sha) == (selected, loss.source)
    assert loss.path.read_bytes() == before


@pytest.fixture
def multi_loss(loss):
    old = loss.point
    new = old.with_name("m__setup__b037")
    old.rename(new)
    for name in ("point", "view", "measurement", "source", "checkpoint", "selected"):
        setattr(loss, name, new / getattr(loss, name).relative_to(old))
    loss.prepared["identity"]["case"] = "b037"
    loss.prepared["output"] = str(new)
    old_row = loss.row
    loss.row = '["deepx_to_trt","m","b037","setup"]'
    loss.root["tasks"][loss.row] = loss.root["tasks"].pop(old_row)
    loss.root["authorized_final_cohort"]["row_ids"] = [loss.row]
    for chain, logical, valid in zip(loss.chains, (0, 1, 2, 0, 0), (False, False, True, False, False)):
        chain.update(logical_repeat=f"repeat:{logical}", valid=valid,
            transport_reasons=[] if valid else ["marker_dropped_samples_nonzero"],
            run_directory=str(new / Path(chain["run_directory"]).relative_to(old)))
    loss.root["sources"]["meter"]["transport_failures"] = 4
    write(loss.path, loss.root)
    return loss


def test_actual_b037_missing_set_has_separate_offsets_and_six_total_starts(multi_loss):
    case = multi_loss
    original = copy.deepcopy(case.root)
    series = authorize(case)
    assert series["logical_repeats"] == ["repeat:0", "repeat:1"]
    assert series["max_starts"] == 3 and series["max_total_starts"] == 6
    with case.path.open("r+") as stream:
        current = budget.EnergyTaskBudget(stream, case.limits, campaign_row=case.row,
            source_id="meter", environment=case.environment, cohort_id=case.run.name)
        assert current.allowed("repeat:0") and current.allowed("repeat:1")
        assert not current.allowed("repeat:2")
        assert current._effective_limits("repeat:0") == {"max_retries": 5, "max_chains": 11, "max_transport_failures": 20}
        assert current._effective_limits("repeat:1") == {"max_retries": 3, "max_chains": 11, "max_transport_failures": 20}
    proof = native._native_phase_debt(case.prepared, case.run, case.prepared["phase_matrix"], None,
                                     logical_repeat="repeat:1")
    assert proof["slot_chain_limits"] == {"repeat:0": 6, "repeat:1": 4, "repeat:2": 3}
    assert proof["case_chain_limit"] == 11
    stored = json.loads(case.path.read_text())
    assert stored["tasks"][case.row]["chains"] == original["tasks"][case.row]["chains"]
    assert stored["tasks"][case.row]["limits"] == case.limits
    assert stored["tasks"][case.row]["stop_reason"] == "task_logical_retry_limit"
    assert stored["sources"] == original["sources"]


def test_multi_slot_grant_never_lends_unused_slot_allowance_to_another(multi_loss):
    case = multi_loss
    authorize(case)
    root = json.loads(case.path.read_text())
    for index in range(4):
        root["tasks"][case.row]["chains"].append({**case.chains[-1], "run_directory": str(case.selected / f"capture_{index}")})
    write(case.path, root)
    with pytest.raises(ValueError, match="original contract changed"):
        budget.native_resume_capture_series(root, case.row, case.environment)


def test_missing_set_stays_fixed_when_first_missing_slot_succeeds_before_interruption(multi_loss):
    case = multi_loss
    original = authorize(case)
    root = json.loads(case.path.read_text())
    series = root["tasks"][case.row]["resume_capture_series"][-1]
    series["parent"] = {**series["parent"], "pid": 999999997, "start_ticks": "1", "session_id": "exited"}
    root["tasks"][case.row]["chains"].append({**case.chains[-1], "logical_repeat": "repeat:0", "valid": True,
        "transport_reasons": [], "run_directory": str(case.selected / "new_valid_0")})
    write(case.path, root)
    reentry._continue_sample_loss_view(case.prepared, {"budget_file": str(case.path), "environment": case.environment},
                                       case.selected, case.command_sha)
    stored = json.loads(case.path.read_text())["tasks"][case.row]
    assert len(stored["resume_capture_series"]) == 1
    assert stored["resume_capture_series"][0]["logical_repeats"] == ["repeat:0", "repeat:1"]
    assert stored["resume_capture_series"][0]["original_chains"] == original["original_chains"]
    with case.path.open("r+") as stream:
        current = budget.EnergyTaskBudget(stream, case.limits, campaign_row=case.row,
            source_id="meter", environment=case.environment, cohort_id=case.run.name)
        assert not current.allowed("repeat:0") and current.allowed("repeat:1")
        assert current._effective_limits("repeat:1")["max_retries"] == 3


def test_later_multi_slot_resume_uses_per_slot_exhaustion_when_other_slot_never_started(multi_loss):
    case = multi_loss
    authorize(case)
    root = json.loads(case.path.read_text())
    task = root["tasks"][case.row]
    previous = task["resume_capture_series"][-1]
    previous.update(stop_reason="task_logical_retry_limit",
        parent={**previous["parent"], "pid": 999999997, "start_ticks": "1", "session_id": "exited"})
    for index in range(3):
        task["chains"].append({**case.chains[-1], "run_directory": str(case.selected / f"new_loss_0_{index}")})
    write(case.path, root)
    result = case.selected / "reports/native_energy_measurements/native_producer_energy_results.partial.json"
    checkpoint = case.selected / "reports/native_energy_measurements/checkpoints/native_energy/rows/row.json"
    write(result, {"ok": False}); write(checkpoint, {"state": "failed"})
    series = authorize(case, predecessor_result=result, predecessor_checkpoint=checkpoint,
                       selected_view=case.selected.parent / "resume_next")
    assert series["logical_repeats"] == ["repeat:0", "repeat:1"] and series["max_total_starts"] == 6
    assert len(json.loads(case.path.read_text())["tasks"][case.row]["resume_capture_series"]) == 2


@pytest.mark.parametrize("accepted_count", [0, 1])
def test_full_selector_preserves_general_incomplete_slot_set(multi_loss, monkeypatch, accepted_count):
    case = multi_loss
    if accepted_count == 0:
        case.chains[2].update(valid=False, transport_reasons=["marker_dropped_samples_nonzero"])
        case.root["sources"]["meter"]["transport_failures"] = 5
        write(case.path, case.root)
    managed_predecessor(case, "running")
    verified = []
    monkeypatch.setattr(budget, "read_verified_campaign_repeat", lambda chain, **kwargs: verified.append(chain["logical_repeat"]))
    context = {"resume": True, "finite_config": {"phase": "complete"}, "budget_file": str(case.path),
               "run_dir": str(case.run), "environment": case.environment}
    selected, predecessor = reentry.select_energy_view(case.prepared, context, case.view, case.command_sha)
    assert predecessor == case.source and selected.parent == case.selected.parent
    assert verified == (["repeat:2"] if accepted_count else [])
    series = json.loads(case.path.read_text())["tasks"][case.row]["resume_capture_series"][-1]
    assert series["logical_repeats"] == (["repeat:0", "repeat:1"] if accepted_count else ["repeat:0", "repeat:1", "repeat:2"])
    assert series["max_total_starts"] == (3 - accepted_count) * 3


@pytest.mark.parametrize("scenario, valid_slots, missing_slots, failed_leaves", [
    (["success", "success", "drop", "drop", "drop", "success"], ["repeat:0", "repeat:1"], ["repeat:2"], 3),
    (["drop", "drop", "success", "drop", "drop", "success", "success"], ["repeat:2"], ["repeat:0", "repeat:1"], 4),
], ids=["b008_two_reused_one_new", "b037_one_reused_two_new"])
def test_portable_normal_collector_reuses_slots_and_imports_completed_point(tmp_path, monkeypatch, scenario, valid_slots, missing_slots, failed_leaves):
    import hashlib
    from contextlib import nullcontext
    from types import SimpleNamespace
    from onnx_splitpoint_tool.native_energy_reporting import collect_native_energy
    from onnx_splitpoint_tool.remote import process_lease
    from test_v2956_a4_raw_energy_cohort import OWNER, rig, write as write_binding
    from test_v283_campaign_energy_budget import measure
    from test_yolo_energy_partial_reporting import publish
    instance, options, identity = rig(tmp_path, monkeypatch)
    instance.scenario.write_text(json.dumps(scenario))
    fake_collector = instance.root / "fake_collector.py"
    fake_collector.write_text(fake_collector.read_text().replace(
        "marker['stream']['dropped_samples'] = 1",
        "marker['stream']['dropped_samples'] = 1\n    marker['valid_for_final_energy'] = False"))
    fake_power = instance.root / "fake_power.py"
    fake_power.write_text(fake_power.read_text().replace(
        "assert '--estimated-duration=3' in sys.argv",
        "assert any(value.startswith('--estimated-duration=') for value in sys.argv)"))
    point = instance.root / "native" / "__".join(identity[k] for k in ("model", "setup_id", "case"))
    view = point / "energy_view"
    measurement = view / "reports/native_energy_measurements"
    prepared = json.loads((instance.root / "native/PREPARED.json").read_text())
    prepared["output"] = str(point)
    descriptor = json.loads(Path(options["capture_admission_file"]).read_text())
    descriptor["prepared"] = write_binding(point / "PREPARED.json", prepared)
    binding = write_binding(point / "CAPTURE_ADMISSION.json", descriptor)
    options.update(capture_admission_file=binding["path"], capture_admission_sha256=binding["sha256"])
    initial_root = measurement / "measurements/original_attempt"
    initial = measure(instance, str(initial_root.relative_to(instance.root)), **options)
    assert not initial["ok"] and initial["task_budget"]["counts"]["valid_logical_repeats"] == len(valid_slots)
    assert initial["error"] == "task_logical_retry_limit"
    assert instance.counts() == {"preflight": 5, "collector": 5, "workload": 5}

    # The portable observer normally writes during-window samples only for a
    # valid marker. Supply its real-shaped samples for rejected fake captures,
    # preserving the marker rejection and frozen scientific summaries.
    for chain in initial["task_budget"]["chains"]:
        if chain["valid"]:
            continue
        leaf = Path(chain["run_directory"])
        entry = json.loads((leaf / "energy_summary.json").read_text())
        command = json.loads((leaf / "collector_storage/command_window_markers.json").read_text())["command"]
        preflight = entry["preflight_evidence"]
        sample = {"phase": "during", "controller_request_start_ns": command["start_realtime_ns"],
            "controller_request_end_ns": command["end_realtime_ns"], "setup_id": identity["setup_id"],
            "run_id": OWNER, "capture_group": str(leaf.parent),
            "cpu_frequency": [{"scaling_cur_freq": 1234}], "host": {"loadavg": [0.1]},
            "workload_roots": [{"preflight_nonce": preflight["nonce"],
                "command_contract_sha256": preflight["expected_command_contract_sha256"],
                "workers": [{"pid": 123, "start_ticks": "456", "allowed_cpus": [0],
                    "threads": [{"tid": 123, "allowed_cpus": [0]}]}]}]}
        (leaf / "environment_samples.jsonl").write_text(json.dumps(sample) + "\n")
        assert budget._pure_sample_loss_failure(chain, options["row"], OWNER, native_a4=True)

    plan = {**identity, "campaign_cohort": OWNER, "diagnostic_only": True,
        "claim_eligible": False, "energy_quality_qualified": False, "measurement_effective_repeats": 3,
        "deepx_runtime_mode": "A4", "deepx_max_inflight": 4}
    predecessor = publish(view, [{"row": plan, "ok": False, "run": {"rc": 1,
        "energy_aggregate": initial, "energy_aggregate_embedded": True,
        "energy_aggregate_complete": False, "energy_aggregate_verified": False}}])
    checkpoint = measurement / "checkpoints/native_energy/rows/fixture.json"
    write_binding(checkpoint, {"state": "failed", "purpose": "portable ownership fixture"})
    budget_path = options["campaign_budget_file"]
    original_root = json.loads(budget_path.read_text())
    original_chains = copy.deepcopy(original_root["tasks"][options["row"]]["chains"])
    preserved = {p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in measurement.rglob("*") if p.is_file()}

    # Physical ownership is external to this local fake-process rig. The new
    # broker identity, finite-series reader, collector and importers stay real.
    new_journal = instance.root / "jobs/remote_process_leases/new"
    resources = ["controller:capture", "controller:nic", "dut:fixture", "source:" + instance.setup.urecs_address]
    config = {"run_id": OWNER, "run_dir": str(instance.root), "session_id": "new",
        "parent_pid": os.getpid(), "parent_start_ticks": _proc_start_time(os.getpid()),
        "setups": {identity["setup_id"]: resources}}
    write_binding(new_journal / "controller.resources.json", config)
    old_journal = new_journal.parent / "old"
    write_binding(old_journal / "controller.resources.json", {**config, "session_id": "old",
        "parent_pid": 999999999, "parent_start_ticks": "1"})
    env = {"ONNX_SPLITPOINT_CONTROLLER_RESOURCES": "required",
        "ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID": OWNER, "ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID": "new",
        "ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR": str(new_journal)}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    instance.env.update(env)
    parent = {"run_id": OWNER, "session_id": "new", "pid": os.getpid(), "start_ticks": _proc_start_time(os.getpid())}
    monkeypatch.setattr(budget, "_budget_parent", lambda environment=None: parent)
    monkeypatch.setattr(budget, "_chain_owner_alive", lambda chain: False)
    monkeypatch.setattr(process_lease, "controller_activity", lambda **kwargs: nullcontext(SimpleNamespace(operation=None)))
    monkeypatch.setattr(process_lease.ControllerResourceClaim, "from_environment", lambda **kwargs: None)

    @contextmanager
    def released(chain, source_id, environment):
        yield {"request_path": str(old_journal / "capture.resource-request.json"),
            "request": {}, "reply": {}, "resources": resources, "parent": parent}

    monkeypatch.setattr(budget, "_released_sample_loss_resources", released)
    selected = measurement / "resume_attempts/resume_portable"
    budget.authorize_native_resume_capture_series(budget_path, row_id=options["row"],
        command_contract_sha256="a" * 64, predecessor_result=predecessor,
        predecessor_checkpoint=checkpoint, selected_view=selected, environment=env)
    series = json.loads(budget_path.read_text())["tasks"][options["row"]]["resume_capture_series"][-1]
    assert series["logical_repeats"] == missing_slots
    assert series["max_starts"] == 3
    assert series["max_total_starts"] == 3 * len(missing_slots)
    before = instance.counts()
    repaired_root = selected / "reports/native_energy_measurements/measurements/new_attempt"
    repaired = measure(instance, str(repaired_root.relative_to(instance.root)),
        reuse_campaign_repeats=True, **options)
    assert repaired["ok"] and repaired["repeat_contract_complete"]
    assert instance.counts() == {key: value + len(missing_slots) for key, value in before.items()}
    assert [bool(entry.get("campaign_reused_repeat")) for entry in repaired["runs"]] == [f"repeat:{i}" in valid_slots for i in range(3)]
    original_valid = {chain["logical_repeat"]: chain for chain in original_chains if chain["valid"]}
    for slot in valid_slots:
        index = int(slot.split(":")[1])
        assert repaired["runs"][index]["campaign_reused_repeat"]["run_directory"] == original_valid[slot]["run_directory"]
    final_source = publish(selected, [{"row": plan, "ok": True, "run": {"rc": 0,
        "energy_aggregate": repaired, "energy_aggregate_embedded": True,
        "energy_aggregate_complete": True, "energy_aggregate_verified": True}}], partial=False)
    write_binding(point / "energy_dispatch.json", {"command": ["normal-runner", "--out-dir", str(final_source.parent)],
        "pre_capture_predecessor": str(predecessor)})
    assert reentry.selected_energy_view(prepared) == selected
    normalized = collect_native_energy(reentry.selected_energy_view(prepared))[0]
    assert normalized["ok"] and normalized["energy_repeat_valid_n"] == 3
    assert normalized["environment_observation_status"] == "pass"
    assert normalized["energy_calibration_verified"] is True
    assert normalized["diagnostic_only"] is True and normalized["claim_eligible"] is False
    assert native.validate_energy_view(selected, final_source)["status"] == "normal_completed_point_imported"
    stored = json.loads(budget_path.read_text())["tasks"][options["row"]]
    assert stored["chains"][:5] == original_chains and len(stored["chains"]) == 5 + len(missing_slots)
    assert sum(c["valid"] is False for c in stored["chains"]) == failed_leaves
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in preserved.items())
    (tmp_path / "independent_completion_result.json").write_text(json.dumps({"status": "pass",
        "fixture_only": True, "hardware_acquisitions": 0, "initial_local_fake_starts": before,
        "final_local_fake_starts": instance.counts(), "reused_slots": valid_slots,
        "new_local_fake_slots": missing_slots, "old_failed_leaves_preserved": failed_leaves,
        "normal_valid_slots": normalized["energy_repeat_valid_n"],
        "selected_view": str(selected)}, indent=2))
