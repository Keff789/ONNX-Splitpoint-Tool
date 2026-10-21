"""Documented technical repair resumes ordinary slots without a lifetime cap."""
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
from test_v2951_acceptance_resume import ordinary_point, write

_REAL_PREPARATION_CLOSURE = numerical._owned_preparation_closure
_REAL_DISPATCH_CLOSURE = numerical._owned_dispatch_closure


@pytest.fixture
def dispatch(ordinary_point, monkeypatch):
    point = ordinary_point
    commands, staged, quarantines = [], [], []
    class Lock:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def commit_quarantine_fence(self, value):
            quarantines.append(value)
    class Transport:
        diagnostics_dir = point.output / "diagnostics"
        host = SimpleNamespace(id="same-dut")
        def run(self, command, **kwargs):
            commands.append(command)
            write(self.diagnostics_dir / f"ssh_{len(commands):02d}.json", {
                "command": command, "target_id": self.host.id, "returncode": 0, "cleanup_proven": True})
            return 0, "normal worker returned"
    monkeypatch.setattr(native, "_transport", lambda *args: Transport())
    monkeypatch.setattr(native, "_verify_current_target", lambda *args: {})
    monkeypatch.setattr(native, "ensure_current_deepx_runtime", lambda *args: None)
    monkeypatch.setattr(native, "_stage_benchmark_set", lambda transport, item, directory: staged.append(directory))
    monkeypatch.setattr(native, "_collect_repeat", lambda transport, item, directory: point.materialize(item, checkpoint=False))
    monkeypatch.setattr(native, "physical_dut_key", lambda *args: "dut:test")
    monkeypatch.setattr(native.EvaluationRunLock, "for_resource", lambda *args, **kwargs: Lock())
    monkeypatch.setattr(native, "bind_workflow_resource_options", lambda **kwargs: nullcontext())
    summary = write(point.output / "reports/native_producer_combined_summary.json", {"rows": []})
    monkeypatch.setattr(native, "_normal_performance_import", lambda *args: summary)
    energy_result = write(point.output / "energy_result.json", {})
    energy_import = write(point.output / "energy_import.json", {})
    monkeypatch.setattr(native, "_energy_capture", lambda *args: {"row": {}, "new_capture_count": 0,
        "retained_valid_repeats": 3, "valid_repeats": 3, "failed_attempts": 0,
        "start_accounting": {}, "energy_result": str(energy_result), "import": str(energy_import)})
    monkeypatch.setattr(native, "_reference_energy_pairs", lambda *args: {"missing_full_energy_comparisons": []})
    context = {"execute_allowed": True, "resume": True, "run_dir": str(point.run),
               "remote_process_registry": SimpleNamespace(active_count=lambda: 0, cancelled=False, configured_scope="same-run")}
    return SimpleNamespace(point=point, context=context, commands=commands, staged=staged, quarantines=quarantines)


def fail_attempt(point, source, retry=0, *, completed=True):
    item = native._performance_retry_item(source, retry) if retry else source
    suffix = f"_retry{retry:02d}" if retry else ""
    directory = point.output / (f"rep{source['repeat']:02d}" + suffix)
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "original_failure.json", {"failure": retry})
    write(directory / "SDK_COMPLETION.json", {"device_completion_proven": completed,
                                               "owned_dispatch_closure": {"proven": completed}})
    point.attempts.append({"repeat": source["repeat"], "argv": item["argv"], "environment": item["env"],
        "started_at": 1. + retry, "completed_at": 2. + retry, "returncode": 1, "technical_replacement": retry > 0})
    write(point.output / "DISPATCH_ATTEMPTS.json", {"attempts": point.attempts})
    return directory


def test_second_repair_retains_slow_valid_slots_and_all_failed_originals(dispatch):
    state = dispatch
    point = state.point
    point.retain(2)
    # A valid but slower process is retained exactly; no speed selection occurs.
    raw = Path(native._same_run_performance_slots(point.prepared)["slots"][1]["report"])
    row = native._read(raw)
    row.update(makespan_ms=100000., fps_makespan=10.)
    write(raw, row)
    proof = native._validate_raw(row, point.prepared["identity"], point.prepared["source_contract"])
    proof.update(report=str(raw), report_sha256=native._sha(raw), repeat=1, runtime_instance_id=row["runtime_instance_id"])
    write(point.output / "rep01/VALIDATED.json", proof)
    base = fail_attempt(point, point.prepared["runs"][2])
    first = fail_attempt(point, point.prepared["runs"][2], 1)
    originals = {path: path.read_bytes() for directory in (point.output / "rep01", base, first)
                 for path in directory.rglob("*") if path.is_file()}
    before = copy.deepcopy(point.attempts)
    result = native._run_native(point.prepared, state.context)
    after = native._read(point.output / "DISPATCH_ATTEMPTS.json")
    assert after["attempts"][:len(before)] == before
    assert len(state.commands) == result["new_performance_processes"] == 1
    assert result["cumulative_performance_processes"] == 5
    assert result["retained_fresh_performance_processes"] == 2
    assert state.staged == [point.output / "rep03_retry02"]
    assert native._same_run_performance_slots(point.prepared)["slots"][1]["fps"] == 10.
    assert len(native._same_run_performance_slots(point.prepared)["slots"]) == 3
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_latest_failed_attempt_must_have_completion_not_only_base(dispatch):
    point = dispatch.point
    point.retain(2)
    fail_attempt(point, point.prepared["runs"][2])
    fail_attempt(point, point.prepared["runs"][2], 1, completed=False)
    before = native._read(point.output / "DISPATCH_ATTEMPTS.json")["attempts"]
    with pytest.raises(RuntimeError, match="normal recovery required"):
        native._run_native(point.prepared, dispatch.context)
    assert dispatch.commands == []
    assert native._read(point.output / "DISPATCH_ATTEMPTS.json")["attempts"] == before


def test_recorded_pre_worker_staging_failure_preserves_leaf_without_a_sdk_start(dispatch, monkeypatch):
    point = dispatch.point
    def broken_stage(*args):
        raise ValueError("concrete staging schema failure")
    monkeypatch.setattr(native, "_stage_benchmark_set", broken_stage)
    with pytest.raises(ValueError, match="staging schema failure"):
        native._run_native(point.prepared, dispatch.context)
    original = native._read(point.output / "DISPATCH_ATTEMPTS.json")
    assert original["attempts"] == [] and dispatch.commands == []
    failed = [item for item in original["preparation_attempts"] if item["phase"] == "pre_worker_staging"][0]
    assert failed["worker_dispatched"] is False and failed["owned_preparation_closure"]["proven"] is True
    assert dispatch.quarantines == []
    monkeypatch.setattr(native, "_stage_benchmark_set", lambda transport, item, directory: dispatch.staged.append(directory))
    result = native._run_native(point.prepared, dispatch.context)
    current = native._read(point.output / "DISPATCH_ATTEMPTS.json")
    assert failed in current["preparation_attempts"]
    assert current["attempts"][0]["argv"] == native._performance_retry_item(point.prepared["runs"][0])["argv"]
    assert result["new_performance_processes"] == result["cumulative_performance_processes"] == 3
    assert (point.output / "rep01").is_dir() and (point.output / "rep01_retry01/VALIDATED.json").is_file()


def test_unrecorded_orphan_preparation_is_preserved_unknown(dispatch):
    path = write(dispatch.point.output / "rep01/unclassified_failure.json", {"evidence": "unknown"})
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="previous preparation completion is unproven"):
        native._run_native(dispatch.point.prepared, dispatch.context)
    assert path.read_bytes() == before and dispatch.commands == []


@pytest.mark.parametrize("cleanup_proven", [True, False])
def test_staging_failure_uses_actual_shared_owned_cleanup_and_pre_worker_receipt(dispatch, monkeypatch, cleanup_proven):
    monkeypatch.setattr(numerical, "_owned_preparation_closure", _REAL_PREPARATION_CLOSURE)
    point = dispatch.point
    def staging(transport, item, directory):
        checkpoint = native._read(point.output / "DISPATCH_ATTEMPTS.json")
        assert checkpoint["attempts"] == []
        assert checkpoint["preparation_attempts"][-1]["phase"] == "pre_worker_staging"
        assert checkpoint["preparation_attempts"][-1]["worker_dispatched"] is False
        write(transport.diagnostics_dir / "ssh_staging.json", {
            "command": "ordinary metadata staging", "target_id": transport.host.id,
            "returncode": 1, "cleanup_proven": cleanup_proven})
        raise ValueError("original metadata guard")
    monkeypatch.setattr(native, "_stage_benchmark_set", staging)
    with pytest.raises(ValueError, match="original metadata guard"):
        native._run_native(point.prepared, dispatch.context)
    record = native._read(point.output / "DISPATCH_ATTEMPTS.json")["preparation_attempts"][-1]
    assert record["owned_preparation_closure"]["proven"] is cleanup_proven
    assert len(dispatch.quarantines) == (0 if cleanup_proven else 1)
    assert dispatch.commands == []


def test_performance_dispatch_requires_actual_matching_owned_ssh_receipts(dispatch, monkeypatch):
    monkeypatch.setattr(numerical, "_owned_dispatch_closure", _REAL_DISPATCH_CLOSURE)
    result = native._run_native(dispatch.point.prepared, dispatch.context)
    assert result["new_performance_processes"] == 3 and dispatch.quarantines == []
    for repeat in (1, 2, 3):
        safety = native._read(dispatch.point.output / f"rep{repeat:02d}/SDK_COMPLETION.json")
        assert safety["owned_dispatch_closure"]["proven"] is True
        path = Path(safety["owned_dispatch_closure"]["diagnostic"]["path"])
        assert safety["owned_dispatch_closure"]["diagnostic"]["sha256"] == native._sha(path)


@pytest.mark.parametrize("cleanup_proven", [True, False])
def test_normal_deployment_failure_uses_pre_worker_owned_closure_without_a_sdk_start(dispatch, monkeypatch, cleanup_proven):
    monkeypatch.setattr(numerical, "_owned_preparation_closure", _REAL_PREPARATION_CLOSURE)
    point = dispatch.point
    def deployment(*args):
        state = native._read(point.output / "DISPATCH_ATTEMPTS.json")
        assert state["attempts"] == [] and state["preparation_attempts"][-1]["phase"] == "pre_worker_deployment"
        assert state["preparation_attempts"][-1]["worker_dispatched"] is False
        write(point.output / "diagnostics/ssh_deployment.json", {
            "command": "ordinary deployment import preflight", "target_id": "same-dut",
            "returncode": 1, "cleanup_proven": cleanup_proven})
        raise ValueError("concrete deployed interpreter failure")
    monkeypatch.setattr(native, "ensure_current_deepx_runtime", deployment)
    with pytest.raises(ValueError, match="deployed interpreter failure"):
        native._run_native(point.prepared, dispatch.context)
    state = native._read(point.output / "DISPATCH_ATTEMPTS.json")
    failed = state["preparation_attempts"][0]
    assert failed["owned_preparation_closure"]["proven"] is cleanup_proven
    assert state["attempts"] == [] and dispatch.commands == [] and dispatch.staged == []
    assert len(dispatch.quarantines) == (0 if cleanup_proven else 1)
    if cleanup_proven:
        monkeypatch.setattr(native, "ensure_current_deepx_runtime", lambda *args: None)
        result = native._run_native(point.prepared, dispatch.context)
        assert result["new_performance_processes"] == 3
        assert native._read(point.output / "DISPATCH_ATTEMPTS.json")["preparation_attempts"][0] == failed


def test_missing_owned_dispatch_cleanup_still_quarantines_even_with_good_raw(dispatch, monkeypatch):
    monkeypatch.setattr(numerical, "_owned_dispatch_closure", lambda *args: {"proven": False})
    with pytest.raises(RuntimeError, match="SDK completion is unsafe or unproven"):
        native._run_native(dispatch.point.prepared, dispatch.context)
    assert len(dispatch.commands) == 1 and len(dispatch.quarantines) == 1
    assert not (dispatch.point.output / "rep01/VALIDATED.json").exists()


def test_worker_failure_is_one_attempt_per_invocation_and_preserves_receipt(dispatch, monkeypatch):
    def failed_collect(transport, item, directory):
        report = dispatch.point.materialize(item, checkpoint=False)
        row = native._read(report)
        row["ok"] = False
        write(report, row)
        return report
    monkeypatch.setattr(native, "_collect_repeat", failed_collect)
    with pytest.raises(ValueError, match="task counts, latency or completion failed"):
        native._run_native(dispatch.point.prepared, dispatch.context)
    assert len(dispatch.commands) == 1
    assert len(native._read(dispatch.point.output / "DISPATCH_ATTEMPTS.json")["attempts"]) == 1
    assert dispatch.quarantines == []
    failed_raw = next((dispatch.point.output / "rep01/download").rglob("result.json"))
    original = failed_raw.read_bytes()
    monkeypatch.setattr(native, "_collect_repeat", lambda transport, item, directory: dispatch.point.materialize(item, checkpoint=False))
    result = native._run_native(dispatch.point.prepared, dispatch.context)
    assert result["new_performance_processes"] == 3
    assert (dispatch.point.output / "rep01_retry01/VALIDATED.json").is_file()
    assert failed_raw.read_bytes() == original


def test_raw_without_owned_closure_cannot_gain_validation_during_resume(dispatch, monkeypatch):
    monkeypatch.setattr(numerical, "_owned_dispatch_closure", lambda *args: {"proven": False})
    with pytest.raises(RuntimeError, match="SDK completion is unsafe or unproven"):
        native._run_native(dispatch.point.prepared, dispatch.context)
    monkeypatch.setattr(numerical, "_owned_dispatch_closure", lambda *args: {"proven": True})
    with pytest.raises(RuntimeError, match="normal recovery required"):
        native._run_native(dispatch.point.prepared, dispatch.context)
    assert len(dispatch.commands) == 1
    assert not (dispatch.point.output / "rep01/VALIDATED.json").exists()


@pytest.mark.parametrize("ceilings", [(0, 0), (1, 6)])
def test_historical_campaign_or_release_ceiling_does_not_reset_records(ordinary_point, ceilings):
    point = ordinary_point
    matrix = Path(point.prepared["phase_matrix"]["path"])
    write(matrix, {"execution_limits": {"performance_technical_replacements_per_cell_max": ceilings[0],
                                      "performance_technical_replacements_global_max": ceilings[1]}})
    point.prepared["phase_matrix"]["sha256"] = native._sha(matrix)
    write(point.output / "PREPARED.json", point.prepared)
    for repeat in (1, 2, 3):
        for retry in range(4):
            point.attempts.append({"repeat": repeat, "technical_replacement": retry > 0})
    path = write(point.output / "DISPATCH_ATTEMPTS.json", {"attempts": point.attempts})
    before = path.read_bytes()
    native._admit_performance_attempt(point.prepared, {}, replacement=True)
    assert path.read_bytes() == before


def test_numbered_full_repair_rewrites_only_writable_destinations():
    source = {"repeat": 1, "native_full": True, "remote_output_root": "/fresh/rep01/outputs",
        "result_json": "/fresh/rep01/outputs/reports/result.json", "full_benchmark_set": "/fresh/rep01/benchmark_set",
        "source_contract": {"path": "/original/immutable.json"},
        "argv": ["python", "full.py", "--root", "/fresh/rep01", "--image-map", '{"full":"/fresh/rep01/benchmark_set/image.jpg"}']}
    original = copy.deepcopy(source)
    repair = native._performance_retry_item(source, 12)
    assert repair["remote_output_root"] == "/fresh/rep01/outputs_retry12"
    assert native._argument(repair["argv"], "--root") == "/fresh/rep01_retry12"
    assert repair["full_benchmark_set"] == "/fresh/rep01_retry12/benchmark_set"
    assert repair["source_contract"] == source["source_contract"] and source == original
