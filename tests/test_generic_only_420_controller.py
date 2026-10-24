"""Local controller/normal-parser checks; synthetic files, no accelerator work."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from onnx_splitpoint_tool import generic_only_reporting as reporting
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from onnx_splitpoint_tool.workflow import generic_completed_task as entry
from onnx_splitpoint_tool.workflow import generic_only_cell_policy as policy
from test_generic_completion_dispatch import prepared_case, LocalTransport


def context(row, tmp_path):
    registry = RemoteProcessLeaseRegistry()
    scope = RemoteProcessLeaseScope("local-controller-test", "same-measurement")
    journal = registry.configure_journal(scope=scope, journal_dir=tmp_path / "journal")
    return {"initial_row": deepcopy(row), "control": policy.CONTROL, "initial_control": policy.CONTROL,
            "registry": registry, "journal": journal, "scope": scope}


def prepared(tmp_path):
    row = prepared_case(tmp_path)
    row["runtime_config"]["config_directory"] = str(Path(row["runtime_config"]["original_run_cfg"]["native_split_quality_binding"]).parent)
    row["runtime_environment_comparison"] = {"status": "observed_no_known_configuration_conflict"}
    return row


def test_normal_dispatch_preserves_valid_index_and_persistent_budget(tmp_path):
    row = prepared(tmp_path)
    ctx = context(row, tmp_path)
    transport = LocalTransport(stop_at=2)
    with pytest.raises(RuntimeError, match="stopped"):
        entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    valid = entry.read_json(Path(row["result_path"]))["repetitions"][0]
    preserved = deepcopy(valid)
    transport.stop_at = None
    entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    assert [index for _, index in transport.calls] == [1, 2, 2, 3]
    assert entry.read_json(Path(row["result_path"]))["repetitions"][0] == preserved
    attempts = policy.read_dispatch_attempts(row["result_path"])
    assert [item["bounded_cell_admission"]["attempt_number"] for item in attempts] == [1, 1, 2, 1]
    before = Path(row["result_path"]).read_bytes()
    entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    assert Path(row["result_path"]).read_bytes() == before


def test_two_failed_original_index_attempts_never_reset_on_reentry(tmp_path):
    row = prepared(tmp_path)
    ctx = context(row, tmp_path)
    transport = LocalTransport(stop_at=1)
    for number in (1, 2):
        transport.stop_at = number
        with pytest.raises(RuntimeError, match="stopped"):
            entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    for _restart in (1, 2):
        with pytest.raises(policy.CellAttemptLimitReached):
            entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    assert [index for _, index in transport.calls] == [1, 1]


def test_unsafe_registry_blocks_before_staging_or_dispatch(tmp_path):
    row = prepared(tmp_path)
    ctx = context(row, tmp_path)
    ctx["registry"].poison()
    transport = LocalTransport()
    with pytest.raises(policy.UnresolvedOwnedProcesses):
        entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    assert transport.calls == []
    assert not Path(row["result_path"]).parent.exists()


def test_h8_detection_quality_raw_input_uses_normal_bound_tensor_loader(tmp_path):
    from onnx_splitpoint_tool.generic_only_plan import validate_quality_input_binding, file_sha256
    array = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    path = tmp_path / "prepared.bin"
    array.tofile(path)
    prepared = {"path": "/existing/dut/prepared.bin", "local_path": str(path), "format": "raw",
        "sha256": file_sha256(path), "shape": list(array.shape), "dtype": "uint8"}
    image = "e" * 64
    row = {"input_binding": {"prepared_input": prepared, "source_image_sha256": image},
        "completion_contract": {"processor_contract": {"original_wh": [640, 426]}}}
    proof = {"representative_input": {"prepared_input": deepcopy(prepared),
        "source_image_sha256": image, "original_wh": [640, 426]},
        "representative_input_comparison": {"prepared_tensor": {"shape": list(array.shape),
            "dtype": "uint8", "tensor_sha256": hashlib.sha256(array.tobytes()).hexdigest()}}}
    validate_quality_input_binding(row, proof)
    proof["representative_input_comparison"]["prepared_tensor"]["tensor_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="actual timing input bytes"):
        validate_quality_input_binding(row, proof)
    path.write_bytes(b"changed raw prepared bytes")
    with pytest.raises(ValueError, match="source missing or changed"):
        validate_quality_input_binding(row, proof)


def test_reservation_counts_even_when_transport_fails_before_remote_invoke(tmp_path):
    row = prepared(tmp_path)
    ctx = context(row, tmp_path)
    class BeforeInvokeFailure(LocalTransport):
        def run_streaming(self, *args, **kwargs):
            raise RuntimeError("controlled failure before any remote process")
    transport = BeforeInvokeFailure()
    for _ in (1, 2):
        with pytest.raises(RuntimeError, match="before any remote process"):
            entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    with pytest.raises(policy.CellAttemptLimitReached):
        entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    assert transport.calls == []
    attempts = policy.read_dispatch_attempts(row["result_path"])
    assert len(attempts) == 2
    assert all(item["status"] == "dispatch_error" and "finished_at" in item for item in attempts)
    assert all("pid" not in item for item in attempts)  # No invented remote process.


def test_parser_valid_work_with_bad_dispatch_exit_is_preserved_but_not_admitted(tmp_path):
    row = prepared(tmp_path)
    ctx = context(row, tmp_path)
    class LateBadExit(LocalTransport):
        def run_streaming(self, command, **kwargs):
            super().run_streaming(command, **kwargs)
            return 1
    transport = LateBadExit()
    with pytest.raises(RuntimeError, match="stopped"):
        entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    before = Path(row["result_path"]).read_bytes()
    with pytest.raises(policy.CellCompletionRejected, match="no verified exit zero"):
        entry.run_case(row, transport, tmp_path / "output", 1, cell_context=ctx)
    assert [index for _, index in transport.calls] == [1]
    assert Path(row["result_path"]).read_bytes() == before


def test_original_start_snapshot_and_append_only_resume_markers(tmp_path):
    output = tmp_path / "output"; output.mkdir()
    bundle = output / "runner-bundle.tar.gz"; bundle.write_bytes(b"synthetic immutable bundle")
    plan_path = output / "plan/execution-plan.json"
    plan = {"scope_mode": "generic_only_completed_task", "execution_id": "same-id",
            "cell_execution_control": deepcopy(policy.CONTROL), "rows": [prepared(tmp_path)]}
    initial, first = policy.admit_controller_attempt(plan, plan_path, output, bundle, resume=False, started_at="first")
    start = (output / "production-start.json").read_bytes()
    with pytest.raises(policy.InvalidResumeBinding, match="explicit --resume"):
        policy.admit_controller_attempt(plan, plan_path, output, bundle, resume=False, started_at="again")
    plan["rows"][0].update(execution_status="failed", execution_error="retained diagnostic")
    _, second = policy.admit_controller_attempt(plan, plan_path, output, bundle, resume=True, started_at="second")
    first.with_suffix(".terminal.json").write_text("{}")
    _, third = policy.admit_controller_attempt(plan, plan_path, output, bundle, resume=True, started_at="third")
    assert [item.name for item in (first, second, third)] == [f"attempt_{i:04d}.json" for i in (1, 2, 3)]
    assert (output / "production-start.json").read_bytes() == start
    plan["rows"][0]["runtime_config"]["execution_id"] = "substitute"
    with pytest.raises(policy.InvalidResumeBinding, match="binding changed"):
        policy.admit_controller_attempt(plan, plan_path, output, bundle, resume=True, started_at="fourth")


def test_default_once_only_controller_on_controlled_owner_leaf(tmp_path, monkeypatch):
    # The existing regression otherwise sees the intentionally live campaign.
    # Substitute only its external owner-observation leaf, not task parsing.
    from test_generic_completion_dispatch import test_normal_cli_accepts_six_pairs_before_remainder_and_finally_reports_stop
    monkeypatch.setattr(entry, "observe_owners", lambda *_, **__: {"ok": True, "synthetic_owner_fixture": True})
    test_normal_cli_accepts_six_pairs_before_remainder_and_finally_reports_stop(tmp_path, monkeypatch)


def independent_plan(tmp_path, monkeypatch, *, cross_setup=False):
    """Controller state fixture, with normal parser and synthetic admission leaf."""
    from onnx_splitpoint_tool.workflow import run_control
    primary = tmp_path / "primary"; primary.mkdir()
    output = tmp_path / "output"; output.mkdir()
    rows = [prepared_case(tmp_path, index=index, setup=(f"setup_{'a' if index == 1 else 'b'}"
             if cross_setup else "test_setup")) for index in (1, 2)]
    for row in rows:
        row["runtime_config"]["config_directory"] = str(Path(row["runtime_config"]["original_run_cfg"]["native_split_quality_binding"]).parent)
    setups = sorted({row["setup_id"] for row in rows})
    entry.write_json(primary / "run_manifest.json", {"hardware_targets": [{"id": setup,
        "remote": {"host": f"192.0.2.{index}", "user": "fixture"}} for index, setup in enumerate(setups, 1)]})
    entry.write_json(primary / "reports/native_host_telemetry_summary.json", {"captures": []})
    with tarfile.open(output / "runner-bundle.tar.gz", "w:gz"):
        pass
    scope = tmp_path / "scope.csv"; scope.write_text("synthetic controlled scope")
    plan = {"scope_mode": "generic_only_completed_task", "cell_execution_control": deepcopy(policy.CONTROL),
        "rows": rows, "execution_id": "new-session", "original_run": str(primary), "output_root": str(output),
        "scope_source": str(scope), "start_case_keys": [row["case_key"] for row in rows] if cross_setup else [rows[0]["case_key"]]}
    path = output / "plan/execution-plan.json"; entry.write_json(path, plan)
    monkeypatch.setattr(entry, "validate_plan", lambda plan, *_: plan["rows"])
    def observe(targets, _output, scoped=False, **context):
        assert scoped is True
        assert context["measurement_identity"] == (primary.name, plan["execution_id"])
        assert set(context["current_physical_locks"]) == set(targets)
        assert all(lock._fh is not None and lock.owner["purpose"] == "generic_completed_task_192"
                   for lock in context["current_physical_locks"].values())
        return {"ok": True, "synthetic_owner_fixture": True}
    monkeypatch.setattr(entry, "observe_owners", observe)
    monkeypatch.setattr(entry, "write_status_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(entry, "terminal_bundle", lambda *_: None)
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: tmp_path / "test-locks/platform.lock")
    monkeypatch.setattr(reporting, "generic_only_admission", lambda observation, _, **kwargs: {
        "eligible_for_generic_only_execution": observation["status"] == "completed"})
    def summary(plan, *_):
        valid = [len(__import__("onnx_splitpoint_tool.generic_completion_reporting", fromlist=["load_valid_repetitions"])
            .load_valid_repetitions(row["result_path"], row)) for row in plan["rows"]]
        return {"all_cases_measured": all(count == 3 for count in valid), "valid_repetitions": sum(valid),
            "status_counts": {"completed": sum(count == 3 for count in valid)}, "planned_cases": len(valid)}
    monkeypatch.setattr(entry, "_write_completion_reports", summary)
    return path, output, rows


def test_independent_controller_continues_after_exhausted_cell_and_resumes_same_id(tmp_path, monkeypatch):
    path, output, rows = independent_plan(tmp_path, monkeypatch)
    class FirstCellFails(LocalTransport):
        def run_streaming(self, command, **kwargs):
            self.stop_at = len(self.calls) + 1 if "b001" in command else None
            return super().run_streaming(command, **kwargs)
    transport = FirstCellFails()
    monkeypatch.setattr(entry, "transport_for", lambda *args, **kwargs: transport)
    assert entry.execute_plan(path) == 1
    assert [index for _, index in transport.calls] == [1, 1, 1, 2, 3]
    terminal = entry.read_json(output / "terminal-status.json")
    assert terminal["status"] == "complete_with_cell_failures"
    assert terminal["summary"]["valid_repetitions"] == 3
    assert terminal["summary"]["status_counts"]["completed"] == 1
    original_start = (output / "production-start.json").read_bytes()
    first_stop = output / "cell-stops/attempt_0001" / (entry.case_name(rows[0]) + ".json")
    original_stop = first_stop.read_bytes()
    preserved_result = Path(rows[1]["result_path"]).read_bytes()
    count = len(transport.calls)
    assert entry.execute_plan(path, resume=True) == 1
    assert len(transport.calls) == count  # Exhausted first cell, valid second cell.
    assert (output / "production-start.json").read_bytes() == original_start
    assert Path(rows[1]["result_path"]).read_bytes() == preserved_result
    assert first_stop.read_bytes() == original_stop
    assert (output / "cell-stops/attempt_0002" / first_stop.name).is_file()
    assert (output / "controller-attempts/attempt_0002.json").is_file()
    assert (output / "controller-attempts/attempt_0001.terminal.json").is_file()


def test_fresh_control_start_cannot_silently_adopt_existing_result(tmp_path):
    output = tmp_path / "output"; output.mkdir()
    bundle = output / "runner-bundle.tar.gz"; bundle.write_bytes(b"synthetic immutable bundle")
    row = prepared(tmp_path)
    result = Path(row["result_path"]); result.parent.mkdir(parents=True); result.write_text("{}")
    plan = {"scope_mode": "generic_only_completed_task", "execution_id": "same-id",
        "cell_execution_control": deepcopy(policy.CONTROL), "rows": [row]}
    with pytest.raises(policy.InvalidResumeBinding, match="cannot adopt existing measurements"):
        policy.admit_controller_attempt(plan, output / "plan/execution-plan.json", output, bundle,
            resume=False, started_at="synthetic")
    assert not (output / "production-start.json").exists()


@pytest.mark.parametrize("fault", ["preflight", "poisoned_ownership", "physical_lock", "staging", "source_integrity"])
def test_affected_setup_stops_without_blocking_independent_clean_setup(tmp_path, monkeypatch, fault):
    path, output, rows = independent_plan(tmp_path, monkeypatch, cross_setup=True)
    transports, registries = {}, {}
    class ControlledTransport(LocalTransport):
        def __init__(self, setup, registry):
            super().__init__()
            self.setup, self.registry = setup, registry
        def run_streaming(self, command, **kwargs):
            if fault == "poisoned_ownership" and self.setup == "setup_a":
                self.stop_at = 1
                rc = super().run_streaming(command, **kwargs)
                self.registry.poison()  # Real normal journal cancellation tombstone.
                return rc
            if fault == "source_integrity" and self.setup == "setup_a":
                self.stop_at = 1
                kwargs["on_line"]("RuntimeError: Generic runtime source differs from frozen bundle")
                return super().run_streaming(command, **kwargs)
            return super().run_streaming(command, **kwargs)
    def transport_for(target, *args, registry, **kwargs):
        setup = target["id"]
        transports[setup] = ControlledTransport(setup, registry)
        registries[setup] = registry
        return transports[setup]
    monkeypatch.setattr(entry, "transport_for", transport_for)
    if fault == "preflight":
        original = entry.verify_runtime_environment
        def verify(transport, row, target):
            if row["setup_id"] == "setup_a": raise ValueError("controlled bound interpreter conflict")
            return original(transport, row, target)
        monkeypatch.setattr(entry, "verify_runtime_environment", verify)
    elif fault == "physical_lock":
        original = entry.EvaluationRunLock.for_resource
        calls = []
        def resource_lock(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1: raise RuntimeError("controlled first DUT lock unavailable")
            return original(*args, **kwargs)
        monkeypatch.setattr(entry.EvaluationRunLock, "for_resource", resource_lock)
    elif fault == "staging":
        original = entry.stage_case
        def stage(row, *args, **kwargs):
            if row["setup_id"] == "setup_a": raise RuntimeError("controlled cache/source staging failure")
            return original(row, *args, **kwargs)
        monkeypatch.setattr(entry, "stage_case", stage)
    assert entry.execute_plan(path) == 1
    assert [index for _, index in transports["setup_b"].calls] == [1, 2, 3]
    if fault == "poisoned_ownership":
        assert [index for _, index in transports["setup_a"].calls] == [1]
        assert registries["setup_a"].cancelled is True
        assert registries["setup_b"].cancelled is False
    elif fault in {"preflight", "staging"}:
        assert transports["setup_a"].calls == []
    elif fault == "source_integrity":
        assert [index for _, index in transports["setup_a"].calls] == [1]
    else:
        assert "setup_a" not in transports
    terminal = entry.read_json(output / "terminal-status.json")
    assert terminal["status"] == "complete_with_cell_failures"
    assert terminal["summary"]["valid_repetitions"] == 3
    cleanup = entry.read_json(output / "cleanup-status.json")
    clean = cleanup["per_backend_journals"]["setup_b"]
    assert clean["journal_empty"] is True and clean["journal_pending_count"] == 0
    assert clean["registry_cancelled_or_poisoned"] is False
    assert not (output / "start-cases-accepted.json").exists()


def classification_observation(tmp_path, policy_name="synchronous", depth=1):
    from onnx_splitpoint_tool.runners.harness import classification
    source = Path(classification.__file__).resolve()
    sources = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    array = np.array([[1, 4, 4, 3, 2, 0]], dtype=np.float32)
    repetitions = []
    for index in (1, 2, 3):
        processor = ClassificationCompletion()
        for _ in range(1100): processor.process({"logits": array})
        path = tmp_path / f"output_{index}.npy"; np.save(path, array)
        row = {"repetition_index": index, "execution_id": "same", "status": "ok", "boot_id": "fixture-boot", "pid": index,
               "process_start_ticks": index, "runtime_instance_id": f"fixture:{index}",
               "warmup_completed_count": 100, "completed_task_count": 1000,
               "stage1_execution": {"policy": policy_name, "inflight": depth},
               "runtime_source_binding": {"before": sources, "after": sources, "unchanged": True},
               "raw_output_samples": {"logits": {"path": path.name}},
               "last_completed_result": {key: value.tolist() for key, value in processor.last_result.items()},
               "postprocess_report": processor.report()}
        if policy_name != "synchronous": row["stage1_peak_uncollected_jobs"] = depth
        repetitions.append(row)
    config = {"task": "classification", "execution_id": "same",
              "completion_contract": {"task": "classification", "operation": "top1_top5", "label_offset": 0},
              "stage1_execution": {"policy": policy_name, "inflight": depth},
              "runtime_source_files": [{"path": name, "sha256": digest} for name, digest in sources.items()]}
    return {"repetitions": repetitions}, {"runtime_config": config, "result_path": str(tmp_path / "result.json")}


@pytest.mark.parametrize("policy_name,depth", [("synchronous", 1), ("hailo_async_slots_v1", 8), ("deepx_async_slots_v1", 4)])
def test_actual_classification_completion_replay_and_policy(tmp_path, policy_name, depth):
    observation, row = classification_observation(tmp_path, policy_name, depth)
    assert len(reporting.validate_completion_attestations(observation, row)) == 3
    changed = deepcopy(observation)
    changed["repetitions"][0]["last_completed_result"]["top1"] = [0]
    changed["repetitions"][0]["last_completed_result"]["top5"] = [[0, 1, 2, 3, 4]]
    changed["repetitions"][0]["postprocess_report"]["classification_topk"] = changed["repetitions"][0]["last_completed_result"]
    with pytest.raises(ValueError, match="actual last raw outputs"):
        reporting.validate_completion_attestations(changed, row)
    changed = deepcopy(observation)
    changed["repetitions"][1]["postprocess_report"]["postprocess_completed_frames"] = 1000
    with pytest.raises(ValueError, match="report/count"):
        reporting.validate_completion_attestations(changed, row)
    changed = deepcopy(observation)
    changed["repetitions"][0]["stage1_peak_uncollected_jobs"] = 1
    with pytest.raises(ValueError, match="host peak"):
        reporting.validate_completion_attestations(changed, row)


def test_reporter_preserves_three_parser_repetitions_but_rejects_durable_bad_exit(tmp_path, monkeypatch):
    # Actual ClassificationCompletion results/source/raw replay; only the
    # independent Quality authority and shared result-header adapter are leaves.
    observation, row = classification_observation(tmp_path)
    config = row["runtime_config"]
    config.update(admission_mode="generic_only", measurement_mode="generic_only_completed_task",
                  config_directory="/session/configs", quality_reference={"synthetic_quality_leaf": True})
    row.update(case_key="synthetic-classifier-cell", original_run=str(tmp_path / "producer"),
        quality_status="accuracy_loss", quality_reference=deepcopy(config["quality_reference"]),
        remote_root="/session", remote_python_executable="/venv/bin/python", case_output_relative="setup/classifier/b001",
        runtime_environment_comparison={"status": "observed_no_known_configuration_conflict"})
    observation.update(case_key=row["case_key"], status="completed", valid_repetition_count=3, completed_task_count=3000,
                       errors=[], raw_attempts=deepcopy(observation["repetitions"]))
    attempts = [{"repetition_index": index, "command": policy.expected_command(row, index),
        "started_at": "synthetic-fixture", "status": "exited", "returncode": 1 if index == 3 else 0}
        for index in (1, 2, 3)]
    (tmp_path / "dispatch_attempts.json").write_text(json.dumps({"attempts": attempts}))
    Path(row["result_path"]).write_text("{}")
    before = {path.name: path.read_bytes() for path in tmp_path.glob("*.npy")}
    decision = reporting.generic_only_admission(observation, row, cell_control=policy.CONTROL)
    assert decision["eligible_for_generic_only_execution"] is False
    assert any("no verified exit zero" in reason for reason in decision["reasons"])
    monkeypatch.setattr(reporting, "validate_quality_reference", lambda _: {"status": "accuracy_loss"})
    monkeypatch.setattr(reporting, "validate_completion_case", lambda *args, **kwargs: deepcopy(observation))
    plan = {"scope_mode": "generic_only_completed_task", "admission_mode": "generic_only",
            "cell_execution_control": deepcopy(policy.CONTROL), "rows": [row]}
    summary = reporting.write_completion_reports(plan, tmp_path, tmp_path / "reports")
    assert summary["status_counts"]["blocked"] == 1
    assert summary["valid_repetitions"] == 3 and summary["functional_admissions"] == 0
    assert summary["all_cases_measured"] is False
    assert {path.name: path.read_bytes() for path in tmp_path.glob("*.npy")} == before
