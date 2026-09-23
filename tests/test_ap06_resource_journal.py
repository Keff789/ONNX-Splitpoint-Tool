"""AP06 parent/child reservations over the existing owned lease journal.

Only external process leaves and temporary attempt logs are synthetic. Resource
admission, source completion checks and parent/child hand-off remain real.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import pytest

from onnx_splitpoint_tool.quality_service import ManagementQualityService, ResourcePauseGate
from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
from onnx_splitpoint_tool.remote.process_lease import (
    ControllerResourceBroker,
    RemoteProcessLeaseJournal,
    RemoteProcessLeaseJournalError,
    RemoteProcessLeaseScope,
)


def _gate():
    return ResourcePauseGate(available_cpu=2, available_memory_bytes=64)


def test_nonblocking_reservation_closes_dispatch_before_activity_drain():
    gate = _gate()
    entered, release = threading.Event(), threading.Event()

    def loader():
        with gate.activity("running_payload_loader", resources=("controller:io",)):
            entered.set()
            assert release.wait(3), "test failed to release its controlled loader"

    with ThreadPoolExecutor(max_workers=1) as workers:
        work = workers.submit(loader)
        token = None
        try:
            assert entered.wait(3)
            token = gate.begin_quiet("capture", resources=("controller:io",), owner="capture-owner")
            assert gate.poll_quiet(token)["state"] == "DRAINING"
            assert gate.acquire_activity("late_transfer", cpu=0,
                                         resources=("controller:io",), timeout=0) is None
            independent = gate.acquire_activity("independent_dut", cpu=0,
                                                resources=("dut:independent",), timeout=0)
            assert independent is not None
            gate.release_activity(independent)
            release.set()
            work.result(timeout=3)
            assert gate.poll_quiet(token)["state"] == "QUIET_CONFIRMED"
            assert gate.acquire_activity("late_transfer", cpu=0,
                                         resources=("controller:io",), timeout=0) is None
            assert gate.end_quiet(token)["state"] == "RELEASED"
            token = None
            after = gate.acquire_activity("after_capture", resources=("controller:io",), timeout=0)
            assert after is not None
            gate.release_activity(after)
        finally:
            release.set()
            if token is not None:
                gate.end_quiet(token)
    assert gate.snapshot()["activities"] == []
    assert gate.snapshot()["reservations"] == []


def test_distinct_capture_owners_collide_while_nested_same_owner_stays_reentrant():
    gate = _gate()
    # Logical setup names differ; the broker has resolved the same physical
    # receive path. The gate must exclude owners, not merely active CPU jobs.
    resources = ("controller:receive", "measurement-source:physical-a")
    outer = gate.begin_quiet("setup_alias_a", resources=resources, owner="first")
    second = nested = None
    try:
        assert gate.poll_quiet(outer)["state"] == "QUIET_CONFIRMED"
        second = gate.begin_quiet("setup_alias_b", resources=resources, owner="second")
        assert gate.poll_quiet(second)["state"] != "QUIET_CONFIRMED"
        nested = gate.begin_quiet("nested_cleanup", resources=resources, owner="first")
        assert gate.poll_quiet(nested)["state"] == "QUIET_CONFIRMED"
        gate.end_quiet(nested)
        nested = None
        assert gate.poll_quiet(second)["state"] != "QUIET_CONFIRMED"
        gate.end_quiet(outer)
        outer = None
        assert gate.poll_quiet(second)["state"] == "QUIET_CONFIRMED"
        assert gate.acquire_activity("new_receive", cpu=0,
                                     resources=resources, timeout=0) is None
    finally:
        for token in (nested, outer, second):
            if token is not None:
                gate.end_quiet(token)
    assert gate.snapshot()["reservations"] == []


def test_reentrant_owner_cannot_take_a_new_resource_from_another_quiet_capture():
    gate = _gate()
    outer = gate.begin_quiet("first_dut", resources=("dut:a",), owner="first")
    second = gate.begin_quiet("second_dut", resources=("dut:b",), owner="second")
    nested = None
    try:
        assert gate.poll_quiet(outer)["state"] == "QUIET_CONFIRMED"
        assert gate.poll_quiet(second)["state"] == "QUIET_CONFIRMED"
        nested = gate.begin_quiet("new_resource", resources=("dut:b",), owner="first")
        assert gate.poll_quiet(nested)["state"] != "QUIET_CONFIRMED", (
            "an earlier claim on independent DUT A must not supersede the active owner of DUT B"
        )
        gate.end_quiet(second)
        second = None
        assert gate.poll_quiet(nested)["state"] == "QUIET_CONFIRMED"
    finally:
        for token in (nested, second, outer):
            if token is not None:
                gate.end_quiet(token)


def test_later_collector_cannot_overtake_already_waiting_statistics_forever():
    gate = _gate()
    first = gate.begin_quiet("first_capture", owner="first")
    assert gate.poll_quiet(first)["state"] == "QUIET_CONFIRMED"
    entered, release = threading.Event(), threading.Event()
    later = None

    def statistics():
        token = gate.acquire_activity("older_statistics_block", timeout=3)
        assert token is not None, "an already waiting draw block starved behind later collectors"
        try:
            entered.set()
            assert release.wait(3)
        finally:
            gate.release_activity(token)

    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(statistics)
        try:
            _without_broker(lambda: gate.snapshot()["waiting"] == 1)
            later = gate.begin_quiet("later_capture", owner="later")
            gate.end_quiet(first)
            first = None
            assert gate.poll_quiet(later)["state"] != "QUIET_CONFIRMED", (
                "a stream of newly reserved captures can continuously leapfrog the older statistics ticket"
            )
            assert entered.wait(3), "the older statistics block must get a turn between captures"
            assert gate.poll_quiet(later)["state"] != "QUIET_CONFIRMED"
            release.set()
            future.result(timeout=3)
            assert gate.poll_quiet(later)["state"] == "QUIET_CONFIRMED"
        finally:
            release.set()
            for token in (first, later):
                if token is not None:
                    gate.end_quiet(token)
    assert gate.snapshot()["activities"] == []
    assert gate.snapshot()["reservations"] == []


def test_ram_blocked_statistics_ticket_does_not_deadlock_collector_fairness():
    gate = _gate()
    memory = gate.acquire_activity("resident_context", cpu=0, memory_bytes=64, resources=())
    first = gate.begin_quiet("first_capture", owner="first")
    assert gate.poll_quiet(first)["state"] == "QUIET_CONFIRMED"
    later = None

    def statistics():
        token = gate.acquire_activity("memory_waiting_statistics", memory_bytes=1, timeout=3)
        assert token is not None
        gate.release_activity(token)

    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(statistics)
        try:
            _without_broker(lambda: gate.snapshot()["waiting"] == 1)
            later = gate.begin_quiet("later_capture", owner="later")
            gate.end_quiet(first)
            first = None
            assert gate.poll_quiet(later)["state"] == "QUIET_CONFIRMED", (
                "fairness must not wait on a statistics ticket whose RAM dependency cannot yet fit"
            )
            gate.end_quiet(later)
            later = None
            gate.release_activity(memory)
            memory = None
            future.result(timeout=3)
        finally:
            for token in (first, later):
                if token is not None:
                    gate.end_quiet(token)
            if memory is not None:
                gate.release_activity(memory)
    assert gate.snapshot()["activities"] == []
    assert gate.snapshot()["reservations"] == []


_COLLECTOR_LEAF = r'''
import json, sys, time
from pathlib import Path
attempt, mode = Path(sys.argv[1]), sys.argv[2]
now = time.monotonic_ns()
print(f"Fast Firmware GO sent monotonic_ns={now} local=fixture:3000 requested_device_us=1", flush=True)
(attempt / "leaf_started.json").write_text(json.dumps({"monotonic": time.monotonic()}))
while not (attempt / "capture.release").exists():
    time.sleep(0.005)
if mode != "missing_end":
    print(f"Fast Firmware protocol end verified monotonic_ns={time.monotonic_ns()} local=fixture:3000 elapsed_us=1", flush=True)
print(f"Fast Firmware source closed monotonic_ns={time.monotonic_ns()} local=fixture:3000 verified=true reason=Ok(())", flush=True)
'''


_CLAIM_CHILD = r'''
import json, sys, time
from pathlib import Path
from onnx_splitpoint_tool.energy.collector import _run_one
from onnx_splitpoint_tool.remote.process_lease import ControllerResourceClaim, RemoteProcessLeaseJournal
attempt, leaf, setup, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
class FileCancellation:
    def is_set(self):
        return (attempt / "cancel.requested").exists()
claim = ControllerResourceClaim(RemoteProcessLeaseJournal.from_environment(required=True),
    setup_id=setup, attempt_dir=attempt, cancel_event=FileCancellation())
try:
    granted = claim.acquire()
    (attempt / "acquired.json").write_text(json.dumps(granted))
    if mode == "missed_acquiring":
        while not (attempt / "dispatch.release").exists():
            time.sleep(0.005)
    result = _run_one([sys.executable, "-B", leaf, str(attempt), mode], cwd=None,
        stdout_path=attempt / "collector_stdout.log", stderr_path=attempt / "collector_stderr.log",
        on_process_started=lambda proc, timestamp: claim.started(proc))
    (attempt / "collector_result.json").write_text(json.dumps(result))
    released = claim.finish(result)
    (attempt / "released.json").write_text(json.dumps(released))
except BaseException as exc:
    (attempt / "error.txt").write_text(type(exc).__name__ + ": " + str(exc))
    raise
while not (attempt / "postprocess.release").exists():
    time.sleep(0.005)
'''


_ACTIVITY_CHILD = r'''
import json, sys, time
from pathlib import Path
from onnx_splitpoint_tool.remote.process_lease import ControllerResourceClaim, RemoteProcessLeaseJournal
attempt, setup, purpose, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
class FileCancellation:
    def is_set(self):
        return (attempt / "cancel.requested").exists()
claim = ControllerResourceClaim(RemoteProcessLeaseJournal.from_environment(required=True),
    setup_id=setup, attempt_dir=attempt, purpose=purpose, cancel_event=FileCancellation())
try:
    active = claim.acquire()
    (attempt / "active.json").write_text(json.dumps(active))
    if mode == "nested_transfer":
        while not (attempt / "nested.requested").exists():
            time.sleep(0.005)
        nested_path = attempt / "nested"
        nested_path.mkdir()
        nested = ControllerResourceClaim(RemoteProcessLeaseJournal.from_environment(required=True),
            setup_id=setup, attempt_dir=nested_path, purpose="transfer",
            continuation=claim.operation if purpose == "prepare" else None)
        nested_active = nested.acquire()
        (nested_path / "active.json").write_text(json.dumps(nested_active))
        while not (nested_path / "activity.release").exists():
            time.sleep(0.005)
        nested_released = nested.finish({"process_started": False})
        (nested_path / "released.json").write_text(json.dumps(nested_released))
    while not (attempt / "activity.release").exists():
        time.sleep(0.005)
    released = claim.finish({"process_started": False})
    (attempt / "released.json").write_text(json.dumps(released))
except BaseException as exc:
    (attempt / "error.txt").write_text(type(exc).__name__ + ": " + str(exc))
    raise
'''


class _JournalRig:
    def __init__(self, root):
        self.root = root
        self.run = root / "run"
        self.run.mkdir()
        self.registry = ProcessTreeRegistry()
        self.children = []
        self.brokers = []
        self.leaf = root / "collector_leaf.py"
        self.leaf.write_text(_COLLECTOR_LEAF)
        self.resources = ("controller:cpu", "controller:io", "source:" + root.name,
                          "controller:capture:" + root.name, "controller:nic:" + root.name,
                          "dut:" + root.name)
        self.other_resources = tuple(resource + ":independent" if resource.startswith(("dut:", "source:"))
                                     else resource for resource in self.resources)
        self.new_session("first")

    def new_session(self, name, *, available_cpu=2, transfer_slots=1, postcalc_slots=1):
        self.gate = ResourcePauseGate(available_cpu=available_cpu, available_memory_bytes=1024**3,
                                     transfer_slots=transfer_slots, postcalc_slots=postcalc_slots)
        self.journal = RemoteProcessLeaseJournal(
            scope=RemoteProcessLeaseScope("resource-fixture", name),
            directory=self.root / ("journal-" + name))
        self.broker = ControllerResourceBroker(self.journal, self.gate, run_dir=self.run,
            setups={"setup-a": self.resources, "setup-alias": self.resources,
                    "setup-b": self.other_resources})
        self.brokers.append(self.broker)

    def spawn(self, name, *, mode="success", setup="setup-a"):
        attempt = self.run / name
        attempt.mkdir()
        env = dict(os.environ, **self.journal.environment(), PYTHONDONTWRITEBYTECODE="1")
        with (attempt / "child.log").open("w") as output:
            child = subprocess.Popen([sys.executable, "-B", "-c", _CLAIM_CHILD,
                str(attempt), str(self.leaf), setup, mode], env=env,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        self.registry.register(child, label="ap06-resource-fixture")
        self.children.append((child, attempt))
        return child, attempt

    def spawn_activity(self, name, *, purpose, setup="setup-a", mode="hold"):
        attempt = self.run / name
        attempt.mkdir()
        env = dict(os.environ, **self.journal.environment(), PYTHONDONTWRITEBYTECODE="1")
        with (attempt / "child.log").open("w") as output:
            child = subprocess.Popen([sys.executable, "-B", "-c", _ACTIVITY_CHILD,
                str(attempt), setup, purpose, mode], env=env, stdout=output,
                stderr=subprocess.STDOUT, start_new_session=True)
        self.registry.register(child, label="ap06-activity-fixture")
        self.children.append((child, attempt))
        return child, attempt

    def until(self, predicate, *, timeout=8):
        deadline = time.monotonic() + timeout
        while True:
            for broker in self.brokers:
                broker.tick()
            value = predicate()
            if value:
                return value
            assert time.monotonic() < deadline, [
                (str(attempt), child.poll(), (attempt / "child.log").read_text())
                for child, attempt in self.children]
            time.sleep(0.005)

    def close(self):
        for child, attempt in self.children:
            (attempt / "dispatch.release").touch()
            (attempt / "activity.release").touch()
            (attempt / "capture.release").touch()
            (attempt / "postprocess.release").touch()
        for child, _ in self.children:
            if child.poll() is None:
                self.registry.terminate_registered(child, grace_s=0.3)
            self.registry.unregister(child)
        for broker in self.brokers:
            broker.close()
        self.registry.assert_quiescent()


@pytest.fixture
def journal_rig(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import run_control
    # Redirect storage only. Actual lease ownership, flock, admission and
    # persistent quarantine code are deliberately left intact.
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path",
                        lambda: tmp_path / "locks" / "workflow.lock")
    rig = _JournalRig(tmp_path)
    try:
        yield rig
    finally:
        rig.close()


def test_child_capture_waits_for_real_parent_drain_and_releases_before_postprocessing(journal_rig):
    rig = journal_rig
    loader = rig.gate.acquire_activity("reference_inference", cpu=2)
    child, attempt = rig.spawn("normal")
    try:
        rig.until(lambda: bool(rig.gate.snapshot()["reservations"]))
        assert any(row["state"] == "DRAINING" and "controller:io" in row["resources"]
                   for row in rig.gate.snapshot()["reservations"])
        assert not (attempt / "acquired.json").exists()
        assert not (attempt / "leaf_started.json").exists()
    finally:
        rig.gate.release_activity(loader)
    rig.until(lambda: (attempt / "leaf_started.json").exists())
    granted = json.loads((attempt / "acquired.json").read_text())
    started = json.loads((attempt / "leaf_started.json").read_text())
    assert started["monotonic"] >= granted["observed_monotonic"]
    assert rig.gate.acquire_activity("late_statistics", timeout=0) is None
    (attempt / "capture.release").touch()
    rig.until(lambda: (attempt / "released.json").exists())
    assert child.poll() is None, "controlled child should still await postprocessing"
    cleanup = json.loads((attempt / "collector_stdout.log.cleanup.json").read_text())
    assert cleanup["owned_tree_quiescent"] is True
    from onnx_splitpoint_tool.energy.task_budget import source_completion
    assert source_completion(attempt)["verified"] is True
    resumed = rig.gate.acquire_activity("statistics_after_capture", timeout=0)
    assert resumed is not None
    rig.gate.release_activity(resumed)
    (attempt / "postprocess.release").touch()
    rig.until(lambda: child.poll() is not None)
    assert child.returncode == 0, (attempt / "child.log").read_text()
    assert rig.gate.snapshot()["reservations"] == []


def test_missing_protocol_end_stops_source_across_new_parent_session(journal_rig):
    rig = journal_rig
    first, attempt = rig.spawn("missing-end", mode="missing_end")
    rig.until(lambda: (attempt / "leaf_started.json").exists())
    (attempt / "capture.release").touch()
    rig.until(lambda: first.poll() is not None)
    assert first.returncode != 0
    assert "campaign_source_completion_unresolved" in (attempt / "error.txt").read_text()
    assert rig.gate.acquire_activity("unsafe_statistics", timeout=0) is None
    assert list((rig.root / "locks").rglob("*.quarantine.json")), "source STOP must survive parent memory"
    rig.new_session("resume")
    second, next_attempt = rig.spawn("resume-alias", setup="setup-alias")
    rig.until(lambda: second.poll() is not None)
    assert second.returncode != 0
    assert not (next_attempt / "leaf_started.json").exists()
    assert "quarantine" in (next_attempt / "error.txt").read_text().lower()


def test_cancel_during_child_reservation_never_dispatches_collector(journal_rig):
    rig = journal_rig
    activity = rig.gate.acquire_activity("reference_still_running", cpu=2)
    child, attempt = rig.spawn("cancel-during-drain")
    try:
        rig.until(lambda: bool(rig.gate.snapshot()["reservations"]))
        (attempt / "cancel.requested").touch()
        rig.until(lambda: child.poll() is not None)
        assert child.returncode != 0
        assert "cancelled_before_dispatch" in (attempt / "error.txt").read_text()
        assert not (attempt / "leaf_started.json").exists()
        assert rig.gate.snapshot()["reservations"] == []
        assert rig.gate.snapshot()["cpu_active"] == 2
        assert not list((rig.root / "locks").rglob("*.quarantine.json"))
    finally:
        rig.gate.release_activity(activity)


def _without_broker(predicate, *, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "controlled child did not reach its next journal phase"
        time.sleep(0.005)


def test_finalizing_keeps_actual_collector_identity_if_parent_missed_acquiring(journal_rig):
    rig = journal_rig
    child, attempt = rig.spawn("missed-acquiring", mode="missed_acquiring")
    rig.until(lambda: (attempt / "acquired.json").exists())
    operation = next(iter(rig.broker.operations.values()))
    assert operation.get("started") is not True
    (attempt / "dispatch.release").touch()
    _without_broker(lambda: (attempt / "leaf_started.json").exists())
    (attempt / "capture.release").touch()
    request_path = next(rig.journal.directory.glob("*.resource-request.json"))
    _without_broker(lambda: json.loads(request_path.read_text())["phase"] == "FINALIZING")
    final_request = json.loads(request_path.read_text())
    assert final_request["process_started"] is True
    assert final_request["collector_pid"] > 0
    assert int(final_request["collector_start_ticks"]) > 0
    # No broker tick occurred between the QUIET grant and FINALIZING. The
    # durable final request must suffice, including the real cleanup receipt.
    rig.until(lambda: (attempt / "released.json").exists())
    assert rig.gate.snapshot()["reservations"] == []
    (attempt / "postprocess.release").touch()
    rig.until(lambda: child.poll() is not None)
    assert child.returncode == 0, (attempt / "child.log").read_text()


_FOREIGN_BROKER = r'''
import json, os, subprocess, sys, time
from pathlib import Path
from onnx_splitpoint_tool.quality_service import ResourcePauseGate
from onnx_splitpoint_tool.remote.process_lease import ControllerResourceBroker, RemoteProcessLeaseJournal, RemoteProcessLeaseScope
from onnx_splitpoint_tool.workflow import run_control
root, lock_path, claim_script, leaf = map(Path, sys.argv[1:5])
resources = json.loads(sys.argv[5])
run_control.platform_workflow_interlock_path = lambda: lock_path
run_dir = root / "run"
run_dir.mkdir()
attempt = run_dir / "capture"
attempt.mkdir()
journal = RemoteProcessLeaseJournal(scope=RemoteProcessLeaseScope("second-gui", "second-session"), directory=root / "journal")
gate = ResourcePauseGate(available_cpu=2, available_memory_bytes=64)
broker = ControllerResourceBroker(journal, gate, run_dir=run_dir, setups={"alias": resources})
with (root / "inner.log").open("w") as output:
    child = subprocess.Popen([sys.executable, "-B", str(claim_script), str(attempt), str(leaf), "alias", "success"],
        env=dict(os.environ, **journal.environment()), stdout=output, stderr=subprocess.STDOUT)
try:
    while child.poll() is None:
        broker.tick()
        time.sleep(0.005)
    broker.tick()
    (root / "foreign_result.json").write_text(json.dumps({"child_returncode": child.returncode,
        "collector_started": (attempt / "leaf_started.json").exists(),
        "error": (attempt / "error.txt").read_text() if (attempt / "error.txt").exists() else ""}))
finally:
    if child.poll() is None:
        child.terminate()
        child.wait(timeout=3)
    broker.close()
'''


def test_second_gui_process_and_setup_alias_collide_on_real_physical_flock(journal_rig):
    rig = journal_rig
    first, attempt = rig.spawn("first-gui")
    rig.until(lambda: (attempt / "leaf_started.json").exists())
    foreign_dir = rig.root / "foreign-gui"
    foreign_dir.mkdir()
    claim_script = rig.root / "claim_child.py"
    claim_script.write_text(_CLAIM_CHILD)
    with (foreign_dir / "child.log").open("w") as output:
        foreign = subprocess.Popen([sys.executable, "-B", "-c", _FOREIGN_BROKER,
            str(foreign_dir), str(rig.root / "locks" / "workflow.lock"),
            str(claim_script), str(rig.leaf), json.dumps(rig.resources)],
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    rig.registry.register(foreign, label="ap06-second-gui-fixture")
    rig.children.append((foreign, foreign_dir))
    rig.until(lambda: foreign.poll() is not None)
    assert foreign.returncode == 0, (foreign_dir / "child.log").read_text()
    result = json.loads((foreign_dir / "foreign_result.json").read_text())
    assert result["child_returncode"] != 0
    assert result["collector_started"] is False
    assert "physical_resource_conflict" in result["error"]
    assert first.poll() is None
    assert not list((rig.root / "locks").rglob("*.quarantine.json"))
    (attempt / "capture.release").touch()
    rig.until(lambda: (attempt / "released.json").exists())
    (attempt / "postprocess.release").touch()
    rig.until(lambda: first.poll() is not None)
    assert first.returncode == 0


def test_request_cannot_claim_a_foreign_process_as_its_owner(journal_rig):
    from onnx_splitpoint_tool.process_control import _proc_start_time
    rig = journal_rig
    attempt = rig.run / "forged-owner"
    attempt.mkdir()
    operation = "capture-" + uuid.uuid4().hex
    path = rig.journal.directory / (operation + ".resource-request.json")
    # The parent of pytest is observable but not owned by this fixture. No
    # signal or other operation is sent to that process.
    rig.journal._write_json_atomic(path, {
        "operation_id": operation, "token": uuid.uuid4().hex,
        "run_id": rig.journal.scope.run_id, "session_id": rig.journal.scope.session_id,
        "owner_pid": os.getppid(), "owner_start_ticks": _proc_start_time(os.getppid()),
        "setup_id": "setup-a", "attempt_dir": str(attempt),
        "sequence": 1, "phase": "RESERVING",
    })
    try:
        with pytest.raises(RemoteProcessLeaseJournalError, match="unowned_request_process"):
            rig.broker.tick()
        assert rig.gate.snapshot()["reservations"] == []
        assert rig.broker.operations == {}
    finally:
        path.unlink()


def test_lost_owned_child_after_grant_keeps_source_stopped(journal_rig):
    rig = journal_rig
    child, attempt = rig.spawn("lost-owner")
    rig.until(lambda: (attempt / "leaf_started.json").exists())
    rig.registry.terminate_registered(child, grace_s=0.1)
    rig.until(lambda: all(operation.get("terminal") for operation in rig.broker.operations.values()))
    reply = json.loads(next(rig.journal.directory.glob("*.resource-reply.json")).read_text())
    assert reply["state"] == "STOP"
    assert reply["reason"] == "controller_resource_owner_lost"
    assert not (attempt / "released.json").exists()
    assert rig.gate.acquire_activity("unsafe_controller_work", timeout=0) is None
    assert list((rig.root / "locks").rglob("*.quarantine.json"))
    unrelated = rig.gate.acquire_activity("independent_dut", cpu=0,
                                          resources=("dut:independent",), timeout=0)
    assert unrelated is not None
    rig.gate.release_activity(unrelated)


def test_actual_child_dut_alias_waits_while_independent_dut_can_run(journal_rig):
    rig = journal_rig
    first, first_path = rig.spawn_activity("dut-first", purpose="dut_job")
    rig.until(lambda: (first_path / "active.json").exists())
    alias, alias_path = rig.spawn_activity("dut-alias", purpose="dut_job", setup="setup-alias")
    rig.until(lambda: rig.gate.snapshot()["waiting"] >= 1)
    independent, other_path = rig.spawn_activity("dut-independent", purpose="dut_job", setup="setup-b")
    rig.until(lambda: (other_path / "active.json").exists())
    assert not (alias_path / "active.json").exists()
    assert not (alias_path / "error.txt").exists(), "same-process DUT contention should queue without poisoning"
    assert first.poll() is independent.poll() is None
    assert rig.gate.snapshot()["cpu_active"] == 0, "remote DUT jobs consume no controller CPU slot"
    (first_path / "activity.release").touch()
    rig.until(lambda: (alias_path / "active.json").exists() and (first_path / "released.json").exists())
    assert (first_path / "released.json").exists()
    for path in (alias_path, other_path):
        (path / "activity.release").touch()
    rig.until(lambda: all(child.poll() is not None for child in (first, alias, independent)))
    assert all(child.returncode == 0 for child in (first, alias, independent))
    assert rig.gate.snapshot()["activities"] == []
    assert rig.gate.snapshot()["waiting"] == 0


@pytest.mark.parametrize("purpose", ["transfer", "prepare", "postcalc"])
def test_child_controller_activity_waits_for_end_then_resumes_during_postprocessing(journal_rig, purpose):
    rig = journal_rig
    collector, capture_path = rig.spawn("capture-before-" + purpose)
    rig.until(lambda: (capture_path / "leaf_started.json").exists())
    child, attempt = rig.spawn_activity(purpose, purpose=purpose)
    rig.until(lambda: rig.gate.snapshot()["waiting"] >= 1)
    assert not (attempt / "active.json").exists()
    assert rig.gate.snapshot()["cpu_active"] == 0
    (capture_path / "capture.release").touch()
    rig.until(lambda: (attempt / "active.json").exists() and (capture_path / "released.json").exists())
    assert (capture_path / "released.json").exists()
    assert collector.poll() is None, "capture helper still awaits noncritical postprocessing"
    assert rig.gate.snapshot()["cpu_active"] == 1
    assert rig.gate.snapshot()["memory_bytes_active"] == 256 * 1024**2
    (attempt / "activity.release").touch()
    (capture_path / "postprocess.release").touch()
    rig.until(lambda: child.poll() is not None and collector.poll() is not None)
    assert child.returncode == collector.returncode == 0


def test_capture_drains_actual_inflight_child_transfer(journal_rig):
    rig = journal_rig
    transfer, transfer_path = rig.spawn_activity("inflight-transfer", purpose="transfer")
    rig.until(lambda: (transfer_path / "active.json").exists())
    collector, capture_path = rig.spawn("capture-after-transfer")
    rig.until(lambda: bool(rig.gate.snapshot()["reservations"]))
    assert any(row["state"] == "DRAINING" and "controller:io" in row["resources"]
               for row in rig.gate.snapshot()["reservations"])
    assert not (capture_path / "leaf_started.json").exists()
    (transfer_path / "activity.release").touch()
    rig.until(lambda: (capture_path / "leaf_started.json").exists() and (transfer_path / "released.json").exists())
    assert (transfer_path / "released.json").exists()
    assert rig.gate.snapshot()["cpu_active"] == 0
    (capture_path / "capture.release").touch()
    rig.until(lambda: (capture_path / "released.json").exists())
    (capture_path / "postprocess.release").touch()
    rig.until(lambda: collector.poll() is not None and transfer.poll() is not None)
    assert collector.returncode == transfer.returncode == 0


def test_child_prepare_and_postcalc_share_cpu_cap_with_reference_activity(journal_rig):
    rig = journal_rig
    reference = rig.gate.acquire_activity("active_reference_inference", cpu=1)
    prepare, prepare_path = rig.spawn_activity("prepare", purpose="prepare")
    rig.until(lambda: (prepare_path / "active.json").exists())
    assert rig.gate.snapshot()["cpu_active"] == 2
    postcalc, postcalc_path = rig.spawn_activity("postcalc", purpose="postcalc")
    try:
        rig.until(lambda: rig.gate.snapshot()["waiting"] >= 1)
        assert not (postcalc_path / "active.json").exists()
        assert rig.gate.snapshot()["cpu_active"] == 2
    finally:
        rig.gate.release_activity(reference)
    rig.until(lambda: (postcalc_path / "active.json").exists())
    assert rig.gate.snapshot()["cpu_active"] == 2
    for path in (prepare_path, postcalc_path):
        (path / "activity.release").touch()
    rig.until(lambda: prepare.poll() is not None and postcalc.poll() is not None)
    assert prepare.returncode == postcalc.returncode == 0
    assert rig.gate.snapshot()["cpu_active"] == 0


def test_idle_owned_statistics_pool_survives_capture_and_resumes_without_new_worker(journal_rig):
    from test_quality_speed_blocks import _request
    rig = journal_rig
    options = {"engine": "optimized_coco_v1", "max_active_requests": 2,
               "prepared_cache_limit_mib": 16, "block_repetitions": 4,
               "checkpoint_blocks": True}
    with ManagementQualityService(rig.root / "tiny-statistics", workers=1,
                                  pause_gate=rig.gate, statistics=options) as service:
        first = _request(7)
        service.submit(first).result(timeout=10)
        owned = set(service._executor._processes)
        assert len(owned) == 1
        collector, attempt = rig.spawn("capture-with-idle-pool")
        rig.until(lambda: (attempt / "leaf_started.json").exists())
        second = replace(first, request_id="after-capture", seed=first.seed + 1)
        with ThreadPoolExecutor(max_workers=1) as submissions:
            pending = submissions.submit(service.submit, second)
            try:
                rig.until(lambda: rig.gate.snapshot()["waiting"] >= 1)
                assert not pending.done()
                assert set(service._executor._processes) == owned
                assert all(process.is_alive() for process in service._executor._processes.values())
                (attempt / "capture.release").touch()
                rig.until(lambda: (attempt / "released.json").exists())
                result = pending.result(timeout=5).result(timeout=10)
                assert result["cache_hit"] is False
                assert {row["worker_pid"] for row in result["statistics_observation"]["shards"]} == owned
                assert set(service._executor._processes) == owned
            finally:
                (attempt / "capture.release").touch()
                rig.until(lambda: (attempt / "released.json").exists())
        (attempt / "postprocess.release").touch()
        rig.until(lambda: collector.poll() is not None)
        assert collector.returncode == 0
    assert service.shutdown_state()["finished"] is True


@pytest.mark.parametrize("quiet_capture", [False, True])
def test_cancel_cleans_owned_spool_when_admitted_and_returns_without_io_during_quiet(tmp_path, quiet_capture):
    from concurrent.futures import CancelledError
    from test_quality_speed_blocks import _request
    gate = ResourcePauseGate(available_cpu=1, available_memory_bytes=1024**3)
    service = ManagementQualityService(tmp_path / "statistics", workers=1, pause_gate=gate,
        statistics={"engine": "optimized_coco_v1", "prepared_cache_limit_mib": 16,
                    "block_repetitions": 4, "checkpoint_blocks": True})
    reservation = None
    foreign = tmp_path / "unrelated-original.json"
    foreign.write_bytes(b"retained original prediction bytes")
    try:
        service.pause("controlled-queued-payload")
        future = service.submit(_request(7))
        _without_broker(lambda: gate.snapshot()["waiting"] >= 1)
        payloads = list((service._payload_scratch / "payloads").glob("*.json"))
        assert len(payloads) == 1
        before = payloads[0].read_bytes()
        if quiet_capture:
            reservation = gate.begin_quiet("actual-controlled-capture", owner="collector")
            assert gate.poll_quiet(reservation)["state"] == "QUIET_CONFIRMED"
        with ThreadPoolExecutor(max_workers=1) as threads:
            stopped = threads.submit(service.shutdown, cancel_futures=True)
            try:
                stopped.result(timeout=5)
            finally:
                if not stopped.done() and reservation is not None:
                    gate.end_quiet(reservation)
                    reservation = None
        assert service.shutdown_state()["finished"] is True
        assert gate.snapshot()["memory_bytes_active"] == 0
        assert gate.snapshot()["activities"] == []
        assert gate.snapshot()["waiting"] == 0
        with pytest.raises(CancelledError):
            future.result(timeout=1)
        if quiet_capture:
            assert gate.poll_quiet(reservation)["state"] == "QUIET_CONFIRMED"
            assert payloads[0].read_bytes() == before
        else:
            assert not payloads[0].exists(), "clean cancellation must remove its owned payload spool"
        assert foreign.read_bytes() == b"retained original prediction bytes"
    finally:
        if reservation is not None:
            gate.end_quiet(reservation)
        service.shutdown(cancel_futures=True)


def test_lost_child_dut_activity_creates_durable_alias_fence(journal_rig):
    rig = journal_rig
    child, attempt = rig.spawn_activity("lost-dut-owner", purpose="dut_job")
    rig.until(lambda: (attempt / "active.json").exists())
    rig.registry.terminate_registered(child, grace_s=0.1)
    rig.until(lambda: all(operation.get("terminal") for operation in rig.broker.operations.values()))
    assert rig.gate.snapshot()["activities"] == []
    assert list((rig.root / "locks").rglob("*.quarantine.json"))
    rig.new_session("dut-resume")
    alias, alias_path = rig.spawn_activity("lost-dut-alias", purpose="dut_job", setup="setup-alias")
    rig.until(lambda: alias.poll() is not None)
    assert alias.returncode != 0
    assert not (alias_path / "active.json").exists()
    assert "quarantine" in (alias_path / "error.txt").read_text().lower()
    other, other_path = rig.spawn_activity("unaffected-dut", purpose="dut_job", setup="setup-b")
    rig.until(lambda: (other_path / "active.json").exists())
    (other_path / "activity.release").touch()
    rig.until(lambda: other.poll() is not None)
    assert other.returncode == 0


@pytest.mark.parametrize("legacy_pause", [False, True])
def test_waiting_capture_allows_owned_dut_job_to_finish_its_nested_transfer(journal_rig, legacy_pause):
    rig = journal_rig
    job, job_path = rig.spawn_activity("job-needing-transfer", purpose="dut_job", mode="nested_transfer")
    rig.until(lambda: (job_path / "active.json").exists())
    if legacy_pause:
        from onnx_splitpoint_tool.quality_service import URECS_RESOURCE_REASON
        rig.gate.pause(URECS_RESOURCE_REASON)
    collector, capture_path = rig.spawn("capture-waiting-on-dut")
    rig.until(lambda: any(row["state"] == "DRAINING" for row in rig.gate.snapshot()["reservations"]))
    assert not (capture_path / "leaf_started.json").exists()
    (job_path / "nested.requested").touch()
    nested_path = job_path / "nested"
    rig.until(lambda: (nested_path / "active.json").exists())
    assert not (capture_path / "leaf_started.json").exists()
    assert job.poll() is None
    assert rig.gate.snapshot()["cpu_active"] == 1
    (nested_path / "activity.release").touch()
    rig.until(lambda: (nested_path / "released.json").exists())
    assert not (capture_path / "leaf_started.json").exists(), "DUT job still owns its physical lease"
    (job_path / "activity.release").touch()
    rig.until(lambda: (capture_path / "leaf_started.json").exists() and (job_path / "released.json").exists())
    assert (job_path / "released.json").exists()
    (capture_path / "capture.release").touch()
    rig.until(lambda: (capture_path / "released.json").exists())
    (capture_path / "postprocess.release").touch()
    rig.until(lambda: collector.poll() is not None and job.poll() is not None)
    assert collector.returncode == job.returncode == 0


@pytest.mark.parametrize("purpose", ["transfer", "postcalc"])
@pytest.mark.parametrize("slots", [1, 2])
def test_generic_local_and_native_child_share_actual_slot_limit(journal_rig, purpose, slots):
    from onnx_splitpoint_tool.process_control import bind_workflow_resource_options, controller_local_activity
    rig = journal_rig
    rig.new_session("shared-slot-cap", available_cpu=4, **{purpose + "_slots": slots})
    children = []
    with bind_workflow_resource_options(controller_gate=rig.gate):
        with controller_local_activity(purpose):
            child, attempt = rig.spawn_activity("child-activity", purpose=purpose)
            children.append((child, attempt))
            if slots == 1:
                rig.until(lambda: rig.gate.snapshot()["waiting"] >= 1)
                assert not (attempt / "active.json").exists()
            else:
                rig.until(lambda: (attempt / "active.json").exists())
                second, second_path = rig.spawn_activity("excess-child-activity", purpose=purpose)
                children.append((second, second_path))
                rig.until(lambda: rig.gate.snapshot()["waiting"] >= 1)
                assert not (second_path / "active.json").exists()
            snapshot = rig.gate.snapshot()
            assert snapshot[purpose + "_capacity"] == slots
            assert sum("controller:" + purpose in activity["resources"]
                       for activity in snapshot["activities"]) == slots
            assert snapshot["cpu_active"] == slots < snapshot["cpu_capacity"], (
                "the tested blocker must be the shared purpose slot, not an exhausted CPU budget"
            )
    rig.until(lambda: all((path / "active.json").exists() for _, path in children))
    for _, path in children:
        (path / "activity.release").touch()
    rig.until(lambda: all(child.poll() is not None for child, _ in children))
    assert all(child.returncode == 0 for child, _ in children)
    assert rig.gate.snapshot()["activities"] == []
    assert rig.gate.snapshot()["waiting"] == 0


def test_owned_prepare_continuation_finishes_nested_transfer_during_capture_drain(journal_rig):
    rig = journal_rig
    prepare, prepare_path = rig.spawn_activity("prepare-needing-transfer", purpose="prepare", mode="nested_transfer")
    rig.until(lambda: (prepare_path / "active.json").exists())
    collector, capture_path = rig.spawn("capture-draining-prepare")
    rig.until(lambda: any(row["state"] == "DRAINING" and "controller:io" in row["resources"]
                         for row in rig.gate.snapshot()["reservations"]))
    (prepare_path / "nested.requested").touch()
    nested = prepare_path / "nested"
    rig.until(lambda: (nested / "active.json").exists())
    assert not (capture_path / "leaf_started.json").exists()
    snapshot = rig.gate.snapshot()
    assert snapshot["cpu_active"] == 1, "a nested continuation must retain the parent's CPU charge"
    assert snapshot["memory_bytes_active"] == 256 * 1024**2
    assert sum("controller:transfer" in activity["resources"]
               for activity in snapshot["activities"]) == 1
    (nested / "activity.release").touch()
    rig.until(lambda: (nested / "released.json").exists())
    assert not (capture_path / "leaf_started.json").exists()
    (prepare_path / "activity.release").touch()
    rig.until(lambda: (capture_path / "leaf_started.json").exists()
              and (prepare_path / "released.json").exists())
    (capture_path / "capture.release").touch()
    rig.until(lambda: (capture_path / "released.json").exists())
    (capture_path / "postprocess.release").touch()
    rig.until(lambda: collector.poll() is not None and prepare.poll() is not None)
    assert collector.returncode == prepare.returncode == 0


def test_child_ignores_only_legacy_urecs_pause_and_keeps_compiler_fence(journal_rig):
    from onnx_splitpoint_tool.quality_service import HAILO_BUILD_RESOURCE_REASON, URECS_RESOURCE_REASON
    rig = journal_rig
    rig.gate.pause(URECS_RESOURCE_REASON)
    rig.gate.pause(HAILO_BUILD_RESOURCE_REASON)
    child, attempt = rig.spawn_activity("compiler-fenced-transfer", purpose="transfer")
    rig.until(lambda: rig.gate.snapshot()["waiting"] >= 1)
    assert not (attempt / "active.json").exists()
    assert rig.gate.snapshot()["cpu_active"] == 0
    rig.gate.resume(HAILO_BUILD_RESOURCE_REASON)
    rig.until(lambda: (attempt / "active.json").exists())
    assert URECS_RESOURCE_REASON in rig.gate.reasons
    (attempt / "activity.release").touch()
    rig.until(lambda: child.poll() is not None)
    assert child.returncode == 0
