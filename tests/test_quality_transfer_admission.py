"""Quality/transfer admission regressions using the real gate and SCP wrapper.

Worker timing and the external SCP leaf are controlled; service preparation,
dispatch, calculations, publication and every admission decision are real.
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from onnx_splitpoint_tool import quality_service as quality
from onnx_splitpoint_tool.process_control import bind_workflow_resource_options
from onnx_splitpoint_tool.remote.ssh_transport import HostConfig, SSHTransport


def _await(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "controlled work did not reach its rendezvous"
        time.sleep(.005)


def _gate():
    return quality.ResourcePauseGate(available_cpu=7, available_memory_bytes=2 * 1024**3)


def _idle(gate):
    snapshot = gate.snapshot()
    assert snapshot["activities"] == []
    assert snapshot["reservations"] == []
    assert snapshot["waiting"] == 0
    assert snapshot["cpu_active"] == snapshot["memory_bytes_active"] == 0


class _HeldExecutor:
    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.release = threading.Event()
        self.started = 0
        self.lock = threading.Lock()

    def submit(self, function, *args, **kwargs):
        def work():
            with self.lock:
                self.started += 1
            assert self.release.wait(10), "test did not release its statistics workers"
            return function(*args, **kwargs)
        return self.pool.submit(work)

    def shutdown(self, **kwargs):
        self.release.set()
        self.pool.shutdown(**kwargs)


@pytest.mark.parametrize("engine", ["legacy", "optimized_coco_v1"])
def test_scp_enters_while_four_quality_workers_and_cpu_blocked_older_ticket_remain(
    tmp_path, monkeypatch, engine,
):
    gate = _gate()
    pool = _HeldExecutor(4)
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kwargs: pool)
    service = quality.ManagementQualityService(tmp_path / "cache", workers=4,
        pause_gate=gate, statistics={"engine": engine, "block_repetitions": 2,
            "checkpoint_blocks": True, "prepared_cache_limit_mib": 16})
    request = quality.QualityEvaluationRequest(
        reference_records=[{"image_id": i, "value": 1.0} for i in range(4)],
        candidate_records=[{"image_id": i, "value": .9} for i in range(4)],
        annotations=[], metric_gate_config={"primary_metric": "mean"},
        repetitions=8, seed=42, confidence_level=.95,
        non_inferiority_margin=.25, evaluator_factory="paired_mean", value_field="value")
    cancelled, entered = threading.Event(), threading.Event()
    transport = SSHTransport(HostConfig(id="offline", label="offline", host="unused.invalid"),
                             cancel_event=cancelled)
    observed, failures = [], []

    def scp_leaf(command, timeout=None):
        observed.append((command, gate.snapshot()))
        entered.set()
        return 0, "controlled external SCP leaf"

    monkeypatch.setattr(transport, "_run_capture", scp_leaf)

    def transfer():
        try:
            with bind_workflow_resource_options(controller_gate=gate):
                assert transport.scp_upload("saved-input.json", "/offline/input.json") == (
                    0, "controlled external SCP leaf")
        except BaseException as exc:
            failures.append(exc)

    older = None
    thread = threading.Thread(target=transfer)
    try:
        client = service.submit(request)
        _await(lambda: pool.started == 4)
        assert gate.snapshot()["cpu_active"] == 4
        older = gate.begin_activity("older_four_cpu_request", cpu=4)
        assert gate.poll_activity(older) is False
        thread.start()
        assert entered.wait(1), "CPU-fitting transfer was blocked behind the unready older request"
        thread.join(2)
        assert not thread.is_alive() and not failures
        assert not pool.release.is_set() and not client.done()
        command, snapshot = observed[0]
        assert command[0] == "scp" and command[-2:] == [
            "saved-input.json", "unused.invalid:/offline/input.json"]
        assert snapshot["cpu_active"] == 5
        assert sum("controller:transfer" in row["resources"]
                   for row in snapshot["activities"]) == 1
        assert gate.poll_activity(older) is False
        pool.release.set()
        # Once capacity becomes available, the older ready request wins over
        # any remaining service publication and can finish without starvation.
        _await(lambda: gate.poll_activity(older))
        gate.release_activity(older)
        older = None
        assert client.result(timeout=5)["status"] == "completed"
    finally:
        cancelled.set()
        if thread.ident is not None:
            thread.join(2)
        if older is not None:
            gate.release_activity(older)
        pool.release.set()
        service.shutdown(wait=True, cancel_futures=True)
    _idle(gate)


def test_ready_older_request_keeps_priority_then_finite_queue_drains():
    gate = _gate()
    running = gate.acquire_activity("running_quality", cpu=4)
    older = gate.begin_activity("older_ready", cpu=2)
    later = gate.begin_activity("later_transfer", cpu=1, resources=("controller:transfer",))
    try:
        assert gate.poll_activity(later) is False
        assert gate.poll_activity(older) is True
        assert gate.poll_activity(later) is True
        assert gate.snapshot()["cpu_active"] == 7
    finally:
        for token in (later, older, running):
            gate.release_activity(token)
    _idle(gate)


@pytest.mark.parametrize("blocker", ["cpu", "ram", "transfer_slot", "draining", "quiet", "dut", "source"])
def test_cpu_fitting_bypass_preserves_each_resource_fence(blocker):
    gate = _gate()
    resources = ("controller:cpu", "controller:io", "controller:transfer", "dut:a", "source:a")
    running = gate.acquire_activity("running_quality", cpu=4,
        resources=() if blocker == "quiet" else ("controller:cpu", "controller:io"))
    older = fence = reservation = transfer = None
    try:
        if blocker in {"draining", "quiet"}:
            reservation = gate.begin_quiet("capture", resources=("controller:io",), owner="capture")
            assert gate.poll_quiet(reservation)["state"] == (
                "QUIET_CONFIRMED" if blocker == "quiet" else "DRAINING")
        else:
            options = {
                "cpu": {"cpu": 3, "resources": ()},
                "ram": {"cpu": 0, "resources": (), "memory_bytes": gate.available_memory_bytes},
                "transfer_slot": {"cpu": 0, "resources": ("controller:transfer",)},
                "dut": {"cpu": 0, "resources": ("dut:a",)},
                "source": {"cpu": 0, "resources": ("source:a",)},
            }[blocker]
            fence = gate.acquire_activity("blocking_" + blocker, **options)
        older = gate.begin_activity("older_cpu_blocked", cpu=4)
        transfer = gate.begin_activity("late_transfer", cpu=1, memory_bytes=1, resources=resources)
        assert gate.poll_activity(transfer) is False
        if reservation is not None:
            gate.end_quiet(reservation)
            reservation = None
        else:
            gate.release_activity(fence)
            fence = None
        assert gate.poll_activity(transfer) is True
    finally:
        if reservation is not None:
            gate.end_quiet(reservation)
        for token in (transfer, fence, older, running):
            if token is not None:
                gate.release_activity(token)
    _idle(gate)


def test_cancelled_scp_admission_releases_ticket_and_transfer_slot(monkeypatch):
    gate = _gate()
    occupied = gate.acquire_activity("occupied_transfer", cpu=1, resources=("controller:transfer",))
    cancel = threading.Event()
    transport = SSHTransport(HostConfig(id="offline", label="offline", host="unused.invalid"),
                             cancel_event=cancel)
    calls, failures = [], []
    monkeypatch.setattr(transport, "_run_capture", lambda *args, **kwargs: (calls.append(args), (0, ""))[1])

    def transfer():
        try:
            with bind_workflow_resource_options(controller_gate=gate):
                transport.scp_upload("saved-input.json", "/offline/input.json")
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=transfer)
    try:
        thread.start()
        _await(lambda: gate.snapshot()["waiting"] == 1)
        cancel.set()
        thread.join(2)
        assert not thread.is_alive()
        assert len(failures) == 1 and "cancelled before admission" in str(failures[0])
        assert calls == [] and gate.snapshot()["waiting"] == 0
        assert gate.snapshot()["cpu_active"] == 1
    finally:
        cancel.set()
        thread.join(2)
        gate.release_activity(occupied)
    cancel.clear()
    with bind_workflow_resource_options(controller_gate=gate):
        assert transport.scp_upload("saved-input.json", "/offline/input.json") == (0, "")
    assert len(calls) == 1
    _idle(gate)
