"""AP06 in-process admission/drain contracts with real controlled threads.

These tests do not start devices or assert cross-process/source integrity.
The short waits bound test coordination only, never measurement duration.
"""
from __future__ import annotations

import threading
import time

import pytest

from onnx_splitpoint_tool.quality_service import ResourcePauseGate


class _TestCancelled(RuntimeError):
    pass


class _Workers:
    def __init__(self):
        self.stop = threading.Event()
        self.events = []
        self.threads = []
        self.errors = []

    def event(self):
        event = threading.Event()
        self.events.append(event)
        return event

    def check_cancelled(self):
        if self.stop.is_set():
            raise _TestCancelled("controlled test cleanup")

    def start(self, target):
        done = threading.Event()

        def run():
            try:
                target()
            except _TestCancelled as exc:
                if not self.stop.is_set():
                    self.errors.append(exc)
            except BaseException as exc:
                self.errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=run, daemon=True)
        self.threads.append(thread)
        thread.start()
        return done


@pytest.fixture
def workers():
    group = _Workers()
    yield group
    group.stop.set()
    for event in group.events:
        event.set()
    for thread in group.threads:
        thread.join(timeout=3)
    assert not any(thread.is_alive() for thread in group.threads), "controlled gate work failed to stop"
    assert not group.errors, repr(group.errors)


def _await(predicate, description):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, description
        time.sleep(0.005)


def _assert_idle(gate):
    snapshot = gate.snapshot()
    assert snapshot["cpu_active"] == 0
    assert snapshot["memory_bytes_active"] == 0
    assert snapshot["activities"] == []
    assert snapshot["reservations"] == []
    assert snapshot["waiting"] == 0


def test_reservation_blocks_new_conflict_before_loader_and_flush_finish(workers):
    gate = ResourcePauseGate(available_cpu=3, available_memory_bytes=128)
    loader_entered, flush_entered = workers.event(), workers.event()
    loader_release, flush_release = workers.event(), workers.event()
    quiet_entered, quiet_release = workers.event(), workers.event()
    late_entered = workers.event()

    def activity(reason, entered, release):
        with gate.activity(reason, memory_bytes=8, resources=("controller:io",),
                           check_cancelled=workers.check_cancelled):
            entered.set()
            release.wait()

    loader_done = workers.start(lambda: activity("payload_loader", loader_entered, loader_release))
    flush_done = workers.start(lambda: activity("checkpoint_flush", flush_entered, flush_release))
    assert loader_entered.wait(3) and flush_entered.wait(3)

    def capture():
        with gate.reserve_quiet("collector", resources=("controller:io",),
                                check_cancelled=workers.check_cancelled) as reservation:
            assert reservation["state"] == "QUIET_CONFIRMED"
            quiet_entered.set()
            quiet_release.wait()

    capture_done = workers.start(capture)
    _await(lambda: bool(gate.snapshot()["reservations"]), "reservation was not published before drain")
    assert gate.snapshot()["reservations"][0]["state"] == "DRAINING"

    def late_work():
        with gate.activity("late_loader", resources=("controller:io",),
                           check_cancelled=workers.check_cancelled):
            late_entered.set()

    late_done = workers.start(late_work)
    _await(lambda: gate.snapshot()["waiting"] == 1, "new conflict did not wait")
    independent_entered = workers.event()

    def independent_dut():
        with gate.activity("dut_b", cpu=0, resources=("dut:b",),
                           check_cancelled=workers.check_cancelled):
            independent_entered.set()

    independent_done = workers.start(independent_dut)
    assert independent_entered.wait(3) and independent_done.wait(3)
    assert not quiet_entered.is_set() and not late_entered.is_set()
    loader_release.set()
    assert loader_done.wait(3)
    assert not quiet_entered.is_set(), "checkpoint flush was excluded from drain"
    assert [a["reason"] for a in gate.snapshot()["activities"]] == ["checkpoint_flush"]
    flush_release.set()
    assert flush_done.wait(3) and quiet_entered.wait(3)
    assert not late_entered.is_set(), "new work entered between drain and capture"
    assert gate.snapshot()["activities"] == []
    quiet_release.set()
    assert capture_done.wait(3) and late_done.wait(3)
    assert late_entered.is_set()
    _assert_idle(gate)


def test_nested_same_reason_quiet_reservations_keep_separate_owners(workers):
    gate = ResourcePauseGate(available_cpu=1, available_memory_bytes=32)
    entered = workers.event()
    with gate.reserve_quiet("collector", resources=("dut:a",)) as outer:
        with gate.reserve_quiet("collector", resources=("dut:a",)) as inner:
            assert outer["id"] != inner["id"]
            assert len(gate.snapshot()["reservations"]) == 2
        assert inner["state"] == "RELEASED"
        assert [r["id"] for r in gate.snapshot()["reservations"]] == [outer["id"]]

        def blocked_work():
            with gate.activity("dut_a", resources=("dut:a",),
                               check_cancelled=workers.check_cancelled):
                entered.set()

        done = workers.start(blocked_work)
        _await(lambda: gate.snapshot()["waiting"] == 1, "outer owner did not retain exclusion")
        assert not entered.is_set()
    assert done.wait(3) and entered.is_set()
    _assert_idle(gate)


def test_nested_legacy_holds_do_not_resume_outer_owner():
    gate = ResourcePauseGate(available_cpu=1, available_memory_bytes=32)
    with gate.hold("collector"):
        with gate.hold("collector"):
            assert gate.paused
        assert gate.paused
        assert not gate.wait_until_resumed(timeout=0)
    assert gate.wait_until_resumed(timeout=0)
    _assert_idle(gate)


def test_cancel_during_drain_releases_reservation_but_keeps_active_work(workers):
    gate = ResourcePauseGate(available_cpu=2, available_memory_bytes=32)
    cancel, cancelled = workers.event(), workers.event()

    def check_cancelled():
        workers.check_cancelled()
        if cancel.is_set():
            raise _TestCancelled("run cancelled before capture")

    def capture():
        with pytest.raises(_TestCancelled, match="run cancelled before capture"):
            with gate.reserve_quiet("collector", check_cancelled=check_cancelled):
                pytest.fail("capture admitted before active loader finished")
        cancelled.set()

    with gate.activity("loader", memory_bytes=8):
        done = workers.start(capture)
        _await(lambda: bool(gate.snapshot()["reservations"]), "capture never entered drain")
        cancel.set()
        assert cancelled.wait(3) and done.wait(3)
        assert gate.snapshot()["reservations"] == []
        assert [a["reason"] for a in gate.snapshot()["activities"]] == ["loader"]
        with gate.activity("after_cancel", memory_bytes=8):
            assert gate.snapshot()["cpu_active"] == 2
    _assert_idle(gate)


def test_exception_after_quiet_confirmation_releases_reservation():
    gate = ResourcePauseGate(available_cpu=1, available_memory_bytes=32)
    with pytest.raises(RuntimeError, match="controlled collector start failure"):
        with gate.reserve_quiet("collector"):
            raise RuntimeError("controlled collector start failure")
    with gate.activity("following_work"):
        assert gate.snapshot()["cpu_active"] == 1
    _assert_idle(gate)


def test_cpu_and_ram_caps_are_shared_across_waiting_contexts(workers):
    gate = ResourcePauseGate(available_cpu=3, available_memory_bytes=10)
    second_entered, second_release = workers.event(), workers.event()

    def second_context():
        with gate.activity("request_2", cpu=2, memory_bytes=5,
                           check_cancelled=workers.check_cancelled):
            second_entered.set()
            second_release.wait()

    with gate.activity("request_1", cpu=2, memory_bytes=6):
        done = workers.start(second_context)
        _await(lambda: gate.snapshot()["waiting"] == 1, "second context did not honor total budget")
        assert gate.snapshot()["cpu_active"] == 2
        assert gate.snapshot()["memory_bytes_active"] == 6
        assert not second_entered.is_set()
        with gate.activity("reference_completion", cpu=1, memory_bytes=4):
            assert gate.snapshot()["cpu_active"] == 3
            assert gate.snapshot()["memory_bytes_active"] == 10
            assert not second_entered.is_set()
        assert not second_entered.is_set()
    assert second_entered.wait(3)
    assert gate.snapshot()["cpu_active"] == 2
    assert gate.snapshot()["memory_bytes_active"] == 5
    second_release.set()
    assert done.wait(3)
    _assert_idle(gate)


def test_ram_waiter_does_not_hold_cpu_needed_by_its_reference(workers):
    gate = ResourcePauseGate(available_cpu=2, available_memory_bytes=8)
    candidate_entered, reference_completed = workers.event(), workers.event()

    def candidate():
        with gate.activity("candidate_waiting_for_reference", cpu=1, memory_bytes=4,
                           check_cancelled=workers.check_cancelled):
            assert reference_completed.is_set()
            candidate_entered.set()

    def reference():
        with gate.activity("reference_completion", cpu=2, memory_bytes=2,
                           check_cancelled=workers.check_cancelled):
            assert gate.snapshot()["cpu_active"] == 2
            assert gate.snapshot()["memory_bytes_active"] == 8
            reference_completed.set()

    with gate.activity("immutable_reference_context", cpu=0, memory_bytes=6, resources=()):
        candidate_done = workers.start(candidate)
        _await(lambda: gate.snapshot()["waiting"] == 1, "candidate did not wait for RAM")
        assert gate.snapshot()["cpu_active"] == 0
        reference_done = workers.start(reference)
        assert reference_completed.wait(3) and reference_done.wait(3), "RAM waiter blocked its reference dependency"
        assert not candidate_entered.is_set()
    assert candidate_done.wait(3) and candidate_entered.is_set()
    _assert_idle(gate)


def test_cpu_waiter_cannot_oversubscribe_but_zero_cpu_flush_can_finish(workers):
    gate = ResourcePauseGate(available_cpu=3, available_memory_bytes=8)
    reference_entered, reference_release = workers.event(), workers.event()

    def reference_session():
        with gate.activity("fixed_two_thread_reference", cpu=2,
                           check_cancelled=workers.check_cancelled):
            reference_entered.set()
            reference_release.wait()

    with gate.activity("two_running_draw_blocks", cpu=2):
        done = workers.start(reference_session)
        _await(lambda: gate.snapshot()["waiting"] == 1, "reference oversubscribed the common CPU cap")
        assert not reference_entered.is_set()
        with gate.activity("release_checkpoint_metadata", cpu=0, memory_bytes=1):
            assert gate.snapshot()["cpu_active"] == 2
            assert gate.snapshot()["memory_bytes_active"] == 1
        assert not reference_entered.is_set()
    assert reference_entered.wait(3)
    assert gate.snapshot()["cpu_active"] == 2
    reference_release.set()
    assert done.wait(3)
    _assert_idle(gate)


def test_cancel_budget_wait_removes_ticket_and_charges_no_tokens(workers):
    gate = ResourcePauseGate(available_cpu=1, available_memory_bytes=8)
    cancel = workers.event()

    def check_cancelled():
        workers.check_cancelled()
        if cancel.is_set():
            raise _TestCancelled("waiting request cancelled")

    def waiting_request():
        with pytest.raises(_TestCancelled, match="waiting request cancelled"):
            with gate.activity("cancelled_request", cpu=1, memory_bytes=8,
                               check_cancelled=check_cancelled):
                pytest.fail("work exceeded occupied CPU/RAM capacity")

    with gate.activity("active_request", cpu=1, memory_bytes=8):
        done = workers.start(waiting_request)
        _await(lambda: gate.snapshot()["waiting"] == 1, "request never waited for admission")
        cancel.set()
        assert done.wait(3)
        assert gate.snapshot()["waiting"] == 0
        assert gate.snapshot()["cpu_active"] == 1
        assert gate.snapshot()["memory_bytes_active"] == 8
    _assert_idle(gate)


def test_own_active_resource_is_a_descriptive_conflict_not_self_drain(workers):
    gate = ResourcePauseGate(available_cpu=1, available_memory_bytes=8)

    def self_conflict():
        with gate.activity("owned_loader", resources=("controller:io",)):
            try:
                with gate.reserve_quiet("collector", resources=("controller:io",),
                                        check_cancelled=workers.check_cancelled):
                    pytest.fail("quiet was claimed while this owner still loaded data")
            except _TestCancelled:
                raise
            except RuntimeError as exc:
                assert any(word in str(exc).lower() for word in ("owner", "self", "conflict")), str(exc)
            else:
                pytest.fail("self-owned drain did not report the resource conflict")
            assert gate.snapshot()["reservations"] == []

    done = workers.start(self_conflict)
    assert done.wait(3), "reserve_quiet waited for its own activity to finish"
    _assert_idle(gate)


@pytest.mark.parametrize("cpu", [0, -1])
def test_nonpositive_explicit_cpu_capacity_is_rejected(cpu):
    with pytest.raises(ValueError):
        ResourcePauseGate(available_cpu=cpu, available_memory_bytes=8)


def test_zero_memory_capacity_is_retained_and_rejects_memory_work():
    gate = ResourcePauseGate(available_cpu=1, available_memory_bytes=0)
    assert gate.available_memory_bytes == 0
    with pytest.raises(ValueError):
        with gate.activity("memory_required", memory_bytes=1):
            pytest.fail("zero RAM budget was replaced by the host default")
    with gate.activity("no_payload", cpu=1, memory_bytes=0):
        assert gate.snapshot()["memory_bytes_active"] == 0
    _assert_idle(gate)


@pytest.mark.parametrize("kwargs", [{"cpu": 4}, {"memory_bytes": 11}, {"cpu": -1}, {"memory_bytes": -1}])
def test_impossible_activity_is_rejected_without_leaking_state(kwargs):
    gate = ResourcePauseGate(available_cpu=3, available_memory_bytes=10)
    with pytest.raises(ValueError):
        with gate.activity("invalid", **kwargs):
            pytest.fail("impossible request was admitted")
    _assert_idle(gate)
