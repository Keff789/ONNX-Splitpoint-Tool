"""Real local setup-queue scheduling with controlled leaf functions.

No device work is started. Registry resolution and scheduling are production
code; events replace only the external workload duration.
"""
from __future__ import annotations

from concurrent.futures import CancelledError
import threading

import pytest
import yaml

from onnx_splitpoint_tool.process_control import (
    ProcessTreeRegistry,
    bind_process_registry,
    current_process_registry,
)
from onnx_splitpoint_tool.workflow import execution_binding
from onnx_splitpoint_tool.workflow.hardware_matrix import normalize_hardware_targets


@pytest.fixture
def setup_queue():
    managers = []
    releases = []

    def make(*, max_workers=2, model_ids=("model_1", "model_2")):
        manager = execution_binding.PhysicalSetupDispatchManager(
            max_workers=max_workers, model_ids=list(model_ids),
        )
        managers.append(manager)
        return manager

    def release_event():
        event = threading.Event()
        releases.append(event)
        return event

    yield make, release_event
    for event in releases:
        event.set()
    for manager in managers:
        manager.shutdown(wait=True, cancel_futures=True)


def _register_all(manager, model_ids=("model_1", "model_2")):
    for model_id in model_ids:
        manager.finish_registration(model_id)


def _registry_targets(tmp_path):
    rows = [
        {"id": "front_hailo", "accelerator": "hailo8", "enabled": True,
         "host": {"address": "192.0.2.10", "user": "nx", "port": 22}},
        {"id": "same_jetson_trt", "accelerator": "tensorrt", "enabled": True,
         "host": {"address": "192.0.2.10", "user": "second_account", "port": 2222}},
        {"id": "separate_hailo", "accelerator": "hailo10h", "enabled": True,
         "host": {"address": "192.0.2.11", "user": "nx", "port": 22}},
    ]
    path = tmp_path / "test_hardware_registry.yaml"
    path.write_text(yaml.safe_dump({"hardware_setups": rows}), encoding="utf-8")
    before = path.read_bytes()
    targets = normalize_hardware_targets({"hardware": {
        "setups_file": str(path), "selected_setups": [row["id"] for row in rows],
    }})
    assert path.read_bytes() == before, "read-only target resolution rewrote the test registry"
    return {target["id"]: target for target in targets}


def test_all_model_registrations_precede_dispatch_including_empty_model(setup_queue):
    make, _ = setup_queue
    manager = make()
    entered = threading.Event()
    future = manager.submit_group("model_1", "dut:a", entered.set)
    manager.finish_registration("model_1")
    assert not entered.wait(0.05), "dispatch crossed the incomplete plan-registration barrier"
    assert not future.done()
    manager.finish_registration("model_2")
    assert entered.wait(3)
    assert future.result(timeout=3) is None


def test_fast_setup_starts_second_model_while_other_setup_runs_first(setup_queue):
    make, release_event = setup_queue
    manager = make(max_workers=2)
    a1_entered, a2_entered = threading.Event(), threading.Event()
    b1_entered, b2_entered = threading.Event(), threading.Event()
    release_a1, release_a2, release_b1 = release_event(), release_event(), release_event()

    def leaf(label, entered, release=None):
        entered.set()
        if release is not None:
            assert release.wait(5), f"controlled test leaf was not released: {label}"
        return label

    # Registration intentionally arrives in the opposite model order.
    a2 = manager.submit_group("model_2", "dut:a", leaf, "a2", a2_entered, release_a2)
    b2 = manager.submit_group("model_2", "dut:b", leaf, "b2", b2_entered)
    manager.finish_registration("model_2")
    a1 = manager.submit_group("model_1", "dut:a", leaf, "a1", a1_entered, release_a1)
    b1 = manager.submit_group("model_1", "dut:b", leaf, "b1", b1_entered, release_b1)
    manager.finish_registration("model_1")
    assert a1_entered.wait(3) and b1_entered.wait(3)
    assert not a2_entered.is_set() and not b2_entered.is_set()
    release_a1.set()
    assert a1.result(timeout=3) == "a1"
    assert a2_entered.wait(3), "fast setup remained behind the old model barrier"
    assert b1.running() and not b1.done()
    assert not b2_entered.is_set(), "same DUT ran two models concurrently"
    release_a2.set()
    release_b1.set()
    assert a2.result(timeout=3) == "a2"
    assert b1.result(timeout=3) == "b1"
    assert b2.result(timeout=3) == "b2"


def test_registry_aliases_share_dut_queue_without_occupying_other_setup_slot(setup_queue, tmp_path):
    make, release_event = setup_queue
    targets = _registry_targets(tmp_path)
    key = execution_binding.physical_dut_key
    first_key = key(targets["front_hailo"])
    alias_key = key(targets["same_jetson_trt"])
    independent_key = key(targets["separate_hailo"])
    assert first_key == alias_key
    assert first_key != independent_key
    manager = make(max_workers=2, model_ids=("model_1",))
    first_entered, alias_entered, independent_entered = [threading.Event() for _ in range(3)]
    release_first, release_independent = release_event(), release_event()

    def leaf(entered, release=None):
        entered.set()
        if release is not None:
            assert release.wait(5)

    first = manager.submit_group("model_1", first_key, leaf, first_entered, release_first)
    alias = manager.submit_group("model_1", alias_key, leaf, alias_entered)
    independent = manager.submit_group("model_1", independent_key, leaf, independent_entered, release_independent)
    manager.finish_registration("model_1")
    assert first_entered.wait(3) and independent_entered.wait(3), "alias waiter occupied the independent DUT slot"
    assert not alias_entered.is_set()
    release_first.set()
    first.result(timeout=3)
    alias.result(timeout=3)
    assert independent.running() and not independent.done()
    release_independent.set()
    independent.result(timeout=3)


def test_global_pool_cap_applies_across_distinct_duts_and_models(setup_queue):
    make, release_event = setup_queue
    manager = make(max_workers=2)
    release = release_event()
    two_active = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0
    active_by_dut = {}

    def leaf(dut, label):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            active_by_dut[dut] = active_by_dut.get(dut, 0) + 1
            assert active <= 2
            assert active_by_dut[dut] == 1
            if active == 2:
                two_active.set()
        try:
            assert release.wait(5)
            return label
        finally:
            with lock:
                active -= 1
                active_by_dut[dut] -= 1

    futures = [manager.submit_group(model, dut, leaf, dut, (model, dut))
               for model in ("model_1", "model_2")
               for dut in ("dut:a", "dut:b", "dut:c")]
    _register_all(manager)
    assert two_active.wait(3), "independent DUTs did not share the available pool"
    with lock:
        assert active == 2
    release.set()
    assert len([future.result(timeout=3) for future in futures]) == 6
    assert peak == 2
    assert active == 0


def test_queued_cancellation_prevents_dispatch_and_unblocks_following_group(setup_queue):
    make, release_event = setup_queue
    manager = make(max_workers=1, model_ids=("model_1",))
    entered = threading.Event()
    release = release_event()
    calls = []

    def first_leaf():
        entered.set()
        assert release.wait(5)
        return "first"

    first = manager.submit_group("model_1", "dut:a", first_leaf)
    cancelled = manager.submit_group("model_1", "dut:a", calls.append, "cancelled")
    following = manager.submit_group("model_1", "dut:a", calls.append, "following")
    manager.finish_registration("model_1")
    assert entered.wait(3)
    assert not first.cancel(), "running physical work was claimed cancelled before cleanup"
    assert cancelled.cancel()
    release.set()
    assert first.result(timeout=3) == "first"
    following.result(timeout=3)
    with pytest.raises(CancelledError):
        cancelled.result(timeout=3)
    assert calls == ["following"]


def test_frozen_model_and_resource_order_resolve_ready_ties(setup_queue):
    make, _ = setup_queue
    manager = make(max_workers=1)
    calls = []
    registration = [
        ("model_2", "dut:b", "m2-b"),
        ("model_1", "dut:b", "m1-b"),
        ("model_2", "dut:a", "m2-a"),
        ("model_1", "dut:a", "m1-a-first"),
        ("model_1", "dut:a", "m1-a-second"),
    ]
    futures = [manager.submit_group(model, dut, calls.append, label)
               for model, dut, label in registration]
    _register_all(manager)
    for future in futures:
        future.result(timeout=3)
    assert calls == ["m1-a-first", "m1-a-second", "m1-b", "m2-a", "m2-b"]


def test_failed_leaf_releases_dut_and_preserves_its_exception(setup_queue):
    make, _ = setup_queue
    manager = make(max_workers=1)

    def failing_leaf():
        raise ValueError("controlled external failure")

    failed = manager.submit_group("model_1", "dut:a", failing_leaf)
    following = manager.submit_group("model_2", "dut:a", lambda: "next model")
    _register_all(manager)
    with pytest.raises(ValueError, match="controlled external failure"):
        failed.result(timeout=3)
    assert following.result(timeout=3) == "next model"


def test_submission_preserves_workflow_process_ownership_context(setup_queue):
    make, _ = setup_queue
    manager = make(max_workers=1)
    first_registry, second_registry = ProcessTreeRegistry(), ProcessTreeRegistry()
    with bind_process_registry(first_registry):
        first = manager.submit_group("model_1", "dut:a", current_process_registry)
    with bind_process_registry(second_registry):
        second = manager.submit_group("model_2", "dut:a", current_process_registry)
    assert current_process_registry() is None
    _register_all(manager)
    assert first.result(timeout=3) is first_registry
    assert second.result(timeout=3) is second_registry


def test_shutdown_cancels_unregistered_plan_without_starting_work(setup_queue):
    make, _ = setup_queue
    manager = make()
    calls = []
    future = manager.submit_group("model_1", "dut:a", calls.append, "unexpected")
    manager.finish_registration("model_1")
    manager.shutdown(wait=True, cancel_futures=True)
    assert future.cancelled()
    assert calls == []


def test_physical_identity_normalizes_ipv6_without_user_port_or_setup_aliases():
    key = execution_binding.physical_dut_key
    first = {"id": "one", "remote": {
        "host": "2001:db8:0:0:0:0:0:1", "user": "nx", "port": 22,
    }}
    alias = {"id": "two", "runtime": {
        "host": "2001:db8::1", "user": "other", "port": 2222,
    }}
    assert key(first) == key(alias)
    assert key(alias) != key({"id": "one", "remote": {"host": "2001:db8::2"}})


def test_explicit_bound_physical_identity_unifies_distinct_interfaces():
    key = execution_binding.physical_dut_key
    first = {"id": "wired", "physical_host_id": "physical-dut-a", "remote": {"host": "192.0.2.20"}}
    alias = {"id": "alternate", "physical_host_id": "physical-dut-a", "runtime": {"host": "198.51.100.20"}}
    assert key(first) == key(alias)
    assert key(first) != key({**alias, "physical_host_id": "physical-dut-b"})


def test_unknown_model_cannot_extend_frozen_queue_scope(setup_queue):
    make, _ = setup_queue
    manager = make(model_ids=("model_1",))
    with pytest.raises((ValueError, RuntimeError)):
        manager.submit_group("unselected_model", "dut:a", lambda: None)
    manager.finish_registration("model_1")
