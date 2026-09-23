"""Legacy finalization keeps resource ownership through real cache publication.

Only the outer executor timing and an I/O rendezvous are controlled. Shard
evaluation, merge, cache format and every gate decision are production code.
"""
from concurrent.futures import Future
from dataclasses import replace
import threading

import pytest

from onnx_splitpoint_tool import quality_service as quality


def _request():
    return quality.QualityEvaluationRequest(
        reference_records=[{"image_id": i, "value": 1.0} for i in range(4)],
        candidate_records=[{"image_id": i, "value": .9} for i in range(4)],
        annotations=[], metric_gate_config={"primary_metric": "top1_accuracy"},
        repetitions=8, seed=42, confidence_level=.95,
        non_inferiority_margin=.25, evaluator_factory="paired_mean", value_field="value",
    )


class ControlledExecutor:
    def __init__(self):
        self.jobs = []
        self.ready = threading.Event()

    def submit(self, function, *args, **kwargs):
        future = Future()
        self.jobs.append((future, function, args, kwargs))
        if len(self.jobs) == 2:
            self.ready.set()
        return future

    def complete(self, index):
        future, function, args, kwargs = self.jobs[index]
        future.set_result(function(*args, **kwargs))

    def shutdown(self, **kwargs):
        if kwargs.get("cancel_futures"):
            for future, *_ in self.jobs:
                future.cancel()


def _service(monkeypatch, tmp_path):
    pool = ControlledExecutor()
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kwargs: pool)
    gate = quality.ResourcePauseGate(available_cpu=2, available_memory_bytes=64 * 1024**2)
    service = quality.ManagementQualityService(tmp_path / "cache", workers=2, pause_gate=gate)
    return service, gate, pool


def _assert_all_worker_permits_released(service):
    acquired = 0
    try:
        for _ in range(2):
            assert service._worker_slots.acquire(blocking=False)
            acquired += 1
        assert not service._worker_slots.acquire(blocking=False)
    finally:
        for _ in range(acquired):
            service._worker_slots.release()


def test_duplicate_earlier_callback_cannot_release_finalizers_cache_io(monkeypatch, tmp_path):
    service, gate, pool = _service(monkeypatch, tmp_path)
    io_entered, release_io = threading.Event(), threading.Event()
    real_put = service.cache.put

    def pending_io(key, value):
        io_entered.set()
        assert release_io.wait(5), "test cache I/O rendezvous was not released"
        return real_put(key, value)

    monkeypatch.setattr(service.cache, "put", pending_io)
    publisher = None
    reservation = None
    try:
        client = service.submit(_request())
        assert pool.ready.wait(5)
        group = next(iter(service._groups.values()))
        pool.complete(0)
        assert group.pending == 1
        publisher = threading.Thread(target=pool.complete, args=(1,))
        publisher.start()
        assert io_entered.wait(5)
        assert group.pending == 0
        assert not client.done()
        assert gate.snapshot()["cpu_active"] == 2

        # The old wrapper saw pending==0 after this duplicate returned and
        # released the finalizer's token despite its outstanding cache I/O.
        service._shard_worker_done(group, pool.jobs[0][0])
        reservation = gate.begin_quiet("collector-after-shards")
        assert gate.poll_quiet(reservation)["state"] == "DRAINING"
        assert gate.snapshot()["cpu_active"] == 2
        assert gate.snapshot()["memory_bytes_active"] > 0
        assert not list((tmp_path / "cache").rglob("*.json"))

        release_io.set()
        publisher.join(5)
        assert not publisher.is_alive()
        result = client.result(timeout=5)
        assert result["status"] == "completed"
        assert service.cache.get(group.key)["status"] == "completed"
        assert gate.poll_quiet(reservation)["state"] == "QUIET_CONFIRMED"
        assert gate.snapshot()["cpu_active"] == 0
        assert gate.snapshot()["memory_bytes_active"] == 0
        _assert_all_worker_permits_released(service)
    finally:
        release_io.set()
        if publisher is not None:
            publisher.join(5)
        if reservation is not None:
            gate.end_quiet(reservation)
        service.shutdown(wait=True, cancel_futures=True)


def test_plan_allocation_failure_releases_cpu_ram_and_worker_permits(monkeypatch, tmp_path):
    service, gate, pool = _service(monkeypatch, tmp_path)
    real_plan = quality.deterministic_resample_plan
    failure = MemoryError("controlled draw-plan allocation failure")

    def fail_first_plan(**kwargs):
        if kwargs["seed"] == 42:
            raise failure
        return real_plan(**kwargs)

    monkeypatch.setattr(quality, "deterministic_resample_plan", fail_first_plan)
    try:
        failed = service.submit(_request())
        with pytest.raises(MemoryError) as caught:
            failed.result(timeout=5)
        assert caught.value is failure
        assert pool.jobs == []
        assert gate.snapshot()["cpu_active"] == 0
        assert gate.snapshot()["memory_bytes_active"] == 0
        assert not gate.snapshot()["activities"]
        assert service._dispatcher.is_alive()
        _assert_all_worker_permits_released(service)
        assert not list((tmp_path / "cache").rglob("*.json"))

        # A later real request must still use every slot and publish normally.
        client = service.submit(replace(_request(), seed=43))
        assert pool.ready.wait(5)
        pool.complete(0)
        pool.complete(1)
        assert client.result(timeout=5)["status"] == "completed"
        assert gate.snapshot()["cpu_active"] == 0
        assert gate.snapshot()["memory_bytes_active"] == 0
        _assert_all_worker_permits_released(service)
    finally:
        service.shutdown(wait=True, cancel_futures=True)
