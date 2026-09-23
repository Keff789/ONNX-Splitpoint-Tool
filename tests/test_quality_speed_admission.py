"""Tiny AP04 resource admission, queue failure, and terminal publication tests."""
from concurrent.futures import CancelledError, Future
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool import quality_service as quality
from onnx_splitpoint_tool import quality_statistics_blocks as blocks
from onnx_splitpoint_tool import quality_statistics_config as settings
from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY
from onnx_splitpoint_tool.quality_lifecycle import cancel_context


def _request(seed=731):
    return quality.QualityEvaluationRequest(
        reference_records=[{"image_id": i, "value": value}
                           for i, value in enumerate((1.0, 0.0, 1.0, 0.0))],
        candidate_records=[{"image_id": i, "value": value}
                           for i, value in enumerate((1.0, 1.0, 0.0, 0.0))],
        annotations=[], metric_gate_config={"primary_metric": "mean",
            "reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY)},
        repetitions=7, seed=seed, confidence_level=0.95,
        non_inferiority_margin=0.1, evaluator_factory="paired_mean",
        request_id=f"synthetic-admission-{seed}")


def _budget(monkeypatch, *, slots=4, available=8 * 1024**3):
    monkeypatch.setattr(settings, "resource_budget", lambda: {
        "affinity_cpus": 8, "quota_cpus": slots + 1,
        "available_memory_bytes": available,
        "statistics_cpu_slots": slots, "controller_cpu_reserve": 1})


def _service(tmp_path, *, workers=2):
    return quality.ManagementQualityService(tmp_path / "cache", workers=workers, statistics={
        "engine": "optimized_coco_v1", "block_repetitions": 2,
        "checkpoint_blocks": True, "prepared_cache_limit_mib": 16})


class InlineExecutor:
    def __init__(self):
        self.calls = 0

    def submit(self, fn, *args, **kwargs):
        self.calls += 1
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass


def _inline(monkeypatch):
    pool = InlineExecutor()
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kwargs: pool)
    return pool


def _assert_restored(service):
    claimed = []
    try:
        for _ in range(service.workers):
            claimed.append(service._worker_slots.acquire(blocking=False))
        assert claimed == [True] * service.workers
        assert service._worker_slots.acquire(blocking=False) is False
    finally:
        for acquired in claimed:
            if acquired:
                service._worker_slots.release()
    assert service.shutdown_state()["finished"] is True
    assert not service._inflight
    assert not service._groups


@pytest.mark.parametrize("requested,slots,effective", [(7, 2, 2), (2, 5, 2), (4, 1, 1)])
def test_resource_budget_reaches_actual_spawn_pool_and_preserves_requested_count(
    tmp_path, monkeypatch, requested, slots, effective,
):
    _budget(monkeypatch, slots=slots)
    original = quality.ProcessPoolExecutor
    observed = []

    class CapturedPool(original):
        def __init__(self, **kwargs):
            observed.append((kwargs["max_workers"], kwargs["mp_context"].get_start_method()))
            super().__init__(**kwargs)

    monkeypatch.setattr(quality, "ProcessPoolExecutor", CapturedPool)
    with _service(tmp_path, workers=requested) as service:
        actual = service.evaluate(_request(), timeout=20)
        processes = list(service._executor._processes.values())
        assert 1 <= len(processes) <= effective
        assert all(process.is_alive() for process in processes)
        assert service.workers == effective
        assert service.workers_requested == requested
    assert observed == [(effective, "spawn")]
    assert actual["bootstrap_workers_requested"] == requested
    assert actual["bootstrap_workers_effective"] == effective
    assert actual["statistics_observation"]["workers_requested"] == requested
    assert actual["statistics_observation"]["workers_effective"] == effective
    _assert_restored(service)


@pytest.mark.parametrize("resource,exception,match", [
    ("ram", MemoryError, "statistics admission needs"),
    ("disk", OSError, "insufficient scratch disk"),
])
def test_admission_rejects_before_plan_generation_or_worker_submission(
    tmp_path, monkeypatch, resource, exception, match,
):
    _budget(monkeypatch, available=1 if resource == "ram" else 8 * 1024**3)
    if resource == "disk":
        monkeypatch.setattr(blocks.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    pool = _inline(monkeypatch)

    def no_plan(*args, **kwargs):
        pytest.fail("resource refusal reached plan materialization")

    monkeypatch.setattr(blocks, "prepare_plan", no_plan)
    with _service(tmp_path) as service:
        client = service.submit(_request())
        completions = []
        client.add_done_callback(completions.append)
        with pytest.raises(exception, match=match):
            client.result(timeout=5)
        assert completions == [client]
        assert pool.calls == 0
        assert not (tmp_path / "cache" / "plans").exists()
        assert not list((tmp_path / "cache").glob("[0-9a-f][0-9a-f]/*.json"))
    _assert_restored(service)


class RejectingExecutor:
    def __init__(self, *, accept_first=False, synchronous=False):
        self.accept_first = accept_first
        self.synchronous = synchronous
        self.accepted = Future()
        self.accepted_result = None
        self.rejected = threading.Event()
        self.error = BrokenProcessPool("synthetic optimized submission rejected")
        self.calls = 0

    def submit(self, fn, *args, **kwargs):
        self.calls += 1
        if self.accept_first and self.calls == 1:
            self.accepted_result = fn(*args, **kwargs)
            if self.synchronous:
                self.accepted.set_result(self.accepted_result)
            return self.accepted
        self.rejected.set()
        raise self.error

    def shutdown(self, **kwargs):
        pass


@pytest.mark.parametrize("accepted,synchronous", [(False, False), (True, False), (True, True)])
def test_initial_and_partial_submit_failures_finish_each_waiter_once_and_restore_slots(
    tmp_path, monkeypatch, accepted, synchronous,
):
    _budget(monkeypatch)
    pool = RejectingExecutor(accept_first=accepted, synchronous=synchronous)
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kwargs: pool)
    service = _service(tmp_path)
    clients = []
    completions = []
    try:
        service.pause("queue-duplicates")
        clients = [service.submit(_request()), service.submit(_request()), service.submit(_request(732))]
        for client in clients:
            client.add_done_callback(completions.append)
        service.resume("queue-duplicates")
        assert pool.rejected.wait(5)
        if accepted and not synchronous:
            assert not clients[0].done()
            group = next(iter(service._groups.values()))
            pool.accepted.set_result(pool.accepted_result)
            service._block_worker_done(group, pool.accepted)  # duplicate terminal callback is inert
        for client in clients:
            with pytest.raises(BrokenProcessPool, match="optimized submission rejected") as caught:
                client.result(timeout=5)
            assert caught.value is pool.error
        assert len(completions) == len(set(completions)) == len(clients)
        assert not list((tmp_path / "cache").glob("[0-9a-f][0-9a-f]/*.json"))
    finally:
        if accepted and not pool.accepted.done():
            pool.accepted.cancel()
        service.shutdown(cancel_futures=True)
    _assert_restored(service)


def test_synchronous_completed_blocks_release_slots_and_publish_deduplicated_clients_once(tmp_path, monkeypatch):
    _budget(monkeypatch)
    pool = _inline(monkeypatch)
    service = _service(tmp_path, workers=4)
    clients, completions = [], []
    try:
        service.pause("queue-duplicates")
        clients = [service.submit(_request()), service.submit(_request())]
        for client in clients:
            client.add_done_callback(completions.append)
        service.resume("queue-duplicates")
        results = [client.result(timeout=5) for client in clients]
        assert pool.calls == 4
        assert results[0] == results[1]
        assert len(completions) == len(set(completions)) == 2
        assert results[0]["primary"]["bootstrap_repetitions"] == 7
        assert results[0]["statistics_observation"]["draws_recomputed"] == 7
    finally:
        service.shutdown()
    _assert_restored(service)


def test_complete_publication_wins_competing_cancel_once_and_cache_is_readable_after_shutdown(
    tmp_path, monkeypatch,
):
    _budget(monkeypatch)
    _inline(monkeypatch)
    service = _service(tmp_path)
    entered, release, stopping = threading.Event(), threading.Event(), threading.Event()
    put_calls, completions = [], []
    original = service.cache.put

    def held_put(key, result):
        put_calls.append(key)
        entered.set()
        assert release.wait(5)
        return original(key, result)

    monkeypatch.setattr(service.cache, "put", held_put)
    stopper_errors = []

    def stop():
        stopping.set()
        try:
            service.shutdown(cancel_futures=True, cancellation_context=cancel_context("publication-test"))
        except BaseException as exc:
            stopper_errors.append(exc)

    stopper = threading.Thread(target=stop)
    try:
        service.pause("queue-duplicates")
        clients = [service.submit(_request()), service.submit(_request())]
        for client in clients:
            client.add_done_callback(completions.append)
        service.resume("queue-duplicates")
        assert entered.wait(5)
        stopper.start()
        assert stopping.wait(5)
        release.set()
        results = [client.result(timeout=5) for client in clients]
        stopper.join(5)
        assert not stopper.is_alive()
        assert not stopper_errors
        assert len(put_calls) == 1
        assert len(completions) == len(set(completions)) == 2
        assert all(result["status"] == "completed" for result in results)
        cached = service.evaluate(_request(), timeout=5)
        assert cached["cache_hit"] is True
        assert cached["statistics_observation"]["draws_recomputed"] == 0
        assert cached["statistics_observation"]["cache_state"] == "complete_result_reused"
        with pytest.raises(quality.QualityServiceClosedError):
            service.submit(_request(999))
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(5)
        service.shutdown(cancel_futures=True)
    _assert_restored(service)


def test_cancel_before_merge_prevents_final_cache_publication_and_finishes_once(tmp_path, monkeypatch):
    _budget(monkeypatch)
    _inline(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = blocks.run_phase

    def held_phase(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(blocks, "run_phase", held_phase)
    service = _service(tmp_path)
    publications = []
    monkeypatch.setattr(service.cache, "put", lambda *args: publications.append(args))
    try:
        client = service.submit(_request())
        completions = []
        client.add_done_callback(completions.append)
        assert entered.wait(5)
        service.shutdown(wait=False, cancel_futures=True,
                         cancellation_context=cancel_context("cancel-before-publication"))
        with pytest.raises(CancelledError):
            client.result(timeout=5)
        release.set()
        service.shutdown()
        assert completions == [client]
        assert publications == []
        assert not list((tmp_path / "cache").glob("[0-9a-f][0-9a-f]/*.json"))
    finally:
        release.set()
        service.shutdown(cancel_futures=True)
    _assert_restored(service)


def test_spool_failure_leaves_no_inflight_waiters_and_next_submission_can_complete(tmp_path, monkeypatch):
    _budget(monkeypatch)
    _inline(monkeypatch)
    with _service(tmp_path) as service:
        with monkeypatch.context() as context:
            def fail_spool(*args, **kwargs):
                raise OSError("synthetic spool write failed")
            context.setattr(blocks, "spool_payload", fail_spool)
            with pytest.raises(OSError, match="spool write failed"):
                service.submit(_request())
            assert not service._inflight
            assert service._queue.qsize() == 0
        actual = service.evaluate(_request(), timeout=5)
        assert actual["status"] == "completed"
    _assert_restored(service)


def test_queue_insert_failure_does_not_leave_a_deduplicated_waiter_without_work(tmp_path, monkeypatch):
    _budget(monkeypatch)
    _inline(monkeypatch)
    service = _service(tmp_path)
    try:
        with monkeypatch.context() as context:
            def fail_put(*args, **kwargs):
                raise OSError("synthetic queue insert failed")
            context.setattr(service._queue, "put", fail_put)
            with pytest.raises(OSError, match="queue insert failed"):
                service.submit(_request())
            assert not service._inflight
            assert not list(service._payload_scratch.rglob("*.json"))
        actual = service.evaluate(_request(), timeout=5)
        assert actual["status"] == "completed"
    finally:
        service.shutdown(cancel_futures=True)
    _assert_restored(service)


def test_corrupt_queued_payload_fails_its_clients_and_does_not_strand_following_request(tmp_path, monkeypatch):
    _budget(monkeypatch)
    _inline(monkeypatch)
    original = blocks.spool_payload
    descriptors = []

    def capture(*args, **kwargs):
        descriptor = original(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(blocks, "spool_payload", capture)
    service = _service(tmp_path)
    completions = []
    try:
        service.pause("queue-corruption-test")
        first = service.submit(_request())
        duplicate = service.submit(_request())
        following = service.submit(_request(732))
        for client in (first, duplicate, following):
            client.add_done_callback(completions.append)
        assert len(descriptors) == 2
        Path(descriptors[0]["path"]).write_text("corrupt synthetic descriptor payload")
        service.resume("queue-corruption-test")
        for client in (first, duplicate):
            with pytest.raises(ValueError, match="descriptor integrity mismatch"):
                client.result(timeout=5)
        assert following.result(timeout=5)["status"] == "completed"
        assert len(completions) == len(set(completions)) == 3
    finally:
        service.shutdown(cancel_futures=True)
    _assert_restored(service)


@pytest.mark.parametrize("ending", ["completed", "load_failure", "queued_cancel"])
def test_terminal_request_removes_only_its_own_spool_and_preserves_retained_artifacts(
    tmp_path, monkeypatch, ending,
):
    _budget(monkeypatch)
    _inline(monkeypatch)
    service = _service(tmp_path)
    original = blocks.spool_payload
    descriptors = []
    retained = {}
    foreign_input = tmp_path / "retained-original-input.json"
    foreign_input.write_bytes(b'{"original": "preserve byte for byte"}')
    foreign_bytes = foreign_input.read_bytes()

    def capture(*args, **kwargs):
        descriptor = original(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(blocks, "spool_payload", capture)
    try:
        service.pause("controlled-spool-inspection")
        client = service.submit(_request())
        assert len(descriptors) == 1
        descriptor = descriptors[0]
        owned_path = Path(descriptor["path"])
        assert owned_path.is_file()
        if ending == "load_failure":
            owned_path.write_bytes(b"invalid synthetic payload")
        if ending == "queued_cancel":
            service.shutdown(cancel_futures=True,
                             cancellation_context=cancel_context("owned-spool-cancel"))
            with pytest.raises(CancelledError):
                client.result(timeout=5)
        else:
            service.resume("controlled-spool-inspection")
            if ending == "load_failure":
                with pytest.raises(ValueError, match="descriptor integrity mismatch"):
                    client.result(timeout=5)
            else:
                actual = client.result(timeout=5)
                key = actual["evaluation_fingerprint"]
                checkpoints = list((service.cache.root / "checkpoints" / key).glob("*.json"))
                plans = list((service.cache.root / "plans").glob("*"))
                complete = service.cache.path_for(key)
                assert len(checkpoints) == 4
                assert any(path.suffix == ".npy" for path in plans)
                assert complete.is_file()
                retained = {path: path.read_bytes() for path in checkpoints + plans + [complete] if path.is_file()}
            service.shutdown()
        assert not owned_path.exists()
        assert not service._payload_scratch.exists()
        assert foreign_input.read_bytes() == foreign_bytes
        assert all(path.is_file() and path.read_bytes() == encoded for path, encoded in retained.items())
    finally:
        service.shutdown(cancel_futures=True)
    _assert_restored(service)


def test_spool_cleanup_refuses_a_foreign_path_even_with_an_owned_key(tmp_path, monkeypatch):
    _budget(monkeypatch)
    _inline(monkeypatch)
    foreign = tmp_path / "original-run-predictions.json"
    foreign.write_bytes(b'{"original": "untouched"}')
    before = foreign.read_bytes()
    with _service(tmp_path) as service:
        with pytest.raises(ValueError, match="foreign statistics payload"):
            service._remove_spooled_payload({"key": "a" * 64, "path": str(foreign)})
        assert foreign.read_bytes() == before
    _assert_restored(service)
