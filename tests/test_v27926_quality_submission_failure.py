"""Worker submission failures must leave every client terminal, not at 3/4."""
from concurrent.futures import CancelledError, Future
from concurrent.futures.process import BrokenProcessPool

import pytest

from onnx_splitpoint_tool import quality_service as quality


def _request(candidate_value=0.5):
    return quality.QualityEvaluationRequest(
        reference_records=[{"image_id": n, "value": 1.0} for n in range(4)],
        candidate_records=[{"image_id": n, "value": candidate_value} for n in range(4)],
        annotations=[],
        metric_gate_config={"primary_metric": "top1_accuracy"},
        repetitions=4,
        seed=42,
        confidence_level=0.95,
        non_inferiority_margin=0.25,
        evaluator_factory="paired_mean",
        value_field="value",
    )


class _BrokenExecutor:
    """Accept at most one shard, then fail every subsequent submission."""

    def __init__(self, accepted=None):
        self.accepted = accepted
        self.calls = 0
        self.submit_error = BrokenProcessPool("submission rejected")
        self.shutdown_calls = []

    def submit(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1 and self.accepted is not None:
            return self.accepted
        raise self.submit_error

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


class _InlineExecutor:
    """Run real shard calculations with synchronous completion callbacks."""

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    def shutdown(self, **kwargs):
        pass


def _service(monkeypatch, tmp_path, executor):
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kwargs: executor)
    return quality.ManagementQualityService(tmp_path / "quality-cache", workers=4)


def _assert_capacity_restored(service):
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
    assert not service._inflight
    assert not service._dispatcher.is_alive()


@pytest.mark.parametrize("callback_order", ["before_error", "after_error"])
@pytest.mark.parametrize("worker_failed", [False, True])
def test_accepted_callback_and_submit_error_always_finish_once(
    monkeypatch, tmp_path, callback_order, worker_failed,
):
    accepted = Future()
    worker_error = BrokenProcessPool("worker exited")

    def complete_worker():
        if worker_failed:
            accepted.set_exception(worker_error)
        else:
            accepted.set_result({"shard_index": 0})

    if callback_order == "before_error":
        complete_worker()
    executor = _BrokenExecutor(accepted)
    service = _service(monkeypatch, tmp_path, executor)
    try:
        client = service.submit(_request())
        completions = []
        client.add_done_callback(lambda done: completions.append(done))
        # A different logical request reaches the executor only after the
        # dispatcher's first submit-error handler has finished. No sleep or
        # timing assumption is needed to exercise callback-after-error. An
        # identical-prediction request needs only one of the three free slots.
        following = service.submit(_request(1.0))
        with pytest.raises(BrokenProcessPool, match="submission rejected"):
            following.result(timeout=5)
        if callback_order == "after_error":
            assert not client.done()
            complete_worker()
        expected = (
            worker_error
            if callback_order == "before_error" and worker_failed
            else executor.submit_error
        )
        with pytest.raises(BrokenProcessPool) as raised:
            client.result(timeout=5)
        assert raised.value is expected
        assert completions == [client]
        assert not list((tmp_path / "quality-cache").rglob("*.json"))
    finally:
        if not accepted.done():
            accepted.cancel()
        service.shutdown(wait=True, cancel_futures=True)
    _assert_capacity_restored(service)


@pytest.mark.parametrize("accept_first", [False, True])
def test_all_four_clients_finish_when_pool_breaks(monkeypatch, tmp_path, accept_first):
    accepted = Future() if accept_first else None
    if accepted is not None:
        accepted.set_exception(BrokenProcessPool("spawn main missing"))
    executor = _BrokenExecutor(accepted)
    service = _service(monkeypatch, tmp_path, executor)
    try:
        clients = [service.submit(_request(value)) for value in (0.1, 0.2, 0.3, 0.4)]
        for client in clients:
            with pytest.raises(BrokenProcessPool):
                client.result(timeout=5)
        assert sum(client.done() for client in clients) == 4
        assert executor.calls == (5 if accept_first else 4)
    finally:
        service.shutdown(wait=True, cancel_futures=True)
    _assert_capacity_restored(service)


def test_success_and_cache_reuse_still_finish_once(monkeypatch, tmp_path):
    service = _service(monkeypatch, tmp_path, _InlineExecutor())
    try:
        service.pause("queue duplicate clients before dispatch")
        request = _request()
        clients = [service.submit(request), service.submit(request)]
        completions = []
        for client in clients:
            client.add_done_callback(lambda done: completions.append(done))
        service.resume("queue duplicate clients before dispatch")
        results = [client.result(timeout=5) for client in clients]
        assert all(result["status"] == "completed" for result in results)
        assert all(result["bootstrap_workers_requested"] == 4 for result in results)
        cached = service.evaluate(request, timeout=5)
        assert cached["cache_hit"] is True
        assert cached["decision"] == results[0]["decision"]
        assert len(list((tmp_path / "quality-cache").rglob("*.json"))) == 1
    finally:
        service.shutdown(wait=True, cancel_futures=True)
    _assert_capacity_restored(service)
    assert completions == clients


def test_cancel_preserves_prior_submission_error_and_late_callback_restores_capacity(monkeypatch, tmp_path):
    accepted = Future()
    executor = _BrokenExecutor(accepted)
    service = _service(monkeypatch, tmp_path, executor)
    try:
        client = service.submit(_request())
        completions = []
        client.add_done_callback(lambda done: completions.append(done))
        # This following request is the dispatch synchronization point.
        following = service.submit(_request(1.0))
        with pytest.raises(BrokenProcessPool):
            following.result(timeout=5)
        assert not client.done()
        service.shutdown(wait=False, cancel_futures=True, terminate_workers=True)
        # v2.80.4: submission already failed before this user cancellation.
        # Preserve that real error instead of replacing it with CancelledError.
        with pytest.raises(BrokenProcessPool, match="submission rejected"):
            client.result(timeout=5)
        accepted.set_exception(BrokenProcessPool("worker stopped after cancellation"))
        assert not client.cancelled()
        assert completions == [client]
        assert not list((tmp_path / "quality-cache").rglob("*.json"))
    finally:
        if not accepted.done():
            accepted.cancel()
        service.shutdown(wait=True, cancel_futures=True)
    _assert_capacity_restored(service)
