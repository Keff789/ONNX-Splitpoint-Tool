"""Two real statistics contexts with tiny inputs and owned spawn workers.

Reference-file locks are controlled external leaves, not replacements for the
product scheduler, resource decision, loader, evaluator, cache or result writer.
All files and child processes belong to the test's external pytest directory.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import json
import multiprocessing as mp
import os
import threading
import time

import pytest

from onnx_splitpoint_tool import quality_statistics_blocks as blocks
from onnx_splitpoint_tool.quality_service import (
    ManagementQualityService,
    ResourcePauseGate,
    _reference_store_file_lock,
    prepare_evaluation,
)

from test_quality_speed_block_review import _legacy, _same_science
from test_quality_speed_blocks import _ordered_vectors, _request as classification_request
from test_quality_speed_coco import _request as detection_request


def _options(contexts=2):
    return {
        "engine": "optimized_coco_v1", "max_active_requests": contexts,
        "block_repetitions": 4, "checkpoint_blocks": True,
        "prepared_cache_limit_mib": 16, "capture_draws": True,
    }


def _wait_until(predicate, *, clients=(), timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        observed = predicate()
        if observed:
            return observed
        for client in clients:
            if client.done() and client.exception() is not None:
                client.result()
        if time.monotonic() >= deadline:
            raise AssertionError("controlled statistics rendezvous did not complete")
        time.sleep(0.01)


def _reference_identity(cache, request):
    _, payload = prepare_evaluation(request)
    plan = blocks.prepare_plan(cache, payload)
    return blocks.reference_key(payload, plan)


def _contexts(service):
    snapshot = service.admission_snapshot()
    contexts = snapshot.get("contexts")
    assert isinstance(contexts, dict), "admission must identify every active request"
    return contexts


@pytest.mark.parametrize("contexts", [1, 2])
def test_independent_short_request_passes_waiting_reference_only_with_two_contexts(tmp_path, contexts):
    first = replace(detection_request((True, False), (False, True), repetitions=17),
                    request_id="held-detection-reference")
    second = replace(classification_request(17), request_id="independent-classification")
    cache = tmp_path / "cache"
    reference = _reference_identity(cache, first)
    with ManagementQualityService(cache, workers=2, statistics=_options(contexts)) as service:
        with _reference_store_file_lock(cache / "reference_vectors", reference):
            waiting = service.submit(first)
            _wait_until(lambda: any(row.get("request_id") == first.request_id and
                                   row.get("phase") == "waiting_reference"
                                   for row in _contexts(service).values()), clients=[waiting])
            ready = service.submit(second)
            if contexts == 2:
                result = ready.result(timeout=10)
                assert result["request_id"] == second.request_id
                assert not waiting.done(), "second request must finish before the held first reference"
            else:
                # A later request remains queued under explicitly retained
                # default/legacy one-context scheduling.
                assert not ready.done()
                assert len(_contexts(service)) == 1
        assert waiting.result(timeout=10)["request_id"] == first.request_id
        assert ready.result(timeout=10)["request_id"] == second.request_id
    assert service.shutdown_state()["finished"] is True


def test_two_contexts_share_one_pool_and_one_cold_reference_with_exact_draws(tmp_path):
    requests = [replace(detection_request((True, False), candidate, repetitions=17),
                        request_id=f"detection-candidate-{index}")
                for index, candidate in enumerate(((True, True), (False, True)))]
    baselines = [_legacy(request) for request in requests]
    cache = tmp_path / "cache"
    reference = _reference_identity(cache, requests[0])
    assert _reference_identity(cache, requests[1]) == reference
    with ManagementQualityService(cache, workers=2, statistics=_options()) as service:
        executor = service._executor
        with _reference_store_file_lock(cache / "reference_vectors", reference):
            clients = [service.submit(request) for request in requests]
            _wait_until(lambda: len(_contexts(service)) == 2, clients=clients)
            assert not any(client.done() for client in clients)
        results = [client.result(timeout=15) for client in clients]
        assert service._executor is executor
        owned = {process.pid for process in executor._processes.values()}
        assert 1 <= len(owned) <= service.workers <= 2
        assert all(pid != os.getpid() for pid in owned)
        observations = [result["statistics_observation"] for result in results]
        assert sorted(observation["reference_cache_hit"] for observation in observations) == [False, True]
        reference_rows = [row for observation in observations for row in observation["reference_shards"]]
        assert sum(row["repetition_stop"] - row["repetition_start"] for row in reference_rows) == 17
        assert len(list((cache / "reference_vectors").glob("*.json"))) == 1
        for result, (baseline, expected) in zip(results, baselines):
            _same_science(result, expected)
            capture = json.loads((cache / "draws" / (result["evaluation_fingerprint"] + ".json")).read_text())
            assert _ordered_vectors(capture["shards"], 17) == _ordered_vectors([baseline], 17)
            for phase in ("shards", "reference_shards"):
                rows = result["statistics_observation"][phase]
                prepared = Counter()
                for row in rows:
                    assert row["worker_pid"] in owned
                    prepared[row["worker_pid"]] += row["prepare_count"]
                    assert row["prepared_cache_bytes"] <= 16 * 1024**2
                assert all(count == 1 for count in prepared.values())
    assert service.shutdown_state()["finished"] is True


def test_admission_keeps_both_request_identities_and_holds_third_context(tmp_path):
    requests = [replace(classification_request(17), seed=731 + index,
                        request_id=f"identified-context-{index}") for index in range(3)]
    cache = tmp_path / "cache"
    keys = [_reference_identity(cache, request) for request in requests]
    with ManagementQualityService(cache, workers=2, statistics=_options()) as service:
        with ExitStack() as held:
            for key in keys:
                held.enter_context(_reference_store_file_lock(cache / "reference_vectors", key))
            clients = [service.submit(request) for request in requests]
            rows = _wait_until(lambda: _contexts(service) if len(_contexts(service)) == 2 else None,
                               clients=clients)
            _wait_until(lambda: all(row.get("phase") == "waiting_reference"
                                   for row in _contexts(service).values()), clients=clients)
            rows = _contexts(service)
            assert {row["request_id"] for row in rows.values()} == {request.request_id for request in requests[:2]}
            assert all(row.get("evaluation_fingerprint", key) == key for key, row in rows.items())
            assert service.admission_snapshot()["queued_descriptors"] >= 1
            assert not any(client.done() for client in clients)
            assert service.progress_snapshot() == [], "reference wait is not active draw computation"
        assert [client.result(timeout=15)["request_id"] for client in clients] == [request.request_id for request in requests]
    assert service.shutdown_state()["finished"] is True


def _block_descriptor(root, request):
    key, payload = prepare_evaluation(request)
    payload["_statistics"] = _options()
    descriptor = blocks.spool_payload(root, key, payload)
    descriptor["plan"] = blocks.prepare_plan(root, payload)
    descriptor["reuse_reference"] = True
    return descriptor


def test_worker_retains_two_private_preparations_across_alternating_requests(tmp_path):
    # An actual one-process worker receives A/B/A/B/C/B/A deterministically. This
    # isolates cache locality from OS scheduling without mocking preparation.
    first = _block_descriptor(tmp_path / "first", replace(classification_request(17), request_id="cache-A"))
    second = _block_descriptor(tmp_path / "second", replace(classification_request(17), seed=732, request_id="cache-B"))
    third = _block_descriptor(tmp_path / "third", replace(classification_request(17), seed=733, request_id="cache-C"))
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        results = [pool.submit(blocks.evaluate_block, descriptor, "reference", start, start + 4).result(timeout=10)
                   for descriptor, start in ((first, 0), (second, 0), (first, 4), (second, 4),
                                             (third, 0), (second, 8), (first, 8))]
    rows = [result["statistics_observation"] for result in results]
    assert len({row["worker_pid"] for row in rows}) == 1
    assert rows[0]["worker_pid"] != os.getpid()
    # The third key evicts the oldest of the two slots. A continuously growing
    # cache would make the final A a hit and must also fail this regression.
    assert [row["prepare_count"] for row in rows] == [1, 1, 0, 0, 1, 0, 1]
    assert [row["prepared_cache_hit"] for row in rows] == [False, False, True, True, False, True, False]
    assert all(row["prepared_cache_bytes"] <= 16 * 1024**2 for row in rows)


def test_registered_reference_cpu_tokens_prevent_statistics_oversubscription(tmp_path):
    gate = ResourcePauseGate(available_cpu=2)
    started = threading.Event()
    finished = threading.Event()
    request = replace(classification_request(17), request_id="shared-cpu-budget")
    with ManagementQualityService(tmp_path / "cache", workers=2, statistics=_options(), pause_gate=gate) as service:
        def evaluate():
            started.set()
            try:
                return service.evaluate(request, timeout=10)
            finally:
                finished.set()

        with ThreadPoolExecutor(max_workers=1) as submitter:
            with gate.activity("controlled-reference-session", cpu=2):
                future = submitter.submit(evaluate)
                assert started.wait(timeout=2)
                assert not finished.wait(timeout=0.1)
                assert gate.snapshot()["cpu_active"] == 2
                assert service.progress_snapshot() == []
            assert future.result(timeout=15)["request_id"] == request.request_id
    assert service.shutdown_state()["finished"] is True
    assert gate.snapshot()["cpu_active"] == 0
    assert gate.snapshot()["memory_bytes_active"] == 0


def test_waiting_second_context_memory_cannot_starve_work_that_releases_first_context():
    gate = ResourcePauseGate(available_cpu=2, available_memory_bytes=1024)
    second_admitted = threading.Event()

    def second_context():
        with gate.activity("second-context-memory", cpu=0, memory_bytes=768,
                           resources=(), timeout=2):
            second_admitted.set()

    with ThreadPoolExecutor(max_workers=1) as submitter:
        with gate.activity("first-context-memory", cpu=0, memory_bytes=768, resources=()):
            future = submitter.submit(second_context)
            _wait_until(lambda: gate.snapshot()["waiting"] >= 1, timeout=1)
            assert not second_admitted.is_set()
            # The first context must still be able to compute its remaining
            # block and release RAM needed by the older waiting admission.
            token = gate.acquire_activity("first-context-next-block", cpu=1, timeout=0.2)
            try:
                assert token is not None, "RAM waiter blocks the CPU work required to release RAM"
                assert gate.snapshot()["memory_bytes_active"] == 768
                assert gate.snapshot()["cpu_active"] == 1
            finally:
                if token is not None:
                    gate.release_activity(token)
        future.result(timeout=2)
    assert second_admitted.is_set()
    assert gate.snapshot()["memory_bytes_active"] == 0
