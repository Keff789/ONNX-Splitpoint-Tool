"""Bounded AP04 checkpoint failure/resume tests on tiny synthetic CPU data."""
from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from onnx_splitpoint_tool import quality_service as quality
from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY
from onnx_splitpoint_tool.quality_cache import json_fingerprint
from onnx_splitpoint_tool.quality_lifecycle import cancel_context


def _request(repetitions=7):
    return quality.QualityEvaluationRequest(
        reference_records=[{"image_id": i, "value": value}
                           for i, value in enumerate((1.0, 0.0, 1.0, 0.0))],
        candidate_records=[{"image_id": i, "value": value}
                           for i, value in enumerate((1.0, 1.0, 0.0, 0.0))],
        annotations=[], metric_gate_config={"primary_metric": "mean",
            "reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY)},
        repetitions=repetitions, seed=731, confidence_level=0.95,
        non_inferiority_margin=0.1, evaluator_factory="paired_mean",
        request_id="tiny-checkpoint-request")


def _service(cache, **options):
    return quality.ManagementQualityService(cache, workers=1, statistics={
        "engine": "optimized_coco_v1", "block_repetitions": 2,
        "checkpoint_blocks": True, "capture_draws": False, **options})


def _wait_for(predicate, *, timeout=15):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("bounded test rendezvous was not reached")
        time.sleep(0.01)


def _science(result):
    values = deepcopy(result)
    for name in ("statistics_observation", "evaluation_fingerprint", "cache_hit",
                 "bootstrap_workers_requested", "bootstrap_workers_effective",
                 "bootstrap_sharding", "worker_model"):
        values.pop(name, None)
    for component in (values["primary"], *values["guardrails"].values()):
        component.pop("bootstrap_elapsed_s", None)
        component.pop("bootstrap_engine", None)
    return values


@pytest.fixture
def expected():
    _, payload = quality.prepare_evaluation(_request())
    return quality._rebind_request_scoped_result(quality._evaluate_payload(payload), payload)


def _checkpoint_files(cache):
    return sorted((Path(cache) / "checkpoints").glob("*/*.json"))


def _fail_before_pair_publication(cache, monkeypatch):
    original = quality._combine_evaluation_shards

    def fail_merge(*args, **kwargs):
        raise RuntimeError("test failure before final pair publication")

    with monkeypatch.context() as context:
        context.setattr(quality, "_combine_evaluation_shards", fail_merge)
        with _service(cache) as service:
            with pytest.raises(RuntimeError, match="before final pair publication"):
                service.evaluate(_request(), timeout=20)
    assert quality._combine_evaluation_shards is original
    paths = _checkpoint_files(cache)
    assert {path.name for path in paths} == {"0-2.json", "2-4.json", "4-6.json", "6-7.json"}
    key = paths[0].parent.name
    assert not service.cache.path_for(key).exists()
    return key, paths


def test_completed_ranges_resume_without_capture_draws_or_final_pair_cache(tmp_path, monkeypatch, expected):
    cache = tmp_path / "cache"
    key, paths = _fail_before_pair_publication(cache, monkeypatch)
    snapshots = {path: path.read_bytes() for path in paths}

    def no_new_draws(*args, **kwargs):
        pytest.fail("complete checkpoint ranges must not submit new draws")

    with _service(cache) as service:
        monkeypatch.setattr(service._executor, "submit", no_new_draws)
        actual = service.evaluate(_request(), timeout=10)
        assert service.cache.path_for(key).is_file()
    assert actual["cache_hit"] is False
    assert actual["primary"]["bootstrap_repetitions"] == 7
    assert _science(actual) == _science(expected)
    observation = actual["statistics_observation"]
    assert observation["checkpoint_reused_draws"] == 7
    assert observation["draws_recomputed"] == 0
    assert observation["plan_reused"] is True
    for shard in observation["shards"]:
        assert shard["checkpoint_hit"] is True
        assert shard["prepare_count"] == 0
        assert shard["worker_pid"] is None
        assert type(shard["historical_worker_pid"]) is int
    assert {path: path.read_bytes() for path in paths} == snapshots
    assert not (cache / "draws").exists()


def test_unpublished_partial_checkpoint_is_ignored_on_resume(tmp_path, monkeypatch, expected):
    cache = tmp_path / "cache"
    key, paths = _fail_before_pair_publication(cache, monkeypatch)
    # Only this test's synthetic complete checkpoint is withdrawn. The partial
    # write must neither count as a completed range nor block a fresh range.
    withdrawn = next(path for path in paths if path.name == "2-4.json")
    withdrawn.unlink()
    partial = withdrawn.with_name(".2-4.json.test-partial.tmp")
    partial.write_text('{"schema": "partial')
    with _service(cache) as service:
        actual = service.evaluate(_request(), timeout=20)
    assert _science(actual) == _science(expected)
    assert withdrawn.is_file()
    assert partial.read_text() == '{"schema": "partial'
    assert service.cache.path_for(key).is_file()
    assert actual["statistics_observation"]["checkpoint_reused_draws"] == 5
    assert actual["statistics_observation"]["draws_recomputed"] == 2


@pytest.mark.parametrize("contents", ["{partial", "null", "[]", "{}"])
def test_corrupt_checkpoint_is_a_controlled_miss(tmp_path, monkeypatch, expected, contents):
    cache = tmp_path / "cache"
    _, paths = _fail_before_pair_publication(cache, monkeypatch)
    corrupt = next(path for path in paths if path.name == "2-4.json")
    corrupt.write_text(contents)
    with _service(cache) as service:
        actual = service.evaluate(_request(), timeout=20)
    assert _science(actual) == _science(expected)
    assert isinstance(json.loads(corrupt.read_text()), dict)
    assert actual["statistics_observation"]["checkpoint_reused_draws"] == 5
    assert actual["statistics_observation"]["draws_recomputed"] == 2


@pytest.mark.parametrize("mutation", ["checksum", "plan", "evaluation", "phase", "undefined_mask",
                                       "absolute_length", "range", "schema"])
def test_checkpoint_integrity_and_scientific_binding_reject_bad_complete_ranges(
    tmp_path, monkeypatch, expected, mutation,
):
    cache = tmp_path / "cache"
    _, paths = _fail_before_pair_publication(cache, monkeypatch)
    corrupt = next(path for path in paths if path.name == "2-4.json")
    record = json.loads(corrupt.read_text())
    if mutation == "checksum":
        record["sha256"] = "0" * 64
    else:
        if mutation in {"plan", "evaluation", "schema", "phase"}:
            record["contract"][mutation] = "wrong-contract"
        elif mutation == "undefined_mask":
            record["undefined_mask"][0] = not record["undefined_mask"][0]
        elif mutation == "absolute_length":
            record["shard"]["absolute_bootstrap"]["primary"]["reference"].pop()
        elif mutation == "range":
            record["shard"]["repetition_offset"] = -1
        record["sha256"] = json_fingerprint({key: value for key, value in record.items() if key != "sha256"})
    corrupt.write_text(json.dumps(record))
    with _service(cache) as service:
        actual = service.evaluate(_request(), timeout=20)
    assert _science(actual) == _science(expected)
    assert actual["statistics_observation"]["checkpoint_reused_draws"] == 5
    assert actual["statistics_observation"]["draws_recomputed"] == 2


def test_checkpoint_contains_absolute_delta_ratio_and_undefined_components_without_capture(tmp_path, monkeypatch):
    _, paths = _fail_before_pair_publication(tmp_path / "cache", monkeypatch)
    for path in paths:
        record = json.loads(path.read_text())
        body = {key: value for key, value in record.items() if key != "sha256"}
        assert record["sha256"] == json_fingerprint(body)
        shard = record["shard"]
        assert record["contract"]["schema"] == "paired-statistics-block-v1"
        assert record["contract"]["phase"] == "candidate"
        assert record["contract"]["B"] == 7
        assert record["contract"]["n"] == 4
        assert shard["block_identity"] == record["contract"]
        count = shard["repetitions"]
        assert len(shard["bootstrap"]["primary"]) == count
        assert len(shard["absolute_bootstrap"]["primary"]["reference"]) == count
        assert len(shard["absolute_bootstrap"]["primary"]["candidate"]) == count
        assert len(shard["relative_loss_draws"]) == count
        assert record["undefined_mask"] == [value is None for value in shard["relative_loss_draws"]]


def test_changed_block_size_preserves_prior_draw_ranges(tmp_path, monkeypatch, expected):
    cache = tmp_path / "cache"
    _, paths = _fail_before_pair_publication(cache, monkeypatch)
    preserved = next(path for path in paths if path.name == "0-2.json")
    original = preserved.read_bytes()
    for path in paths:
        if path != preserved:
            path.unlink()
    with _service(cache, block_repetitions=3) as service:
        actual = service.evaluate(_request(), timeout=20)
    assert _science(actual) == _science(expected)
    assert preserved.read_bytes() == original
    assert actual["statistics_observation"]["checkpoint_reused_draws"] == 2
    assert actual["statistics_observation"]["draws_recomputed"] == 5
    assert {path.name for path in _checkpoint_files(cache)} == {"0-2.json", "2-5.json", "5-7.json"}


def test_plan_write_failure_resolves_future_without_a_partial_result(tmp_path, monkeypatch):
    cache = tmp_path / "cache"

    def disk_full(*args, **kwargs):
        raise OSError("test plan scratch full")

    with monkeypatch.context() as context:
        context.setattr(np, "save", disk_full)
        with _service(cache) as service:
            with pytest.raises(OSError, match="plan scratch full"):
                service.evaluate(_request(), timeout=10)
    assert service.shutdown_state()["finished"] is True
    assert not _checkpoint_files(cache)
    assert not list(cache.glob("[0-9a-f][0-9a-f]/*.json"))


def test_published_plan_is_exact_pcg64_int64_readonly_mmap(tmp_path):
    cache = tmp_path / "cache"
    with _service(cache) as service:
        service.evaluate(_request(), timeout=20)
    files = list((cache / "plans").glob("*.npy"))
    assert len(files) == 1
    plan = np.load(files[0], mmap_mode="r", allow_pickle=False)
    expected = quality.deterministic_resample_plan(image_count=4, repetitions=7, seed=731)
    assert isinstance(plan, np.memmap)
    assert plan.dtype == np.dtype("int64")
    assert plan.shape == (7, 4)
    assert not plan.flags.writeable
    np.testing.assert_array_equal(plan, expected)


def test_corrupt_plan_is_recreated_identically_and_keeps_valid_completed_ranges(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _, checkpoints = _fail_before_pair_publication(cache, monkeypatch)
    snapshots = {path: path.read_bytes() for path in checkpoints}
    path, = (cache / "plans").glob("*.npy")
    original = path.read_bytes()
    path.write_bytes(b"invalid numpy plan")
    with _service(cache) as service:
        actual = service.evaluate(_request(), timeout=20)
    assert path.read_bytes() == original
    assert actual["statistics_observation"]["plan_reused"] is False
    assert actual["statistics_observation"]["checkpoint_reused_draws"] == 7
    assert actual["statistics_observation"]["draws_recomputed"] == 0
    assert {path: path.read_bytes() for path in checkpoints} == snapshots


def test_paused_queue_holds_only_descriptors_and_does_not_materialize_waiting_payloads(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import quality_statistics_blocks as blocks
    prepared = []
    original_load = blocks.load_payload

    def observed_load(descriptor):
        prepared.append(descriptor["key"])
        return original_load(descriptor)

    monkeypatch.setattr(blocks, "load_payload", observed_load)
    service = _service(tmp_path / "cache")
    descriptors = []
    original_put = service._queue.put

    def observed_put(item, *args, **kwargs):
        if item is not None:
            descriptors.append(item.payload)
        return original_put(item, *args, **kwargs)

    monkeypatch.setattr(service._queue, "put", observed_put)
    service.pause("test-owned-resource-window")
    clients = []
    try:
        for index in range(12):
            request = _request()
            for row in request.reference_records + request.candidate_records:
                row["synthetic_extra_payload"] = "x" * 16384
            clients.append(service.submit(replace(request, seed=731 + index)))
        assert len(descriptors) == 12
        assert prepared == []
        assert service._groups == {}
        for descriptor in descriptors:
            assert "reference_records" not in descriptor
            assert "candidate_records" not in descriptor
            assert "annotations" not in descriptor
            assert len(json.dumps(descriptor)) < 4096
            assert descriptor["size_bytes"] > 128 * 1024
            assert Path(descriptor["path"]).is_file()
        assert not (getattr(service._executor, "_processes", None) or {})
        service.shutdown(cancel_futures=True, terminate_workers=True,
                         cancellation_context=cancel_context("synthetic-queued-run"))
        assert all(client.cancelled() for client in clients)
        assert prepared == []
    finally:
        service.shutdown(cancel_futures=True, terminate_workers=True)


def _blocking_request(tmp_path, monkeypatch):
    module = tmp_path / "ap04_checkpoint_test_worker.py"
    gate = tmp_path / "block-enabled"
    entered = tmp_path / "blocked-worker.json"
    gate.write_text("enabled")
    module.write_text('''import json, os, time
from pathlib import Path
from onnx_splitpoint_tool.quality_service import _PairedMeanEvaluator
class Evaluator(_PairedMeanEvaluator):
    def __init__(self, reference, candidate, annotations, config):
        super().__init__(reference, candidate, annotations, config)
        self.calls = 0
        self.gate = Path(annotations["gate"])
        self.entered = Path(annotations["entered"])
    def evaluate(self, multiplicities):
        self.calls += 1
        if self.calls >= 4 and self.gate.exists():
            self.entered.write_text(json.dumps({"pid": os.getpid()}))
            deadline = time.monotonic() + 30
            while self.gate.exists() and time.monotonic() < deadline:
                time.sleep(.01)
        return super().evaluate(multiplicities)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    request = replace(_request(), evaluator_factory="ap04_checkpoint_test_worker:Evaluator",
                      annotations={"gate": str(gate), "entered": str(entered)})
    return request, gate, entered


@pytest.mark.parametrize("ending", ["cancel", "worker_crash"])
def test_cancel_or_worker_crash_preserves_only_complete_ranges_for_resume(tmp_path, monkeypatch, ending):
    cache = tmp_path / "cache"
    request, gate, entered = _blocking_request(tmp_path, monkeypatch)
    service = _service(cache)
    try:
        client = service.submit(request)
        _wait_for(entered.is_file)
        _wait_for(lambda: bool(_checkpoint_files(cache)))
        paths = _checkpoint_files(cache)
        assert [path.name for path in paths] == ["0-2.json"]
        preserved = paths[0].read_bytes()
        key = paths[0].parent.name
        assert not service.cache.path_for(key).exists()
        owned = list(service._executor._processes.values())
        active_pid = json.loads(entered.read_text())["pid"]
        assert active_pid in [process.pid for process in owned]
        if ending == "cancel":
            service.shutdown(cancel_futures=True, terminate_workers=True,
                             cancellation_context=cancel_context("synthetic-checkpoint-run"))
            with pytest.raises(CancelledError):
                client.result(timeout=5)
        else:
            # Crash only the worker created by this controlled synthetic test.
            process = next(process for process in owned if process.pid == active_pid)
            process.terminate()
            process.join(timeout=5)
            from concurrent.futures.process import BrokenProcessPool
            with pytest.raises(BrokenProcessPool):
                client.result(timeout=10)
            service.shutdown()
        assert all(not process.is_alive() for process in owned)
        assert not service.cache.path_for(key).exists()
        assert paths[0].read_bytes() == preserved
    finally:
        gate.unlink(missing_ok=True)
        service.shutdown(cancel_futures=True, terminate_workers=True)

    with _service(cache) as resumed:
        actual = resumed.evaluate(request, timeout=20)
    assert actual["primary"]["bootstrap_repetitions"] == 7
    assert actual["primary"]["ci_computed"] is True
    assert paths[0].read_bytes() == preserved
    assert len(_checkpoint_files(cache)) == 4
    assert resumed.cache.path_for(key).is_file()
    assert actual["statistics_observation"]["checkpoint_reused_draws"] == 2
    assert actual["statistics_observation"]["draws_recomputed"] == 5
