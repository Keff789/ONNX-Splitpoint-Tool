"""Small CPU-only regressions for truthful statistics progress and cache reuse."""
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool import quality_service, quality_statistics
from onnx_splitpoint_tool.quality_service import ManagementQualityService, QualityEvaluationRequest
from onnx_splitpoint_tool.quality_statistics import WorkerProgress


def _progress(tmp_path):
    return WorkerProgress({
        "request_id": "actually-running",
        "evaluation_fingerprint": "actual-key",
        "_statistics": {"progress_dir": str(tmp_path)},
    }, 128, 256)


def _clock(monkeypatch):
    clock = SimpleNamespace(monotonic=100.0, wall=1000.0)
    monkeypatch.setattr(quality_statistics.time, "perf_counter", lambda: clock.monotonic)
    monkeypatch.setattr(quality_statistics.time, "time", lambda: clock.wall)
    return clock


def test_phase_changes_are_immediate_but_same_phase_writes_are_throttled(tmp_path, monkeypatch):
    clock = _clock(monkeypatch)
    writes = []
    monkeypatch.setattr(quality_service, "_atomic_write_json",
                        lambda path, row: writes.append((Path(path), deepcopy(row))))
    progress = _progress(tmp_path)
    progress.emit("preparing")
    for elapsed in (0.1, 1.0, 4.999):
        clock.monotonic = 100.0 + elapsed
        clock.wall = 1000.0 + elapsed
        progress.emit("preparing")
    assert len(writes) == 1
    clock.monotonic, clock.wall = 105.0, 1005.0
    progress.emit("preparing")
    assert len(writes) == 2
    progress.emit("matching")
    assert len(writes) == 3
    progress.emit("point", cache_state="cold")
    assert len(writes) == 4
    assert [row["phase"] for _, row in writes] == ["preparing", "preparing", "matching", "point"]
    for path, row in writes:
        assert path == tmp_path / f"{os.getpid()}.json"
        assert row["worker_pid"] == os.getpid()
        assert row["request_id"] == "actually-running"
        assert row["evaluation_fingerprint"] == "actual-key"
        assert (row["repetition_start"], row["repetition_stop"]) == (128, 256)
        assert row["draws_requested"] == 128
        assert row["draws_completed"] == 0
        assert row["last_progress_at"] is None


def test_heartbeat_advances_time_without_inventing_draw_progress(tmp_path, monkeypatch):
    clock = _clock(monkeypatch)
    progress = _progress(tmp_path)
    progress.emit("accumulating", completed=3, cache_state="reference_hit")
    before = json.loads((tmp_path / f"{os.getpid()}.json").read_text())
    waits = []

    class OneHeartbeat:
        def wait(self, duration):
            waits.append(duration)
            clock.monotonic += duration
            clock.wall += duration
            return len(waits) > 1

        def set(self):
            pass

    class InlineThread:
        def __init__(self, *, target, daemon):
            assert daemon is True
            self.target = target
            self.joined = False

        def start(self):
            self.target()

        def join(self):
            self.joined = True

    # Run one actual heartbeat callback deterministically, without a 5 s sleep.
    progress._stop = OneHeartbeat()
    monkeypatch.setattr(threading, "Thread", InlineThread)
    with progress:
        after = json.loads((tmp_path / f"{os.getpid()}.json").read_text())
    assert waits == [5, 5]
    assert progress._thread.joined
    assert after["last_heartbeat_at"] == before["last_heartbeat_at"] + 5
    assert after["elapsed_s"] == before["elapsed_s"] + 5
    assert after["last_progress_at"] == before["last_progress_at"]
    assert after["draws_completed"] == before["draws_completed"] == 3
    assert after["phase"] == "accumulating"
    assert after["cache_state"] == "reference_hit"


def test_exception_publishes_failed_terminal_and_preserves_actual_draw_count(tmp_path):
    progress = _progress(tmp_path)
    with pytest.raises(ValueError, match="failed draw"):
        with progress:
            progress.emit("accumulating", completed=7)
            raise ValueError("failed draw")
    result = json.loads((tmp_path / f"{os.getpid()}.json").read_text())
    assert result["phase"] == "failed"
    assert result["error_type"] == "ValueError"
    assert result["draws_completed"] == 7
    assert result["draws_requested"] == 128
    assert result["last_progress_at"] is not None
    assert progress._stop.is_set()
    assert not progress._thread.is_alive()


def _snapshot_service(tmp_path, *, alive=True):
    service = ManagementQualityService.__new__(ManagementQualityService)
    service.cache = SimpleNamespace(root=tmp_path)
    service._lock = threading.RLock()
    service._started_at = 1000.0
    service._inflight = {"first-pending": [], "second-pending": [], "actual-key": []}
    service._executor = SimpleNamespace(_processes={
        413: SimpleNamespace(pid=413, is_alive=lambda: alive),
        219: SimpleNamespace(pid=219, is_alive=lambda: True),
    })
    (tmp_path / "progress").mkdir()
    return service


def _row(**changes):
    row = {"request_id": "actually-running", "evaluation_fingerprint": "actual-key",
           "worker_pid": 413, "phase": "accumulating", "draws_completed": 7,
           "draws_requested": 128, "repetition_start": 128, "repetition_stop": 256,
           "last_heartbeat_at": 1001.0, "last_progress_at": 1000.5}
    row.update(changes)
    return row


def _write_row(tmp_path, row, filename="413.json"):
    (tmp_path / "progress" / filename).write_text(json.dumps(row))


def test_snapshot_reports_owned_live_workers_instead_of_first_pending_entries(tmp_path):
    service = _snapshot_service(tmp_path)
    first = _row()
    second = _row(worker_pid=219, phase="matching", draws_completed=0, last_progress_at=None)
    _write_row(tmp_path, first)
    _write_row(tmp_path, second, "219.json")
    assert service.progress_snapshot() == [second, first]
    assert all(row["request_id"] == "actually-running" for row in service.progress_snapshot())
    # Pending submissions alone are not evidence that a worker is executing.
    (tmp_path / "progress" / "413.json").unlink()
    (tmp_path / "progress" / "219.json").unlink()
    assert service.progress_snapshot() == []


@pytest.mark.parametrize("changes,filename,alive", [
    ({"worker_pid": 999}, "999.json", True),
    ({}, "413.json", False),
    ({}, "another-worker.json", True),
    ({"evaluation_fingerprint": "another-service-key"}, "413.json", True),
    ({"evaluation_fingerprint": ["actual-key"]}, "413.json", True),
    ({"evaluation_fingerprint": {"key": "actual-key"}}, "413.json", True),
    ({"phase": "completed"}, "413.json", True),
    ({"phase": "failed"}, "413.json", True),
    ({"phase": ["accumulating"]}, "413.json", True),
    ({"last_heartbeat_at": 999.0}, "413.json", True),
    ({"last_heartbeat_at": "1001"}, "413.json", True),
    ({"last_heartbeat_at": None}, "413.json", True),
    ({"last_heartbeat_at": float("nan")}, "413.json", True),
    ({"last_heartbeat_at": float("inf")}, "413.json", True),
    ({"worker_pid": "413"}, "413.json", True),
    ({"worker_pid": True}, "True.json", True),
])
def test_snapshot_ignores_stale_foreign_terminal_or_malformed_rows(tmp_path, changes, filename, alive):
    service = _snapshot_service(tmp_path, alive=alive)
    _write_row(tmp_path, _row(**changes), filename)
    assert service.progress_snapshot() == []


@pytest.mark.parametrize("content", ["{partial", "null", "[]", "123", '"text"', "{}"])
def test_snapshot_survives_partial_write_and_nonrecord_json(tmp_path, content):
    service = _snapshot_service(tmp_path)
    (tmp_path / "progress" / "413.json").write_text(content)
    _write_row(tmp_path, _row(worker_pid=219), "219.json")
    assert service.progress_snapshot() == [_row(worker_pid=219)]


def test_snapshot_ignores_progress_after_own_request_is_no_longer_inflight(tmp_path):
    service = _snapshot_service(tmp_path)
    _write_row(tmp_path, _row())
    del service._inflight["actual-key"]
    assert service.progress_snapshot() == []


def test_snapshot_after_executor_shutdown_has_no_active_workers(tmp_path):
    service = _snapshot_service(tmp_path)
    _write_row(tmp_path, _row())
    service._executor._processes = None
    assert service.progress_snapshot() == []


@pytest.mark.parametrize("engine", ["legacy", "optimized_coco_v1"])
def test_complete_pair_cache_hit_reports_zero_draw_recomputation(tmp_path, monkeypatch, engine):
    request = QualityEvaluationRequest(
        reference_records=[{"image_id": i, "value": float(i % 2)} for i in range(6)],
        candidate_records=[{"image_id": i, "value": float(i % 2) - 0.01} for i in range(6)],
        annotations=[], metric_gate_config={"primary_metric": "mean"},
        repetitions=17, seed=20260710, confidence_level=0.95,
        non_inferiority_margin=0.1, evaluator_factory="paired_mean",
        request_id="first-execution")
    cache = tmp_path / "cache"
    with ManagementQualityService(cache, workers=1, statistics={"engine": engine}) as service:
        original = service.evaluate(request, timeout=30)
    assert original["cache_hit"] is False
    assert original["primary"]["bootstrap_repetitions_requested"] == 17

    def unexpected_work(*args, **kwargs):
        pytest.fail("complete pair-cache hit submitted fresh worker work")

    with ManagementQualityService(cache, workers=1, statistics={"engine": engine}) as service:
        monkeypatch.setattr(service._executor, "submit", unexpected_work)
        reused = service.evaluate(request, timeout=5)
        assert service.progress_snapshot() == []
    assert reused["cache_hit"] is True
    assert reused["primary"] == original["primary"]
    assert reused["statistics_observation"]["cache_state"] == "complete_result_reused"
    assert reused["statistics_observation"]["draws_recomputed"] == 0
    assert reused["statistics_observation"]["engine_requested"] == engine
    if engine == "optimized_coco_v1":
        assert original["statistics_observation"]["draws_recomputed"] == 17
    else:
        assert "draws_recomputed" not in original["statistics_observation"]
