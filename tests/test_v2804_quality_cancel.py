"""Run-scoped cancellation across real futures/processes and report readers."""
from concurrent.futures import CancelledError, Future
from dataclasses import replace
import json
import threading
import time

import pytest

from onnx_splitpoint_tool import quality_service as quality
from onnx_splitpoint_tool.quality_lifecycle import (
    cancel_context, completed_outcome, exception_outcome,
    is_bootstrap_eta_sample, progress_text, stamp_exception, summarize_requests,
    replay_historical_cancellation,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import project_central_quality_status


def request():
    return quality.QualityEvaluationRequest(
        reference_records=[{"image_id": i, "value": 1.0} for i in range(4)],
        candidate_records=[{"image_id": i, "value": .9} for i in range(4)],
        annotations=[], metric_gate_config={"primary_metric": "top1_accuracy"},
        repetitions=8, seed=42, confidence_level=.95,
        non_inferiority_margin=.25, evaluator_factory="paired_mean", value_field="value",
    )


def closed_context(run_id="one"):
    context = cancel_context(run_id)
    context["shutdown_monotonic"] = time.monotonic()
    return context


@pytest.mark.parametrize("kind", [CancelledError, quality.QualityServiceClosedError])
def test_closed_requires_matching_time_bound_run_context(kind):
    context = closed_context()
    exc = stamp_exception(kind("stopped"), context)
    assert exception_outcome(exc, run_id="one")["status"] == "cancelled"
    assert exception_outcome(exc, run_id="other")["status"] == "failed"
    assert exception_outcome(kind("no context"), run_id="one")["status"] == "failed"
    early = stamp_exception(kind("observed before cancellation"))
    stamp_exception(early, context)
    assert exception_outcome(early, run_id="one")["status"] == "failed"
    impossible = dict(context, shutdown_monotonic=time.monotonic() + 60)
    assert exception_outcome(stamp_exception(kind(), impossible), run_id="one")["status"] == "failed"


def test_genuine_error_never_becomes_cancel_by_later_context():
    error = stamp_exception(ValueError("candidate SHA mismatch"), closed_context())
    out = exception_outcome(error, run_id="one")
    assert out["completion_reason"] == "technical_error"
    assert out["error"] == "ValueError: candidate SHA mismatch"


class PendingExecutor:
    def __init__(self, count=4):
        self.futures = []
        self.ready = threading.Event()
        self.count = count

    def submit(self, *args, **kwargs):
        future = Future()
        self.futures.append(future)
        if len(self.futures) == self.count:
            self.ready.set()
        return future

    def shutdown(self, **kwargs):
        if kwargs.get("cancel_futures"):
            for future in self.futures:
                future.cancel()


def test_cancel_before_dispatch_and_deduplicated_waiters(monkeypatch, tmp_path):
    pool = PendingExecutor()
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kw: pool)
    service = quality.ManagementQualityService(tmp_path / "cache", workers=4)
    service.pause("urecs-owned-window")
    clients = [service.submit(request()) for _ in range(3)]
    callbacks = []
    for client in clients:
        client.add_done_callback(lambda f: callbacks.append(f))
    context = cancel_context("one")
    service.shutdown(cancel_futures=True, terminate_workers=True, cancellation_context=context)
    service.shutdown(cancel_futures=True, cancellation_context=context)
    assert pool.futures == []
    assert not service.pause_gate.paused
    assert len(callbacks) == len(set(callbacks)) == 3
    for client in clients:
        with pytest.raises(CancelledError):
            client.result()
        assert client.quality_cancel_context["run_id"] == "one"
    with pytest.raises(quality.QualityServiceClosedError) as caught:
        service.evaluate(request())
    assert exception_outcome(caught.value, run_id="one")["completion_reason"] == "service_closed_after_cancel"
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_prior_shard_failure_survives_cancel_and_duplicate_callback(monkeypatch, tmp_path):
    pool = PendingExecutor()
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kw: pool)
    service = quality.ManagementQualityService(tmp_path / "cache", workers=4)
    client = service.submit(request())
    assert pool.ready.wait(5)
    group = next(iter(service._groups.values()))
    earlier = ValueError("worker model failure before cancel")
    pool.futures[0].set_exception(earlier)
    service._shard_worker_done(group, pool.futures[0])  # duplicate callback is inert
    service.shutdown(cancel_futures=True, cancellation_context=cancel_context("one"))
    with pytest.raises(ValueError) as caught:
        client.result()
    assert caught.value is earlier
    assert exception_outcome(caught.value, run_id="one")["status"] == "failed"
    assert not list((tmp_path / "cache").rglob("*.json"))
    assert not service._inflight
    assert all(service._worker_slots.acquire(blocking=False) for _ in range(4))


def test_error_observation_and_storage_cannot_be_split_by_cancel(monkeypatch, tmp_path):
    pool = PendingExecutor()
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kw: pool)
    service = quality.ManagementQualityService(tmp_path / "cache", workers=4)
    client = service.submit(request())
    assert pool.ready.wait(5)
    observed, release = threading.Event(), threading.Event()
    original_stamp = quality.stamp_exception
    error = ValueError("observed before cancel acquired group lock")
    def held_stamp(exc, context=None):
        value = original_stamp(exc, context)
        if exc is error and not observed.is_set():
            observed.set()
            assert release.wait(5)
        return value
    monkeypatch.setattr(quality, "stamp_exception", held_stamp)
    publisher = threading.Thread(target=pool.futures[0].set_exception, args=(error,))
    publisher.start()
    assert observed.wait(5)
    stopper = threading.Thread(target=service.shutdown, kwargs={
        "cancel_futures": True, "cancellation_context": cancel_context("one"),
    })
    stopper.start()
    release.set()
    publisher.join(5)
    stopper.join(5)
    assert not publisher.is_alive() and not stopper.is_alive()
    with pytest.raises(ValueError) as caught:
        client.result()
    assert caught.value is error
    assert not client.cancelled()


class InlineExecutor:
    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    def shutdown(self, **kwargs):
        pass


def test_completed_publication_wins_competing_cancel_once(monkeypatch, tmp_path):
    monkeypatch.setattr(quality, "ProcessPoolExecutor", lambda **kw: InlineExecutor())
    service = quality.ManagementQualityService(tmp_path / "cache", workers=4)
    entered, release = threading.Event(), threading.Event()
    original_put = service.cache.put
    def held_put(key, value):
        entered.set()
        assert release.wait(5)
        return original_put(key, value)
    monkeypatch.setattr(service.cache, "put", held_put)
    client = service.submit(request())
    assert entered.wait(5)
    stopper = threading.Thread(target=service.shutdown, kwargs={
        "cancel_futures": True, "cancellation_context": cancel_context("one"),
    })
    stopper.start()
    release.set()
    result = client.result(timeout=5)
    stopper.join(5)
    assert not stopper.is_alive()
    assert result["status"] == "completed"
    assert service.evaluate(request())["cache_hit"] is True
    assert len(list((tmp_path / "cache").rglob("*.json"))) == 1


def test_actual_spawned_shard_cancel_releases_pause_and_joins_only_own_processes(monkeypatch, tmp_path):
    # The custom evaluator runs inside the real spawn pool and blocks at its
    # first metric call. The marker proves the actual process boundary.
    module = tmp_path / "v2804_cancel_worker.py"
    marker = tmp_path / "entered.json"
    module.write_text('''import json, os, time
from pathlib import Path
class Evaluator:
    def __init__(self, reference, candidate, annotations, config):
        self.marker = annotations["marker"]
    def evaluate(self, multiplicities):
        Path(self.marker).write_text(json.dumps({"pid": os.getpid()}))
        time.sleep(30)
        return {"candidate": .9, "reference": 1., "delta": -.1}
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    service = quality.ManagementQualityService(tmp_path / "cache", workers=1)
    req = replace(request(), evaluator_factory="v2804_cancel_worker:Evaluator", annotations={"marker": str(marker)})
    try:
        client = service.submit(req)
        deadline = time.monotonic() + 10
        while not marker.is_file() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.is_file(), "real quality worker did not reach evaluator"
        owned = list(service._executor._processes.values())
        assert json.loads(marker.read_text())["pid"] in [p.pid for p in owned]
        service.pause("urecs-owned-window")
        queued = service.submit(replace(req, seed=43))
        service.shutdown(cancel_futures=True, terminate_workers=True, cancellation_context=cancel_context("one"))
        assert all(not p.is_alive() for p in owned)
        assert not service._dispatcher.is_alive()
        assert not service.pause_gate.paused
        assert client.cancelled() and queued.cancelled()
        assert not list((tmp_path / "cache").rglob("*.json"))
    finally:
        service.shutdown(cancel_futures=True, terminate_workers=True)


def test_counts_and_scientific_projection_keep_negative_observations():
    # Derived metadata fixture: the historical 43/18/2+60 population. Original
    # Q5 bytes are tested independently by the evidence replay/export gate.
    rows = [
        {"status": "completed", "technical_status": "completed", "decision": decision,
         "variant": "full", "source_request": f"{decision}/{n}"}
        for decision, count in (("pass", 43), ("fail", 18), ("inconclusive", 2))
        for n in range(count)
    ]
    rows += [exception_outcome(stamp_exception(CancelledError(), closed_context()), run_id="one") for _ in range(4)]
    rows += [exception_outcome(stamp_exception(quality.QualityServiceClosedError("closed"), closed_context()), run_id="one") for _ in range(56)]
    counts = summarize_requests(rows)
    assert counts["request_count"] == counts["terminal_count"] == 123
    assert counts["completed_count"] == counts["evaluated_count"] == 63
    assert counts["cancelled_count"] == 60 and counts["technical_failed_count"] == 0
    assert counts["quality_decision_counts"] == {"pass": 43, "fail": 18, "inconclusive": 2}
    report = project_central_quality_status({"status": "cancelled", "results": rows, **counts})
    for name in ("terminal_count", "evaluated_count", "cancelled_count", "technical_failed_count", "quality_decision_counts"):
        assert report[name] == counts[name]
    assert report["technical_status"] == "cancelled"
    assert report["quality_decision"] == "fail"
    assert report["scientific_pass"] is False
    assert "63/123 evaluated; 60 cancelled; 0 technical errors" in progress_text(counts)


def test_normal_stage_json_csv_and_log_share_counts(tmp_path, monkeypatch):
    import csv
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
    logs = []
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)), log=logs.append)
    runner.run_dir = tmp_path / "run"
    runner.run_dir.mkdir()
    runner.run_id = "one"
    runner.run_log_path = runner.run_dir / "evaluation_workflow.log"
    runner.parent_log_path = tmp_path / "latest.log"
    runner.profile_payload = {"execution_preset": {"id": "standard"}, "models": [],
                              "quality_gate": {"statistics": {"execution_location": "central_management", "workers": 1}}}
    monkeypatch.setattr(runner, "_merge_central_quality_results", lambda rows: {})
    monkeypatch.setattr(runner, "_refresh_validation_after_central_quality", lambda models: [])
    values = [
        {"status": "completed", "technical_status": "completed", "decision": "fail", "variant": "full"},
        exception_outcome(stamp_exception(CancelledError(), closed_context()), run_id="one"),
        exception_outcome(stamp_exception(quality.QualityServiceClosedError("closed"), closed_context()), run_id="one"),
        exception_outcome(ValueError("original error"), run_id="one"),
    ]
    for i, value in enumerate(values):
        future = Future()
        future.set_result(value)
        runner._central_quality_futures[str(i)] = future
    artifacts, metrics, message, status = runner._stage_evaluate_quality()
    summary = json.loads(artifacts["central_quality_summary_json"].read_text())
    assert status == "cancelled"
    assert summary["request_count"] == summary["terminal_count"] == 4
    assert summary["completed_count"] == summary["evaluated_count"] == 1
    assert summary["cancelled_count"] == 2
    assert summary["failed_count"] == summary["technical_failed_count"] == 1
    assert summary["quality_decision"] == "fail"
    assert metrics["quality_decision_counts"] == {"pass": 0, "fail": 1, "inconclusive": 0}
    with artifacts["central_quality_summary_csv"].open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["technical_status"] for row in rows].count("cancelled") == 2
    assert "1/4 evaluated; 2 cancelled; 1 technical errors" in message
    assert "4/4 completed" not in "\n".join(logs)


def test_cancel_context_write_failure_still_stops_owned_service(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, WorkflowOptions
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    stopped = []
    runner._central_quality_service = SimpleNamespace(shutdown=lambda **kwargs: stopped.append(kwargs))
    monkeypatch.setattr(runner, "_write_workflow_control", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        runner.request_cancel()
    assert stopped[0]["cancel_futures"] and stopped[0]["terminate_workers"]
    assert stopped[0]["cancellation_context"]["persistence_error"] == "OSError: disk full"


def test_historical_replay_requires_exact_event_binding_and_preserves_original(tmp_path):
    import copy
    import subprocess
    import sys
    from pathlib import Path
    rows = [{"status": "completed", "technical_status": "completed", "decision": "fail", "variant": "full"}]
    for index in range(3):
        rows.append({"status": "failed", "technical_status": "failed", "decision": "unavailable",
                     "source_request": str(index), "source_request_sha256": str(index) * 64,
                     "error": "QualityServiceClosedError: closed"})
    summary = {"status": "cancelled", "run_id": "one", "request_count": 4, "failed_count": 3, "results": rows}
    evidence = {"run_id": "one", "source_evidence": ["synthetic controlled event log; not original Q5"],
                "cancel_requested_at": "2026-09-12T10:00:00Z", "service_shutdown_at": "2026-09-12T10:00:01Z",
                "shutdown_completed_at": "2026-09-12T10:00:03Z", "events": {}}
    for index in range(3):
        evidence["events"][str(index)] = {"source_request_sha256": str(index) * 64,
            "error": rows[index + 1]["error"], "observed_at": "2026-09-12T10:00:02Z"}
    evidence["events"]["1"]["observed_at"] = "2026-09-12T09:59:59Z"  # earlier real failure
    evidence["events"]["2"]["source_request_sha256"] = "wrong"
    original = copy.deepcopy(summary)
    replay = replay_historical_cancellation(summary, evidence)
    assert summary == original
    assert replay["cancelled_count"] == 1 and replay["technical_failed_count"] == replay["failed_count"] == 2
    assert replay["results"][1]["original_status"] == "failed"
    assert replay["results"][0] == rows[0]
    with pytest.raises(ValueError, match="run_binding"):
        replay_historical_cancellation(summary, dict(evidence, run_id="other"))
    source, audit = tmp_path / "summary.json", tmp_path / "audit.json"
    source.write_text(json.dumps(summary)); audit.write_text(json.dumps(evidence))
    before = source.read_bytes()
    output = tmp_path / "review"
    script = Path(__file__).resolve().parents[1] / "scripts/replay_quality_cancel_v2804.py"
    result = subprocess.run([sys.executable, "-I", "-B", str(script), "--summary", str(source),
                             "--evidence", str(audit), "--output-dir", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == before
    assert json.loads((output / "central_quality_cancel_replay.json").read_text())["cancelled_count"] == 1


@pytest.mark.parametrize("change", [
    {"status": "cancelled"}, {"status": "failed"}, {"cache_hit": True},
    {"primary": {"bootstrap_repetitions": 0, "bootstrap_skipped_reason": "candidate_reference_identical"}},
    {"primary": {"bootstrap_repetitions": 0, "bootstrap_skipped_reason": "point_estimate_below_non_inferiority_margin"}},
])
def test_eta_excludes_cancel_errors_cache_and_fast_paths(change):
    result = {"status": "completed", "decision": "pass", "primary": {"bootstrap_repetitions": 5000, "ci_computed": True}}
    assert is_bootstrap_eta_sample(result)
    assert not is_bootstrap_eta_sample(dict(result, **change))
    assert completed_outcome(dict(result, cache_hit=True))["completion_reason"] == "reused_result"
