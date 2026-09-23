"""Real campaign-budget transactions between owned local processes.

The external leaf is a controlled preflight without a collector or workload.
Budget decisions, persistence, source ownership and re-entry remain production
code; these tests make no energy or hardware claim.
"""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy.task_budget import EnergyTaskBudget, bounded_energy_task


def _task_process(checkpoint, output, row, source, sender, release, *,
                  logical="repeat:0", repeats=2, retries=0, failures=2,
                  lose_owner=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    @bounded_energy_task
    def controlled_preflight(_command, out_dir, **kwargs):
        budget = kwargs["_task_budget"]
        attempt = Path(out_dir) / "attempt"
        attempt.mkdir()
        chain = budget.reserve(logical, attempt, True)
        if chain is None:
            return {"ok": False, "status": "not_dispatched", "runs": []}
        sender.send(("reserved", {"pid": os.getpid(), "row": row, "source": source}))
        if lose_owner:
            os._exit(23)  # Only this test-owned process exits, leaving its actual reservation.
        if release is not None and not release.wait(10):
            raise AssertionError("controlled preflight release was not delivered")
        entry = {"ok": False, "status": "preflight_failed", "collector_started": False,
                 "diagnostic_only": True, "error": "controlled external preflight stopped before collector"}
        budget.finish(chain, entry, attempt)
        return {"ok": False, "status": "preflight_failed", "runs": [entry]}

    try:
        result = controlled_preflight(
            "controlled-local-preflight", output,
            setup=SimpleNamespace(urecs_address=source),
            campaign_budget_file=checkpoint, campaign_row_id=row,
            campaign_repeats=repeats, campaign_max_retries=retries,
            campaign_max_transport_failures=failures,
            invalid_repeat_max_retries=0,
        )
        sender.send(("done", result))
    except BaseException as exc:
        sender.send(("error", {"type": type(exc).__name__, "message": str(exc)}))
    finally:
        sender.close()


@pytest.fixture
def processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    owned = []
    events = []
    checkpoint = tmp_path / "energy_task_budget.json"

    def start(row, source, *, label=None, hold=False, **kwargs):
        receiver, sender = context.Pipe(duplex=False)
        release = context.Event() if hold else None
        if release is not None:
            events.append(release)
        process = context.Process(
            target=_task_process,
            args=(str(checkpoint), str(tmp_path / (label or row)), row, source, sender, release),
            kwargs=kwargs,
        )
        process.start()
        sender.close()
        owned.append((process, receiver))
        return process, receiver, release

    yield start, checkpoint
    for event in events:
        event.set()
    for process, receiver in owned:
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=3)
        receiver.close()
        assert not process.is_alive(), "owned test child did not exit"


def _message(connection):
    assert connection.poll(10), "owned budget process did not answer"
    return connection.recv()


def _finish(process, connection):
    kind, result = _message(connection)
    assert kind == "done", result
    process.join(timeout=3)
    assert process.exitcode == 0
    return result


def _run_completed(start, row, source, **kwargs):
    process, connection, _ = start(row, source, **kwargs)
    kind, result = _message(connection)
    assert kind == "reserved", result
    return _finish(process, connection)


def _checkpoint(path):
    return json.loads(path.read_text())


def test_independent_sources_reserve_and_finish_concurrently_in_one_checkpoint(processes):
    start, path = processes
    first, first_pipe, release_first = start("dut_a", "192.0.2.10:5000", hold=True)
    assert _message(first_pipe)[0] == "reserved"
    second, second_pipe, release_second = start("dut_b", "192.0.2.11:5000", hold=True)
    kind, info = _message(second_pipe)
    assert kind == "reserved", info
    assert first.is_alive() and second.is_alive()
    active = _checkpoint(path)
    assert set(active["tasks"]) == {"dut_a", "dut_b"}
    assert all(not task["chains"][0]["finished"] for task in active["tasks"].values())
    assert all(not source["stop_reason"] for source in active["sources"].values())
    # Finish in reverse order to exercise stale in-memory handles during reload.
    release_second.set()
    assert _finish(second, second_pipe)["task_budget"]["counts"]["begun_chains"] == 1
    assert first.is_alive()
    release_first.set()
    assert _finish(first, first_pipe)["task_budget"]["counts"]["begun_chains"] == 1
    final = _checkpoint(path)
    assert set(final["tasks"]) == {"dut_a", "dut_b"}
    assert all(task["chains"][0]["finished"] for task in final["tasks"].values())
    assert all(not source["stop_reason"] for source in final["sources"].values())
    assert sum(chain["collector_started"] for task in final["tasks"].values() for chain in task["chains"]) == 0


def test_live_same_source_is_busy_without_poisoning_source_or_dispatching(processes):
    start, path = processes
    first, first_pipe, release = start("first", "physical-source", hold=True)
    assert _message(first_pipe)[0] == "reserved"
    second, second_pipe, _ = start("competing", "physical-source")
    kind, outcome = _message(second_pipe)
    assert kind in {"error", "done"}, "same source admitted concurrent external work"
    assert "busy" in json.dumps(outcome).lower(), outcome
    second.join(timeout=3)
    assert second.exitcode == 0
    busy = _checkpoint(path)
    assert busy["sources"]["physical-source"]["stop_reason"] == ""
    assert not busy["tasks"].get("competing", {}).get("chains")
    release.set()
    _finish(first, first_pipe)
    # Busy is transient: the same waiting row can enter once its owner leaves.
    _run_completed(start, "competing", "physical-source", label="competing-retry")
    assert not _checkpoint(path)["sources"]["physical-source"]["stop_reason"]


def test_distinct_rows_in_same_process_still_serialize_one_source_and_resume(tmp_path):
    path = tmp_path / "campaign.json"
    limits = {"max_chains": 2, "max_retries": 0, "max_transport_failures": 2}
    with path.open("w+") as first_handle, path.open("r+") as second_handle:
        first = EnergyTaskBudget(first_handle, limits, campaign_row="first", source_id="same-source")
        first_dir = tmp_path / "first-attempt"
        first_dir.mkdir()
        chain = first.reserve("repeat:0", first_dir, True)
        assert chain is not None
        second = EnergyTaskBudget(second_handle, limits, campaign_row="second", source_id="same-source")
        second_dir = tmp_path / "second-attempt"
        second_dir.mkdir()
        assert second.reserve("repeat:0", second_dir, True) is None
        assert second.source["stop_reason"] == ""
        assert second.data["stop_reason"] == ""
        entry = {"ok": False, "status": "preflight_failed", "collector_started": False,
                 "diagnostic_only": True, "error": "controlled preflight only"}
        first.finish(chain, entry, first_dir)
        following = second.reserve("repeat:0", second_dir, True)
        assert following is not None, "transient same-process source busy became persistent"
        second.finish(following, entry, second_dir)
    final = _checkpoint(path)
    assert all(task["chains"][0]["finished"] for task in final["tasks"].values())
    assert final["sources"]["same-source"]["stop_reason"] == ""


def test_lost_process_owner_persistently_stops_only_its_source(processes):
    start, path = processes
    lost, lost_pipe, _ = start("lost", "source-a", lose_owner=True)
    assert _message(lost_pipe)[0] == "reserved"
    lost.join(timeout=3)
    assert lost.exitcode == 23
    for suffix in ("first_reentry", "second_reentry"):
        process, connection, _ = start(suffix, "source-a")
        kind, result = _message(connection)
        assert kind == "done", result
        assert result["status"] == "BLOCKED"
        assert result["execution_status"] == "NOT_RUN"
        assert "unresolved" in result["error"]
        process.join(timeout=3)
        assert process.exitcode == 0
    _run_completed(start, "independent", "source-b")
    final = _checkpoint(path)
    assert "unresolved" in final["sources"]["source-a"]["stop_reason"]
    assert final["sources"]["source-b"]["stop_reason"] == ""
    assert len(final["tasks"]["lost"]["chains"]) == 1
    assert final["tasks"]["lost"]["chains"][0]["finished"] is False


def test_old_unfinished_chain_without_owner_remains_fail_closed(processes):
    start, path = processes
    limits = {"max_chains": 2, "max_retries": 0, "max_transport_failures": 2}
    path.write_text(json.dumps({"campaign": True, "max_transport_failures": 2,
        "sources": {"old-source": {"stop_reason": ""}}, "tasks": {"old-row": {
            "source_id": "old-source", "limits": limits, "stop_reason": "", "chains": [{
                "logical_repeat": "repeat:0", "run_directory": str(path.parent / "old-attempt"),
                "preflight_requested": True, "collector_started": False, "finished": False,
            }]}}}))
    process, connection, _ = start("new-row", "old-source")
    kind, result = _message(connection)
    assert kind == "done", result
    assert result["status"] == "BLOCKED" and result["execution_status"] == "NOT_RUN"
    assert "unresolved" in result["error"]
    process.join(timeout=3)
    assert process.exitcode == 0
    assert len(_checkpoint(path)["tasks"]["old-row"]["chains"]) == 1


@pytest.mark.parametrize("repeats,logical,reason", [
    (1, "repeat:1", "task_chain_limit"),
    (2, "repeat:0", "task_logical_retry_limit"),
])
def test_reentry_keeps_previous_chain_and_retry_limits(processes, repeats, logical, reason):
    start, path = processes
    _run_completed(start, "row", "source", label="attempt-one", repeats=repeats)
    process, connection, _ = start("row", "source", label="attempt-two", repeats=repeats, logical=logical)
    kind, result = _message(connection)
    assert kind == "done", result
    assert result["status"] == "BLOCKED" and result["error"] == reason
    assert result["task_budget"]["counts"]["begun_chains"] == 1
    process.join(timeout=3)
    assert process.exitcode == 0
    task = _checkpoint(path)["tasks"]["row"]
    assert len(task["chains"]) == 1 and task["chains"][0]["finished"] is True


@pytest.mark.parametrize("changed", [{"repeats": 3}, {"retries": 1}, {"failures": 1}])
def test_reentry_rejects_changed_persisted_limits_without_rewriting_checkpoint(processes, changed):
    start, path = processes
    _run_completed(start, "row", "source", label="original")
    before = path.read_bytes()
    process, connection, _ = start("row", "source", label="changed", **changed)
    kind, result = _message(connection)
    assert kind == "error" and result["type"] == "ValueError", result
    assert "differ" in result["message"], result
    process.join(timeout=3)
    assert process.exitcode == 0
    assert path.read_bytes() == before


def test_finished_legacy_chain_and_limits_survive_new_source_transaction(processes):
    start, path = processes
    limits = {"max_chains": 2, "max_retries": 0, "max_transport_failures": 2}
    old = {"source_id": "old-source", "limits": limits, "stop_reason": "", "chains": [{
        "logical_repeat": "repeat:0", "run_directory": str(path.parent / "old-attempt"),
        "preflight_requested": True, "collector_started": False, "finished": True,
    }]}
    path.write_text(json.dumps({"campaign": True, "max_transport_failures": 2,
        "sources": {"old-source": {"stop_reason": ""}}, "tasks": {"old-row": old}}))
    _run_completed(start, "new-row", "new-source")
    assert _checkpoint(path)["tasks"]["old-row"] == old
