"""Profile/CLI retry contracts with controlled local collector processes only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import threading
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy import collector, task_budget
from test_v283_campaign_energy_budget import campaign_rig, measure


POLICY = {"enabled": True, "max_retries": 2, "max_transport_failures": 20}


def run(rig, name="series", **kwargs):
    return measure(rig, name, campaign_max_retries=2, campaign_max_transport_failures=20,
                   invalid_repeat_max_retries=None, **kwargs)


def record(rig, result, record_property):
    record_property("controlled_retry_evidence", json.dumps({
        "process_starts": rig.counts(), "task_budget": result["task_budget"],
        "selected_attempts": [row["selected_repeat_attempt_index"] for row in result.get("runs", [])],
        "ok": result["ok"], "error": result.get("error"),
    }))


def test_twenty_uses_one_validator_from_profile_cli_and_task(tmp_path, monkeypatch):
    seen = []
    real = task_budget.positive_transport_failure_limit

    def validate(value):
        seen.append(value)
        return real(value)

    monkeypatch.setattr(task_budget, "positive_transport_failure_limit", validate)
    profile_args = task_budget.campaign_budget_profile_args({"energy": {"task_budget": POLICY}}, tmp_path)
    assert seen == [20]
    parser = argparse.ArgumentParser()
    task_budget.add_campaign_budget_arguments(parser, measurement=True)
    args = parser.parse_args(profile_args + ["--campaign-row-id", "row", "--campaign-repeats", "3"])
    assert seen == [20, 20]
    assert task_budget.campaign_budget_forward_args(args) == profile_args
    assert seen == [20, 20, 20]
    entered = []

    @task_budget.bounded_energy_task
    def leaf(command, output, **kwargs):
        entered.append(kwargs["invalid_repeat_max_retries"])
        return {"ok": True, "runs": []}

    result = leaf("controlled", tmp_path / "task", setup=SimpleNamespace(urecs_address="fixture"),
                  **task_budget.campaign_budget_measurement_kwargs(args))
    assert entered == [2]
    assert set(seen) == {20} and len(seen) >= 5
    assert result["task_budget"]["limits"] == {
        "max_chains": 9, "max_retries": 2, "max_transport_failures": 20}


@pytest.mark.parametrize("value", [True, False, None, 0, -1, 1.5, 20.0, "20"])
def test_invalid_failure_budget_rejected_by_all_python_entries(tmp_path, value):
    policy = {**POLICY, "max_transport_failures": value}
    with pytest.raises(ValueError, match="positive integer"):
        task_budget.campaign_budget_policy({"energy": {"task_budget": policy}})
    args = SimpleNamespace(campaign_budget_file=str(tmp_path / "budget.json"),
                           campaign_max_retries=2, campaign_max_transport_failures=value)
    with pytest.raises(ValueError, match="positive integer"):
        task_budget.campaign_budget_forward_args(args)

    @task_budget.bounded_energy_task
    def never(*args, **kwargs):
        raise AssertionError("invalid policy reached acquisition")

    with pytest.raises(ValueError, match="positive integer"):
        never("controlled", tmp_path / "task", campaign_budget_file=tmp_path / "budget.json",
              campaign_row_id="row", campaign_repeats=3, campaign_max_retries=2,
              campaign_max_transport_failures=value)
    assert not (tmp_path / "budget.json").exists()


@pytest.mark.parametrize("text", ["true", "false", "null", "0", "-1", "1.5", "20.0"])
def test_invalid_failure_budget_rejected_by_cli_parser(text):
    parser = argparse.ArgumentParser()
    task_budget.add_campaign_budget_arguments(parser)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--campaign-max-transport-failures", text])
    assert exc.value.code == 2


def test_two_invalid_then_first_valid_uses_exactly_three_attempts(tmp_path, monkeypatch, record_property):
    rig = campaign_rig(tmp_path, monkeypatch, ["drop", "success", "success", "drop", "success"])
    assert rig.defaults.invalid_repeat_max_retries == 1  # campaign must reach the outer loop
    result = run(rig)
    assert result["ok"] and result["repeat_contract_complete"]
    assert result["invalid_repeat_max_retries"] == result["invalid_repeat_max_retries_requested"] == 2
    assert rig.counts() == {"preflight": 5, "collector": 5, "workload": 5}
    chains = result["task_budget"]["chains"]
    assert [c["valid"] for c in chains if c["logical_repeat"] == "repeat:0"] == [False, False, True]
    assert all(c["source_completion_verified"] for c in chains)
    assert [r["selected_repeat_attempt_index"] for r in result["runs"]] == [2, 0, 0]
    assert result["task_budget"]["counts"]["valid_logical_repeats"] == 3
    assert result["task_budget"]["campaign_source"]["transport_failures"] == 2
    assert result["task_budget"]["campaign_source"]["stop_reason"] == ""
    first = result["runs"][0]
    assert first["repeat_attempt_count"] == 3
    assert [h["selected"] for h in first["repeat_attempt_history"]] == [False, False, True]
    for history in first["repeat_attempt_history"][:2]:
        old = json.loads((Path(history["run_directory"]) / "energy_summary.json").read_text())
        assert old["final_energy_gate_status"] == "fail"
        assert (Path(history["run_directory"]) / "collector_storage/fast_firmware.parquet").exists()
    assert len(list((tmp_path / "series/repeat_retry_attempts").rglob("collector_reconnect_evidence.json"))) == 2
    record(rig, result, record_property)


def test_three_invalid_attempts_never_start_fourth_and_reentry_keeps_history(tmp_path, monkeypatch, record_property):
    rig = campaign_rig(tmp_path, monkeypatch, ["success", "success", "drop"])
    result = run(rig)
    assert not result["ok"] and result["error"] == "task_logical_retry_limit"
    assert rig.counts()["collector"] == 5
    assert sum(c["logical_repeat"] == "repeat:2" for c in result["task_budget"]["chains"]) == 3
    original_chains = result["task_budget"]["chains"]
    again = run(rig, "reentry")
    assert rig.counts()["collector"] == 5
    assert again["task_budget"]["chains"] == original_chains
    assert again["task_budget"]["campaign_source"]["transport_failures"] == 3
    record(rig, again, record_property)


def test_three_repeat_row_reserves_at_most_nine_preflight_chains(tmp_path, monkeypatch, record_property):
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    rig.env["R3_PREFLIGHT_FAIL"] = "1"
    for logical in range(3):
        for attempt in range(3):
            result = run(rig, f"preflight-{logical}-{attempt}", run_count=1, exact_run_count=True,
                         _task_logical_repeat=f"repeat:{logical}")
            assert result["runs"][0]["status"] == "preflight_failed"
    result = run(rig, "tenth", run_count=1, exact_run_count=True, _task_logical_repeat="repeat:3")
    assert result["error"] == "task_chain_limit"
    assert rig.counts() == {"preflight": 9, "collector": 0, "workload": 0}
    assert result["task_budget"]["counts"]["begun_chains"] == 9
    record(rig, result, record_property)


def test_successful_retry_and_later_failure_preserve_count_until_twentieth(tmp_path, monkeypatch, record_property):
    rig = campaign_rig(tmp_path, monkeypatch, ["drop", "success", "success", "success", "drop"])
    first = run(rig, "recovered", row="first-model")
    assert first["ok"] and first["task_budget"]["campaign_source"]["transport_failures"] == 1
    # Independent model/row transactions use the existing checkpoint; successful
    # retry and later re-entry may not reset its physical-source counter.
    for failure_count in range(2, 21):
        result = run(rig, f"model-{failure_count}", row=f"model-{failure_count}",
                     run_count=1, exact_run_count=True)
        source = result["task_budget"]["campaign_source"]
        assert source["transport_failures"] == failure_count
        if failure_count < 20:
            assert source["stop_reason"] == ""
        else:
            assert result["error"] == "campaign_source_transport_failure_limit"
    before = rig.counts()
    blocked = run(rig, "blocked-model", row="blocked-model")
    assert blocked["execution_status"] == "NOT_RUN" and rig.counts() == before
    rig.setup.urecs_address = "INDEPENDENT_SOURCE"
    rig.scenario.write_text(json.dumps(["success"]))
    independent = run(rig, "independent", row="independent")
    assert independent["ok"]
    checkpoint = json.loads((tmp_path / "campaign.json").read_text())
    assert checkpoint["sources"]["fake_no_socket"]["transport_failures"] == 20
    assert checkpoint["sources"]["independent_source"]["transport_failures"] == 0
    record(rig, independent, record_property)


def test_missing_source_end_still_stops_before_retry_with_twenty_available(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ["first_sample", "success"])
    result = run(rig)
    assert result["error"] == "campaign_source_completion_unresolved"
    assert rig.counts()["collector"] == 1
    again = run(rig, "new-row", row="new-row")
    assert again["execution_status"] == "NOT_RUN" and rig.counts()["collector"] == 1


def test_cancel_during_existing_backoff_still_prevents_retry(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ["drop", "success", "success", "success"])
    cancel = threading.Event()
    original = collector._collector_reconnect_backoff
    calls = []

    def cancel_after_initial_repeats(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        if len(calls) == 1:  # drop uses the backoff immediately before its retry
            cancel.set()
        return result

    monkeypatch.setattr(collector, "_collector_reconnect_backoff", cancel_after_initial_repeats)
    result = run(rig, cancel_event=cancel)
    assert result["error"] == "task_cancelled"
    assert rig.counts()["collector"] == 3
    assert len(calls) == 1


def test_live_owner_still_blocks_another_row_with_twenty_available(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    limits = {"max_chains": 9, "max_retries": 2, "max_transport_failures": 20}
    with (tmp_path / "campaign.json").open("w+") as handle:
        owner = task_budget.EnergyTaskBudget(handle, limits, campaign_row="active-row",
                                            source_id="fake_no_socket")
        attempt = tmp_path / "owned-attempt"
        attempt.mkdir()
        chain = owner.reserve("repeat:0", attempt, True)
        assert chain is not None
        result = run(rig, "competing", row="competing")
        assert result["status"] == "not_dispatched" and "energy_source_busy" in result["reason"]
        assert rig.counts() == {"preflight": 0, "collector": 0, "workload": 0}
        assert not result["task_budget"]["campaign_source"]["stop_reason"]
        owner.finish(chain, {"status": "preflight_failed", "collector_started": False}, attempt)


def test_larger_contract_cannot_rewrite_an_existing_campaign(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    measure(rig, "old-row", run_count=1, exact_run_count=True)
    path = tmp_path / "campaign.json"
    original = path.read_bytes()
    before = rig.counts()
    with pytest.raises(ValueError, match="differs from existing checkpoint"):
        run(rig, "changed-budget", row="another-row")
    assert path.read_bytes() == original and rig.counts() == before


@pytest.mark.parametrize("completed_repeats", [1, 3])
def test_direct_cli_reentry_cannot_reserve_an_already_valid_campaign_repeat(
    tmp_path, monkeypatch, completed_repeats, record_property,
):
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    first = run(rig, "original", run_count=completed_repeats, exact_run_count=True)
    assert first["ok"]
    original = (tmp_path / "campaign.json").read_bytes()
    counts = rig.counts()
    again = run(rig, "fresh-output")
    assert rig.counts() == counts
    assert again["status"] == "not_dispatched"
    assert again["reason"] == "energy_repeat_already_valid:repeat:0"
    assert not again["task_budget"]["stop_reason"]
    assert not again["task_budget"]["campaign_source"]["stop_reason"]
    assert (tmp_path / "campaign.json").read_bytes() == original
    record(rig, again, record_property)
    if completed_repeats == 3:
        other_row = run(rig, "different-row", row="different-row")
        assert other_row["ok"] and rig.counts()["collector"] == counts["collector"] + 3
        rig.setup.urecs_address = "INDEPENDENT_SOURCE"
        other_source = run(rig, "other-source", row="other-source")
        assert other_source["ok"] and rig.counts()["collector"] == counts["collector"] + 6
        checkpoint = json.loads((tmp_path / "campaign.json").read_text())
        assert checkpoint["tasks"]["row-a"]["chains"] == first["task_budget"]["chains"]
        assert all(not source["stop_reason"] for source in checkpoint["sources"].values())


def test_normal_plan_measure_cli_builds_collector_with_profile_budget(tmp_path, monkeypatch, capsys, record_property):
    from scripts import native_producer_energy_plan as planner, energy_measurement_cli as cli
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    rig.defaults.pre_duration_s = rig.defaults.post_duration_s = 5
    rig.defaults.physical_scope = "FS"
    rig.setup.setup_id = "orin_nx_hailo8_01"
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": [{
        "ok": True, "backend": "native_full_hailo8", "model": "fixture_resnet",
        "case": "full", "precision": "runtime_precision", "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8", "task": "classification", "fps_makespan": 8.,
        "full_command_contract": {"contract_sha256": "a" * 64},
    }]}))

    def verified_contract(raw, *, expected_identity):
        return {**expected_identity, "contract_sha256": "a" * 64,
                "runtime_options": {}, "energy_workload": {}, "artifacts": {}}, "controlled_contract"

    monkeypatch.setattr(planner, "_verify_full_command_contract", verified_contract)
    monkeypatch.setattr(planner, "_full_runtime_argv", lambda *a, **k: ["controlled-workload"])
    monkeypatch.setattr(planner, "_full_preflight_argv", lambda *a, **k: ["controlled-preflight"])
    monkeypatch.setattr(planner, "_process_local_runtime_environment", lambda contract: {})
    budget_args = task_budget.campaign_budget_profile_args({"energy": {"task_budget": POLICY}}, tmp_path)
    plan_dir = tmp_path / "plan"
    monkeypatch.setattr(sys, "argv", [str(planner.__file__), "--summary", str(summary),
        "--out-dir", str(plan_dir), "--hailo8-ssh", "fixture-host", "--duration-s", "60",
        "--runs", "3", "--physical-scope", "FS", "--window-label", "command",
        "--require-runtime-work-units", "--require-command-window-alignment",
        "--measure-all-runtime-successful", "--allow-unpaired", *budget_args])
    assert planner.main() == 0
    plan = json.loads((plan_dir / "native_producer_energy_plan.json").read_text())
    assert len(plan["rows"]) == 1, plan.get("excluded_rows")
    command = shlex.split(plan["rows"][0]["measure_command"])
    measure_args = command[command.index("measure"):]
    assert measure_args[measure_args.index("--campaign-max-transport-failures") + 1] == "20"
    assert measure_args[measure_args.index("--campaign-max-retries") + 1] == "2"
    assert measure_args[measure_args.index("--invalid-repeat-max-retries") + 1] == "2"
    assert measure_args[measure_args.index("--duration") + 1] == "60"
    assert measure_args[measure_args.index("--runs") + 1] == "3"
    # The planner, parser and collector stay real. Replace only fixture-owned
    # remote command leaves before reaching the actual collector process boundary.
    Path(measure_args[measure_args.index("--command-file") + 1]).write_text(rig.command)
    Path(measure_args[measure_args.index("--preflight-command-file") + 1]).write_text(rig.preflight)
    monkeypatch.setattr(cli, "_measurement_context", lambda *args: (rig.defaults, rig.setup))
    observed = []
    actual_run_one = collector._run_one

    class ControlledCollectorBoundary(Exception):
        pass

    def stop_at_collector(command, **kwargs):
        if command[0] == rig.defaults.collector_binary and "--help" not in command:
            observed.append(command)
            raise ControlledCollectorBoundary
        return actual_run_one(command, **kwargs)

    monkeypatch.setattr(collector, "_run_one", stop_at_collector)
    with pytest.raises(ControlledCollectorBoundary):
        cli.main(measure_args)
    assert len(observed) == 1
    collector_command = observed[0]
    assert "fast-firmware" in collector_command and "--sample-rate=2000" in collector_command
    assert all(flag in collector_command for flag in ("-d=76s", "-b=5s", "-e=5s", "--channel=0"))
    assert collector_command[collector_command.index("--run-id") + 1].startswith("native_")
    checkpoint = json.loads((tmp_path / "energy_task_budget.json").read_text())
    assert checkpoint["max_transport_failures"] == 20
    row = next(iter(checkpoint["tasks"].values()))
    assert row["limits"] == {"max_chains": 9, "max_retries": 2, "max_transport_failures": 20}
    assert len(row["chains"]) == 1 and row["chains"][0]["collector_started"] is False
    assert rig.counts() == {"preflight": 1, "collector": 0, "workload": 0}
    record_property("normal_plan_cli_collector", json.dumps({
        "measure_command": command, "collector_command": collector_command,
        "budget": checkpoint, "physical_collector_starts": 0,
    }))
