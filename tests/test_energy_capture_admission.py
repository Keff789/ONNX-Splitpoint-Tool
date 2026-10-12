"""Admission is checked before each normal physical reserve, including retry.

Independent local fake leaves record preflight/collector/workload starts; these
tests never connect to hardware or modify an old product campaign.
"""
import argparse
import copy
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy import collector
from onnx_splitpoint_tool.workflow import hardware_acceptance_native
from test_v283_campaign_energy_budget import campaign_rig, measure


HOOK = "onnx_splitpoint_tool.workflow.hardware_acceptance_native:admit_energy_capture"


def descriptor(tmp_path, debt=None):
    path = tmp_path / "admission.json"
    path.write_text(json.dumps({"prior": debt or {"repeat:0": 1, "repeat:1": 0, "repeat:2": 0}}))
    return dict(capture_admission_file=str(path), capture_admission_sha256=collector._sha256_path(path),
                capture_admission_hook=HOOK)


def normal_guard(monkeypatch, calls, *, mutate_after=None, refuse=False):
    def admit(bound, request):
        budget = json.loads(Path(request["campaign_budget_file"]).read_text())
        chains = budget["tasks"][request["campaign_row_id"]]["chains"]
        calls.append({"request": copy.deepcopy(request), "existing_chains": copy.deepcopy(chains)})
        assert all(chain["finished"] is True for chain in chains)
        assert not any(chain["run_directory"] == request["run_directory"] for chain in chains)
        assert request["campaign_source_id"] == "fake_no_socket"
        assert request["duration_s"] == 1.0 and request["command_contract_sha256"] == "a" * 64
        assert request["setup_id"] == "fake" and request["require_runtime_work_units"] is True
        assert request["require_command_window_alignment"] is True
        counts = dict(bound["prior"])
        for chain in chains:
            counts[chain["logical_repeat"]] += 1
        if refuse or counts[request["logical_repeat"]] >= 3 or sum(counts.values()) >= 9:
            raise ValueError("normal cumulative admission refuses exhausted repeat/case")
        if mutate_after == len(calls):
            path = Path(request["campaign_budget_file"]).parent / "admission.json"
            path.write_text(json.dumps({**bound, "changed": True}))
        return {"ok": True, "prior_and_current_chains": counts}
    monkeypatch.setattr(hardware_acceptance_native, "admit_energy_capture", admit, raising=False)


def capture(rig, name="measurement", **kwargs):
    options = dict(campaign_max_retries=2, campaign_max_transport_failures=20, invalid_repeat_max_retries=2)
    options.update(kwargs)
    return measure(rig, name, **options)


def test_prior_debt_charges_every_internal_retry_before_reserve(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ["drop", "success", "success", "drop", "success"])
    options = descriptor(tmp_path)
    original = Path(options["capture_admission_file"]).read_bytes()
    calls = []
    normal_guard(monkeypatch, calls)
    with pytest.raises(ValueError, match="normal cumulative admission"):
        capture(rig, **options)
    assert len(calls) == 5
    assert [call["request"]["logical_repeat"] for call in calls] == ["repeat:0", "repeat:1", "repeat:2", "repeat:0", "repeat:0"]
    assert [len(call["existing_chains"]) for call in calls] == [0, 1, 2, 3, 4]
    assert rig.counts() == {"preflight": 4, "collector": 4, "workload": 4}
    budget = json.loads((tmp_path / "campaign.json").read_text())
    assert len(budget["tasks"]["row-a"]["chains"]) == 4
    assert Path(options["capture_admission_file"]).read_bytes() == original
    refused = list((tmp_path / "measurement").rglob("capture_admission.json"))
    assert len(refused) == 5 and sum(json.loads(path.read_text())["status"] == "refused" for path in refused) == 1


def test_remaining_attempt_succeeds_with_regular_three_repeats_and_retry_policy(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ["drop", "success", "success", "success"])
    options = descriptor(tmp_path)
    calls = []
    normal_guard(monkeypatch, calls)
    result = capture(rig, **options)
    assert result["ok"] is True and result["repeat_contract_complete"] is True
    assert result["run_count"] == 3 and result["invalid_repeat_max_retries"] == 2
    assert result["task_budget"]["limits"] == {"max_chains": 9, "max_retries": 2, "max_transport_failures": 20}
    assert len(calls) == 4 and rig.counts() == {"preflight": 4, "collector": 4, "workload": 4}
    assert result["task_budget"]["counts"]["valid_logical_repeats"] == 3
    for call in calls:
        receipt = json.loads((Path(call["request"]["run_directory"]) / "capture_admission.json").read_text())
        assert receipt["status"] == "admitted" and receipt["request"] == call["request"]
        assert receipt["descriptor_sha256"] == options["capture_admission_sha256"]


def test_reused_valid_slots_do_not_reserve_or_call_physical_admission_again(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    options = descriptor(tmp_path, {"repeat:0": 0, "repeat:1": 0, "repeat:2": 0})
    calls = []
    normal_guard(monkeypatch, calls)
    original = capture(rig, name="initial", **options)
    assert original["ok"] is True
    replay = capture(rig, name="reuse", reuse_campaign_repeats=True, **options)
    assert replay["ok"] is True and all(entry.get("campaign_reused_repeat") for entry in replay["runs"])
    assert len(calls) == 3 and rig.counts() == {"preflight": 3, "collector": 3, "workload": 3}


@pytest.mark.parametrize("failure", ["hash", "symlink", "refused", "non_object"])
def test_refusal_precedes_any_reserve_preflight_or_collector(tmp_path, monkeypatch, failure):
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    options = descriptor(tmp_path)
    path = Path(options["capture_admission_file"])
    calls = []
    normal_guard(monkeypatch, calls, refuse=failure == "refused")
    if failure == "hash":
        path.write_text("{}")
    elif failure == "symlink":
        link = tmp_path / "symlink.json"
        link.symlink_to(path)
        options["capture_admission_file"] = str(link)
    elif failure == "non_object":
        path.write_text("[]")
        options["capture_admission_sha256"] = collector._sha256_path(path)
    with pytest.raises(ValueError):
        capture(rig, **options)
    assert rig.counts() == {"preflight": 0, "collector": 0, "workload": 0}
    assert json.loads((tmp_path / "campaign.json").read_text())["tasks"]["row-a"]["chains"] == []
    receipt = json.loads((tmp_path / "measurement/run_000/capture_admission.json").read_text())
    assert receipt["status"] == "refused"


def test_descriptor_hash_is_rechecked_on_internal_retry(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ["drop", "success", "success", "success"])
    options = descriptor(tmp_path)
    calls = []
    normal_guard(monkeypatch, calls, mutate_after=3)
    with pytest.raises(ValueError, match="descriptor changed"):
        capture(rig, **options)
    assert len(calls) == 3 and rig.counts() == {"preflight": 3, "collector": 3, "workload": 3}
    assert len(json.loads((tmp_path / "campaign.json").read_text())["tasks"]["row-a"]["chains"]) == 3


def test_existing_admission_evidence_is_never_overwritten(tmp_path, monkeypatch):
    options = descriptor(tmp_path)
    run = tmp_path / "original/run_000"
    run.mkdir(parents=True)
    receipt = run / "capture_admission.json"
    receipt.write_text('{"status":"original immutable evidence"}')
    original = receipt.read_bytes()
    with pytest.raises(ValueError, match="attempt output already exists"):
        collector._capture_admission_before_reserve(options, {"run_directory": str(run)})
    assert receipt.read_bytes() == original


@pytest.mark.parametrize("changed", ["missing_hash", "relative_path", "private_module", "inline_code", "bad_hash"])
def test_cli_options_only_accept_complete_normal_product_binding(tmp_path, changed):
    options = descriptor(tmp_path)
    if changed == "missing_hash":
        options["capture_admission_sha256"] = ""
    elif changed == "relative_path":
        options["capture_admission_file"] = "relative.json"
    elif changed == "private_module":
        options["capture_admission_hook"] = "private_code.guard:admit"
    elif changed == "inline_code":
        options["capture_admission_hook"] = "onnx_splitpoint_tool.guard:lambda request: True"
    else:
        options["capture_admission_sha256"] = "bad"
    with pytest.raises(ValueError, match="capture admission requires"):
        collector.capture_admission_forward_args(SimpleNamespace(**options))


def test_normal_measurement_cli_passes_hash_bound_admission_to_collector(tmp_path, monkeypatch, capsys):
    from scripts import energy_measurement_cli as cli
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    options = descriptor(tmp_path)
    monkeypatch.setattr(cli, "_measurement_context", lambda *args: (rig.defaults, rig.setup))
    observed = []
    def collect(**kwargs):
        observed.append(kwargs)
        return {"ok": True}
    monkeypatch.setattr(cli, "run_fast_firmware_measurement", collect)
    flags = collector.capture_admission_forward_args(SimpleNamespace(**options))
    assert cli.main(["measure", "--setup-id", "fake", "--out", str(tmp_path / "cli"), "--command", rig.command, *flags]) == 0
    assert {key: observed[0][key] for key in options} == options
    assert rig.counts() == {"preflight": 0, "collector": 0, "workload": 0}
    capsys.readouterr()


def test_normal_native_wrapper_forwards_admission_to_local_planner(tmp_path, monkeypatch):
    from scripts import run_native_producer_energy_from_summary as wrapper
    options = descriptor(tmp_path)
    flags = collector.capture_admission_forward_args(SimpleNamespace(**options))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": []}))
    observed = []
    def local_plan(argv, **kwargs):
        assert kwargs["label"] == "energy_plan"
        observed.append(argv)
        return {"rc": 1}
    monkeypatch.setattr(wrapper, "_run", local_plan)
    monkeypatch.setattr(sys, "argv", ["normal-wrapper", "--summary", str(summary), "--out-dir", str(tmp_path / "parent"), *flags])
    assert wrapper.main() == 2
    assert len(observed) == 1
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        assert observed[0][observed[0].index(flag) + 1] == value


def test_normal_planner_forwards_admission_to_every_measurement_command(tmp_path, monkeypatch):
    from test_p03_native_energy_planner_invariant import NativeEnergyPlannerInvariantTests
    flags = collector.capture_admission_forward_args(SimpleNamespace(**descriptor(tmp_path)))
    parse = argparse.ArgumentParser.parse_args
    def parse_with_flags(parser, args=None, namespace=None):
        return parse(parser, [*(sys.argv[1:] if args is None else args), *flags], namespace)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_with_flags)
    plan, _, _ = NativeEnergyPlannerInvariantTests()._run_plan({"resnet50": ("b001",)}, mixed_annotations=False)
    commands = [shlex.split(row["measure_command"]) for row in plan["rows"] if row.get("measure_command")]
    assert commands
    for command in commands:
        for flag, value in zip(flags[::2], flags[1::2]):
            assert command[command.index(flag) + 1] == value


@pytest.mark.parametrize("name", ["energy_measurement_cli.py", "run_native_producer_energy_from_summary.py", "native_producer_energy_plan.py"])
def test_changed_normal_cli_resource_mirrors_are_identical(name):
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / name).read_bytes() == (root / "onnx_splitpoint_tool/resources/remote_scripts" / name).read_bytes()
