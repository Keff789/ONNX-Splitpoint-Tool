"""One diagnostic setup capture through the real CLI/collector/budget; SDK-free."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy import collector, task_budget
from scripts import energy_measurement_cli as cli
from test_v283_campaign_energy_budget import campaign_rig


def _rig(tmp_path, monkeypatch, modes):
    rig = campaign_rig(tmp_path, monkeypatch, modes)
    rig.setup.urecs_address = "127.0.0.1"
    rig.defaults.pre_duration_s = rig.defaults.post_duration_s = 5.0
    rig.defaults.invalid_repeat_max_retries = 3
    assert rig.defaults.min_active_duration_s == 30
    # Existing software process leaves provide a two-second sample window.
    # The normal CLI still executes its real `sleep 2` workload.
    path = tmp_path / "fake_collector.py"
    path.write_text(path.read_text().replace("1999", "3999").replace("2500", "4500")
                    .replace("27000000", "28000000"))
    path = tmp_path / "fake_power.py"
    path.write_text(path.read_text().replace("1999", "3999"))
    monkeypatch.setattr(cli, "_measurement_context", lambda *_: (rig.defaults, rig.setup))
    return rig


def _argv(rig, name="test", *, budget=True):
    result = ["test-setup", "--setup-id", rig.setup.setup_id, "--acquire", "--sleep-s", "2",
              "--out", str(rig.root / name)]
    if budget:
        result += ["--campaign-budget-file", str(rig.root / "campaign.json"),
                   "--campaign-row-id", "diagnostic-source-test", "--campaign-repeats", "1",
                   "--campaign-max-retries", "0", "--campaign-max-transport-failures", "20",
                   "--task-logical-repeat", "diagnostic-source-test:0"]
    return result


def _read_cli(args, capsys):
    rc = cli.main(args)
    return rc, json.loads(capsys.readouterr().out)["measurement"]


def test_one_real_cli_capture_is_diagnostic_and_campaign_accounted(tmp_path, monkeypatch, capsys):
    rig = _rig(tmp_path, monkeypatch, ["success"])
    guard = tmp_path / "controlled-starts.json"
    guard.write_text(json.dumps({"schema": "onnx-splitpoint/controlled-test-starts",
                                "limits": {"collector": 1}, "entries": []}))
    monkeypatch.setenv("ONNX_SPLITPOINT_TEST_START_JOURNAL", str(guard))
    rc, measurement = _read_cli(_argv(rig), capsys)
    assert rc == 0 and measurement["ok"] is True
    assert rig.counts()["collector"] == 1
    assert rig.counts()["preflight"] == 0
    assert measurement["run_count"] == 1
    assert measurement["invalid_repeat_max_retries"] == 0
    assert measurement["energy_window_method_ab"]["exact_run_count_requested"] is True
    assert measurement["energy_window_method_ab"]["enabled"] is False
    assert measurement["diagnostic_only"] is True
    assert measurement["scientific_primary_claim_eligible"] is False
    assert collector._energy_repeat_contract_complete(
        measurement["runs"], requested_count=1, require_ab=False)
    entry = measurement["runs"][0]
    assert entry["collector_rc"] == entry["workload_command_rc"] == 0
    assert entry["collector_expected_duration_s"] == 28
    assert entry["command_window_trace_binding"]["verified"] is True
    assert entry["energy_configured_workload_duration_s"] == 2
    assert task_budget.source_completion(tmp_path / "test/run_000")["verified"] is True
    row = measurement["task_budget"]
    assert row["limits"] == {"max_chains": 1, "max_transport_failures": 20, "max_retries": 0}
    assert row["counts"]["collector_starts"] == row["counts"]["valid_logical_repeats"] == 1
    assert row["chains"][0]["source_completion_verified"] is True
    assert row["chains"][0]["logical_repeat"] == "diagnostic-source-test:0"
    assert len(json.loads(guard.read_text())["entries"]) == 1
    checkpoint_before = json.loads((tmp_path / "campaign.json").read_text())
    rc, again = _read_cli(_argv(rig, "reentry"), capsys)
    assert rc != 0 and again["ok"] is False
    assert rig.counts()["collector"] == 1
    assert again["task_budget"]["chains"] == row["chains"]
    assert json.loads((tmp_path / "campaign.json").read_text())["max_transport_failures"] == checkpoint_before["max_transport_failures"]


@pytest.mark.parametrize("mode", ["drop", "first_sample"])
def test_invalid_capture_has_no_retry_and_original_source_failure_stays(tmp_path, monkeypatch, capsys, mode):
    rig = _rig(tmp_path, monkeypatch, [mode, "success"])
    rc, measurement = _read_cli(_argv(rig), capsys)
    assert rc != 0 and measurement["ok"] is False
    assert rig.counts()["collector"] == 1
    assert measurement["invalid_repeat_max_retries"] == 0
    assert measurement["task_budget"]["counts"]["valid_logical_repeats"] == 0
    assert measurement["runs"][0]["final_energy_gate_status"] == "fail"
    expected_end = mode == "drop"
    assert task_budget.source_completion(tmp_path / "test/run_000")["verified"] is expected_end
    old_chains = measurement["task_budget"]["chains"]
    rc, again = _read_cli(_argv(rig, "later"), capsys)
    assert rc != 0 and again["task_budget"]["chains"] == old_chains
    assert rig.counts()["collector"] == 1
    if mode == "first_sample":
        assert measurement["task_budget"]["campaign_source"]["stop_reason"] == "campaign_source_completion_unresolved"


def test_existing_one_start_guard_blocks_before_collector(tmp_path, monkeypatch):
    rig = _rig(tmp_path, monkeypatch, ["success"])
    guard = tmp_path / "controlled-starts.json"
    guard.write_text(json.dumps({"schema": "onnx-splitpoint/controlled-test-starts",
        "limits": {"collector": 1}, "entries": [{"kind": "collector", "identity": "already-used"}]}))
    before = guard.read_bytes()
    monkeypatch.setenv("ONNX_SPLITPOINT_TEST_START_JOURNAL", str(guard))
    with pytest.raises(RuntimeError, match="physical start budget exhausted"):
        cli.main(_argv(rig))
    assert rig.counts()["collector"] == 0
    assert guard.read_bytes() == before


def test_setup_without_campaign_keeps_one_capture_defaults(tmp_path, monkeypatch, capsys):
    rig = _rig(tmp_path, monkeypatch, ["success"])
    rc, measurement = _read_cli(_argv(rig, budget=False), capsys)
    assert rc == 0 and measurement["ok"] is True
    assert rig.counts()["collector"] == 1
    assert measurement["run_count"] == 1 and measurement["invalid_repeat_max_retries"] == 0
    assert not (tmp_path / "campaign.json").exists()


def test_setup_cannot_change_existing_source_failure_limit(tmp_path, monkeypatch):
    rig = _rig(tmp_path, monkeypatch, ["success"])
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps({"campaign": True, "max_transport_failures": 20,
                              "tasks": {}, "sources": {}}))
    before = path.read_bytes()
    args = _argv(rig)
    args[args.index("--campaign-max-transport-failures") + 1] = "21"
    with pytest.raises(ValueError, match="contract differs"):
        cli.main(args)
    assert rig.counts()["collector"] == 0 and path.read_bytes() == before


def test_energy_cli_mirror_is_exact():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts/energy_measurement_cli.py").read_bytes() == (
        root / "onnx_splitpoint_tool/resources/remote_scripts/energy_measurement_cli.py").read_bytes()
