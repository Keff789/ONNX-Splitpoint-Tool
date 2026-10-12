"""Normal diagnostic raw energy admission, evidence and bounded reuse.

The independent process fixtures are local fakes; no DUT/collector socket or
SSH transport is used. Environment samples exercise the real evidence reader.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy import collector, environment_observer, task_budget
from onnx_splitpoint_tool.native_energy_reporting import (
    _cohort_environment_fields, collect_native_energy,
)
from test_v283_campaign_energy_budget import campaign_rig, measure
from test_yolo_energy_partial_reporting import publish


ROW = json.dumps(["hailo8_to_trt", "mobilenet_v3_large", "b001", "fake"], separators=(",", ":"))
OWNER = "actual-normal-parent"


def _local_environment(monkeypatch, *, invalid=False):
    class LocalObserver:
        def __init__(self, setup, output, environment, run_id):
            assert run_id == OWNER
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None

    monkeypatch.setattr(environment_observer, "EnergyEnvironmentObserver", LocalObserver)
    monkeypatch.setattr(task_budget, "_budget_parent", lambda environment=None: {"run_id": OWNER})
    validate = environment_observer.validate_environment

    def local_sample(root, entry, *, freeze=False):
        root = Path(root)
        trace = entry.get("command_window_trace_binding") or {}
        start, end = trace.get("command_start_realtime_ns"), trace.get("command_end_realtime_ns")
        if freeze and start and end:
            preflight = entry["preflight_evidence"]
            sample = {
                "phase": "during", "controller_request_start_ns": start,
                "controller_request_end_ns": end, "setup_id": "fake",
                "run_id": OWNER, "capture_group": str(root.parent),
                "cpu_frequency": [{"scaling_cur_freq": 1234}],
                "host": {"loadavg": [0.1, 0.1, 0.1]},
                "workload_roots": [{
                    "preflight_nonce": "foreign" if invalid else preflight["nonce"],
                    "command_contract_sha256": preflight["expected_command_contract_sha256"],
                    "workers": [{"pid": 123, "start_ticks": "456", "allowed_cpus": [0, 1],
                                 "threads": [{"tid": 123, "allowed_cpus": [0, 1]}]}],
                }],
            }
            (root / "environment_samples.jsonl").write_text(json.dumps(sample) + "\n")
        return validate(root, entry, freeze=freeze)

    monkeypatch.setattr(environment_observer, "validate_environment", local_sample)


def _measure(rig, name="raw", **kwargs):
    options = dict(row=ROW, campaign_max_retries=2, campaign_max_transport_failures=20,
                   invalid_repeat_max_retries=2, diagnostic_only=True,
                   require_environment_observation=True, environment_run_id=OWNER,
                   claim_exclusion_reason="quality_or_pairing_not_claim_eligible")
    options.update(kwargs)
    return measure(rig, name, **options)


def _plan():
    return {"backend": "hailo8_to_trt", "model": "mobilenet_v3_large", "case": "b001",
            "setup_id": "fake", "environment_observation_required": True,
            "environment_observation_run_id": OWNER, "diagnostic_only": True,
            "claim_eligible": False, "measurement_effective_repeats": 3}


def _raw_projection(root, aggregate):
    item = {"row": _plan(), "ok": aggregate["ok"], "run": {
        "rc": 0 if aggregate["ok"] else 1, "energy_aggregate": aggregate,
        "energy_aggregate_embedded": True, "energy_aggregate_complete": aggregate["ok"],
        "energy_aggregate_verified": aggregate["ok"],
    }}
    publish(root, [item])
    return collect_native_energy(root, include_partial=True)[0]


def test_diagnostic_environment_admission_starts_normal_observer_first(tmp_path, monkeypatch):
    events = []

    class Budget:
        def __init__(self, handle, limits, *args, **kwargs):
            assert limits == {"max_chains": 9, "max_retries": 2, "max_transport_failures": 20}
            assert kwargs["campaign_row"] == ROW and kwargs["cohort_id"] is None
        def is_set(self):
            return False
        def report(self, result, output):
            return result

    class Observer:
        def __init__(self, setup, output, environment, run_id):
            assert run_id == OWNER
        def __enter__(self):
            events.append("environment_started")
        def __exit__(self, *args):
            events.append("environment_finished")

    monkeypatch.setattr(task_budget, "EnergyTaskBudget", Budget)
    monkeypatch.setattr(task_budget, "_budget_parent", lambda environment=None: {"run_id": OWNER})
    monkeypatch.setattr(environment_observer, "EnergyEnvironmentObserver", Observer)

    @task_budget.bounded_energy_task
    def capture(command, output, **kwargs):
        assert events == ["environment_started"]
        assert kwargs["diagnostic_only"] is True
        assert kwargs["_task_budget"].require_environment_observation is True
        assert kwargs["invalid_repeat_max_retries"] == 2
        events.append("capture")
        return {"ok": True}

    assert capture("unchanged", tmp_path / "raw", campaign_budget_file=tmp_path / "budget",
                   campaign_row_id=ROW, campaign_repeats=3, campaign_max_retries=2,
                   campaign_max_transport_failures=20, diagnostic_only=True,
                   require_environment_observation=True, environment_run_id=OWNER,
                   require_runtime_work_units=True, require_command_window_alignment=True,
                   setup=SimpleNamespace(urecs_address="fake"))["ok"]
    assert events == ["environment_started", "capture", "environment_finished"]


@pytest.mark.parametrize("missing", ["campaign_budget_file", "require_runtime_work_units", "require_command_window_alignment"])
def test_environment_required_diagnostic_keeps_command_window_budget_guards(tmp_path, missing):
    @task_budget.bounded_energy_task
    def forbidden(*args, **kwargs):
        pytest.fail("capture must remain blocked")
    options = dict(campaign_budget_file=tmp_path / "budget", diagnostic_only=True,
                   require_environment_observation=True, require_runtime_work_units=True,
                   require_command_window_alignment=True)
    options.pop(missing)
    with pytest.raises(ValueError, match="required environment requires campaign command-window"):
        forbidden("unchanged", tmp_path / "raw", **options)
    assert not (tmp_path / "budget").exists()


def test_final_cohort_diagnostic_is_rejected_before_any_start(tmp_path, monkeypatch):
    _local_environment(monkeypatch)
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    with pytest.raises(ValueError, match="final cohort requires non-diagnostic"):
        _measure(rig, campaign_cohort="scientific-final")
    assert rig.counts() == {"preflight": 0, "collector": 0, "workload": 0}
    assert not (tmp_path / "campaign.json").exists()


def test_three_diagnostic_raw_repeats_pass_actual_environment_and_keep_claims_false(tmp_path, monkeypatch):
    _local_environment(monkeypatch)
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    result = _measure(rig)
    assert result["ok"] and result["repeat_contract_complete"]
    assert result["valid_postprocessed_runs"] == 3
    assert rig.counts() == {"preflight": 3, "collector": 3, "workload": 3}
    assert collector._clamp_diagnostic_claim_fields(result) == result
    assert result["diagnostic_only"] is True and result["energy_efficiency_claim_eligible"] is False
    for entry in result["runs"]:
        assert entry["environment_observation"]["ok"] is True
        assert entry["campaign_run_id"] == OWNER and entry["campaign_row"] == ROW
        assert collector._clamp_diagnostic_claim_fields(entry) == entry
    row = _raw_projection(tmp_path / "projection", result)
    assert row["ok"] is True and row["energy_repeat_valid_n"] == 3
    assert row["environment_observation_status"] == "pass"
    assert row["diagnostic_only"] is True and row["claim_eligible"] is False and row["claim_ok"] is False


def test_actual_environment_failure_blocks_raw_acceptance_and_import(tmp_path, monkeypatch):
    _local_environment(monkeypatch, invalid=True)
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    result = _measure(rig)
    assert result["ok"] is False and result["repeat_contract_complete"] is False
    # Marker postprocessing remains visible; required environment admission is
    # separate and must not promote those observations into accepted repeats.
    assert result["valid_postprocessed_runs"] == 3
    assert all(not entry["environment_observation"]["ok"] for entry in result["runs"])
    assert all(not collector._energy_repeat_aggregation_eligible(entry, require_ab=False) for entry in result["runs"])
    assert result["task_budget"]["counts"]["valid_logical_repeats"] == 0
    row = _raw_projection(tmp_path / "projection", result)
    assert row["ok"] is False and row["environment_observation_status"] == "fail"
    assert row["claim_eligible"] is False


def test_normal_two_retry_limit_keeps_diagnostic_claim_clamps(tmp_path, monkeypatch):
    _local_environment(monkeypatch)
    rig = campaign_rig(tmp_path, monkeypatch, ["drop", "success", "success", "drop", "success"])
    result = _measure(rig)
    assert result["ok"] is True and result["repeat_contract_complete"] is True
    assert result["repeat_retry_attempt_count"] == 2 and result["repeat_retry_recovered_count"] == 1
    assert result["runs"][0]["selected_repeat_attempt_index"] == 2
    assert rig.counts() == {"preflight": 5, "collector": 5, "workload": 5}
    assert result["task_budget"]["counts"]["valid_logical_repeats"] == 3
    assert collector._clamp_diagnostic_claim_fields(result) == result
    failed = json.loads((tmp_path / "raw/run_000/energy_summary.json").read_text())
    assert failed["final_energy_gate_status"] == "fail" and failed["diagnostic_only"] is True


def test_partial_valid_diagnostic_slots_are_reused_without_new_starts_or_rewrites(tmp_path, monkeypatch):
    _local_environment(monkeypatch)
    rig = campaign_rig(tmp_path, monkeypatch, ["success", "drop", "success", "success"])
    partial = _measure(rig, "partial", invalid_repeat_max_retries=0)
    assert partial["ok"] is False and partial["task_budget"]["counts"]["valid_logical_repeats"] == 2
    original_paths = [tmp_path / f"partial/run_{index:03d}/energy_summary.json" for index in range(3)]
    original = [path.read_bytes() for path in original_paths]
    old_chains = copy.deepcopy(partial["task_budget"]["chains"])
    result = _measure(rig, "retry", reuse_campaign_repeats=True)
    assert result["ok"] is True and result["repeat_contract_complete"] is True
    assert [bool(entry.get("campaign_reused_repeat")) for entry in result["runs"]] == [True, False, True]
    assert rig.counts() == {"preflight": 4, "collector": 4, "workload": 4}
    assert result["task_budget"]["chains"][:3] == old_chains
    assert [path.read_bytes() for path in original_paths] == original
    assert result["task_budget"]["counts"]["begun_chains"] == 4
    assert collector._clamp_diagnostic_claim_fields(result) == result
    assert _raw_projection(tmp_path / "projection", result)["environment_observation_status"] == "pass"


@pytest.fixture
def accepted_repeat(tmp_path, monkeypatch):
    _local_environment(monkeypatch)
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    result = _measure(rig, run_count=1, exact_run_count=True, invalid_repeat_max_retries=0)
    assert result["ok"] is True
    chain = result["task_budget"]["chains"][0]
    options = dict(command=rig.command, command_contract_sha256="a" * 64, duration_s=1, require_ab=False)
    return rig, chain, options


def test_verified_raw_repeat_reader_is_read_only(accepted_repeat):
    rig, chain, options = accepted_repeat
    paths = [Path(chain["run_directory"]) / "energy_summary.json", rig.root / "campaign.json"]
    original = [path.read_bytes() for path in paths]
    entry = task_budget.read_verified_campaign_repeat(chain, **options)
    assert entry["diagnostic_only"] is True and entry["environment_observation"]["ok"] is True
    assert collector._clamp_diagnostic_claim_fields(entry) == entry
    assert [path.read_bytes() for path in paths] == original


def test_required_environment_reuse_rejects_removed_purpose_flag_before_skipping_slot(accepted_repeat):
    rig, chain, options = accepted_repeat
    path = Path(chain["run_directory"]) / "energy_summary.json"
    entry = json.loads(path.read_text())
    entry.pop("environment_observation_required")
    path.write_text(json.dumps(entry))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="environment/purpose binding invalid"):
        task_budget.read_verified_campaign_repeat(chain, **options, require_environment_observation=True)
    with pytest.raises(ValueError, match="environment/purpose binding invalid"):
        _measure(rig, "removed-flag", run_count=1, exact_run_count=True, reuse_campaign_repeats=True)
    assert rig.counts() == {"preflight": 1, "collector": 1, "workload": 1}
    assert path.read_bytes() == original


@pytest.mark.parametrize("mutation", ["source_close", "environment_hash", "nonce", "claim", "command", "contract", "duration", "cohort"])
def test_diagnostic_raw_reuse_retains_all_original_proof_guards(accepted_repeat, mutation):
    rig, chain, options = accepted_repeat
    root = Path(chain["run_directory"])
    entry = json.loads((root / "energy_summary.json").read_text())
    if mutation == "source_close":
        path = root / "collector_stdout.log"
        path.write_bytes(path.read_bytes().replace(b"verified=true", b"verified=false"))
    elif mutation == "environment_hash":
        entry["environment_observation"]["evidence_sha256"] = "f" * 64
    elif mutation == "nonce":
        path = Path(entry["environment_observation"]["evidence_path"])
        frozen = json.loads(path.read_text())
        frozen["observations"][0]["workload_roots"][0]["preflight_nonce"] = "f" * 64
        path.write_text(json.dumps(frozen))
        entry["environment_observation"]["evidence_sha256"] = collector._sha256_path(path)
    elif mutation == "claim":
        entry["nested"] = {"semantic_claim_ok": True}
    elif mutation == "command":
        options["command"] = "changed"
    elif mutation == "contract":
        options["command_contract_sha256"] = "f" * 64
    elif mutation == "duration":
        options["duration_s"] = 60
    elif mutation == "cohort":
        chain["cohort_id"] = entry["campaign_cohort"] = OWNER
    (root / "energy_summary.json").write_text(json.dumps(entry))
    with pytest.raises(ValueError):
        task_budget.read_verified_campaign_repeat(chain, **options)


def test_diagnostic_repeat_cannot_be_promoted_by_non_diagnostic_collector_reuse(accepted_repeat):
    rig, chain, options = accepted_repeat
    original = (Path(chain["run_directory"]) / "energy_summary.json").read_bytes()
    with pytest.raises(ValueError, match="diagnostic campaign repeat cannot be reused for non-diagnostic"):
        _measure(rig, "scientific-promotion", diagnostic_only=False, run_count=1,
                 exact_run_count=True, reuse_campaign_repeats=True)
    assert rig.counts() == {"preflight": 1, "collector": 1, "workload": 1}
    assert (Path(chain["run_directory"]) / "energy_summary.json").read_bytes() == original


@pytest.mark.parametrize("mutation", ["parent", "row", "hash", "cohort", "count"])
def test_native_environment_import_keeps_exact_three_run_row_and_hash_binding(tmp_path, monkeypatch, mutation):
    _local_environment(monkeypatch)
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    result = _measure(rig)
    entries = copy.deepcopy(result["runs"])
    if mutation == "parent":
        entries[0]["campaign_run_id"] = "foreign"
    elif mutation == "row":
        entries[0]["campaign_row"] = ROW.replace("b001", "b999")
    elif mutation == "hash":
        entries[0]["environment_observation"]["evidence_sha256"] = "f" * 64
    elif mutation == "cohort":
        entries[0]["campaign_cohort"] = "foreign"
    else:
        entries.pop()
    ok, fields = _cohort_environment_fields(_plan(), entries)
    assert ok is False and fields["environment_observation_status"] == "fail"


def test_explicit_cohort_diagnostic_remains_ineligible_for_repeat_and_import(tmp_path, monkeypatch):
    _local_environment(monkeypatch)
    rig = campaign_rig(tmp_path, monkeypatch, ["success"])
    result = _measure(rig)
    entries = copy.deepcopy(result["runs"])
    for entry in entries:
        entry["campaign_cohort"] = OWNER
        assert collector._energy_repeat_aggregation_eligible(entry, require_ab=False) is False
    ok, fields = _cohort_environment_fields({**_plan(), "campaign_cohort": OWNER}, entries)
    assert ok is False and fields["environment_observation_status"] == "fail"
