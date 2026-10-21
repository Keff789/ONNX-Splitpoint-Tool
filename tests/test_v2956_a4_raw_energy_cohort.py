"""Finite A4 capture admission keeps raw energy separate from quality claims.

External process leaves and DUT telemetry are local fixtures. The ordinary
collector, final cohort budget, capture callback and evidence readers run.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy import collector, environment_observer, task_budget
from onnx_splitpoint_tool.native_energy_reporting import _cohort_environment_fields, collect_native_energy
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from test_v27923_fs_calibration_gate import _setup
from test_v283_campaign_energy_budget import campaign_rig, measure
from test_yolo_energy_partial_reporting import publish


OWNER = "actual-a4-parent"
HOOK = "onnx_splitpoint_tool.workflow.hardware_acceptance_native:admit_energy_capture"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": collector._sha256_path(path)}


def rig(tmp_path, monkeypatch, backend="deepx_to_trt"):
    root = tmp_path / OWNER
    root.mkdir()
    instance = campaign_rig(root, monkeypatch, ["success"])
    instance.setup = _setup(root)
    instance.defaults = replace(instance.defaults, physical_scope="FS")
    identity = {"backend": backend, "model": "mobilenet_v3_large",
                "case": "full" if backend == "native_full_deepx" else "b027", "setup_id": "test_fs"}
    row = json.dumps([identity[key] for key in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    prepared = {"schema": "onnx-splitpoint/hardware-acceptance-native-plan/v1", "identity": identity,
                "deepx_runtime_mode": "A4", "output": str(root / "native"),
                "target_binding": {"urecs_address": instance.setup.urecs_address},
                "energy_recipe": {"duration_s": 60, "repeats": 3, "warmup": 0, "frames_override": 0,
                                  "physical_scope": "FS", "window": "whole_command"}}
    descriptor = {"schema": "onnx-splitpoint/hardware-acceptance-native-capture-admission/v1",
                  "run_dir": str(root), "phase_matrix": {}, "command_contract_sha256": "a" * 64,
                  "prepared": write(root / "native/PREPARED.json", prepared)}
    binding = write(root / "native/CAPTURE_ADMISSION.json", descriptor)
    checkpoint = root / "campaign.json"
    write(checkpoint, {"campaign": True, "tasks": {},
                       "sources": {instance.setup.urecs_address: {"stop_reason": ""}}, "max_transport_failures": 20})
    task_budget.authorize_final_cohort(checkpoint, cohort_id=OWNER, parent_run_dir=root,
        row_ids=[row], source_id=instance.setup.urecs_address,
        authorization={"purpose": "final", "repeats": 3, "row_ids": [row],
                       "generation": OWNER, "explicit_user_request": {"fixture": True}})
    journal = root / "journal"
    write(journal / "controller.resources.json", {"run_dir": str(root)})
    instance.env["ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR"] = str(journal)
    monkeypatch.setattr(task_budget, "_budget_parent", lambda environment=None: {"run_id": OWNER})

    class Observer:
        def __init__(self, setup, output, environment, run_id):
            assert run_id == OWNER
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(environment_observer, "EnergyEnvironmentObserver", Observer)
    validate = environment_observer.validate_environment

    def sample(directory, entry, *, freeze=False):
        directory = Path(directory)
        trace = entry.get("command_window_trace_binding") or {}
        start, end = trace.get("command_start_realtime_ns"), trace.get("command_end_realtime_ns")
        if freeze and start and end:
            preflight = entry["preflight_evidence"]
            write_sample = {"phase": "during", "controller_request_start_ns": start,
                "controller_request_end_ns": end, "setup_id": identity["setup_id"], "run_id": OWNER,
                "capture_group": str(directory.parent), "cpu_frequency": [{"scaling_cur_freq": 1234}],
                "host": {"loadavg": [0.1, 0.1, 0.1]}, "workload_roots": [{
                    "preflight_nonce": preflight["nonce"],
                    "command_contract_sha256": preflight["expected_command_contract_sha256"],
                    "workers": [{"pid": 123, "start_ticks": "456", "allowed_cpus": [0],
                                 "threads": [{"tid": 123, "allowed_cpus": [0]}]}]}]}
            (directory / "environment_samples.jsonl").write_text(json.dumps(write_sample) + "\n")
        return validate(directory, entry, freeze=freeze)

    monkeypatch.setattr(environment_observer, "validate_environment", sample)
    # Only physical phase inventory is external to this portable process rig.
    # The real callback still validates its descriptor and every collector request.
    monkeypatch.setattr(native, "_native_phase_debt", lambda *args, **kwargs: {
        "campaign_row_id": row, "campaign_source_id": instance.setup.urecs_address,
        "prior_and_current_chains": {f"repeat:{i}": 0 for i in range(3)}})
    # The callback binds the existing ordinary energy budget filename.
    budget = root / "energy/energy_task_budget.json"
    budget.parent.mkdir()
    checkpoint.rename(budget)
    options = dict(row=row, campaign_budget_file=budget, campaign_cohort=OWNER,
        campaign_max_retries=2, campaign_max_transport_failures=20, invalid_repeat_max_retries=2,
        diagnostic_only=True, claim_exclusion_reason="quality_or_pairing_not_claim_eligible",
        require_environment_observation=True, environment_run_id=OWNER,
        capture_admission_file=binding["path"], capture_admission_sha256=binding["sha256"],
        capture_admission_hook=HOOK, duration_s=60, physical_scope="FS")
    return instance, options, identity


def capture(instance, options, name="first", **overrides):
    return measure(instance, "native/energy_view/" + name, **{**options, **overrides})


@pytest.mark.parametrize("backend", ["deepx_to_trt", "native_full_deepx"])
def test_admitted_a4_three_raw_repeats_reuse_and_normal_import_keep_claims_false(tmp_path, monkeypatch, backend):
    instance, options, identity = rig(tmp_path, monkeypatch, backend)
    result = capture(instance, options)
    assert result["ok"] and result["repeat_contract_complete"]
    assert instance.counts() == {"preflight": 3, "collector": 3, "workload": 3}
    assert result["task_budget"]["counts"]["valid_logical_repeats"] == 3
    assert collector._clamp_diagnostic_claim_fields(result) == result
    assert result["diagnostic_only"] is True and result["claim_eligible"] is False
    original = {}
    for entry, chain in zip(result["runs"], result["task_budget"]["chains"]):
        assert entry["energy_configured_workload_duration_s"] == 60
        assert entry["environment_observation"]["ok"] is True
        assert entry["full_system_current_scale_verified"] is True
        assert entry["runtime_work_unit_gate_status"] == "pass"
        assert task_budget.native_a4_diagnostic_cohort_repeat(entry)
        verified = task_budget.read_verified_campaign_repeat(chain, command=instance.command,
            command_contract_sha256="a" * 64, duration_s=60, require_ab=False)
        assert verified["diagnostic_only"] is True and verified["scientific_primary_claim_eligible"] is False
        path = Path(chain["run_directory"]) / "energy_summary.json"
        original[path] = path.read_bytes()
    plan = {**identity, "campaign_cohort": OWNER, "diagnostic_only": True,
            "claim_eligible": False, "energy_quality_qualified": False, "measurement_effective_repeats": 3,
            "deepx_runtime_mode": "A4", "deepx_max_inflight": 4}
    publish(instance.root / "projection", [{"row": plan, "ok": True, "run": {
        "rc": 0, "energy_aggregate": result, "energy_aggregate_embedded": True,
        "energy_aggregate_complete": True, "energy_aggregate_verified": True}}])
    normalized = collect_native_energy(instance.root / "projection", include_partial=True)[0]
    assert normalized["ok"] is True and normalized["energy_repeat_valid_n"] == 3
    assert normalized["environment_observation_status"] == "pass"
    assert normalized["diagnostic_only"] is True and normalized["claim_eligible"] is False
    assert normalized["energy_quality_qualified"] is False
    reused = capture(instance, options, "reused", reuse_campaign_repeats=True)
    assert reused["ok"] and all(entry.get("campaign_reused_repeat") for entry in reused["runs"])
    assert instance.counts() == {"preflight": 3, "collector": 3, "workload": 3}
    assert all(path.read_bytes() == content for path, content in original.items())


def test_narrow_a4_raw_repeat_rejects_claim_receipt_and_binding_mutations(tmp_path, monkeypatch):
    instance, options, identity = rig(tmp_path, monkeypatch)
    result = capture(instance, options)
    assert result["ok"]
    entry = result["runs"][0]
    plan = {**identity, "campaign_cohort": OWNER, "diagnostic_only": True}
    assert _cohort_environment_fields(plan, result["runs"])[0]
    for field in ("claim_eligible", "energy_efficiency_claim_eligible", "scientific_primary_claim_eligible"):
        for value in (None, True, "false"):
            changed = copy.deepcopy(entry)
            changed[field] = value
            assert not task_budget.native_a4_diagnostic_cohort_repeat(changed)
            assert not collector._energy_repeat_aggregation_eligible(changed, require_ab=False)
            assert not _cohort_environment_fields(plan, [changed, *result["runs"][1:]])[0]
        changed = copy.deepcopy(entry)
        changed.pop(field)
        assert not task_budget.native_a4_diagnostic_cohort_repeat(changed)
    changed = copy.deepcopy(entry)
    changed["nested"] = {"semantic_claim_ok": True}
    assert not task_budget.native_a4_diagnostic_cohort_repeat(changed)
    receipt_path = Path(entry["storage_dir"]).parent / "capture_admission.json"
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    for field, value in (("status", "refused"), ("hook", "unrelated:consumer"),
                         ("request", []), ("result", [])):
        write(receipt_path, {**receipt, field: value})
        assert not task_budget.native_a4_diagnostic_cohort_repeat(entry)
    receipt_path.write_bytes(original)
    for field, value in (("campaign_source_id", "foreign"), ("setup_id", "foreign"),
                         ("campaign_budget_file", str(instance.root / "other-budget")),
                         ("run_directory", str(instance.root / "outside-energy-view"))):
        changed_receipt = copy.deepcopy(receipt)
        changed_receipt["request"][field] = value
        write(receipt_path, changed_receipt)
        assert not task_budget.native_a4_diagnostic_cohort_repeat(entry)
    receipt_path.write_bytes(original)
    descriptor_path = Path(options["capture_admission_file"])
    descriptor_bytes = descriptor_path.read_bytes()
    write(descriptor_path, [])
    assert not task_budget.native_a4_diagnostic_cohort_repeat(entry)
    descriptor_path.write_bytes(descriptor_bytes)
    assert task_budget.native_a4_diagnostic_cohort_repeat(entry)
    with pytest.raises(ValueError, match="diagnostic campaign repeat cannot be reused"):
        capture(instance, options, "promotion", diagnostic_only=False, reuse_campaign_repeats=True)
    assert instance.counts() == {"preflight": 3, "collector": 3, "workload": 3}


@pytest.mark.parametrize("mutation", ["hook", "duration", "prepared_mode", "backend"])
def test_unrelated_diagnostic_cohort_is_rejected_before_start(tmp_path, monkeypatch, mutation):
    instance, options, identity = rig(tmp_path, monkeypatch)
    if mutation == "hook":
        options["capture_admission_hook"] = "unrelated:consumer"
    elif mutation == "duration":
        options["duration_s"] = 59
    else:
        prepared_path = instance.root / "native/PREPARED.json"
        prepared = json.loads(prepared_path.read_text())
        if mutation == "prepared_mode":
            prepared["deepx_runtime_mode"] = "S1"
        else:
            prepared["identity"]["backend"] = "hailo8_to_trt"
            options["row"] = json.dumps(list(prepared["identity"][key] for key in
                ("backend", "model", "case", "setup_id")), separators=(",", ":"))
        descriptor_path = Path(options["capture_admission_file"])
        descriptor = json.loads(descriptor_path.read_text())
        descriptor["prepared"] = write(prepared_path, prepared)
        options["capture_admission_sha256"] = write(descriptor_path, descriptor)["sha256"]
    with pytest.raises(ValueError, match="final cohort requires non-diagnostic"):
        capture(instance, options)
    assert instance.counts() == {"preflight": 0, "collector": 0, "workload": 0}
