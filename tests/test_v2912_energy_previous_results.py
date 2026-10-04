"""Previous-results admits a new plan only for positively unstarted history."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys

import pytest

import test_v2712_energy_resume as fixture
from onnx_splitpoint_tool.workflow.checkpoints import AtomicRowJournal


COUNTERS = (
    "measurement_wrapper_started_count", "started_measurement_count",
    "collector_started_repeat_count", "workload_started_repeat_count",
)
IDENTITY = ("deepx_to_trt", "mobilenet_v3_large", "b001", "orin_nx_deepx_m1_01")


def _campaign_args(root):
    return (" --campaign-budget-file " + shlex.quote(str(root / "campaign.json"))
            + " --campaign-row-id row-a --campaign-repeats 3"
            + " --campaign-max-retries 2 --campaign-max-transport-failures 20")


def _row(base: Path, *, attempt: str, identity=IDENTITY):
    row = fixture._plan_row(
        base, backend=identity[0], model=identity[1], case=identity[2],
        setup_id=identity[3], attempt=attempt,
    )
    row.update(measurement_profile_requested_repeats=3, measurement_effective_repeats=3,
               measurement_repeat_expansion_applied=False,
               measurement_repeat_expansion_reason="none")
    return row


def _plan(rows, *, attempt):
    return {**fixture._plan(rows, attempt=attempt),
            "energy_profile_requested_runs_per_row": 3, "energy_effective_runs_per_row": 3,
            "energy_repeat_expansion_applied": False, "energy_repeat_expansion_reason": "none"}


def _history(tmp_path: Path):
    namespace = fixture._namespace()
    old = _row(tmp_path / "previous/measurements/old", attempt="old-plan")
    fresh = _row(tmp_path / "current/measurements/new", attempt="new-plan")
    for row in (old, fresh):
        row["measure_command"] += _campaign_args(tmp_path)
    old_plan = _plan([old], attempt="old-plan")
    plan = _plan([fresh], attempt="new-plan")
    previous = {
        "ok": False, "complete": False, "requested": True, "dry_run": False,
        "status": "blocked_energy_plan_cohort_invalid",
        "blocked_reason": "energy_plan_cohort_contract_invalid",
        **{field: 0 for field in COUNTERS},
        "plan_payload": old_plan,
        "rows": [{"row": copy.deepcopy(old), "ok": False,
                  "skipped": "blocked_before_measurement",
                  "blocked_reason": "energy_plan_cohort_contract_invalid",
                  "measurement_started": False}],
    }
    command = ["python", "energy_plan.py", "--out-dir", str(tmp_path / "previous/plan")]
    for field, value in fixture._EXECUTION_CONTEXT.items():
        command += ["--" + field.replace("_", "-"), value]
    previous["plan"] = {"cmd": command}
    previous["result_ledger"] = namespace["_result_ledger_from_results"](previous["rows"])
    previous["result_ledger_valid"] = True
    # Reproduce the old plan alias: its script now references another attempt's
    # local contract. The frozen report and its technical hashes are unchanged.
    path = Path(old["preflight_command_file"])
    original_preflight = path.read_text()
    path.write_text(original_preflight.replace(old["command_contract_file"], "/later/attempt/contract.json"))
    source = tmp_path / "previous/native_producer_energy_results.json"
    source.write_text(json.dumps(previous))
    return namespace, previous, plan, source, original_preflight


def _files(root: Path):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


def _call(namespace, previous, plan, **kwargs):
    return namespace["_supplement_previous_results"](previous, plan, **kwargs)


def _journal(namespace, previous, source):
    rows = previous["plan_payload"]["rows"]
    plan_hash, identities, row_hashes = namespace["_managed_journal_contract"](
        rows, previous["plan_payload"],
    )
    journal = AtomicRowJournal.create(
        source.parent / "checkpoints/native_energy", plan_hash=plan_hash,
        rows=rows, identities=identities, row_contract_hashes=row_hashes,
    )
    journal.mark_run_state("failed")
    return journal


def test_unstarted_old_alias_allows_extended_current_cohort_without_reuse(tmp_path):
    namespace, previous, plan, _source, _ = _history(tmp_path)
    added = _row(tmp_path / "current/measurements/full", attempt="new-plan",
                 identity=("native_full_tensorrt", "yolo26s", "full", "orin_nx_deepx_m1_01"))
    plan["rows"].append(added)
    before = copy.deepcopy(previous), copy.deepcopy(plan), _files(tmp_path)
    with pytest.raises(ValueError, match="not bound to its local contract"):
        namespace["_validate_resume_remote_commands"](
            previous["plan_payload"]["rows"][0], plan["rows"][0], IDENTITY,
        )
    assert _call(namespace, previous, plan) == {}
    assert (previous, plan, _files(tmp_path)) == before


@pytest.mark.parametrize("field", COUNTERS)
@pytest.mark.parametrize("value", [None, False, 1])
def test_missing_false_or_positive_start_counter_cannot_claim_nullstart(tmp_path, field, value):
    namespace, previous, plan, _, _ = _history(tmp_path)
    if value is None:
        previous.pop(field)
    else:
        previous[field] = value
    with pytest.raises(ValueError, match="remote command drift"):
        _call(namespace, previous, plan)


@pytest.mark.parametrize("evidence", ["partial", "complete", "started_failure", "empty_run"])
def test_zero_summary_does_not_hide_started_or_partial_result(tmp_path, evidence):
    namespace, previous, plan, _, _ = _history(tmp_path)
    row = previous["rows"][0]["row"]
    if evidence == "complete":
        previous["rows"][0] = fixture._complete_result(row, marker="preserve")
    else:
        run = {} if evidence == "empty_run" else {
            "rc": 1, "measurement_start_observation": {"measurement_started": True},
        }
        if evidence == "partial":
            run["energy_aggregate"] = fixture._aggregate(
                Path(row["measurement_output_dir"]), setup_id=row["setup_id"],
                run_id=row["measurement_run_id"], valid=2,
            )
        previous["rows"][0]["run"] = run
    before = copy.deepcopy(previous)
    with pytest.raises(ValueError, match="remote command drift"):
        _call(namespace, previous, plan)
    assert previous == before


@pytest.mark.parametrize("drift", ["old_result_precision", "current_precision", "old_result_comparison",
                                   "current_comparison", "current_hash", "context", "campaign"])
def test_nullstart_keeps_source_measurement_and_execution_contracts_strict(tmp_path, drift):
    namespace, previous, plan, _, _ = _history(tmp_path)
    if drift == "old_result_precision":
        previous["rows"][0]["row"]["precision"] = "fp32"
    elif drift == "current_precision":
        plan["rows"][0]["precision"] = "fp32"
    elif drift == "old_result_comparison":
        previous["rows"][0]["row"]["comparison_backend"] = "hailo8"
    elif drift == "current_comparison":
        plan["rows"][0]["comparison_backend"] = "hailo8"
    elif drift == "current_hash":
        plan["rows"][0]["successful_command_contract_sha256"] = "b" * 64
    elif drift == "context":
        plan["execution_context"]["deepx_ssh"] = "nx@foreign"
    else:
        plan["rows"][0]["measure_command"] = plan["rows"][0]["measure_command"].replace(
            "--campaign-repeats 3", "--campaign-repeats 4",
        )
    with pytest.raises(ValueError, match="(contract drift|execution context drift)"):
        _call(namespace, previous, plan)


@pytest.mark.parametrize("bad", ["preflight_unbound", "workload_missing", "workload_symlink", "measure_binding"])
def test_current_commands_are_still_validated_for_unstarted_previous_plan(tmp_path, bad):
    namespace, previous, plan, _, _ = _history(tmp_path)
    row = plan["rows"][0]
    if bad == "preflight_unbound":
        Path(row["preflight_command_file"]).write_text("#!/bin/sh\nexit 0\n")
    elif bad == "workload_missing":
        Path(row["command_file"]).unlink()
    elif bad == "workload_symlink":
        path = Path(row["command_file"])
        target = path.with_suffix(".original")
        path.rename(target)
        path.symlink_to(target)
    else:
        row["measure_command"] = row["measure_command"].replace(
            str(row["command_file"]), "/foreign/workload.sh",
        )
    with pytest.raises(ValueError, match="remote command drift"):
        _call(namespace, previous, plan)


@pytest.mark.parametrize("representation", ["plan", "result"])
@pytest.mark.parametrize("field", ["measurement_output_dir", "measurement_output_base_dir"])
def test_physical_attempt_output_contradicts_nullstart(tmp_path, representation, field):
    namespace, previous, plan, _, _ = _history(tmp_path)
    row = previous["plan_payload"]["rows"][0] if representation == "plan" else previous["rows"][0]["row"]
    output = tmp_path / f"physical-{representation}-{field}"
    row[field] = str(output)
    repeat = output / "attempt_saved/run_000"
    repeat.mkdir(parents=True)
    (repeat / "energy_summary.json").write_text('{"collector_started":true}')
    before = _files(tmp_path)
    with pytest.raises(ValueError, match="remote command drift"):
        _call(namespace, previous, plan)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("chain", [
    {"collector_started": False, "finished": False},
    {"collector_started": True, "finished": True, "valid": False},
    {"collector_started": True, "finished": True, "valid": True, "logical_repeat": 0},
])
def test_existing_campaign_chain_prevents_nullstart_exception(tmp_path, chain):
    namespace, previous, plan, _, _ = _history(tmp_path)
    budget = tmp_path / "campaign.json"
    budget.write_text(json.dumps({"campaign": True, "tasks": {"row-a": {"chains": [chain]}}}))
    before = budget.read_bytes()
    with pytest.raises(ValueError, match="remote command drift"):
        _call(namespace, previous, plan)
    assert budget.read_bytes() == before


@pytest.mark.parametrize('task', [None, {}, {'chains': None}])
def test_malformed_existing_campaign_task_is_not_absent(tmp_path, task):
    namespace, previous, plan, _, _ = _history(tmp_path)
    budget = tmp_path / 'campaign.json'
    budget.write_text(json.dumps({'campaign': True, 'tasks': {'row-a': task}}))
    before = budget.read_bytes()
    with pytest.raises(ValueError, match='remote command drift'):
        _call(namespace, previous, plan)
    assert budget.read_bytes() == before


def test_complete_measured_result_keeps_exact_aggregate_and_provenance(tmp_path):
    namespace, previous, plan, _, original_preflight = _history(tmp_path)
    old_row = previous["plan_payload"]["rows"][0]
    Path(old_row["preflight_command_file"]).write_text(original_preflight)
    item = fixture._complete_result(old_row, marker="original-source")
    aggregate = Path(old_row["measurement_output_dir"]) / "energy_aggregate.json"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(json.dumps(item["run"]["energy_aggregate"]))
    item["run"].update(energy_aggregate_path=str(aggregate),
                       energy_aggregate_sha256=hashlib.sha256(aggregate.read_bytes()).hexdigest())
    previous.update(rows=[item], status="completed", started_measurement_count=1)
    before = copy.deepcopy(previous), _files(tmp_path)
    assert _call(namespace, previous, plan) == {IDENTITY: item}
    assert (previous, _files(tmp_path)) == before
    aggregate.write_text("changed original aggregate")
    with pytest.raises(ValueError, match="aggregate missing or changed"):
        _call(namespace, previous, plan)


@pytest.mark.parametrize("running", [False, True])
def test_authoritative_row_journal_overrules_stale_manifest_read_only(tmp_path, running):
    namespace, previous, plan, source, _ = _history(tmp_path)
    journal = _journal(namespace, previous, source)
    if running:
        old_manifest = journal.manifest_path.read_bytes()
        journal.transition(0, "running", execution={"phase": "measurement_child_starting",
                                                  "measurement_wrapper_started": False})
        # Real commit order: authoritative row changed, derived manifest lagged.
        journal.manifest_path.write_bytes(old_manifest)
    before = _files(tmp_path)
    if running:
        with pytest.raises(ValueError, match="(remote command drift|not.started|journal)"):
            _call(namespace, previous, plan)
    else:
        assert _call(namespace, previous, plan) == {}
    assert _files(tmp_path) == before


def test_previous_results_main_passes_resume_checks_before_measurement_leaf(tmp_path, monkeypatch):
    namespace, previous, _plan, source, _ = _history(tmp_path)
    _journal(namespace, previous, source)
    (tmp_path / "campaign.json").write_text(json.dumps({
        "campaign": True, "tasks": {"row-a": {"chains": [], "stop_reason": "", "source_id": "fixture"}},
        "sources": {"fixture": {"stop_reason": ""}},
    }))
    summary = tmp_path / "summary.json"
    summary.write_text('{"rows":[]}')
    out = tmp_path / "current-main"
    def with_campaign(row, _index, _plan_root):
        row["measure_command"] += _campaign_args(tmp_path)
        row.update(measurement_profile_requested_repeats=3, measurement_effective_repeats=3,
                   measurement_repeat_expansion_applied=False,
                   measurement_repeat_expansion_reason="none")
        return row
    original_plan = fixture._plan
    monkeypatch.setattr(fixture, "_plan", lambda *a, **kw: {
        **original_plan(*a, **kw), "energy_profile_requested_runs_per_row": 3,
        "energy_effective_runs_per_row": 3, "energy_repeat_expansion_applied": False,
        "energy_repeat_expansion_reason": "none",
    })
    calls = fixture._install_fake_runner(
        namespace, out=out, old_plan=previous["plan_payload"], row_transform=with_campaign,
    )
    observed = []
    namespace["main"].__globals__["_prepare_resume_measurement_cohort"] = (
        lambda rows, **kwargs: observed.append(copy.deepcopy(rows)) or
        {"ok": True, "measurement_wrapper_allowed": True}
    )
    argv = ["runner", "--summary", str(summary), "--out-dir", str(out),
            "--previous-results", str(source)]
    for field, value in fixture._EXECUTION_CONTEXT.items():
        argv += ["--" + field.replace("_", "-"), value]
    monkeypatch.setattr(sys, "argv", argv)
    before = _files(source.parent)
    assert namespace["main"]() == 0
    assert calls == ["|".join(IDENTITY)]
    assert len(observed) == 1 and len(observed[0]) == 1
    report = json.loads((out / "native_producer_energy_results.json").read_text())
    assert report["supplement_reused_row_count"] == 0
    assert report["started_measurement_count"] == 1
    assert _files(source.parent) == before
