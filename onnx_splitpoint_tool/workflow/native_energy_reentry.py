"""Select scoped normal A4 Resume attempts while preserving predecessors.

Original managed checkpoints remain immutable. Physical admission continues
through the existing campaign budget and Native capture callback.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from .checkpoints import AtomicRowJournal, load_stage_checkpoint
from ..energy.task_budget import campaign_budget_stop_reason


def _read(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError("Native energy predecessor is not a canonical regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Native energy predecessor is not an object")
    return value


def selected_energy_view(prepared):
    """Resolve the real normal report destination from its saved dispatch."""
    original = Path(prepared["output"]) / "energy_view"
    dispatch_path = Path(prepared["output"]) / "energy_dispatch.json"
    if not dispatch_path.exists():
        return original
    dispatch = _read(dispatch_path)
    command = dispatch.get("command") or []
    if not command and not dispatch.get("pre_capture_predecessor"):
        return original
    if command.count("--out-dir") != 1:
        raise ValueError("Native energy predecessor invocation lacks its output binding")
    selected = Path(command[command.index("--out-dir") + 1])
    measurement = original / "reports/native_energy_measurements"
    if selected == measurement:
        return original
    attempt = selected.parent.parent
    predecessor = Path(dispatch.get("pre_capture_predecessor") or "")
    if (prepared.get("deepx_runtime_mode") != "A4"
            or prepared["identity"]["backend"] not in {"deepx_to_trt", "native_full_deepx"}
            or selected != attempt / "reports/native_energy_measurements"
            or attempt.parent != measurement / "resume_attempts"
            or not attempt.name.startswith("resume_") or attempt.resolve() != attempt
            or not predecessor.is_relative_to(measurement)
            or predecessor.resolve() != predecessor
            or predecessor.name not in {"native_producer_energy_results.json", "native_producer_energy_results.partial.json"}
            or not predecessor.is_file() or predecessor.is_symlink()):
        raise ValueError("Native energy selected attempt escapes its original point or predecessor")
    return attempt


def select_energy_view(prepared, context, original_view, command_hash):
    """Return the selected normal view and its preserved predecessor result."""
    original_view = Path(original_view)
    if (context.get("resume") is not True or prepared.get("deepx_runtime_mode") != "A4"
            or not context.get("finite_config") or prepared["identity"]["backend"] not in {
                "deepx_to_trt", "native_full_deepx"}):
        return original_view, None
    from ..resources.remote_scripts import run_native_producer_energy_from_summary as runner

    measurement = original_view / "reports/native_energy_measurements"
    dispatch_path = Path(prepared["output"]) / "energy_dispatch.json"
    previous_results = None
    if dispatch_path.exists():
        dispatch = _read(dispatch_path)
        selected_view = selected_energy_view(prepared)
        if selected_view != original_view:
            original_view = selected_view
            measurement = original_view / "reports/native_energy_measurements"
            previous_results = Path(dispatch["pre_capture_predecessor"])
    stage_path = measurement / "stages/native_energy/stage_result.json"
    if not stage_path.exists():
        _continue_sample_loss_view(prepared, context, original_view, command_hash)
        return original_view, previous_results
    saved_stage = _read(stage_path)
    stage, reason = load_stage_checkpoint(stage_path, stage="native_energy",
        input_hash=saved_stage.get("input_hash"), run_root=measurement)
    if stage is None:
        raise ValueError("Native energy predecessor checkpoint invalid: " + reason)
    if stage["state"] not in {"failed", "running"}:
        return original_view, previous_results
    source = measurement / "native_producer_energy_results.json"
    if not source.exists():
        source = source.with_name("native_producer_energy_results.partial.json")
    if not source.exists():
        _continue_sample_loss_view(prepared, context, original_view, command_hash)
        return original_view, previous_results
    previous = _read(source)
    zero_start = all(type(previous.get(key)) is int and previous[key] == 0 for key in (
            "started_measurement_count", "collector_started_repeat_count", "workload_started_repeat_count"))
    plan = previous.get("plan_payload") or {}
    rows = plan.get("rows") or []
    if (len(rows) != 1 or any(rows[0].get(key) != value for key, value in prepared["identity"].items())
            or rows[0].get("deepx_runtime_mode") != "A4" or rows[0].get("deepx_max_inflight") != 4
            or rows[0].get("successful_command_contract_sha256") != command_hash
            or rows[0].get("duration_s") != 60
            or rows[0].get("measurement_requested_repeats") != 3
            or rows[0].get("measurement_effective_repeats") != 3
            or plan.get("physical_scope") != "FS" or plan.get("window_label") != "command"
            or plan.get("require_runtime_work_units") is not True
            or plan.get("require_command_window_alignment") is not True):
        raise ValueError("Native energy predecessor changed its accepted performance contract")
    plan_hash, identities, row_hashes = runner._managed_journal_contract(rows, plan)
    journal = AtomicRowJournal.open_for_resume(measurement / "checkpoints/native_energy",
        plan_hash=plan_hash, identities=identities, row_contract_hashes=row_hashes, refresh_manifest=False)
    checkpoint = journal.rows[0]
    if not zero_start:
        return _select_sample_loss_view(prepared, context, original_view, source, checkpoint,
                                        measurement / "checkpoints/native_energy/rows" / journal._row_filename(0, identities[0]),
                                        command_hash, previous_results)
    known_plan_error = (checkpoint["state"] == "not_started" and checkpoint["attempt_count"] == 0
        and previous.get("status") == "blocked_energy_plan_cohort_invalid"
        and [item.get("error") for item in previous.get("cohort_validation_errors") or []]
        == ["ValueError: energy_quality_admission_precision_drift"])
    result = checkpoint.get("result") or {}
    run = result.get("run") or {}
    error_text = str(run.get("stdout") or run.get("stdout_tail") or "")
    known_claim_error = (checkpoint["state"] == "failed" and run.get("rc") == 1
        and not run.get("cancelled") and not run.get("timed_out")
        and "ValueError: final cohort requires non-diagnostic exact command-window task measurement" in error_text)
    if not (known_plan_error or known_claim_error):
        return original_view, previous_results
    if known_claim_error:
        observation = run.get("measurement_start_observation") or {}
        if (observation.get("measurement_started") is not False
                or any(type(observation.get(key)) is not int or observation[key] != 0 for key in (
                    "collector_started_repeat_count", "workload_started_repeat_count"))):
            raise ValueError("Native energy predecessor lacks exact zero-start evidence")
    execution = checkpoint.get("execution") or {}
    output = execution.get("output_dir")
    if output:
        output = Path(output)
        if (not output.is_absolute() or not output.is_relative_to(measurement)
                or output.resolve() != output or (output.exists() and any(output.iterdir()))):
            raise ValueError("Native energy predecessor has physical or unknown attempt output")
    budget = _read(context["budget_file"])
    row_id = json.dumps([prepared["identity"][key] for key in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    source_id = prepared["target_binding"]["urecs_address"]
    task = budget["tasks"][row_id]
    if (task.get("chains") != [] or campaign_budget_stop_reason(budget, row_id, context.get("environment"),
            source_id=source_id, parent_run_dir=context["run_dir"])):
        raise ValueError("Native energy pre-capture repair cannot bypass physical history or STOP")
    attempt = Path(prepared["output"]) / "energy_view/reports/native_energy_measurements/resume_attempts" / ("resume_" + uuid.uuid4().hex)
    return attempt, source


def _select_sample_loss_view(prepared, context, view, source, checkpoint, checkpoint_path,
                             command_hash, previous_results):
    """Explicit normal Resume keeps accepted slots and starts one bounded series."""
    from ..energy.task_budget import authorize_native_resume_capture_series, read_verified_campaign_repeat
    budget = _read(context["budget_file"])
    row_id = json.dumps([prepared["identity"][key] for key in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    task = budget["tasks"][row_id]
    history = task.get("resume_capture_series") or []
    existing = history[-1] if history else None
    # An interrupted pending selected attempt continues its original series.
    # No failed managed checkpoint is reopened or relabeled.
    if checkpoint["state"] != "failed":
        if existing is not None and existing["selected_view"] == str(view):
            authorize_native_resume_capture_series(context["budget_file"], row_id=row_id,
                command_contract_sha256=command_hash,
                predecessor_result=existing["predecessor_result"]["path"],
                predecessor_checkpoint=existing["predecessor_checkpoint"]["path"],
                selected_view=view, environment=context.get("environment"))
        return view, previous_results
    valid = {c["logical_repeat"] for c in task.get("chains") or []
             if c.get("valid") is True and c.get("cohort_id") == Path(context["run_dir"]).name}
    if task.get("stop_reason") != "task_logical_retry_limit" or len(valid) >= 3:
        return view, previous_results
    previous = _read(source)
    if checkpoint.get("result") != (previous.get("rows") or [None])[0]:
        raise ValueError("Native sample-loss predecessor differs from its immutable managed result")
    command_path = Path(previous["plan_payload"]["rows"][0]["command_file"])
    energy_root = Path(prepared["output"]) / "energy_view"
    if (command_path.resolve() != command_path or command_path.is_symlink()
            or not command_path.is_file() or not command_path.is_relative_to(energy_root)):
        raise ValueError("Native sample-loss predecessor command escapes its normal energy view")
    command = command_path.read_text(encoding="utf-8").strip()
    for chain in task["chains"]:
        if chain.get("valid") is True:
            read_verified_campaign_repeat(chain, command=command, command_contract_sha256=command_hash,
                                          duration_s=60, require_ab=False)
    attempt = Path(prepared["output"]) / "energy_view/reports/native_energy_measurements/resume_attempts" / ("resume_" + uuid.uuid4().hex)
    series = authorize_native_resume_capture_series(context["budget_file"], row_id=row_id,
        command_contract_sha256=command_hash, predecessor_result=source,
        predecessor_checkpoint=checkpoint_path, selected_view=attempt, environment=context.get("environment"))
    return Path(series["selected_view"]), source


def _continue_sample_loss_view(prepared, context, view, command_hash):
    from ..energy.task_budget import authorize_native_resume_capture_series
    row_id = json.dumps([prepared["identity"][key] for key in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    history = (_read(context["budget_file"])["tasks"][row_id].get("resume_capture_series") or [])
    if history and history[-1]["selected_view"] == str(view):
        series = history[-1]
        authorize_native_resume_capture_series(context["budget_file"], row_id=row_id,
            command_contract_sha256=command_hash, predecessor_result=series["predecessor_result"]["path"],
            predecessor_checkpoint=series["predecessor_checkpoint"]["path"], selected_view=view,
            environment=context.get("environment"))
