"""Explicit reentry of the bound YOLO starter attempts.

The previous terminal checkpoint remains a failed attempt.  New checkpoints
use the existing supplement attempt directory; no measured leaf is reset.
"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Any

from .checkpoints import _strict_json, canonical_json_sha256, load_stage_checkpoint


COORDINATOR = Path("stages/run_native_producers/native_coordinator/stage_result.json")
PERFORMANCE = Path("stages/run_native_producers/native_performance/stage_result.json")
QUALITY_NONSTART = "augmentation startcase quality release failed: no completed Native split quality binding for "
CANCEL_NONSTART = "augmentation startcase quality release failed: workflow_cancelled_before_native_case_dispatch"
COMPLETION_FAILURE = "augmentation Generic/Native technical pair failed: "
# Existing native_producer_final_report input names, not a new result registry.
NATIVE_RAW_PATTERNS = (
    "native_fifo_results.json", "hailo10_native_fifo_e2e_results.json",
    "deepx_native_fifo_e2e_results.json", "deepx_native_fifo_e2e_status.json",
    "deepx_native_fifo_probe.json", "native_fifo_eval_runner*.json",
    "native_hailo10h_producer_e2e_eval*.json", "native_deepx_producer_e2e_eval*.json",
    "native_full_baseline_eval.json",
)
UNPUBLISHED_NATIVE_BASELINE = frozenset({
    "reports/native_expected_matrix.json", "reports/native_producer_summary.json",
})


def _object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("augmentation_reentry_missing_original:" + str(path))
    value = _strict_json(path)
    if not isinstance(value, dict):
        raise ValueError("augmentation_reentry_invalid_original:" + str(path))
    return value


def validate_augmentation_reentry(run: str | Path, plan_path: str | Path,
                                 original_root: str | Path | None = None) -> dict[str, Any]:
    """Validate the actual failed predecessor, without changing any file."""
    run = Path(run).resolve()
    plan_path = Path(plan_path).resolve()
    source = Path(original_root).absolute() if original_root else run
    attempt_root = ""
    if original_root:
        try:
            parts = source.relative_to(run / "jobs/supplement_attempts").parts
        except ValueError as exc:
            raise ValueError("augmentation_reentry_original_attempt_outside_run") from exc
        if (len(parts) != 2 or parts[1] != "original" or parts[0] in {"", ".", ".."}
                or source.resolve() != source):
            raise ValueError("augmentation_reentry_original_attempt_invalid")
        attempt_root = str(source.parent)
    plan = _object(plan_path)
    if (plan.get("cohort_id") != "yolo_min3_capability_augmentation" or plan.get("frozen") is not True
            or Path(str(plan.get("augmentation_run") or "")).resolve() != run):
        raise ValueError("augmentation_reentry_frozen_run_binding_invalid")
    manifest = _object(source / "run_manifest.json")
    if manifest.get("run_id") != run.name:
        raise ValueError("augmentation_reentry_original_run_mismatch")
    stage = _object(source / "reports/native_producer_stage.json")
    if str(stage.get("error") or "").startswith(COMPLETION_FAILURE):
        return _measured_completion_reentry(run, plan_path, plan, source, attempt_root, stage)
    cancelled = stage.get("error") == CANCEL_NONSTART
    coordinator_path, performance_path = COORDINATOR, PERFORMANCE
    ancestor = None
    if cancelled:
        prior = (stage.get("config", {}).get("augmentation_startcase_barrier") or {}).get("reentry")
        # This is the one observed continuation of the original Quality
        # nonstart, not a general retry of cancelled or partially measured work.
        if not isinstance(prior, Mapping) or prior.get("kind") != "augmentation_quality_release_nonstart":
            raise ValueError("augmentation_reentry_cancel_ancestor_invalid")
        prior_root = prior.get("original_root")
        if (not isinstance(prior_root, str) or not prior_root
                or not str(_object(Path(prior_root) / "reports/native_producer_stage.json").get("error") or "").startswith(QUALITY_NONSTART)):
            raise ValueError("augmentation_reentry_cancel_ancestor_not_quality_nonstart")
        ancestor = validate_augmentation_reentry(run, plan_path, prior.get("original_root"))
        if dict(prior) != ancestor or not ancestor.get("attempt_root"):
            raise ValueError("augmentation_reentry_cancel_ancestor_changed")
        native_relative = Path(ancestor["attempt_root"]).relative_to(run) / "native"
        coordinator_path = native_relative / "native_coordinator/stage_result.json"
        performance_path = native_relative / "native_performance/stage_result.json"
        from .supplement import _regular_evidence
        for relative in (coordinator_path, performance_path,
                         native_relative / "native_invocation_config.json"):
            _regular_evidence(source, relative.as_posix())
    raw = _object(source / coordinator_path)
    checkpoint, reason = load_stage_checkpoint(source / coordinator_path, stage="native_coordinator",
        input_hash=str(raw.get("input_hash") or ""), run_root=source)
    if checkpoint is None:
        raise ValueError("augmentation_reentry_predecessor_invalid:" + reason)
    if (checkpoint.get("state") != "failed" or checkpoint.get("complete") is not True
            or checkpoint.get("details", {}).get("return_code") != 2):
        raise ValueError("augmentation_reentry_requires_failed_quality_nonstart")
    records = checkpoint.get("artifacts", [])
    if not any(row.get("path") == "reports/native_producer_stage.json" for row in records):
        raise ValueError("augmentation_reentry_original_stage_unbound")
    error = str(stage.get("error") or "")
    if (stage.get("failure_class") != "augmentation_startcase_barrier"
            or stage.get("status") != "failed" or stage.get("state") != "failed"
            or stage.get("complete") is not True or stage.get("run_id") != run.name
            or error != checkpoint.get("error") or error != stage.get("failure_reason")
            or not (error == CANCEL_NONSTART if cancelled else error.startswith(QUALITY_NONSTART))
            or stage.get("variant_results") != []
            or stage.get("native_energy", {}).get("status") != "not_started_startcase_barrier"):
        raise ValueError("augmentation_reentry_not_premeasurement_quality_failure")
    cfg = _object(source / "reports/native_producer_stage_config.json")
    if stage.get("config") != cfg:
        raise ValueError("augmentation_reentry_original_config_changed")
    barrier = cfg.get("augmentation_startcase_barrier") or {}
    if (cfg.get("native_release_mode") != "per_case"
            or barrier.get("plan_sha256") != canonical_json_sha256(plan)
            or Path(str(barrier.get("plan_path") or "")).resolve() != plan_path):
        raise ValueError("augmentation_reentry_original_plan_changed")
    if cancelled:
        _validate_cancel_invocation(run, source, native_relative, cfg, checkpoint, plan)
    performance_raw = _object(source / performance_path)
    performance, reason = load_stage_checkpoint(source / performance_path, stage="native_performance",
        input_hash=str(performance_raw.get("input_hash") or ""), run_root=source)
    if (performance is None or performance.get("state") != "running"
            or performance.get("complete") is not False or performance.get("artifacts") != []):
        raise ValueError("augmentation_reentry_performance_nonstart_invalid:" + reason)
    # In the initial admission inspect the real leaf namespace.  During this
    # attempt, newly completed leaves must not invalidate its saved predecessor.
    if not original_root and any((run / "reports/native_producer_variants").glob("*/performance_stage_result.json")):
        raise ValueError("augmentation_reentry_native_leaf_already_present")
    if not original_root:
        native_root = run / "native_producers"
        if any(path.is_file() for pattern in NATIVE_RAW_PATTERNS for path in native_root.rglob(pattern)):
            raise ValueError("augmentation_reentry_native_raw_observation_already_present")
    for name in ("native_producer_summary.json", "native_producer_combined_summary.json"):
        path = source / "reports" / name
        if path.exists() and _object(path).get("rows"):
            raise ValueError("augmentation_reentry_original_native_results_present")
    result = {
        "kind": "augmentation_native_cancel_nonstart" if cancelled else "augmentation_quality_release_nonstart",
        "run_dir": str(run), "plan_path": str(plan_path),
        "plan_sha256": canonical_json_sha256(plan), "original_root": str(source),
        "attempt_root": attempt_root,
        "predecessor_checkpoint": str(source / coordinator_path),
        "predecessor_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "predecessor_input_hash": checkpoint["input_hash"],
        "predecessor_error": error,
    }
    if cancelled:
        result["quality_nonstart_ancestor"] = ancestor
        result["predecessor_native_relative"] = native_relative.as_posix()
    return result


def _validate_cancel_invocation(run: Path, source: Path, native_relative: Path,
                                cfg: dict, checkpoint: dict, plan: dict) -> None:
    """Bind the existing invocation and finished Quality through normal readers."""
    from onnx_splitpoint_tool.resources.remote_scripts import run_evalrun_native_producer_variants as coordinator
    from .checkpoints import native_coordinator_input_hash
    from .supplement import completed_quality

    invocation_path = source / native_relative / "native_invocation_config.json"
    invocation = _object(invocation_path)
    expanded, _ = coordinator._bind_evalrun_quality_gate_policy(run, invocation)
    policy_sha = coordinator.AccuracyGatePolicy.from_mapping(expanded.get("quality_gate_policy") or {}).sha256()
    expected_hash = native_coordinator_input_hash(run, invocation_path,
        quality_gate_policy_sha256=policy_sha)
    if checkpoint.get("input_hash") != expected_hash:
        raise ValueError("augmentation_reentry_cancel_invocation_changed")
    leaves = coordinator._per_case_variants(run, expanded, expanded.get("variants") or [])
    expanded.update(variants=leaves, native_split_quality_binding_sets_by_variant={},
                    trt_quality_producer_sets_by_setup={})
    if expanded != cfg:
        raise ValueError("augmentation_reentry_cancel_expanded_config_changed")
    summary_path = source / "quality_management/central_quality_summary.json"
    summary = _object(summary_path)
    saved = summary.get("results") or []
    completed, _ = completed_quality(run)
    actual_by_request = {row["source_request"]: row for row in completed}
    expected = {(str(row.get("model_id")), str(row.get("case_id")), str(row.get("setup_id")))
                for row in plan.get("rows", []) if row.get("selection_role") == "primary"}
    observed = {(str(row.get("model_id")), str(row.get("case_id")),
                 str(row.get("source_setup_id") or row.get("setup_id"))) for row in saved}
    if (not expected or len(saved) != len(expected) or observed != expected
            or any(row.get("status") != "completed" or row.get("technical_status") != "completed"
                   or actual_by_request.get(row.get("source_request")) != row for row in saved)):
        raise ValueError("augmentation_reentry_cancel_quality_incomplete")
    # Pure read/bind: no request export, statistics submission or report writer.
    for leaf in leaves:
        merged = coordinator._merged_variant(cfg, leaf)
        for physical in coordinator._variant_backend_bindings(cfg, leaf):
            selections = coordinator._split_selections_for_variant(run, merged, physical)
            bound = coordinator.load_split_binding_set_from_central_quality_summary(summary_path,
                eval_run_id=run.name, setup_id=physical["setup_id"], selections=selections)
            coordinator._validate_supplied_split_binding_set(bound, eval_run_id=run.name,
                setup_id=physical["setup_id"], selections=selections)


def _measured_completion_reentry(run: Path, plan_path: Path, plan: dict,
                                 source: Path, attempt_root: str, stage: dict) -> dict:
    """Resume only the bound first-pair or two-starter projection failure.

    The saved terminal attempt and its actual Native observations remain the
    authority. This is deliberately separate from either nonstart admission.
    """
    from onnx_splitpoint_tool.resources.remote_scripts import run_evalrun_native_producer_variants as coordinator
    from onnx_splitpoint_tool.native_command_contract import verify_native_command_contract
    from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields
    from .supplement import _regular_evidence, _verify_original_binding

    cfg = _object(source / "reports/native_producer_stage_config.json")
    barrier = cfg.get("augmentation_startcase_barrier") or {}
    prior = barrier.get("reentry")
    completed_pair = isinstance(prior, Mapping) and prior.get("kind") == "augmentation_completed_native_first_generic"
    two_pairs = completed_pair and isinstance(prior.get("completed_pair_ancestor"), Mapping)
    if (not isinstance(prior, Mapping) or prior.get("kind") not in {
            "augmentation_native_cancel_nonstart", "augmentation_completed_native_first_generic"}
            or not prior.get("original_root")):
        raise ValueError("augmentation_measured_reentry_ancestor_invalid")
    # The saved first-pair repair may lead to the second fixed starter. No
    # recursive repair of that two-starter state is admitted.
    prior_source = Path(str(prior["original_root"]))
    prior_stage = _object(prior_source / "reports/native_producer_stage.json")
    if completed_pair:
        parent = (prior_stage.get("config", {}).get("augmentation_startcase_barrier") or {}).get("reentry") or {}
        if (prior.get("two_completed_pairs_ancestor")
                or (two_pairs and (parent != prior["completed_pair_ancestor"]
                    or parent.get("completed_pair_ancestor")
                    or parent.get("kind") != "augmentation_completed_native_first_generic"))
                or (not two_pairs and parent.get("kind") != "augmentation_native_cancel_nonstart")
                or not str(prior_stage.get("error") or "").startswith(COMPLETION_FAILURE)):
            raise ValueError("augmentation_measured_reentry_completed_pair_ancestor_invalid")
    elif prior_stage.get("error") != CANCEL_NONSTART:
        raise ValueError("augmentation_measured_reentry_ancestor_not_cancel_nonstart")
    ancestor = validate_augmentation_reentry(run, plan_path, prior_source)
    if dict(prior) != ancestor or not ancestor.get("attempt_root"):
        raise ValueError("augmentation_measured_reentry_ancestor_changed")
    relative = Path(ancestor["attempt_root"]).relative_to(run) / "native"
    cp_path = relative / "native_coordinator/stage_result.json"
    perf_path = relative / "native_performance/stage_result.json"
    index = {row["path"]: row for row in _object(source / "artifact_index.json").get("artifacts", [])}
    for raw in (cp_path.as_posix(), perf_path.as_posix(), (relative / "native_invocation_config.json").as_posix(),
                "reports/native_producer_stage.json", "reports/native_producer_stage_config.json",
                "reports/native_producer_summary.json", "reports/native_producer_combined_summary.json",
                "reports/native_expected_matrix.json"):
        _verify_original_binding(_regular_evidence(source, raw), raw, index)
    raw = _object(source / cp_path)
    cp, reason = load_stage_checkpoint(source / cp_path, stage="native_coordinator",
        input_hash=str(raw.get("input_hash") or ""), run_root=source)
    if (cp is None or cp.get("state") != "failed" or cp.get("complete") is not True
            or cp.get("details", {}).get("return_code") != 2):
        raise ValueError("augmentation_measured_reentry_checkpoint_invalid:" + reason)
    _validate_cancel_invocation(run, source, relative, cfg, cp, plan)
    perf = _object(source / perf_path)
    checked_perf, reason = load_stage_checkpoint(source / perf_path, stage="native_performance",
        input_hash=str(perf.get("input_hash") or ""), run_root=source)
    if (checked_perf is None or perf.get("state") != "running" or perf.get("complete") is not False
            or perf.get("artifacts") != []):
        raise ValueError("augmentation_measured_reentry_performance_invalid:" + reason)
    if (cfg.get("native_release_mode") != "per_case" or barrier.get("plan_sha256") != canonical_json_sha256(plan)
            or Path(str(barrier.get("plan_path") or "")).resolve() != plan_path
            or barrier.get("technical_terminals") or not barrier.get("startcases")):
        raise ValueError("augmentation_measured_reentry_plan_changed")
    # The only additional state is the two fixed starters after the first
    # completed pair was accepted and the second pair's projection failed.
    count = 2 if two_pairs else 1
    starters = barrier["startcases"][:count]
    keys = [coordinator._augmentation_terminal_key(row) for row in starters]
    frozen = {coordinator._augmentation_terminal_key(row): row for row in plan.get("rows", [])
              if row.get("selection_role") == "primary"}
    if len(keys) != count or len(set(keys)) != count or any(key not in frozen for key in keys):
        raise ValueError("augmentation_measured_reentry_frozen_starter_invalid")
    error = str(stage.get("error") or "")
    executions = stage.get("variant_results") or []
    if (stage.get("failure_class") != "augmentation_startcase_barrier"
            or stage.get("status") != "failed" or stage.get("state") != "failed"
            or stage.get("complete") is not True or stage.get("run_id") != run.name
            or error != cp.get("error") or error != stage.get("failure_reason")
            or not error.startswith(COMPLETION_FAILURE + keys[-1] + ";")
            or len(executions) != count or any(row.get("ok") is not True or row.get("rc") != 0 for row in executions)
            or stage.get("native_energy", {}).get("status") != "not_started_startcase_barrier"):
        raise ValueError("augmentation_measured_reentry_not_first_completion_failure")
    if two_pairs:
        released = (stage.get("augmentation_startcase_barrier") or {}).get("completed_startcases") or {}
        if (set(released) != {keys[0]} or released[keys[0]].get("completion_callback", {}).get("rc") != 0
                or executions[1].get("reused") is True):
            raise ValueError("augmentation_measured_reentry_second_starter_progress_invalid")
        if source == run:
            from onnx_splitpoint_tool.generic_completion_plan import completion_identity
            observed = _object(plan_path.parent.parent / "completion/execution-plan.json").get("rows") or []
            if (len(observed) != 2 or {completion_identity(row.get("identity") or {}) for row in observed}
                    != {completion_identity(frozen[key]) for key in keys}):
                raise ValueError("augmentation_completed_pair_generic_scope_changed")
    expected_leaves = {source / "reports/native_producer_variants" / str(row.get("artifact_namespace") or "")
                       / "performance_stage_result.json" for row in executions}
    if (len(expected_leaves) != count or set((source / "reports/native_producer_variants").glob(
            "*/performance_stage_result.json")) != expected_leaves):
        raise ValueError("augmentation_measured_reentry_unexpected_native_leaves")
    rows = _object(source / "reports/native_producer_summary.json").get("rows") or []
    if (len(rows) != count or {coordinator._augmentation_terminal_key(row) for row in rows} != set(keys)
            or any(row.get("ok") is not True for row in rows)):
        raise ValueError("augmentation_measured_reentry_native_summary_invalid")
    import copy
    projected_cfg = copy.deepcopy(cfg)
    projected_cfg["native_split_quality_binding_sets_by_variant"] = {}
    preserved_leaves, preserved_generic = [], []
    for position, (key, record) in enumerate(zip(keys, executions)):
        starter = frozen[key]
        namespace = str(record.get("artifact_namespace") or "")
        leaf_path = Path("reports/native_producer_variants") / namespace / "performance_stage_result.json"
        if not namespace:
            raise ValueError("augmentation_measured_reentry_unexpected_native_leaves")
        _verify_original_binding(_regular_evidence(source, leaf_path.as_posix()), leaf_path.as_posix(), index)
        leaf = _object(source / leaf_path)
        prepared = leaf.get("details", {}).get("variant") or {}
        execution = leaf.get("details", {}).get("execution") or {}
        leaf_index = prepared.get("frozen_leaf_index")
        if type(leaf_index) is not int or not 0 <= leaf_index < len(projected_cfg.get("variants", [])):
            raise ValueError("augmentation_measured_reentry_leaf_index_invalid")
        projected_cfg["variants"][leaf_index] = prepared
        projected_cfg["native_split_quality_binding_sets_by_variant"][namespace] = prepared.get("native_split_quality_binding_sets_by_setup") or {}
        command = record.get("cmd")
        if two_pairs:
            leaf_coordinator_hash = (ancestor["preserved_native_coordinator_input_hash"]
                                     if position == 0 else cp["input_hash"])
        else:
            leaf_coordinator_hash = ancestor["predecessor_input_hash"] if completed_pair else cp["input_hash"]
        expected_input = canonical_json_sha256({"coordinator_input_hash": leaf_coordinator_hash, "variant": prepared, "cmd": command})
        checked_leaf, reason = load_stage_checkpoint(source / leaf_path, stage="native_case_performance",
            input_hash=expected_input, run_root=source)
        if (checked_leaf is None or leaf.get("state") != "completed" or leaf.get("complete") is not True
                or execution.get("rc") != 0 or execution.get("cmd") != command
                or coordinator._augmentation_leaf_key(cfg, prepared) != key
                or any(record.get(field) != value for field, value in execution.items())):
            raise ValueError("augmentation_measured_reentry_leaf_invalid:" + reason)
        if completed_pair and position == 0 and (record.get("reused") is not True or ancestor.get("preserved_native_leaves") != [{
                "path": leaf_path.as_posix(), "checkpoint_sha256": leaf["checkpoint_sha256"],
                "input_hash": expected_input, "case_key": key}]):
            raise ValueError("augmentation_measured_reentry_completed_native_changed")
        snapshot = Path(str(prepared.get("central_quality_summary") or "")).relative_to(run)
        _verify_original_binding(_regular_evidence(source, snapshot.as_posix()), snapshot.as_posix(), index)
        from .artifacts import sha256_file
        if sha256_file(source / snapshot).removeprefix("sha256:") != prepared.get("quality_snapshot_sha256"):
            raise ValueError("augmentation_measured_reentry_quality_snapshot_changed")
        row = next(row for row in rows if coordinator._augmentation_terminal_key(row) == key)
        raw_path = Path(str(row.get("report") or "")).relative_to(run)
        _verify_original_binding(_regular_evidence(run, raw_path.as_posix()), raw_path.as_posix(), index)
        native = _object(run / raw_path)
        verified, reason = verify_native_command_contract(native.get("native_command_contract"), expected_identity={
            "model": starter["model_id"], "case": starter["case_id"], "setup_id": starter["setup_id"],
            "precision": starter["precision"], "backend": row.get("backend")})
        rates = rate_endpoint_fields(native)
        from onnx_splitpoint_tool.resources.remote_scripts.native_producer_final_report import (
            _repeat_fields, _aggregate_repetitions, _validate_repeat_claim_evidence,
        )
        repeats = _repeat_fields((native.get("endpoint_results") or {}).get("completed_task") or native, native)
        independent, repeat_reasons = _validate_repeat_claim_evidence(repeats)
        projected_repeats = _aggregate_repetitions([{**row, **repeats}])
        if (not independent or len(projected_repeats) != 1
                or projected_repeats[0].get("repetition_records") != row.get("repetition_records")):
            raise ValueError("augmentation_measured_reentry_raw_repetitions_changed:" + ",".join(repeat_reasons))
        if (verified is None or verified.get("eval_run_id") != run.name or native.get("ok") is not True
                or native.get("repetition_count_requested") != 3 or native.get("repetition_count_valid") != 3
                or native.get("repetition_count_attempted") != 3 or native.get("repetition_independence_verified") is not True
                or rates.get("completed_task_work_unit_counts") != [1000, 1000, 1000]
                or row.get("fps_repetition_samples") != rates.get("fps_repetition_samples")
                or row.get("native_command_contract_sha256") != native.get("native_command_contract_sha256")):
            raise ValueError("augmentation_measured_reentry_native_repetitions_invalid:" + reason)
        # Preserve the actual bound observation even when older report aliases
        # already carry a precision/claim conflict. Those flags are not upgraded.
        current = [item for item in _object(run / "reports/native_producer_summary.json").get("rows", [])
                   if coordinator._augmentation_terminal_key(item) == key and item.get("backend") == row.get("backend")]
        measured_fields = ("precision", "report", "source_reports", "native_command_contract_sha256",
            "fps_makespan", "fps_repetition_samples", "repetition_records", "repetition_count_requested",
            "repetition_count_attempted", "repetition_count_valid", "frames", "warmup",
            "completed_task_work_unit_counts", "completed_task_measurement_times_s")
        if len(current) != 1 or any(current[0].get(field) != row.get(field) for field in measured_fields):
            raise ValueError("augmentation_measured_reentry_original_measurement_changed")
        preserved_leaf = {"path": leaf_path.as_posix(), "checkpoint_sha256": leaf["checkpoint_sha256"],
                          "input_hash": expected_input, "case_key": key}
        if two_pairs:
            preserved_leaf["coordinator_input_hash"] = leaf_coordinator_hash
        preserved_leaves.append(preserved_leaf)
        if completed_pair:
            preserved_generic.append(_completed_first_generic(run, plan_path, plan, starter, native,
                allow_additions=two_pairs or source != run))
    if stage.get("config") != projected_cfg:
        raise ValueError("augmentation_measured_reentry_released_config_changed")
    result = {"kind": "augmentation_completed_native_first_generic", "run_dir": str(run),
        "plan_path": str(plan_path), "plan_sha256": canonical_json_sha256(plan),
        "original_root": str(source), "attempt_root": attempt_root,
        "predecessor_checkpoint": str(source / cp_path), "predecessor_checkpoint_sha256": cp["checkpoint_sha256"],
        "predecessor_input_hash": cp["input_hash"], "predecessor_error": error,
        "predecessor_native_relative": relative.as_posix(),
        "cancel_nonstart_ancestor": ancestor["cancel_nonstart_ancestor"] if completed_pair else ancestor,
        "preserved_native_leaves": preserved_leaves}
    if two_pairs:
        result.update(two_completed_pairs_ancestor=ancestor,
            preserved_generic_results=preserved_generic)
    elif completed_pair:
        result.update(completed_pair_ancestor=ancestor, preserved_native_coordinator_input_hash=leaf_coordinator_hash,
            preserved_generic_result=preserved_generic[0])
    return result


def _completed_first_generic(run: Path, plan_path: Path, plan: dict, starter: dict, native: dict,
                             *, allow_additions: bool = False) -> dict:
    """Verify the existing three observations through the unchanged runtime parser."""
    from onnx_splitpoint_tool.generic_completion_plan import completion_identity
    from onnx_splitpoint_tool.generic_completion_reporting import load_valid_repetitions
    completion_root = plan_path.parent.parent / "completion"
    execution = _object(completion_root / "execution-plan.json")
    rows = execution.get("rows") or []
    identities = [completion_identity(row.get("identity") or {}) for row in rows]
    allowed = {completion_identity(row) for row in plan.get("rows", []) if row.get("selection_role") == "primary"}
    identity = completion_identity(starter)
    matches = [row for row, observed in zip(rows, identities) if observed == identity]
    if (not rows or len(identities) != len(set(identities)) or not set(identities).issubset(allowed)
            or len(matches) != 1 or (not allow_additions and len(rows) != 1)):
        raise ValueError("augmentation_completed_pair_generic_scope_changed")
    case = matches[0]
    runtime = case.get("runtime_config") or {}
    expected_path = completion_root / "results" / starter["setup_id"] / starter["model_id"] / starter["case_id"] / "generic_completion_results.json"
    if (completion_identity(case.get("identity") or {}) != identity
            or completion_identity(runtime.get("identity") or {}) != identity
            or case.get("original_run") != str(run) or runtime.get("original_run") != str(run)
            or case.get("model_sha256") != starter.get("model_sha256")
            or case.get("native_command_contract_sha256") != native.get("native_command_contract_sha256")
            or Path(str(case.get("result_path") or "")) != expected_path
            or expected_path.resolve() != expected_path):
        raise ValueError("augmentation_completed_pair_generic_binding_changed")
    parsed = load_valid_repetitions(expected_path, case)
    if len(parsed) != 3:
        raise ValueError("augmentation_completed_pair_generic_incomplete")
    return {"path": str(expected_path), "case_key": case["case_key"],
        "repetitions": [{key: row.get(key) for key in ("repetition_index", "repetition_id",
            "runtime_instance_id", "completed_task_fps", "measured_makespan_seconds")} for row in parsed]}


def completed_native_leaf_input(run: Path, cfg: Mapping[str, Any], leaf_path: Path,
                                variant: Mapping[str, Any], command: list[str], default: str) -> str:
    """Use the old coordinator hash only for its exact, still-bound completed leaf."""
    descriptor = (cfg.get("augmentation_startcase_barrier") or {}).get("reentry") or {}
    if descriptor.get("kind") != "augmentation_completed_native_first_generic":
        return default
    native_reentry_root(run, cfg)  # Full immutable predecessor check, no writes.
    logical = leaf_path.relative_to(run).as_posix()
    selected = [row for row in descriptor["preserved_native_leaves"] if row["path"] == logical]
    if not selected:
        return default
    previous = _object(leaf_path)
    if len(selected) != 1:
        raise ValueError("augmentation_measured_reentry_current_leaf_contract_changed")
    expected = canonical_json_sha256({"coordinator_input_hash": selected[0].get("coordinator_input_hash", descriptor.get(
        "preserved_native_coordinator_input_hash", descriptor["predecessor_input_hash"])),
                                     "variant": dict(variant), "cmd": command})
    if (len(selected) != 1 or expected != selected[0]["input_hash"]
            or previous.get("checkpoint_sha256") != selected[0]["checkpoint_sha256"]):
        raise ValueError("augmentation_measured_reentry_current_leaf_contract_changed")
    return expected


def augmentation_reentry_snapshot_paths(run: Path) -> tuple[Path, ...]:
    """Preserve the failed invocation and any strictly admitted completed leaf."""
    path = run / "reports/native_producer_stage.json"
    if not path.is_file() or not (str(_object(path).get("error") or "").startswith(COMPLETION_FAILURE)
                                  or _object(path).get("error") == CANCEL_NONSTART):
        return ()
    stage = _object(path)
    plan_path = stage["config"]["augmentation_startcase_barrier"]["plan_path"]
    checked = validate_augmentation_reentry(run, plan_path)
    native = run / checked["predecessor_native_relative"]
    paths = [native / name for name in ("native_invocation_config.json",
        "native_coordinator/stage_result.json", "native_performance/stage_result.json")]
    for row in checked.get("preserved_native_leaves", []):
        paths.extend((run / row["path"]).parent.glob("*.json"))
    return tuple(paths)


def native_reentry_root(run: str | Path, cfg: Mapping[str, Any]) -> Path | None:
    """Resolve a checked supplement attempt for both parent and child."""
    barrier = cfg.get("augmentation_startcase_barrier") or {}
    descriptor = barrier.get("reentry")
    if descriptor is None:
        return None
    if not isinstance(descriptor, Mapping) or not descriptor.get("attempt_root"):
        raise ValueError("augmentation_reentry_runtime_attempt_missing")
    checked = validate_augmentation_reentry(run, barrier.get("plan_path") or "",
                                          descriptor.get("original_root"))
    if dict(descriptor) != checked:
        raise ValueError("augmentation_reentry_descriptor_changed")
    return Path(checked["attempt_root"]) / "native"


def unpublished_native_baseline(run: Path, original: Path,
                                source_index: Mapping[str, Any]) -> frozenset[str]:
    """Recognize the bound augmentation nonstart, never missing measurements.

    An ordinary supplement requires both published Native baseline reports.
    The first predecessor never reached their writers. Keep its source index:
    the current attempt must either also be a nonstart or have the separately
    verified first Native completion. The latter's real published measurements
    are checked above; they are never represented as absent or unstarted.
    """
    from .supplement import _regular_evidence, _verify_original_binding

    cfg_path = original / "reports/native_producer_stage_config.json"
    if not cfg_path.is_file() or cfg_path.is_symlink():
        return frozenset()
    cfg = _object(cfg_path)
    barrier = cfg.get("augmentation_startcase_barrier")
    if not isinstance(barrier, Mapping) or not barrier.get("plan_path"):
        return frozenset()
    measured = any((run / raw).exists() for raw in UNPUBLISHED_NATIVE_BASELINE)
    current_proof = None
    if measured:
        current_proof = validate_augmentation_reentry(run, barrier["plan_path"])
        if current_proof.get("kind") != "augmentation_completed_native_first_generic":
            raise ValueError("augmentation_reentry_native_basis_not_absent:published_native_not_bound")
    for raw in UNPUBLISHED_NATIVE_BASELINE:
        if raw in source_index:
            raise ValueError("augmentation_reentry_native_basis_previously_bound:" + raw)
        for root in ((original,) if measured else (run, original)):
            path = root / raw
            if path.exists() or path.is_symlink():
                raise ValueError("augmentation_reentry_native_basis_not_absent:" + str(path))
    # Absence on disk cannot erase an indexed measurement or dispatch attempt.
    for raw in source_index:
        path = Path(raw)
        if ((raw.startswith("native_producers/")
                and any(fnmatchcase(path.name, pattern) for pattern in NATIVE_RAW_PATTERNS))
                or (raw.startswith("reports/native_producer_variants/")
                    and path.name == "performance_stage_result.json")):
            raise ValueError("augmentation_reentry_original_native_observation_bound:" + raw)
    for raw in (COORDINATOR.as_posix(), PERFORMANCE.as_posix(),
                "reports/native_producer_stage.json", "reports/native_producer_stage_config.json"):
        _verify_original_binding(_regular_evidence(original, raw), raw, source_index)
    plan_path = barrier["plan_path"]
    original_proof = validate_augmentation_reentry(run, plan_path, original)
    current_proof = current_proof or validate_augmentation_reentry(run, plan_path)
    if current_proof["kind"] == "augmentation_completed_native_first_generic":
        current_proof = current_proof["cancel_nonstart_ancestor"]
    if current_proof["kind"] == "augmentation_native_cancel_nonstart":
        current_proof = current_proof["quality_nonstart_ancestor"]
    for field in ("run_dir", "plan_path", "plan_sha256", "predecessor_checkpoint_sha256",
                  "predecessor_input_hash", "predecessor_error"):
        if original_proof[field] != current_proof[field]:
            raise ValueError("augmentation_reentry_native_nonstart_proof_changed:" + field)
    return UNPUBLISHED_NATIVE_BASELINE
