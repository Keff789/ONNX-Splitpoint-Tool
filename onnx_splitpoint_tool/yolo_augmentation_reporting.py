"""Report the separate YOLO capability augmentation without rewriting its base.

This is a consumer of the existing Completion and Native/Energy reporters.  It
does not admit measurements, select boundaries, or change any claim decision.
The historical and new observations retain their own run/source identities.
"""
from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .generic_completion_plan import completion_identity
from .generic_completion_reporting import _csv, completion_groups
from .native_energy_reporting import (
    build_native_energy_pairs, collect_native_energy,
    summarize_native_energy_reporting, native_energy_results_path,
)
from .native_performance_identity import canonical_native_backend
from .workflow.checkpoints import atomic_write_json


COHORT = "yolo_min3_capability_augmentation"
MODELS = ("yolo11l", "yolo26m", "yolo26s", "yolov7_paper")
STRATUM = ("model_id", "setup_id", "direction", "precision",
           "comparison_backend", "output_endpoint_id", "boundary_class")


def _load(value: Any, filename: str = "") -> tuple[Any, str]:
    if value is None:
        return {}, ""
    if isinstance(value, (Mapping, list)):
        return copy.deepcopy(value), "provided_payload"
    path = Path(value)
    if path.is_dir() and filename:
        path = path / filename
    if not path.exists():
        return {}, str(path)
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def _rows(payload: Any, key: str = "rows") -> list[dict[str, Any]]:
    values = payload if isinstance(payload, list) else payload.get(key, [])
    if not isinstance(values, list) or any(not isinstance(row, Mapping) for row in values):
        raise ValueError(f"augmentation {key} must be a list of objects")
    return [copy.deepcopy(dict(row)) for row in values]


def _case(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("model_id") or row.get("model") or ""),
            str(row.get("case_id") or row.get("case") or ""),
            str(row.get("setup_id") or ""))


def _unique(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    keys = [_case(row) for row in rows]
    if any(not all(key) for key in keys) or len(keys) != len(set(keys)):
        raise ValueError(f"{label} requires unique complete model/case/setup identities")


def _stratum(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(key) or "") for key in STRATUM)


def _annotate(pair: Mapping[str, Any], *, origin: str, source: str,
              observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    row = copy.deepcopy(dict(pair))
    # Validate, but do not change the existing endpoint and backend identities.
    completion_identity(row)
    row.update(observation_origin=origin,
               augmentation_cohort_id=COHORT if origin == "augmentation" else "",
               source_completion_report=source)
    observed = observation or {}
    row["generic_repetition_statistics"] = {
        key: copy.deepcopy(value) for key, value in observed.items()
        if key.startswith("completed_task_fps") or key.startswith("measured_makespan")
        or key in ("repetitions", "valid_repetition_count", "completed_task_count")
    }
    row["generic_measurement_dates"] = [
        {key: rep[key] for key in ("started_at", "completed_at", "runtime_instance_id") if key in rep}
        for rep in observed.get("repetitions", []) if isinstance(rep, Mapping)
    ]
    row["measurement_date_status"] = "recorded" if any(
        "started_at" in rep or "completed_at" in rep for rep in row["generic_measurement_dates"]
    ) else "see_original_measurement_source"
    for tier in ("technical", "quality", "claim"):
        if row.get(f"eligible_for_{tier}_transfer") is True:
            for metric in ("generic_completed_task_fps", "native_completed_task_fps"):
                number = row.get(metric)
                if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number <= 0:
                    raise ValueError(f"eligible augmentation pair has invalid {metric}")
    return row


def _groups(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse the ranker; add the existing Cross-Runner Top-1 cycle regret."""
    groups = completion_groups(copy.deepcopy(pairs), minimum_candidates=3)
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_stratum[_stratum(row)].append(row)
    for group in groups:
        rows = by_stratum[_stratum(group)]
        close = [row for row in rows if row.get("eligible_for_quality_transfer") is True
                 and row.get("generic_task_quality_status") == "reference_close"
                 and row.get("native_task_quality_status") == "reference_close"]
        filtered = completion_groups(copy.deepcopy(close), minimum_candidates=3)
        for key in ("candidate_count", "spearman_rho", "kendall_tau_b", "pairwise_concordance", "status"):
            group[f"reference_close_{key}"] = (filtered[0].get(f"quality_{key}") if filtered else
                0 if key == "candidate_count" else "insufficient_candidates" if key == "status" else None)
        for tier in ("technical", "quality", "claim", "reference_close"):
            eligible = close if tier == "reference_close" else [
                row for row in rows if row.get(f"eligible_for_{tier}_transfer") is True]
            group[f"{tier}_native_best_hit_at_1"] = None
            group[f"{tier}_native_regret_at_1"] = None
            if len(eligible) >= 3:
                # Identical ordering, cycle coordinate and stable tie handling
                # to cross_runner_reporting.add_shortlist_metrics(k=1).
                native_best = min(eligible, key=lambda row: 1000.0 / row["native_completed_task_fps"])
                selected = min(eligible, key=lambda row: 1000.0 / row["generic_completed_task_fps"])
                best_cycle = 1000.0 / native_best["native_completed_task_fps"]
                selected_cycle = 1000.0 / selected["native_completed_task_fps"]
                group[f"{tier}_native_best_hit_at_1"] = selected is native_best
                group[f"{tier}_native_regret_at_1"] = (selected_cycle - best_cycle) / best_cycle
        group["top3_at_n3_is_trivial"] = group["technical_candidate_count"] == 3
        group["top3_is_performance_claim"] = False
        group["quality_cohort_meaning"] = "valid_accuracy_observation_including_accuracy_loss"
        group["reference_close_cohort_meaning"] = "both_runners_reference_close_with_existing_quality_checks"
        group["observation_origins"] = sorted({row["observation_origin"] for row in rows})
        group["runtime_conditions_comparable"] = all(row.get("runtime_conditions_comparable") is True for row in rows)
    return groups


def _completion(value: Any) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]], str]:
    payload, source = _load(value, "generic_completion_cross_runner.json")
    if not isinstance(payload, Mapping):
        raise ValueError("augmentation completion report must be an object")
    observations = payload.get("observations", [])
    if source not in ("", "provided_payload"):
        obs, _ = _load(Path(source).with_name("generic_completion_observations.json"))
        observations = obs.get("rows", []) if isinstance(obs, Mapping) else []
    _unique(observations, "completion observations")
    return dict(payload), {_case(row): row for row in observations}, source


def _energy(value: Any, run_hint: Any = None) -> tuple[list[dict[str, Any]], Any, str, str]:
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(item, (str, Path)) for item in value
    ):
        # The authorized rest blocks keep separate normal run layouts. Read
        # each actual aggregate with its own validation and retain its source;
        # never flatten raw rows into invented normalized observations.
        roots = [Path(item).resolve() for item in value]
        if len(roots) != len(set(roots)):
            raise ValueError("duplicate augmentation energy block")
        observations, blocks = [], []
        for root in roots:
            source = native_energy_results_path(root, include_partial=True)
            if not root.is_dir() or not source.is_file():
                raise ValueError("augmentation energy block aggregate missing: " + str(source))
            raw, _ = _load(source)
            observations.extend(collect_native_energy(root, include_partial=True))
            blocks.append({"source": str(source), "results": raw})
        _unique(observations, "augmentation energy block observations")
        return (observations, {"blocks": blocks},
                json.dumps([item["source"] for item in blocks]),
                "normal_native_energy_collector_separate_blocks")
    if value is not None and not isinstance(value, (Mapping, list)) and Path(value).is_dir():
        root = Path(value)
        projected = root / "native_energy_observations.json"
        if projected.exists():
            data, source = _load(projected)
            return _rows(data), data, source, "existing_native_energy_observations"
        source = native_energy_results_path(root, include_partial=True)
        raw, _ = _load(source)
        return collect_native_energy(root, include_partial=True), raw, str(source), "normal_native_energy_collector"
    data, source = _load(value)
    rows = _rows(data)
    if not rows or all("execution_mode" in row for row in rows):
        return rows, data, source, "existing_native_energy_observations"
    # A raw aggregate is not a normalized observation.  Invoke the normal
    # collector only at its real canonical run path with its own validation.
    run = Path(run_hint) if run_hint else None
    if source not in ("", "provided_payload"):
        path = Path(source)
        if path.name in {"native_producer_energy_results.json", "native_producer_energy_results.partial.json"} and path.parent.name == "native_energy_measurements" and path.parent.parent.name == "reports":
            run = path.parents[2]
    if run is not None:
        canonical = native_energy_results_path(run, include_partial=True)
        actual, _ = _load(canonical)
        if actual != data:
            raise ValueError("energy payload differs from canonical new-run results")
        return collect_native_energy(run, include_partial=True), data, str(canonical), "normal_native_energy_collector"
    return [], data, source, "raw_results_retained_canonical_run_required_for_projection"


def _bound_native_rows(native: Mapping[str, Any], plan: Mapping[str, Any], planned_by_case) -> list[dict[str, Any]]:
    if not isinstance(native, Mapping):
        raise ValueError("new Native summary must be a normal summary object")
    rows = _rows(native)
    _unique(rows, "new Native observations")
    if not rows:
        return rows
    if not plan.get("augmentation_run"):
        raise ValueError("new Native observations require their augmentation run binding")
    run = Path(plan["augmentation_run"]).resolve()
    authority = native.get("native_split_quality_authority") or {}
    run_ids = [value for value in (native.get("run_id"), native.get("eval_run_id"),
               authority.get("run_id"), authority.get("stage_run_id")) if value]
    if not run_ids or any(value != run.name for value in run_ids):
        raise ValueError("new Native summary run binding conflicts with augmentation")
    for row in rows:
        selected = planned_by_case.get(_case(row))
        if selected is None:
            raise ValueError("new native row outside frozen augmentation plan")
        if selected.get("selection_role") == "reserve" and selected.get("activated") is not True:
            raise ValueError("new Native observation belongs to an inactive reserve")
        expected_backend = canonical_native_backend(selected.get("direction") or selected.get("backend"))
        if (not row.get("backend") or canonical_native_backend(row["backend"]) != expected_backend
                or canonical_native_backend(row.get("comparison_backend")) != expected_backend
                or (row.get("direction") and canonical_native_backend(row["direction"]) != expected_backend)):
            raise ValueError("new Native backend conflicts with augmentation plan")
        if selected.get("precision") and row.get("precision") != selected["precision"]:
            raise ValueError("new Native precision conflicts with augmentation plan")
        for field in ("model_sha256", "output_endpoint_id"):
            if selected.get(field) and row.get(field) and selected[field] != row[field]:
                raise ValueError(f"new Native {field} conflicts with augmentation plan")
        command = row.get("native_command_contract") or {}
        if any(value and value != run.name for value in (row.get("eval_run_id"), command.get("eval_run_id"))):
            raise ValueError("new Native row run binding conflicts with augmentation")
        for field in ("report", "source_root", "technical_terminal_source"):
            if row.get(field):
                source = Path(row[field])
                if not source.is_absolute():
                    source = run / source
                if not source.resolve().is_relative_to(run):
                    raise ValueError(f"new Native {field} is outside augmentation run")
    return rows


def write_augmentation_reports(plan: Any, augmentation_completion_reports: Any = None,
                               new_native_summary: Any = None, new_energy_results: Any = None,
                               output_root: str | Path | None = None) -> dict[str, Any]:
    """Write external baseline references, new-cohort accounting and exact union.

    Inputs may be paths or loaded normal-reporter payloads.  Energy additionally
    accepts its own run directory, or ``native_energy_observations.json``.  Raw
    results without their canonical run/validation are exported without inferred
    eligibility.  Missing inputs remain missing; they never become measurements.
    """
    plan, plan_source = _load(plan)
    if not isinstance(plan, Mapping) or plan.get("cohort_id") != COHORT:
        raise ValueError("augmentation cohort identity missing or changed")
    if output_root is None:
        raise ValueError("augmentation output root required")
    out = Path(output_root).resolve()
    base_root = Path(str(plan["base_completion_root"])).resolve()
    original = Path(str(plan["original_run"])).resolve()
    if out.is_relative_to(original) or out.is_relative_to(base_root):
        raise ValueError("augmentation output must remain outside original and base completion runs")
    planned = _rows(plan)
    _unique(planned, "augmentation plan")
    planned_by_case = {_case(row): row for row in planned}
    native, native_source = _load(new_native_summary, "native_producer_combined_summary.json")
    native_rows = _bound_native_rows(native, plan, planned_by_case)
    native_by_case = {_case(row): (index, row) for index, row in enumerate(native_rows)}
    base_input = base_root / "reports" if (base_root / "reports").is_dir() else base_root
    base, base_obs, base_source = _completion(base_input)
    if not base.get("pairs"):
        raise ValueError("base completion pairs missing")
    added, added_obs, added_source = _completion(augmentation_completion_reports)
    base_rows, added_rows = _rows(base, "pairs"), _rows(added, "pairs")
    _unique(base_rows, "base pairs")
    _unique(added_rows, "new pairs")
    base_keys = {_case(row) for row in base_rows}
    if base_keys & set(planned_by_case):
        raise ValueError("augmentation plan includes an existing base boundary")
    for row in added_rows:
        key = _case(row)
        if key not in planned_by_case:
            raise ValueError("completion pair outside frozen augmentation plan")
        selected = planned_by_case[key]
        for field in ("direction", "precision", "comparison_backend", "output_endpoint_id"):
            expected = selected.get(field)
            if expected and (canonical_native_backend(expected) != canonical_native_backend(row.get(field))
                             if field in ("direction", "comparison_backend") else expected != row.get(field)):
                raise ValueError(f"augmentation completion {field} conflicts with plan")
    base_pairs = [_annotate(row, origin="base", source=base_source, observation=base_obs.get(_case(row))) for row in base_rows]
    new_pairs = [_annotate(row, origin="augmentation", source=added_source, observation=added_obs.get(_case(row))) for row in added_rows]
    union = base_pairs + new_pairs
    groups = _groups(union)
    added_by_case = {_case(row): row for row in new_pairs}
    accounting = []
    for planned_row in planned:
        pair = added_by_case.get(_case(planned_row), {})
        native_index, native_row = native_by_case.get(_case(planned_row), (None, {}))
        row = copy.deepcopy(planned_row)
        row.update(observation_origin="augmentation", augmentation_cohort_id=COHORT,
                   generic_completion_status=pair.get("generic_completion_status", "unstarted"),
                   technical_pair_measured=pair.get("eligible_for_technical_transfer") is True,
                   generic_task_quality_status=pair.get("generic_task_quality_status", "unknown"),
                   native_task_quality_status=pair.get("native_task_quality_status", native_row.get("task_quality_status", "unknown")),
                   claim_pair_eligible=pair.get("eligible_for_claim_transfer") is True,
                   completion_report_source=added_source)
        # Preserve explicit compile/runtime terminal states even with no result.
        row["accounting_status"] = (pair.get("generic_completion_status") or planned_row.get("execution_status")
                                     or planned_row.get("status") or "unstarted")
        if row["accounting_status"] == "unstarted":
            row["accounting_status"] = (planned_row.get("execution_status") or planned_row.get("status") or "unstarted")
        row.update(native_result_present=bool(native_row), native_execution_status=native_row.get("status", "unstarted"),
                   native_summary_source=native_source, native_summary_row_index=native_index,
                   native_result_provenance={field: copy.deepcopy(native_row[field]) for field in (
                       "eval_run_id", "report", "source_root", "native_command_contract_sha256", "source_request_sha256",
                       "failure_stage", "failure_reason", "primary_failure_reason", "upstream_evidence_path", "error",
                       "technical_terminal_kind", "technical_terminal_source",
                       "compiler_dispatched", "compiler_dispatch_count", "runtime_started", "measurement_started",
                       "repetition_count_attempted", "repetition_count_valid", "build_exclusion", "actual_compile_failure",
                       "upstream_build_observation") if field in native_row})
        if row["accounting_status"] == "unstarted" and native_row:
            native_status = str(native_row.get("status") or "unknown")
            row["accounting_status"] = ("native_succeeded_generic_pending"
                if native_status in {"ok", "completed"} and native_row.get("ok") is True else native_status)
            row["accounting_status_source"] = "normal_native_summary"
        else:
            row["accounting_status_source"] = "generic_completion_or_plan"
        if not pair and not native_row and planned_row.get("selection_role") == "reserve" and not planned_row.get("activated"):
            row["accounting_status"] = planned_row.get("execution_status") or "standby"
        accounting.append(row)
    coverage = []
    model_setups = sorted({(row["model_id"], row["setup_id"]) for row in base_pairs if row["model_id"] in MODELS}
                         | {(key[0], key[2]) for key in planned_by_case})
    for model, setup in model_setups:
        old = [row for row in base_pairs if row["model_id"] == model and row["setup_id"] == setup]
        new = [row for row in new_pairs if row["model_id"] == model and row["setup_id"] == setup]
        selected = [row for row in accounting if _case(row)[::2] == (model, setup)]
        strata = [g for g in groups if g["model_id"] == model and g["setup_id"] == setup]
        maximum = max((g["technical_candidate_count"] for g in strata), default=0)
        coverage.append({"model_id": model, "setup_id": setup,
            "base_natural_boundary_count": len({_case(row)[1] for row in old}),
            "primary_boundaries": [row.get("case_id", row.get("case")) for row in selected if row.get("selection_role") == "primary"],
            "reserve_boundaries": [row.get("case_id", row.get("case")) for row in selected if row.get("selection_role") == "reserve"],
            "new_technical_boundary_count": sum(row.get("eligible_for_technical_transfer") is True for row in new),
            "new_accuracy_classes": dict(Counter(row.get("generic_task_quality_status", "unknown") for row in new if row.get("generic_completion_status") == "completed")),
            "union_technical_natural_boundary_count": len({row["case_id"] for row in old + new if row.get("eligible_for_technical_transfer") is True}),
            "largest_exact_comparable_stratum_count": maximum,
            "minimum_candidates": 3, "shortfall": max(0, 3 - maximum),
            "shortfall_reason": "" if maximum >= 3 else "fewer_than_three_measured_boundaries_in_one_exact_stratum",
            "strata": strata})
    energy, energy_raw, energy_source, energy_status = _energy(new_energy_results, plan.get("augmentation_run"))
    for row in energy:
        if _case(row) not in planned_by_case or row.get("execution_mode") != "native_split":
            raise ValueError("new energy observation outside frozen split augmentation plan")
    _unique(energy, "new energy observations")
    base_energy_input = plan.get("base_energy_observations")
    if not base_energy_input and plan.get("base_scientific_reports"):
        base_energy_input = Path(plan["base_scientific_reports"]) / "native_energy_observations.json"
    base_energy, _, base_energy_source, _ = _energy(base_energy_input)
    full = [row for row in base_energy if row.get("execution_mode") == "native_full_baseline"]
    energy_pairs = build_native_energy_pairs(energy + full) if energy else []
    energy_summary = summarize_native_energy_reporting(energy, energy_pairs)
    energy_report = {"projection_status": energy_status, "new_results_source": energy_source,
        "base_observations_source": base_energy_source, "baseline_full_reference_count": len(full),
        "baseline_full_remeasured_count": 0, "new_observations": energy,
        "pairs_against_original_full": energy_pairs, "summary": energy_summary,
        "temporal_comparison": "new_split_measurements_against_historical_full_references",
        "runtime_conditions_comparable": plan.get("energy_runtime_conditions_comparable", "unknown"),
        "raw_full_system_energy_is_primary": True,
        "generic_energy_measurement_performed": False}
    summary = {"cohort_id": COHORT, "selection_scope": "post_planned_capability_graph_selection_not_holdout",
        "base_pair_count": len(base_pairs), "planned_new_identity_count": len(planned),
        "new_completion_pair_count": len(new_pairs), "union_pair_row_count": len(union),
        "new_technical_pair_count": sum(row.get("eligible_for_technical_transfer") is True for row in new_pairs),
        "union_technical_pair_count": sum(row.get("eligible_for_technical_transfer") is True for row in union),
        "new_case_status_counts": dict(Counter(row["accounting_status"] for row in accounting)),
        "yolo_model_setup_group_count": len(coverage),
        "yolo_groups_at_minimum_three": sum(row["shortfall"] == 0 for row in coverage),
        "new_native_row_count": len(native_rows), "energy": energy_summary,
        "claim_decisions_inherited_unchanged": True}
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "base_reference.json", {"original_run": str(original), "base_completion_root": str(base_root),
        "source_completion_report": base_source, "summary": base.get("summary", {}),
        "groups": base.get("groups", []), "source_energy_observations": base_energy_source})
    atomic_write_json(out / "augmentation_cases.json", {"plan_source": plan_source, "rows": accounting,
        "native_summary_source": native_source, "native_summary": native})
    atomic_write_json(out / "augmentation_completion.json", {"pairs": new_pairs, "groups": _groups(new_pairs)})
    atomic_write_json(out / "augmented_union.json", {"pairs": union, "groups": groups, "coverage": coverage, "summary": summary})
    atomic_write_json(out / "augmentation_energy.json", energy_report)
    atomic_write_json(out / "augmentation_energy_raw_results.json", energy_raw)
    atomic_write_json(out / "augmentation_summary.json", summary)
    for name, rows in (("augmentation_cases", accounting), ("augmentation_pairs", new_pairs),
                       ("augmented_union_pairs", union), ("augmented_union_groups", groups),
                       ("yolo_group_coverage", coverage), ("augmentation_energy_observations", energy),
                       ("augmentation_energy_pairs", energy_pairs)):
        _csv(out / f"{name}.csv", rows)
    return summary
