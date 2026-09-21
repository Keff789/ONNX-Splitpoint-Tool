"""Canonical scientific reporting for BenchmarkSets and EvaluationRuns.

Version 60 replaces the historic collection of overlapping claim tables with a
single, versioned report contract.  The reporter executes the task-quality gate
and the pre-registered ranking-method comparison as part of the run.  It never
upgrades legacy point estimates to a final non-inferiority pass and never emits
Top-k claims unless the prediction freeze and candidate universe are auditable.
"""
from __future__ import annotations

from onnx_splitpoint_tool.v60z_integration import augment_scientific_report
from onnx_splitpoint_tool.native_energy_reporting import (
    scientific_energy_rows,
    collect_native_energy,
    build_native_energy_pairs,
    build_native_energy_ab_aggregates,
)
from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
from onnx_splitpoint_tool.energy.comparison import resolve_energy_comparison

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from ..ranking_methods import (
    METHOD_LABELS,
    METHOD_ORDER,
    METHOD_UNITS,
    canonical_direction,
    canonical_runner,
    compute_ranking_predictions,
    method_macro_sort_key,
    ranking_method_policy,
)
from ..campaign import (
    EVALUATED_MATRIX_CLAIM_SCOPE,
    RANKING_GENERALIZATION_CLAIM_SCOPE,
    resolve_campaign_claim_scope,
    verify_prediction_freeze_approval,
)
from ..protocol_freeze import is_confirmatory_holdout, normalize_evaluation_role
from ..quality_result_contract import (UNCERTAINTY_FIELDS, project_quality_result, project_quality_component, project_flat_quality_uncertainty)
from ..accuracy_reporting import assessment_fields
from ..validation.accuracy_gates import AccuracyGatePolicy, apply_accuracy_gate_to_row
from .artifacts import now_iso, read_json, relpath, sha256_file, sha256_json, write_csv, write_json, write_text
from .cross_runner_reporting import compute_cross_runner_report, markdown_for_cross_runner
from .evidence_status import (project_native_evidence_status, workflow_completion_projection, project_historical_workflow_status)

REPORT_SCHEMA = "onnx-splitpoint/scientific-report"
REPORT_SCHEMA_VERSION = 3
PERFORMANCE_ROW_ROLE = "performance_observation"
NATIVE_ENERGY_ROW_ROLE = "native_energy_measurement"
CENTRAL_QUALITY_ROW_ROLE = "central_quality_result"
RANKING_COMPARISON_STRATA = (
    "model_id",
    "direction",
    "runner_regime",
    "setup_id",
    "runtime_precision_identity",
    "comparison_backend",
    "comparison_output_endpoint_id",
    "comparison_endpoint_contract_hash",
    "e2e_scope",
    "comparison_endpoint_stratum",
    "measurement_concurrency",
    "completed_task_stage",
    "completed_task_completion_mode",
    "frozen_host_postprocess_contract_sha256",
    "host_postprocessing_available",
    "host_postprocess_required",
)
FULL_ONLY_QUALITY_ACCEPTANCE_IDENTITY_SCHEMA = (
    "onnx-splitpoint/full-only-quality-acceptance-identity-contract"
)
STANDARD_SETUP_LOCAL_QUALITY_ACCEPTANCE_IDENTITY_SCHEMA = (
    "onnx-splitpoint/standard-setup-local-tensorrt-quality-"
    "acceptance-identity-contract"
)
FULL_ONLY_QUALITY_ACCEPTANCE_IDENTITY_SCHEMA_VERSION = 1
QUALITY_ACCEPTANCE_IDENTITY_CONTRACT_SCOPES = {
    FULL_ONLY_QUALITY_ACCEPTANCE_IDENTITY_SCHEMA: "full_only",
    STANDARD_SETUP_LOCAL_QUALITY_ACCEPTANCE_IDENTITY_SCHEMA: (
        "standard_quality_setup_local_tensorrt"
    ),
}
FULL_ONLY_QUALITY_ACCEPTANCE_IDENTITY_KEY_FIELDS = (
    "model_id",
    "source_run_id",
    "setup_id",
    "backend",
    "variant",
    "execution_role",
    "performance_claims_emitted",
)
CLAIM_PERFORMANCE_CSV_FIELDS = (
    "model_id", "task", "case_id", "backend", "variant", "setup_id",
    "comparison_backend", "throughput_fps", "latency_ms",
    "performance_eligible", "performance_claim_eligible",
    "precision_quality_binding_verified", "accuracy_gate_pass",
    "task_quality_status", "eligibility_status", "exclusion_reason",
    "performance_claim_exclusion_reasons",
)
CLAIM_ENERGY_CSV_FIELDS = (
    "model_id", "task", "case_id", "backend", "setup_id",
    "comparison_backend", "energy_per_work_j", "average_power_w",
    "measurement_ok", "energy_eligible", "energy_claim_eligible",
    "precision_quality_binding_verified",
    "task_quality_observation_valid", "accuracy_gate_pass",
    "quality_provenance_complete", "quality_claim_result_verified",
    "screening_comparable", "claim_comparable", "exclusion_reason",
    "scientific_claim_exclusion_reasons",
)
CLAIM_EXCLUSION_DETAIL_CSV_FIELDS = (
    "claim_kind", "model_id", "backend", "setup_id", "reason",
    "task", "case_id", "variant", "row_status", "eligibility_status",
)
CLAIM_EXCLUSION_GROUP_CSV_FIELDS = (
    "claim_kind", "model_id", "backend", "setup_id", "reason", "count",
)
CLAIM_EXCLUSION_DIMENSION_CSV_FIELDS = (
    "claim_kind", "dimension", "value", "count",
)
CLAIM_EXCLUSION_DIMENSIONS = (
    ("model", "model_id"),
    ("backend", "backend"),
    ("setup", "setup_id"),
    ("reason", "reason"),
)

# These are presentation artefacts only.  Raw benchmark results, native dumps,
# task predictions, energy summaries and traces remain untouched.
LEGACY_REPORT_FILES = {
    "claim_table_best_splits.csv",
    "claim_table_comprehensive.csv",
    "claim_table_comprehensive.json",
    "claim_table_comprehensive.md",
    "claim_table_full_models.csv",
    "energy_claim_summary.csv",
    "energy_claim_summary.md",
    "prediction_calibration.csv",
    "prediction_vs_benchmark.csv",
    "thesis_metrics.csv",
    "thesis_metrics.json",
    "thesis_metrics.md",
    "backend_speedups.csv",
    "full_backend_throughput.csv",
    "heterogeneous_pipeline_throughput.csv",
    "pipeline_throughput.csv",
    "same_backend_split_diagnostics.csv",
    "energy_pipeline.csv",
    "energy_summary.csv",
    "validation_summary.csv",
    "model_summary.csv",
    "summary.csv",
    "result_dashboard.csv",
    "thesis_section.md",
    "ranking_validation.csv",
    "ranking_method_comparison.csv",
    "ranking_method_macro.csv",
}
LEGACY_TABLE_FILES = {
    "claim_table_comprehensive.tex",
    "claim_table_best_splits.tex",
    "claim_table_full_models.tex",
    "energy_claim_summary.tex",
    "thesis_best_summary.tex",
    "full_backend_throughput.tex",
    "heterogeneous_pipeline_throughput.tex",
    "pipeline_throughput.tex",
    "backend_speedups.tex",
    "prediction_accuracy.tex",
    "model_selection.tex",
    "ranking_validation.tex",
    "ranking_method_comparison.tex",
}
LEGACY_FIGURE_NAMES = {
    "latency_by_model.png",
    "pipeline_fps_by_model.png",
    "predicted_vs_measured.png",
    "claim_pipeline_fps_bar.png",
    "claim_pipeline_latency_ms_bar.png",
    "ranking_method_spearman.png",
    "ranking_method_hit_at_5.png",
    "ranking_method_regret_at_5.png",
}
LEGACY_REPORT_DIRS = {"accuracy_gates"}


def _f(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _b(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "ok", "pass", "passed", "valid", "claim_ok", "eligible"}:
        return True
    if text in {"0", "false", "no", "fail", "failed", "invalid", "error"}:
        return False
    return None


def _claim_eligible_axis(
    row: Mapping[str, Any],
    *,
    explicit_key: str,
    legacy_key: str,
) -> bool:
    """Null is an absent assertion; false and conflicting evidence are vetoes."""
    if (
        any(_b(row.get(key)) is False for key in (
            "claim_eligible", "declared_claim_eligible", "runtime_executable",
            legacy_key,
        ))
        or _b(row.get("diagnostic_only")) is True
        or row.get("identity_conflicts")
        or row.get("completed_v2_projection_conflicts")
        or _claim_exclusion_tokens(row.get(
            "performance_claim_exclusion_reasons" if explicit_key == "performance_claim_eligible"
            else "scientific_claim_exclusion_reasons"
        ))
    ):
        return False
    if row.get(explicit_key) is not None:
        return _b(row.get(explicit_key)) is True
    return _b(row.get(legacy_key)) is True


def _performance_claim_eligible(row: Mapping[str, Any]) -> bool:
    return _claim_eligible_axis(
        row,
        explicit_key="performance_claim_eligible",
        legacy_key="performance_eligible",
    )


def _energy_claim_eligible(row: Mapping[str, Any]) -> bool:
    return _claim_eligible_axis(
        row,
        explicit_key="energy_claim_eligible",
        legacy_key="energy_eligible",
    )


def _claim_exclusion_tokens(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return _claim_exclusion_tokens(parsed)
        return [text]
    if isinstance(value, (list, tuple, set)):
        return sorted({
            token
            for item in value
            for token in _claim_exclusion_tokens(item)
            if token
        })
    return [str(value).strip()] if str(value).strip() else []


def _claim_exclusion_reasons(
    row: Mapping[str, Any],
    *,
    claim_kind: str,
) -> list[str]:
    if claim_kind == "performance":
        structured_key = "performance_claim_exclusion_reasons"
        fallback_keys = (
            "exclusion_reason",
            "ranking_exclusion_reason",
            "eligibility_gate_reason",
        )
        default = "not_performance_eligible"
    else:
        structured_key = "scientific_claim_exclusion_reasons"
        fallback_keys = (
            "exclusion_reason",
            "measurement_failure_reason",
        )
        default = "not_energy_eligible"
    reasons = _claim_exclusion_tokens(row.get(structured_key))
    if not reasons:
        for key in fallback_keys:
            reasons = _claim_exclusion_tokens(row.get(key))
            if reasons:
                break
    return sorted(set(reasons or [default]))


def _claim_exclusion_dimension(
    row: Mapping[str, Any],
    keys: Sequence[str],
    fallback: str,
) -> str:
    for key in keys:
        value = str(row.get(key) if row.get(key) is not None else "").strip()
        if value:
            return value
    return fallback


def _claim_exclusion_detail_rows(
    performance_rows: Sequence[Mapping[str, Any]],
    energy_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for claim_kind, source_rows in (
        ("performance", performance_rows),
        ("energy", energy_rows),
    ):
        for row in source_rows:
            model_id = _claim_exclusion_dimension(
                row, ("model_id", "model"), "unknown_model",
            )
            backend = _claim_exclusion_dimension(
                row,
                ("backend", "split_backend", "run_id"),
                "unknown_backend",
            )
            setup_id = _claim_exclusion_dimension(
                row,
                ("setup_id", "measurement_setup_id", "source_setup_id"),
                "unknown_setup",
            )
            for reason in _claim_exclusion_reasons(
                row, claim_kind=claim_kind,
            ):
                details.append({
                    "claim_kind": claim_kind,
                    "model_id": model_id,
                    "backend": backend,
                    "setup_id": setup_id,
                    "reason": reason,
                    "task": str(row.get("task") or ""),
                    "case_id": str(
                        row.get("case_id") or row.get("case") or ""
                    ),
                    "variant": str(row.get("variant") or ""),
                    "row_status": _row_status(row),
                    "eligibility_status": str(
                        row.get("eligibility_status") or ""
                    ),
                })
    kind_order = {"performance": 0, "energy": 1}
    return sorted(
        details,
        key=lambda row: (
            kind_order.get(str(row.get("claim_kind") or ""), 99),
            str(row.get("model_id") or ""),
            str(row.get("backend") or ""),
            str(row.get("setup_id") or ""),
            str(row.get("reason") or ""),
            str(row.get("task") or ""),
            str(row.get("case_id") or ""),
            str(row.get("variant") or ""),
            str(row.get("row_status") or ""),
            str(row.get("eligibility_status") or ""),
        ),
    )


def _claim_exclusion_group_rows(
    details: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(
        (
            str(row.get("claim_kind") or ""),
            str(row.get("model_id") or ""),
            str(row.get("backend") or ""),
            str(row.get("setup_id") or ""),
            str(row.get("reason") or ""),
        )
        for row in details
    )
    kind_order = {"performance": 0, "energy": 1}
    return [
        {
            "claim_kind": key[0],
            "model_id": key[1],
            "backend": key[2],
            "setup_id": key[3],
            "reason": key[4],
            "count": count,
        }
        for key, count in sorted(
            counts.items(),
            key=lambda item: (
                kind_order.get(item[0][0], 99),
                item[0][1],
                item[0][2],
                item[0][3],
                item[0][4],
            ),
        )
    ]


def _claim_exclusion_dimension_rows(
    details: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for claim_kind in ("performance", "energy"):
        kind_rows = [
            row for row in details
            if str(row.get("claim_kind") or "") == claim_kind
        ]
        for dimension, key in CLAIM_EXCLUSION_DIMENSIONS:
            counts = Counter(str(row.get(key) if row.get(key) is not None else "") for row in kind_rows)
            rows.extend({
                "claim_kind": claim_kind,
                "dimension": dimension,
                "value": value,
                "count": count,
            } for value, count in sorted(counts.items()))
    return rows


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file() or yaml is None:
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return dict(value or {}) if isinstance(value, Mapping) else {}
    except Exception:
        return {}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _archived_scientific_performance_rows(
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Recover trimmed-debug-pack observations without treating them as raw.

    Complete EvaluationRuns use normalized benchmark results. Compact debug
    packs may intentionally retain only the canonical ``row_eligibility``
    projection. That projection is still factual measurement evidence, but its
    provenance must remain explicit because it cannot repair fields omitted by
    the older reporter.
    """

    path = run_dir / "reports" / "scientific" / "row_eligibility.csv"
    rows = []
    for raw in _read_csv_rows(path):
        role = str(raw.get("row_role") or PERFORMANCE_ROW_ROLE).strip()
        if role != PERFORMANCE_ROW_ROLE:
            continue
        row = dict(raw)
        row["report_replay_input_source"] = (
            "archived_scientific_row_eligibility"
        )
        row["report_replay_input_path"] = str(path)
        for field in (
            "runtime_numeric_input_identity",
            "runtime_input_encoding_identity",
            "output_endpoint_attestation",
        ):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip().startswith("{"):
                continue
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed, Mapping):
                row[field] = dict(parsed)
        rows.append(row)
    return rows


def _enrich_archived_setup_identity(
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Restore exact request identities omitted by old compact CSV reports.

    The central quality ledger is frozen EvaluationRun evidence and retains
    the variant-local request identity that produced each archived
    performance row.  Replay may restore that existing identity only for one
    exact model/case/run/variant match; it never borrows an adjacent case or
    backend identity.
    """

    central = read_json(
        run_dir / "quality_management" / "central_quality_summary.json",
        default={},
    ) or {}
    central_results = (
        list(central.get("results") or [])
        if isinstance(central, Mapping) else []
    )

    def variant(value: Any) -> str:
        token = str(value or "").strip().lower()
        return "split" if token in {"split", "composed"} else token

    def run_identity(row: Mapping[str, Any]) -> str:
        raw = (
            row.get("source_run_id") or row.get("run_id")
            or row.get("backend") or row.get("direction") or ""
        )
        return canonical_direction(raw)

    def case_identity(row: Mapping[str, Any]) -> str:
        return (
            "full"
            if variant(row.get("variant")) == "full"
            else str(row.get("case_id") or "").strip()
        )

    setups_by_key: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    identities_by_key: dict[
        tuple[str, str, str, str], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for raw in central_results:
        if not isinstance(raw, Mapping):
            continue
        request_identity = (
            raw.get("request_identity")
            if isinstance(raw.get("request_identity"), Mapping) else {}
        )
        setup_id = str(
            raw.get("setup_id") or raw.get("source_setup_id")
            or request_identity.get("setup_id") or ""
        ).strip()
        key = (
            str(raw.get("model_id") or "").strip(),
            case_identity(raw),
            run_identity(raw),
            variant(raw.get("variant")),
        )
        if setup_id and all(key):
            setups_by_key[key].add(setup_id)
        if request_identity and all(key):
            canonical = json.dumps(
                dict(request_identity),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            identities_by_key[key].setdefault(canonical, dict(request_identity))

    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        existing_setup = str(row.get("setup_id") or "").strip()
        key = (
            str(row.get("model_id") or "").strip(),
            case_identity(row),
            run_identity(row),
            variant(row.get("variant")),
        )
        if existing_setup:
            row["setup_identity_status"] = "source_row_explicit"
        candidates = sorted(setups_by_key.get(key, set()))
        if not existing_setup and len(candidates) == 1:
            row["setup_id"] = candidates[0]
            row["setup_identity_status"] = "central_quality_exact_join"
        elif not existing_setup and candidates:
            row["setup_identity_status"] = "ambiguous_central_quality_join"
        elif not existing_setup:
            row["setup_identity_status"] = "setup_identity_unavailable"
        row["setup_identity_candidates"] = candidates

        identity_candidates = list(identities_by_key.get(key, {}).values())
        if len(identity_candidates) == 1:
            request_identity = dict(identity_candidates[0])
            request_variant = str(
                request_identity.get("variant") or "composed"
            ).strip().lower()
            if request_variant in {"", "split", "pipeline"}:
                request_variant = "composed"
            existing = row.get("quality_request_identities_by_variant")
            projected = {
                str(name): dict(value)
                for name, value in (
                    existing.items() if isinstance(existing, Mapping) else []
                )
                if isinstance(value, Mapping)
            }
            previous = projected.get(request_variant)
            if previous and json.dumps(
                previous, sort_keys=True, separators=(",", ":"), default=str,
            ) != json.dumps(
                request_identity,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ):
                row["quality_request_identity_status"] = (
                    "conflicting_archived_and_central_identity"
                )
            else:
                projected[request_variant] = request_identity
                row["quality_request_identities_by_variant"] = projected
                row["quality_request_identity_status"] = (
                    "central_quality_exact_join"
                )
        elif identity_candidates:
            row["quality_request_identity_status"] = (
                "ambiguous_central_quality_join"
            )
        else:
            row["quality_request_identity_status"] = "identity_unavailable"
        enriched.append(row)
    return enriched


def _profile_model_entries(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    suite = profile.get("model_suite") if isinstance(profile.get("model_suite"), Mapping) else {}
    for tier in ("primary", "reserve"):
        for item in list(suite.get(tier) or []):
            if isinstance(item, Mapping):
                if item.get("enabled") is False:
                    continue
                model_id = str(item.get("id") or "").strip()
                if model_id:
                    row = dict(item)
                    row.setdefault("suite_tier", tier)
                    row.setdefault("evaluation_role", "development")
                    row["evaluation_role"] = normalize_evaluation_role(row.get("evaluation_role"))
                    out[model_id] = row
            elif item:
                out[str(item)] = {"id": str(item), "suite_tier": tier, "evaluation_role": "development"}
    return out


def _normalized_scientific_task(value: Any, model_id: Any) -> str:
    """Return the report-contract task instead of leaking ``auto``/blank."""
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"classification", "image_classification", "classifier"}:
        return "classification"
    if raw in {"detection", "object_detection", "detector"}:
        return "detection"
    model = str(model_id or "").strip().lower()
    return "detection" if any(token in model for token in ("yolo", "detr")) else "classification"


def _policy_from_profile(profile: Mapping[str, Any]) -> AccuracyGatePolicy:
    block = profile.get("quality_gate") if isinstance(profile.get("quality_gate"), Mapping) else {}
    return AccuracyGatePolicy.from_mapping(block or None)


def _ranking_policy(profile: Mapping[str, Any]) -> dict[str, Any]:
    raw = profile.get("ranking_validation") if isinstance(profile.get("ranking_validation"), Mapping) else {}
    method_policy = ranking_method_policy(raw)
    return {
        "enabled": bool(raw.get("enabled", True)),
        "holdout_unit": str(raw.get("holdout_unit") or "model_direction_runner"),
        "k_values": sorted({int(x) for x in list(raw.get("k_values") or [1, 3, 5]) if int(x) > 0}),
        "elite_q_values": sorted({int(x) for x in list(raw.get("elite_q_values") or [1, 3]) if int(x) > 0}),
        "primary_k": int(raw.get("primary_k") or 5),
        "minimum_candidates_for_correlation": int(raw.get("minimum_candidates_for_correlation") or 3),
        "candidate_universe": str(raw.get("candidate_universe") or "all_feasible"),
        "require_frozen_predictions": bool(raw.get("require_frozen_predictions", True)),
        "require_complete_candidate_universe": bool(raw.get("require_complete_candidate_universe", True)),
        "near_optimal_relative_epsilon": float(raw.get("near_optimal_relative_epsilon") or 0.01),
        # Compatibility default is explicit and aligned with Standard.  A
        # confirmatory profile may still pre-register 2,000 (or another value),
        # but the reporter must not invent a hidden Final-sized workload.
        "bootstrap_repetitions": int(raw.get("bootstrap_repetitions") or 500),
        "bootstrap_seed": int(raw.get("bootstrap_seed") or 20260710),
        "targets": list(raw.get("targets") or ["pipeline_cycle_ms"]),
        "methods": list(method_policy.get("methods") or METHOD_ORDER),
        "method_policy": method_policy,
    }


def _ranking_audit_request(
    profile: Mapping[str, Any],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recover the score-independent audit request from the frozen run inputs.

    The archived selection policy is the execution plan.  A prospective
    candidate-universe freeze is an independent second source and also keeps
    failed/partially generated audit runs identifiable when no ranking group
    could be formed.
    """

    selection = (
        dict(profile.get("selection_policy") or {})
        if isinstance(profile.get("selection_policy"), Mapping)
        else {}
    )
    strategy = (
        str(selection.get("selection_strategy") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    plan_requested = bool(
        strategy in {
            "score_independent_audit",
            "ranking_audit",
            "deterministic_audit",
        }
        or _b(selection.get("score_independent_audit_enabled")) is True
    )
    frozen_model_ids = sorted(
        str(model_id)
        for model_id, prediction in predictions.items()
        if isinstance(prediction, Mapping)
        and isinstance(prediction.get("_prediction_freeze"), Mapping)
        and str(
            prediction["_prediction_freeze"].get("candidate_universe_scope")
            or ""
        ).strip()
        == "predeclared_audit_universe"
    )
    freeze_requested = bool(frozen_model_ids)
    sources = [
        source
        for source, enabled in (
            ("selection_plan", plan_requested),
            ("prediction_freeze", freeze_requested),
        )
        if enabled
    ]
    native_producers = (
        dict(profile.get("native_producers") or {})
        if isinstance(profile.get("native_producers"), Mapping)
        else {}
    )
    native_enabled = (
        native_producers.get("enabled")
        if isinstance(native_producers.get("enabled"), bool)
        else None
    )
    native_enabled_source = (
        "profile.native_producers.enabled"
        if native_enabled is not None else "unavailable"
    )
    return {
        "requested": bool(plan_requested or freeze_requested),
        "source": "+".join(sources) if sources else "none",
        "selection_strategy": strategy,
        "score_independent_audit_enabled": bool(plan_requested),
        "frozen_audit_model_ids": frozen_model_ids,
        "native_execution_enabled": native_enabled,
        "native_execution_enabled_source": native_enabled_source,
    }


def clean_legacy_reports(reports_dir: Path) -> dict[str, Any]:
    removed: list[str] = []
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted(LEGACY_REPORT_FILES):
        path = reports_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    tables = reports_dir / "tables"
    for name in sorted(LEGACY_TABLE_FILES):
        path = tables / name
        if path.is_file():
            path.unlink()
            removed.append(f"tables/{name}")
    figures = reports_dir / "figures"
    if figures.is_dir():
        for path in list(figures.rglob("*")):
            if path.is_file() and path.name in LEGACY_FIGURE_NAMES:
                path.unlink()
                removed.append(relpath(path, reports_dir))
        for path in sorted(figures.rglob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
    for name in sorted(LEGACY_REPORT_DIRS):
        path = reports_dir / name
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name + "/")
    payload = {
        "schema": "onnx-splitpoint/report-cleanup",
        "schema_version": 2,
        "created_at": now_iso(),
        "removed": removed,
        "removed_count": len(removed),
        "preserved_note": "Raw benchmark data, native validation artefacts, prediction freezes, energy traces and provenance files were not removed.",
    }
    write_json(reports_dir / "scientific" / "report_cleanup.json", payload)
    return payload


def _row_score(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        1 if _b(row.get("runtime_ok")) is True or _b(row.get("runtime_executable")) is True else 0,
        1 if _b(row.get("final_pass_all")) is not None else 0,
        1 if _f(row.get("throughput_primary_fps")) is not None or _f(row.get("pipeline_cycle_selected_ms")) is not None else 0,
        sum(1 for value in row.values() if value not in (None, "", [], {})),
    )


def _dedupe_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = (
            str(row.get("model_id") or ""),
            str(row.get("run_id") or row.get("backend") or ""),
            str(row.get("backend") or ""),
            str(row.get("case_id") or ""),
            str(row.get("variant") or ""),
            str(
                row.get("setup_id")
                or row.get("measurement_setup_id")
                or row.get("source_setup_id")
                or ""
            ),
        )
        current = best.get(key)
        if current is None or _row_score(row) > _row_score(current):
            best[key] = row
    return sorted(
        best.values(),
        key=lambda row: (
            str(row.get("model_id") or ""),
            str(row.get("backend") or ""),
            str(row.get("case_id") or ""),
            str(row.get("variant") or ""),
            str(
                row.get("setup_id")
                or row.get("measurement_setup_id")
                or row.get("source_setup_id")
                or ""
            ),
        ),
    )


def _central_quality_setup_id(result: Mapping[str, Any]) -> str:
    """Return the physical setup identity without guessing across setups."""

    nested = (
        result.get("request_identity")
        if isinstance(result.get("request_identity"), Mapping)
        else {}
    )
    producer = (
        result.get("producer_identity")
        if isinstance(result.get("producer_identity"), Mapping)
        else {}
    )
    nested_producer = (
        nested.get("producer_identity")
        if isinstance(nested.get("producer_identity"), Mapping)
        else {}
    )
    return str(
        _first(
            result,
            ("setup_id", "source_setup_id", "measurement_setup_id"),
        )
        or _first(nested, ("setup_id", "source_setup_id"))
        or _first(producer, ("setup_id", "source_setup_id"))
        or _first(nested_producer, ("setup_id", "source_setup_id"))
        or ""
    ).strip()


def _canonical_quality_decision(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token in {"reference_close", "accuracy_loss", "not_estimable"}:
        return token
    if token in {"pass", "passed", "ok", "success", "successful"}:
        return "pass"
    if token in {"fail", "failed", "failure"}:
        return "fail"
    if token in {
        "inconclusive",
        "legacy_point_estimate_only",
        "screening_only",
    }:
        return "inconclusive"
    return "not_evaluated"


def _configured_quality_guardrails(
    result: Mapping[str, Any], metric_gate_config: Mapping[str, Any],
) -> list[str]:
    """Recover the complete guardrail contract, including legacy results.

    v2.75.21 stored the configured guardrails only in
    ``metric_gate_config.guardrails``.  The evaluator then materialised AP50
    but silently omitted the configured AP75 component.  Reporting must not
    turn such a historical PASS into a contract-complete PASS merely because
    the newer explicit ``configured_guardrails`` field is absent.
    """

    configured: set[str] = set()
    explicit = result.get("configured_guardrails")
    if isinstance(explicit, Sequence) and not isinstance(
        explicit, (str, bytes, bytearray),
    ):
        configured.update(
            str(name).strip()
            for name in explicit
            if str(name).strip()
        )
    raw = metric_gate_config.get("guardrails")
    if not isinstance(raw, Mapping):
        return sorted(configured)
    task = str(
        result.get("task") or metric_gate_config.get("task") or ""
    ).strip().lower()
    if not task:
        primary_metric = str(
            metric_gate_config.get("primary_metric") or ""
        ).strip().lower()
        if primary_metric.startswith(("coco_", "ap")):
            task = "detection"
        elif primary_metric.startswith("top") or "accuracy" in primary_metric:
            task = "classification"
    for configured_name in raw:
        name = str(configured_name).strip()
        if not name:
            continue
        metric = name[:-7] if name.endswith("_margin") else name
        if task == "detection" and metric.startswith("top"):
            continue
        if task == "classification" and metric.startswith("ap"):
            continue
        configured.add(metric)
    return sorted(configured)


def _quality_component_projection(
    component: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = project_quality_component(component) if isinstance(component, Mapping) else {}
    return {
        key: source.get(key)
        for key in (
            "metric",
            "candidate",
            "reference",
            "delta",
            "ci_low",
            "ci_high",
            "margin",
            "decision",
            "status",
            "n",
            "bootstrap_repetitions_requested",
            "bootstrap_repetitions",
            "bootstrap_engine",
            "bootstrap_skipped_reason",
            "bootstrap_elapsed_s",
            *UNCERTAINTY_FIELDS,
        )
    }


def _central_quality_gate_reasons(
    result: Mapping[str, Any],
    primary: Mapping[str, Any],
    guardrails: Mapping[str, Any],
) -> list[str]:
    """Project only reasons, decisions and skips present in source evidence."""

    reasons: list[str] = []

    def add(value: Any) -> None:
        token = str(value or "").strip()
        if token and token not in reasons:
            reasons.append(token)

    for field in (
        "decision_reason", "gate_reason", "reason", "skip_reason",
        "failure_reason", "inconclusive_reason",
    ):
        value = result.get(field)
        if value not in (None, ""):
            add(f"result:{field}:{value}")

    for scope, component in (
        ("primary", primary),
        *(
            (f"guardrail:{name}", value)
            for name, value in sorted(guardrails.items())
            if isinstance(value, Mapping)
        ),
    ):
        metric = str(component.get("metric") or scope.split(":")[-1]).strip()
        component_scope = (
            scope if scope.split(":")[-1] == metric else f"{scope}:{metric}"
        )
        decision = str(
            component.get("decision") or component.get("status") or ""
        ).strip()
        if decision:
            add(f"{component_scope}:decision:{decision}")
        for field in (
            "decision_reason", "reason", "skip_reason",
            "bootstrap_skipped_reason",
        ):
            value = component.get(field)
            if value not in (None, ""):
                add(f"{component_scope}:{field}:{value}")
    return reasons


def _central_quality_result_projection(
    result: Mapping[str, Any],
    *,
    source_index: int,
    dataset_tier: str = "",
) -> dict[str, Any]:
    """Create the lossless, setup-aware canonical quality result row.

    These are quality observations, not performance observations.  In
    particular, a setup-local ``full_quality_only`` TensorRT companion must
    never acquire latency/FPS or performance-claim eligibility merely because
    its semantic result is retained in the canonical report.
    """

    result = project_quality_result(result)
    primary = (
        dict(result.get("primary") or {})
        if isinstance(result.get("primary"), Mapping)
        else {}
    )
    guardrails = (
        dict(result.get("guardrails") or {})
        if isinstance(result.get("guardrails"), Mapping)
        else {}
    )
    metric_gate_config = (
        dict(result.get("metric_gate_config") or {})
        if isinstance(result.get("metric_gate_config"), Mapping)
        else {}
    )
    configured_guardrails = _configured_quality_guardrails(
        result, metric_gate_config,
    )
    observed_guardrails = {
        str(name).strip() for name in guardrails if str(name).strip()
    }
    explicit_missing = result.get("missing_guardrails")
    missing_guardrails = {
        str(name).strip()
        for name in (
            explicit_missing
            if isinstance(explicit_missing, Sequence)
            and not isinstance(explicit_missing, (str, bytes, bytearray))
            else []
        )
        if str(name).strip()
    }
    missing_guardrails.update(
        set(configured_guardrails) - observed_guardrails
    )
    source_contract_complete = result.get("guardrail_contract_complete")
    if missing_guardrails or source_contract_complete is False:
        guardrail_contract_complete: bool | None = False
    elif configured_guardrails or source_contract_complete is True:
        guardrail_contract_complete = True
    else:
        guardrail_contract_complete = None
    technical_status = str(
        result.get("technical_status") or result.get("status") or "unavailable"
    ).strip().lower()
    source_decision = _canonical_quality_decision(
        result.get("decision") or result.get("scientific_status")
    )
    # A known fail or inconclusive result remains scientifically meaningful,
    # but an apparent pass with an incomplete configured contract fails closed.
    decision = (
        "not_evaluated"
        if source_decision == "pass" and guardrail_contract_complete is False
        else source_decision
    )
    setup_id = _central_quality_setup_id(result)
    source_request_sha256 = str(
        result.get("source_request_sha256") or ""
    ).strip().lower().removeprefix("sha256:")
    result_id = str(result.get("evaluation_fingerprint") or "").strip()
    if not result_id:
        result_id = sha256_json({
            "model_id": result.get("model_id"),
            "run_id": result.get("run_id") or result.get("source_run_id"),
            "setup_id": setup_id,
            "case_id": result.get("case_id"),
            "variant": result.get("variant"),
            "source_request_sha256": source_request_sha256,
            "producer_identity_sha256": result.get(
                "producer_identity_sha256"
            ),
        })
    gate_reasons = _central_quality_gate_reasons(
        result, primary, guardrails,
    )
    from ..accuracy_reporting import assessment_fields
    projected = {
        **assessment_fields(result.get("accuracy_assessment")),
        "secondary_accuracy_assessments": result.get("secondary_accuracy_assessments"),
        "accuracy_warnings": result.get("accuracy_warnings", []),
        "legacy_decision": result.get("legacy_decision"),
        "observed_image_ids": result.get("observed_image_ids", []),
        "evaluated_images": result.get("evaluated_images"),
        "collection_eval_run_id": result.get("collection_eval_run_id"),
        "row_role": CENTRAL_QUALITY_ROW_ROLE,
        "quality_result_id": result_id,
        "source_index": int(source_index),
        "model_id": result.get("model_id"),
        "task": result.get("task"),
        "case_id": result.get("case_id"),
        "backend": (
            result.get("backend")
            or result.get("source_run_id")
            or result.get("run_id")
        ),
        "setup_id": setup_id,
        "run_id": result.get("run_id") or result.get("source_run_id"),
        "source_run_id": result.get("source_run_id") or result.get("run_id"),
        "variant": result.get("variant"),
        "execution_role": result.get("execution_role"),
        "performance_claims_emitted": result.get(
            "performance_claims_emitted"
        ),
        "technical_status": technical_status,
        "task_quality_tier": dataset_tier,
        "completion_reason": result.get("completion_reason"),
        "task_quality_status": decision,
        "task_quality_decision": decision,
        "source_task_quality_decision": source_decision,
        "task_quality_metric": primary.get("metric"),
        "task_quality_candidate": primary.get("candidate"),
        "task_quality_reference": primary.get("reference"),
        "task_quality_delta": primary.get("delta"),
        "task_quality_ci_low": primary.get("ci_low"),
        "task_quality_ci_high": primary.get("ci_high"),
        "task_quality_margin": primary.get("margin"),
        "task_quality_bootstrap_repetitions_requested": primary.get(
            "bootstrap_repetitions_requested"
        ),
        "task_quality_bootstrap_repetitions": primary.get(
            "bootstrap_repetitions"
        ),
        "task_quality_bootstrap_engine": primary.get("bootstrap_engine"),
        "task_quality_bootstrap_skipped_reason": primary.get(
            "bootstrap_skipped_reason"
        ),
        "task_quality_bootstrap_elapsed_s": primary.get(
            "bootstrap_elapsed_s"
        ),
        "task_quality_gate_reason": ";".join(gate_reasons),
        "task_quality_gate_reasons": gate_reasons,
        "validation_evaluated_count": result.get("n") or primary.get("n"),
        "runtime_precision_identity": result.get(
            "runtime_precision_identity"
        ),
        "quality_policy_sha256": (
            result.get("policy_sha256")
            or result.get("task_quality_policy_sha256")
        ),
        "validation_dataset_manifest_sha256": result.get(
            "validation_dataset_manifest_sha256"
        ),
        "validation_dataset_sha256": result.get(
            "validation_dataset_sha256"
        ),
        "validation_image_ids_sha256": result.get(
            "validation_image_ids_sha256"
        ),
        "validation_ground_truth_sha256": result.get(
            "validation_ground_truth_sha256"
        ),
        "preprocessing_contract_sha256": result.get(
            "preprocessing_contract_sha256"
        ),
        "decoder_contract_sha256": result.get("decoder_contract_sha256"),
        "nms_contract_sha256": result.get("nms_contract_sha256"),
        "quality_contract_sha256": result.get("quality_contract_sha256"),
        "algorithm_version": result.get("algorithm_version"),
        "quality_result_contract_version": result.get(
            "quality_result_contract_version"
        ),
        "metric_gate_config": metric_gate_config,
        "evaluation_fingerprint": result.get("evaluation_fingerprint"),
        "reference_identity": result.get("reference_identity"),
        "reference_predictions_sha256": result.get(
            "reference_predictions_sha256"
        ),
        "candidate_predictions_sha256": result.get(
            "candidate_predictions_sha256"
        ),
        "annotations_sha256": result.get("annotations_sha256"),
        "source_request": result.get("source_request"),
        "source_request_sha256": source_request_sha256,
        "producer_identity_sha256": result.get("producer_identity_sha256"),
        "producer_binding_eligible": result.get("producer_binding_eligible"),
        "guardrails": guardrails,
        "configured_guardrails": configured_guardrails,
        "missing_guardrails": sorted(missing_guardrails),
        "guardrail_contract_complete": guardrail_contract_complete,
        "source_guardrail_contract_complete": source_contract_complete,
        "row_status": (
            "available"
            if technical_status in {"completed", "ok", "success"}
            else "technical_failure"
        ),
    }
    for field in UNCERTAINTY_FIELDS:
        projected[f"task_quality_{field}"] = primary.get(field)
    for field in ("quality_result_source_contract_version", "quality_result_projection_contract_version", "legacy_quality_result"):
        projected[field] = result.get(field)
    for guardrail_name in ("ap50", "ap75"):
        component = _quality_component_projection(
            guardrails.get(guardrail_name)
            if isinstance(guardrails.get(guardrail_name), Mapping)
            else None
        )
        for field, value in component.items():
            projected[f"task_quality_{guardrail_name}_{field}"] = value
    return projected


def project_central_quality_status(
    summary: Mapping[str, Any] | None,
    *,
    dataset_tier: str = "",
) -> dict[str, Any]:
    """Separate execution success from the aggregate scientific decision."""

    from onnx_splitpoint_tool.quality_lifecycle import summarize_requests

    source = dict(summary or {}) if isinstance(summary, Mapping) else {}
    raw_results = [
        dict(result)
        for result in list(source.get("results") or [])
        if isinstance(result, Mapping)
    ]
    results = [
        _central_quality_result_projection(
            result,
            source_index=index,
            dataset_tier=dataset_tier,
        )
        for index, result in enumerate(raw_results)
    ]
    # Sorting makes regenerated reports deterministic.  ``source_index`` and
    # the request/fingerprint identities remain present, and setup_id is part
    # of the key, so distinct setup-local companions are never collapsed.
    results.sort(key=lambda row: (
        str(row.get("model_id") or ""),
        str(row.get("run_id") or ""),
        str(row.get("setup_id") or ""),
        str(row.get("case_id") or ""),
        str(row.get("variant") or ""),
        str(row.get("source_request_sha256") or ""),
        int(row.get("source_index") or 0),
    ))

    def _int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    request_count = _int(source.get("request_count"))
    if not request_count and results:
        request_count = len(results)
    completed_count = sum(
        1
        for row in results
        if str(row.get("technical_status") or "")
        in {"completed", "ok", "success"}
    )
    request_counts = summarize_requests(raw_results)
    failed_count = max(
        _int(source.get("failed_count")),
        request_counts["technical_failed_count"],
    )
    merge = source.get("merge") if isinstance(source.get("merge"), Mapping) else {}
    unmatched_count = _int(merge.get("unmatched_result_count"))
    result_shortfall_count = max(0, request_count - len(results))
    source_status = str(source.get("status") or "").strip().lower()
    skipped = source_status.startswith("skipped") or source_status in {
        "disabled", "not_applicable",
    }
    if not source:
        technical_status = "unavailable"
    elif skipped and request_count == 0:
        technical_status = "not_applicable"
    elif source_status == "cancelled" or request_counts["cancelled_count"]:
        technical_status = "cancelled"
    elif source_status in {"failed", "error"}:
        technical_status = "failed"
    elif (
        source_status in {"partial", "incomplete"}
        or failed_count
        or unmatched_count
        or result_shortfall_count
    ):
        technical_status = "partial"
    elif source_status in {"ok", "completed", "success"}:
        technical_status = "ok"
    else:
        technical_status = "unavailable"

    decision_counts = Counter(
        _canonical_quality_decision(row.get("task_quality_decision"))
        for row in results
    )
    all_full_results = [
        row for row in results
        if str(row.get("variant") or "").strip().lower() == "full"
    ]
    raw_identity_contract = source.get("quality_acceptance_identity_contract")
    explicit_identity_contract = raw_identity_contract is not None
    identity_contract = (
        dict(raw_identity_contract)
        if isinstance(raw_identity_contract, Mapping)
        else {}
    )
    expected_identity_rows: list[dict[str, Any]] = []
    missing_identity_rows: list[dict[str, Any]] = []
    duplicate_identity_rows: list[dict[str, Any]] = []
    unexpected_identity_rows: list[dict[str, Any]] = []
    duplicate_expected_identity_rows: list[dict[str, Any]] = []
    identity_contract_definition_errors: list[str] = []

    if explicit_identity_contract and not isinstance(
        raw_identity_contract, Mapping,
    ):
        identity_contract_definition_errors.append(
            "quality_acceptance_identity_contract_must_be_mapping"
        )
    identity_contract_schema = str(
        identity_contract.get("schema") or ""
    )
    expected_execution_scope = (
        QUALITY_ACCEPTANCE_IDENTITY_CONTRACT_SCOPES.get(
            identity_contract_schema
        )
    )
    if explicit_identity_contract and expected_execution_scope is None:
        identity_contract_definition_errors.append(
            "quality_acceptance_identity_contract_schema_mismatch"
        )
    if (
        explicit_identity_contract
        and expected_execution_scope is not None
        and str(identity_contract.get("execution_scope") or "")
        != expected_execution_scope
    ):
        scope_error = (
            "quality_acceptance_identity_contract_scope_must_be_full_only"
            if expected_execution_scope == "full_only"
            else "quality_acceptance_identity_contract_scope_must_be_"
            "standard_quality_setup_local_tensorrt"
        )
        identity_contract_definition_errors.append(scope_error)
    raw_contract_version = identity_contract.get("schema_version")
    if explicit_identity_contract and (
        type(raw_contract_version) is not int
        or raw_contract_version
        != FULL_ONLY_QUALITY_ACCEPTANCE_IDENTITY_SCHEMA_VERSION
    ):
        identity_contract_definition_errors.append(
            "quality_acceptance_identity_contract_schema_version_mismatch"
        )
    raw_identity_key_fields = identity_contract.get("identity_key_fields")
    if explicit_identity_contract and (
        not isinstance(raw_identity_key_fields, Sequence)
        or isinstance(raw_identity_key_fields, (str, bytes, bytearray))
        or list(raw_identity_key_fields)
        != list(FULL_ONLY_QUALITY_ACCEPTANCE_IDENTITY_KEY_FIELDS)
    ):
        identity_contract_definition_errors.append(
            "quality_acceptance_identity_contract_key_fields_mismatch"
        )

    def _backend_token(value: Any) -> str:
        token = str(value or "").strip().lower().replace("-", "_")
        return {
            "trt": "tensorrt",
            "ort_tensorrt": "tensorrt",
            "native_tensorrt": "tensorrt",
            "tensor_rt": "tensorrt",
            "hailo_8": "hailo8",
            "hailo10": "hailo10h",
            "hailo_10": "hailo10h",
        }.get(token, token)

    def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        performance_claims = row.get("performance_claims_emitted")
        performance_claim_token = (
            "false"
            if performance_claims is False
            else "invalid_not_literal_false:"
            + type(performance_claims).__name__
            + ":"
            + str(performance_claims).strip().lower()
        )
        return (
            str(row.get("model_id") or "").strip(),
            str(row.get("source_run_id") or row.get("run_id") or "")
            .strip().lower().replace("-", "_"),
            str(row.get("setup_id") or "").strip(),
            _backend_token(row.get("backend")),
            str(row.get("variant") or "").strip().lower().replace("-", "_"),
            str(row.get("execution_role") or "").strip(),
            performance_claim_token,
        )

    def _identity_prefix(row: Mapping[str, Any]) -> tuple[Any, ...]:
        """Project the stable producer prefix of an acceptance identity."""

        key = _identity_key(row)
        return key[:4]

    def _is_acceptance_scoped_result(row: Mapping[str, Any]) -> bool:
        return bool(
            str(row.get("variant") or "").strip().lower() == "full"
            and str(row.get("execution_role") or "").strip()
            == "full_quality_only"
            and row.get("performance_claims_emitted") is False
        )

    def _declares_acceptance_scope(row: Mapping[str, Any]) -> bool:
        """Keep malformed Full-only declarations fail-closed.

        Ordinary Generic quality rows have no Full-only execution role and
        therefore remain diagnostics.  A row that explicitly declares the
        acceptance role but violates the literal Full/False scope is not a
        diagnostic escape hatch: it is retained as unexpected evidence.
        """

        return (
            str(row.get("execution_role") or "").strip()
            == "full_quality_only"
        )

    if explicit_identity_contract:
        raw_expected = identity_contract.get("expected_identities")
        if not isinstance(raw_expected, Sequence) or isinstance(
            raw_expected, (str, bytes, bytearray),
        ):
            identity_contract_definition_errors.append(
                "quality_acceptance_expected_identities_must_be_array"
            )
            raw_expected = []
        if any(not isinstance(row, Mapping) for row in raw_expected):
            identity_contract_definition_errors.append(
                "quality_acceptance_expected_identity_must_be_mapping"
            )
        expected_base = [
            dict(row) for row in raw_expected if isinstance(row, Mapping)
        ]
        if not expected_base:
            identity_contract_definition_errors.append(
                "quality_acceptance_expected_identities_empty"
            )
        raw_model_ids = identity_contract.get("model_ids")
        if not isinstance(raw_model_ids, Sequence) or isinstance(
            raw_model_ids, (str, bytes, bytearray),
        ):
            identity_contract_definition_errors.append(
                "quality_acceptance_model_ids_must_be_array"
            )
            raw_model_ids = []
        model_ids = (
            [
                str(value).strip() for value in raw_model_ids
                if str(value).strip()
            ]
        )
        if not model_ids:
            identity_contract_definition_errors.append(
                "quality_acceptance_model_ids_empty"
            )
        for model_id in model_ids:
            for identity in expected_base:
                missing_key_fields = [
                    field for field in (
                        "source_run_id", "setup_id", "backend", "variant",
                        "execution_role", "performance_claims_emitted",
                    )
                    if field not in identity
                ]
                if missing_key_fields:
                    identity_contract_definition_errors.append(
                        "quality_acceptance_expected_identity_missing_key_fields:"
                        + str(identity.get("id") or "<unnamed>")
                        + ":"
                        + ",".join(missing_key_fields)
                    )
                    continue
                invalid_scope_fields: list[str] = []
                if str(identity.get("variant") or "").strip().lower() != "full":
                    invalid_scope_fields.append("variant")
                if str(identity.get("execution_role") or "").strip() != (
                    "full_quality_only"
                ):
                    invalid_scope_fields.append("execution_role")
                if identity.get("performance_claims_emitted") is not False:
                    invalid_scope_fields.append("performance_claims_emitted")
                if invalid_scope_fields:
                    identity_contract_definition_errors.append(
                        "quality_acceptance_expected_identity_invalid_full_only_scope:"
                        + str(identity.get("id") or "<unnamed>")
                        + ":"
                        + ",".join(invalid_scope_fields)
                    )
                    continue
                expected_row = {
                    **identity,
                    "model_id": model_id,
                }
                if not all(
                    _identity_key(expected_row)[index]
                    for index in range(7)
                ):
                    identity_contract_definition_errors.append(
                        "quality_acceptance_expected_identity_incomplete:"
                        + str(identity.get("id") or "<unnamed>")
                    )
                    continue
                expected_identity_rows.append(expected_row)
        expected_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        actual_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in expected_identity_rows:
            expected_by_key[_identity_key(row)].append(row)
        expected_identity_prefixes = {
            _identity_prefix(row) for row in expected_identity_rows
        }
        malformed_acceptance_rows: list[dict[str, Any]] = []
        for row in results:
            if _is_acceptance_scoped_result(row):
                actual_by_key[_identity_key(row)].append(row)
            elif (
                _declares_acceptance_scope(row)
                or _identity_prefix(row) in expected_identity_prefixes
            ):
                malformed_acceptance_rows.append(row)
        duplicate_expected_identity_rows = [
            dict(rows_for_key[0])
            for rows_for_key in expected_by_key.values()
            if len(rows_for_key) != 1
        ]
        full_results = []
        for key, expected_rows_for_key in expected_by_key.items():
            if len(expected_rows_for_key) != 1:
                continue
            matches = actual_by_key.get(key, [])
            if not matches:
                missing_identity_rows.append(dict(expected_rows_for_key[0]))
            else:
                # Preserve every duplicate as evidence so an observed FAIL or
                # INCONCLUSIVE remains dominant even though completeness is
                # fail-closed.
                full_results.extend(matches)
                if len(matches) != 1:
                    duplicate_identity_rows.append(dict(expected_rows_for_key[0]))
        expected_keys = set(expected_by_key)
        unexpected_identity_rows = malformed_acceptance_rows + [
            row for key, rows_for_key in actual_by_key.items()
            if key not in expected_keys for row in rows_for_key
        ]
        diagnostic_full_results = [
            row for row in all_full_results if row not in full_results
        ]
    else:
        # Compatibility projection for historical runs without an explicit
        # acceptance contract.  A setup-local Native TensorRT Full companion
        # supersedes generic ``ort_tensorrt/full`` on the same model/setup.
        native_companion_keys = {
            (
                str(row.get("model_id") or ""),
                str(row.get("setup_id") or ""),
            )
            for row in all_full_results
            if str(row.get("source_run_id") or row.get("run_id") or "")
            .strip().lower() == "native_full_tensorrt"
        }
        diagnostic_full_results = [
            row for row in all_full_results
            if str(row.get("source_run_id") or row.get("run_id") or "")
            .strip().lower() == "ort_tensorrt"
            and (
                str(row.get("model_id") or ""),
                str(row.get("setup_id") or ""),
            ) in native_companion_keys
        ]
        diagnostic_full_result_ids = {
            id(row) for row in diagnostic_full_results
        }
        full_results = [
            row for row in all_full_results
            if id(row) not in diagnostic_full_result_ids
        ]
    identity_contract_issue_count = (
        len(missing_identity_rows)
        + len(duplicate_identity_rows)
        + len(unexpected_identity_rows)
        + len(duplicate_expected_identity_rows)
        + len(identity_contract_definition_errors)
    )
    identity_contract_complete = (
        explicit_identity_contract
        and bool(expected_identity_rows)
        and identity_contract_issue_count == 0
    )
    full_decision_counts = Counter(
        _canonical_quality_decision(row.get("task_quality_decision"))
        for row in full_results
    )
    usable_full_result_count = sum(
        1
        for row in full_results
        if str(row.get("technical_status") or "")
        in {"completed", "ok", "success"}
        and _canonical_quality_decision(row.get("task_quality_decision"))
        in {"fail", "inconclusive", "pass", "reference_close", "accuracy_loss", "not_estimable"}
    )
    missing_guardrail_count = sum(
        1
        for row in results
        if row.get("guardrail_contract_complete") is False
        or bool(row.get("missing_guardrails"))
    )
    not_evaluated_count = (
        int(decision_counts.get("not_evaluated") or 0)
        + result_shortfall_count
        + unmatched_count
        + missing_guardrail_count
    )
    if not usable_full_result_count:
        aggregate_decision = "not_evaluated"
    elif full_decision_counts.get("fail"):
        aggregate_decision = "fail"
    elif full_decision_counts.get("inconclusive"):
        aggregate_decision = "inconclusive"
    elif explicit_identity_contract and not identity_contract_complete:
        aggregate_decision = "not_evaluated"
    elif not_evaluated_count or technical_status in {
        "failed", "partial", "unavailable",
    }:
        aggregate_decision = "not_evaluated"
    elif (
        full_results
        and int(full_decision_counts.get("pass") or 0)
        == len(full_results)
    ):
        aggregate_decision = "pass"
    else:
        aggregate_decision = "not_evaluated"

    # The standard setup-local TRT companions establish a Full reference.
    # Their success must not turn failed/missing accelerator or split quality
    # into a successful campaign.  Explicit Full-only canaries keep their
    # deliberately narrower acceptance scope.
    full_reference_decision = aggregate_decision
    standard_campaign = (
        expected_execution_scope == "standard_quality_setup_local_tensorrt"
    )
    evidence = (
        source.get("evidence_state_summary")
        if isinstance(source.get("evidence_state_summary"), Mapping)
        else {}
    )
    campaign_missing_count = max(
        _int(source.get("quality_missing")),
        _int(evidence.get("quality_missing")),
        max(0, _int(source.get("quality_applicable"))
            - _int(source.get("quality_completed")))
        if "quality_applicable" in source and "quality_completed" in source
        else 0,
    )
    if standard_campaign:
        if decision_counts.get("fail"):
            aggregate_decision = "fail"
        elif (
            campaign_missing_count or not_evaluated_count
            or not identity_contract_complete
            or technical_status != "ok"
            or not results
        ):
            aggregate_decision = "not_evaluated"
        elif decision_counts.get("inconclusive"):
            aggregate_decision = "inconclusive"
        elif full_reference_decision == "pass" and (
            int(decision_counts.get("pass") or 0) == len(results)
        ):
            aggregate_decision = "pass"
        else:
            aggregate_decision = "not_evaluated"

    if results and all(row.get("accuracy_assessment") for row in results) and technical_status == "ok" and not not_evaluated_count and not campaign_missing_count:
        aggregate_decision = "accuracy_loss" if decision_counts.get("accuracy_loss") else "reference_close" if decision_counts.get("reference_close") else "not_estimable"
    scientific_pass = bool(
        technical_status == "ok" and aggregate_decision == "pass"
    )
    return {
        "schema": "onnx-splitpoint/central-quality-reporting-status",
        "schema_version": 1,
        "source_present": bool(source),
        "source_status": source_status or "unavailable",
        "technical_status": technical_status,
        "quality_decision": aggregate_decision,
        "full_reference_quality_decision": full_reference_decision,
        "full_reference_scientific_pass": bool(
            technical_status == "ok" and full_reference_decision == "pass"
        ),
        "campaign_quality_decision": (
            aggregate_decision if standard_campaign else "not_applicable"
        ),
        "campaign_quality_missing_count": (
            campaign_missing_count if standard_campaign else 0
        ),
        "scientific_status": (
            aggregate_decision
            if technical_status in {"ok", "not_applicable"}
            else "not_evaluated"
        ),
        "scientific_pass": scientific_pass,
        "request_count": request_count,
        "result_count": len(results),
        "completed_count": completed_count,
        "terminal_count": request_counts["terminal_count"],
        "evaluated_count": request_counts["evaluated_count"],
        "cancelled_count": request_counts["cancelled_count"],
        "technical_failed_count": request_counts["technical_failed_count"],
        "queued_count": _int(source.get("queued_count")),
        "running_count": _int(source.get("running_count")),
        "quality_decision_counts": request_counts["quality_decision_counts"],
        "quality_uncertainty_counts": request_counts["quality_uncertainty_counts"],
        "failed_count": failed_count,
        "unmatched_result_count": unmatched_count,
        "result_shortfall_count": result_shortfall_count,
        "missing_guardrail_count": missing_guardrail_count,
        "not_evaluated_count": not_evaluated_count,
        "decision_counts": dict(sorted(decision_counts.items())),
        "quality_acceptance_identity_contract": (
            identity_contract if explicit_identity_contract else None
        ),
        "aggregate_full_result_count": len(full_results),
        "aggregate_selection_mode": (
            "explicit_full_only_identity_contract"
            if explicit_identity_contract else "historical_full_compatibility"
        ),
        "aggregate_expected_full_result_count": len(expected_identity_rows),
        "aggregate_identity_contract_complete": (
            identity_contract_complete if explicit_identity_contract else None
        ),
        "aggregate_identity_contract_issue_count": identity_contract_issue_count,
        "aggregate_missing_identity_count": len(missing_identity_rows),
        "aggregate_duplicate_identity_count": len(duplicate_identity_rows),
        "aggregate_unexpected_identity_count": len(unexpected_identity_rows),
        "aggregate_duplicate_expected_identity_count": len(
            duplicate_expected_identity_rows
        ),
        "aggregate_identity_contract_definition_errors": (
            identity_contract_definition_errors
        ),
        "aggregate_missing_identities": missing_identity_rows,
        "aggregate_duplicate_identities": duplicate_identity_rows,
        "aggregate_unexpected_identity_result_ids": [
            str(row.get("quality_result_id") or "")
            for row in unexpected_identity_rows
        ],
        "aggregate_all_full_result_count": len(all_full_results),
        "aggregate_excluded_diagnostic_full_result_count": len(
            diagnostic_full_results
        ),
        "aggregate_excluded_diagnostic_full_result_ids": [
            str(row.get("quality_result_id") or "")
            for row in diagnostic_full_results
        ],
        "aggregate_usable_full_result_count": usable_full_result_count,
        "aggregate_decision_counts": dict(
            sorted(full_decision_counts.items())
        ),
        "results_complete": bool(
            technical_status == "ok"
            and request_count == len(results)
            and not not_evaluated_count
            and not (standard_campaign and campaign_missing_count)
        ),
        "results": results,
        "status_semantics": {
            "technical_status": (
                "Execution/computation completeness only; metric decisions do "
                "not turn a technically completed run into a crash."
            ),
            "quality_decision": (
                "Standard campaigns include all evaluated Full/split decisions "
                "and required quality coverage. Explicit Full-only canaries "
                "retain their exact acceptance scope. The Full reference "
                "decision is reported separately."
            ),
        },
    }


def _normalize_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


def _same_sha256(left: Any, right: Any) -> bool:
    a = _normalize_sha256(left)
    b = _normalize_sha256(right)
    return bool(a and b and a == b)


def _candidate_prediction_map(prediction: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for candidate in list(prediction.get("candidates") or []):
        if not isinstance(candidate, Mapping):
            continue
        case_id = str(candidate.get("case_id") or "")
        if not case_id and candidate.get("boundary") is not None:
            try:
                case_id = f"b{int(candidate.get('boundary')):03d}"
            except Exception:
                case_id = ""
        if case_id:
            out[case_id] = dict(candidate)
    return out


def _load_prediction_freeze(
    model_dir: Path,
    prediction_path: Path,
    *,
    require_candidate_universe: bool = False,
) -> dict[str, Any]:
    """Load and hash-verify the prospective prediction and method freeze."""
    analysis_dir = model_dir / "analysis"
    manifest_path = analysis_dir / "prediction_freeze_manifest.json"
    manifest = read_json(manifest_path, default={}) or {}
    if not isinstance(manifest, Mapping) or not manifest:
        return {
            "present": False,
            "valid": False,
            "prospective": False,
            "valid_for_holdout": False,
            "status": "missing",
            "manifest_path": "",
            "ranking_predictions_valid": False,
            "ranking_method_predictions": [],
        }
    prediction_csv = analysis_dir / str(manifest.get("prediction_csv") or "predictions_frozen.csv")
    ranking_csv = analysis_dir / str(manifest.get("ranking_prediction_csv") or "ranking_predictions_frozen.csv")
    universe_declared = bool(
        manifest.get("candidate_universe_manifest")
        or manifest.get("candidate_universe_sha256")
    )
    universe_required = bool(
        require_candidate_universe
        or is_confirmatory_holdout(manifest.get("evaluation_role"))
        or _b(manifest.get("valid_for_holdout")) is True
    )
    universe_path = analysis_dir / str(
        manifest.get("candidate_universe_manifest") or "candidate_universe_manifest.json"
    )
    universe_csv = analysis_dir / str(
        manifest.get("candidate_universe_csv") or "candidate_universe.csv"
    )
    universe = read_json(universe_path, default={}) or {}
    prediction_hash = sha256_file(prediction_path) if prediction_path.is_file() else None
    prediction_csv_hash = sha256_file(prediction_csv) if prediction_csv.is_file() else None
    ranking_csv_hash = sha256_file(ranking_csv) if ranking_csv.is_file() else None
    core_hash_ok = bool(
        prediction_hash
        and prediction_csv_hash
        and _same_sha256(manifest.get("prediction_sha256"), prediction_hash)
        and _same_sha256(manifest.get("prediction_csv_sha256"), prediction_csv_hash)
    )
    ranking_hash_ok = bool(
        ranking_csv_hash
        and _same_sha256(manifest.get("ranking_prediction_csv_sha256"), ranking_csv_hash)
    )
    if universe_declared and isinstance(universe, Mapping) and universe:
        universe_self_hash = sha256_json({
            key: value
            for key, value in universe.items()
            if key not in {"universe_sha256", "created_at"}
        })
        universe_manifest_valid = bool(
            str(universe.get("universe_sha256") or "") == universe_self_hash
            and str(manifest.get("candidate_universe_sha256") or "") == universe_self_hash
        )
    else:
        universe_self_hash = ""
        universe_manifest_valid = False
    universe_hash_ok = bool(
        universe_manifest_valid
        or (not universe_declared and not universe_required)
    )
    expected_universe_csv_hash = str(manifest.get("candidate_universe_csv_sha256") or "")
    universe_csv_hash = sha256_file(universe_csv) if universe_csv.is_file() else None
    universe_csv_hash_ok = bool(
        not expected_universe_csv_hash
        or (universe_csv_hash and _same_sha256(expected_universe_csv_hash, universe_csv_hash))
    )
    prospective = bool(manifest.get("prospective"))
    conflict = (analysis_dir / "prediction_freeze_conflict.json").is_file()
    valid = bool(
        core_hash_ok
        and ranking_hash_ok
        and universe_hash_ok
        and universe_csv_hash_ok
        and prospective
        and not conflict
    )
    status = str(manifest.get("freeze_status") or ("prospective_frozen" if valid else "invalid"))
    if conflict:
        status = "hash_conflict"
    elif not core_hash_ok:
        status = "hash_mismatch"
    elif not ranking_hash_ok:
        status = "ranking_prediction_hash_mismatch"
    elif not universe_hash_ok:
        status = (
            "candidate_universe_required_missing"
            if universe_required and not universe_declared
            else "candidate_universe_hash_mismatch"
        )
    elif not universe_csv_hash_ok:
        status = "candidate_universe_csv_hash_mismatch"
    elif not prospective:
        status = "retrospective_not_valid"
    audit = universe.get("audit") if isinstance(universe, Mapping) and isinstance(universe.get("audit"), Mapping) else {}
    try:
        audit_minimum_valid_candidates = max(0, int(audit.get("minimum_valid_candidates") or 0))
    except Exception:
        audit_minimum_valid_candidates = 0
    return {
        **dict(manifest),
        "present": True,
        "valid": valid,
        "prospective": prospective,
        "valid_for_holdout": bool(valid and manifest.get("valid_for_holdout")),
        "status": status,
        "hash_ok": core_hash_ok,
        "ranking_predictions_valid": ranking_hash_ok,
        "candidate_universe_required": universe_required,
        "candidate_universe_valid": universe_manifest_valid and universe_csv_hash_ok,
        "candidate_universe_manifest_path": relpath(universe_path, model_dir) if universe_path.is_file() else "",
        "candidate_universe_csv_path": relpath(universe_csv, model_dir) if universe_csv.is_file() else "",
        "candidate_universe_mode": str(universe.get("mode") or "") if isinstance(universe, Mapping) else "",
        "candidate_universe_scope": str(universe.get("claim_scope") or manifest.get("candidate_universe_scope") or "") if isinstance(universe, Mapping) else str(manifest.get("candidate_universe_scope") or ""),
        "candidate_universe_complete": (
            manifest.get("candidate_universe_complete")
            if "candidate_universe_complete" in manifest
            else (
                universe.get("declared_complete")
                if isinstance(universe, Mapping)
                else None
            )
        ),
        "candidate_universe_selected_case_ids": [
            str(value)
            for value in list(universe.get("selected_case_ids") or [])
            if str(value)
        ] if isinstance(universe, Mapping) else [],
        "candidate_universe_minimum_valid_candidates": audit_minimum_valid_candidates,
        "manifest_path": relpath(manifest_path, model_dir),
        "prediction_csv_path": relpath(prediction_csv, model_dir) if prediction_csv.is_file() else "",
        "ranking_prediction_csv_path": relpath(ranking_csv, model_dir) if ranking_csv.is_file() else "",
        "ranking_method_predictions": _read_csv_rows(ranking_csv),
    }


def _load_evalrun_rows(
    run_dir: Path,
    profile: Mapping[str, Any],
    policy: AccuracyGatePolicy,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    entries = _profile_model_entries(profile)
    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, Any]] = {}
    model_facts: list[dict[str, Any]] = []
    run_manifest = read_json(run_dir / "run_manifest.json", default={}) or {}
    evaluation_run_id = str(run_manifest.get("run_id") or run_dir.name).strip()
    models_root = run_dir / "models"
    if not models_root.is_dir():
        return rows, predictions, model_facts
    for model_dir in sorted(path for path in models_root.iterdir() if path.is_dir()):
        model_id = model_dir.name
        manifest = read_json(model_dir / "model_manifest.json", default={}) or {}
        entry = dict(entries.get(model_id) or manifest.get("profile_entry") or {})
        task = _normalized_scientific_task(
            entry.get("task") or manifest.get("task"), model_id
        )
        role = normalize_evaluation_role(entry.get("evaluation_role") or "development")
        prediction_path = model_dir / "analysis" / "prediction.json"
        prediction = dict(read_json(prediction_path, default={}) or {})
        ranking_block = profile.get("ranking_validation") if isinstance(profile.get("ranking_validation"), Mapping) else {}
        prediction["_prediction_freeze"] = _load_prediction_freeze(
            model_dir,
            prediction_path,
            require_candidate_universe=bool(
                is_confirmatory_holdout(role)
                or ranking_block.get("require_complete_candidate_universe") is True
            ),
        )
        prediction["_ranking_method_predictions"] = list(
            (prediction["_prediction_freeze"] or {}).get("ranking_method_predictions") or []
        )
        predictions[model_id] = prediction
        prediction_map = _candidate_prediction_map(prediction)
        normalized = read_json(model_dir / "benchmark_results" / "normalized_results.json", default={}) or {}
        for source in list(normalized.get("results") or []):
            if not isinstance(source, Mapping):
                continue
            row = dict(source)
            if str(row.get("model_id") or "").strip().lower() in {"", "model"}:
                row["model_id"] = model_id
            if not str(row.get("task") or "").strip():
                row["task"] = _normalized_scientific_task(
                    _first(
                        row,
                        (
                            "benchmark_task_used",
                            "benchmark_task_requested",
                            "validation_dataset_task",
                        ),
                    )
                    or task,
                    model_id,
                )
            if not str(row.get("run_id") or "").strip():
                row["run_id"] = str(
                    _first(row, ("source_tag", "benchmark_run_id", "run_profile_id"))
                    or row.get("backend")
                    or evaluation_run_id
                ).strip()
            row["evaluation_role"] = role
            row["dataset_tier"] = str(entry.get("validation_tier") or entry.get("dataset_tier") or policy.dataset_tier)
            candidate = prediction_map.get(str(row.get("case_id") or "")) or {}
            for destination, keys in {
                "prediction_rank": ("rank", "source_rank"),
                "predicted_cycle_ms": ("predicted_stream_cycle_ms_calibrated", "predicted_stream_cycle_ms", "predicted_total_latency_ms"),
                "predicted_fps": ("predicted_stream_fps_calibrated", "predicted_stream_fps"),
                "predicted_latency_ms": ("predicted_total_latency_ms", "pred_latency_total_ms"),
                "predicted_handover_ms": ("predicted_handover_ms_calibrated", "predicted_handover_ms"),
                "prediction_score": ("score_pred", "accelerator_fit_score"),
            }.items():
                if row.get(destination) in (None, ""):
                    row[destination] = _first(candidate, keys)
            apply_accuracy_gate_to_row(row, policy)
            rows.append(row)
        facts = (
            read_json(model_dir / "analysis" / "model_facts.json", default={})
            or read_json(model_dir / "model_manifest.json", default={})
            or {}
        )
        model_facts.append(
            {
                "model_id": model_id,
                "task": task,
                "evaluation_role": role,
                **{
                    key: facts.get(key)
                    for key in ("family", "node_count", "parameter_count", "gflops", "model_sha256", "resolved_path")
                },
            }
        )
    return _dedupe_rows(rows), predictions, model_facts


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for item in range(position, end):
            result[order[item]] = rank
        position = end
    return result


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [value - mean_x for value in x], [value - mean_y for value in y]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denominator if denominator > 0 else None


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return _pearson(_average_ranks(x), _average_ranks(y))


def _kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for left in range(len(x)):
        for right in range(left + 1, len(x)):
            sign_x = (x[left] > x[right]) - (x[left] < x[right])
            sign_y = (y[left] > y[right]) - (y[left] < y[right])
            if sign_x == 0 and sign_y == 0:
                continue
            if sign_x == 0:
                ties_x += 1
            elif sign_y == 0:
                ties_y += 1
            elif sign_x == sign_y:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denominator if denominator > 0 else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(numbers) / len(numbers) if numbers else None


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    numbers = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not numbers:
        return None
    count = len(numbers)
    return numbers[count // 2] if count % 2 else (numbers[count // 2 - 1] + numbers[count // 2]) / 2.0


def _runner_regime(row: Mapping[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) if row.get(key) is not None else "").lower()
        for key in ("runner_regime", "run_id", "backend", "runner", "source", "validation_claim_level")
    )
    return canonical_runner(text)


def _direction(row: Mapping[str, Any]) -> str:
    return canonical_direction(
        str(row.get("direction") or row.get("backend") or row.get("run_id") or ""),
        stage1=str(row.get("stage1_provider") or ""),
        stage2=str(row.get("stage2_provider") or ""),
    )


def _measured_cycle(row: Mapping[str, Any]) -> Optional[float]:
    return _f(
        _first(
            row,
            (
                "pipeline_cycle_selected_ms",
                "native_measured_cycle_ms",
                "cycle_ms",
                "total_latency_ms",
                "split_latency_e2e_ms",
            ),
        )
    )


def _stratum_text(value: Any) -> str:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    parsed = _b(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if parsed is not None and str(value or "").strip().lower() in {
        "true", "false", "yes", "no", "1", "0",
    }:
        return "true" if parsed else "false"
    return str(value or "").strip()


def _comparison_output_endpoint_id(row: Mapping[str, Any]) -> str:
    """Return the completed-task endpoint, falling back to legacy aliases."""

    return str(
        _first(
            row,
            (
                "completed_task_comparison_output_endpoint_id",
                "completed_task_output_endpoint_id",
                "comparison_output_endpoint_id",
                "physical_output_endpoint_id",
                "output_endpoint_id",
            ),
        )
        or ""
    ).strip()


def _comparison_endpoint_contract_hash(row: Mapping[str, Any]) -> str:
    """Return the completed-task endpoint contract, with legacy fallback."""

    return str(
        _first(
            row,
            (
                "completed_task_comparison_endpoint_contract_hash",
                "completed_task_endpoint_contract_hash",
                "endpoint_contract_hash",
            ),
        )
        or ""
    ).strip()


def _case_local_native_split_stratum(
    row: Mapping[str, Any], value: Any,
) -> bool:
    """Identify the legacy ``[model, case, native_split]`` identity.

    That tuple identifies an observation, not a comparison endpoint.  Keeping
    it in the group key makes every split candidate a singleton.  It is safe
    to ignore only when the row carries a complete canonical comparison
    identity; otherwise retaining it is the fail-closed behaviour.
    """

    if canonical_runner(_runner_regime(row)) != "native_fifo":
        return False
    items: list[Any]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray),
    ):
        items = list(value)
    elif isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return False
        if not isinstance(parsed, list):
            return False
        items = parsed
    else:
        return False
    if len(items) != 3:
        return False
    model_id = str(row.get("model_id") or row.get("model") or "").strip()
    case_id = str(row.get("case_id") or row.get("case") or "").strip()
    variant = str(items[2] or "").strip().lower()
    if (
        str(items[0] or "").strip() != model_id
        or str(items[1] or "").strip() != case_id
        or variant not in {"split", "composed", "native_split", "native_producer"}
    ):
        return False
    endpoint_id = _comparison_output_endpoint_id(row)
    endpoint_hash = _comparison_endpoint_contract_hash(row)
    e2e_scope = str(row.get("e2e_scope") or "").strip().lower()
    host_required = _b(row.get("host_postprocess_required"))
    if (
        not endpoint_id
        or not endpoint_hash
        or e2e_scope in {"", "unknown", "unavailable"}
        or host_required is None
    ):
        return False
    if host_required is True and (
        _b(row.get("host_postprocessing_available")) is not True
        or not str(
            row.get("frozen_host_postprocess_contract_sha256") or ""
        ).strip()
    ):
        return False
    return True


def _comparison_endpoint_stratum(row: Mapping[str, Any]) -> str:
    explicit = row.get("comparison_endpoint_stratum")
    if _stratum_text(explicit):
        return _stratum_text(explicit)
    legacy = row.get("output_endpoint_comparison_stratum")
    if _case_local_native_split_stratum(row, legacy):
        return ""
    return _stratum_text(legacy)


def _resolved_report_setup_id_v27550(row: Mapping[str, Any]) -> str:
    """Project one exact setup identity without guessing across setups."""

    explicit = str(
        _first(
            row,
            ("setup_id", "measurement_setup_id", "source_setup_id", "hardware_setup_id"),
        )
        or ""
    ).strip()
    if explicit:
        return explicit
    candidates: set[str] = set()
    for value in list(row.get("quality_source_setup_ids") or []):
        text = str(value or "").strip()
        if text:
            candidates.add(text)
    identities = row.get("quality_request_identities_by_variant")
    if isinstance(identities, Mapping):
        for identity in identities.values():
            if not isinstance(identity, Mapping):
                continue
            for value in list(identity.get("setup_ids") or []):
                text = str(value or "").strip()
                if text:
                    candidates.add(text)
            text = str(identity.get("setup_id") or "").strip()
            if text:
                candidates.add(text)
    return next(iter(candidates)) if len(candidates) == 1 else ""


def _ranking_stratum(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": str(row.get("model_id") or "").strip(),
        "direction": _direction(row),
        "runner_regime": _runner_regime(row),
        "setup_id": _resolved_report_setup_id_v27550(row),
        "setup_identity_status": row.get("setup_identity_status"),
        "setup_identity_candidates": row.get("setup_identity_candidates") or [],
        "runtime_precision_identity": str(
            _first(
                row,
                (
                    "runtime_precision_identity",
                    "execution_precision",
                    "full_runtime_precision",
                    "precision",
                ),
            )
            or ""
        ).strip(),
        "comparison_backend": str(row.get("comparison_backend") or "").strip(),
        "comparison_output_endpoint_id": _comparison_output_endpoint_id(row),
        "comparison_endpoint_contract_hash": (
            _comparison_endpoint_contract_hash(row)
        ),
        "e2e_scope": str(row.get("e2e_scope") or "").strip(),
        "comparison_endpoint_stratum": _comparison_endpoint_stratum(row),
        "measurement_concurrency": _stratum_text(
            row.get("measurement_concurrency")
        ),
        "completed_task_stage": str(
            row.get("completed_task_stage") or ""
        ).strip(),
        "completed_task_completion_mode": str(
            row.get("completed_task_completion_mode") or ""
        ).strip(),
        "frozen_host_postprocess_contract_sha256": str(
            row.get("frozen_host_postprocess_contract_sha256") or ""
        ).strip(),
        "host_postprocessing_available": _stratum_text(
            row.get("host_postprocessing_available")
        ),
        "host_postprocess_required": _stratum_text(
            row.get("host_postprocess_required")
        ),
    }


def _ranking_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("variant") or "").lower()
        if variant not in {"split", "composed", "native_split", "native_producer"}:
            continue
        if _measured_cycle(row) is None:
            continue
        direction = _direction(row)
        if not direction or "_to_" not in direction:
            continue
        stratum = _ranking_stratum(row)
        groups[tuple(stratum[field] for field in RANKING_COMPARISON_STRATA)].append(
            dict(row)
        )
    return groups


def _ranking_quality_decision(row: Mapping[str, Any]) -> str:
    """Resolve contradictory aliases with fail/inconclusive dominance."""

    if isinstance(row.get("accuracy_assessment"), Mapping):
        return str(row["accuracy_assessment"].get("accuracy_class") or "not_estimable")
    decisions: list[str] = []
    for key in (
        "task_quality_decision",
        "accuracy_gate_decision",
        "task_quality_status",
        "quality_gate_status",
    ):
        raw = str(row.get(key) if row.get(key) is not None else "").strip().lower().replace("-", "_")
        aliases = {
            "quality_pass": "pass",
            "quality_passed": "pass",
            "quality_failed": "fail",
            "quality_fail": "fail",
            "quality_inconclusive": "inconclusive",
        }
        decision = _canonical_quality_decision(aliases.get(raw, raw))
        if decision != "not_evaluated":
            decisions.append(decision)
    if "fail" in decisions:
        return "fail"
    if "inconclusive" in decisions:
        return "inconclusive"
    if "pass" in decisions:
        return "pass"
    if any(
        _b(row.get(key)) is False
        for key in (
            "task_quality_pass",
            "quality_accuracy_gate_pass",
            "accuracy_gate_pass",
        )
        if row.get(key) not in (None, "")
    ):
        return "fail"
    return "not_evaluated"


def _ranking_quality_vetoed(row: Mapping[str, Any]) -> bool:
    return _ranking_quality_decision(row) in {"fail", "inconclusive"}


def _technical_ranking_measurement(row: Mapping[str, Any]) -> bool:
    """Require a technically successful numeric observation for ranking."""

    if _measured_cycle(row) is None:
        return False
    if _b(row.get("measurement_valid")) is False:
        return False
    if _b(row.get("terminal_failure")) is True:
        return False
    return bool(
        _b(row.get("runtime_executable")) is True
        or _b(row.get("ok")) is True
        or _b(row.get("ranking_eligible")) is True
    )


def _ranking_sensitivity_cohort_eligible(
    row: Mapping[str, Any], cohort: str,
) -> bool:
    if not _technical_ranking_measurement(row):
        return False
    if cohort == "technical":
        return True
    decision = _ranking_quality_decision(row)
    if cohort == "quality_pass":
        return decision == "pass"
    if cohort == "quality_pass_or_inconclusive":
        return decision in {"pass", "inconclusive"}
    raise ValueError(f"Unknown ranking sensitivity cohort: {cohort}")


def _ranking_exclusion_reasons_for_analysis(
    row: Mapping[str, Any],
) -> set[str]:
    reasons: set[str] = set()
    for key in (
        "ranking_exclusion_reasons",
        "ranking_exclusion_reason",
        "performance_claim_exclusion_reasons",
        "exclusion_reason",
        "eligibility_gate_reason",
        "accuracy_gate_reason",
        "gate_status",
    ):
        reasons.update(_claim_exclusion_tokens(row.get(key)))
    return {
        str(reason).strip()
        for reason in reasons
        if str(reason).strip()
        and str(reason).strip().lower() not in {"eligible", "available"}
    }


def _development_screening_ranking_exception(
    row: Mapping[str, Any],
    *,
    role: str,
    profile: Mapping[str, Any],
    audit_scoped: bool,
) -> bool:
    """Admit one Generic Standard PASS to development-only audit analysis.

    This does not change claim eligibility on the source/scientific row.  It
    only prevents the screening dataset tier from erasing otherwise valid
    measurements in an explicitly score-independent, predeclared development
    audit.  Every scope check is fail-closed so ordinary Standard screening
    rows never become ranking evidence through this reporting-only exception.
    """

    if audit_scoped is not True:
        return False
    if normalize_evaluation_role(role) != "development":
        return False
    if canonical_runner(_runner_regime(row)) != "generic":
        return False
    selection = (
        profile.get("selection_policy")
        if isinstance(profile.get("selection_policy"), Mapping)
        else {}
    )
    strategy = (
        str(selection.get("selection_strategy") or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if strategy != "score_independent_audit":
        return False
    # The flag is written alongside the strategy by current profiles.  Treat
    # an explicit contradiction as invalid while retaining old profiles that
    # predate the compatibility flag but already carry the exact strategy.
    if _b(selection.get("score_independent_audit_enabled")) is False:
        return False
    execution_preset = (
        profile.get("execution_preset")
        if isinstance(profile.get("execution_preset"), Mapping)
        else {}
    )
    # The resolved execution-preset snapshot is authoritative.  A missing id
    # must not silently inherit Standard semantics during offline reporting.
    explicit_mode = str(execution_preset.get("id") or "").strip().lower()
    if explicit_mode != "standard":
        return False
    if _ranking_quality_decision(row) != "pass":
        return False
    if any(
        _b(row.get(key)) is not True
        for key in (
            "buildable",
            "runtime_executable",
            "contract_consistent",
            "accuracy_gate_pass",
        )
    ):
        return False
    if _b(row.get("quality_policy_match")) is False or _b(
        row.get("accuracy_gate_policy_match")
    ) is False:
        return False
    if _b(row.get("precision_quality_binding_verified")) is False:
        return False
    if _b(row.get("numerical_similarity_pass")) is False:
        return False
    return _ranking_exclusion_reasons_for_analysis(row) == {"screening_only"}


def _ranking_analysis_eligible(
    row: Mapping[str, Any],
    *,
    role: str,
    profile: Mapping[str, Any],
    audit_scoped: bool,
) -> bool:
    # Explicit quality FAIL/INCONCLUSIVE always dominates a stale positive
    # boolean in imported or producer-shaped rows.
    if _ranking_quality_vetoed(row):
        return False
    if _b(row.get("ranking_eligible")) is True:
        return True
    return _development_screening_ranking_exception(
        row,
        role=role,
        profile=profile,
        audit_scoped=audit_scoped,
    )


def _native_ranking_base_identity(
    row: Mapping[str, Any],
) -> tuple[str, ...] | None:
    stratum = _ranking_stratum(row)
    case_id = str(row.get("case_id") or row.get("case") or "").strip()
    values = (
        stratum["model_id"],
        stratum["direction"],
        stratum["runner_regime"],
        case_id,
        stratum["setup_id"],
        stratum["runtime_precision_identity"],
        stratum["comparison_backend"],
    )
    if stratum["runner_regime"] != "native_fifo" or not all(values):
        return None
    return values


def _merge_native_ranking_rows(
    rows: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich or append Native rows without creating duplicate strata.

    Older normalized Native projections can carry the stable execution
    identity but omit newer endpoint/host-tail stratum fields.  A compatible
    authoritative Native matrix observation enriches that row in place.  A
    conflicting complete stratum remains a separate measurement context.
    """

    merged = [dict(row) for row in rows if isinstance(row, Mapping)]
    for source in native_rows:
        native = dict(source)
        base_identity = _native_ranking_base_identity(native)
        if base_identity is None:
            merged.append(native)
            continue
        native_stratum = _ranking_stratum(native)
        compatible_indices: list[int] = []
        for index, existing in enumerate(merged):
            if _native_ranking_base_identity(existing) != base_identity:
                continue
            existing_stratum = _ranking_stratum(existing)
            if all(
                not existing_stratum[field]
                or not native_stratum[field]
                or existing_stratum[field] == native_stratum[field]
                for field in RANKING_COMPARISON_STRATA
            ):
                compatible_indices.append(index)
        if compatible_indices:
            compatible_strata = [
                _ranking_stratum(merged[index])
                for index in compatible_indices
            ]
            mutually_compatible = all(
                all(
                    not left[field]
                    or not right[field]
                    or left[field] == right[field]
                    for field in RANKING_COMPARISON_STRATA
                )
                for left_index, left in enumerate(compatible_strata)
                for right in compatible_strata[left_index + 1 :]
            )
            if not mutually_compatible:
                # An incomplete matrix projection can be compatible with two
                # otherwise distinct endpoint/host-tail strata.  There is no
                # score-independent way to choose one of them, and merging
                # both would silently destroy a comparison context.  Retain
                # the already-normalized rows unchanged and ignore this
                # ambiguous enrichment.
                continue
            index = compatible_indices[0]
            enriched: dict[str, Any] = dict(merged[index])
            for compatible_index in compatible_indices:
                for key, value in merged[compatible_index].items():
                    current = enriched.get(key)
                    if current is None or (
                        isinstance(current, str) and not current.strip()
                    ):
                        enriched[key] = value
            # The matrix is the authoritative Native projection for cycle and
            # row-local quality; existing normalized-only fields are retained.
            # Empty matrix aliases must not erase a complete endpoint stratum.
            native_nonempty = {
                key: value
                for key, value in native.items()
                if value is not None
                and not (isinstance(value, str) and not value.strip())
            }
            merged[index] = {**enriched, **native_nonempty}
            for duplicate_index in reversed(compatible_indices[1:]):
                del merged[duplicate_index]
        else:
            merged.append(native)
    return merged


def _method_rows_for_context(
    prediction: Mapping[str, Any],
    *,
    direction: str,
    runner: str,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    frozen = [
        dict(row)
        for row in list(prediction.get("_ranking_method_predictions") or [])
        if isinstance(row, Mapping)
    ]
    matched = [
        row
        for row in frozen
        if canonical_direction(row.get("direction")) == canonical_direction(direction)
        and canonical_runner(row.get("runner_regime")) == canonical_runner(runner)
    ]
    if matched:
        return matched, "frozen"
    candidates = [dict(candidate) for candidate in list(prediction.get("candidates") or []) if isinstance(candidate, Mapping)]
    if not candidates:
        return [], "unavailable"
    normalized_direction = canonical_direction(direction)
    stage1, stage2 = (normalized_direction.split("_to_", 1) + [""])[:2]
    derived = compute_ranking_predictions(
        candidates,
        [{"direction": normalized_direction, "stage1": stage1, "stage2": stage2, "runner_regime": runner}],
        policy.get("method_policy") if isinstance(policy.get("method_policy"), Mapping) else {},
    )
    return [dict(row) for row in derived], "posthoc_derived"


def _candidate_universe_status(
    model_id: str,
    measured_ids: set[str],
    expected_ids: set[str],
    profile: Mapping[str, Any],
    ranking_policy: Mapping[str, Any],
    freeze: Mapping[str, Any] | None = None,
) -> tuple[bool, str, bool, str, int]:
    freeze = freeze if isinstance(freeze, Mapping) else {}
    frozen_complete = _b(freeze.get("candidate_universe_complete"))
    frozen_valid = _b(freeze.get("candidate_universe_valid"))
    expected = len(expected_ids)
    if frozen_complete is not None:
        if frozen_complete is not True:
            return (
                False,
                "frozen_candidate_universe_not_declared_complete",
                False,
                "measurement_coverage_not_interpretable_without_declared_universe",
                expected,
            )
        if frozen_valid is not True:
            return (
                False,
                "frozen_candidate_universe_not_verified",
                False,
                "measurement_coverage_not_interpretable_without_verified_universe",
                expected,
            )
        if not expected_ids:
            return (
                False,
                "declared_complete_but_frozen_universe_missing",
                False,
                "expected_candidate_ids_missing",
                0,
            )
        missing = expected_ids - measured_ids
        return (
            True,
            "declared_complete_and_verified",
            not missing,
            "complete" if not missing else f"missing_{len(missing)}_of_{expected}",
            expected,
        )
    entry = _profile_model_entries(profile).get(model_id) or {}
    explicit = entry.get("candidate_universe_complete")
    if explicit is None:
        raw = profile.get("ranking_validation") if isinstance(profile.get("ranking_validation"), Mapping) else {}
        explicit = raw.get("candidate_universe_complete")
    if explicit is True:
        if not expected_ids:
            return (
                False,
                "declared_complete_but_candidate_ids_missing",
                False,
                "expected_candidate_ids_missing",
                0,
            )
        missing = expected_ids - measured_ids
        return (
            True,
            "declared_complete",
            not missing,
            "complete" if not missing else f"missing_{len(missing)}_of_{expected}",
            expected,
        )
    return (
        False,
        (
            "not_declared_complete_but_not_required_by_policy"
            if not bool(ranking_policy.get("require_complete_candidate_universe", True))
            else "not_declared_complete"
        ),
        bool(expected_ids and expected_ids <= measured_ids),
        f"measured_{len(measured_ids)}_of_{expected or 'unknown'}",
        expected,
    )


def _ranking_observed_scope(rows, profile, policy):
    """Describe distinct candidates using the same report strata and eligibility."""
    minimum = max(2, int(policy.get("minimum_candidates_for_correlation") or 3))
    entries = _profile_model_entries(profile)
    groups = []
    for key, observations in sorted(_ranking_groups(rows).items()):
        stratum = dict(zip(RANKING_COMPARISON_STRATA, key))
        role = normalize_evaluation_role((entries.get(stratum["model_id"]) or {}).get("evaluation_role") or "development")
        by_case = defaultdict(list)
        for row in observations:
            if row.get("case_id"):
                by_case[str(row["case_id"])].append(row)
        valid, excluded = [], []
        for case, repeats in sorted(by_case.items()):
            # A duplicate is never an extra candidate. Conflicting observations
            # remain excluded from this descriptive scope; no best-of selection.
            eligibility = [_ranking_analysis_eligible(row, role=role, profile=profile, audit_scoped=False)
                           and _measured_cycle(row) is not None for row in repeats]
            if all(eligibility):
                valid.append(case)
            else:
                excluded.append({"case_id": case, "reason": "technical_or_quality_or_binding_not_eligible",
                                 "observed_reasons": sorted({str(row.get("failure_reason") or row.get("status") or "") for row in repeats})})
        groups.append({**stratum, "distinct_candidate_count": len(by_case),
            "valid_candidate_count": len(valid), "valid_candidate_ids": valid,
            "excluded_candidates": excluded, "minimum_candidates": minimum,
            "status": "candidate_count_sufficient_other_gates_required" if len(valid) >= minimum else "insufficient_candidates"})
    return {"groups": groups, "group_count": len(groups),
            "measured_model_count": len({str(row.get("model_id")) for row in rows if row.get("model_id")}),
            "candidate_count_semantics": "distinct_case_per_existing_comparison_stratum",
            "correlation_claim_from_counts_alone": False}


def _ranking_method_comparison(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    profile: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from ..execution_plan import profile_experiment_purpose
    purpose = profile_experiment_purpose(profile)
    if purpose in {"coverage_integration", "deepx_preprocessing_ab"}:
        scope = _ranking_observed_scope(rows, profile, policy)
        return [], [], {
            "status": "not_requested_for_integration", "experiment_purpose": purpose,
            "method_count": 0, "group_method_row_count": 0, "best_method_id": "",
            "best_method_label": "", "best_method_basis": "not_requested_for_integration",
            "cohort_sensitivity_rows": [], "ranking_scope": scope,
            "metric_scope_note": "Coverage/integration does not request Spearman, Kendall or Top-k ranking claims.",
        }
    details: list[dict[str, Any]] = []
    cohort_sensitivity_rows: list[dict[str, Any]] = []
    k_values = list(policy.get("k_values") or [1, 3, 5])
    elite_q_values = list(policy.get("elite_q_values") or [1, 3])
    min_candidates = int(policy.get("minimum_candidates_for_correlation") or 3)
    epsilon = float(policy.get("near_optimal_relative_epsilon") or 0.01)
    methods = list(policy.get("methods") or METHOD_ORDER)
    entries = _profile_model_entries(profile)

    for group_key, source_group in sorted(_ranking_groups(rows).items()):
        stratum = dict(zip(RANKING_COMPARISON_STRATA, group_key))
        model_id = stratum["model_id"]
        direction = stratum["direction"]
        runner = stratum["runner_regime"]
        by_case: dict[str, dict[str, Any]] = {}
        for source in source_group:
            case_id = str(source.get("case_id") or "")
            if case_id and (case_id not in by_case or _row_score(source) > _row_score(by_case[case_id])):
                by_case[case_id] = dict(source)
        unscoped_measured_group = list(by_case.values())
        prediction = predictions.get(model_id) or {}
        freeze = prediction.get("_prediction_freeze") if isinstance(prediction.get("_prediction_freeze"), Mapping) else {}
        role = normalize_evaluation_role(
            (entries.get(model_id) or {}).get("evaluation_role")
            or (unscoped_measured_group[0].get("evaluation_role") if unscoped_measured_group else "development")
            or "development"
        )
        candidate_universe_scope = str(freeze.get("candidate_universe_scope") or "")
        audit_case_ids = {
            str(value)
            for value in list(freeze.get("candidate_universe_selected_case_ids") or [])
            if str(value)
        }
        # A predeclared audit is score-independent for development and
        # confirmatory roles alike.  Applying this only to holdouts allowed a
        # deployment shortlist row to change development-audit metrics.
        audit_scoped = bool(
            candidate_universe_scope == "predeclared_audit_universe"
        )
        measured_group = (
            [row for row in unscoped_measured_group if str(row.get("case_id") or "") in audit_case_ids]
            if audit_scoped
            else unscoped_measured_group
        )
        deployment_only_measurement_count = len(unscoped_measured_group) - len(measured_group)
        technically_measured_group = [
            row for row in measured_group
            if _technical_ranking_measurement(row)
        ]
        measured_by_case = {
            str(row.get("case_id") or ""): row
            for row in technically_measured_group
            if str(row.get("case_id") or "")
        }
        measured_ids = set(measured_by_case)
        valid_rows = [
            row
            for row in measured_group
            if _ranking_analysis_eligible(
                row,
                role=role,
                profile=profile,
                audit_scoped=bool(
                    audit_scoped
                    and str(row.get("case_id") or "") in audit_case_ids
                ),
            )
            and _measured_cycle(row) is not None
        ]
        measured_performance_rows = technically_measured_group
        quality_vetoed_candidate_count = sum(
            1 for row in measured_performance_rows
            if _ranking_quality_vetoed(row)
        )
        development_screening_exception_count = sum(
            1 for row in measured_performance_rows
            if _b(row.get("ranking_eligible")) is not True
            and _development_screening_ranking_exception(
                row,
                role=role,
                profile=profile,
                audit_scoped=bool(
                    audit_scoped
                    and str(row.get("case_id") or "") in audit_case_ids
                ),
            )
        )
        actual_sorted = sorted(valid_rows, key=lambda row: float(_measured_cycle(row) or math.inf))
        actual_best = actual_sorted[0] if actual_sorted else None
        best_cycle = _measured_cycle(actual_best or {})
        near_set = {
            str(row.get("case_id") or "")
            for row in actual_sorted
            if best_cycle is not None and (_measured_cycle(row) or math.inf) <= best_cycle * (1.0 + epsilon)
        }
        elite_sets = {
            int(q): {
                str(row.get("case_id") or "")
                for row in actual_sorted[: min(int(q), len(actual_sorted))]
                if str(row.get("case_id") or "")
            }
            for q in elite_q_values
            if int(q) > 0
        }
        context_rows, prediction_origin = _method_rows_for_context(
            prediction,
            direction=direction,
            runner=runner,
            policy=policy,
        )
        if audit_scoped:
            context_rows = [
                method_row
                for method_row in context_rows
                if str(method_row.get("case_id") or "") in audit_case_ids
            ]
        try:
            audit_minimum = int(freeze.get("candidate_universe_minimum_valid_candidates") or 0)
        except Exception:
            audit_minimum = 0
        effective_min_candidates = max(min_candidates, audit_minimum if audit_scoped else 0)

        for method_id in methods:
            method_rows = [row for row in context_rows if str(row.get("method_id") or "") == method_id]
            method_row_ids = {str(row.get("case_id") or "") for row in method_rows if str(row.get("case_id") or "")}
            expected_ids = set(audit_case_ids) if audit_scoped else set(method_row_ids)
            (
                universe_declared_complete,
                universe_declaration_status,
                measurement_coverage_complete,
                measurement_coverage_status,
                expected_count,
            ) = _candidate_universe_status(
                model_id,
                measured_ids,
                expected_ids,
                profile,
                policy,
                freeze,
            )
            available = [
                row
                for row in method_rows
                if _b(row.get("prediction_available")) is not False and _f(row.get("predicted_value")) is not None
            ]
            prediction_by_case = {
                str(row.get("case_id") or ""): row
                for row in available
                if str(row.get("case_id") or "")
            }
            available_ids = set(prediction_by_case)
            prediction_coverage_complete = bool(
                expected_ids
                and expected_ids <= method_row_ids
                and expected_ids <= available_ids
            )
            pairs: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
            for measured in valid_rows:
                case_id = str(measured.get("case_id") or "")
                predicted = prediction_by_case.get(case_id)
                predicted_value = _f((predicted or {}).get("predicted_value"))
                measured_value = _measured_cycle(measured)
                if predicted_value is not None and measured_value is not None:
                    pairs.append((float(predicted_value), float(measured_value), measured, predicted or {}))
            # Development diagnostics answer a different question from strict
            # claim eligibility: how well does a frozen predictor order every
            # technically measured member of the declared audit subset?  Keep
            # that correlation visible even when screening quality or partial
            # accelerator build coverage excludes rows from claims.  Global
            # top-k/regret below still requires complete measurement coverage.
            diagnostic_pairs: list[
                tuple[float, float, dict[str, Any], dict[str, Any]]
            ] = []
            for measured in measured_performance_rows:
                case_id = str(measured.get("case_id") or "")
                predicted = prediction_by_case.get(case_id)
                predicted_value = _f((predicted or {}).get("predicted_value"))
                measured_value = _measured_cycle(measured)
                if predicted_value is not None and measured_value is not None:
                    diagnostic_pairs.append(
                        (
                            float(predicted_value),
                            float(measured_value),
                            measured,
                            predicted or {},
                        )
                    )
            diagnostic_correlation_available = (
                len(diagnostic_pairs) >= effective_min_candidates
            )
            prediction_unit = str(
                (available[0].get("prediction_unit") if available else METHOD_UNITS.get(method_id)) or ""
            )
            absolute_comparable = prediction_unit == "ms"
            errors = [abs(predicted - measured) for predicted, measured, _, _ in pairs] if absolute_comparable else []
            percentage_errors = [
                abs(predicted - measured) / abs(measured) * 100.0
                for predicted, measured, _, _ in pairs
                if absolute_comparable and abs(measured) > 1e-12
            ]
            biases = [predicted - measured for predicted, measured, _, _ in pairs] if absolute_comparable else []
            measured_median = _median(measured for _, measured, _, _ in pairs)
            predicted_sorted = sorted(
                available,
                key=lambda row: (
                    float(_f(row.get("predicted_value")) or math.inf),
                    int(_f(row.get("predicted_rank")) or 10**9),
                ),
            )
            source_labels = sorted(
                {str(row.get("prediction_source") or "") for row in method_rows if str(row.get("prediction_source") or "")}
            )
            predictions_frozen = bool(
                freeze.get("valid")
                and freeze.get("ranking_predictions_valid")
                and prediction_origin == "frozen"
            )
            if candidate_universe_scope == "predeclared_audit_universe":
                ranking_metric_scope = "audit_relative"
            elif candidate_universe_scope == "complete_feasible_universe":
                ranking_metric_scope = "global_feasible_universe"
            else:
                ranking_metric_scope = "declared_universe"
            mean_error = _mean(errors)
            diagnostic_pair_ids = {
                str(measured.get("case_id") or "")
                for _, _, measured, _ in diagnostic_pairs
                if str(measured.get("case_id") or "")
            }
            diagnostic_coverage_fraction = (
                len(diagnostic_pair_ids & expected_ids) / len(expected_ids)
                if expected_ids else None
            )
            diagnostic_leader_exclusion_reasons: list[str] = []
            if is_confirmatory_holdout(role):
                diagnostic_leader_exclusion_reasons.append(
                    "confirmatory_group_not_development_diagnostic"
                )
            if len(diagnostic_pairs) < effective_min_candidates:
                diagnostic_leader_exclusion_reasons.append(
                    "below_effective_minimum_candidate_count"
                )
            if not predictions_frozen:
                diagnostic_leader_exclusion_reasons.append(
                    "predictions_not_frozen"
                )
            if universe_declared_complete is not True:
                diagnostic_leader_exclusion_reasons.append(
                    "candidate_universe_not_declared_complete"
                )
            if (
                diagnostic_coverage_fraction is None
                or diagnostic_coverage_fraction < 0.8
            ):
                diagnostic_leader_exclusion_reasons.append(
                    "diagnostic_coverage_below_80_percent"
                )
            detail: dict[str, Any] = {
                **stratum,
                "model_id": model_id,
                "evaluation_role": role,
                "direction": direction,
                "runner_regime": runner,
                # Stable compatibility aliases for the Native audit exports.
                "precision": stratum["runtime_precision_identity"],
                "output_endpoint_id": stratum[
                    "comparison_output_endpoint_id"
                ],
                "method_id": method_id,
                "method_label": METHOD_LABELS.get(method_id, method_id),
                "prediction_unit": prediction_unit,
                "prediction_origin": prediction_origin,
                "prediction_source": ";".join(source_labels[:8]),
                "prediction_available_count": len(available),
                "prediction_expected_count": len(expected_ids),
                "prediction_coverage_complete": prediction_coverage_complete,
                "prediction_missing_candidate_count": len(expected_ids - available_ids),
                "unscoped_measured_candidate_count": len(unscoped_measured_group),
                "measured_candidate_count": len(measured_group),
                "deployment_only_measurement_count_excluded": deployment_only_measurement_count,
                "audit_scope_enforced": audit_scoped,
                "valid_candidate_count": len(valid_rows),
                "ranking_ineligible_candidate_count": (
                    len(measured_performance_rows) - len(valid_rows)
                ),
                "quality_vetoed_candidate_count": (
                    quality_vetoed_candidate_count
                ),
                "development_screening_exception_candidate_count": (
                    development_screening_exception_count
                ),
                "paired_candidate_count": len(pairs),
                "minimum_valid_candidates_required": effective_min_candidates,
                "expected_candidate_count": expected_count,
                # Compatibility field retains its historical strict meaning:
                # declared/verified *and* fully measured. New fields expose
                # the two independent facts explicitly.
                "candidate_universe_complete": bool(
                    universe_declared_complete and measurement_coverage_complete
                ),
                "candidate_universe_status": (
                    "complete"
                    if universe_declared_complete and measurement_coverage_complete
                    else measurement_coverage_status
                ),
                "candidate_universe_declared_complete": universe_declared_complete,
                "candidate_universe_declaration_status": universe_declaration_status,
                "candidate_measurement_coverage_complete": measurement_coverage_complete,
                "candidate_measurement_coverage_status": measurement_coverage_status,
                "candidate_measurement_coverage_fraction": (
                    len(measured_ids & expected_ids) / len(expected_ids)
                    if expected_ids else None
                ),
                "candidate_universe_scope": candidate_universe_scope,
                "ranking_metric_scope": ranking_metric_scope,
                "predictions_frozen": predictions_frozen,
                "prediction_freeze_status": str(freeze.get("status") or "missing"),
                "prediction_freeze_manifest": str(freeze.get("manifest_path") or ""),
                "prediction_sha256": str(freeze.get("prediction_sha256") or ""),
                "ranking_prediction_csv": str(freeze.get("ranking_prediction_csv_path") or ""),
                "mae_ms": mean_error if absolute_comparable else None,
                "mape_percent": _mean(percentage_errors) if absolute_comparable else None,
                "bias_ms": _mean(biases) if absolute_comparable else None,
                "nmae": (
                    mean_error / measured_median
                    if absolute_comparable and mean_error is not None and measured_median not in (None, 0)
                    else None
                ),
                "spearman_rho": (
                    _spearman([predicted for predicted, _, _, _ in pairs], [measured for _, measured, _, _ in pairs])
                    if len(pairs) >= effective_min_candidates and prediction_coverage_complete
                    else None
                ),
                "kendall_tau_b": (
                    _kendall_tau_b([predicted for predicted, _, _, _ in pairs], [measured for _, measured, _, _ in pairs])
                    if len(pairs) >= effective_min_candidates and prediction_coverage_complete
                    else None
                ),
                "diagnostic_paired_candidate_count": len(diagnostic_pairs),
                "diagnostic_minimum_candidates_required": effective_min_candidates,
                "diagnostic_candidate_coverage_fraction": (
                    diagnostic_coverage_fraction
                ),
                "development_diagnostic_leader_group_eligible": not bool(
                    diagnostic_leader_exclusion_reasons
                ),
                "development_diagnostic_leader_group_exclusion_reasons": (
                    diagnostic_leader_exclusion_reasons
                ),
                "diagnostic_spearman_rho": (
                    _spearman(
                        [predicted for predicted, _, _, _ in diagnostic_pairs],
                        [measured for _, measured, _, _ in diagnostic_pairs],
                    )
                    if diagnostic_correlation_available else None
                ),
                "diagnostic_kendall_tau_b": (
                    _kendall_tau_b(
                        [predicted for predicted, _, _, _ in diagnostic_pairs],
                        [measured for _, measured, _, _ in diagnostic_pairs],
                    )
                    if diagnostic_correlation_available else None
                ),
                "diagnostic_correlation_scope": (
                    "technically_measured_declared_subset"
                    if diagnostic_correlation_available else "unavailable"
                ),
                "diagnostic_correlation_claim_eligible": False,
                "best_measured_case": str((actual_best or {}).get("case_id") or ""),
                "best_measured_cycle_ms": best_cycle,
                "near_optimal_relative_epsilon": epsilon,
                "near_optimal_case_count": len(near_set),
                "validity_at_k_is_diagnostic_only": True,
            }
            paired_valid_ids = {
                str(measured.get("case_id") or "")
                for _, _, measured, _ in pairs
                if str(measured.get("case_id") or "")
            }
            for k in k_values:
                k_int = int(k)
                top_k = predicted_sorted[: min(k_int, len(predicted_sorted))]
                top_k_ids = {str(row.get("case_id") or "") for row in top_k}
                eligible_top_k = [
                    measured_by_case[case_id]
                    for case_id in top_k_ids
                    if case_id in measured_by_case
                    and _ranking_analysis_eligible(
                        measured_by_case[case_id],
                        role=role,
                        profile=profile,
                        audit_scoped=bool(
                            audit_scoped and case_id in audit_case_ids
                        ),
                    )
                ]
                measured_values = [_measured_cycle(row) for row in eligible_top_k if _measured_cycle(row) is not None]
                best_top_k = min(measured_values) if measured_values else None
                rank_quality_available = bool(
                    universe_declared_complete
                    and measurement_coverage_complete
                    and prediction_coverage_complete
                    and len(pairs) >= effective_min_candidates
                    and k_int <= len(pairs)
                    and len(top_k) == k_int
                    and len(top_k_ids) == k_int
                    and top_k_ids <= paired_valid_ids
                )
                detail[f"hit_at_{k}"] = (
                    str((actual_best or {}).get("case_id") or "") in top_k_ids
                    if rank_quality_available and actual_best
                    else None
                )
                detail[f"near_optimal_hit_at_{k}"] = (
                    bool(top_k_ids & near_set) if rank_quality_available and near_set else None
                )
                detail[f"validity_at_{k}"] = len(eligible_top_k) / len(top_k) if top_k else None
                detail[f"diagnostic_validity_at_{k}"] = detail[f"validity_at_{k}"]
                detail[f"regret_at_{k}"] = (
                    (best_top_k - best_cycle) / best_cycle
                    if rank_quality_available and best_top_k is not None and best_cycle not in (None, 0)
                    else None
                )
                for q, elite_ids in elite_sets.items():
                    detail[f"elite_recall_at_{k}_q{q}"] = (
                        len(top_k_ids & elite_ids) / len(elite_ids)
                        if rank_quality_available and int(q) <= len(pairs) and elite_ids
                        else None
                    )
            if not available:
                status = (
                    "handover_model_unavailable"
                    if method_id == "cycle_time_with_handover" and runner == "native_fifo"
                    else "method_unavailable"
                )
            elif not prediction_coverage_complete:
                status = "frozen_prediction_coverage_incomplete"
            elif len(pairs) < effective_min_candidates:
                status = (
                    "insufficient_valid_audit_candidates"
                    if audit_scoped
                    else "insufficient_candidates_for_correlation"
                )
            elif not (universe_declared_complete and measurement_coverage_complete):
                status = "candidate_universe_not_auditable"
            elif bool(policy.get("require_frozen_predictions", True)) and not predictions_frozen:
                status = "predictions_not_frozen"
            elif not is_confirmatory_holdout(role):
                status = "development_evidence_only"
            else:
                status = "holdout_validated"
            detail["status"] = status
            for cohort_id in (
                "technical",
                "quality_pass",
                "quality_pass_or_inconclusive",
            ):
                cohort_measurements = [
                    row for row in measured_group
                    if _ranking_sensitivity_cohort_eligible(row, cohort_id)
                ]
                cohort_pairs: list[tuple[float, float, str]] = []
                for measured in cohort_measurements:
                    case_id = str(measured.get("case_id") or "")
                    predicted = prediction_by_case.get(case_id)
                    predicted_value = _f(
                        (predicted or {}).get("predicted_value")
                    )
                    measured_value = _measured_cycle(measured)
                    if predicted_value is not None and measured_value is not None:
                        cohort_pairs.append((
                            float(predicted_value),
                            float(measured_value),
                            case_id,
                        ))
                cohort_pair_ids = {
                    case_id for _, _, case_id in cohort_pairs if case_id
                }
                cohort_coverage_fraction = (
                    len(cohort_pair_ids & expected_ids) / len(expected_ids)
                    if expected_ids else None
                )
                cohort_complete = bool(
                    universe_declared_complete
                    and expected_ids
                    and expected_ids <= cohort_pair_ids
                    and prediction_coverage_complete
                )
                correlation_available = (
                    len(cohort_pairs) >= effective_min_candidates
                )
                global_hit_at_1 = None
                global_regret_at_1 = None
                if cohort_id == "technical" and cohort_complete:
                    measured_order = sorted(
                        cohort_pairs, key=lambda item: item[1]
                    )
                    predicted_order = sorted(
                        cohort_pairs, key=lambda item: item[0]
                    )
                    if measured_order and predicted_order:
                        best_case = measured_order[0][2]
                        selected_case = predicted_order[0][2]
                        best_value = measured_order[0][1]
                        selected_value = next(
                            measured_value
                            for _, measured_value, case_id in cohort_pairs
                            if case_id == selected_case
                        )
                        global_hit_at_1 = selected_case == best_case
                        global_regret_at_1 = (
                            (selected_value - best_value) / best_value
                            if best_value > 0 else None
                        )
                cohort_sensitivity_rows.append({
                    **stratum,
                    "evaluation_role": role,
                    "method_id": method_id,
                    "method_label": METHOD_LABELS.get(
                        method_id, method_id
                    ),
                    "cohort": cohort_id,
                    "cohort_semantics": {
                        "technical": "technically successful numeric observations",
                        "quality_pass": "technical observations with task-quality pass",
                        "quality_pass_or_inconclusive": (
                            "technical observations with task-quality pass or inconclusive"
                        ),
                    }[cohort_id],
                    "declared_candidate_count": len(expected_ids),
                    "cohort_candidate_count": len(cohort_measurements),
                    "paired_candidate_count": len(cohort_pairs),
                    "minimum_candidates_required": effective_min_candidates,
                    "candidate_coverage_fraction": cohort_coverage_fraction,
                    "declared_universe_complete_for_cohort": cohort_complete,
                    "spearman_rho": (
                        _spearman(
                            [item[0] for item in cohort_pairs],
                            [item[1] for item in cohort_pairs],
                        ) if correlation_available else None
                    ),
                    "kendall_tau_b": (
                        _kendall_tau_b(
                            [item[0] for item in cohort_pairs],
                            [item[1] for item in cohort_pairs],
                        ) if correlation_available else None
                    ),
                    "correlation_scope": (
                        "diagnostic_measured_cohort_subset"
                        if correlation_available else "unavailable"
                    ),
                    "correlation_claim_eligible": False,
                    "global_hit_at_1": global_hit_at_1,
                    "global_regret_at_1": global_regret_at_1,
                    "global_top1_status": (
                        "available_complete_declared_universe"
                        if cohort_id == "technical" and cohort_complete
                        else (
                            "filtered_sensitivity_cohort_not_global_universe"
                            if cohort_id != "technical"
                            else "unavailable_incomplete_declared_universe"
                        )
                    ),
                    "status": (
                        "diagnostic_available"
                        if correlation_available
                        else "insufficient_candidates"
                    ),
                })
            details.append(detail)

    macro_rows: list[dict[str, Any]] = []
    primary_k = int(policy.get("primary_k") or 5)
    for method_id in methods:
        method_details = [row for row in details if str(row.get("method_id") or "") == method_id]
        holdout = [row for row in method_details if is_confirmatory_holdout(row.get("evaluation_role"))]
        validated = [row for row in holdout if str(row.get("status") or "") == "holdout_validated"]
        source = validated or holdout or method_details
        diagnostic_leader_details = [
            row for row in method_details
            if row.get("development_diagnostic_leader_group_eligible") is True
            and _f(row.get("diagnostic_spearman_rho")) is not None
        ]
        development_details = [
            row for row in method_details
            if not is_confirmatory_holdout(row.get("evaluation_role"))
        ]
        diagnostic_strata = sorted({
            json.dumps(
                [str(row.get(field) or "") for field in RANKING_COMPARISON_STRATA],
                separators=(",", ":"),
            )
            for row in diagnostic_leader_details
        })
        metric_scopes = sorted({str(row.get("ranking_metric_scope") or "") for row in source if str(row.get("ranking_metric_scope") or "")})
        macro: dict[str, Any] = {
            "method_id": method_id,
            "method_label": METHOD_LABELS.get(method_id, method_id),
            "prediction_unit": METHOD_UNITS.get(method_id, ""),
            "group_count": len(method_details),
            "development_group_count": len(development_details),
            "holdout_group_count": len(holdout),
            "validated_holdout_group_count": len(validated),
            "ranking_metric_scope": metric_scopes[0] if len(metric_scopes) == 1 else ("mixed" if metric_scopes else ""),
            "macro_mae_ms": _mean(_f(row.get("mae_ms")) for row in source),
            "macro_mape_percent": _mean(_f(row.get("mape_percent")) for row in source),
            "macro_spearman_rho": _mean(_f(row.get("spearman_rho")) for row in source),
            "macro_kendall_tau_b": _mean(_f(row.get("kendall_tau_b")) for row in source),
            "strict_correlation_group_count": sum(
                _f(row.get("spearman_rho")) is not None for row in source
            ),
            "diagnostic_correlation_group_count": sum(
                _f(row.get("diagnostic_spearman_rho")) is not None
                for row in source
            ),
            "macro_diagnostic_spearman_rho": _mean(
                _f(row.get("diagnostic_spearman_rho")) for row in source
            ),
            "macro_diagnostic_kendall_tau_b": _mean(
                _f(row.get("diagnostic_kendall_tau_b")) for row in source
            ),
            "development_diagnostic_eligible_group_count": len(
                diagnostic_leader_details
            ),
            "development_diagnostic_macro_spearman_rho": _mean(
                _f(row.get("diagnostic_spearman_rho"))
                for row in diagnostic_leader_details
            ),
            "development_diagnostic_macro_kendall_tau_b": _mean(
                _f(row.get("diagnostic_kendall_tau_b"))
                for row in diagnostic_leader_details
            ),
            "development_diagnostic_actual_strata": diagnostic_strata,
            "development_diagnostic_min_coverage_fraction": (
                min(
                    float(row.get("diagnostic_candidate_coverage_fraction"))
                    for row in diagnostic_leader_details
                    if row.get("diagnostic_candidate_coverage_fraction") is not None
                )
                if diagnostic_leader_details else None
            ),
            "development_diagnostic_claim_eligible": False,
            "status": (
                "holdout_available"
                if validated
                else ("holdout_incomplete" if holdout else ("development_only" if method_details else "unavailable"))
            ),
        }
        for k in k_values:
            macro[f"macro_hit_at_{k}"] = _mean(
                float(bool(row.get(f"hit_at_{k}")))
                for row in source
                if row.get(f"hit_at_{k}") is not None
            )
            macro[f"macro_near_optimal_hit_at_{k}"] = _mean(
                float(bool(row.get(f"near_optimal_hit_at_{k}")))
                for row in source
                if row.get(f"near_optimal_hit_at_{k}") is not None
            )
            macro[f"macro_validity_at_{k}"] = _mean(_f(row.get(f"validity_at_{k}")) for row in source)
            macro[f"macro_regret_at_{k}"] = _mean(_f(row.get(f"regret_at_{k}")) for row in source)
            for q in elite_q_values:
                macro[f"macro_elite_recall_at_{k}_q{q}"] = _mean(
                    _f(row.get(f"elite_recall_at_{k}_q{q}")) for row in source
                )
        macro_rows.append(macro)

    comparable = [
        row
        for row in macro_rows
        if int(row.get("holdout_group_count") or 0) > 0
        and int(row.get("validated_holdout_group_count") or 0)
        == int(row.get("holdout_group_count") or 0)
        and (
            row.get(f"macro_hit_at_{primary_k}") is not None
            or (
                row.get("macro_spearman_rho") is not None
                and int(row.get("strict_correlation_group_count") or 0)
                == int(row.get("holdout_group_count") or 0)
            )
        )
    ]
    best = (
        sorted(comparable, key=lambda row: method_macro_sort_key(row, primary_k))[0]
        if comparable else None
    )
    best_basis = "strict_complete_groups" if best else ""

    # A development diagnostic may guide the next model iteration, but it is
    # never the report's ``best_method``.  Publish a separately named leader
    # only when every compared method uses the same actually observed strata,
    # every included actual stratum satisfies its effective minimum,
    # predictions are frozen and at least 80 percent of each declared
    # candidate universe is paired.  Development groups without qualified
    # actual evidence remain visible in the macro ledger, but do not have to
    # become qualified merely for another stratum to support the diagnostic.
    diagnostic_pool = [
        row for row in macro_rows
        if int(row.get("development_group_count") or 0) > 0
    ]
    diagnostic_signatures = {
        tuple(row.get("development_diagnostic_actual_strata") or [])
        for row in diagnostic_pool
        if int(row.get("development_diagnostic_eligible_group_count") or 0)
        > 0
        and _f(row.get("development_diagnostic_macro_spearman_rho"))
        is not None
    }
    diagnostic_comparable = [
        row for row in diagnostic_pool
        if int(row.get("development_diagnostic_eligible_group_count") or 0)
        > 0
        and _f(row.get("development_diagnostic_macro_spearman_rho"))
        is not None
    ]
    identical_actual_strata = bool(
        len(diagnostic_comparable) >= 2
        and len(diagnostic_comparable) == len(diagnostic_pool)
        and len(diagnostic_signatures) == 1
        and next(iter(diagnostic_signatures), ())
    )
    diagnostic_leader = (
        max(
            diagnostic_comparable,
            key=lambda row: (
                float(
                    _f(row.get("development_diagnostic_macro_spearman_rho"))
                    or -math.inf
                ),
                float(
                    _f(row.get("development_diagnostic_macro_kendall_tau_b"))
                    or -math.inf
                ),
                str(row.get("method_id") or ""),
            ),
        )
        if identical_actual_strata else None
    )
    non_unavailable = [row for row in macro_rows if str(row.get("status") or "") != "unavailable"]
    all_holdout_valid = bool(non_unavailable) and all(
        str(row.get("status") or "") == "holdout_available" for row in non_unavailable
    )
    overall = {
        "status": (
            "holdout_available"
            if all_holdout_valid
            else (
                "holdout_incomplete"
                if any(row.get("holdout_group_count") for row in macro_rows)
                else ("development_only" if details else "unavailable")
            )
        ),
        "method_count": len(macro_rows),
        "group_method_row_count": len(details),
        "best_method_id": str((best or {}).get("method_id") or ""),
        "best_method_label": str((best or {}).get("method_label") or ""),
        "best_method_basis": best_basis,
        "development_diagnostic_leader_method_id": str(
            (diagnostic_leader or {}).get("method_id") or ""
        ),
        "development_diagnostic_leader_method_label": str(
            (diagnostic_leader or {}).get("method_label") or ""
        ),
        "development_diagnostic_leader_basis": (
            "frozen_minimum_80pct_identical_actual_strata"
            if diagnostic_leader else ""
        ),
        "development_diagnostic_leader_spearman_rho": (
            _f(
                (diagnostic_leader or {}).get(
                    "development_diagnostic_macro_spearman_rho"
                )
            )
        ),
        "development_diagnostic_leader_kendall_tau_b": (
            _f(
                (diagnostic_leader or {}).get(
                    "development_diagnostic_macro_kendall_tau_b"
                )
            )
        ),
        "development_diagnostic_leader_min_coverage_fraction": (
            _f(
                (diagnostic_leader or {}).get(
                    "development_diagnostic_min_coverage_fraction"
                )
            )
        ),
        "development_diagnostic_leader_actual_strata": list(
            (diagnostic_leader or {}).get(
                "development_diagnostic_actual_strata"
            ) or []
        ),
        "development_diagnostic_leader_claim_eligible": False,
        "development_diagnostic_identical_actual_strata": (
            identical_actual_strata
        ),
        "primary_k": primary_k,
        "metric_scope_note": "Top-k, elite recall and regret are audit-relative whenever candidate_universe_scope=predeclared_audit_universe; they are global only for a complete feasible universe.",
        "cohort_sensitivity_rows": cohort_sensitivity_rows,
        "cohort_sensitivity_semantics": (
            "Identical method-by-stratum recomputation on technical, "
            "quality-pass, and pass-or-inconclusive cohorts. Subset "
            "correlations are diagnostic; global Hit@1/Regret@1 require the "
            "complete declared technical universe."
        ),
    }
    overall["experiment_purpose"] = purpose
    overall["ranking_scope"] = _ranking_observed_scope(rows, profile, policy)
    return details, macro_rows, overall


def _scientific_row(row: Mapping[str, Any]) -> dict[str, Any]:
    energy_comparison = resolve_energy_comparison(row)
    metrics = row.get("accuracy_gate_metrics") if isinstance(row.get("accuracy_gate_metrics"), Mapping) else {}
    cardinality = row.get("validation_cardinality_contract") if isinstance(row.get("validation_cardinality_contract"), Mapping) else {}
    model_id = str(
        row.get("model_id")
        or row.get("model")
        or row.get("model_name")
        or ""
    ).strip()
    if model_id.lower() == "model":
        model_id = str(row.get("model") or row.get("model_name") or "").strip()
    model_identity_resolved = bool(model_id and model_id.lower() != "model")
    task = _normalized_scientific_task(
        row.get("task")
        or _first(
            row,
            ("benchmark_task_used", "benchmark_task_requested", "validation_dataset_task"),
        ),
        model_id,
    )
    run_id = str(
        row.get("run_id")
        or _first(row, ("source_tag", "benchmark_run_id", "run_profile_id"))
        or row.get("backend")
        or "unknown_run"
    ).strip()
    scientific_row = {
        "row_role": PERFORMANCE_ROW_ROLE,
        "normalization_error": row.get("normalization_error"),
        "model_id": model_id or "unknown_model",
        "model_identity_resolved": model_identity_resolved,
        "setup_id": row.get("setup_id") or row.get("energy_setup_id"),
        "source_run_id": row.get("source_run_id") or row.get("run_id"),
        "source_release": row.get("release") or row.get("tool_release") or row.get("tool_version"),
        "diagnostic_only": _b(row.get("diagnostic_only")) is True,
        "declared_claim_eligible": _b(row.get("claim_eligible")),
        "task": task,
        "evaluation_role": row.get("evaluation_role"),
        "case_id": row.get("case_id"),
        "backend": row.get("backend"),
        "setup_id": _resolved_report_setup_id_v27550(row),
        "setup_identity_status": row.get("setup_identity_status"),
        "setup_identity_candidates": row.get("setup_identity_candidates") or [],
        "comparison_backend": row.get("comparison_backend"),
        "run_id": run_id,
        "variant": row.get("variant"),
        "runtime_precision_identity": _first(
            row, ("runtime_precision_identity", "execution_precision", "precision"),
        ),
        "runtime_numeric_input_identity": (
            dict(row.get("runtime_numeric_input_identity") or {})
            if isinstance(row.get("runtime_numeric_input_identity"), Mapping)
            else {}
        ),
        "runtime_numeric_input_sha256": str(
            row.get("runtime_numeric_input_sha256") or ""
        ).strip().lower().removeprefix("sha256:"),
        "runtime_input_encoding_identity": (
            dict(row.get("runtime_input_encoding_identity") or {})
            if isinstance(row.get("runtime_input_encoding_identity"), Mapping)
            else {}
        ),
        "runtime_input_encoding_sha256": str(
            row.get("runtime_input_encoding_sha256") or ""
        ).strip().lower().removeprefix("sha256:"),
        "runtime_numeric_input_identity_status": str(
            row.get("runtime_numeric_input_identity_status") or "unavailable"
        ).strip().lower(),
        "repetition_index": _first(
            row,
            ("repetition_index", "process_local_repetition_index", "repeat_idx", "repeat_index"),
        ),
        "repetition_id": row.get("repetition_id"),
        "repetition_count_requested": row.get("repetition_count_requested"),
        "repetition_count_attempted": row.get("repetition_count_attempted"),
        "repetition_count_valid": row.get("repetition_count_valid"),
        "repetition_status": row.get("repetition_status"),
        "repetition_aggregation": row.get("repetition_aggregation"),
        "repetition_runtime_scope": row.get("repetition_runtime_scope"),
        "repetition_independence_verified": row.get(
            "repetition_independence_verified"
        ),
        "runner_regime": _runner_regime(row),
        "direction": _direction(row),
        "buildable": row.get("buildable"),
        "runtime_executable": row.get("runtime_executable"),
        # Preserve the measured endpoint and its existing completion evidence
        # through Scientific JSON/CSV; never infer completion from a rate.
        "measurement_endpoint": row.get("measurement_endpoint"),
        "postprocess_included": row.get("postprocess_included"),
        "postprocess_completion_verified": row.get("postprocess_completion_verified"),
        "postprocess_completed_frames": row.get("postprocess_completed_frames"),
        "decoder_id": row.get("decoder_id"),
        "host_postprocessing_evidence_status": row.get("host_postprocessing_evidence_status"),
        "host_postprocessing_evidence_source": row.get("host_postprocessing_evidence_source"),
        "raw_stage_mean_ms": row.get("raw_stage_mean_ms"),
        "host_tail_mean_ms": row.get("host_tail_mean_ms"),
        "completed_task_mean_ms": row.get("completed_task_mean_ms"),
        "measurement_payload_present": row.get(
            "measurement_payload_present"
        ),
        "measurement_valid": row.get("measurement_valid"),
        "terminal_failure": row.get("terminal_failure"),
        "terminal_reason": row.get("terminal_reason"),
        "terminal_reason_scope": row.get("terminal_reason_scope"),
        "runner_returncode": row.get("runner_returncode"),
        "runner_signal_number": row.get("runner_signal_number"),
        "runtime_failure_evidence_source": row.get(
            "runtime_failure_evidence_source"
        ),
        "runtime_failure_evidence_line_number": row.get(
            "runtime_failure_evidence_line_number"
        ),
        **assessment_fields(row.get("accuracy_assessment")),
        "contract_consistent": row.get("contract_consistent"),
        "task_quality_tier": (
            row.get("accuracy_gate_tier") or row.get("task_quality_tier")
        ),
        # v60i separates the task-quality decision from the overall row
        # eligibility status.  Contract/tier failures must not hide a real
        # bootstrap pass/fail/inconclusive result.
        "task_quality_status": (
            row.get("accuracy_gate_decision")
            or row.get("task_quality_decision")
            or row.get("task_quality_status")
            or row.get("quality_gate_status")
        ),
        "task_quality_decision": (
            row.get("accuracy_gate_decision")
            or row.get("task_quality_decision")
        ),
        "task_quality_gate_reason": (
            row.get("accuracy_gate_trigger_reason")
            or row.get("task_quality_gate_reason")
        ),
        "task_quality_gate_reasons": (
            row.get("accuracy_gate_trigger_reasons")
            or row.get("task_quality_gate_reasons")
            or []
        ),
        "contract_gate_reason": row.get("contract_gate_reason"),
        "eligibility_gate_reason": (
            row.get("accuracy_gate_reason")
            or row.get("eligibility_gate_reason")
        ),
        "eligibility_status": row.get("gate_status") or row.get("eligibility_status"),
        "task_quality_metric": (
            row.get("accuracy_gate_metric") or row.get("task_quality_metric")
        ),
        "task_quality_candidate": metrics.get(
            "candidate", row.get("task_quality_candidate")
        ),
        "task_quality_reference": metrics.get(
            "reference", row.get("task_quality_reference")
        ),
        "task_quality_delta": (
            row.get("accuracy_gate_delta")
            if row.get("accuracy_gate_delta") not in (None, "")
            else row.get("task_quality_delta")
        ),
        "task_quality_ci_low": row.get("accuracy_gate_ci_low", row.get("task_quality_ci_low")),
        "task_quality_ci_high": row.get("accuracy_gate_ci_high", row.get("task_quality_ci_high")),
        "task_quality_margin": (
            row.get("accuracy_gate_threshold")
            if row.get("accuracy_gate_threshold") not in (None, "")
            else row.get("task_quality_margin")
        ),
        "task_quality_bootstrap_repetitions_requested": metrics.get(
            "bootstrap_repetitions_requested",
            row.get("accuracy_gate_bootstrap_repetitions_requested")
            or row.get("task_quality_bootstrap_repetitions_requested"),
        ),
        "task_quality_bootstrap_repetitions": metrics.get(
            "bootstrap_repetitions",
            row.get("accuracy_gate_bootstrap_repetitions")
            or row.get("task_quality_bootstrap_repetitions"),
        ),
        "task_quality_bootstrap_engine": metrics.get(
            "bootstrap_engine",
            row.get("accuracy_gate_bootstrap_engine")
            or row.get("task_quality_bootstrap_engine"),
        ),
        "task_quality_bootstrap_skipped_reason": metrics.get(
            "bootstrap_skipped_reason",
            row.get("accuracy_gate_bootstrap_skipped_reason")
            or row.get("task_quality_bootstrap_skipped_reason"),
        ),
        "task_quality_bootstrap_elapsed_s": metrics.get(
            "bootstrap_elapsed_s",
            row.get("accuracy_gate_bootstrap_elapsed_s")
            or row.get("task_quality_bootstrap_elapsed_s"),
        ),
        "task_quality_bootstrap_candidate_event_count": metrics.get("bootstrap_candidate_event_count"),
        "task_quality_bootstrap_reference_event_count": metrics.get("bootstrap_reference_event_count"),
        "quality_policy_sha256": (
            row.get("accuracy_gate_policy_sha256")
            or row.get("quality_policy_sha256")
        ),
        "runtime_quality_policy_sha256": row.get("runtime_quality_gate_policy_sha256") or row.get("task_quality_policy_sha256"),
        "quality_policy_match": row.get("accuracy_gate_policy_match"),
        "benchmark_task_requested": row.get("benchmark_task_requested"),
        "benchmark_task_used": row.get("benchmark_task_used"),
        "validation_dataset_task": row.get("validation_dataset_task"),
        "validation_requested_count": cardinality.get("requested_count"),
        "validation_evaluated_count": cardinality.get("evaluated_count"),
        "validation_cardinality_status": cardinality.get("status"),
        "validation_cardinality_pass": cardinality.get("pass"),
        "performance_eligible": row.get("performance_eligible"),
        "performance_claim_eligible": row.get(
            "performance_claim_eligible"
        ),
        "performance_claim_exclusion_reasons": row.get(
            "performance_claim_exclusion_reasons"
        ) or [],
        "precision_quality_binding_verified": row.get(
            "precision_quality_binding_verified"
        ),
        "accuracy_gate_pass": row.get("accuracy_gate_pass"),
        "ranking_eligible": row.get("ranking_eligible"),
        "energy_eligible": row.get("energy_eligible"),
        "energy_claim_eligible": row.get("energy_claim_eligible"),
        "pareto_eligible": row.get("pareto_eligible"),
        "exclusion_reason": row.get("exclusion_reason"),
        "latency_ms": _first(row, (
            "split_latency_e2e_ms", "total_latency_ms", "full_e2e_latency_ms",
            "latency_ms",
        )),
        "cycle_ms": _first(row, (
            "pipeline_cycle_selected_ms", "total_latency_ms", "cycle_ms",
        )),
        "throughput_fps": _first(row, (
            "throughput_primary_fps", "pipeline_fps_selected",
            "heterogeneous_pipeline_fps", "throughput_fps",
        )),
        "average_power_w": energy_comparison.get("comparison_average_power_w"),
        "raw_average_power_w": energy_comparison.get("raw_average_power_w"),
        "host_normalized_average_power_est_w": energy_comparison.get("host_normalized_average_power_est_w"),
        "average_power_ci_low_w": _first(row, ("energy_streaming_avg_power_w_ci_low", "avg_power_w_ci_low")),
        "average_power_ci_high_w": _first(row, ("energy_streaming_avg_power_w_ci_high", "avg_power_w_ci_high")),
        "energy_per_work_j": energy_comparison.get("comparison_energy_per_work_j"),
        "raw_energy_per_work_j": energy_comparison.get("raw_energy_per_work_j"),
        "host_normalized_energy_per_work_est_j": energy_comparison.get("host_normalized_energy_per_work_est_j"),
        "energy_comparison_basis": energy_comparison.get("energy_comparison_basis"),
        "energy_comparison_status": energy_comparison.get("energy_comparison_status"),
        "energy_comparison_claim_ready": energy_comparison.get("energy_comparison_claim_ready"),
        "energy_per_work_sample_stddev_j": energy_comparison.get("comparison_energy_per_work_sample_stddev_j"),
        "energy_per_work_ci_low_j": energy_comparison.get("comparison_energy_per_work_ci_low_j"),
        "energy_per_work_ci_high_j": energy_comparison.get("comparison_energy_per_work_ci_high_j"),
        "energy_repeat_n": _first(row, ("energy_streaming_repeat_n", "energy_latency_repeat_n", "energy_valid_window_count")),
        "energy_confidence_level": row.get("energy_confidence_level"),
        "energy_scope": _first(row, ("energy_physical_scope", "measurement_scope", "energy_scope")),
        "energy_window": _first(row, ("energy_window_label", "energy_measurement_window", "energy_window", "window_type")),
        "prediction_rank": row.get("prediction_rank"),
        "predicted_cycle_ms": row.get("predicted_cycle_ms"),
        "predicted_fps": row.get("predicted_fps"),
        "prediction_score": row.get("prediction_score"),
    }
    scientific_row["row_status"] = str(
        row.get("row_status")
        or (
            "available"
            if _performance_observation(scientific_row)
            else "unavailable"
        )
    )
    if (
        str(row.get("host_normalization_role") or "none").strip().lower()
        == "tensorrt_full"
        and energy_comparison.get("energy_comparison_claim_ready") is not True
    ):
        scientific_row["energy_eligible"] = False
        scientific_row["energy_claim_eligible"] = False
        scientific_row["energy_per_work_j"] = None
        scientific_row["average_power_w"] = None
    if (
        not model_identity_resolved
        or scientific_row.get("diagnostic_only") is True
        or scientific_row.get("declared_claim_eligible") is False
        or row.get("identity_conflicts")
        or row.get("completed_v2_projection_conflicts")
    ):
        scientific_row["ranking_eligible"] = False
        scientific_row["performance_eligible"] = False
        scientific_row["performance_claim_eligible"] = False
        scientific_row["energy_eligible"] = False
        scientific_row["energy_claim_eligible"] = False
        scientific_row["pareto_eligible"] = False
    for field in UNCERTAINTY_FIELDS:
        if field in metrics or f"accuracy_gate_{field}" in row or f"task_quality_{field}" in row:
            scientific_row[f"task_quality_{field}"] = metrics.get(field, row.get(f"accuracy_gate_{field}", row.get(f"task_quality_{field}")))
    return project_flat_quality_uncertainty(scientific_row)


def _performance_observation(row: Mapping[str, Any]) -> bool:
    return bool(
        _b(row.get("runtime_executable")) is True
        and _b(row.get("measurement_valid")) is not False
        and _b(row.get("terminal_failure")) is not True
        and (_f(row.get("throughput_fps")) is not None or _f(row.get("latency_ms")) is not None)
    )


def _performance_cohort_projection(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expose technical, quality and claim cohorts without changing gates.

    The projection is descriptive: it reuses the existing observation,
    task-quality and claim predicates and records disagreements rather than
    silently promoting or dropping a row.
    """

    projected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        technical = _performance_observation(row)
        quality_decision = _canonical_quality_decision(
            row.get("task_quality_decision")
            or row.get("task_quality_status")
        )
        quality = bool(technical and (quality_decision == "pass" or (row.get("accuracy_assessment") and quality_decision in {"reference_close", "accuracy_loss", "not_estimable"})))
        claim = _performance_claim_eligible(row)
        technical_reasons: list[str] = []
        if _b(row.get("runtime_executable")) is not True:
            technical_reasons.append("runtime_not_executable")
        if _b(row.get("measurement_valid")) is False:
            technical_reasons.append("measurement_invalid")
        if _b(row.get("terminal_failure")) is True:
            technical_reasons.append("terminal_process_failure")
        if (
            _f(row.get("throughput_fps")) is None
            and _f(row.get("latency_ms")) is None
        ):
            technical_reasons.append("performance_metric_unavailable")
        quality_reasons = [] if quality else (
            ["not_in_technical_cohort"]
            if not technical else [f"task_quality_{quality_decision}"]
        )
        claim_reasons = (
            [] if claim else _claim_exclusion_reasons(
                row, claim_kind="performance",
            )
        )
        consistency_errors: list[str] = []
        if quality and not technical:
            consistency_errors.append("quality_without_technical")
        if claim and not quality:
            consistency_errors.append("claim_without_quality")
        row.update({
            "technical_cohort_eligible": technical,
            "quality_cohort_eligible": quality,
            "claim_cohort_eligible": claim,
            "task_quality_decision_for_cohort": quality_decision,
            "technical_cohort_exclusion_reasons": technical_reasons,
            "quality_cohort_exclusion_reasons": quality_reasons,
            "claim_cohort_exclusion_reasons": claim_reasons,
            "cohort_consistency_errors": consistency_errors,
            "cohort_membership": [
                name
                for name, eligible in (
                    ("technical", technical),
                    ("quality", quality),
                    ("claim", claim),
                )
                if eligible
            ],
        })
        projected.append(row)

    summary = {
        "schema": "onnx-splitpoint/performance-cohort-summary",
        "schema_version": 1,
        "performance_row_count": len(projected),
        "technical_cohort_count": sum(
            bool(row["technical_cohort_eligible"]) for row in projected
        ),
        "quality_cohort_count": sum(
            bool(row["quality_cohort_eligible"]) for row in projected
        ),
        "claim_cohort_count": sum(
            bool(row["claim_cohort_eligible"]) for row in projected
        ),
        "cohort_consistency_error_count": sum(
            bool(row["cohort_consistency_errors"]) for row in projected
        ),
        "semantics": {
            "technical": (
                "runtime_executable with a latency or throughput observation"
            ),
            "quality": "technical cohort and task-quality decision pass",
            "claim": "existing performance-claim eligibility predicate",
        },
    }
    return projected, summary


def _energy_observation(row: Mapping[str, Any]) -> bool:
    return bool(
        _f(row.get("energy_per_work_j")) is not None
        or _f(row.get("average_power_w")) is not None
    )


def _row_role(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("row_role") or "").strip()
    if explicit:
        return explicit
    if str(row.get("source_kind") or "").strip() == NATIVE_ENERGY_ROW_ROLE:
        return NATIVE_ENERGY_ROW_ROLE
    return PERFORMANCE_ROW_ROLE


def _is_performance_row(row: Mapping[str, Any]) -> bool:
    return _row_role(row) == PERFORMANCE_ROW_ROLE


def _is_native_energy_attempt(row: Mapping[str, Any]) -> bool:
    return _row_role(row) == NATIVE_ENERGY_ROW_ROLE


def _row_status(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("row_status") or "").strip()
    if explicit:
        return explicit
    if _is_native_energy_attempt(row):
        return "available" if _b(row.get("measurement_ok")) is True else "measurement_failed"
    return "available" if _performance_observation(row) else "unavailable"


def _refresh_scientific_summary(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute row counters without mixing performance and energy roles."""
    result = dict(summary or {})
    scientific_rows = [dict(row) for row in rows if isinstance(row, Mapping)]
    performance_rows = [
        row for row in scientific_rows if _is_performance_row(row)
    ]
    native_energy_attempts = [
        row for row in scientific_rows if _is_native_energy_attempt(row)
    ]
    native_energy_successes = [
        row
        for row in native_energy_attempts
        if _b(row.get("measurement_ok")) is True
        or (
            row.get("measurement_ok") in (None, "")
            and _row_status(row) == "available"
        )
    ]
    native_energy_failures = [
        row
        for row in native_energy_attempts
        if _b(row.get("measurement_ok")) is False
        or _row_status(row) == "measurement_failed"
    ]

    quality_status_counts = dict(
        Counter(
            str(
                row.get("task_quality_decision")
                or row.get("task_quality_status")
                or "unavailable"
            )
            for row in performance_rows
        )
    )
    eligibility_status_counts = dict(
        Counter(
            str(row.get("eligibility_status") or "unavailable")
            for row in performance_rows
        )
    )
    result.update(
        {
            "row_count": len(scientific_rows),
            "row_role_counts": dict(
                Counter(_row_role(row) for row in scientific_rows)
            ),
            "row_status_counts": dict(
                Counter(_row_status(row) for row in scientific_rows)
            ),
            "performance_payload_row_count": len(performance_rows),
            "performance_observation_count": sum(
                1 for row in performance_rows
                if _performance_observation(row)
            ),
            "task_quality_row_count": len(performance_rows),
            "eligibility_row_count": len(performance_rows),
            "native_energy_attempt_count": len(native_energy_attempts),
            "native_energy_success_count": len(native_energy_successes),
            "native_energy_failed_count": len(native_energy_failures),
            # Explicit aliases keep the distinction between attempts and
            # successful numeric observations visible to downstream readers.
            "native_energy_observation_count": len(native_energy_successes),
            "native_energy_failed_attempt_count": len(native_energy_failures),
            "ranking_eligible_count": sum(
                1
                for row in performance_rows
                if _b(row.get("ranking_eligible")) is True
            ),
            "performance_eligible_count": sum(
                1
                for row in performance_rows
                if _b(row.get("performance_eligible")) is True
            ),
            "energy_eligible_count": sum(
                1
                for row in scientific_rows
                if _b(row.get("energy_eligible")) is True
            ),
            "quality_failed_count": sum(
                1
                for row in performance_rows
                if str(row.get("task_quality_decision") or "").strip().lower()
                == "fail"
            ),
            "quality_inconclusive_count": sum(
                1
                for row in performance_rows
                if str(row.get("task_quality_decision") or "").strip().lower()
                in {
                    "inconclusive",
                    "legacy_point_estimate_only",
                    "screening_only",
                }
            ),
            "quality_status_counts": quality_status_counts,
            "eligibility_status_counts": eligibility_status_counts,
            # Compatibility alias retained for dashboards that expect it.
            "status_counts": eligibility_status_counts,
            "screening_performance_observation_count": (
                sum(
                    1
                    for row in performance_rows
                    if _performance_observation(row)
                    and _b(row.get("performance_eligible")) is not True
                )
            ),
            "screening_energy_observation_count": sum(
                1
                for row in scientific_rows
                if _energy_observation(row)
                and _b(row.get("energy_eligible")) is not True
            ),
            "screening_energy_attempt_count": sum(
                1
                for row in native_energy_attempts
                if _b(row.get("energy_eligible")) is not True
            ),
            "model_count": len(
                {
                    str(row.get("model_id") or "")
                    for row in scientific_rows
                    if row.get("model_id")
                }
            ),
        }
    )
    return result


def _md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]], *, max_rows: int = 200) -> str:
    if not rows:
        return "_No rows available._\n"
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows[:max_rows]:
        values: list[str] = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value if value not in (None, "") else "").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_Only the first {max_rows} of {len(rows)} rows are shown._")
    return "\n".join(lines) + "\n"


def _task_quality_bound_display(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Present a computed bound or an explicitly uncomputed point estimate.

    Legacy projections are in-memory only; the original result bytes and
    hashes are never updated by reporting.
    """

    projected = project_flat_quality_uncertainty(row)
    repetitions_raw = projected.get("task_quality_bootstrap_repetitions")
    repetitions: int | None
    if isinstance(repetitions_raw, bool):
        repetitions = None
    else:
        try:
            repetitions = int(repetitions_raw)
        except (TypeError, ValueError):
            repetitions = None
    skipped_reason = str(
        projected.get("task_quality_bootstrap_skipped_reason") or ""
    ).strip()
    ci_low = projected.get("task_quality_ci_low")
    computed = bool(
        repetitions is not None
        and repetitions > 0
        and ci_low not in (None, "")
        and not skipped_reason
    )
    if computed:
        projected.update({
            "task_quality_bound_value": ci_low,
            "task_quality_bound_evidence": "computed_95_percent_lcb",
            "task_quality_ci_computed": True,
        })
    else:
        reason = skipped_reason or (
            "zero_bootstrap_repetitions"
            if repetitions == 0
            else "bootstrap_computation_not_attested"
        )
        projected.update({
            "task_quality_bound_value": None,
            "task_quality_bound_evidence": f"not_computed:{reason}",
            "task_quality_ci_computed": False,
            "task_quality_uncertainty_label": "Unsicherheit nicht berechnet",
        })
    return projected


def _tex_escape(value: Any) -> str:
    text = str(value if value not in (None, "") else "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _tex_header(value: Any) -> str:
    """Escape prose headers while retaining trusted, package-owned math mode."""
    text = str(value if value not in (None, "") else "")
    if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        return text
    return _tex_escape(text)


def _tex_value(value: Any, digits: int = 3) -> str:
    number = _f(value)
    return f"{number:.{digits}f}" if number is not None else "N/A"


def _write_tex_table(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str, str]],
    caption: str,
    label: str,
    *, preview: bool = False,
) -> Path:
    alignment = "".join("r" if kind in {"number", "integer"} else "l" for _, _, kind in columns)
    if any(kind in {"number", "integer"} and _f(row.get(key)) is None for row in rows for key, _, kind in columns):
        caption += " N/A: metric unavailable or not estimable; see the row status/reason and source CSV for details."
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{_tex_escape(caption)}}}",
        f"\\label{{{_tex_escape(label)}}}",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(_tex_header(title) for _, title, _ in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values: list[str] = []
        for key, _, kind in columns:
            value = row.get(key)
            values.append(_tex_value(value, 0 if kind == "integer" else 3) if kind in {"number", "integer"} else _tex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    if preview and rows:
        from ..reporting_figures import table_preview
        cells = [[_tex_value(row.get(key), 0 if kind == "integer" else 3) if kind in {"number", "integer"} else str(row.get(key) if row.get(key) is not None else "")
                  for key, _, kind in columns] for row in rows]
        figure = table_preview([title for _, title, _ in columns], cells, caption=caption)
        _figure_outputs(figure, path.with_suffix(""))
    return write_text(path, "\n".join(lines))


def _figure_outputs(figure: Any, base: Path) -> list[Path]:
    import textwrap
    for axis in figure.axes:
        if len(axis.get_title()) > 80:
            axis.set_title(textwrap.fill(axis.get_title(), 80))
    output: list[Path] = []
    for extension in ("pdf", "png"):
        path = base.with_suffix(f".{extension}")
        figure.savefig(path, bbox_inches="tight", dpi=220 if extension == "png" else None)
        output.append(path)
    return output


def _figure_row_label(row):
    family = "Native" if canonical_runner(row.get("runner_regime")) == "native_fifo" or str(row.get("execution_mode") or "").startswith("native") else "Generic"
    return f"{family}: {row.get('model_id')} / {row.get('backend')} / {row.get('case_id')}"


def _energy_figure_details(figure, axis, rows):
    """Label the plotted comparison basis and only already stored intervals."""
    from matplotlib.patches import Patch
    bases = set()
    for position, (bar, row) in enumerate(zip(axis.patches, rows)):
        normalized = str(row.get("energy_comparison_basis") or "").startswith("host_normalized")
        bar.set_hatch("//" if normalized else None)
        bases.add(normalized)
        low, high = _f(row.get("energy_per_work_ci_low_j")), _f(row.get("energy_per_work_ci_high_j"))
        if (_f(row.get("energy_repeat_n")) or 0) >= 2 and low is not None and high is not None:
            axis.hlines(position, 1000 * low, 1000 * high, color="black", linewidth=1)
    legends = [Patch(facecolor="C0", hatch="//" if normalized else None,
                     label="TRT host-normalized estimate" if normalized else "Measured input energy")
               for normalized in sorted(bases)]
    axis.legend(handles=legends, loc="best", fontsize=9)
    counts = sorted({str(row.get("energy_repeat_n")) for row in rows if row.get("energy_repeat_n") is not None})
    durations = [v for row in rows if (v := _f(row.get("active_duration_s"))) is not None]
    window = f"mean window {min(durations):.2f}–{max(durations):.2f} s" if durations else "window duration N/A: not recorded"
    scopes = ", ".join(sorted({str(row.get("energy_scope") or "N/A") for row in rows}))
    intervals = sorted({str(100 * float(row["energy_confidence_level"])) + "%" for row in rows if _f(row.get("energy_confidence_level")) is not None})
    ci = "/".join(intervals) + " stored repetition CI" if intervals else "CI level N/A: not recorded"
    windows = ", ".join(sorted({str(row.get("energy_window_effective") or row.get("energy_window") or "N/A") for row in rows}))
    figure.text(.5, -.025, f"Scope: {scopes}; {windows}; n={','.join(counts) or 'N/A'}; {window}.\n"
                f"Bars: mean energy/image; lines: {ci} when n ≥ 2. Descriptive values; quality is a separate axis.",
                ha="center", va="top", fontsize=9)
    axis.set_xlabel("Comparison energy per completed image [mJ/image]")


def _make_figures(
    scientific_rows: Sequence[Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    ranking_macro: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    try:
        from ..reporting_figures import export_subplots
    except ImportError:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []

    performance_rows = [
        row for row in scientific_rows if _is_performance_row(row)
    ]
    counts = Counter(
        str(row.get("task_quality_status") or "unavailable")
        for row in performance_rows
    )
    if counts:
        figure, axis = export_subplots(figsize=(8.2, 4.3))
        labels, values = list(counts.keys()), list(counts.values())
        axis.bar(range(len(labels)), values)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set_ylabel("Rows")
        axis.set_title("Task-quality and eligibility status")
        axis.grid(axis="y", alpha=0.25)
        output.extend(_figure_outputs(figure, figures_dir / "task_quality_status_counts"))

    quality_rows = [
        row
        for row in performance_rows
        if _f(row.get("task_quality_delta")) is not None
    ]
    if quality_rows:
        quality_rows = quality_rows[:40]
        figure, axis = export_subplots(figsize=(9.2, max(4.0, 0.32 * len(quality_rows))))
        positions = list(range(len(quality_rows)))
        values = [float(_f(row.get("task_quality_delta")) or 0.0) for row in quality_rows]
        computed_rows = []
        point_only_rows = []
        for position, row, value in zip(positions, quality_rows, values):
            row = _task_quality_bound_display(row)
            lower, upper = _f(row.get("task_quality_ci_low")), _f(row.get("task_quality_ci_high"))
            if row.get("task_quality_ci_computed") is True and lower is not None and upper is not None:
                computed_rows.append((position, value, lower, upper))
            else:
                point_only_rows.append((position, value))
        if computed_rows:
            # Draw the interval itself, including the legitimate case where a
            # percentile interval does not contain the point estimate.
            axis.hlines([item[0] for item in computed_rows], [item[2] for item in computed_rows], [item[3] for item in computed_rows], label="Berechnetes Bootstrap-Intervall")
            axis.plot([item[1] for item in computed_rows], [item[0] for item in computed_rows], "o", linestyle="none")
        if point_only_rows:
            axis.plot([item[1] for item in point_only_rows], [item[0] for item in point_only_rows], "x", linestyle="none", label="Unsicherheit nicht berechnet")
        axis.legend(loc="best")
        margins = [float(_f(row.get("task_quality_margin")) or 0.01) for row in quality_rows]
        if margins:
            axis.axvline(-max(margins), linestyle="--", linewidth=1)
        axis.axvline(0.0, linewidth=1)
        axis.set_yticks(
            positions,
            [_figure_row_label(row) for row in quality_rows],
        )
        axis.set_xlabel("Candidate minus canonical reference")
        axis.set_title("Task-quality non-inferiority results")
        axis.grid(axis="x", alpha=0.25)
        output.extend(_figure_outputs(figure, figures_dir / "task_quality_noninferiority"))

    macro_rows = [row for row in ranking_macro if str(row.get("status") or "") != "unavailable"]
    if macro_rows:
        labels = [str(row.get("method_label") or row.get("method_id") or "") for row in macro_rows]
        for field, y_label, title, filename in (
            ("macro_spearman_rho", "Macro Spearman rho", "Ranking fidelity by method", "ranking_method_spearman"),
            ("macro_hit_at_5", "Macro Hit@5", "Best-candidate retention by method", "ranking_method_hit_at_5"),
            ("macro_regret_at_5", "Macro regret@5", "Shortlist regret by method", "ranking_method_regret_at_5"),
        ):
            values = [_f(row.get(field)) for row in macro_rows]
            if not any(value is not None for value in values):
                continue
            figure, axis = export_subplots(figsize=(9.0, 4.8))
            axis.bar(range(len(labels)), [float(value) if value is not None else math.nan for value in values])
            axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
            axis.set_ylabel(y_label)
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.25)
            output.extend(_figure_outputs(figure, figures_dir / filename))

    performance = [
        row
        for row in scientific_rows
        if _b(row.get("performance_eligible")) is True and _f(row.get("throughput_fps")) is not None
    ]
    if performance:
        performance = sorted(
            performance,
            key=lambda row: float(_f(row.get("throughput_fps")) or 0),
            reverse=True,
        )[:30]
        figure, axis = export_subplots(figsize=(9.2, max(4.0, 0.30 * len(performance))))
        labels = [_figure_row_label(row) for row in performance]
        values = [float(_f(row.get("throughput_fps")) or 0.0) for row in performance]
        axis.barh(range(len(performance)), values)
        axis.set_yticks(range(len(performance)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Throughput [FPS]")
        axis.set_title("Eligible full and split performance rows")
        axis.grid(axis="x", alpha=0.25)
        output.extend(_figure_outputs(figure, figures_dir / "eligible_throughput"))

    screening_performance = [
        row for row in scientific_rows
        if _performance_observation(row) and _b(row.get("performance_eligible")) is not True
    ]
    if screening_performance:
        screening_performance = sorted(
            screening_performance,
            key=lambda row: float(_f(row.get("throughput_fps")) or 0),
            reverse=True,
        )[:30]
        figure, axis = export_subplots(figsize=(9.2, max(4.0, 0.30 * len(screening_performance))))
        labels = [_figure_row_label(row) for row in screening_performance]
        values = [float(_f(row.get("throughput_fps")) or 0.0) for row in screening_performance]
        axis.barh(range(len(screening_performance)), values)
        axis.set_yticks(range(len(screening_performance)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Throughput [FPS]")
        axis.set_title("Development/screening throughput observations (not claim eligible)")
        axis.grid(axis="x", alpha=0.25)
        output.extend(_figure_outputs(figure, figures_dir / "screening_throughput_observations"))

    energy = [
        row
        for row in scientific_rows
        if _b(row.get("energy_eligible")) is True and _f(row.get("energy_per_work_j")) is not None
    ]
    if energy:
        energy = sorted(energy, key=lambda row: float(_f(row.get("energy_per_work_j")) or math.inf))[:30]
        figure, axis = export_subplots(figsize=(9.2, max(4.0, 0.30 * len(energy))))
        labels = [_figure_row_label(row) for row in energy]
        values = [1000.0 * float(_f(row.get("energy_per_work_j")) or 0.0) for row in energy]
        axis.barh(range(len(energy)), values)
        axis.set_yticks(range(len(energy)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Energy per completed work unit [mJ]")
        axis.set_title("Scope-compatible energy rows")
        axis.grid(axis="x", alpha=0.25)
        _energy_figure_details(figure, axis, energy)
        output.extend(_figure_outputs(figure, figures_dir / "eligible_energy_per_work"))

    screening_energy = [
        row for row in scientific_rows
        if _energy_observation(row) and _b(row.get("energy_eligible")) is not True
    ]
    if screening_energy:
        screening_energy = sorted(screening_energy, key=lambda row: float(_f(row.get("energy_per_work_j")) or math.inf))[:30]
        figure, axis = export_subplots(figsize=(9.2, max(4.0, 0.30 * len(screening_energy))))
        labels = [_figure_row_label(row) for row in screening_energy]
        values = [1000.0 * float(_f(row.get("energy_per_work_j")) or 0.0) for row in screening_energy]
        axis.barh(range(len(screening_energy)), values)
        axis.set_yticks(range(len(screening_energy)), labels)
        axis.invert_yaxis()
        axis.set_xlabel("Energy per completed work unit [mJ]")
        axis.set_title("Development/screening energy observations (not claim eligible)")
        axis.grid(axis="x", alpha=0.25)
        _energy_figure_details(figure, axis, screening_energy)
        output.extend(_figure_outputs(figure, figures_dir / "screening_energy_observations"))
    return output



def _runtime_quality_gate_artifact_present(source_root: Path) -> bool:
    """Detect the v60 per-run task-quality block in raw validation artefacts.

    Normalized result rows intentionally flatten the gate fields and therefore
    do not retain a nested ``task_quality_gate`` mapping.  Looking only at the
    normalized rows incorrectly labelled current runs as pre-v60.
    """
    root = Path(source_root)
    patterns = (
        "models/*/benchmark_results/**/validation_report.json",
        "models/*/validation/**/*.json",
        "b*/results*/validation_report.json",
        "b*/validation_report.json",
    )
    seen: set[str] = set()
    checked = 0
    for pattern in patterns:
        try:
            candidates = root.glob(pattern)
        except Exception:
            continue
        for path in candidates:
            if not path.is_file():
                continue
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            checked += 1
            # Keep report generation bounded for unusually large copied trees.
            if checked > 5000:
                return False
            payload = read_json(path, default={}) or {}
            if not isinstance(payload, Mapping):
                continue
            if isinstance(payload.get("task_quality_gate"), Mapping):
                return True
            gates = payload.get("task_quality_gates_by_variant")
            if isinstance(gates, Mapping) and gates:
                return True
    return False

def _build_report_payload(
    *,
    source_kind: str,
    source_root: Path,
    profile_id: str,
    tool_version: str,
    workflow_version: str,
    profile: Mapping[str, Any],
    policy: AccuracyGatePolicy,
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
    model_facts: Sequence[Mapping[str, Any]],
    ranking_input_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    scientific_rows = [_scientific_row(row) for row in rows]
    ranking_policy = _ranking_policy(profile)
    ranking_audit_request = _ranking_audit_request(profile, predictions)
    if ranking_policy.get("enabled"):
        ranking_rows, ranking_macro_rows, ranking_summary = _ranking_method_comparison(
            ranking_input_rows if ranking_input_rows is not None else rows,
            predictions,
            profile,
            ranking_policy,
        )
    else:
        ranking_rows, ranking_macro_rows, ranking_summary = [], [], {"status": "disabled"}
    ranking_cohort_sensitivity = list(
        ranking_summary.pop("cohort_sensitivity_rows", [])
    )
    open_items: list[dict[str, Any]] = []
    campaign_readiness: dict[str, Any] = {}
    for candidate in (
        source_root / "campaign" / "campaign_readiness.json",
        source_root / "campaign_readiness.json",
    ):
        if candidate.is_file():
            campaign_readiness = dict(read_json(candidate, default={}) or {})
            break
    claim_scope = resolve_campaign_claim_scope(profile)
    embedded_scope = str(campaign_readiness.get("claim_scope") or "")
    if embedded_scope in {
        EVALUATED_MATRIX_CLAIM_SCOPE,
        RANKING_GENERALIZATION_CLAIM_SCOPE,
    }:
        claim_scope = {
            "scope": embedded_scope,
            "valid": bool(campaign_readiness.get("claim_scope_valid", True)),
            "explicit": bool(campaign_readiness.get("claim_scope_explicit", False)),
            "source": "embedded_campaign_readiness",
        }
    generalization_claim = bool(
        not claim_scope.get("valid")
        or claim_scope.get("scope") == RANKING_GENERALIZATION_CLAIM_SCOPE
    )
    if not claim_scope.get("valid"):
        open_items.append(
            {
                "id": "campaign_claim_scope_invalid",
                "severity": "high",
                "detail": "The campaign claim scope is invalid. Reporting remains fail-closed under ranking_generalization requirements.",
            }
        )
    if campaign_readiness:
        campaign_mode = str(campaign_readiness.get("mode") or "development").lower()
        if campaign_mode == "final" and not bool(campaign_readiness.get("final_ready") or campaign_readiness.get("ready")):
            open_items.append(
                {
                    "id": "campaign_not_ready",
                    "severity": "high",
                    "detail": f"Final-campaign preflight status is {campaign_readiness.get('status', 'incomplete')} with {campaign_readiness.get('required_failure_count', 0)} required failures.",
                }
            )
        elif campaign_mode != "final":
            open_items.append(
                {
                    "id": "campaign_development_mode",
                    "severity": "informational",
                    "detail": f"Campaign status is {campaign_readiness.get('status', 'development_ready')}; {campaign_readiness.get('deferred_final_requirement_count', 0)} final-only requirements remain deferred.",
                }
            )
    if not policy.frozen_before_final_campaign:
        open_items.append(
            {
                "id": "quality_profile_not_frozen",
                "severity": "required_for_final",
                "detail": "Set quality_gate.frozen_before_final_campaign=true before the final campaign and archive the profile hash.",
            }
        )
    if policy.dataset_tier != "final":
        open_items.append(
            {
                "id": "screening_dataset_only",
                "severity": "required_for_final",
                "detail": "The configured task-quality tier is screening; these rows are not final task-quality evidence.",
            }
        )
    if not generalization_claim:
        open_items.append(
            {
                "id": "ranking_generalization_out_of_scope",
                "severity": "informational",
                "detail": "The scientific claim is limited to the evaluated workload/hardware matrix; unseen-model ranking generalisation is not claimed.",
            }
        )
    if generalization_claim and ranking_summary.get("status") != "holdout_available":
        open_items.append(
            {
                "id": "independent_holdout_missing",
                "severity": "required_for_final",
                "detail": "No complete, hash-verified model-level hold-out comparison is available for all configured ranking methods.",
            }
        )
    if generalization_claim and any(row.get("candidate_universe_complete") is False for row in ranking_rows):
        open_items.append(
            {
                "id": "candidate_universe_incomplete",
                "severity": "required_for_final",
                "detail": "Top-k recall and regret remain unavailable until every candidate in the declared hold-out universe is measured or explicitly accounted for.",
            }
        )
    if generalization_claim and bool(ranking_policy.get("require_frozen_predictions", True)) and any(
        is_confirmatory_holdout(row.get("evaluation_role")) and row.get("predictions_frozen") is not True
        for row in ranking_rows
    ):
        open_items.append(
            {
                "id": "predictions_not_frozen",
                "severity": "required_for_final",
                "detail": "At least one hold-out method/context lacks a hash-verified prospective ranking prediction artefact created before benchmark execution.",
            }
        )
    if generalization_claim and any(str(row.get("status") or "") == "handover_model_unavailable" for row in ranking_rows):
        open_items.append(
            {
                "id": "native_handover_model_unconfigured",
                "severity": "required_for_final",
                "detail": "The Native FIFO cycle-time-with-handover baseline requires a frozen runner- and direction-specific handover model or prospective native handover fields.",
            }
        )
    if generalization_claim and any(str(row.get("status") or "") == "method_unavailable" for row in ranking_rows):
        open_items.append(
            {
                "id": "ranking_method_inputs_missing",
                "severity": "medium",
                "detail": "At least one ranking method lacks candidate features or calibrated backend parameters required for a prediction.",
            }
        )
    if any(row.get("accuracy_gate_policy_match") is False for row in rows):
        open_items.append(
            {
                "id": "runtime_quality_policy_mismatch",
                "severity": "high",
                "detail": "At least one runtime task-quality gate used a policy hash that differs from the active evaluation profile. The measured gate is reported, but the row is not claim eligible.",
            }
        )
    if any(str(row.get("gate_status") or "") == "legacy_point_estimate_only" for row in rows):
        open_items.append(
            {
                "id": "legacy_quality_rows",
                "severity": "medium",
                "detail": "Some imported rows contain only point-estimate accuracy metrics and remain inconclusive under the new gate.",
            }
        )
    runtime_gate_present = bool(
        any(isinstance(row.get("task_quality_gate"), Mapping) for row in rows)
        or _runtime_quality_gate_artifact_present(source_root)
    )
    if rows and not runtime_gate_present:
        open_items.append(
            {
                "id": "runtime_quality_gate_not_embedded",
                "severity": "high",
                "detail": "No v60 per-sample task-quality block was found in the normalized rows or raw validation reports; final confidence-bound decisions cannot be reconstructed from aggregate rows alone.",
            }
        )
    summary = _refresh_scientific_summary(
        {
            "ranking": ranking_summary,
        },
        scientific_rows,
    )
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": source_root.name if source_kind == "evaluation_run" else "",
        "created_at": now_iso(),
        "source_kind": source_kind,
        "source_root": str(source_root),
        "profile_id": profile_id,
        "tool_version": tool_version,
        "workflow_version": workflow_version,
        "quality_gate_policy": policy.as_dict(),
        "ranking_validation_policy": ranking_policy,
        "ranking_audit_request": ranking_audit_request,
        "claim_scope": str(claim_scope.get("scope") or ""),
        "campaign_claim_scope": dict(claim_scope),
        "campaign_readiness": campaign_readiness,
        "summary": summary,
        "model_facts": list(model_facts),
        "rows": scientific_rows,
        "ranking_method_comparison": ranking_rows,
        "ranking_method_macro": ranking_macro_rows,
        "ranking_method_cohort_sensitivity": ranking_cohort_sensitivity,
        "open_items": open_items,
    }


def _native_ranking_row_has_usable_metric(row: Mapping[str, Any]) -> bool:
    if str(row.get("status") or "").strip() not in {
        "development_evidence_only",
        "holdout_validated",
    }:
        return False
    if any(
        _f(row.get(key)) is not None
        for key in ("spearman_rho", "kendall_tau_b")
    ):
        return True
    return any(
        (
            str(key).startswith(
                (
                    "hit_at_",
                    "near_optimal_hit_at_",
                    "regret_at_",
                    "elite_recall_at_",
                )
            )
            and value is not None
        )
        for key, value in row.items()
    )


def _native_ranking_audit_summary(
    native_ranking: Sequence[Mapping[str, Any]],
    audit_request: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build an audit status that reflects request, scope and usable output."""

    rows = [dict(row) for row in native_ranking if isinstance(row, Mapping)]
    request = dict(audit_request or {}) if isinstance(audit_request, Mapping) else {}
    row_declares_audit = any(
        _b(row.get("audit_scope_enforced")) is True
        or str(row.get("candidate_universe_scope") or "").strip()
        == "predeclared_audit_universe"
        for row in rows
    )
    requested = bool(_b(request.get("requested")) is True or row_declares_audit)
    native_execution_enabled = _b(request.get("native_execution_enabled"))
    audit_rows = [
        row
        for row in rows
        if _b(row.get("audit_scope_enforced")) is True
        or str(row.get("candidate_universe_scope") or "").strip()
        == "predeclared_audit_universe"
    ]
    usable_rows = [
        row for row in audit_rows if _native_ranking_row_has_usable_metric(row)
    ]

    def _count(row: Mapping[str, Any], key: str) -> int:
        try:
            return max(0, int(row.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    quality_shortfalls = [
        row
        for row in audit_rows
        if _count(row, "minimum_valid_candidates_required") > 0
        and _count(row, "measured_candidate_count")
        >= _count(row, "minimum_valid_candidates_required")
        and _count(row, "valid_candidate_count")
        < _count(row, "minimum_valid_candidates_required")
        and _count(row, "quality_vetoed_candidate_count") > 0
    ]
    measured_shortfalls = [
        row
        for row in audit_rows
        if _count(row, "minimum_valid_candidates_required") > 0
        and _count(row, "measured_candidate_count")
        < _count(row, "minimum_valid_candidates_required")
    ]
    if not requested:
        status = "not_requested"
        detail = (
            "The frozen selection plan and prediction freezes do not declare "
            "a score-independent ranking audit."
        )
        scope = "not_requested"
    elif native_execution_enabled is False and not rows:
        status = "native_disabled"
        detail = (
            "The score-independent audit was requested, but Native execution "
            "was explicitly disabled in the frozen profile. No missing "
            "Native metric is inferred from that planned configuration."
        )
        scope = "predeclared_score_independent_candidates"
    elif usable_rows:
        status = "available"
        detail = "At least one audit-scoped Native predictor metric is available."
        scope = "predeclared_score_independent_candidates"
    elif quality_shortfalls:
        status = "insufficient_valid_candidates_after_quality_veto"
        detail = (
            "Audit candidates were measured, but row-local task-quality "
            "vetoes left no usable Native metric at the required minimum."
        )
        scope = "predeclared_score_independent_candidates"
    elif measured_shortfalls:
        status = "insufficient_measured_candidates"
        detail = (
            "The requested Native audit did not measure the required number "
            "of candidates in a comparable stratum."
        )
        scope = "predeclared_score_independent_candidates"
    elif not audit_rows:
        status = "requested_but_unavailable"
        detail = (
            "The audit was requested, but no audit-scoped Native ranking "
            "group was produced."
        )
        scope = "predeclared_score_independent_candidates"
    else:
        status = "requested_but_no_usable_metrics"
        detail = (
            "Audit-scoped Native rows exist, but no correlation, hit or regret "
            "metric passed all ranking prerequisites."
        )
        scope = "predeclared_score_independent_candidates"
    return {
        "schema": "onnx-splitpoint/native-ranking-audit",
        "schema_version": 1,
        "status": status,
        "status_detail": detail,
        "audit_requested": requested,
        "native_execution_enabled": native_execution_enabled,
        "native_execution_enabled_source": str(
            request.get("native_execution_enabled_source") or "unavailable"
        ),
        "audit_request_source": str(
            request.get("source")
            or ("row_scope" if row_declares_audit else "none")
        ),
        "quality_gate_policy": "row_local_veto",
        "comparison_strata": list(RANKING_COMPARISON_STRATA),
        "audit_scope": scope,
        "group_method_row_count": len(rows),
        "audit_group_method_row_count": len(audit_rows),
        "usable_metric_row_count": len(usable_rows),
        "quality_veto_shortfall_group_method_row_count": len(
            quality_shortfalls
        ),
        "measured_shortfall_group_method_row_count": len(measured_shortfalls),
        "row_status_counts": dict(
            Counter(str(row.get("status") or "unavailable") for row in rows)
        ),
        "rows": rows,
    }


def _write_reports(
    report_root: Path,
    payload: Mapping[str, Any],
    *,
    compatibility_root: Optional[Path] = None,
) -> dict[str, Path]:
    payload = dict(payload)
    payload_rows = [
        project_flat_quality_uncertainty(row)
        for row in list(payload.get("rows") or [])
        if isinstance(row, Mapping)
    ]
    payload["rows"] = payload_rows
    payload["summary"] = _refresh_scientific_summary(
        payload.get("summary")
        if isinstance(payload.get("summary"), Mapping)
        else {},
        payload_rows,
    )
    central_quality_reporting = (
        dict(payload.get("central_quality_reporting") or {})
        if isinstance(payload.get("central_quality_reporting"), Mapping)
        else {}
    )
    if central_quality_reporting.get("source_present"):
        decision_counts = dict(
            central_quality_reporting.get("decision_counts") or {}
        )
        refreshed_summary = dict(payload.get("summary") or {})
        refreshed_summary.update({
            "task_quality_row_count": int(
                central_quality_reporting.get("result_count") or 0
            ),
            "quality_status_counts": decision_counts,
            "quality_failed_count": int(decision_counts.get("fail") or 0),
            "quality_inconclusive_count": int(
                decision_counts.get("inconclusive") or 0
            ),
            "quality_not_evaluated_count": int(
                central_quality_reporting.get("not_evaluated_count") or 0
            ),
            **{
                "central_quality_" + name: central_quality_reporting.get(name, 0)
                for name in (
                    "terminal_count", "evaluated_count", "cancelled_count",
                    "technical_failed_count", "queued_count", "running_count",
                )
            },
        })
        payload["summary"] = refreshed_summary
    native_evidence = (
        dict(payload.get("native_evidence_status") or {})
        if isinstance(payload.get("native_evidence_status"), Mapping)
        else {}
    )
    native_evidence_projection = project_native_evidence_status(
        native_evidence
    )
    payload["native_evidence_summary"] = native_evidence_projection
    # The subtree contains presentation artefacts only. Recreate it atomically
    # at report granularity so superseded files such as ranking_validation.csv
    # cannot survive a v60 regeneration. Raw benchmark rows and traces live
    # outside this directory and are never touched here.
    if report_root.is_dir():
        shutil.rmtree(report_root, ignore_errors=True)
    report_root.mkdir(parents=True, exist_ok=True)
    tables = report_root / "thesis_tables"
    figures = report_root / "figures"
    if tables.is_dir():
        shutil.rmtree(tables, ignore_errors=True)
    if figures.is_dir():
        shutil.rmtree(figures, ignore_errors=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rows = list(payload.get("rows") or [])
    performance_rows = [
        row for row in rows if _is_performance_row(row)
    ]
    central_quality_rows = [
        _task_quality_bound_display(row)
        for row in list(payload.get("central_quality_results") or [])
        if isinstance(row, Mapping)
    ]
    # Central management results are the canonical task-quality result set
    # when present.  They include semantic-only, setup-local Full TensorRT
    # companions that intentionally have no normalized performance row.
    task_quality_rows = central_quality_rows or [
        _task_quality_bound_display(row) for row in performance_rows
    ]
    payload["central_quality_results"] = central_quality_rows
    if payload.get("quality_reporting_policy"):
        write_json(report_root / "quality_policy.json", payload["quality_reporting_policy"])
        from ..accuracy_reporting import observed_coverage
        write_json(report_root / "sentinel_coverage.json", observed_coverage(
            central_quality_rows, run_id=str(payload.get("run_id") or ""),
            run_root=report_root.parent.parent))
    ranking = list(payload.get("ranking_method_comparison") or [])
    ranking_macro = list(payload.get("ranking_method_macro") or [])
    ranking_cohort_sensitivity = list(
        payload.get("ranking_method_cohort_sensitivity") or []
    )
    cross_runner_pairs = list(payload.get("cross_runner_candidate_pairs") or [])
    cross_runner_groups = list(payload.get("cross_runner_ranking_validation") or [])
    cross_runner_macro = payload.get("cross_runner_ranking_macro") if isinstance(payload.get("cross_runner_ranking_macro"), Mapping) else {}
    native_energy_observations = list(payload.get("native_energy_observations") or [])
    native_energy_pairs = list(payload.get("native_energy_pair_comparison") or [])
    native_energy_ab_aggregates = list(payload.get("native_energy_ab_aggregates") or [])
    native_performance_matrix = payload.get("native_performance_matrix") if isinstance(payload.get("native_performance_matrix"), Mapping) else {}
    native_performance_observations = [
        dict(row) for row in list(native_performance_matrix.get("observations") or [])
        if isinstance(row, Mapping)
    ]
    native_comparison_eligible_observations = [
        dict(row)
        for row in list(native_performance_matrix.get("comparison_eligible_observations") or [])
        if isinstance(row, Mapping)
    ]
    native_evidence_columns = [
        ("model_id", "Model"), ("setup_id", "Setup"),
        ("backend", "Backend"), ("case_id", "Case"),
        ("execution_mode", "Mode"),
        ("claim_ok_source", "Claim"),
        ("claim_input_source", "Input claim"),
        ("claim_structural_gate_pass", "Claim structure"),
        (
            "claim_ok_structural_clamped",
            "Claim structurally clamped",
        ),
        (
            "quality_physical_evidence_conflict",
            "Physical evidence conflict",
        ),
        (
            "quality_physical_evidence_conflict_fields",
            "Physical conflict fields",
        ),
        ("numerical_similarity_mean_iou", "Mean IoU"),
        (
            "numerical_similarity_mean_iou_threshold",
            "Mean IoU threshold",
        ),
        ("source_e2e_scope", "Source E2E scope"),
        ("e2e_scope", "E2E scope"),
        ("e2e_claim_eligible", "E2E eligible"),
        ("e2e_contract_reason", "E2E reason"),
        ("comparison_endpoint_stratum", "Endpoint stratum"),
        ("measurement_concurrency", "Concurrency"),
        ("stage", "Physical stage"),
        ("output_format", "Output format"),
        ("contract_family", "Physical contract"),
        ("contract_source", "Contract source"),
        ("endpoint_contract_complete", "Physical contract complete"),
        ("endpoint_contract_hash", "Physical contract hash"),
        ("accelerator_output_stage", "Accelerator stage"),
        (
            "accelerator_output_contract_family",
            "Accelerator contract",
        ),
        (
            "accelerator_endpoint_contract_hash",
            "Accelerator contract hash",
        ),
        ("comparison_output_endpoint_id", "Comparison endpoint"),
        ("physical_output_endpoint_id", "Physical endpoint"),
        ("output_endpoint_match", "Physical endpoint match"),
        ("comparison_endpoint_match", "Comparison endpoint match"),
        (
            "comparison_stratum_explicit",
            "Comparison stratum explicit",
        ),
        (
            "requires_host_decode_nms",
            "Host decode/NMS required",
        ),
        ("postprocess_included", "Postprocess included"),
        ("postprocess_location", "Postprocess location"),
        ("host_postprocess_frozen", "Host tail frozen"),
        (
            "host_postprocess_required",
            "Host postprocess required",
        ),
        ("host_tail_required", "Host tail required"),
        (
            "host_postprocessing_available",
            "Host postprocess available",
        ),
        ("host_tail_available", "Host tail available"),
        (
            "host_postprocessing_evidence_status",
            "Host postprocess evidence",
        ),
        (
            "host_postprocessing_evidence_source",
            "Host evidence source",
        ),
        (
            "host_postprocessing_legacy_alias_conflict",
            "Host alias conflict",
        ),
        ("decoder_contract_pass", "Decoder contract"),
        ("nms_ok", "NMS"),
        ("decoder_id", "Decoder ID"),
        (
            "postprocess_completed_frames",
            "Postprocess completed frames",
        ),
        (
            "postprocess_completion_verified",
            "Postprocess completion verified",
        ),
        (
            "frozen_host_postprocess_contract_sha256",
            "Frozen host contract hash",
        ),
        ("completed_task_stage", "Completed stage"),
        (
            "completed_task_contract_family",
            "Completed contract",
        ),
        (
            "completed_task_endpoint_contract_hash",
            "Completed contract hash",
        ),
        (
            "completed_task_output_endpoint_id",
            "Completed endpoint",
        ),
        (
            "completed_task_comparison_endpoint_contract_hash",
            "Completed comparison contract hash",
        ),
        (
            "completed_task_comparison_output_endpoint_id",
            "Completed comparison endpoint",
        ),
        (
            "completed_task_completion_mode",
            "Completion mode",
        ),
        (
            "completed_task_endpoint_attested",
            "Completed endpoint attested",
        ),
        (
            "completed_task_endpoint_attestation_status",
            "Completed attestation status",
        ),
        ("native_measured_throughput_fps", "Completed Task FPS median"),
        ("fps_ci95_low", "Completed Task CI95 low"),
        ("fps_ci95_high", "Completed Task CI95 high"),
        ("p2_output_fps", "P2 output FPS median"),
        ("p2_output_fps_ci95_low", "P2 CI95 low"),
        ("p2_output_fps_ci95_high", "P2 CI95 high"),
        ("completed_task_fps_unavailable_reason", "Task FPS unavailable reason"),
        ("historical_fps", "Historical diagnostic FPS"),
        ("historical_fps_ci95_low", "Historical CI95 low"),
        ("historical_fps_ci95_high", "Historical CI95 high"),
        ("historical_performance_endpoint", "Historical endpoint"),
        ("historical_fps_source", "Historical rate source"),
        ("request_latency_mean_ms", "Einbildlatenz Mean [ms] ab vorbereitetem Input"),
        ("request_latency_p50_ms", "Einbildlatenz P50 [ms]"),
        ("request_latency_p95_ms", "Einbildlatenz P95 [ms]"),
        ("request_latency_count", "Latenz n"),
        ("request_latency_expected_count", "Latenz expected n"),
        ("request_latency_semantics", "Latenzsemantik"),
        ("request_latency_status", "Latenznachweis"),
        ("request_latency_unavailable_reason", "Latenz fehlender Nachweis"),
        ("host_output_latency_mean_ms", "Hostoutputlatenz Mean [ms] ohne Task-Postprocessing"),
        ("host_output_fps", "Hostoutput FPS ohne Task-Postprocessing"),
        ("host_output_fps_ci95_low", "Hostoutput FPS CI95 low"),
        ("host_output_fps_ci95_high", "Hostoutput FPS CI95 high"),
        ("host_output_rate_endpoint", "Hostoutput-Endpunkt"),
        ("latency_median_ms", "Legacy latency median [ms]"),
        ("latency_ci95_low_ms", "Latency CI95 low [ms]"),
        ("latency_ci95_high_ms", "Latency CI95 high [ms]"),
        ("repetition_count_valid", "valid n"),
        ("repetition_aggregation", "Aggregation"),
        (
            "native_theoretical_cycle_rate_fps",
            "Theoretical cycle rate",
        ),
        ("structural_contract_status", "Structure"),
        ("numerical_similarity_status", "Numerical"),
        ("task_quality_status", "Task quality"),
    ]
    native_evidence_number_fields = {
        "request_latency_mean_ms", "request_latency_p50_ms", "request_latency_p95_ms",
        "request_latency_count", "request_latency_expected_count", "host_output_latency_mean_ms",
        "historical_fps", "historical_fps_ci95_low", "historical_fps_ci95_high",
        "fps_ci95_low", "fps_ci95_high", "p2_output_fps", "p2_output_fps_ci95_low", "p2_output_fps_ci95_high",
        "numerical_similarity_mean_iou",
        "numerical_similarity_mean_iou_threshold",
        "measurement_concurrency",
        "postprocess_completed_frames",
        "native_measured_throughput_fps",
        "latency_median_ms",
        "latency_ci95_low_ms",
        "latency_ci95_high_ms",
        "repetition_count_valid",
        "native_theoretical_cycle_rate_fps",
    }
    native_evidence_tex_columns = [
        (
            key,
            label,
            "number" if key in native_evidence_number_fields else "text",
        )
        for key, label in native_evidence_columns
    ]
    cross_runner_identity_diagnostics = (
        dict(payload.get("cross_runner_identity_diagnostics") or {})
        if isinstance(payload.get("cross_runner_identity_diagnostics"), Mapping)
        else {}
    )
    cross_runner_payload = {
        # The JSON report already carries these values.  Propagate the same
        # bounded-denominator diagnostics into the human-readable report so a
        # failed join cannot be rendered misleadingly as 0/0.
        **cross_runner_identity_diagnostics,
        "schema": payload.get("cross_runner_schema") or "onnx-splitpoint/cross-runner-ranking-transfer",
        "schema_version": int(payload.get("cross_runner_schema_version") or 1),
        "status": payload.get("cross_runner_ranking_status") or cross_runner_macro.get("status") or "unavailable",
        "pair_count": len(cross_runner_pairs),
        "technical_pair_count": sum(
            bool(row.get("eligible_for_technical_transfer"))
            for row in cross_runner_pairs if isinstance(row, Mapping)
        ),
        "quality_pair_count": sum(
            bool(row.get("eligible_for_quality_transfer"))
            for row in cross_runner_pairs if isinstance(row, Mapping)
        ),
        "claim_pair_count": sum(
            bool(row.get("eligible_for_claim_transfer"))
            for row in cross_runner_pairs if isinstance(row, Mapping)
        ),
        "eligible_pair_count": sum(bool(row.get("eligible_for_transfer")) for row in cross_runner_pairs if isinstance(row, Mapping)),
        "pairs": cross_runner_pairs,
        "groups": cross_runner_groups,
        "macro": dict(cross_runner_macro),
    }
    campaign_readiness = payload.get("campaign_readiness") if isinstance(payload.get("campaign_readiness"), Mapping) else {}
    campaign_claim_scope = payload.get("campaign_claim_scope") if isinstance(payload.get("campaign_claim_scope"), Mapping) else {}
    endpoint_lifecycle = (
        dict(payload.get("endpoint_lifecycle") or {})
        if isinstance(payload.get("endpoint_lifecycle"), Mapping)
        else {}
    )
    endpoint_lifecycle_rows = [
        dict(row)
        for row in list(endpoint_lifecycle.get("rows") or [])
        if isinstance(row, Mapping)
    ]
    endpoint_lifecycle_summary = {
        "schema": "onnx-splitpoint/endpoint-lifecycle-summary",
        "schema_version": 1,
        **(
            dict(endpoint_lifecycle.get("summary") or {})
            if isinstance(endpoint_lifecycle.get("summary"), Mapping)
            else {}
        ),
        "adapter": (
            dict(endpoint_lifecycle.get("adapter") or {})
            if isinstance(endpoint_lifecycle.get("adapter"), Mapping)
            else {}
        ),
        "diagnostics": (
            dict(endpoint_lifecycle.get("diagnostics") or {})
            if isinstance(endpoint_lifecycle.get("diagnostics"), Mapping)
            else {}
        ),
    }
    artifacts: dict[str, Path] = {}
    artifacts["scientific_report_json"] = write_json(report_root / "scientific_report.json", payload)
    artifacts["row_eligibility_csv"] = write_csv(report_root / "row_eligibility.csv", rows)
    artifacts["endpoint_lifecycle_ledger_csv"] = write_csv(
        report_root / "endpoint_lifecycle_ledger.csv",
        endpoint_lifecycle_rows,
    )
    artifacts["endpoint_lifecycle_ledger_json"] = write_json(
        report_root / "endpoint_lifecycle_ledger.json",
        endpoint_lifecycle,
    )
    artifacts["endpoint_lifecycle_summary_json"] = write_json(
        report_root / "endpoint_lifecycle_summary.json",
        endpoint_lifecycle_summary,
    )
    artifacts["central_quality_results_json"] = write_json(
        report_root / "central_quality_results.json", central_quality_rows,
    )
    artifacts["central_quality_results_csv"] = write_csv(
        report_root / "central_quality_results.csv", central_quality_rows,
    )
    artifacts["task_quality_json"] = write_json(
        report_root / "task_quality.json", task_quality_rows,
    )
    artifacts["task_quality_csv"] = write_csv(
        report_root / "task_quality.csv",
        [
            {
                key: row.get(key)
                for key in (
                    "model_id",
                    "task",
                    "case_id",
                    "backend",
                    "setup_id",
                    "run_id",
                    "source_run_id",
                    "variant",
                    "execution_role",
                    "technical_status",
                    "accuracy_assessment", "accuracy_class", "accuracy_relative_loss", "accuracy_absolute_loss_pp",
                    "accuracy_relative_loss_ci", "accuracy_uncertainty", "accuracy_uncertainty_reason", "accuracy_policy_id",
                    "task_quality_tier",
                    "task_quality_status",
                    "task_quality_decision",
                    "task_quality_gate_reason",
                    "contract_gate_reason",
                    "eligibility_gate_reason",
                    "task_quality_metric",
                    "task_quality_candidate",
                    "task_quality_reference",
                    "task_quality_delta",
                    "task_quality_ci_low",
                    "task_quality_ci_high",
                    "task_quality_decision_basis",
                    "task_quality_gate_bound_value",
                    "task_quality_ci_computed",
                    "task_quality_uncertainty_status",
                    "task_quality_uncertainty_label",
                    "quality_result_source_contract_version",
                    "quality_result_projection_contract_version",
                    "legacy_quality_result",
                    "task_quality_margin",
                    "task_quality_bootstrap_repetitions_requested",
                    "task_quality_bootstrap_repetitions",
                    "task_quality_bootstrap_engine",
                    "task_quality_bootstrap_skipped_reason",
                    "task_quality_bootstrap_elapsed_s",
                    "task_quality_ci_computed",
                    "task_quality_bound_value",
                    "task_quality_bound_evidence",
                    "task_quality_ap50_candidate",
                    "task_quality_ap50_reference",
                    "task_quality_ap50_delta",
                    "task_quality_ap50_ci_low",
                    "task_quality_ap50_ci_high",
                    "task_quality_ap50_margin",
                    "task_quality_ap50_decision",
                    "task_quality_ap75_candidate",
                    "task_quality_ap75_reference",
                    "task_quality_ap75_delta",
                    "task_quality_ap75_ci_low",
                    "task_quality_ap75_ci_high",
                    "task_quality_ap75_margin",
                    "task_quality_ap75_decision",
                    "task_quality_bootstrap_candidate_event_count",
                    "task_quality_bootstrap_reference_event_count",
                    "validation_requested_count",
                    "validation_evaluated_count",
                    "validation_cardinality_status",
                    "validation_cardinality_pass",
                    "contract_consistent",
                    "ranking_eligible",
                    "exclusion_reason",
                    "quality_result_id",
                    "source_task_quality_decision",
                    "algorithm_version",
                    "quality_result_contract_version",
                    "metric_gate_config",
                    "configured_guardrails",
                    "missing_guardrails",
                    "guardrail_contract_complete",
                    "reference_predictions_sha256",
                    "candidate_predictions_sha256",
                    "annotations_sha256",
                    "source_request_sha256",
                    "producer_identity_sha256",
                )
            }
            for row in task_quality_rows
        ],
    )
    performance_observations = [
        row for row in performance_rows if _performance_observation(row)
    ]
    performance = [
        row
        for row in performance_observations
        if _performance_claim_eligible(row)
    ]
    energy = [row for row in rows if _energy_claim_eligible(row)]
    performance_cohorts, performance_cohort_summary = (
        _performance_cohort_projection(performance_rows)
    )
    cohort_summary_projection = dict(payload.get("summary") or {})
    cohort_summary_projection.update({
        "performance_technical_cohort_count": int(
            performance_cohort_summary["technical_cohort_count"]
        ),
        "performance_quality_cohort_count": int(
            performance_cohort_summary["quality_cohort_count"]
        ),
        "performance_claim_cohort_count": int(
            performance_cohort_summary["claim_cohort_count"]
        ),
        "performance_cohort_consistency_error_count": int(
            performance_cohort_summary["cohort_consistency_error_count"]
        ),
    })
    payload["summary"] = cohort_summary_projection
    payload["performance_cohort_summary"] = performance_cohort_summary
    generic_screening_performance = [
        {**dict(row), "claim_eligible": False, "observation_tier": "development_or_screening"}
        for row in performance_rows
        if _performance_observation(row)
        and not _performance_claim_eligible(row)
    ]
    # ``native_performance_matrix.observations`` is a second projection of the
    # same normalized performance rows.  Appending it here counted the Native
    # observations twice (48 normalized observations became 93 CSV rows in the
    # reference replay).  Keep the dedicated Native matrix artefacts below,
    # while the screening table remains one row per normalized observation.
    screening_performance = generic_screening_performance
    screening_energy = [
        {**dict(row), "claim_eligible": False, "observation_tier": "development_or_screening"}
        for row in rows
        if (
            _is_native_energy_attempt(row)
            or _energy_observation(row)
        )
        and not _energy_claim_eligible(row)
    ]
    artifacts["claim_eligible_performance_csv"] = write_csv(
        report_root / "claim_eligible_performance.csv",
        performance,
        fieldnames=CLAIM_PERFORMANCE_CSV_FIELDS,
    )
    artifacts["claim_eligible_energy_csv"] = write_csv(
        report_root / "claim_eligible_energy.csv",
        energy,
        fieldnames=CLAIM_ENERGY_CSV_FIELDS,
    )
    performance_excluded_rows = [
        row for row in performance_rows
        if not _performance_claim_eligible(row)
    ]
    energy_excluded_rows = [
        row for row in rows
        if (
            _is_native_energy_attempt(row) or _energy_observation(row)
        )
        and not _energy_claim_eligible(row)
    ]
    exclusion_details = _claim_exclusion_detail_rows(
        performance_excluded_rows, energy_excluded_rows,
    )
    grouped_exclusion_counts = _claim_exclusion_group_rows(
        exclusion_details
    )
    dimension_exclusion_counts = _claim_exclusion_dimension_rows(
        exclusion_details
    )
    # Compatibility reason maps remain one primary reason per excluded source
    # row. The detailed/grouped contracts retain every explicit reason
    # assignment and document that counting semantic separately.
    performance_exclusions = Counter(
        _claim_exclusion_reasons(
            row, claim_kind="performance",
        )[0]
        for row in performance_excluded_rows
    )
    energy_exclusions = Counter(
        _claim_exclusion_reasons(row, claim_kind="energy")[0]
        for row in energy_excluded_rows
    )
    claim_exclusion_summary = {
        "schema": "onnx-splitpoint/claim-exclusion-summary",
        "schema_version": 2,
        "performance_claim_eligible_count": len(performance),
        "energy_claim_eligible_count": len(energy),
        "performance_excluded_count": len(
            performance_excluded_rows
        ),
        "energy_excluded_count": len(energy_excluded_rows),
        "excluded_row_count": (
            len(performance_excluded_rows) + len(energy_excluded_rows)
        ),
        "exclusion_reason_assignment_count": len(exclusion_details),
        "performance_exclusion_reason_counts": dict(sorted(
            performance_exclusions.items()
        )),
        "energy_exclusion_reason_counts": dict(sorted(
            energy_exclusions.items()
        )),
        "exclusion_details": exclusion_details,
        "grouped_exclusion_counts": grouped_exclusion_counts,
        "dimension_exclusion_counts": dimension_exclusion_counts,
        "group_dimensions": [
            "model_id", "backend", "setup_id", "reason",
        ],
        "dimension_order": [
            dimension for dimension, _key
            in CLAIM_EXCLUSION_DIMENSIONS
        ],
        "count_semantics": (
            "details_and_grouped_counts_count_one record per excluded "
            "claim-kind/source-row/reason assignment; excluded_count fields "
            "count source rows"
        ),
        "stable_sort_contract": {
            "details": [
                "claim_kind", "model_id", "backend", "setup_id", "reason",
                "task", "case_id", "variant", "row_status",
                "eligibility_status",
            ],
            "grouped_counts": [
                "claim_kind", "model_id", "backend", "setup_id", "reason",
            ],
            "dimension_counts": [
                "claim_kind", "dimension_order", "value",
            ],
        },
    }
    artifacts["claim_exclusion_details_csv"] = write_csv(
        report_root / "claim_exclusion_details.csv",
        exclusion_details,
        fieldnames=CLAIM_EXCLUSION_DETAIL_CSV_FIELDS,
    )
    artifacts["claim_exclusion_grouped_counts_csv"] = write_csv(
        report_root / "claim_exclusion_grouped_counts.csv",
        grouped_exclusion_counts,
        fieldnames=CLAIM_EXCLUSION_GROUP_CSV_FIELDS,
    )
    artifacts["claim_exclusion_dimension_counts_csv"] = write_csv(
        report_root / "claim_exclusion_dimension_counts.csv",
        dimension_exclusion_counts,
        fieldnames=CLAIM_EXCLUSION_DIMENSION_CSV_FIELDS,
    )
    artifacts["claim_exclusion_summary_json"] = write_json(
        report_root / "claim_exclusion_summary.json",
        claim_exclusion_summary,
    )
    payload_summary = dict(payload.get("summary") or {})
    payload_summary.update({
        "performance_claim_excluded_count": len(
            performance_excluded_rows
        ),
        "energy_claim_excluded_count": len(energy_excluded_rows),
        "claim_exclusion_reason_assignment_count": len(
            exclusion_details
        ),
    })
    payload["summary"] = payload_summary
    payload["claim_exclusion_summary"] = claim_exclusion_summary
    # The canonical report JSON is initially created with the other base
    # artifacts. Rewrite it before the manifest so the embedded breakdown and
    # its checksum are part of the canonical report contract.
    artifacts["scientific_report_json"] = write_json(
        report_root / "scientific_report.json", payload,
    )
    claim_exclusion_markdown = [
        "# Claim exclusion summary",
        "",
        (
            "Performance claim-eligible: "
            f"{len(performance)}; excluded rows: "
            f"{len(performance_excluded_rows)}."
        ),
        (
            "Energy claim-eligible: "
            f"{len(energy)}; excluded rows: "
            f"{len(energy_excluded_rows)}."
        ),
        (
            "Detailed and grouped counts represent one reason assignment per "
            "excluded claim-kind/source row. A source row with multiple "
            "explicit reasons therefore has multiple detail rows."
        ),
        "",
        "## Performance exclusions by primary reason",
        "",
        *(
            [
                f"- `{reason}`: {count}"
                for reason, count in sorted(
                    performance_exclusions.items()
                )
            ]
            or ["- none"]
        ),
        "",
        "## Energy exclusions by primary reason",
        "",
        *(
            [
                f"- `{reason}`: {count}"
                for reason, count in sorted(energy_exclusions.items())
            ]
            or ["- none"]
        ),
        "",
        "## Grouped exclusions: model / backend / setup / reason",
        "",
        _md_table(
            grouped_exclusion_counts,
            [
                ("claim_kind", "Claim kind"),
                ("model_id", "Model"),
                ("backend", "Backend"),
                ("setup_id", "Setup"),
                ("reason", "Reason"),
                ("count", "Count"),
            ],
        ),
        "## Per-dimension counts",
        "",
        _md_table(
            dimension_exclusion_counts,
            [
                ("claim_kind", "Claim kind"),
                ("dimension", "Dimension"),
                ("value", "Value"),
                ("count", "Count"),
            ],
        ),
        "## Stable exclusion detail rows",
        "",
        _md_table(
            exclusion_details,
            [
                ("claim_kind", "Claim kind"),
                ("model_id", "Model"),
                ("backend", "Backend"),
                ("setup_id", "Setup"),
                ("reason", "Reason"),
                ("task", "Task"),
                ("case_id", "Case"),
                ("variant", "Variant"),
                ("row_status", "Row status"),
                ("eligibility_status", "Eligibility"),
            ],
        ),
    ]
    artifacts["claim_exclusion_summary_md"] = write_text(
        report_root / "claim_exclusion_summary.md",
        "\n".join(claim_exclusion_markdown) + "\n",
    )
    # Keep the long-standing ``performance_results`` contract claim-only for
    # existing analysis-pack consumers.  The complete measured surface is a
    # new, explicitly named observations table.
    artifacts["performance_results_csv"] = write_csv(
        report_root / "performance_results.csv", performance,
    )
    artifacts["performance_results_json"] = write_json(
        report_root / "performance_results.json", performance,
    )
    artifacts["performance_observations_csv"] = write_csv(
        report_root / "performance_observations.csv", performance_observations,
    )
    artifacts["performance_observations_json"] = write_json(
        report_root / "performance_observations.json", performance_observations,
    )
    artifacts["performance_cohorts_csv"] = write_csv(
        report_root / "performance_cohorts.csv", performance_cohorts,
    )
    artifacts["performance_cohorts_json"] = write_json(
        report_root / "performance_cohorts.json", performance_cohorts,
    )
    artifacts["performance_cohort_summary_json"] = write_json(
        report_root / "performance_cohort_summary.json",
        performance_cohort_summary,
    )
    artifacts["energy_results_csv"] = write_csv(report_root / "energy_results.csv", energy)
    artifacts["screening_performance_observations_csv"] = write_csv(report_root / "screening_performance_observations.csv", screening_performance)
    artifacts["screening_performance_observations_json"] = write_json(report_root / "screening_performance_observations.json", screening_performance)
    artifacts["native_comparison_eligible_observations_csv"] = write_csv(
        report_root / "native_comparison_eligible_observations.csv",
        native_comparison_eligible_observations,
    )
    artifacts["native_comparison_eligible_observations_json"] = write_json(
        report_root / "native_comparison_eligible_observations.json",
        native_comparison_eligible_observations,
    )
    artifacts["screening_energy_observations_csv"] = write_csv(report_root / "screening_energy_observations.csv", screening_energy)
    artifacts["screening_energy_observations_json"] = write_json(report_root / "screening_energy_observations.json", screening_energy)
    artifacts["ranking_method_comparison_csv"] = write_csv(
        report_root / "ranking_method_comparison.csv",
        ranking,
    )
    artifacts["ranking_method_macro_csv"] = write_csv(report_root / "ranking_method_macro.csv", ranking_macro)
    artifacts["ranking_method_cohort_sensitivity_csv"] = write_csv(
        report_root / "ranking_method_cohort_sensitivity.csv",
        ranking_cohort_sensitivity,
    )
    artifacts["ranking_method_cohort_sensitivity_json"] = write_json(
        report_root / "ranking_method_cohort_sensitivity.json",
        ranking_cohort_sensitivity,
    )
    cohort_md = [
        "# Ranking method cohort sensitivity",
        "",
        (
            "Identical predictor methods are recomputed on technical, "
            "quality-pass, and pass-or-inconclusive cohorts. Subset "
            "correlations are diagnostic; global Hit@1 and Regret@1 remain "
            "unavailable unless the complete declared technical universe is "
            "paired."
        ),
        "",
        _md_table(
            ranking_cohort_sensitivity,
            [
                ("model_id", "Model"),
                ("direction", "Direction"),
                ("method_id", "Method"),
                ("cohort", "Cohort"),
                ("paired_candidate_count", "n"),
                ("candidate_coverage_fraction", "Coverage"),
                ("spearman_rho", "Spearman"),
                ("kendall_tau_b", "Kendall"),
                ("global_hit_at_1", "Global Hit@1"),
                ("global_regret_at_1", "Global Regret@1"),
                ("status", "Status"),
            ],
            max_rows=240,
        ),
        "",
    ]
    artifacts["ranking_method_cohort_sensitivity_md"] = write_text(
        report_root / "ranking_method_cohort_sensitivity.md",
        "\n".join(cohort_md),
    )
    native_ranking = [
        dict(row) for row in ranking
        if canonical_runner(row.get("runner_regime")) == "native_fifo"
    ]
    native_ranking_summary = _native_ranking_audit_summary(
        native_ranking,
        payload.get("ranking_audit_request")
        if isinstance(payload.get("ranking_audit_request"), Mapping)
        else {},
    )
    artifacts["native_ranking_audit_csv"] = write_csv(
        report_root / "native_ranking_audit.csv", native_ranking
    )
    artifacts["native_ranking_audit_json"] = write_json(
        report_root / "native_ranking_audit.json", native_ranking_summary
    )
    native_md = [
        "# Native ranking audit", "",
        f"Status: **{native_ranking_summary['status']}**", "",
        str(native_ranking_summary.get("status_detail") or ""), "",
        "Native makespan results are ranked only inside identical model, direction, setup, precision and endpoint-contract strata.",
        "Rows failing the task-quality gate remain reported but are excluded from performance ranking.", "",
        _md_table(native_ranking, [
            ("model_id", "Model"), ("direction", "Direction"),
            ("setup_id", "Setup"),
            ("comparison_output_endpoint_id", "Endpoint"),
            ("comparison_endpoint_stratum", "Endpoint stratum"),
            ("method_id", "Predictor"), ("valid_candidate_count", "Valid n"),
            ("spearman_rho", "Spearman"), ("kendall_tau_b", "Kendall"),
            ("hit_at_1", "Hit@1"), ("regret_at_1", "Regret@1"),
            ("status", "Status"),
        ]),
    ]
    artifacts["native_ranking_audit_md"] = write_text(
        report_root / "native_ranking_audit.md", "\n".join(native_md) + "\n"
    )
    artifacts["cross_runner_candidate_pairs_csv"] = write_csv(
        report_root / "cross_runner_candidate_pairs.csv", cross_runner_pairs
    )
    artifacts["cross_runner_candidate_pairs_json"] = write_json(
        report_root / "cross_runner_candidate_pairs.json", cross_runner_pairs
    )
    artifacts["cross_runner_ranking_validation_csv"] = write_csv(
        report_root / "cross_runner_ranking_validation.csv", cross_runner_groups
    )
    artifacts["cross_runner_ranking_validation_json"] = write_json(
        report_root / "cross_runner_ranking_validation.json", cross_runner_groups
    )
    artifacts["cross_runner_ranking_macro_json"] = write_json(
        report_root / "cross_runner_ranking_macro.json", dict(cross_runner_macro)
    )
    artifacts["cross_runner_ranking_md"] = write_text(
        report_root / "cross_runner_ranking.md", markdown_for_cross_runner(cross_runner_payload)
    )
    artifacts["native_energy_observations_json"] = write_json(
        report_root / "native_energy_observations.json", native_energy_observations
    )
    artifacts["native_energy_observations_csv"] = write_csv(
        report_root / "native_energy_observations.csv", native_energy_observations
    )
    artifacts["native_energy_pair_comparison_csv"] = write_csv(
        report_root / "native_energy_pair_comparison.csv", native_energy_pairs
    )
    artifacts["native_energy_pair_comparison_json"] = write_json(
        report_root / "native_energy_pair_comparison.json", native_energy_pairs
    )
    artifacts["native_performance_matrix_json"] = write_json(
        report_root / "native_performance_matrix.json", dict(native_performance_matrix)
    )
    artifacts["native_performance_observations_csv"] = write_csv(
        report_root / "native_performance_observations.csv", native_performance_observations
    )
    artifacts["native_performance_observations_json"] = write_json(
        report_root / "native_performance_observations.json", native_performance_observations
    )
    artifacts["native_energy_ab_aggregates_csv"] = write_csv(
        report_root / "native_energy_ab_aggregates.csv", native_energy_ab_aggregates
    )
    artifacts["native_energy_ab_aggregates_json"] = write_json(
        report_root / "native_energy_ab_aggregates.json", native_energy_ab_aggregates
    )
    native_energy_md = [
        "# Native energy pair comparison", "",
        f"Observations: **{len(native_energy_observations)}**", f"Pairs: **{len(native_energy_pairs)}**", "",
    ]
    for pair in native_energy_pairs:
        native_energy_md.append(
            f"- {pair.get('model')} / {pair.get('split_backend')} / {pair.get('case')} vs "
            f"{pair.get('baseline_backend')} on {pair.get('setup_id')}: {pair.get('comparison_status')}"
        )
    artifacts["native_energy_pair_comparison_md"] = write_text(
        report_root / "native_energy_pair_comparison.md", "\n".join(native_energy_md) + "\n"
    )
    native_performance_md = [
        "# Native performance matrix", "",
        f"Status: **{native_performance_matrix.get('status', 'unavailable')}**  ",
        (
            "Rows: present "
            f"**{native_performance_matrix.get('present_expected_row_count', len(native_performance_observations))}**"
            f" / expected **{native_performance_matrix.get('expected_row_count', 0)}**; "
            f"successful **{native_performance_matrix.get('successful_expected_row_count', 0)}**; "
            f"failed **{native_performance_matrix.get('failed_expected_row_count', 0)}**; "
            f"missing **{native_performance_matrix.get('missing_expected_row_count', 0)}**  "
        ),
        "All rows in this matrix are development/screening observations. Final claim eligibility remains controlled by the protocol and task-quality gates.", "",
        _md_table(
            native_performance_observations,
            native_evidence_columns,
            max_rows=240,
        ),
    ]
    artifacts["native_performance_matrix_md"] = write_text(
        report_root / "native_performance_matrix.md", "\n".join(native_performance_md) + "\n"
    )
    native_ab_md = [
        "# Native energy A/B aggregates", "",
        f"Aggregates: **{len(native_energy_ab_aggregates)}**", "",
        _md_table(
            native_energy_ab_aggregates,
            [
                ("model", "Model"), ("backend", "Backend"), ("case", "Case"),
                ("setup_id", "Setup"), ("energy_repeat_valid_n", "valid n"),
                ("energy_repeat_requested_n", "requested n"), ("energy_repeat_status", "Repeat status"),
                ("ab_energy_relative_percent_mean", "Legacy-command [%]"),
                ("ab_energy_relative_percent_ci_low", "CI low"),
                ("ab_energy_relative_percent_ci_high", "CI high"),
                ("energy_ab_status", "A/B status"),
            ],
            max_rows=120,
        ),
    ]
    artifacts["native_energy_ab_aggregates_md"] = write_text(
        report_root / "native_energy_ab_aggregates.md", "\n".join(native_ab_md) + "\n"
    )
    if campaign_readiness:
        artifacts["campaign_readiness_csv"] = write_csv(
            report_root / "campaign_readiness.csv",
            [dict(row) for row in list(campaign_readiness.get("checks") or []) if isinstance(row, Mapping)],
        )

    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    ranking_summary = summary.get("ranking") if isinstance(summary.get("ranking"), Mapping) else {}
    def evidence_text(value: Any) -> str:
        return "unavailable" if value is None else str(value)

    markdown = [
        "# Scientific Evaluation Report",
        "",
        str((payload.get("completion") or {}).get("message") or ""),
        "",
        f"Generated: `{payload.get('created_at')}`",
        f"Profile: `{payload.get('profile_id')}`",
        "",
        "## Decision summary",
        "",
        (
            "- Technical execution status: "
            f"**{payload.get('technical_status', 'unavailable')}**"
        ),
        (
            "- Quality-evaluation technical status: "
            f"**{(payload.get('decision_axes') or {}).get('quality_evaluation_technical_status', 'unavailable')}**"
        ),
        (
            "- Quality-evaluation status: "
            f"**{(payload.get('decision_axes') or {}).get('quality_evaluation_status', 'unavailable')}**"
        ),
        (
            "- Aggregate quality decision: "
            f"**{payload.get('quality_decision', 'not_evaluated')}**"
        ),
        (
            "- Scientific status: "
            f"**{payload.get('scientific_status', 'not_evaluated')}**"
        ),
        (
            "- Development-analysis status: "
            f"**{(payload.get('decision_axes') or {}).get('development_analysis_status', 'unavailable')}**"
        ),
        (
            "- Claim readiness: "
            f"**{(payload.get('decision_axes') or {}).get('claim_readiness_status', 'blocked')}** "
            f"({(payload.get('decision_axes') or {}).get('claim_eligible_row_count', 0)} claim-eligible row(s))"
        ),
        (
            "- Central quality evaluated: "
            f"**{summary.get('central_quality_evaluated_count', 0)} / "
            f"{summary.get('central_quality_request_count', 0)}** "
            f"across **{summary.get('central_quality_setup_count', 0)}** setup(s); "
            f"cancelled **{summary.get('central_quality_cancelled_count', 0)}**, "
            f"technical failures **{summary.get('central_quality_technical_failed_count', 0)}**, "
            f"terminal **{summary.get('central_quality_terminal_count', 0)}**"
        ),
        f"- Rows: **{summary.get('row_count', 0)}**",
        f"- Performance observation rows: **{summary.get('performance_observation_count', 0)}**",
        (
            "- Native energy attempts: "
            f"**{summary.get('native_energy_attempt_count', 0)}** "
            f"(**{summary.get('native_energy_success_count', 0)}** successful, "
            f"**{summary.get('native_energy_failed_count', 0)}** failed)"
        ),
        f"- Ranking eligible: **{summary.get('ranking_eligible_count', 0)}**",
        f"- Performance eligible: **{summary.get('performance_eligible_count', 0)}**",
        f"- Energy eligible: **{summary.get('energy_eligible_count', 0)}**",
        f"- Development/screening performance observations: **{summary.get('screening_performance_observation_count', 0)}**",
        f"- Native setup-specific performance observations: **{summary.get('native_performance_observation_count', 0)}** "
        f"(matrix: **{summary.get('native_performance_matrix_status', 'unavailable')}**)",
        f"- Native completed comparison endpoints: **{summary.get('native_completed_comparison_endpoint_count', 0)}**",
        (
            "- Native host postprocessing: "
            f"required **{summary.get('native_host_postprocess_required_count', 0)}**, "
            f"available **{summary.get('native_host_postprocessing_available_count', 0)}**, "
            "legacy-alias conflicts "
            f"**{summary.get('native_host_postprocessing_legacy_alias_conflict_count', 0)}**"
        ),
        f"- Native runtime evidence: **{native_evidence_projection['runtime_status']}**",
        f"- Native semantic evidence: **{native_evidence_projection['semantic_status']}**",
        f"- Native claim evidence: **{native_evidence_projection['claim_status']}**",
        f"- Native energy evidence: **{native_evidence_projection['energy_status']}**",
        f"- Technical status: **{native_evidence_projection['technical_status']}**",
        "- Claim decisions complete: "
        f"**{evidence_text(native_evidence_projection['claim_decisions_complete'])}**",
        "- Native energy attempts: "
        f"**{evidence_text(native_evidence_projection['energy_measurement_attempted_count'])} started, "
        f"{evidence_text(native_evidence_projection['energy_not_started_preflight_count'])} not started**",
        f"- Accounting: {native_evidence_projection.get('energy_accounting_summary') or 'unavailable for legacy evidence'}",
        "- Energy plan completion: "
        f"**{evidence_text(native_evidence_projection['energy_measurement_success_count'])}/"
        f"{evidence_text(native_evidence_projection['energy_plan_denominator_count'])}**",
        "- Energy matrix coverage: "
        f"**{evidence_text(native_evidence_projection['energy_measurement_success_count'])}/"
        f"{evidence_text(native_evidence_projection['energy_matrix_denominator_count'])}**",
        "- Energy claim eligible: "
        f"**{evidence_text(native_evidence_projection['energy_claim_eligible_count'])}**",
        "- Final all-split energy complete: "
        f"**{evidence_text(native_evidence_projection['final_all_split_energy_complete'])}**",
        f"- Scientific status: **{native_evidence_projection['scientific_status']}**",
        "- Scientific ready: "
        f"**{evidence_text(native_evidence_projection['scientific_ready'])}**",
        f"- Development/screening energy observations: **{summary.get('screening_energy_observation_count', 0)}**",
        f"- Campaign claim scope: **{campaign_claim_scope.get('scope', campaign_readiness.get('claim_scope', 'unknown'))}**",
        f"- Ranking-method comparison: **{ranking_summary.get('status', 'unavailable')}**",
        (
            "- Best method under a common comparison coverage: "
            f"**{ranking_summary.get('best_method_label') or 'not available'}** "
            f"({ranking_summary.get('best_method_basis') or 'insufficient common coverage'})"
        ),
        (
            "- Development diagnostic leader (not claim eligible): "
            f"**{ranking_summary.get('development_diagnostic_leader_method_label') or 'not available'}** "
            f"({ranking_summary.get('development_diagnostic_leader_basis') or 'minimum/coverage/strata prerequisites not met'})"
        ),
        f"- Generic-to-Native ranking transfer: **{cross_runner_payload.get('status', 'unavailable')}** ({len(cross_runner_pairs)} paired candidates)",
        "",
        "## Claim exclusion breakdown",
        "",
        (
            "- Performance excluded rows: "
            f"**{len(performance_excluded_rows)}**"
        ),
        (
            "- Energy excluded rows: "
            f"**{len(energy_excluded_rows)}**"
        ),
        (
            "- Exclusion reason assignments: "
            f"**{len(exclusion_details)}**"
        ),
        "",
        "### Grouped by model / backend / setup / reason",
        "",
        _md_table(
            grouped_exclusion_counts,
            [
                ("claim_kind", "Claim kind"),
                ("model_id", "Model"),
                ("backend", "Backend"),
                ("setup_id", "Setup"),
                ("reason", "Reason"),
                ("count", "Count"),
            ],
        ),
        "### Stable detail rows",
        "",
        _md_table(
            exclusion_details,
            [
                ("claim_kind", "Claim kind"),
                ("model_id", "Model"),
                ("backend", "Backend"),
                ("setup_id", "Setup"),
                ("reason", "Reason"),
                ("case_id", "Case"),
            ],
        ),
        "## Final-campaign readiness",
        "",
        f"- Claim scope: **{campaign_readiness.get('claim_scope', campaign_claim_scope.get('scope', 'unknown'))}**",
        f"- Status: **{campaign_readiness.get('status', 'not embedded')}**",
        f"- Required failures: **{campaign_readiness.get('required_failure_count', 'n/a')}**",
        "",
        _md_table(
            list(campaign_readiness.get("checks") or []),
            [
                ("id", "Check"),
                ("status", "Status"),
                ("severity", "Severity"),
                ("detail", "Detail"),
            ],
            max_rows=80,
        ) if campaign_readiness else "No campaign preflight was embedded in this source.",
        "",
        "## Task-quality gates",
        "",
        _md_table(
            task_quality_rows,
            [
                ("model_id", "Model"),
                ("backend", "Backend"),
                ("setup_id", "Setup"),
                ("case_id", "Case"),
                ("variant", "Variant"),
                ("technical_status", "Technical"),
                ("task_quality_tier", "Tier"),
                ("task_quality_metric", "Metric"),
                ("task_quality_delta", "Delta"),
                ("task_quality_bound_value", "Lower bound"),
                ("task_quality_bound_evidence", "Bound evidence"),
                ("task_quality_uncertainty_label", "Uncertainty"),
                ("task_quality_margin", "Margin"),
                ("task_quality_status", "Status"),
                ("task_quality_gate_reason", "Gate reason"),
            ],
            max_rows=80,
        ),
        "## Development/screening performance observations",
        "",
        "These rows are retained for debugging and method development but are explicitly not claim eligible.",
        "",
        _md_table(
            screening_performance,
            [
                ("model_id", "Model"),
                ("backend", "Backend"),
                ("case_id", "Boundary"),
                ("latency_ms", "Latency ms"),
                ("throughput_fps", "FPS"),
                ("task_quality_status", "Quality"),
                ("eligibility_status", "Eligibility"),
                ("exclusion_reason", "Reason"),
            ],
            max_rows=100,
        ),
        "## Native setup-specific performance matrix",
        "",
        "Eligible Native comparison rows use the median and 95-percent confidence interval of independently initialized repetitions, never as a best-of value. Rows without verified independence remain visible and explicitly ineligible. FPS remains independent from any theoretical stage-cycle rate. These observations still require final-protocol admission before thesis claims.",
        "",
        _md_table(
            native_performance_observations,
            native_evidence_columns,
            max_rows=240,
        ),
        "## Development/screening energy observations",
        "",
        "This table includes every Native energy attempt. Failed attempts remain visible, carry no invented numeric result and are always ineligible.",
        "",
        _md_table(
            screening_energy,
            [
                ("model_id", "Model"),
                ("backend", "Backend"),
                ("case_id", "Boundary"),
                ("row_status", "Status"),
                ("average_power_w", "Power W"),
                ("energy_per_work_j", "Energy/work J"),
                ("energy_scope", "Scope"),
                ("energy_window", "Window"),
                ("measurement_failure_reason", "Failure"),
                ("exclusion_reason", "Reason"),
            ],
            max_rows=100,
        ),
        "## Ranking-method comparison (macro)",
        "",
        _md_table(
            ranking_macro,
            [
                ("method_label", "Method"),
                ("group_count", "Groups"),
                ("validated_holdout_group_count", "Validated hold-outs"),
                ("macro_mae_ms", "MAE ms"),
                ("macro_mape_percent", "MAPE %"),
                ("macro_spearman_rho", "Spearman"),
                ("macro_kendall_tau_b", "Kendall"),
                ("diagnostic_correlation_group_count", "Diagnostic groups"),
                ("macro_diagnostic_spearman_rho", "Diagnostic Spearman"),
                ("macro_diagnostic_kendall_tau_b", "Diagnostic Kendall"),
                ("macro_hit_at_5", "Hit@5"),
                ("macro_elite_recall_at_5_q3", "Elite R@5"),
                ("macro_regret_at_5", "Regret@5"),
                ("status", "Status"),
            ],
            max_rows=20,
        ),
        "## Ranking-method comparison (per model/direction/runner)",
        "",
        _md_table(
            ranking,
            [
                ("model_id", "Model"),
                ("evaluation_role", "Role"),
                ("direction", "Direction"),
                ("runner_regime", "Runner"),
                ("method_label", "Method"),
                ("paired_candidate_count", "n"),
                ("candidate_universe_complete", "Universe"),
                ("candidate_universe_declared_complete", "Declared"),
                ("candidate_measurement_coverage_fraction", "Coverage"),
                ("predictions_frozen", "Frozen"),
                ("mae_ms", "MAE ms"),
                ("mape_percent", "MAPE %"),
                ("spearman_rho", "Spearman"),
                ("kendall_tau_b", "Kendall"),
                ("diagnostic_paired_candidate_count", "Diagnostic n"),
                ("diagnostic_spearman_rho", "Diagnostic Spearman"),
                ("diagnostic_kendall_tau_b", "Diagnostic Kendall"),
                ("hit_at_5", "Hit@5"),
                ("elite_recall_at_5_q3", "Elite R@5"),
                ("regret_at_5", "Regret@5"),
                ("status", "Status"),
            ],
            max_rows=160,
        ),
        "## Generic-to-Native ranking transfer",
        "",
        "The same semantically valid candidate boundaries are compared under Generic and Native execution. Absolute Native throughput and energy remain Native-runner evidence; this section tests whether Generic ordering is a useful shortlist surrogate.",
        "",
        _md_table(
            cross_runner_groups,
            [
                ("model_id", "Model"),
                ("direction", "Direction"),
                ("contract_class", "Contract"),
                ("technical_candidate_count", "Technical n"),
                ("quality_candidate_count", "Quality n"),
                ("claim_candidate_count", "Claim n"),
                ("eligible_candidate_count", "n"),
                ("technical_spearman_rho", "Technical Spearman"),
                ("technical_pairwise_concordance", "Technical Concordance"),
                ("technical_native_best_hit_at_1", "Technical Hit@1"),
                ("technical_native_regret_at_1", "Technical Regret@1"),
                ("quality_spearman_rho", "Quality Spearman"),
                ("quality_pairwise_concordance", "Quality Concordance"),
                ("quality_native_best_hit_at_1", "Quality Hit@1"),
                ("quality_native_regret_at_1", "Quality Regret@1"),
                ("spearman_rho", "Spearman"),
                ("kendall_tau_b", "Kendall"),
                ("pairwise_concordance", "Concordance"),
                ("native_best_hit_at_1", "Native Hit@1"),
                ("native_regret_at_1", "Native Regret@1"),
                ("ratio_median", "Generic/Native ratio"),
                ("status", "Status"),
            ],
            max_rows=80,
        ),
        "## Open items",
        "",
    ]
    for item in list(payload.get("open_items") or []):
        markdown.append(f"- **{item.get('severity', 'open')} — {item.get('id')}**: {item.get('detail')}")
    markdown.append("")
    from ..measurement_configuration import measurement_configuration_lines
    if isinstance(payload.get("measurement_configuration"), Mapping):
        markdown.extend(["", "## Resolved measurement configuration", ""])
        markdown.extend("- " + line for line in measurement_configuration_lines(payload["measurement_configuration"]))
    artifacts["scientific_report_md"] = write_text(report_root / "scientific_report.md", "\n".join(markdown))

    quality_rows = [
        row for row in task_quality_rows if row.get("task_quality_metric")
    ]
    artifacts["task_quality_tex"] = _write_tex_table(
        tables / "task_quality_gates.tex",
        quality_rows,
        [
            ("model_id", "Model", "text"),
            ("backend", "Backend", "text"),
            ("setup_id", "Setup", "text"),
            ("case_id", "Boundary", "text"),
            ("task_quality_metric", "Metric", "text"),
            ("task_quality_delta", "$\\Delta$", "number"),
            ("task_quality_bound_value", "Lower bound", "number"),
            ("task_quality_bound_evidence", "Bound evidence", "text"),
            ("task_quality_uncertainty_label", "Uncertainty", "text"),
            ("task_quality_margin", "Margin", "number"),
            ("task_quality_status", "Decision", "text"),
        ],
        "Pre-registered task-quality non-inferiority gates. Only final-tier pass rows are optimisation-eligible.",
        "tab:task-quality-gates",
    )
    artifacts["ranking_method_comparison_tex"] = _write_tex_table(
        tables / "ranking_method_comparison.tex",
        ranking_macro,
        [
            ("method_label", "Method", "text"),
            ("validated_holdout_group_count", "$n_{HO}$", "number"),
            ("macro_mae_ms", "MAE [ms]", "number"),
            ("macro_mape_percent", "MAPE [percent]", "number"),
            ("macro_spearman_rho", "$\\rho$", "number"),
            ("macro_kendall_tau_b", "$\\tau_b$", "number"),
            ("macro_hit_at_5", "Hit@5", "number"),
            ("macro_elite_recall_at_5_q3", "Elite R@5", "number"),
            ("macro_regret_at_5", "Regret@5", "number"),
            ("status", "Status", "text"),
        ],
        "Comparison of the frozen Cut Bytes workflow selector with four pre-registered alternative ranking baselines. Top-k and regret are audit-relative for deterministic-audit universes. Absolute error is reported only for cycle-time predictors.",
        "tab:ranking-method-comparison",
    )
    artifacts["ranking_method_cohort_sensitivity_tex"] = _write_tex_table(
        tables / "ranking_method_cohort_sensitivity.tex",
        ranking_cohort_sensitivity,
        [
            ("model_id", "Model", "text"),
            ("direction", "Direction", "text"),
            ("method_label", "Method", "text"),
            ("cohort", "Cohort", "text"),
            ("paired_candidate_count", "$n$", "number"),
            ("candidate_coverage_fraction", "Coverage", "number"),
            ("spearman_rho", "$\\rho$", "number"),
            ("kendall_tau_b", "$\\tau_b$", "number"),
            ("global_hit_at_1", "Global Hit@1", "text"),
            ("global_regret_at_1", "Global Regret@1", "number"),
        ],
        (
            "Ranking-method sensitivity on identical technical, "
            "quality-pass, and pass-or-inconclusive cohorts. Subset "
            "correlations are diagnostic; global top-1 metrics require the "
            "complete declared technical universe."
        ),
        "tab:ranking-method-cohort-sensitivity",
    )
    artifacts["cross_runner_ranking_validation_tex"] = _write_tex_table(
        tables / "cross_runner_ranking_validation.tex",
        cross_runner_groups,
        [
            ("model_id", "Model", "text"),
            ("direction", "Direction", "text"),
            ("technical_candidate_count", "$n_T$", "number"),
            ("technical_pairwise_concordance", "$C_T$", "number"),
            ("technical_native_best_hit_at_1", "Hit@1$_T$", "text"),
            ("technical_native_regret_at_1", "Regret@1$_T$", "number"),
            ("quality_candidate_count", "$n_Q$", "number"),
            ("quality_pairwise_concordance", "$C_Q$", "number"),
            ("quality_native_best_hit_at_1", "Hit@1$_Q$", "text"),
            ("quality_native_regret_at_1", "Regret@1$_Q$", "number"),
            ("claim_candidate_count", "$n_C$", "number"),
            ("pairwise_concordance", "$C_C$", "number"),
            ("native_best_hit_at_1", "Hit@1$_C$", "text"),
            ("native_regret_at_1", "Regret@1$_C$", "number"),
            ("status", "Status", "text"),
        ],
        "Generic-to-Native ranking-transfer validation on identical candidate boundaries, separated into technical, quality-screened, and claim cohorts.",
        "tab:cross-runner-ranking-transfer",
    )
    if campaign_readiness:
        artifacts["campaign_readiness_tex"] = _write_tex_table(
            tables / "campaign_readiness.tex",
            [dict(row) for row in list(campaign_readiness.get("checks") or []) if isinstance(row, Mapping)],
            [
                ("id", "Requirement", "text"),
                ("status", "Status", "text"),
                ("severity", "Severity", "text"),
            ],
            "Software-side preflight for the frozen final measurement campaign. Hardware measurements remain separate evidence.",
            "tab:campaign-readiness",
        )
    artifacts["performance_results_tex"] = _write_tex_table(
        tables / "performance_results.tex",
        performance,
        [
            ("model_id", "Model", "text"),
            ("backend", "Backend", "text"),
            ("case_id", "Boundary", "text"),
            ("runner_regime", "Runner", "text"),
            ("latency_ms", "Latency [ms]", "number"),
            ("throughput_fps", "FPS", "number"),
            ("task_quality_status", "Quality", "text"),
        ],
        "Strict claim-eligible performance results. Technically measured screening and development rows remain in the separate performance-observations table.",
        "tab:performance-results",
    )
    artifacts["performance_observations_tex"] = _write_tex_table(
        tables / "performance_observations.tex",
        performance_observations,
        [
            ("model_id", "Model", "text"),
            ("backend", "Backend", "text"),
            ("case_id", "Boundary", "text"),
            ("runner_regime", "Runner", "text"),
            ("latency_ms", "Latency [ms]", "number"),
            ("throughput_fps", "FPS", "number"),
            ("task_quality_status", "Quality", "text"),
        ],
        "Complete technically measured performance surface. Rows in this table are observations and are not automatically thesis-claim eligible.",
        "tab:performance-observations",
    )
    artifacts["screening_performance_observations_tex"] = _write_tex_table(
        tables / "screening_performance_observations.tex",
        screening_performance,
        [
            ("model_id", "Model", "text"),
            ("backend", "Backend", "text"),
            ("case_id", "Boundary", "text"),
            ("runner_regime", "Runner", "text"),
            ("latency_ms", "Latency [ms]", "number"),
            ("throughput_fps", "FPS", "number"),
            ("task_quality_status", "Quality", "text"),
            ("eligibility_status", "Eligibility", "text"),
        ],
        "Development and screening performance observations. These rows are not claim eligible and must not be used in the final Pareto front.",
        "tab:screening-performance-observations",
    )
    artifacts["native_performance_observations_tex"] = _write_tex_table(
        tables / "native_performance_observations.tex",
        native_performance_observations,
        native_evidence_tex_columns,
        "Complete setup-specific Native performance matrix retained as development/screening evidence. Eligible comparison rows use the median and 95-percent confidence interval of independently initialized repetitions, never a best-of selection; unverified rows stay explicitly marked. Measured FPS and theoretical cycle-model rate are distinct columns. Final claims additionally require protocol admission.",
        "tab:native-performance-observations",
    )
    artifacts["screening_energy_observations_tex"] = _write_tex_table(
        tables / "screening_energy_observations.tex",
        screening_energy,
        [
            ("model_id", "Model", "text"),
            ("backend", "Backend", "text"),
            ("case_id", "Boundary", "text"),
            ("row_status", "Status", "text"),
            ("average_power_w", "Power [W]", "number"),
            ("energy_per_work_j", "Energy/work [J]", "number"),
            ("energy_scope", "Scope", "text"),
            ("energy_window", "Window", "text"),
            ("eligibility_status", "Eligibility", "text"),
            ("measurement_failure_reason", "Failure", "text"),
        ],
        "Development and screening energy attempts. Successful observations and failed measurements remain visible; failed attempts carry no invented numeric result and are never claim eligible.",
        "tab:screening-energy-observations",
    )
    artifacts["energy_results_tex"] = _write_tex_table(
        tables / "energy_results.tex",
        energy,
        [
            ("model_id", "Model", "text"),
            ("backend", "Backend", "text"),
            ("case_id", "Boundary", "text"),
            ("average_power_w", "Power [W]", "number"),
            ("energy_per_work_j", "Energy/work [J]", "number"),
            ("energy_per_work_ci_low_j", "CI low [J]", "number"),
            ("energy_per_work_ci_high_j", "CI high [J]", "number"),
            ("energy_repeat_n", "$n$", "number"),
            ("energy_scope", "Scope", "text"),
            ("energy_window", "Window", "text"),
        ],
        "Energy results under compatible declared scopes and workload windows.",
        "tab:energy-results",
    )

    artifacts["native_energy_observations_tex"] = _write_tex_table(
        tables / "native_energy_observations.tex",
        native_energy_observations,
        [
            ("model", "Model", "text"),
            ("backend", "Backend", "text"),
            ("case", "Boundary", "text"),
            ("setup_id", "Setup", "text"),
            ("measurement_status", "Status", "text"),
            ("energy_total_j", "Raw energy [J]", "number"),
            ("energy_per_work_j", "Energy/work [J]", "number"),
            ("energy_scope", "Scope", "text"),
            ("energy_window_effective", "Window", "text"),
            ("claim_eligible", "Claim", "text"),
            ("measurement_failure_reason", "Failure", "text"),
        ],
        "Native energy attempts with raw input energy as the primary measurement and separately gated per-work normalisation. Failed attempts remain explicit and contain no fabricated result.",
        "tab:native-energy-observations",
    )

    artifacts["native_energy_pair_comparison_tex"] = _write_tex_table(
        tables / "native_energy_pair_comparison.tex",
        native_energy_pairs,
        [
            ("model", "Model", "text"),
            ("split_backend", "Split", "text"),
            ("case", "Boundary", "text"),
            ("setup_id", "Setup", "text"),
            ("baseline_kind", "Baseline", "text"),
            ("split_energy_per_work_j", "Split J/work", "number"),
            ("baseline_energy_per_work_j", "Baseline J/work", "number"),
            ("energy_ratio", "Ratio", "number"),
            ("comparison_status", "Status", "text"),
        ],
        "Setup-local Native Split versus Native Full energy comparison.",
        "tab:native-energy-pair-comparison",
    )
    artifacts["native_energy_ab_aggregates_tex"] = _write_tex_table(
        tables / "native_energy_ab_aggregates.tex",
        native_energy_ab_aggregates,
        [
            ("model", "Model", "text"),
            ("backend", "Backend", "text"),
            ("case", "Boundary", "text"),
            ("setup_id", "Setup", "text"),
            ("energy_repeat_valid_n", "$n_{valid}$", "number"),
            ("energy_repeat_requested_n", "$n_{requested}$", "number"),
            ("ab_energy_relative_percent_mean", "$\\Delta E$ [percent]", "number"),
            ("ab_energy_relative_percent_ci_low", "CI low", "number"),
            ("ab_energy_relative_percent_ci_high", "CI high", "number"),
            ("energy_ab_status", "Status", "text"),
        ],
        "Same-capture Chapter-4 versus candidate-window A/B aggregates. Incomplete repeat sets are marked explicitly and are not silently promoted to complete evidence.",
        "tab:native-energy-ab-aggregates",
    )

    for index, path in enumerate(_make_figures(rows, ranking, ranking_macro, figures)):
        artifacts[f"scientific_figure_{index:02d}"] = path

    manifest = {
        "schema": "onnx-splitpoint/scientific-report-manifest",
        "schema_version": 2,
        "created_at": now_iso(),
        "artifacts": [
            {
                "path": relpath(path, report_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
            for path in artifacts.values()
            if path.is_file()
        ],
    }
    artifacts["scientific_report_manifest"] = write_json(report_root / "report_manifest.json", manifest)

    if compatibility_root is not None:
        compatibility_root.mkdir(parents=True, exist_ok=True)
        dashboard = {
            "schema": REPORT_SCHEMA,
            "schema_version": REPORT_SCHEMA_VERSION,
            "created_at": payload.get("created_at"),
            "profile_id": payload.get("profile_id"),
            "technical_status": payload.get("technical_status"),
            "technical_execution_status": payload.get(
                "technical_execution_status"
            ),
            "quality_decision": payload.get("quality_decision"),
            "aggregate_quality_decision": payload.get(
                "aggregate_quality_decision"
            ),
            "scientific_status": payload.get("scientific_status"),
            "scientific_pass": payload.get("scientific_pass"),
            "decision_axes": payload.get("decision_axes"),
            "completion": payload.get("completion"),
            "summary": payload.get("summary"),
            "central_quality_reporting": payload.get(
                "central_quality_reporting"
            ),
            "central_quality_results": payload.get(
                "central_quality_results"
            ),
            "native_evidence_status": payload.get("native_evidence_status"),
            "claim_exclusion_summary": payload.get(
                "claim_exclusion_summary"
            ),
            "open_items": payload.get("open_items"),
            "canonical_report": relpath(artifacts["scientific_report_json"], compatibility_root),
        }
        artifacts["result_dashboard_json"] = write_json(compatibility_root / "result_dashboard.json", dashboard)
        artifacts["result_dashboard_md"] = write_text(
            compatibility_root / "result_dashboard.md",
            (report_root / "scientific_report.md").read_text(encoding="utf-8"),
        )
        artifacts["summary_csv"] = write_csv(compatibility_root / "summary.csv", rows)
        artifacts["model_summary_csv"] = write_csv(
            compatibility_root / "model_summary.csv",
            list(payload.get("model_facts") or []),
        )
    return artifacts


def _scientific_decision_axes(
    payload: Mapping[str, Any],
    *,
    technical_execution_status: str,
    central_quality_reporting: Mapping[str, Any],
) -> dict[str, Any]:
    """Project independent execution, analysis and claim-readiness axes.

    A successful quality aggregate is not by itself a scientific pass.  The
    report must also contain at least one row that is eligible for a declared
    performance or energy claim.  Development diagnostics remain visible on a
    separate axis and can never satisfy that prerequisite.
    """

    technical = str(technical_execution_status or "pending").strip().lower()
    if technical in {"completed", "success"}:
        technical = "ok"
    quality_technical = str(
        central_quality_reporting.get("technical_status") or "unavailable"
    ).strip().lower()
    aggregate_quality = str(
        central_quality_reporting.get("quality_decision") or "not_evaluated"
    ).strip().lower()
    if quality_technical in {"ok", "not_applicable"}:
        quality_evaluation_status = (
            "evaluated"
            if aggregate_quality in {"pass", "fail", "inconclusive"}
            else "not_evaluated"
        )
    else:
        quality_evaluation_status = quality_technical

    rows = [
        row for row in list(payload.get("rows") or [])
        if isinstance(row, Mapping)
    ]
    performance_claim_count = sum(
        _performance_claim_eligible(row) for row in rows
    )
    energy_claim_count = sum(_energy_claim_eligible(row) for row in rows)
    claim_row_count = sum(
        _performance_claim_eligible(row) or _energy_claim_eligible(row)
        for row in rows
    )

    ranking_summary = (
        (payload.get("summary") or {}).get("ranking")
        if isinstance(payload.get("summary"), Mapping) else {}
    )
    if not isinstance(ranking_summary, Mapping):
        ranking_summary = {}
    ranking_status = str(ranking_summary.get("status") or "unavailable")
    cross_runner_status = str(
        payload.get("cross_runner_ranking_status") or "unavailable"
    )
    diagnostic_available = bool(
        ranking_summary.get("development_diagnostic_leader_method_id")
        or any(
            _f(row.get("diagnostic_spearman_rho")) is not None
            for row in list(payload.get("ranking_method_comparison") or [])
            if isinstance(row, Mapping)
        )
        or cross_runner_status in {
            "ok", "quality_screening_only", "technical_diagnostic_only",
        }
    )
    if ranking_status == "holdout_available":
        development_analysis_status = "confirmatory_and_development_available"
    elif diagnostic_available:
        development_analysis_status = "development_diagnostic_available"
    elif payload.get("ranking_method_comparison"):
        development_analysis_status = "incomplete"
    else:
        development_analysis_status = "unavailable"

    readiness_reasons: list[str] = []
    if technical != "ok":
        readiness_reasons.append("technical_execution_not_ok")
    if quality_technical not in {"ok", "not_applicable"}:
        readiness_reasons.append("quality_evaluation_not_technically_complete")
    if aggregate_quality != "pass":
        readiness_reasons.append(
            f"aggregate_quality_{aggregate_quality or 'not_evaluated'}"
        )
    if claim_row_count == 0:
        readiness_reasons.append("no_claim_eligible_rows")
    claim_readiness_status = "ready" if not readiness_reasons else "blocked"

    if quality_technical not in {"ok", "not_applicable"}:
        scientific_status = "not_evaluated"
    elif aggregate_quality in {"fail", "inconclusive"}:
        scientific_status = aggregate_quality
    elif technical != "ok":
        scientific_status = "not_claim_ready"
    elif claim_row_count == 0:
        scientific_status = "not_claim_ready"
    elif aggregate_quality == "pass":
        scientific_status = "pass"
    else:
        scientific_status = "not_evaluated"

    return {
        # Compatibility key retained for existing dashboards.
        "technical_status": technical,
        "technical_execution_status": technical,
        "quality_evaluation_technical_status": quality_technical,
        "quality_evaluation_status": quality_evaluation_status,
        "quality_decision": aggregate_quality,
        "aggregate_quality_decision": aggregate_quality,
        "aggregate_quality_status": aggregate_quality,
        "development_analysis_status": development_analysis_status,
        "claim_readiness_status": claim_readiness_status,
        "claim_readiness_reasons": readiness_reasons,
        "claim_eligible_row_count": claim_row_count,
        "performance_claim_eligible_row_count": performance_claim_count,
        "energy_claim_eligible_row_count": energy_claim_count,
        "scientific_status": scientific_status,
        "scientific_pass": bool(
            claim_readiness_status == "ready"
            and scientific_status == "pass"
        ),
        "status_semantics": dict(
            central_quality_reporting.get("status_semantics") or {}
        ),
    }


def _full_quality_only_stale_hardware_smoke_projection(
    *,
    run_status: Mapping[str, Any] | None,
    run_manifest: Mapping[str, Any] | None,
    central_quality_reporting: Mapping[str, Any],
) -> dict[str, Any]:
    """Recognize one historical, non-applicable hardware-smoke downgrade.

    Offline replay cannot rerun ``_derive_final_status``.  It must nevertheless
    avoid copying a stale ``partial`` axis when an exact Full-quality-only
    acceptance contract proves all requested hardware quality endpoints
    completed and the *only* blocker is a completed, error-free hardware smoke
    that expected performance rows which are deliberately out of scope.
    """

    status_source = (
        dict(run_status or {}) if isinstance(run_status, Mapping) else {}
    )
    manifest = (
        dict(run_manifest or {}) if isinstance(run_manifest, Mapping) else {}
    )
    contract = central_quality_reporting.get(
        "quality_acceptance_identity_contract"
    )
    blockers = [
        dict(row)
        for row in list(status_source.get("blocking_reasons") or [])
        if isinstance(row, Mapping)
    ]
    base = {
        "schema": (
            "onnx-splitpoint/"
            "full-quality-only-technical-status-projection"
        ),
        "schema_version": 1,
        "applied": False,
        "source_technical_status": str(
            status_source.get("technical_status")
            or status_source.get("status")
            or manifest.get("technical_status")
            or manifest.get("status")
            or "pending"
        ).strip().lower(),
        "projected_technical_status": "",
        "reason": "guards_not_satisfied",
        "stale_hardware_smoke_models": [],
    }
    if not isinstance(contract, Mapping):
        return base
    if (
        base["source_technical_status"] != "partial"
        or central_quality_reporting.get("technical_status") != "ok"
        or central_quality_reporting.get("results_complete") is not True
        or central_quality_reporting.get(
            "aggregate_identity_contract_complete"
        ) is not True
        or str(contract.get("schema") or "")
        != FULL_ONLY_QUALITY_ACCEPTANCE_IDENTITY_SCHEMA
        or str(contract.get("execution_scope") or "") != "full_only"
        or not blockers
        or int(status_source.get("blocking_reason_count") or len(blockers))
        != len(blockers)
    ):
        return base
    if any(
        str(row.get("kind") or "") != "stage_status"
        or str(row.get("stage") or "") != "hardware_smoke"
        or str(row.get("status") or "").strip().lower()
        not in {"partial", "warn"}
        for row in blockers
    ):
        return base

    expected_models = {
        str(value).strip()
        for value in list(contract.get("model_ids") or [])
        if str(value).strip()
    }
    blocked_models = {
        str(row.get("model_id") or "").strip() for row in blockers
    }
    if not blocked_models or not blocked_models.issubset(expected_models):
        return base

    manifest_models = (
        manifest.get("models")
        if isinstance(manifest.get("models"), Mapping) else {}
    )
    verified_blocked_models: set[str] = set()
    for model_id, model in manifest_models.items():
        if not isinstance(model, Mapping):
            continue
        stages = (
            model.get("stages")
            if isinstance(model.get("stages"), Mapping) else {}
        )
        if any(
            isinstance(stage, Mapping)
            and str(stage.get("status") or "").strip().lower() == "failed"
            for stage in stages.values()
        ):
            return base
        if str(model_id) not in blocked_models:
            continue
        hardware = stages.get("hardware_smoke")
        if not isinstance(hardware, Mapping) or not (
            str(hardware.get("status") or "").strip().lower()
            in {"partial", "warn"}
            and str(hardware.get("state") or "").strip().lower()
            == "completed"
            and hardware.get("complete") is True
            and not str(hardware.get("error_class") or "").strip()
            and not str(hardware.get("error_detail") or "").strip()
        ):
            return base
        verified_blocked_models.add(str(model_id))
    if verified_blocked_models != blocked_models:
        return base
    root_stages = (
        manifest.get("root_stages")
        if isinstance(manifest.get("root_stages"), Mapping) else {}
    )
    if any(
        isinstance(stage, Mapping)
        and str(stage.get("status") or "").strip().lower() == "failed"
        for stage in root_stages.values()
    ):
        return base

    model_results = [
        row
        for row in list(central_quality_reporting.get("results") or [])
        if isinstance(row, Mapping)
        and str(row.get("model_id") or "") in blocked_models
        and str(row.get("execution_role") or "") == "full_quality_only"
        and str(row.get("variant") or "").strip().lower() == "full"
        and row.get("performance_claims_emitted") is False
        and str(row.get("technical_status") or "")
        in {"completed", "ok", "success"}
    ]
    expected_per_model = len([
        row for row in list(contract.get("expected_identities") or [])
        if isinstance(row, Mapping)
    ])
    if expected_per_model <= 0 or len(model_results) != (
        expected_per_model * len(blocked_models)
    ):
        return base

    base.update({
        "applied": True,
        "projected_technical_status": "ok",
        "reason": (
            "exact_full_quality_only_central_acceptance;"
            "hardware_smoke_not_applicable_to_performance"
        ),
        "stale_hardware_smoke_models": sorted(blocked_models),
    })
    return base


def build_scientific_reports(
    run_dir: Path,
    *,
    profile_id: str = "",
    tool_version: str = "",
    workflow_version: str = "",
    cleanup_legacy: bool = True,
    output_dir: Optional[Path] = None,
    native_source_dir: Optional[Path] = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    reports_dir = run_dir / "reports"
    external_output = output_dir is not None
    if external_output:
        from ..filesystem_admission import require_output_outside_source

        _source, report_root = require_output_outside_source(
            run_dir,
            Path(output_dir).expanduser(),
            operation="Offline scientific report replay",
        )
        compatibility_root: Optional[Path] = None
    else:
        report_root = reports_dir / "scientific"
        compatibility_root = reports_dir
    profile = _load_yaml(run_dir / "profile.yaml")
    native_source_root = (
        Path(native_source_dir).expanduser().resolve(strict=True)
        if native_source_dir is not None else run_dir
    )
    if not native_source_root.is_dir():
        raise RuntimeError(
            "Native replay source is not a directory: "
            f"{native_source_root}"
        )
    policy = _policy_from_profile(profile)
    # A replay target outside the EvaluationRun must never clean presentation
    # files in the source run.  ``_write_reports`` recreates only its explicit
    # destination, so an external output remains a self-contained projection.
    if cleanup_legacy and not external_output:
        clean_legacy_reports(reports_dir)
    rows, predictions, model_facts = _load_evalrun_rows(run_dir, profile, policy)
    report_input_source = "normalized_benchmark_results"
    if not rows:
        archived_rows = _archived_scientific_performance_rows(run_dir)
        if archived_rows:
            rows = _enrich_archived_setup_identity(run_dir, archived_rows)
            report_input_source = "archived_scientific_row_eligibility"
    from .endpoint_lifecycle import classify_runtime_failure_measurements

    (
        rows,
        runtime_failure_evidence,
        runtime_failure_diagnostics,
    ) = classify_runtime_failure_measurements(run_dir, rows)
    # Native performance rows live outside the Generic normalization tree.
    # Ingest them before the report payload computes ranking metrics, while
    # retaining the existing Generic scientific-row table as its own surface.
    native_performance_matrix = collect_native_performance_matrix(
        native_source_root
    )
    native_matrix_rows = [
        dict(row)
        for row in list(native_performance_matrix.get("observations") or [])
        if isinstance(row, Mapping)
    ]
    ranking_input_rows = _merge_native_ranking_rows(rows, native_matrix_rows)
    payload = _build_report_payload(
        source_kind="evaluation_run",
        source_root=run_dir,
        profile_id=profile_id or str(profile.get("name") or run_dir.name),
        tool_version=tool_version,
        workflow_version=workflow_version,
        profile=profile,
        policy=policy,
        rows=rows,
        predictions=predictions,
        model_facts=model_facts,
        ranking_input_rows=ranking_input_rows,
    )
    from ..measurement_configuration import measurement_configuration
    payload["measurement_configuration"] = measurement_configuration(
        profile, read_json(run_dir / "effective_execution_plan.json", default={}),
    )
    payload["report_input_source"] = report_input_source
    payload["native_source_binding"] = {
        "schema": "onnx-splitpoint/native-replay-source-binding",
        "schema_version": 1,
        "generic_run_dir": str(run_dir),
        "native_run_dir": str(native_source_root),
        "native_source_is_separate": native_source_root != run_dir,
        "native_run_id": native_source_root.name,
        "binding_status": "explicit" if native_source_dir is not None else "co_located",
    }
    payload["runtime_failure_reclassification"] = {
        "schema": "onnx-splitpoint/runtime-failure-reclassification",
        "schema_version": 1,
        "status": "applied" if runtime_failure_evidence else "not_present",
        "reclassified_endpoint_count": len(runtime_failure_evidence),
        "evidence": runtime_failure_evidence,
        "diagnostics": runtime_failure_diagnostics,
    }
    input_summary = dict(payload.get("summary") or {})
    input_summary["report_input_source"] = report_input_source
    input_summary["archived_projection_input_row_count"] = (
        len(rows)
        if report_input_source == "archived_scientific_row_eligibility"
        else 0
    )
    payload["summary"] = input_summary
    from .endpoint_lifecycle import build_evalrun_endpoint_lifecycle

    try:
        endpoint_lifecycle = build_evalrun_endpoint_lifecycle(
            run_dir, list(rows), profile=profile,
        )
    except Exception as exc:
        # Report replay must remain read-only and diagnostic. A malformed
        # historical plan is exposed as an unavailable ledger rather than
        # being replaced by endpoints inferred from result rows.
        endpoint_lifecycle = {
            "schema": "onnx-splitpoint/endpoint-lifecycle-ledger",
            "schema_version": 1,
            "rows": [],
            "summary": {
                "planned": 0,
                "materialized": 0,
                "measurement_payload_present": 0,
                "payload_missing": 0,
                "measured": 0,
                "valid_measurement": 0,
                "invalid_measurement_count": 0,
                "nonmeasured": 0,
                "terminal": 0,
                "nonterminal": 0,
            },
            "adapter": {
                "status": "error",
                "evidence_completeness_status": "error",
                "measurement_completeness_status": "error",
                "evidence_completeness_reasons": [
                    "endpoint_lifecycle_adapter_error"
                ],
                "read_only": True,
                "errors": [f"{type(exc).__name__}: {exc}"],
            },
        }
    payload["endpoint_lifecycle"] = endpoint_lifecycle
    lifecycle_summary = (
        dict(endpoint_lifecycle.get("summary") or {})
        if isinstance(endpoint_lifecycle.get("summary"), Mapping)
        else {}
    )
    lifecycle_adapter = (
        dict(endpoint_lifecycle.get("adapter") or {})
        if isinstance(endpoint_lifecycle.get("adapter"), Mapping)
        else {}
    )
    payload_summary = dict(payload.get("summary") or {})
    payload_summary.update({
        "endpoint_lifecycle_status": str(
            lifecycle_adapter.get("status") or "unavailable"
        ),
        "planned_endpoint_count": int(lifecycle_summary.get("planned") or 0),
        "materialized_endpoint_count": int(
            lifecycle_summary.get("materialized") or 0
        ),
        "measured_endpoint_count": int(
            lifecycle_summary.get("measured") or 0
        ),
        "measurement_payload_present_endpoint_count": int(
            lifecycle_summary.get("measurement_payload_present") or 0
        ),
        "valid_measurement_endpoint_count": int(
            lifecycle_summary.get("valid_measurement") or 0
        ),
        "invalid_measurement_endpoint_count": int(
            lifecycle_summary.get("invalid_measurement_count") or 0
        ),
        "missing_measurement_payload_endpoint_count": int(
            lifecycle_summary.get("missing_measurement_payload_count") or 0
        ),
        "without_valid_measurement_endpoint_count": int(
            lifecycle_summary.get("without_valid_measurement_count") or 0
        ),
        "nonmeasured_endpoint_count": int(
            lifecycle_summary.get("nonmeasured") or 0
        ),
        "terminal_endpoint_count": int(
            lifecycle_summary.get("terminal") or 0
        ),
        "nonterminal_endpoint_count": int(
            lifecycle_summary.get("nonterminal") or 0
        ),
        "endpoint_lifecycle_evidence_completeness_status": str(
            lifecycle_adapter.get("evidence_completeness_status")
            or "unavailable"
        ),
        "endpoint_lifecycle_evidence_completeness_reasons": list(
            lifecycle_adapter.get("evidence_completeness_reasons") or []
        ),
        "endpoint_lifecycle_measurement_completeness_status": str(
            lifecycle_adapter.get("measurement_completeness_status")
            or "unavailable"
        ),
        "runtime_failure_reclassified_endpoint_count": len(
            runtime_failure_evidence
        ),
    })
    payload["summary"] = payload_summary
    central_quality_source = read_json(
        run_dir / "quality_management" / "central_quality_summary.json",
        default={},
    ) or {}
    from .evidence_state_model import summarize_run_scope
    current_scope = summarize_run_scope(run_dir)
    if (isinstance(central_quality_source, Mapping)
            and current_scope["matrix_required"] > 0
            and current_scope["matrix_required"] == central_quality_source.get("matrix_required")):
        # Re-evaluate only applicability from the same sealed model scopes.
        # Historical results, decisions and source files remain untouched.
        central_quality_source = dict(central_quality_source)
        payload["quality_scope_projection"] = {
            "source_quality_missing": central_quality_source.get("quality_missing"),
            "source_evidence_state_summary": central_quality_source.get("evidence_state_summary"),
            "projected_evidence_state_summary": current_scope,
        }
        central_quality_source.update({key: current_scope[key] for key in (
            "matrix_required", "matrix_present", "quality_applicable", "quality_completed",
            "quality_blocked", "quality_not_applicable", "quality_missing",
        )})
        central_quality_source["evidence_state_summary"] = current_scope
    central_quality_reporting = project_central_quality_status(
        central_quality_source
        if isinstance(central_quality_source, Mapping)
        else {},
        dataset_tier=str(policy.dataset_tier or ""),
    )
    payload["central_quality_reporting"] = {
        key: value
        for key, value in central_quality_reporting.items()
        if key != "results"
    }
    payload["central_quality_results"] = list(
        central_quality_reporting.get("results") or []
    )
    payload["quality_reporting_policy"] = dict(policy.reporting_policy or {})
    from .run_discovery import build_measurement_set_contract

    measurement_set = build_measurement_set_contract(run_dir)
    payload["run_id"] = str(run_dir.name)
    payload["measurement_set_sha256"] = str(
        measurement_set.get("measurement_set_sha256") or ""
    )
    payload["measurement_result_count"] = int(
        measurement_set.get("result_count") or 0
    )
    # Cross-runner identity joins need the enriched normalized contracts,
    # including nested quality-request identities. The compact scientific-row
    # projection intentionally omits several of those fields.
    cross_runner = compute_cross_runner_report(
        run_dir,
        list(rows),
        minimum_candidates=3,
        native_run_dir=native_source_root,
        generic_input_source=report_input_source,
    )
    payload["cross_runner_schema"] = cross_runner.get("schema")
    payload["cross_runner_schema_version"] = cross_runner.get("schema_version")
    payload["cross_runner_candidate_pairs"] = list(cross_runner.get("pairs") or [])
    payload["cross_runner_ranking_validation"] = list(cross_runner.get("groups") or [])
    payload["cross_runner_ranking_macro"] = dict(cross_runner.get("macro") or {})
    payload["cross_runner_ranking_status"] = cross_runner.get("status") or "unavailable"
    payload["cross_runner_identity_diagnostics"] = {
        key: cross_runner.get(key)
        for key in (
            "planned_native_intersection_source",
            "planned_native_intersection_explicit",
            "planned_native_intersection_count",
            "planned_native_intersection_paired_count",
            "planned_native_intersection_complete",
            "generic_inside_planned_intersection_count",
            "generic_outside_planned_intersection_count",
            "generic_identity_projection_count",
            "generic_input_source",
            "legacy_endpoint_contract_default_migration_count",
            "legacy_endpoint_contract_default_migrations",
            "native_outside_planned_intersection_count",
            "identity_exclusion_count",
            "identity_exclusion_reason_counts",
            "identity_exclusions",
        )
    }

    payload["native_performance_matrix"] = native_performance_matrix
    summary = dict(payload.get("summary") or {})
    summary["native_performance_observation_count"] = int(native_performance_matrix.get("observation_count") or 0)
    summary["native_performance_expected_row_count"] = int(native_performance_matrix.get("expected_row_count") or 0)
    summary["native_performance_matrix_status"] = str(native_performance_matrix.get("status") or "unavailable")
    summary["native_e2e_scope_counts"] = dict(Counter(
        str(row.get("e2e_scope") or "unavailable")
        for row in native_matrix_rows
    ))
    summary["native_completed_endpoint_attested_count"] = sum(
        1 for row in native_matrix_rows
        if row.get("completed_task_endpoint_attested") is True
    )
    summary["native_completed_comparison_endpoint_count"] = sum(
        1 for row in native_matrix_rows
        if str(
            row.get("completed_task_comparison_output_endpoint_id") or ""
        ).strip()
    )
    summary["native_comparison_endpoint_counts"] = dict(Counter(
        str(
            row.get("comparison_output_endpoint_id")
            or row.get("completed_task_comparison_output_endpoint_id")
            or "unavailable"
        )
        for row in native_matrix_rows
    ))
    summary["native_frozen_host_postprocess_count"] = sum(
        1 for row in native_matrix_rows
        if row.get("host_postprocess_frozen") is True
    )
    summary["native_host_postprocess_required_count"] = sum(
        1 for row in native_matrix_rows
        if row.get("host_postprocess_required") is True
    )
    summary["native_host_postprocessing_available_count"] = sum(
        1 for row in native_matrix_rows
        if row.get("host_postprocessing_available") is True
    )
    summary["native_host_postprocessing_evidence_status_counts"] = dict(
        Counter(
            str(
                row.get("host_postprocessing_evidence_status")
                or "unavailable"
            )
            for row in native_matrix_rows
        )
    )
    summary["native_host_postprocessing_legacy_alias_conflict_count"] = sum(
        1 for row in native_matrix_rows
        if row.get("host_postprocessing_legacy_alias_conflict") is True
    )
    payload["summary"] = summary

    native_observations = collect_native_energy(run_dir)
    native_scientific_rows = scientific_energy_rows(run_dir)
    if native_scientific_rows:
        payload_rows = list(payload.get("rows") or [])
        payload_rows.extend(native_scientific_rows)
        payload["rows"] = payload_rows
    payload["summary"] = _refresh_scientific_summary(
        payload.get("summary")
        if isinstance(payload.get("summary"), Mapping)
        else {},
        list(payload.get("rows") or []),
    )
    payload["native_energy_observations"] = native_observations
    payload["native_energy_pair_comparison"] = build_native_energy_pairs(native_observations)
    payload["native_energy_ab_aggregates"] = build_native_energy_ab_aggregates(native_observations)
    native_evidence = read_json(
        reports_dir / "native_evidence_status.json", default={},
    ) or {}
    if isinstance(native_evidence, Mapping) and native_evidence:
        payload["native_evidence_status"] = dict(native_evidence)
        native_evidence_projection = (
            project_native_evidence_status(native_evidence)
        )
        summary = dict(payload.get("summary") or {})
        summary.update({
            "technical_status": str(
                native_evidence_projection.get("technical_status")
                or "unavailable"
            ),
            "runtime_complete": bool(
                (native_evidence.get("runtime") or {}).get("complete")
            ),
            "semantic_validation_complete": bool(
                (native_evidence.get("semantics") or {}).get("complete")
            ),
            "claim_validation_complete": bool(
                native_evidence_projection.get(
                    "claim_decisions_complete"
                )
            ),
            "scientific_status": str(
                native_evidence_projection.get("scientific_status")
                or "unavailable"
            ),
            "energy_execution_status": str(
                native_evidence_projection.get("energy_status")
                or "unavailable"
            ),
            "energy_requested": native_evidence_projection.get(
                "energy_requested"
            ),
            "energy_coverage_contract_active": (
                native_evidence_projection.get(
                    "energy_coverage_contract_active"
                )
            ),
            "energy_coverage_contract_status": str(
                native_evidence_projection.get(
                    "energy_coverage_contract_status"
                ) or "unavailable"
            ),
            "energy_matrix_expected_count": (
                native_evidence_projection.get(
                    "energy_matrix_expected_count"
                )
            ),
            "energy_plan_included_count": (
                native_evidence_projection.get(
                    "energy_plan_included_count"
                )
            ),
            "energy_plan_excluded_count": (
                native_evidence_projection.get(
                    "energy_plan_excluded_count"
                )
            ),
            "energy_measurement_success_count": (
                native_evidence_projection.get(
                    "energy_measurement_success_count"
                )
            ),
            "energy_measurement_failed_count": (
                native_evidence_projection.get(
                    "energy_measurement_failed_count"
                )
            ),
            "energy_measurement_started_count": (
                native_evidence_projection.get(
                    "energy_measurement_started_count"
                )
            ),
            "energy_not_started_preflight_count": (
                native_evidence_projection.get(
                    "energy_not_started_preflight_count"
                )
            ),
            "energy_claim_eligible_count": (
                native_evidence_projection.get(
                    "energy_claim_eligible_count"
                )
            ),
            "energy_plan_completion_fraction": (
                native_evidence_projection.get(
                    "energy_plan_completion_fraction"
                )
            ),
            "energy_planned_completion_fraction": (
                native_evidence_projection.get(
                    "energy_planned_completion_fraction"
                )
            ),
            "energy_planned_matrix_coverage_fraction": (
                native_evidence_projection.get(
                    "energy_planned_matrix_coverage_fraction"
                )
            ),
            "energy_successful_matrix_coverage_fraction": (
                native_evidence_projection.get(
                    "energy_successful_matrix_coverage_fraction"
                )
            ),
            "energy_matrix_coverage_fraction": (
                native_evidence_projection.get(
                    "energy_matrix_coverage_fraction"
                )
            ),
            "energy_coverage_contract_valid": (
                native_evidence_projection.get(
                    "energy_coverage_contract_valid"
                )
            ),
            "energy_plan_ledger_valid": (
                native_evidence_projection.get(
                    "energy_plan_ledger_valid"
                )
            ),
            "energy_result_ledger_valid": (
                native_evidence_projection.get(
                    "energy_result_ledger_valid"
                )
            ),
            "final_all_split_energy_required": (
                native_evidence_projection.get(
                    "final_all_split_energy_required"
                )
            ),
            "final_all_split_energy_complete": (
                native_evidence_projection.get(
                    "final_all_split_energy_complete"
                )
            ),
            "technical_quality_failure": bool(
                native_evidence.get("technical_quality_failure")
            ),
            "scientific_ready": bool(
                native_evidence_projection.get("scientific_ready")
            ),
        })
        payload["summary"] = summary

    run_status = read_json(
        reports_dir / "run_status_summary.json", default={},
    ) or {}
    run_manifest = read_json(run_dir / "run_manifest.json", default={}) or {}
    technical_execution_status = str(
        (
            run_status.get("technical_status")
            if isinstance(run_status, Mapping)
            else ""
        )
        or (
            run_status.get("status")
            if isinstance(run_status, Mapping)
            else ""
        )
        or (
            run_manifest.get("technical_status")
            if isinstance(run_manifest, Mapping)
            else ""
        )
        or (
            run_manifest.get("status")
            if isinstance(run_manifest, Mapping)
            else ""
        )
        or "pending"
    ).strip().lower()
    if technical_execution_status in {"completed", "success"}:
        technical_execution_status = "ok"
    technical_status_projection = (
        _full_quality_only_stale_hardware_smoke_projection(
            run_status=(
                run_status if isinstance(run_status, Mapping) else {}
            ),
            run_manifest=(
                run_manifest if isinstance(run_manifest, Mapping) else {}
            ),
            central_quality_reporting=central_quality_reporting,
        )
    )
    if technical_status_projection.get("applied") is True:
        technical_execution_status = str(
            technical_status_projection.get("projected_technical_status")
            or technical_execution_status
        )
    if technical_status_projection.get("applied") is not True:
        historical_projection = project_historical_workflow_status(run_status, native_evidence)
        if historical_projection.get("applied") is True:
            technical_status_projection = historical_projection
            technical_execution_status = historical_projection["projected_technical_status"]
    payload["technical_status_projection"] = technical_status_projection
    payload["completion"] = workflow_completion_projection(
        technical_execution_status, native_evidence=native_evidence,
        central_quality=central_quality_reporting,
        other_warning_count=int(run_status.get("non_blocking_reason_count") or 0),
        generic_excluded_count=int(((run_status.get("completion") or {}).get("counts") or {}).get("generic_excluded") or 0),
    )
    decision_axes = _scientific_decision_axes(
        payload,
        technical_execution_status=technical_execution_status,
        central_quality_reporting=central_quality_reporting,
    )
    technical_execution_status = str(
        decision_axes["technical_execution_status"]
    )
    quality_decision = str(decision_axes["aggregate_quality_decision"])
    scientific_status = str(decision_axes["scientific_status"])
    payload["decision_axes"] = decision_axes
    payload["technical_status"] = technical_execution_status
    payload["technical_execution_status"] = technical_execution_status
    payload["quality_decision"] = quality_decision
    payload["aggregate_quality_decision"] = quality_decision
    payload["quality_evaluation_status"] = decision_axes[
        "quality_evaluation_status"
    ]
    payload["aggregate_quality_status"] = decision_axes[
        "aggregate_quality_status"
    ]
    payload["development_analysis_status"] = decision_axes[
        "development_analysis_status"
    ]
    payload["claim_readiness_status"] = decision_axes[
        "claim_readiness_status"
    ]
    payload["scientific_status"] = scientific_status
    payload["scientific_pass"] = decision_axes["scientific_pass"]
    summary = dict(payload.get("summary") or {})
    summary.update({
        "technical_status": technical_execution_status,
        "technical_execution_status": technical_execution_status,
        "quality_evaluation_technical_status": decision_axes[
            "quality_evaluation_technical_status"
        ],
        "quality_decision": quality_decision,
        "aggregate_quality_decision": quality_decision,
        "quality_evaluation_status": decision_axes[
            "quality_evaluation_status"
        ],
        "aggregate_quality_status": decision_axes[
            "aggregate_quality_status"
        ],
        "development_analysis_status": decision_axes[
            "development_analysis_status"
        ],
        "claim_readiness_status": decision_axes[
            "claim_readiness_status"
        ],
        "claim_readiness_reasons": list(
            decision_axes["claim_readiness_reasons"]
        ),
        "claim_eligible_row_count": int(
            decision_axes["claim_eligible_row_count"]
        ),
        "performance_claim_eligible_row_count": int(
            decision_axes["performance_claim_eligible_row_count"]
        ),
        "energy_claim_eligible_row_count": int(
            decision_axes["energy_claim_eligible_row_count"]
        ),
        "scientific_status": scientific_status,
        "scientific_pass": decision_axes["scientific_pass"],
        "central_quality_result_count": int(
            central_quality_reporting.get("result_count") or 0
        ),
        "central_quality_request_count": int(
            central_quality_reporting.get("request_count") or 0
        ),
        "central_quality_setup_count": len({
            str(row.get("setup_id") or "")
            for row in list(payload.get("central_quality_results") or [])
            if isinstance(row, Mapping) and str(row.get("setup_id") or "")
        }),
        "central_quality_decision_counts": dict(
            central_quality_reporting.get("decision_counts") or {}
        ),
    })
    payload["summary"] = summary

    artifacts = _write_reports(
        report_root,
        payload,
        compatibility_root=compatibility_root,
    )
    quality_artifacts = _augment_v60z_quality_evidence(
        run_dir,
        report_root,
        profile,
        (
            list(payload.get("central_quality_results") or [])
            or list(payload.get("rows") or [])
        ),
    )
    payload["quality_evidence"] = quality_artifacts
    canonical_payload = read_json(
        report_root / "scientific_report.json", default=payload,
    )
    final_report_payload = (
        dict(canonical_payload)
        if isinstance(canonical_payload, Mapping)
        else dict(payload)
    )
    final_report_payload["quality_evidence"] = quality_artifacts
    posthook_paths = {
        "task_quality_reference_comparison_csv": (
            report_root / "task_quality_reference_comparison.csv"
        ),
        "task_quality_reference_comparison_json": (
            report_root / "task_quality_reference_comparison.json"
        ),
        "task_quality_reference_comparison_md": (
            report_root / "task_quality_reference_comparison.md"
        ),
        "task_quality_loss_decomposition_csv": (
            report_root / "task_quality_loss_decomposition.csv"
        ),
        "task_quality_loss_decomposition_json": (
            report_root / "task_quality_loss_decomposition.json"
        ),
        "task_quality_loss_decomposition_md": (
            report_root / "task_quality_loss_decomposition.md"
        ),
        "task_quality_reference_comparison_tex": (
            report_root / "thesis_tables"
            / "task_quality_reference_comparison.tex"
        ),
        "task_quality_loss_decomposition_tex": (
            report_root / "thesis_tables"
            / "task_quality_loss_decomposition.tex"
        ),
        "quality_evidence_v61a_json": (
            report_root / "quality_evidence_v61a.json"
        ),
        "quality_evidence_v60z_json": (
            report_root / "quality_evidence_v60z.json"
        ),
    }
    artifacts.update({
        key: path for key, path in posthook_paths.items() if path.is_file()
    })
    # The quality hook runs after the base tables. Re-seal the already
    # existing report manifest so its inventory and canonical JSON include the
    # newly generated, evidence-derived surfaces as well.
    artifacts["scientific_report_json"] = write_json(
        report_root / "scientific_report.json", final_report_payload,
    )
    artifacts.pop("scientific_report_manifest", None)
    manifest = {
        "schema": "onnx-splitpoint/scientific-report-manifest",
        "schema_version": 2,
        "created_at": now_iso(),
        "artifacts": [
            {
                "path": relpath(path, report_root),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts.values()
            if path.is_file()
        ],
    }
    artifacts["scientific_report_manifest"] = write_json(
        report_root / "report_manifest.json", manifest,
    )
    evidence_completeness_status = str(
        lifecycle_adapter.get("evidence_completeness_status") or "error"
    )
    measurement_completeness_status = str(
        lifecycle_adapter.get("measurement_completeness_status") or "error"
    )
    report_generation_status = "ok"
    replay_status = (
        "ok" if evidence_completeness_status == "complete"
        else f"ok_with_{evidence_completeness_status}_evidence"
    )
    return {
        "artifacts": artifacts,
        "model_count": int(payload.get("summary", {}).get("model_count") or 0),
        "scientific_row_count": len(payload.get("rows") or []),
        "ranking_method_comparison_rows": len(payload.get("ranking_method_comparison") or []),
        "ranking_method_count": len(payload.get("ranking_method_macro") or []),
        "figure_count": len([path for key, path in artifacts.items() if key.startswith("scientific_figure_")]),
        "open_item_count": len(payload.get("open_items") or []),
        "status": replay_status,
        "report_generation_status": report_generation_status,
        "evidence_completeness_status": evidence_completeness_status,
        "evidence_completeness_reasons": list(
            lifecycle_adapter.get("evidence_completeness_reasons") or []
        ),
        "measurement_completeness_status": measurement_completeness_status,
        "technical_status": technical_execution_status,
        "quality_decision": quality_decision,
        "scientific_status": scientific_status,
    }


def _load_suite_prediction(suite_root: Path, plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    model_id = str(
        plan.get("model_id")
        or next((row.get("model_id") for row in rows if row.get("model_id")), "")
        or suite_root.name
    )
    prediction_path = suite_root / "prediction.json"
    prediction = dict(read_json(prediction_path, default={}) or {})
    manifest_path = suite_root / "prediction_freeze_manifest.json"
    manifest = dict(read_json(manifest_path, default={}) or {})
    prediction_csv = suite_root / str(manifest.get("prediction_csv") or "predictions_frozen.csv")
    ranking_csv = suite_root / str(manifest.get("ranking_prediction_csv") or "ranking_predictions_frozen.csv")
    core_hash_ok = bool(
        prediction_path.is_file()
        and prediction_csv.is_file()
        and _same_sha256(sha256_file(prediction_path), manifest.get("prediction_sha256"))
        and _same_sha256(sha256_file(prediction_csv), manifest.get("prediction_csv_sha256"))
    )
    ranking_hash_ok = bool(
        ranking_csv.is_file()
        and _same_sha256(sha256_file(ranking_csv), manifest.get("ranking_prediction_csv_sha256"))
    )
    universe_path = suite_root / str(manifest.get("candidate_universe_manifest") or "candidate_universe_manifest.json")
    universe = dict(read_json(universe_path, default={}) or {})
    universe_declared = bool(manifest.get("candidate_universe_manifest") or manifest.get("candidate_universe_sha256"))
    plan_entries = _profile_model_entries({"model_suite": plan.get("model_suite") or {}})
    plan_role = normalize_evaluation_role((plan_entries.get(model_id) or {}).get("evaluation_role"))
    row_role = normalize_evaluation_role(next((row.get("evaluation_role") for row in rows if row.get("evaluation_role")), ""))
    ranking_cfg = plan.get("ranking_validation") if isinstance(plan.get("ranking_validation"), Mapping) else {}
    universe_required = bool(
        is_confirmatory_holdout(manifest.get("evaluation_role"))
        or _b(manifest.get("valid_for_holdout")) is True
        or is_confirmatory_holdout(plan_role)
        or is_confirmatory_holdout(row_role)
        or ranking_cfg.get("require_complete_candidate_universe") is True
    )
    universe_self_hash = (
        sha256_json({
            key: value
            for key, value in universe.items()
            if key not in {"universe_sha256", "created_at"}
        })
        if universe_declared and universe_path.is_file() and universe
        else ""
    )
    universe_manifest_valid = bool(
        universe_declared
        and universe_self_hash
        and str(universe.get("universe_sha256") or "") == universe_self_hash
        and str(manifest.get("candidate_universe_sha256") or "") == universe_self_hash
    )
    universe_hash_ok = bool(
        universe_manifest_valid
        or (not universe_declared and not universe_required)
    )
    universe_csv = suite_root / str(manifest.get("candidate_universe_csv") or "candidate_universe.csv")
    expected_universe_csv_hash = str(manifest.get("candidate_universe_csv_sha256") or "")
    actual_universe_csv_hash = sha256_file(universe_csv) if universe_csv.is_file() else None
    universe_csv_hash_ok = bool(
        not expected_universe_csv_hash
        or (
            actual_universe_csv_hash
            and _same_sha256(expected_universe_csv_hash, actual_universe_csv_hash)
        )
    )
    campaign_cfg = plan.get("campaign") if isinstance(plan.get("campaign"), Mapping) else {}
    approval_required = bool(campaign_cfg.get("require_prediction_freeze_approval"))
    approval_path = suite_root / "prediction_freeze_approval.json"
    approval_verification = verify_prediction_freeze_approval(
        approval_path,
        public_key=(suite_root / "prediction_freeze_public_key.pem") if (suite_root / "prediction_freeze_public_key.pem").is_file() else None,
        require_cryptographic_signature=bool(campaign_cfg.get("require_cryptographic_prediction_signature")),
    ) if approval_path.is_file() else {"ok": not approval_required, "signature_status": "missing"}
    approval_ok = bool(approval_verification.get("ok"))
    conflict = (suite_root / "prediction_freeze_conflict.json").is_file()
    valid = bool(
        core_hash_ok
        and ranking_hash_ok
        and universe_hash_ok
        and universe_csv_hash_ok
        and approval_ok
        and manifest.get("prospective")
        and not conflict
    )
    status = str(manifest.get("freeze_status") or ("prospective_frozen" if valid else "invalid"))
    if conflict:
        status = "hash_conflict"
    elif not core_hash_ok:
        status = "hash_mismatch"
    elif not ranking_hash_ok:
        status = "ranking_prediction_hash_mismatch"
    elif not universe_hash_ok:
        status = (
            "candidate_universe_required_missing"
            if universe_required and not universe_declared
            else "candidate_universe_hash_mismatch"
        )
    elif not universe_csv_hash_ok:
        status = "candidate_universe_csv_hash_mismatch"
    elif not approval_ok:
        status = "prediction_freeze_approval_missing_or_invalid"
    audit = universe.get("audit") if isinstance(universe.get("audit"), Mapping) else {}
    try:
        audit_minimum_valid_candidates = max(0, int(audit.get("minimum_valid_candidates") or 0))
    except Exception:
        audit_minimum_valid_candidates = 0
    prediction["_prediction_freeze"] = {
        **manifest,
        "valid": valid,
        "ranking_predictions_valid": ranking_hash_ok,
        "candidate_universe_required": universe_required,
        "candidate_universe_valid": universe_manifest_valid and universe_csv_hash_ok,
        "candidate_universe_scope": universe.get("claim_scope", ""),
        "candidate_universe_complete": universe.get("declared_complete"),
        "candidate_universe_selected_case_ids": [
            str(value)
            for value in list(universe.get("selected_case_ids") or [])
            if str(value)
        ],
        "candidate_universe_minimum_valid_candidates": audit_minimum_valid_candidates,
        "candidate_universe_manifest_path": universe_path.name if universe_path.is_file() else "",
        "candidate_universe_csv_path": universe_csv.name if universe_csv.is_file() else "",
        "prediction_freeze_approval_required": approval_required,
        "prediction_freeze_approval_valid": approval_ok,
        "prediction_freeze_approval": approval_verification,
        "status": status,
        "manifest_path": "prediction_freeze_manifest.json" if manifest_path.is_file() else "",
        "ranking_prediction_csv_path": ranking_csv.name if ranking_csv.is_file() else "",
    }
    prediction["_ranking_method_predictions"] = _read_csv_rows(ranking_csv)
    return {model_id: prediction} if model_id else {}


def build_benchmarkset_scientific_report(
    suite_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Optional[Mapping[str, Any]] = None,
    tool_version: str = "",
) -> dict[str, Any]:
    """Write the canonical report directly into a generated BenchmarkSet."""
    suite_root = Path(suite_root)
    plan = dict(plan or read_json(suite_root / "benchmark_plan.json", default={}) or {})
    profile = {
        "quality_gate": plan.get("quality_gate") or {},
        "ranking_validation": plan.get("ranking_validation") or {},
        "model_suite": plan.get("model_suite") or {},
        "campaign": plan.get("campaign") or {},
    }
    policy = _policy_from_profile(profile)
    profile_entries = _profile_model_entries(profile)
    model_suite = plan.get("model_suite") if isinstance(plan.get("model_suite"), Mapping) else {}
    primary_model = model_suite.get("primary")
    if isinstance(primary_model, Mapping):
        primary_model = primary_model.get("id") or primary_model.get("model_id")
    canonical_model_candidates = {
        str(value).strip()
        for value in (plan.get("model_id"), primary_model)
        if str(value or "").strip() and str(value).strip().lower() != "model"
    }
    canonical_placeholder_model = (
        next(iter(canonical_model_candidates))
        if len(canonical_model_candidates) == 1 else ""
    )
    enriched: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        model_id = str(row.get("model_id") or "").strip()
        if model_id.lower() in {"", "model"}:
            model_id = canonical_placeholder_model or "unknown_model"
            row["model_id"] = model_id
            row["model_identity_source"] = (
                "benchmark_plan_single_canonical_model"
                if canonical_placeholder_model else "unresolved_placeholder"
            )
        declared_role = (profile_entries.get(model_id) or {}).get("evaluation_role")
        row["evaluation_role"] = normalize_evaluation_role(
            row.get("evaluation_role") or declared_role or "development"
        )
        apply_accuracy_gate_to_row(row, policy)
        enriched.append(row)
    deduped = _dedupe_rows(enriched)
    predictions = _load_suite_prediction(suite_root, plan, deduped)
    payload = _build_report_payload(
        source_kind="benchmark_set",
        source_root=suite_root,
        profile_id=str((plan.get("evaluation_profile") or {}).get("profile_id") or suite_root.name),
        tool_version=tool_version,
        workflow_version="benchmark_suite",
        profile=profile,
        policy=policy,
        rows=deduped,
        predictions=predictions,
        model_facts=[],
    )
    artifacts = _write_reports(suite_root / "scientific_report", payload)
    return {
        "artifacts": artifacts,
        "status": "ok",
        "row_count": len(payload.get("rows") or []),
        "ranking_method_count": len(payload.get("ranking_method_macro") or []),
    }


# Backward-compatible name used by pre-v60 callers.
def build_result_dashboard_reports(
    run_dir: Path,
    *,
    profile_id: str = "",
    tool_version: str = "",
    workflow_version: str = "",
) -> dict[str, Any]:
    return build_scientific_reports(
        run_dir,
        profile_id=profile_id,
        tool_version=tool_version,
        workflow_version=workflow_version,
    )


__all__ = [
    "build_scientific_reports",
    "build_benchmarkset_scientific_report",
    "build_result_dashboard_reports",
    "clean_legacy_reports",
    "project_central_quality_status",
]


def _augment_v60z_quality_evidence(run_dir, report_dir, profile, rows):
    """Best-effort official-COCO and task-quality decomposition post-hook."""
    try:
        return augment_scientific_report(run_dir=run_dir, report_dir=report_dir, profile=profile or {}, rows=list(rows or []))
    except Exception as exc:
        from pathlib import Path
        import json
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {"status": "error", "reason": str(exc)}
        (out / "quality_evidence_v60z.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
