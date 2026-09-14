from __future__ import annotations

"""Canonical, claim-safe Native performance matrix ingestion.

Native Split and setup-local Native Full rows are produced outside the Generic
normalisation tree.  They therefore need an explicit reporting contract instead
of being silently omitted from the Scientific Report.  This module preserves
the full measured matrix as development/screening evidence while deliberately
leaving final claim admission to the existing quality and protocol gates.
"""

from pathlib import Path
from typing import Any, Mapping
import json
import math

from .workflow.full_only_quality_canary import (
    resolve_full_only_quality_canary,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, Mapping) else {}
    except Exception:
        return {}


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _truth(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "ok", "pass", "passed", "claim_ok", "eligible"}:
        return True
    if text in {"0", "false", "no", "fail", "failed", "invalid", "error"}:
        return False
    return None


def _explicit_count(payload: Mapping[str, Any], key: str, fallback: int) -> int:
    """Preserve authoritative zero counters instead of treating them as absent."""

    value = payload.get(key)
    if value is None:
        return int(fallback)
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _case(value: Any, backend: Any = "") -> str:
    text = str(value or "").strip().lower()
    if text in {"", "none"}:
        return "full" if str(backend or "").startswith("native_full_") else ""
    if text == "full":
        return text
    if text.startswith("b") and text[1:].isdigit():
        return f"b{int(text[1:]):03d}"
    if text.isdigit():
        return f"b{int(text):03d}"
    return text


def _task(model: Any, explicit: Any) -> str:
    raw = str(explicit or "").strip().lower().replace("-", "_")
    if raw in {"classification", "image_classification", "classifier"}:
        return "classification"
    if raw in {"detection", "object_detection", "detector"}:
        return "detection"
    model_text = str(model or "").strip().lower()
    return "detection" if any(token in model_text for token in ("yolo", "detr", "detect")) else "classification"


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    backend = str(row.get("backend") or "").strip()
    return (
        str(row.get("model") or row.get("model_id") or "").strip(),
        backend,
        _case(row.get("case") or row.get("case_id"), backend),
        str(row.get("precision") or "").strip(),
        str(row.get("setup_id") or "").strip(),
        str(row.get("comparison_backend") or "").strip(),
    )


def _scope(value: Any) -> str:
    source = str(value or "").strip()
    return {
        "accelerator_only": "accelerator_output_endpoint",
        "accelerator_output": "accelerator_output_endpoint",
    }.get(source, source or "unavailable")


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row.get(name) not in (None, ""):
            return row.get(name)
    return None


def _comparison_endpoint_id(row: Mapping[str, Any]) -> str:
    """Prefer the backend-independent task endpoint without hiding physics."""
    return str(
        _row_value(
            row,
            "comparison_output_endpoint_id",
            "completed_task_comparison_output_endpoint_id",
            "physical_output_endpoint_id",
            "output_endpoint_id",
        )
        or ""
    ).strip()


def _contract_bool(value: Any) -> str:
    parsed = _truth(value)
    return "" if parsed is None else ("true" if parsed else "false")


def _measurement_contract_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        _scope(row.get("e2e_scope")),
        str(row.get("comparison_endpoint_stratum") or "").strip(),
        str(row.get("measurement_concurrency") or "").strip(),
        str(row.get("completed_task_stage") or "").strip(),
        _comparison_endpoint_id(row),
        str(
            _row_value(
                row,
                "completed_task_comparison_endpoint_contract_hash",
                "completed_task_endpoint_contract_hash",
            )
            or ""
        ).strip(),
        str(row.get("completed_task_completion_mode") or "").strip(),
        str(row.get("frozen_host_postprocess_contract_sha256") or "").strip(),
        _contract_bool(
            _row_value(
                row,
                "host_postprocessing_available",
                "host_tail_available",
            )
        ),
        _contract_bool(
            _row_value(
                row,
                "host_postprocess_required",
                "host_tail_required",
            )
        ),
    )


def _pick(
    source: Mapping[str, Any],
    detail: Mapping[str, Any],
    *names: str,
) -> Any:
    for container in (detail, source):
        for name in names:
            if name in container and container.get(name) not in (None, ""):
                return container.get(name)
    return None


def _detail_score(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    contract = row.get("native_command_contract") if isinstance(row.get("native_command_contract"), Mapping) else {}
    return (
        1 if _truth(row.get("ok")) is True else 0,
        1 if (
            _num(row.get("fps_median")) is not None
            or _num(row.get("latency_median_ms")) is not None
            or str(row.get("repetition_aggregation") or "").strip()
        ) else 0,
        1 if _num(row.get("fps_makespan")) is not None else 0,
        1 if str(row.get("setup_id") or contract.get("setup_id") or "").strip() else 0,
    )


def _detail_for(
    row: Mapping[str, Any],
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    model, backend, case, precision, setup_id, comparison = _row_key(row)
    candidates: list[dict[str, Any]] = []
    for detail in details:
        d_model, d_backend, d_case, d_precision, d_setup, d_comparison = _row_key(detail)
        if (d_model, d_backend, d_case) != (model, backend, case):
            continue
        if precision and d_precision and precision != d_precision:
            continue
        if setup_id and d_setup and setup_id != d_setup:
            continue
        if comparison and d_comparison and comparison != d_comparison:
            continue
        candidates.append(detail)
    source_contract = _measurement_contract_key(row)
    if any(value not in {"", "unavailable"} for value in source_contract):
        exact = [
            candidate for candidate in candidates
            if all(
                expected in {"", "unavailable"} or observed == expected
                for observed, expected in zip(
                    _measurement_contract_key(candidate),
                    source_contract,
                )
            )
        ]
        if exact:
            candidates = exact
    return max(candidates, key=_detail_score) if candidates else {}


def _theoretical_cycle(detail: Mapping[str, Any]) -> tuple[float | None, str]:
    for key in ("paper_equivalent_cycle_ms", "pipeline_cycle_ms", "cycle_ms"):
        value = _num(detail.get(key))
        if value is not None and value > 0:
            return value, key
    paper_fps = _num(detail.get("paper_fps") or detail.get("paper_equivalent_fps"))
    if paper_fps is not None and paper_fps > 0:
        return 1000.0 / paper_fps, "paper_fps"
    thread_cycles = [
        value
        for value in (_num(detail.get("p1_thread_ms")), _num(detail.get("p2_thread_ms")))
        if value is not None and value > 0
    ]
    return (max(thread_cycles), "max(p1_thread_ms,p2_thread_ms)") if thread_cycles else (None, "")


def collect_native_performance_matrix(run_dir: str | Path) -> dict[str, Any]:
    """Return all Native performance rows under a development-only contract."""
    run = Path(run_dir)
    reports = run / "reports"
    effective_plan = _load(run / "effective_execution_plan.json")
    profile = _load(run / "profile.yaml")
    if not profile:
        try:
            import yaml

            loaded = yaml.safe_load(
                (run / "profile.yaml").read_text(encoding="utf-8")
            )
            profile = dict(loaded) if isinstance(loaded, Mapping) else {}
        except Exception:
            profile = {}
    canary_contract = resolve_full_only_quality_canary(
        profile, plan_rows=None,
    )
    full_only_canary = bool(
        effective_plan.get("quality_canary_enabled") is True
        and str(
            effective_plan.get("quality_canary_execution_scope") or ""
        ).strip().lower() == "full_only"
        and effective_plan.get("native_enabled") is False
        and int(
            effective_plan.get("expected_full_quality_results_total") or 0
        ) > 0
        and effective_plan.get("performance_claims_emitted") is False
        and canary_contract.get("enabled") is True
        and canary_contract.get("ok") is True
        and str(
            canary_contract.get("execution_scope") or ""
        ).strip().lower() == "full_only"
        and canary_contract.get("performance_claims_emitted") is False
        and len(list(
            canary_contract.get("expected_full_quality_identities") or []
        )) == int(
            effective_plan.get("expected_full_quality_results_total") or 0
        )
    )
    concise_path = reports / "native_stage_concise_summary.json"
    concise = _load(concise_path)
    concise_rows = [dict(row) for row in list(concise.get("rows") or []) if isinstance(row, Mapping)]

    detail_path = reports / "native_producer_combined_summary.json"
    detail_payload = _load(detail_path)
    if not detail_payload:
        detail_path = reports / "native_producer_summary.json"
        detail_payload = _load(detail_path)
    details = [dict(row) for row in list(detail_payload.get("rows") or []) if isinstance(row, Mapping)]

    # Older/manual runs may lack the concise matrix.  Deduplicate the detailed
    # summary by its complete setup-local execution identity in that case.
    if not concise_rows:
        best: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
        for row in details:
            key = _row_key(row)
            current = best.get(key)
            if current is None or _detail_score(row) > _detail_score(current):
                best[key] = row
        concise_rows = list(best.values())
        concise_path = detail_path

    expected_path = reports / "native_expected_matrix.json"
    from .native_job_identity import project_known_build_exclusions
    expected = project_known_build_exclusions(_load(expected_path))
    expected_rows = [dict(row) for row in list(expected.get("present_expected_rows") or []) if isinstance(row, Mapping)]
    expected_keys = {_row_key(row)[:3] for row in expected_rows}

    # Full-only Quality Canaries intentionally disable the Native performance
    # producer.  Expose an explicit N/A contract only when both archived
    # profile and effective execution plan agree and no stale Native artifacts
    # leaked into the run.  Ordinary empty matrices remain partial/fail-closed.
    if full_only_canary and not concise_rows and not details and not expected_rows:
        explicit_counts = any(
            int(expected.get(key) or 0) != 0
            for key in (
                "expected_row_count",
                "present_expected_row_count",
                "successful_expected_row_count",
                "failed_expected_row_count",
                "missing_expected_row_count",
            )
        )
        if not explicit_counts:
            return {
                "schema": "onnx-splitpoint/native-performance-matrix",
                "schema_version": 5,
                "status": "not_applicable_quality_evidence_only",
                "applicable": False,
                "expected_row_count": 0,
                "present_expected_row_count": 0,
                "successful_expected_row_count": 0,
                "failed_expected_row_count": 0,
                "missing_expected_row_count": 0,
                "row_presence_complete": None,
                "execution_success_complete": None,
                "matrix_complete": None,
                "observation_count": 0,
                "comparison_eligible_observation_count": 0,
                "claim_eligibility_policy": (
                    "not_applicable; Full-only Quality Canary has Native "
                    "performance disabled"
                ),
                "source_path": "",
                "expected_matrix_source_path": (
                    str(expected_path) if expected_path.is_file() else ""
                ),
                "observations": [],
                "all_observations": [],
                "comparison_eligible_observations": [],
            }

    observations: list[dict[str, Any]] = []
    for source in concise_rows:
        detail = _detail_for(source, details)
        contract = detail.get("native_command_contract") if isinstance(detail.get("native_command_contract"), Mapping) else {}
        backend = str(source.get("backend") or detail.get("backend") or "").strip()
        model = str(source.get("model") or source.get("model_id") or detail.get("model") or "").strip()
        case = _case(source.get("case") or source.get("case_id") or detail.get("case"), backend)
        execution_mode = str(
            source.get("execution_mode")
            or detail.get("execution_mode")
            or ("native_full_baseline" if backend.startswith("native_full_") else "native_split")
        ).strip()
        # The combined Native report is authoritative for repeated runs.  Its
        # median must win over the concise row's compatibility ``fps`` alias;
        # otherwise a legacy single/best row could silently replace the
        # independent-run aggregate in the scientific observation.
        measured_fps = next((
            value for value in (
                _num(source.get("p2_output_fps")),
                _num(detail.get("p2_output_fps")),
                _num(source.get("throughput_primary_fps")),
                _num(detail.get("throughput_primary_fps")),
                _num(source.get("raw_model_outputs_fps_median")),
                _num(detail.get("raw_model_outputs_fps_median")),
                _num(source.get("fps_median")),
                _num(detail.get("fps_median")),
                _num(source.get("fps")),
                _num(source.get("fps_makespan")),
                _num(detail.get("fps_makespan")),
                _num(detail.get("fps")),
            ) if value is not None
        ), None)
        measured_source = (
            "native_stage_concise_summary.p2_output_fps"
            if _num(source.get("p2_output_fps")) is not None
            else "native_summary.p2_output_fps"
            if _num(detail.get("p2_output_fps")) is not None
            else "native_stage_concise_summary.throughput_primary_fps"
            if _num(source.get("throughput_primary_fps")) is not None
            else "native_summary.throughput_primary_fps"
            if _num(detail.get("throughput_primary_fps")) is not None
            else "native_stage_concise_summary.raw_model_outputs_fps_median"
            if _num(source.get("raw_model_outputs_fps_median")) is not None
            else "native_summary.raw_model_outputs_fps_median"
            if _num(detail.get("raw_model_outputs_fps_median")) is not None
            else "native_stage_concise_summary.fps_median"
            if _num(source.get("fps_median")) is not None
            else "native_summary.fps_median"
            if _num(detail.get("fps_median")) is not None
            else "native_stage_concise_summary.fps"
            if _num(source.get("fps")) is not None
            else "native_summary.fps_makespan"
            if _num(detail.get("fps_makespan")) is not None
            else "native_summary.fps"
            if _num(detail.get("fps")) is not None
            else "unavailable"
        )
        theoretical_cycle, theoretical_source = _theoretical_cycle(detail)
        setup_id = str(
            source.get("setup_id")
            or detail.get("setup_id")
            or contract.get("setup_id")
            or ""
        ).strip()
        comparison_backend = str(
            source.get("comparison_backend")
            or detail.get("comparison_backend")
            or contract.get("comparison_backend")
            or ""
        ).strip()
        runtime_ok = str(source.get("runtime_status") or detail.get("status") or "").strip().lower() in {
            "ok", "pass", "passed", "success", "completed"
        } or _truth(detail.get("ok")) is True
        semantic_ok = _truth(source.get("semantic_ok"))
        structural_contract_pass = _truth(
            _pick(
                source,
                detail,
                "structural_contract_pass",
                "contract_consistent",
            )
        )
        contract_consistent = structural_contract_pass
        performance_contract_eligible = _truth(
            detail.get("performance_claim_eligible")
            if detail.get("performance_claim_eligible") is not None
            else source.get("performance_claim_eligible")
        ) is True
        exclusion_reasons = (
            detail.get("performance_claim_exclusion_reasons")
            or source.get("performance_claim_exclusion_reasons")
            or []
        )
        if isinstance(exclusion_reasons, str):
            exclusion_reasons = [exclusion_reasons] if exclusion_reasons else []
        else:
            exclusion_reasons = list(exclusion_reasons)
        latency_median_ms = next((
            value for value in (
                _num(detail.get("latency_median_ms")),
                _num(source.get("latency_median_ms")),
                _num(detail.get("latency_mean_ms")),
                _num(source.get("latency_mean_ms")),
            ) if value is not None
        ), None)
        latency_ci95_low_ms = next((
            value for value in (
                _num(detail.get("latency_ci95_low_ms")),
                _num(source.get("latency_ci95_low_ms")),
            ) if value is not None
        ), None)
        latency_ci95_high_ms = next((
            value for value in (
                _num(detail.get("latency_ci95_high_ms")),
                _num(source.get("latency_ci95_high_ms")),
            ) if value is not None
        ), None)
        repetition_aggregation = str(
            detail.get("repetition_aggregation")
            or source.get("repetition_aggregation")
            or ""
        )
        aggregation_lower = repetition_aggregation.strip().lower().replace("-", "_")
        # ``median_never_best_of`` is the name of the safe contract, not a
        # best-of selection.  A substring check incorrectly rejected every
        # compliant row because the contract deliberately contains the words
        # ``best_of`` in its prohibition.
        best_of_used = any(token in aggregation_lower for token in (
            "best_of_", "fastest", "minimum_of", "maximum_of",
        )) and aggregation_lower != "median_never_best_of"
        repetition_runtime_scope = str(
            detail.get("repetition_runtime_scope")
            or source.get("repetition_runtime_scope")
            or ""
        ).strip().lower()
        repetition_independence_verified = _truth(
            detail.get("repetition_independence_verified")
            if detail.get("repetition_independence_verified") is not None
            else source.get("repetition_independence_verified")
        )
        repeat_claim_gate_pass = _truth(
            detail.get("repeat_claim_gate_pass")
            if detail.get("repeat_claim_gate_pass") is not None
            else source.get("repeat_claim_gate_pass")
        )
        independent_repetitions = (
            repetition_runtime_scope in {
                "fresh_runtime_per_repetition",
                "fresh_process_per_repetition",
                "fresh_hailo_vstreams_trt_completion_runtime_per_repetition",
            }
            and repetition_independence_verified is True
        )
        if best_of_used:
            performance_contract_eligible = False
            if "best_of_aggregation_not_allowed" not in exclusion_reasons:
                exclusion_reasons.append("best_of_aggregation_not_allowed")
        if "median" in aggregation_lower and not independent_repetitions:
            performance_contract_eligible = False
            if "repetition_independence_not_verified" not in exclusion_reasons:
                exclusion_reasons.append("repetition_independence_not_verified")
        if performance_contract_eligible and repeat_claim_gate_pass is not True:
            performance_contract_eligible = False
            if "raw_repeat_claim_gate_not_passed" not in exclusion_reasons:
                exclusion_reasons.append("raw_repeat_claim_gate_not_passed")
        repetition_count_valid = int(
            _num(detail.get("repetition_count_valid") or source.get("repetition_count_valid")) or 0
        )
        identity_key = (model, backend, case)
        source_e2e_scope = str(
            _pick(source, detail, "source_e2e_scope", "e2e_scope") or ""
        )
        e2e_scope = _scope(_pick(source, detail, "e2e_scope"))
        output_endpoint_id = str(
            _pick(source, detail, "output_endpoint_id") or ""
        )
        physical_output_endpoint_id = str(
            _pick(
                source,
                detail,
                "physical_output_endpoint_id",
                "output_endpoint_id",
            )
            or ""
        )
        completed_task_comparison_output_endpoint_id = str(
            _pick(
                source,
                detail,
                "completed_task_comparison_output_endpoint_id",
            )
            or ""
        )
        comparison_output_endpoint_id = str(
            _pick(source, detail, "comparison_output_endpoint_id")
            or completed_task_comparison_output_endpoint_id
            or physical_output_endpoint_id
        )
        native_measured_cycle_ms = (
            (1000.0 / measured_fps)
            if measured_fps is not None and measured_fps > 0
            else None
        )
        runtime_precision_identity = str(
            _pick(
                source,
                detail,
                "runtime_precision_identity",
                "execution_precision",
                "full_runtime_precision",
                "precision",
            )
            or ""
        ).strip()
        precision_quality_binding_verified = _truth(
            _pick(source, detail, "precision_quality_binding_verified")
        )
        task_quality_observation_valid = _truth(
            _pick(source, detail, "task_quality_observation_valid")
        )
        quality_claim_result_verified = _truth(
            _pick(source, detail, "quality_claim_result_verified")
        )
        quality_central_evidence_verified = _truth(
            _pick(source, detail, "quality_central_evidence_verified")
        )
        quality_accuracy_gate_pass = _truth(
            _pick(source, detail, "quality_accuracy_gate_pass")
        )
        quality_eligible_for_ranking = _truth(
            _pick(source, detail, "quality_eligible_for_ranking")
        )
        task_quality_pass = _truth(
            _pick(source, detail, "task_quality_pass", "task_valid")
        )
        quality_gate_status = str(
            _pick(source, detail, "quality_gate_status") or ""
        ).strip()
        normalized_quality_gate_status = (
            quality_gate_status.lower().replace("-", "_").replace(" ", "_")
        )
        ranking_exclusion_reasons: list[str] = []
        if not runtime_ok:
            ranking_exclusion_reasons.append("native_runtime_not_successful")
        if native_measured_cycle_ms is None:
            ranking_exclusion_reasons.append("native_measured_cycle_missing")
        for field, value in (
            (
                "precision_quality_binding_not_verified",
                precision_quality_binding_verified,
            ),
            (
                "task_quality_observation_not_valid",
                task_quality_observation_valid,
            ),
            (
                "central_quality_evidence_not_verified",
                quality_central_evidence_verified,
            ),
            ("task_quality_not_passed", task_quality_pass),
            ("quality_accuracy_gate_not_passed", quality_accuracy_gate_pass),
        ):
            if value is not True:
                ranking_exclusion_reasons.append(field)
        if normalized_quality_gate_status in {
            "fail",
            "failed",
            "inconclusive",
            "error",
            "blocked",
            "invalid",
            "rejected",
        }:
            ranking_exclusion_reasons.append(
                "quality_gate_status_not_admissible:"
                + normalized_quality_gate_status
            )
        ranking_eligible = not ranking_exclusion_reasons
        observations.append({
            "model_id": model,
            "model": model,
            "task": _task(model, source.get("task") or detail.get("task")),
            "stage": str(_pick(source, detail, "stage") or ""),
            "output_format": str(
                _pick(source, detail, "output_format") or ""
            ),
            "contract_family": str(
                _pick(source, detail, "contract_family") or ""
            ),
            "contract_source": str(
                _pick(source, detail, "contract_source") or ""
            ),
            "endpoint_contract_complete": _truth(
                _pick(source, detail, "endpoint_contract_complete")
            ),
            "endpoint_contract_hash": str(
                _pick(source, detail, "endpoint_contract_hash") or ""
            ),
            "output_endpoint_attestation": _pick(
                source, detail, "output_endpoint_attestation"
            ),
            "accelerator_output_stage": str(
                _pick(source, detail, "accelerator_output_stage") or ""
            ),
            "accelerator_output_contract_family": str(
                _pick(
                    source,
                    detail,
                    "accelerator_output_contract_family",
                )
                or ""
            ),
            "accelerator_endpoint_contract_hash": str(
                _pick(
                    source,
                    detail,
                    "accelerator_endpoint_contract_hash",
                )
                or ""
            ),
            "accelerator_output_endpoint_attestation": _pick(
                source,
                detail,
                "accelerator_output_endpoint_attestation",
            ),
            "case_id": case,
            "case": case,
            "backend": backend,
            "variant": execution_mode,
            "execution_mode": execution_mode,
            "runner_regime": "native_full" if execution_mode == "native_full_baseline" else "native_fifo",
            "setup_id": setup_id,
            "comparison_backend": comparison_backend,
            "precision": str(source.get("precision") or detail.get("precision") or ""),
            "runtime_precision_identity": runtime_precision_identity,
            "direction": str(
                _pick(source, detail, "direction") or backend
            ).strip(),
            "runtime_status": str(source.get("runtime_status") or detail.get("status") or "unavailable"),
            "runtime_executable": runtime_ok,
            "native_measured_throughput_fps": measured_fps,
            "throughput_fps": measured_fps,
            "native_measured_fps_source": measured_source,
            "fps_median": _num(detail.get("fps_median") or source.get("fps_median") or measured_fps),
            "fps_ci95_low": _num(detail.get("fps_ci95_low") or source.get("fps_ci95_low")),
            "fps_ci95_high": _num(detail.get("fps_ci95_high") or source.get("fps_ci95_high")),
            "repetition_count_requested": int(_num(detail.get("repetition_count_requested") or source.get("repetition_count_requested")) or 0),
            "repetition_count_attempted": int(_num(detail.get("repetition_count_attempted") or source.get("repetition_count_attempted")) or 0),
            "repetition_count_valid": repetition_count_valid,
            "repetition_status": str(detail.get("repetition_status") or source.get("repetition_status") or ""),
            "repetition_aggregation": repetition_aggregation,
            "repetition_runtime_scope": repetition_runtime_scope,
            "repetition_independence_verified": repetition_independence_verified,
            "performance_statistic": (
                "median_of_independent_repetitions"
                if "median" in aggregation_lower and independent_repetitions
                else "median_with_independence_unverified"
                if "median" in aggregation_lower
                else "unsupported_best_of"
                if best_of_used
                else "single_observation"
                if repetition_count_valid == 1
                else "unavailable"
            ),
            "best_of_used": best_of_used,
            # The compatibility primary is explicitly the independent-run
            # median; it must never expose a fastest repetition as latency.
            "latency_ms": latency_median_ms,
            "latency_mean_ms": latency_median_ms,
            "latency_median_ms": latency_median_ms,
            "latency_ci95_low_ms": latency_ci95_low_ms,
            "latency_ci95_high_ms": latency_ci95_high_ms,
            "latency_p50_ms": _num(detail.get("latency_p50_ms") or source.get("latency_p50_ms")),
            "latency_p95_ms": _num(detail.get("latency_p95_ms") or source.get("latency_p95_ms")),
            "latency_semantics": str(detail.get("latency_semantics") or source.get("latency_semantics") or ""),
            "completion_interval_mean_ms": _num(detail.get("completion_interval_mean_ms") or source.get("completion_interval_mean_ms")),
            "native_measured_cycle_ms": native_measured_cycle_ms,
            # Compatibility aliases let the common ranking implementation use
            # the measured Native makespan without deriving it a second time.
            "pipeline_cycle_selected_ms": native_measured_cycle_ms,
            "throughput_primary_fps": measured_fps,
            "performance_endpoint": str(
                _pick(source, detail, "performance_endpoint") or ""
            ),
            "primary_performance_endpoint": str(
                _pick(source, detail, "primary_performance_endpoint") or ""
            ),
            "application_performance_endpoint": str(
                _pick(source, detail, "application_performance_endpoint") or ""
            ),
            "p2_output_fps": _num(
                _pick(source, detail, "p2_output_fps", "raw_model_outputs_fps_median")
            ),
            "completed_detection_fps": _num(
                _pick(source, detail, "completed_detection_fps", "completed_task_fps_median")
            ),
            "application_throughput_fps": _num(
                _pick(source, detail, "application_throughput_fps", "completed_detection_fps", "completed_task_fps_median")
            ),
            "completed_to_p2_ratio": _num(
                _pick(source, detail, "completed_to_p2_ratio")
            ),
            "p2_output_contract_family": str(
                _pick(source, detail, "p2_output_contract_family") or ""
            ),
            "postprocess_adapter_id": str(
                _pick(source, detail, "postprocess_adapter_id") or ""
            ),
            "postprocess_location": str(
                _pick(source, detail, "postprocess_location") or ""
            ),
            "stage_timings": _pick(source, detail, "stage_timings"),
            "directly_measured": _truth(
                _pick(source, detail, "directly_measured")
            ),
            "projection_source": str(
                _pick(source, detail, "projection_source") or ""
            ),
            "native_theoretical_cycle_ms": theoretical_cycle,
            "native_theoretical_cycle_rate_fps": (
                1000.0 / theoretical_cycle
                if theoretical_cycle is not None and theoretical_cycle > 0
                else None
            ),
            "native_theoretical_cycle_source": theoretical_source,
            "handoff_ms": _num(source.get("handoff_ms") or detail.get("handoff_ms")),
            "semantic_status": str(source.get("semantic_status") or "unavailable"),
            "semantic_ok": semantic_ok,
            "contract_consistent": contract_consistent,
            "structural_contract_pass": structural_contract_pass,
            "structural_contract_status": str(
                _pick(source, detail, "structural_contract_status") or ""
            ),
            "structural_contract_reason": str(
                _pick(source, detail, "structural_contract_reason") or ""
            ),
            "numerical_similarity_pass": _truth(
                _pick(source, detail, "numerical_similarity_pass")
            ),
            "numerical_similarity_status": str(
                _pick(source, detail, "numerical_similarity_status") or ""
            ),
            "numerical_similarity_reason": str(
                _pick(source, detail, "numerical_similarity_reason") or ""
            ),
            "numerical_similarity_scope": str(
                _pick(source, detail, "numerical_similarity_scope") or ""
            ),
            "numerical_similarity_metric": str(
                _pick(source, detail, "numerical_similarity_metric") or ""
            ),
            "numerical_similarity_value": _num(
                _pick(source, detail, "numerical_similarity_value")
            ),
            "numerical_similarity_threshold": _num(
                _pick(source, detail, "numerical_similarity_threshold")
            ),
            "numerical_similarity_mean_iou": _num(
                _pick(source, detail, "numerical_similarity_mean_iou")
            ),
            "numerical_similarity_mean_iou_threshold": _num(
                _pick(
                    source,
                    detail,
                    "numerical_similarity_mean_iou_threshold",
                )
            ),
            "numerical_similarity_policy_id": str(
                _pick(source, detail, "numerical_similarity_policy_id") or ""
            ),
            "task_quality_pass": _truth(
                _pick(source, detail, "task_quality_pass", "task_valid")
            ),
            "task_quality_status": str(
                _pick(source, detail, "task_quality_status") or ""
            ),
            "task_quality_reason": str(
                _pick(source, detail, "task_quality_reason") or ""
            ),
            # Physical and semantic endpoint identities are both retained.
            # ``comparison_output_endpoint_id`` is the backend-independent
            # primary when a completed-task comparison attestation exists;
            # legacy consumers continue to receive the physical
            # ``output_endpoint_id`` unchanged.
            "output_endpoint_id": output_endpoint_id,
            "physical_output_endpoint_id": physical_output_endpoint_id,
            "comparison_output_endpoint_id": comparison_output_endpoint_id,
            "output_endpoint_match": _truth(detail.get("output_endpoint_match") if detail.get("output_endpoint_match") is not None else source.get("output_endpoint_match")),
            "comparison_endpoint_match": _truth(
                _pick(
                    source,
                    detail,
                    "comparison_endpoint_match",
                    "output_endpoint_match",
                )
            ),
            "output_endpoint_comparison_stratum": _pick(
                source,
                detail,
                "output_endpoint_comparison_stratum",
            ),
            "comparison_stratum_explicit": _truth(
                _pick(source, detail, "comparison_stratum_explicit")
            ),
            "source_e2e_scope": source_e2e_scope,
            "e2e_scope": e2e_scope,
            "e2e_claim_eligible": _truth(
                _pick(
                    source,
                    detail,
                    "e2e_claim_eligible",
                    "claim_eligible_e2e",
                )
            ),
            "e2e_contract_reason": str(
                _pick(source, detail, "e2e_contract_reason") or ""
            ),
            "comparison_endpoint_stratum": str(
                _pick(source, detail, "comparison_endpoint_stratum") or ""
            ),
            "measurement_concurrency": _pick(
                source, detail, "measurement_concurrency"
            ),
            "requires_host_decode_nms": _truth(
                _pick(source, detail, "requires_host_decode_nms")
            ),
            "postprocess_included": _truth(
                _pick(source, detail, "postprocess_included")
            ),
            "postprocess_location": str(
                _pick(source, detail, "postprocess_location") or ""
            ),
            "host_postprocess_frozen": _truth(
                _pick(source, detail, "host_postprocess_frozen")
            ),
            "host_postprocess_required": _truth(
                _pick(
                    source,
                    detail,
                    "host_postprocess_required",
                    "host_tail_required",
                )
            ),
            "host_postprocessing_available": _truth(
                _pick(
                    source,
                    detail,
                    "host_postprocessing_available",
                    "host_tail_available",
                )
            ),
            "host_postprocessing_evidence_status": str(
                _pick(
                    source,
                    detail,
                    "host_postprocessing_evidence_status",
                )
                or ""
            ),
            "host_postprocessing_evidence_source": str(
                _pick(
                    source,
                    detail,
                    "host_postprocessing_evidence_source",
                )
                or ""
            ),
            "host_postprocessing_legacy_alias_conflict": _truth(
                _pick(
                    source,
                    detail,
                    "host_postprocessing_legacy_alias_conflict",
                )
            ),
            "decoder_contract_pass": _truth(
                _pick(source, detail, "decoder_contract_pass")
            ),
            "nms_ok": _truth(_pick(source, detail, "nms_ok")),
            "decoder_id": str(
                _pick(source, detail, "decoder_id") or ""
            ),
            # Keep compatibility aliases losslessly available to older report
            # consumers while exposing the canonical names above.
            "host_tail_required": _truth(
                _pick(
                    source,
                    detail,
                    "host_tail_required",
                    "host_postprocess_required",
                )
            ),
            "host_tail_available": _truth(
                _pick(
                    source,
                    detail,
                    "host_tail_available",
                    "host_postprocessing_available",
                )
            ),
            "postprocess_completed_frames": (
                int(_num(_pick(
                    source, detail, "postprocess_completed_frames",
                )))
                if _num(_pick(
                    source, detail, "postprocess_completed_frames",
                )) is not None
                else None
            ),
            "postprocess_completion_verified": _truth(
                _pick(source, detail, "postprocess_completion_verified")
            ),
            "frozen_host_postprocess_contract": _pick(
                source, detail, "frozen_host_postprocess_contract"
            ),
            "frozen_host_postprocess_contract_sha256": str(
                _pick(
                    source,
                    detail,
                    "frozen_host_postprocess_contract_sha256",
                    "frozen_postprocess_contract_sha256",
                )
                or ""
            ),
            "frozen_host_postprocess_result": _pick(
                source, detail, "frozen_host_postprocess_result"
            ),
            "completed_task_stage": str(
                _pick(source, detail, "completed_task_stage") or ""
            ),
            "completed_task_contract_family": str(
                _pick(
                    source, detail, "completed_task_contract_family",
                )
                or ""
            ),
            "completed_task_endpoint_contract": _pick(
                source, detail, "completed_task_endpoint_contract"
            ),
            "completed_task_endpoint_contract_hash": str(
                _pick(
                    source, detail, "completed_task_endpoint_contract_hash",
                )
                or ""
            ),
            "completed_task_output_endpoint_id": str(
                _pick(
                    source, detail, "completed_task_output_endpoint_id",
                )
                or ""
            ),
            "completed_task_comparison_endpoint_contract": _pick(
                source,
                detail,
                "completed_task_comparison_endpoint_contract",
            ),
            "completed_task_comparison_endpoint_contract_hash": str(
                _pick(
                    source,
                    detail,
                    "completed_task_comparison_endpoint_contract_hash",
                )
                or ""
            ),
            "completed_task_comparison_output_endpoint_id": (
                completed_task_comparison_output_endpoint_id
            ),
            "completed_task_completion_mode": str(
                _pick(
                    source,
                    detail,
                    "completed_task_completion_mode",
                )
                or ""
            ),
            "completed_task_endpoint_attested": _truth(
                _pick(
                    source, detail, "completed_task_endpoint_attested",
                )
            ),
            "completed_task_endpoint_attestation": _pick(
                source, detail, "completed_task_endpoint_attestation"
            ),
            "completed_task_endpoint_attestation_status": str(
                _pick(
                    source,
                    detail,
                    "completed_task_endpoint_attestation_status",
                )
                or ""
            ),
            "runtime_precision_explicit": _truth(detail.get("runtime_precision_explicit") if detail.get("runtime_precision_explicit") is not None else source.get("runtime_precision_explicit")),
            "precision_quality_verified": _truth(detail.get("precision_quality_verified") if detail.get("precision_quality_verified") is not None else source.get("precision_quality_verified")),
            "precision_quality_binding_verified": (
                precision_quality_binding_verified
            ),
            "task_quality_observation_valid": task_quality_observation_valid,
            "quality_claim_result_verified": quality_claim_result_verified,
            "quality_central_evidence_verified": (
                quality_central_evidence_verified
            ),
            "quality_accuracy_gate_pass": quality_accuracy_gate_pass,
            "quality_eligible_for_ranking": quality_eligible_for_ranking,
            "quality_gate_status": quality_gate_status,
            "quality_accuracy_gate_reason": str(
                _pick(source, detail, "quality_accuracy_gate_reason") or ""
            ),
            "quality_ranking_exclusion_reason": str(
                _pick(source, detail, "quality_ranking_exclusion_reason") or ""
            ),
            "repeat_claim_gate_pass": repeat_claim_gate_pass,
            "performance_contract_eligible": performance_contract_eligible,
            "performance_claim_exclusion_reasons": list(exclusion_reasons),
            "claim_ok_source": _truth(
                _pick(source, detail, "claim_ok")
            ),
            "claim_input_source": _truth(
                _pick(source, detail, "claim_ok_source")
            ),
            "claim_structural_gate_pass": _truth(
                _pick(source, detail, "claim_structural_gate_pass")
            ),
            "claim_structural_gate_reason": str(
                _pick(source, detail, "claim_structural_gate_reason") or ""
            ),
            "claim_ok_structural_clamped": _truth(
                _pick(source, detail, "claim_ok_structural_clamped")
            ),
            "quality_physical_evidence_conflict": _truth(
                _pick(
                    source,
                    detail,
                    "quality_physical_evidence_conflict",
                )
            ),
            "quality_physical_evidence_conflict_fields": _pick(
                source,
                detail,
                "quality_physical_evidence_conflict_fields",
            ),
            "input_image": str(source.get("input_image") or detail.get("input_image") or ""),
            "input_image_source": str(source.get("input_image_source") or detail.get("input_image_source") or ""),
            "failure_reason": str(source.get("failure_reason") or detail.get("failure_reason") or ""),
            "expected_matrix_row": identity_key in expected_keys if expected_keys else None,
            "observation_tier": "development_or_screening",
            "source_kind": "native_performance_matrix",
            "source_path": str(concise_path),
            # A Smoke/development matrix is evidence, not an automatic final
            # claim.  Existing protocol/quality gates remain authoritative.
            "claim_eligible": False,
            "performance_eligible": performance_contract_eligible,
            # Ranking admission is deliberately separate from final claim
            # admission. ``screening_only`` can veto a final claim through
            # quality_claim_result_verified/quality_eligible_for_ranking while
            # an exactly bound, centrally verified positive quality result
            # remains valid development-audit evidence.
            "ranking_eligible": ranking_eligible,
            "ranking_eligibility_status": (
                "eligible" if ranking_eligible else "quality_veto"
            ),
            "ranking_exclusion_reasons": ranking_exclusion_reasons,
            "energy_eligible": False,
            "eligibility_status": "development_or_screening",
            "exclusion_reason": (
                "native_development_observation_requires_final_protocol_admission"
                if performance_contract_eligible
                else ";".join(str(reason) for reason in exclusion_reasons)
                or "native_performance_comparison_contract_not_satisfied"
            ),
        })

    observations.sort(key=lambda row: (
        str(row.get("model_id") or ""), str(row.get("setup_id") or ""),
        str(row.get("backend") or ""), str(row.get("case_id") or ""),
    ))
    expected_count = _explicit_count(
        expected,
        "expected_row_count",
        len(expected_rows) or len(observations),
    )
    present_count = _explicit_count(
        expected,
        "present_expected_row_count",
        len(observations),
    )
    missing_count = _explicit_count(
        expected,
        "missing_expected_row_count",
        max(0, expected_count - present_count),
    )
    observed_success_count = sum(
        1 for row in observations if row.get("runtime_executable") is True
    )
    successful_count = int(
        expected.get("successful_expected_row_count")
        if expected.get("successful_expected_row_count") is not None
        else min(present_count, observed_success_count)
    )
    failed_count = int(
        expected.get("failed_expected_row_count")
        if expected.get("failed_expected_row_count") is not None
        else max(0, present_count - successful_count)
    )
    row_presence_complete = bool(
        expected_count and missing_count == 0 and present_count >= expected_count
    )
    execution_success_complete = bool(
        row_presence_complete
        and failed_count == 0
        and successful_count >= expected_count
    )
    comparison_eligible = [
        dict(row) for row in observations
        if row.get("performance_contract_eligible") is True
    ]
    return {
        "schema": "onnx-splitpoint/native-performance-matrix",
        "schema_version": 5,
        "status": ("complete" if execution_success_complete else "complete_with_exclusions"
                   if expected.get("technical_execution_complete") else "partial"),
        "expected_row_count": expected_count,
        "present_expected_row_count": present_count,
        "successful_expected_row_count": successful_count,
        "failed_expected_row_count": failed_count,
        "missing_expected_row_count": missing_count,
        "row_presence_complete": row_presence_complete,
        "execution_success_complete": execution_success_complete,
        "technical_execution_complete": expected.get("technical_execution_complete", execution_success_complete),
        "excluded_expected_row_count": int(expected.get("excluded_expected_row_count") or 0),
        "excluded_expected_rows": list(expected.get("excluded_expected_rows") or []),
        "matrix_complete": execution_success_complete,
        "observation_count": len(observations),
        "comparison_eligible_observation_count": len(comparison_eligible),
        "claim_eligibility_policy": "development_observations_only; final claims require the existing protocol and task-quality gates",
        "source_path": str(concise_path),
        "expected_matrix_source_path": str(expected_path) if expected_path.is_file() else "",
        # Compatibility alias plus explicit scientific tiers.  Consumers can
        # retain every diagnostic row without mistaking it for an admissible
        # comparison observation.
        "observations": observations,
        "all_observations": observations,
        "comparison_eligible_observations": comparison_eligible,
    }


__all__ = ["collect_native_performance_matrix"]
