from __future__ import annotations

"""Conservative Central-Quality joins for v2.79.2.

The primary join key is the exact source-request SHA together with logical
measurement identity.  Direct setup representations outrank setup-less mirrors;
a mirror is eligible only when its provenance relation was explicitly verified.
Companion results are never allowed to fill the primary measurement matrix.
"""

from typing import Any, Callable, Mapping, Sequence

from .logical_measurement import (
    canonical_backend,
    canonical_run_id,
    direct_setup_ids,
    request_sha,
    selected_variant,
)


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _sha(value: Any) -> str:
    token = _token(value)
    if token.startswith("sha256:"):
        token = token[7:]
    return (
        token
        if len(token) == 64
        and all(character in "0123456789abcdef" for character in token)
        else ""
    )


def _canonical_variant(value: Any) -> str:
    token = _token(value)
    return "composed" if token in {"split", "complete"} else token


def _result_request_sha(
    result: Mapping[str, Any], result_identity: Mapping[str, Any],
) -> str:
    for value in (
        result_identity.get("source_request_sha256"),
        result.get("source_request_sha256"),
        result.get("request_sha256"),
    ):
        token = _sha(value)
        if token:
            return token
    return ""


def _permitted_full_cases(row: Mapping[str, Any], row_case: str) -> set[str]:
    permitted = {row_case} if row_case else set()
    if row_case == "full":
        for value in (
            row.get("source_case_id"), row.get("original_case_id"),
        ):
            token = _token(value)
            if token:
                permitted.add(token)
        raw = row.get("full_source_case_ids")
        if isinstance(raw, (str, bytes, bytearray)):
            raw = [raw]
        for value in list(raw or []):
            token = _token(value)
            if token:
                permitted.add(token)
    return permitted


def exact_request_sha_candidates(
    *,
    rows: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    variant: str,
    row_identity_getter: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
    contract_matcher: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
    primary_variant: bool = True,
) -> list[int]:
    """Return only the best exactly request-bound representation candidates."""

    observed_sha = _result_request_sha(result, result_identity)
    if not observed_sha:
        return []
    result_setup = _token(result_identity.get("setup_id") or result.get("setup_id"))
    result_model = _token(
        result_identity.get("model_id") or result.get("model_id")
    )
    result_run = canonical_run_id(
        result_identity.get("source_run_id")
        or result.get("source_run_id")
        or result.get("run_id")
    )
    result_case = _token(
        result_identity.get("case_id") or result.get("case_id")
    )
    result_task = _token(
        result_identity.get("task") or result.get("task")
    )
    wanted_variant = _canonical_variant(variant)

    scored: list[tuple[int, int]] = []
    for index, row in enumerate(rows):
        row_variant = selected_variant(row)
        if primary_variant:
            if wanted_variant == "full" and row_variant != "full":
                continue
            if wanted_variant == "composed" and row_variant == "full":
                continue
        identity = dict(row_identity_getter(row, variant) or {})
        expected_sha = _sha(
            identity.get("source_request_sha256") or request_sha(row, variant)
        )
        if expected_sha != observed_sha:
            continue
        if result_model and _token(
            identity.get("model_id") or row.get("model_id")
        ) != result_model:
            continue
        row_case = _token(identity.get("case_id") or row.get("case_id"))
        if result_case and result_case not in _permitted_full_cases(row, row_case):
            continue
        row_run = canonical_run_id(
            identity.get("source_run_id")
            or row.get("quality_source_run_id")
            or row.get("source_run_id")
            or row.get("run_id")
        )
        if result_run and row_run != result_run:
            continue
        if result_task and _token(
            identity.get("task") or row.get("task")
        ) not in {"", result_task}:
            continue
        if not contract_matcher(identity, result_identity):
            continue

        direct_setups = direct_setup_ids({**row, **identity})
        direct = direct_setups[0] if len(direct_setups) == 1 else ""
        mirrors = {
            _token(value)
            for value in list(row.get("mirror_setup_ids") or [])
            if _token(value)
        }
        proven_mirror = bool(row.get("mirror_provenance_verified") is True)
        primary = bool(row.get("logical_measurement_primary") is True)
        if result_setup and direct == result_setup:
            score = 0 if primary else 1
        elif result_setup and not direct and proven_mirror and result_setup in mirrors:
            score = 2
        elif not result_setup and direct:
            score = 3 if primary else 4
        elif not result_setup and not direct and proven_mirror:
            score = 5
        else:
            continue
        scored.append((score, index))
    if not scored:
        return []
    best = min(score for score, _ in scored)
    return [index for score, index in scored if score == best]


def is_companion_result(result: Mapping[str, Any]) -> bool:
    role = _token(
        result.get("execution_role")
        or result.get("result_class")
        or result.get("source_kind")
    )

    run_id = canonical_run_id(
        result.get("source_run_id") or result.get("run_id")
    )
    return bool(
        role in {
            "full_quality_only", "summary_only", "native_full_companion",
            "companion",
        }
        or run_id == "native_full_tensorrt"
        or result.get("performance_claims_emitted") is False
        and role == "full_quality_only"
    )


def cancelled_request_identity(result, *, run_id, cancellation, run_dir=None):
    """Bind a terminal cancellation without asserting successful execution.

    An independently recorded run cancellation and its original observation
    interval are required. Missing success-only producer fields do not turn
    an abandoned calculation into a new runtime failure.
    """
    context = result.get("cancel_context") or {}
    if not isinstance(context, Mapping) or not isinstance(cancellation, Mapping):
        return None
    if (result.get("status") != "cancelled" or result.get("technical_status") != "cancelled"
            or context.get("run_id") != run_id or cancellation.get("run_id") != run_id
            or not context.get("reason") or context.get("reason") != cancellation.get("reason")
            or context.get("requested_at") != cancellation.get("requested_at")
            or context.get("requested_monotonic") != cancellation.get("requested_monotonic")
            or result.get("exception_type") not in {"CancelledError", "QualityServiceClosedError"}):
        return None
    try:
        if not 0 < float(context["requested_monotonic"]) <= float(context["shutdown_monotonic"]) <= float(result["failure_observed_monotonic"]):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    nested = result.get("request_identity") or {}
    if not isinstance(nested, Mapping):
        return None
    if result.get("error") and str(result["error"]).split(":", 1)[0] != result["exception_type"]:
        return None
    identity = dict(nested)
    for name in ("eval_run_id", "model_id", "setup_id", "source_run_id", "case_id", "variant", "task", "source_request_sha256"):
        values = [result.get(name), nested.get(name)]
        if name == "eval_run_id":
            values.append(result.get("collection_eval_run_id"))
        canonical = _sha if name == "source_request_sha256" else str
        tokens = {canonical(value) for value in values if value not in (None, "")}
        if len(tokens) != 1 or not next(iter(tokens)):
            return None
        identity[name] = next(iter(tokens))
    if identity["eval_run_id"] != run_id:
        return None
    expected_reason = "user_cancelled" if result["exception_type"] == "CancelledError" else "service_closed_after_cancel"
    if result.get("completion_reason") != expected_reason:
        return None
    for alias, field, canonical in (("source_setup_id", "setup_id", _token),
                                     ("run_id", "source_run_id", canonical_run_id)):
        if result.get(alias) and canonical(result[alias]) != canonical(identity[field]):
            return None
    if is_companion_result(result):
        # Companions have no measured primary row to bind their SHA. Verify
        # their registered local request manifest, without loading predictions
        # or requiring successful execution/completion attestations.
        import hashlib
        import json
        from pathlib import Path
        if run_dir is None:
            return None
        root = Path(run_dir).resolve()
        path = root / str(result.get("source_request") or "")
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                return None
            encoded = path.read_bytes()
            if hashlib.sha256(encoded).hexdigest() != identity["source_request_sha256"]:
                return None
            request = json.loads(encoded)
        except (OSError, ValueError):
            return None
        if not isinstance(request, Mapping):
            return None
        for name in ("eval_run_id", "model_id", "setup_id", "source_run_id", "case_id", "variant", "task"):
            canonical = canonical_run_id if name == "source_run_id" else str
            if canonical(request.get(name) or "") != canonical(identity[name]):
                return None
    identity["producer_binding_eligible"] = False
    identity["cancellation_only"] = True
    return identity


def cancelled_row_identity(row, variant):
    """Retain the original request identity after an earlier normal join."""
    identities = row.get("quality_request_identities_by_variant") or {}
    identity = identities.get(variant) or {}
    if not identity and selected_variant(row) == variant:
        identity = row.get("central_quality_request_identity") or {}
    return dict(identity)


def cancelled_contract_matches(expected, observed):
    # exact_request_sha_candidates also checks physical setup, model, case,
    # run and variant. Success-only completion fields are intentionally absent.
    return all(not expected.get(name) or str(expected[name]) == str(observed.get(name) or "")
               for name in ("eval_run_id", "model_id", "setup_id", "source_run_id", "case_id", "variant", "task"))


def join_quality_results_by_request_sha(
    *,
    rows: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Generic read-only join used by the v2.79.2 reconciliation tool.

    This helper does not replace the normal runner's stronger contract matcher;
    it provides a conservative portable replay when only normalized rows and
    Central Quality result identities are available.
    """

    matched_rows: set[tuple[int, str]] = set()
    joins: list[dict[str, Any]] = []
    companions: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for result_index, result in enumerate(results):
        if is_companion_result(result):
            companions.append({"result_index": result_index, "result": dict(result)})
            continue
        variant = _canonical_variant(result.get("variant") or "full")
        observed_sha = _result_request_sha(result, result)
        candidates: list[tuple[int, int]] = []
        for row_index, row in enumerate(rows):
            if (row_index, variant) in matched_rows:
                continue
            if request_sha(row, variant) != observed_sha or not observed_sha:
                continue
            if _token(row.get("model_id")) != _token(result.get("model_id")):
                continue
            if _canonical_variant(selected_variant(row)) != variant:
                continue
            result_run = canonical_run_id(
                result.get("source_run_id") or result.get("run_id")
            )
            row_run = canonical_run_id(
                row.get("quality_source_run_id")
                or row.get("source_run_id")
                or row.get("run_id")
            )
            if result_run and row_run and result_run != row_run:
                continue
            result_backend = canonical_backend(
                result.get("backend")
                or result.get("comparison_backend")
                or result.get("producer_backend")
            )
            row_backend = canonical_backend(
                row.get("backend")
                or row.get("comparison_backend")
                or row.get("producer_backend")
            )
            if result_backend and row_backend and result_backend != row_backend:
                continue
            result_case = _token(result.get("case_id") or "full")
            row_case = _token(row.get("case_id") or "full")
            if result_case not in _permitted_full_cases(row, row_case):
                continue
            result_setup = _token(result.get("setup_id"))
            direct = direct_setup_ids(row)
            direct_setup = direct[0] if len(direct) == 1 else ""
            mirrors = {_token(value) for value in list(row.get("mirror_setup_ids") or [])}
            if result_setup and direct_setup == result_setup:
                rank = 0 if row.get("logical_measurement_primary") is True else 1
            elif (
                result_setup
                and not direct_setup
                and row.get("mirror_provenance_verified") is True
                and result_setup in mirrors
            ):
                rank = 2
            elif not result_setup and direct_setup:
                rank = 3
            else:
                continue
            candidates.append((rank, row_index))
        if candidates:
            best = min(rank for rank, _ in candidates)
            selected = [idx for rank, idx in candidates if rank == best]
        else:
            selected = []
        detail = {
            "result_index": result_index,
            "source_request_sha256": observed_sha,
            "candidate_indices": selected,
            "candidate_count": len(selected),
        }
        if len(selected) == 1:
            row_index = selected[0]
            matched_rows.add((row_index, variant))
            joins.append({**detail, "status": "matched", "row_index": row_index})
        elif len(selected) > 1:
            ambiguous.append({**detail, "status": "ambiguous"})
        else:
            unmatched.append({**detail, "status": "unmatched"})

    return {
        "joins": joins,
        "companions": companions,
        "unmatched_results": unmatched,
        "ambiguous_results": ambiguous,
        "matched_primary_count": len(joins),
        "companion_count": len(companions),
        "unmatched_count": len(unmatched),
        "ambiguous_count": len(ambiguous),
    }


def apply_central_quality_result_to_row(row, result, result_identity, *, variant,
                                        match_mode, match_role, result_cancelled, smoke_quality):
    """Apply exactly the normal runner's existing per-row Quality projection.

    Identity selection, evaluation, file writing and counters stay with the
    normal controller. This extracted block is also replayed read-only to
    verify source-bound technical provenance after a later normal join.
    """
    import copy
    from .artifacts import sha256_json
    from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
    row["central_quality_join_mode"] = match_mode
    row["central_quality_join_source_request_sha256"] = str(
        result_identity.get("source_request_sha256") or ""
    )
    result_completed = (
        str(result.get("status") or "") == "completed"
        and str(result.get("technical_status") or "completed") == "completed"
    )
    if result_completed:
        gate = {
            "schema": "onnx-splitpoint/task-quality-gate",
            "schema_version": 2,
            "task": str(result.get("task") or row.get("task") or ""),
            "variant": variant,
            "tier": str((row.get("task_quality_policy") or {}).get("dataset_tier") or "screening")
            if isinstance(row.get("task_quality_policy"), Mapping) else "screening",
            "canonical_reference": "management_cpu_ort_full_onnx",
            "decision": str(result.get("decision") or "unavailable"),
            "status": str(result.get("decision") or "unavailable"),
            "primary": dict(result.get("primary") or {}),
            "guardrails": dict(result.get("guardrails") or {}),
            "n": int(result.get("n") or 0),
            "execution_location": "central_management",
            "management_result_sha256": sha256_json(result),
            "evaluation_fingerprint": str(result.get("evaluation_fingerprint") or ""),
            "reference_identity": str(result.get("reference_identity") or ""),
            "technical_status": str(result.get("technical_status") or "completed"),
            "scientific_status": str(result.get("scientific_status") or result.get("decision") or ""),
        }
        for reporting_field in ("accuracy_assessment", "reporting_policy", "secondary_accuracy_assessments", "accuracy_warnings", "legacy_decision"):
            if reporting_field in result:
                gate[reporting_field] = copy.deepcopy(result[reporting_field])
    else:
        gate = {
            "schema": "onnx-splitpoint/task-quality-gate",
            "schema_version": 2,
            "task": str(result.get("task") or row.get("task") or ""),
            "variant": variant,
            "tier": str((row.get("task_quality_policy") or {}).get("dataset_tier") or "screening")
            if isinstance(row.get("task_quality_policy"), Mapping) else "screening",
            "canonical_reference": "management_cpu_ort_full_onnx",
            "decision": "unavailable",
            "status": "unavailable",
            "n": 0,
            "execution_location": "central_management",
            "technical_status": "cancelled" if result_cancelled else "failed",
            "scientific_status": "unavailable",
            "reason": str(result.get("completion_reason")) if result_cancelled else "central_quality_evaluation_failed",
            "error": str(result.get("error") or "central quality evaluation failed"),
        }
    canonical_request_identity = {
        "schema": "onnx-splitpoint/central-quality-request-identity",
        "schema_version": 2,
        **result_identity,
    }
    gate["source_request_sha256"] = str(result_identity.get("source_request_sha256") or "")
    gate["quality_request_identity"] = canonical_request_identity
    by_variant = dict(row.get("task_quality_gates_by_variant") or {})
    by_variant[variant] = gate
    row["task_quality_gates_by_variant"] = by_variant
    identities_by_variant = dict(row.get("quality_request_identities_by_variant") or {})
    identities_by_variant[variant] = canonical_request_identity
    row["quality_request_identities_by_variant"] = identities_by_variant
    row_setup_ids = {
        str(value).strip().lower()
        for value in list(row.get("quality_source_setup_ids") or [])
        if str(value).strip()
    }
    if result_identity.get("setup_id"):
        row_setup_ids.add(str(result_identity["setup_id"]).strip().lower())
    row["quality_source_setup_ids"] = sorted(row_setup_ids)
    row["task_quality_policy_sha256"] = str(result.get("policy_sha256") or row.get("task_quality_policy_sha256") or "")
    row["runtime_quality_gate_policy_sha256"] = row["task_quality_policy_sha256"]
    row["quality_execution_location"] = "central_management"
    if match_role == "primary":
        row["task_quality_gate"] = gate
        row["quality_evaluation_pending"] = False
        # Benchmark completion belongs to the measured accelerator
        # row and is independent from a failed semantic evaluator.
        row["central_quality_technical_status"] = str(
            result.get("technical_status") or ("completed" if result_completed else "failed")
        )
        row["scientific_status"] = gate["decision"]
        row["central_quality_result_path"] = str(result.get("source_request") or "")
        row["central_quality_source_request_sha256"] = str(result_identity.get("source_request_sha256") or "")
        row["central_quality_request_identity"] = canonical_request_identity
        policy = row.get("task_quality_policy") if isinstance(row.get("task_quality_policy"), Mapping) else None
        try:
            apply_accuracy_gate_to_row(row, policy)
        except Exception as exc:
            row["central_quality_gate_application_error"] = f"{type(exc).__name__}: {exc}"
        if smoke_quality:
            metric_miss = str(gate.get("decision") or "").lower() in {
                "fail", "failed",
            }
            row.update({
                "diagnostic_only": True,
                "claim_eligible": False,
                "scientific_claim_eligible": False,
                "performance_claim_eligible": False,
                "energy_efficiency_claim_eligible": False,
                "eligible_for_ranking": False,
                "ranking_eligible": False,
                "performance_eligible": False,
                "energy_eligible": False,
                "pareto_eligible": False,
                "thesis_valid": False,
                "metric_threshold_enforced": False,
                "metric_threshold_warning": bool(metric_miss),
                "quality_enforcement_mode": "smoke_diagnostic",
            })
            if metric_miss:
                row["accuracy_gate_reason"] = "smoke_metric_threshold_warning"
                row["ranking_exclusion_reason"] = "smoke_diagnostic_only"
                row["exclusion_reason"] = "smoke_diagnostic_only"
    else:
        supplemental = dict(row.get("central_quality_supplemental_results") or {})
        supplemental[variant] = {
            "status": "completed" if result_completed else "cancelled" if result_cancelled else "failed",
            "source_request": str(result.get("source_request") or ""),
            "source_request_sha256": str(result_identity.get("source_request_sha256") or ""),
            "request_identity": canonical_request_identity,
            "management_result_sha256": sha256_json(result),
        }
        row["central_quality_supplemental_results"] = supplemental
    return result_completed
