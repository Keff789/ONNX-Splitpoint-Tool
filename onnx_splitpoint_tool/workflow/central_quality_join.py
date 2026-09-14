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
