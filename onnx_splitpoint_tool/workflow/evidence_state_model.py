from __future__ import annotations

"""Layered measurement-state projection used by v2.79.2 reports.

The projection keeps seven questions separate:

1. Is there a raw representation?
2. Which logical measurement does it represent?
3. What is the terminal build/runtime outcome?
4. Is task Quality applicable to the measured endpoint?
5. Did the Quality evaluation complete?
6. What decision did it make?
7. Is the row eligible for a scientific claim?

This prevents a P2-only technical measurement from being reported as a missing
completed-task quality result.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .logical_measurement import (
    annotate_logical_measurements,
    canonical_backend,
    canonical_run_id,
    selected_variant,
    setup_ids,
)


def _token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    token = _token(value)
    if token in {"1", "true", "yes", "ok", "pass", "passed", "completed", "success"}:
        return True
    if token in {"0", "false", "no", "fail", "failed", "error"}:
        return False
    return None


def _actual_endpoint(row: Mapping[str, Any]) -> str:
    return _token(
        row.get("measurement_endpoint")
        or row.get("performance_endpoint")
        or row.get("output_endpoint")
        or row.get("endpoint_mode")
        or row.get("stage")
    )


def _quality_applicability(
    row: Mapping[str, Any], scope: Mapping[str, Any],
) -> str:
    from ..native_job_identity import known_build_exclusion
    if known_build_exclusion(row):
        return "not_applicable"
    # Actual runtime evidence takes precedence over a pre-dispatch
    # ``conditional`` scope declaration.
    explicit_row = _token(row.get("quality_applicability"))
    if explicit_row in {"applicable", "not_applicable"}:
        return explicit_row
    if (
        row.get("quality_not_applicable") is True
        or row.get("technical_validation_only") is True
        or _token(row.get("row_scope")) == "technical_validation_only"
    ):
        return "not_applicable"
    explicit_scope = _token(scope.get("quality_applicability"))
    if explicit_scope in {"applicable", "not_applicable"}:
        return explicit_scope
    # Historical normalized rows record the P2-only contract in the
    # variant-local quality gate.  It is authoritative evidence that no
    # completed-task Quality endpoint existed; classifying it as ``missing``
    # would invent an obligation that the measured endpoint never had.
    variant = selected_variant(row)
    gates = row.get("task_quality_gates_by_variant")
    gate = gates.get(variant) if isinstance(gates, Mapping) else None
    if not isinstance(gate, Mapping):
        candidate = row.get("task_quality_gate")
        gate = candidate if isinstance(candidate, Mapping) else {}
    gate_status = _token(gate.get("status"))
    gate_decision = _token(gate.get("decision"))
    if (
        gate_decision == "not_applicable"
        or gate_status in {
            "not_applicable", "technical_only", "technical_validation_only",
        }
    ):
        return "not_applicable"
    endpoint = _actual_endpoint(row)
    if endpoint in {
        "part2", "p2_output", "raw_model_outputs", "technical_only",
        "technical_validation_only",
    }:
        return "not_applicable"
    if endpoint in {
        "completed_task", "completed_detection", "classification_logits",
        "decoded_nms", "full", "model_outputs",
    }:
        return "applicable"

    variant = _token(row.get("variant") or scope.get("variant"))
    task = _token(row.get("task") or scope.get("task"))
    if variant == "full" or task == "classification":
        return "applicable"
    return "conditional"


def _quality_payload(
    row: Mapping[str, Any], quality_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    if quality_result:
        return quality_result
    variant = selected_variant(row)
    by_variant = row.get("task_quality_gates_by_variant")
    if isinstance(by_variant, Mapping):
        gate = by_variant.get(variant)
        if isinstance(gate, Mapping):
            return gate
    gate = row.get("task_quality_gate")
    if isinstance(gate, Mapping):
        return gate
    return {}


def project_evidence_state(
    row: Mapping[str, Any], *,
    scope: Mapping[str, Any] | None = None,
    quality_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scope = dict(scope or {})
    row = dict(row or {})
    quality = dict(_quality_payload(row, dict(quality_result or {})))

    presence = _token(row.get("presence"))
    raw_representation = (
        "missing"
        if not row or presence in {"missing", "absent"}
        else "present"
    )
    logical_id = str(
        row.get("logical_measurement_id")
        or scope.get("logical_measurement_id")
        or scope.get("logical_identity_sha256")
        or ""
    )

    status = _token(
        row.get("semantic_status")
        or row.get("technical_status")
        or row.get("status")
    )
    timed_out = bool(
        row.get("timed_out")
        or row.get("timeout_kind")
        or status in {"timeout", "timed_out", "hard_timeout", "idle_timeout"}
    )
    unsupported = bool(row.get("unsupported_reason")) or status == "unsupported"
    runtime_ok = _boolean(
        row.get("runtime_ok")
        if row.get("runtime_ok") is not None
        else row.get("execution_ok")
    )
    compile_ok = _boolean(row.get("compile_ok"))
    from ..native_job_identity import known_build_exclusion
    exclusion = known_build_exclusion(row)
    if raw_representation == "missing":
        build_runtime = "missing"
    elif timed_out:
        build_runtime = "timeout"
    elif unsupported:
        build_runtime = "unsupported"
    elif exclusion or compile_ok is False or status in {"build_failed", "compile_failed"}:
        build_runtime = "build_failed"
    elif runtime_ok is False or status in {"runtime_failed", "failed", "error"}:
        build_runtime = "runtime_failed"
    elif runtime_ok is True or compile_ok is True or status in {
        "ok", "success", "completed", "pass", "failed_quality",
    }:
        build_runtime = "completed"
    else:
        build_runtime = status or "unknown"

    applicability = _quality_applicability(row, scope)
    if applicability == "not_applicable":
        completion = "not_applicable"
        decision = "not_applicable"
    else:
        technical = _token(
            quality.get("technical_status")
            or row.get("central_quality_technical_status")
            or row.get("quality_technical_status")
        )
        gate_decision = ""
        for value in (
            quality.get("decision"), quality.get("scientific_status"),
            quality.get("status"), row.get("central_quality_decision"),
            row.get("scientific_status"), row.get("quality_decision"),
        ):
            token = _token(value)
            if token in {"pass", "fail", "inconclusive", "unavailable", "not_evaluated"}:
                gate_decision = token
                break
        if technical == "completed" or gate_decision in {"pass", "fail", "inconclusive"}:
            completion = "completed"
        elif build_runtime in {
            "build_failed", "runtime_failed", "timeout", "unsupported",
        }:
            completion = "blocked"
        else:
            completion = "missing"
        decision = gate_decision or (
            "not_evaluated" if completion == "completed" else "unavailable"
        )

    claim = bool(
        row.get("claim_eligible") is True
        or row.get("scientific_claim_eligible") is True
        or (
            isinstance(row.get("row_eligibility"), Mapping)
            and row.get("row_eligibility", {}).get("claim_eligible") is True
        )
    )
    if completion != "completed" or decision != "pass":
        claim = False

    return {
        "raw_representation": raw_representation,
        "logical_measurement_id": logical_id,
        "build_runtime_state": build_runtime,
        "quality_applicability": applicability,
        "quality_completion": completion,
        "quality_decision": decision,
        "claim_eligible": claim,
        "measurement_endpoint": _actual_endpoint(row),
        **({
            "build_exclusion": exclusion,
            "upstream_evidence_path": str(row.get("upstream_evidence_path") or ""),
        } if exclusion else {}),
    }


def summarize_evidence_states(
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    applicability = Counter(_token(s.get("quality_applicability")) for s in states)
    completion = Counter(_token(s.get("quality_completion")) for s in states)
    decisions = Counter(_token(s.get("quality_decision")) for s in states)
    build_runtime = Counter(_token(s.get("build_runtime_state")) for s in states)
    return {
        "matrix_required": len(states),
        "matrix_present": sum(
            s.get("raw_representation") == "present" for s in states
        ),
        "quality_applicable": sum(
            s.get("quality_applicability") != "not_applicable" for s in states
        ),
        "quality_completed": sum(
            s.get("quality_completion") == "completed" for s in states
        ),
        "quality_blocked": sum(
            s.get("quality_completion") == "blocked" for s in states
        ),
        "quality_not_applicable": sum(
            s.get("quality_applicability") == "not_applicable" for s in states
        ),
        "quality_missing": sum(
            s.get("quality_completion") == "missing" for s in states
        ),
        "companions": 0,
        "claim_eligible": sum(bool(s.get("claim_eligible")) for s in states),
        "applicability_counts": dict(sorted(applicability.items())),
        "completion_counts": dict(sorted(completion.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "build_runtime_counts": dict(sorted(build_runtime.items())),
    }


def _scope_match(
    row: Mapping[str, Any], entry: Mapping[str, Any],
) -> bool:
    if _token(row.get("model_id")) != _token(entry.get("model_id")):
        return False
    if _token(row.get("case_id") or "full") != _token(entry.get("case_id") or "full"):
        return False
    if canonical_backend(row.get("backend")) != canonical_backend(entry.get("backend")):
        return False
    if selected_variant(row) != selected_variant(entry):
        return False
    expected_run = canonical_run_id(entry.get("run_id"))
    observed_run = canonical_run_id(
        row.get("quality_source_run_id")
        or row.get("source_run_id")
        or row.get("run_id")
    )
    if expected_run and observed_run and expected_run != observed_run:
        return False
    expected_setup = _token(entry.get("expected_setup_id") or entry.get("setup_id"))
    if expected_setup:
        direct = setup_ids(row)
        mirrors = {_token(value) for value in list(row.get("mirror_setup_ids") or [])}
        if direct == [expected_setup]:
            return True
        return bool(
            not direct
            and row.get("mirror_provenance_verified") is True
            and expected_setup in mirrors
        )
    return True


def _select_scope_representation(
    candidates: Sequence[Mapping[str, Any]], entry: Mapping[str, Any],
) -> tuple[dict[str, Any], bool, int]:
    if not candidates:
        return {}, False, 0
    annotated = annotate_logical_measurements(candidates)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in annotated:
        groups.setdefault(str(row.get("logical_measurement_id") or ""), []).append(row)
    groups.pop("", None)
    if len(groups) != 1:
        return {}, len(groups) > 1, len(candidates)
    group = next(iter(groups.values()))
    expected_setup = _token(entry.get("expected_setup_id") or entry.get("setup_id"))
    eligible = []
    for row in group:
        if expected_setup and not _scope_match(row, entry):
            continue
        eligible.append(row)
    if not eligible:
        return {}, False, len(candidates)
    eligible.sort(key=lambda row: (
        0 if row.get("logical_measurement_primary") is True else 1,
        0 if setup_ids(row) else 1,
        str(row.get("source_path") or ""),
    ))
    return dict(eligible[0]), False, len(candidates)


def summarize_run_scope(run_dir: Any) -> dict[str, Any]:
    """Project sealed per-model scope identities onto normalized result rows."""

    import json

    root = Path(run_dir)
    states: list[dict[str, Any]] = []
    for scope_path in sorted(
        root.glob("models/*/benchmark_set/required_run_scope.json")
    ):
        try:
            scope_payload = json.loads(scope_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_id = str(
            scope_payload.get("model_id") or scope_path.parents[1].name
        )
        normalized_path = (
            root / "models" / model_id / "benchmark_results"
            / "normalized_results.json"
        )
        try:
            normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
        except Exception:
            normalized = {}
        raw_rows = [
            dict(row)
            for row in list(normalized.get("results") or [])
            if isinstance(row, Mapping)
        ]
        rows = annotate_logical_measurements(raw_rows)
        # The build stage already owns exact recipe/endpoint exclusion proof.
        # Reuse its verifier and role-conflict checks for missing composed
        # requests. A measured P2 remains an independent observation.
        try:
            build_stage = json.loads((
                root / "models" / model_id
                / "stages/build_backend_artifacts/stage_result.json"
            ).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            build_stage = {}
        readiness = (build_stage.get("details") or {}).get("deferred_build_readiness") or {}
        try:
            required_matrix = json.loads((normalized_path.parent / "required_profile_matrix.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            required_matrix = {}
        for entry in list(scope_payload.get("identities") or []):
            if not isinstance(entry, Mapping):
                continue
            candidates = [row for row in rows if _scope_match(row, entry)]
            selected, ambiguous, representation_count = _select_scope_representation(
                candidates, entry,
            )
            if not selected and not ambiguous:
                from ..native_job_identity import required_profile_build_exclusions, known_build_exclusion
                stored = [item for item in required_matrix.get("excluded_results", [])
                          if isinstance(item, Mapping) and _scope_match(item, entry)
                          and item.get("logical_identity_sha256") == entry.get("logical_identity_sha256")
                          and known_build_exclusion(item)]
                excluded = required_profile_build_exclusions(
                    stored if len(stored) == 1 else [],
                    readiness, rows,
                )
                if len(excluded) == 1:
                    selected = excluded[0]
            state = project_evidence_state(selected, scope=entry)
            state.update({
                "model_id": _token(entry.get("model_id") or model_id),
                "case_id": _token(entry.get("case_id") or "full"),
                "run_id": canonical_run_id(entry.get("run_id")),
                "backend": canonical_backend(entry.get("backend")),
                "setup_id": str(entry.get("expected_setup_id") or entry.get("setup_id") or ""),
                "variant": selected_variant(entry),
                "scope_path": str(scope_path),
                "scope_identity_sha256": str(
                    entry.get("logical_identity_sha256") or ""
                ),
                "representation_count": representation_count,
                "representation_ambiguous": ambiguous,
            })
            states.append(state)
    summary = summarize_evidence_states(states)
    summary["states"] = states
    return summary
