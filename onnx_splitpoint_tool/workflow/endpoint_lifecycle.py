"""Evidence-preserving endpoint lifecycle projection.

The scientific reporter needs to account for every prospectively planned
performance endpoint, including endpoints that never produced a measurement.
This module deliberately does not discover or execute hardware work.  Its only
job is to join explicit plan, generation, result, and terminal evidence into a
fail-closed ledger.

An absent result is never interpreted as a failure and never synthesized into
a measurement.  A non-measured endpoint is terminal only when an explicit
terminal event (for example a generator rejection or runtime failure) supplies
a concrete reason.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_IDENTITY_FIELDS = (
    "model_id",
    "case_id",
    "setup_id",
    "backend",
    "direction",
    "precision",
    "endpoint_contract_hash",
)

_ALIASES = {
    "model_id": ("model_id", "model", "model_name"),
    "case_id": ("case_id", "candidate_id", "split_id", "run_id"),
    "setup_id": ("setup_id", "hardware_setup_id", "target_setup_id"),
    "backend": ("backend", "producer_backend", "runtime_backend"),
    "direction": ("direction", "split_direction", "execution_direction"),
    "precision": ("precision", "runtime_precision_identity"),
    "endpoint_contract_hash": (
        "endpoint_contract_hash",
        "endpoint_hash",
        "contract_hash",
    ),
}

_MEASUREMENT_FIELDS = (
    "latency_ms",
    "total_latency_ms",
    "pipeline_cycle_selected_ms",
    "throughput_fps",
    "throughput_primary_fps",
    "pipeline_fps_selected",
    "heterogeneous_pipeline_fps",
)

_REASON_FIELDS = (
    "terminal_reason",
    "failure_reason",
    "reject_reason",
    "exclude_reason",
    "error_detail",
    "error_class",
    "reason",
    "detail",
)

_TERMINAL_FAILURE_STATUSES = {
    "failed",
    "failure",
    "error",
    "runtime_failed",
    "compile_failed",
    "build_failed",
    "validation_failed",
    "measurement_failed",
    "rejected",
    "rejected_by_benchmark_generator",
    "unsupported",
    "unsupported_op",
    "resource_infeasible",
    "missing_artifact",
    "timeout",
    "cancelled",
}

_MATERIALIZED_STATUSES = {
    "accepted",
    "accepted_by_benchmark_generator",
    "accepted_split_exported",
    "built",
    "compiled",
    "complete",
    "completed",
    "materialized",
    "ok",
    "ready",
    "runtime_failed",
    "validation_failed",
    "measurement_failed",
}


# Only recipes whose row expansion is explicitly defined by the Generic
# runner are admitted.  Similar-looking unknown IDs are not guessed.
_GENERIC_RUN_PROFILE_SPECS: Dict[str, Dict[str, Any]] = {
    "ort_cpu": {
        "kind": "reference",
        "backend": "ort_cpu",
        "direction": "ort_cpu",
        "aliases": {"ort_cpu", "cpu_ort", "cpu"},
    },
    "ort_cuda": {
        "kind": "reference",
        "backend": "ort_cuda",
        "direction": "ort_cuda",
        "aliases": {"ort_cuda", "cuda_ort", "cuda"},
    },
    "ort_tensorrt": {
        "kind": "reference",
        "backend": "ort_tensorrt",
        "direction": "ort_tensorrt",
        "aliases": {"ort_tensorrt", "tensorrt", "trt"},
    },
    "hailo8": {
        "kind": "full",
        "backend": "hailo8",
        "direction": "hailo8",
        "aliases": {"hailo8", "hailo8_full"},
    },
    "hailo8_to_trt": {
        "kind": "split",
        "backend": "hailo8_to_trt",
        "direction": "hailo8_to_trt",
        "aliases": {"hailo8_to_trt", "hailo8_to_tensorrt"},
    },
    "hailo10": {
        "kind": "full",
        "backend": "hailo10",
        "direction": "hailo10",
        "aliases": {"hailo10", "hailo10h", "hailo10_full", "hailo10h_full"},
    },
    "hailo10_to_tensorrt": {
        "kind": "split",
        "backend": "hailo10_to_tensorrt",
        "direction": "hailo10_to_tensorrt",
        "aliases": {
            "hailo10_to_trt",
            "hailo10_to_tensorrt",
            "hailo10h_to_trt",
            "hailo10h_to_tensorrt",
        },
    },
    "deepx_m1_full": {
        "kind": "full",
        "backend": "deepx_m1",
        "direction": "deepx_m1",
        "aliases": {"deepx_m1_full", "deepx_full", "deepx_m1"},
    },
    "deepx_m1_to_tensorrt": {
        "kind": "split",
        "backend": "deepx_m1_to_tensorrt",
        "direction": "deepx_m1_to_tensorrt",
        "aliases": {
            "deepx_m1_to_tensorrt",
            "deepx_m1_to_trt",
            "deepx_to_tensorrt",
            "deepx_to_trt",
        },
    },
}

_GENERIC_RUN_PROFILE_ALIASES = {
    alias: canonical
    for canonical, spec in _GENERIC_RUN_PROFILE_SPECS.items()
    for alias in spec["aliases"]
}


def _text(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _first(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return ""


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    if text in {"1", "true", "yes", "ok", "pass", "passed"}:
        return True
    if text in {"0", "false", "no", "fail", "failed"}:
        return False
    return None


def _number(value: Any) -> float | None:
    try:
        if value in (None, "") or isinstance(value, bool):
            return None
        number = float(value)
        # Avoid importing a scientific dependency merely for finite checking.
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError, OverflowError):
        return None


def _reason_tokens(value: Any) -> List[str]:
    """Normalize archived JSON/list reason surfaces without literal ``[]``."""

    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                decoded = json.loads(text)
            except (TypeError, ValueError):
                decoded = None
            if isinstance(decoded, Sequence) and not isinstance(
                decoded, (str, bytes)
            ):
                return _reason_tokens(decoded)
        return [token.strip() for token in text.split(";") if token.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            token
            for item in value
            for token in _reason_tokens(item)
            if token
        ]
    text = str(value).strip()
    return [text] if text else []


def _identity_values(row: Mapping[str, Any]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for canonical in _IDENTITY_FIELDS:
        raw = _first(row, _ALIASES[canonical])
        if canonical == "precision" and isinstance(raw, Mapping):
            raw = (
                raw.get("precision")
                or raw.get("dtype")
                or raw.get("mode")
                or raw.get("name")
                or ""
            )
        values[canonical] = _text(raw)
    return values


def endpoint_identity(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """Return a stable join key without hashing or inventing identity.

    A producer supplied ``endpoint_id`` is authoritative.  Otherwise the key is
    the normalized explicit identity tuple.  Model and case are mandatory in
    the fallback because joining anonymous evidence would silently conflate
    unrelated endpoints.
    """

    explicit = _text(row.get("endpoint_id"))
    if explicit:
        return ("endpoint_id", explicit)
    values = _identity_values(row)
    if not values["model_id"] or not values["case_id"]:
        raise ValueError(
            "Endpoint evidence requires endpoint_id or both model_id and case_id."
        )
    return ("identity",) + tuple(values[field] for field in _IDENTITY_FIELDS)


def _status_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for field in ("status", "row_status", "generation_status", "build_status"):
        value = _text(row.get(field)).lower()
        if value:
            tokens.add(value)
    return tokens


def _has_measurement(row: Mapping[str, Any]) -> bool:
    """Require a real numeric performance value; flags alone are insufficient."""

    numeric = _has_measurement_payload(row)
    if not numeric:
        return False
    explicit = _bool(row.get("measured"))
    measurement_ok = _bool(row.get("measurement_ok"))
    if explicit is False or measurement_ok is False:
        return False
    if _bool(row.get("measurement_valid")) is False:
        return False
    return True


def _has_measurement_payload(row: Mapping[str, Any]) -> bool:
    """Return whether a numeric payload exists, independent of its validity."""

    return any(
        _number(row.get(field)) is not None for field in _MEASUREMENT_FIELDS
    )


def _is_materialized(row: Mapping[str, Any]) -> bool:
    explicit = _bool(row.get("materialized"))
    if explicit is not None:
        return explicit
    if _has_measurement(row):
        return True
    return bool(_status_tokens(row) & _MATERIALIZED_STATUSES)


def _concrete_reason(row: Mapping[str, Any]) -> str:
    for field in _REASON_FIELDS:
        value = _text(row.get(field))
        if value:
            return value
    # Specific terminal status codes are useful causal classes.  Generic
    # ``failed``/``error``/``rejected`` labels are not concrete enough and must
    # not close the endpoint without additional evidence.
    specific_statuses = _status_tokens(row) & (
        _TERMINAL_FAILURE_STATUSES
        - {"failed", "failure", "error", "rejected", "cancelled"}
    )
    if specific_statuses:
        return "status:" + sorted(specific_statuses)[0]
    return ""


def _terminal_failure(row: Mapping[str, Any]) -> bool:
    explicit = _bool(row.get("terminal"))
    if explicit is not None:
        return explicit
    if _bool(row.get("complete")) is True and _bool(row.get("measurement_ok")) is False:
        return True
    return bool(_status_tokens(row) & _TERMINAL_FAILURE_STATUSES)


def _index_evidence(
    rows: Sequence[Mapping[str, Any]] | None,
) -> tuple[Dict[Tuple[str, ...], List[Dict[str, Any]]], List[Dict[str, Any]]]:
    indexed: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    invalid: List[Dict[str, Any]] = []
    for raw in rows or ():
        if not isinstance(raw, Mapping):
            invalid.append({"reason": "not_a_mapping", "evidence": raw})
            continue
        row = dict(raw)
        try:
            key = endpoint_identity(row)
        except ValueError as exc:
            invalid.append({"reason": str(exc), "evidence": row})
            continue
        indexed.setdefault(key, []).append(row)
    return indexed, invalid


def build_endpoint_lifecycle_ledger(
    planned_endpoints: Sequence[Mapping[str, Any]],
    *,
    materialization_evidence: Sequence[Mapping[str, Any]] | None = None,
    measurement_evidence: Sequence[Mapping[str, Any]] | None = None,
    terminal_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Join explicit endpoint evidence into one complete planned ledger.

    Unmatched and malformed evidence is retained in diagnostics.  It never
    creates an unplanned ledger row.  Duplicate planned identities are rejected
    because otherwise lifecycle counts would be ambiguous.
    """

    materialized_index, invalid_materialization = _index_evidence(materialization_evidence)
    measurement_index, invalid_measurement = _index_evidence(measurement_evidence)
    terminal_index, invalid_terminal = _index_evidence(terminal_evidence)

    planned_keys: set[Tuple[str, ...]] = set()
    ledger: List[Dict[str, Any]] = []
    for raw in planned_endpoints:
        if not isinstance(raw, Mapping):
            raise ValueError("Every planned endpoint must be a mapping.")
        plan = dict(raw)
        key = endpoint_identity(plan)
        if key in planned_keys:
            raise ValueError(f"Duplicate planned endpoint identity: {key!r}")
        planned_keys.add(key)

        material_rows = materialized_index.get(key, [])
        measurement_rows = measurement_index.get(key, [])
        terminal_rows = terminal_index.get(key, [])
        payload_rows = [
            row for row in measurement_rows if _has_measurement_payload(row)
        ]
        measured_rows = [row for row in measurement_rows if _has_measurement(row)]
        valid_measurement = bool(measured_rows)
        measurement_payload_present = bool(payload_rows)
        # ``measured`` is a compatibility count for an archived numeric
        # payload.  ``valid_measurement`` is the fail-closed scientific fact.
        measured = measurement_payload_present
        materialized = bool(
            measurement_payload_present
            or any(_is_materialized(row) for row in material_rows)
            or any(_is_materialized(row) for row in terminal_rows)
        )

        terminal_failure_rows = [
            row for row in terminal_rows if _terminal_failure(row)
        ]
        # A case runner can persist a numeric payload and then terminate with
        # a non-zero return code.  The payload remains observable evidence but
        # is not a valid measurement for ranking or technical-success counts.
        # Reclassified measurement rows carry the same explicit failure fields
        # so a compact offline replay remains fail-closed even when the source
        # pack contains no structured terminal sidecar.
        terminal_failure_rows.extend(
            row for row in payload_rows
            if _terminal_failure(row) and row not in terminal_failure_rows
        )
        terminal_reasons = [
            _concrete_reason(row) for row in terminal_failure_rows
            if _concrete_reason(row)
        ]

        if valid_measurement and not terminal_reasons:
            terminal = True
            terminal_reason = "measurement_available"
            lifecycle_status = "measured"
            measurement_evidence_status = "valid"
        elif measurement_payload_present and terminal_reasons:
            valid_measurement = False
            terminal = True
            terminal_reason = terminal_reasons[0]
            lifecycle_status = "measured_with_terminal_failure"
            measurement_evidence_status = (
                "payload_present_but_terminal_process_failure"
            )
        else:
            if not terminal_failure_rows:
                # Explicit generator rejection is also terminal evidence when
                # callers pass generation decisions in materialization_evidence.
                terminal_failure_rows = [
                    row for row in material_rows if _terminal_failure(row)
                ]
            terminal_reasons = [
                _concrete_reason(row) for row in terminal_failure_rows
                if _concrete_reason(row)
            ]
            # An explicit terminal flag without a reason is not enough for a
            # scientific lifecycle claim.  Keep the endpoint open/fail-closed.
            terminal = bool(terminal_failure_rows and terminal_reasons)
            if terminal:
                terminal_reason = terminal_reasons[0]
                lifecycle_status = "terminal_without_measurement"
            elif materialized:
                terminal_reason = "not_terminal_missing_terminal_evidence"
                lifecycle_status = "materialized_not_terminal"
            else:
                terminal_reason = "not_terminal_missing_materialization_evidence"
                lifecycle_status = "planned_not_materialized"
            measurement_evidence_status = (
                "payload_missing" if not measurement_payload_present
                else "payload_present_invalid_without_terminal_reason"
            )

        identity = _identity_values(plan)
        endpoint_id = _text(plan.get("endpoint_id"))
        ledger.append({
            "endpoint_id": endpoint_id,
            **identity,
            "planned": True,
            "materialized": materialized,
            "measurement_payload_present": measurement_payload_present,
            "measured": measured,
            "valid_measurement": valid_measurement,
            "terminal": terminal,
            "terminal_failure": lifecycle_status in {
                "measured_with_terminal_failure",
                "terminal_without_measurement",
            },
            "lifecycle_status": lifecycle_status,
            "terminal_reason": terminal_reason,
            "terminal_reason_scope": str(
                _first(
                    terminal_failure_rows[0] if terminal_failure_rows else {},
                    ("terminal_reason_scope", "reason_scope"),
                ) or ""
            ),
            "measurement_evidence_status": measurement_evidence_status,
            "technical_measurement_valid": bool(valid_measurement),
            "eligible_for_technical_observation": bool(valid_measurement),
            "runner_returncode": _first(
                terminal_failure_rows[0] if terminal_failure_rows else {},
                ("runner_returncode", "returncode"),
            ),
            "causal_backend": _first(
                terminal_failure_rows[0] if terminal_failure_rows else {},
                ("causal_backend", "backend", "hw_arch"),
            ),
            "causal_hw_arch": _first(
                terminal_failure_rows[0] if terminal_failure_rows else {},
                ("causal_hw_arch", "hw_arch"),
            ),
            "builder_backend": _first(
                terminal_failure_rows[0] if terminal_failure_rows else {},
                ("builder_backend",),
            ),
            "plan_evidence": plan,
            "materialization_evidence_count": len(material_rows),
            "measurement_evidence_count": len(measurement_rows),
            "valid_measurement_evidence_count": len(measured_rows),
            "terminal_evidence_count": len(terminal_rows),
        })

    all_evidence_keys = (
        set(materialized_index) | set(measurement_index) | set(terminal_index)
    )
    unmatched_keys = sorted(all_evidence_keys - planned_keys)
    summary = summarize_endpoint_lifecycle(ledger)
    return {
        "schema": "onnx-splitpoint/endpoint-lifecycle-ledger",
        "schema_version": 1,
        "rows": ledger,
        "summary": summary,
        "diagnostics": {
            "unmatched_evidence_count": sum(
                len(materialized_index.get(key, ()))
                + len(measurement_index.get(key, ()))
                + len(terminal_index.get(key, ()))
                for key in unmatched_keys
            ),
            "unmatched_evidence_identities": [list(key) for key in unmatched_keys],
            "invalid_evidence": (
                invalid_materialization + invalid_measurement + invalid_terminal
            ),
        },
    }


def summarize_endpoint_lifecycle(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    planned = sum(_bool(row.get("planned")) is True for row in rows)
    materialized = sum(_bool(row.get("materialized")) is True for row in rows)
    payload_present = sum(
        _bool(row.get("measurement_payload_present")) is True for row in rows
    )
    measured = sum(_bool(row.get("measured")) is True for row in rows)
    valid_measurement = sum(
        _bool(row.get("valid_measurement")) is True for row in rows
    )
    terminal = sum(_bool(row.get("terminal")) is True for row in rows)
    return {
        "planned": planned,
        "materialized": materialized,
        "measurement_payload_present": payload_present,
        "payload_missing": planned - payload_present,
        "measured": measured,
        "valid_measurement": valid_measurement,
        "valid_measurement_count": valid_measurement,
        "invalid_measurement_terminal": sum(
            str(row.get("lifecycle_status") or "")
            == "measured_with_terminal_failure"
            for row in rows
        ),
        "invalid_measurement_count": payload_present - valid_measurement,
        "missing_measurement_payload_count": planned - payload_present,
        "without_valid_measurement_count": planned - valid_measurement,
        "nonmeasured": planned - measured,
        "terminal": terminal,
        "nonterminal": planned - terminal,
    }


def compare_historical_invariants(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare compact historical facts without manufacturing endpoint rows."""

    mismatches: List[Dict[str, Any]] = []
    for field in (
        "planned",
        "measurement_payload_present",
        "payload_missing",
        "valid_measurement",
        "invalid_measurement_count",
    ):
        if field not in expected:
            continue
        actual = observed.get(field)
        wanted = expected.get(field)
        if actual != wanted:
            mismatches.append({"field": field, "expected": wanted, "observed": actual})

    observed_coverage = observed.get("audit_coverage")
    expected_coverage = expected.get("audit_coverage")
    if not isinstance(observed_coverage, Mapping):
        observed_coverage = {}
    if not isinstance(expected_coverage, Mapping):
        expected_coverage = {}
    for model_id, wanted in expected_coverage.items():
        actual = observed_coverage.get(model_id)
        if isinstance(wanted, Mapping) and isinstance(actual, Mapping):
            matches = all(actual.get(key) == value for key, value in wanted.items())
        else:
            matches = actual == wanted
        if not matches:
            mismatches.append({
                "field": f"audit_coverage.{model_id}",
                "expected": wanted,
                "observed": actual,
            })
    return {"ok": not mismatches, "mismatches": mismatches}


def _canonical_run_profile(value: Any) -> str:
    token = _text(value).lower().replace("-", "_")
    return _GENERIC_RUN_PROFILE_ALIASES.get(token, "")


def _case_id(row: Mapping[str, Any]) -> str:
    value = _first(row, ("case_id", "candidate_id", "split_id", "case"))
    text = _text(value)
    if text.lower() == "full":
        return "full"
    if text:
        if text.lower().startswith("b"):
            try:
                return f"b{int(text[1:]):03d}"
            except ValueError:
                return text
        try:
            return f"b{int(text):03d}"
        except ValueError:
            return text
    boundary = _first(row, ("boundary", "split_index", "boundary_index"))
    try:
        return f"b{int(boundary):03d}" if boundary not in (None, "") else ""
    except (TypeError, ValueError):
        return ""


def _generic_endpoint_id(model_id: str, run_profile_id: str, case_id: str) -> str:
    return f"generic:{model_id}:{run_profile_id}:{case_id}"


def _read_json_mapping(path: Path) -> tuple[Dict[str, Any], str]:
    if not path.is_file():
        return {}, f"missing:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, f"unreadable:{path}:{type(exc).__name__}:{exc}"
    if not isinstance(payload, Mapping):
        return {}, f"not_a_mapping:{path}"
    return dict(payload), ""


_REMOTE_CASE_START_RE = re.compile(
    r"^\[(?P<run>[^\]]+)\]\s+\[\d+/\d+\]\s+Running\s+"
    r"(?P<case>b\d+)(?:\s+.*)?$"
)
_REMOTE_CASE_FAILURE_RE = re.compile(
    r"^\[warn\]\s+case failed:\s+(?P<case>b\d+)\s+"
    r"\(rc=(?P<rc>-?\d+)\)$"
)


def historical_runtime_case_failures(
    run_dir: str | Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read exact case-runner failures retained by historical remote logs.

    v2.75.49 stored the suite/service return code in structured status files,
    but two case-local SIGSEGVs existed only in the explicitly bound stdout.
    This parser is intentionally stateful and narrow: a failure line is
    admitted only when it matches the currently active, known run/case pair.
    Free-form substrings and the duplicated workflow log are ignored.
    """

    root = Path(run_dir)
    failures: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    status_paths = sorted(
        root.glob(
            "models/*/benchmark_results/remote_benchmark_status_*.json"
        )
    )
    for status_path in status_paths:
        status, error = _read_json_mapping(status_path)
        if error:
            diagnostics.append({
                "reason": error,
                "status_path": str(status_path),
            })
            continue
        model_id = _text(status.get("model_id"))
        setup_id = _text(status.get("hardware_target_id"))
        stdout_value = _text(status.get("stdout_path"))
        if not model_id or not setup_id or not stdout_value:
            diagnostics.append({
                "reason": "remote_status_missing_model_setup_or_stdout",
                "status_path": str(status_path),
            })
            continue
        stdout_path = Path(stdout_value)
        if not stdout_path.is_absolute():
            stdout_path = root / stdout_path
        # The archived path can point to the original absolute EvalRun.  The
        # compact pack keeps a same-name sibling next to the status object.
        if not stdout_path.is_file():
            sibling = status_path.with_name(
                f"remote_benchmark_stdout_{setup_id}.txt"
            )
            stdout_path = sibling if sibling.is_file() else stdout_path
        if not stdout_path.is_file():
            diagnostics.append({
                "reason": "bound_remote_stdout_missing",
                "status_path": str(status_path),
                "stdout_path": str(stdout_path),
            })
            continue
        active: tuple[str, str] | None = None
        try:
            lines = stdout_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError as exc:
            diagnostics.append({
                "reason": f"remote_stdout_unreadable:{type(exc).__name__}:{exc}",
                "status_path": str(status_path),
                "stdout_path": str(stdout_path),
            })
            continue
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            start = _REMOTE_CASE_START_RE.fullmatch(line)
            if start:
                run_profile_id = _canonical_run_profile(start.group("run"))
                case_id = _case_id({"case_id": start.group("case")})
                active = (
                    (run_profile_id, case_id)
                    if run_profile_id and case_id else None
                )
                if active is None:
                    diagnostics.append({
                        "reason": "unknown_remote_case_start_identity",
                        "status_path": str(status_path),
                        "stdout_path": str(stdout_path),
                        "line_number": line_number,
                        "raw_evidence_line": line,
                    })
                continue
            failure = _REMOTE_CASE_FAILURE_RE.fullmatch(line)
            if not failure:
                continue
            failed_case = _case_id({"case_id": failure.group("case")})
            if active is None or active[1] != failed_case:
                diagnostics.append({
                    "reason": "remote_case_failure_without_matching_active_case",
                    "status_path": str(status_path),
                    "stdout_path": str(stdout_path),
                    "line_number": line_number,
                    "raw_evidence_line": line,
                })
                continue
            run_profile_id, case_id = active
            returncode = int(failure.group("rc"))
            key = (model_id, run_profile_id, case_id, returncode)
            if key in seen:
                continue
            seen.add(key)
            failures.append({
                "model_id": model_id,
                "case_id": case_id,
                "setup_id": setup_id,
                "run_profile_id": run_profile_id,
                "endpoint_id": _generic_endpoint_id(
                    model_id, run_profile_id, case_id
                ),
                "terminal": True,
                "terminal_failure": True,
                "measurement_valid": False,
                "terminal_reason": (
                    f"case_runner_nonzero_rc:{returncode}"
                ),
                "terminal_reason_scope": "case_runtime_process",
                "runner_returncode": returncode,
                "runner_signal_number": (
                    abs(returncode) if returncode < 0 else None
                ),
                "error_class": (
                    "case_runner_signal_sigsegv"
                    if returncode == -11 else "case_runner_nonzero_returncode"
                ),
                "evidence_source": str(stdout_path),
                "evidence_status_path": str(status_path),
                "evidence_line_number": line_number,
                "raw_evidence_line": line,
            })
    return failures, diagnostics


def classify_runtime_failure_measurements(
    run_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Mark payload-bearing failed case processes as invalid, without loss."""

    failures, diagnostics = historical_runtime_case_failures(run_dir)
    failures_by_endpoint = {
        _text(row.get("endpoint_id")): row for row in failures
        if _text(row.get("endpoint_id"))
    }
    projected: List[Dict[str, Any]] = []
    matched: set[str] = set()
    for source in rows or ():
        row = dict(source)
        model_id = _text(_first(row, ("model_id", "model", "model_name")))
        case_id = _case_id(row)
        run_profile_id = _run_profile_from_result(row)
        endpoint_id = (
            _generic_endpoint_id(model_id, run_profile_id, case_id)
            if model_id and run_profile_id and case_id else ""
        )
        failure = failures_by_endpoint.get(endpoint_id)
        if failure and _has_measurement_payload(row):
            matched.add(endpoint_id)
            reasons = _reason_tokens(
                row.get("performance_claim_exclusion_reasons")
            )
            if "terminal_process_failure" not in reasons:
                reasons.append("terminal_process_failure")
            row.update({
                "measurement_payload_present": True,
                "measurement_valid": False,
                "terminal": True,
                "terminal_failure": True,
                "runtime_executable": False,
                "row_status": "terminal_process_failure",
                "terminal_reason": failure["terminal_reason"],
                "terminal_reason_scope": failure["terminal_reason_scope"],
                "runner_returncode": failure["runner_returncode"],
                "runner_signal_number": failure["runner_signal_number"],
                "performance_claim_eligible": False,
                "ranking_eligible": False,
                "performance_claim_exclusion_reasons": reasons,
                "runtime_failure_evidence_source": failure["evidence_source"],
                "runtime_failure_evidence_line_number": failure[
                    "evidence_line_number"
                ],
            })
        projected.append(row)
    for failure in failures:
        endpoint_id = _text(failure.get("endpoint_id"))
        if endpoint_id not in matched:
            diagnostics.append({
                "reason": "runtime_failure_without_numeric_payload_row",
                "endpoint_id": endpoint_id,
                "evidence_source": failure.get("evidence_source"),
                "evidence_line_number": failure.get("evidence_line_number"),
            })
    return projected, failures, diagnostics


def _run_profile_from_result(row: Mapping[str, Any]) -> str:
    # Explicit logical/run identity wins.  Backend/direction are compatibility
    # fallbacks for normalized historical rows that did not retain run_id.
    for field in (
        "run_profile_id",
        "logical_run_id",
        "run_id",
        "direction",
        "backend",
        "producer_backend",
    ):
        canonical = _canonical_run_profile(row.get(field))
        if canonical:
            return canonical
    return ""


def _planned_generic_endpoints(
    *,
    plan: Mapping[str, Any],
    model_candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if "effective_generic_run_ids" in plan:
        raw_profiles = plan.get("effective_generic_run_ids")
        profile_source = "effective_generic_run_ids"
    else:
        raw_profiles = plan.get("logical_run_profiles")
        profile_source = "logical_run_profiles"
    if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)):
        raw_profiles = []

    profiles: List[str] = []
    unknown: List[str] = []
    for raw in raw_profiles:
        raw_id = (
            _first(raw, ("id", "run_id", "name"))
            if isinstance(raw, Mapping)
            else raw
        )
        canonical = _canonical_run_profile(raw_id)
        if not canonical:
            if _text(raw_id):
                unknown.append(_text(raw_id))
            continue
        if canonical not in profiles:
            profiles.append(canonical)

    rows: List[Dict[str, Any]] = []
    for model_id, candidate_rows in model_candidates.items():
        candidates = []
        seen_cases: set[str] = set()
        for candidate in candidate_rows:
            case_id = _case_id(candidate)
            if not case_id or case_id == "full" or case_id in seen_cases:
                continue
            seen_cases.add(case_id)
            candidates.append((case_id, dict(candidate)))
        for run_profile_id in profiles:
            spec = _GENERIC_RUN_PROFILE_SPECS[run_profile_id]
            kind = str(spec["kind"])
            if kind in {"reference", "split"}:
                for case_id, candidate in candidates:
                    rows.append({
                        "endpoint_id": _generic_endpoint_id(
                            model_id, run_profile_id, case_id
                        ),
                        "model_id": model_id,
                        "case_id": case_id,
                        "backend": spec["backend"],
                        "direction": spec["direction"],
                        "run_profile_id": run_profile_id,
                        "endpoint_scope": "generic_split",
                        "candidate_origin": _text(candidate.get("origin")),
                        "candidate_execution_roles": list(
                            candidate.get("candidate_execution_roles") or []
                        ) if isinstance(
                            candidate.get("candidate_execution_roles"), Sequence
                        ) and not isinstance(
                            candidate.get("candidate_execution_roles"),
                            (str, bytes),
                        ) else [],
                    })
            if kind in {"reference", "full"}:
                rows.append({
                    "endpoint_id": _generic_endpoint_id(
                        model_id, run_profile_id, "full"
                    ),
                    "model_id": model_id,
                    "case_id": "full",
                    "backend": spec["backend"],
                    "direction": spec["direction"],
                    "run_profile_id": run_profile_id,
                    "endpoint_scope": "generic_full_baseline",
                })
    return rows, {
        "run_profile_source": profile_source,
        "mapped_run_profiles": profiles,
        "unknown_run_profiles": unknown,
    }


def _hailo_state_run_profiles(
    state: Mapping[str, Any], run_profiles: Iterable[str],
) -> List[str]:
    """Map an explicit Hailo artifact state only to dependent run profiles."""

    hw = _text(state.get("hw_arch")).lower().replace("hailo10h", "hailo10")
    variant = _text(state.get("variant")).lower()
    if hw not in {"hailo8", "hailo10"} or variant != "part1":
        return []
    out: List[str] = []
    for run_profile_id in run_profiles:
        canonical = _canonical_run_profile(run_profile_id)
        if canonical == f"{hw}_to_trt":
            out.append(canonical)
    return out


def _backend_terminal_states(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    for field in ("backend_terminal_states", "hailo_backend_terminal_states"):
        value = row.get(field)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def build_evalrun_endpoint_lifecycle(
    run_dir: str | Path,
    normalized_rows: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a read-only Generic lifecycle ledger from one EvalRun.

    The optional profile is accepted for caller provenance only.  It is not
    used to expand missing execution plans: planned endpoint identity remains
    authoritative only when ``effective_execution_plan.json`` and each final
    candidate plan provide it explicitly.
    """

    root = Path(run_dir)
    plan_path = root / "effective_execution_plan.json"
    plan, plan_error = _read_json_mapping(plan_path)
    adapter_errors: List[str] = [plan_error] if plan_error else []

    raw_model_ids = plan.get("models")
    model_ids = [
        _text(row.get("id") if isinstance(row, Mapping) else row)
        for row in raw_model_ids or []
    ] if isinstance(raw_model_ids, Sequence) and not isinstance(raw_model_ids, (str, bytes)) else []
    model_ids = list(dict.fromkeys(model_id for model_id in model_ids if model_id))

    model_candidates: Dict[str, List[Dict[str, Any]]] = {}
    audit_candidates_by_model: Dict[str, List[Dict[str, Any]]] = {}
    candidate_plan_paths: Dict[str, str] = {}
    for model_id in model_ids:
        candidate_path = (
            root / "models" / model_id / "analysis" / "final_candidate_plan.json"
        )
        candidate_plan_paths[model_id] = str(candidate_path)
        candidate_plan, candidate_error = _read_json_mapping(candidate_path)
        if candidate_error:
            adapter_errors.append(candidate_error)
        selected = candidate_plan.get("selected_candidates")
        model_candidates[model_id] = [
            dict(row) for row in selected or () if isinstance(row, Mapping)
        ] if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)) else []
        audit = candidate_plan.get("audit_candidates")
        audit_candidates_by_model[model_id] = [
            dict(row) for row in audit or () if isinstance(row, Mapping)
        ] if isinstance(audit, Sequence) and not isinstance(audit, (str, bytes)) else []

    planned, plan_diagnostics = _planned_generic_endpoints(
        plan=plan, model_candidates=model_candidates
    )
    planned_ids = {_text(row.get("endpoint_id")) for row in planned}
    split_profiles = {
        run_id
        for run_id in plan_diagnostics["mapped_run_profiles"]
        if _GENERIC_RUN_PROFILE_SPECS[run_id]["kind"] in {"reference", "split"}
    }

    materialization_evidence: List[Dict[str, Any]] = []
    terminal_evidence: List[Dict[str, Any]] = []
    generation_paths: Dict[str, str] = {}
    for model_id in model_ids:
        generation_path = (
            root / "models" / model_id / "benchmark_set" / "generation_decisions.json"
        )
        generation_paths[model_id] = str(generation_path)
        generation, generation_error = _read_json_mapping(generation_path)
        if generation_error:
            # Generation decisions are useful but optional lifecycle evidence;
            # absence does not make the plan invalid or terminal.
            adapter_errors.append(generation_error)
            continue
        accepted = generation.get("accepted_cases")
        for raw in accepted or ():
            if not isinstance(raw, Mapping):
                continue
            case_id = _case_id(raw)
            if not case_id:
                continue
            backend_states = _backend_terminal_states(raw)
            explicitly_scoped_profiles = {
                run_profile_id
                for state in backend_states
                for run_profile_id in _hailo_state_run_profiles(
                    state, split_profiles,
                )
            }
            for run_profile_id in split_profiles:
                # New partial-retention decisions provide an explicit Hailo
                # artifact lifecycle.  They replace (rather than supplement)
                # the legacy blanket case-materialized assumption for the
                # dependent Hailo profile. Non-Hailo profiles remain case-
                # materialized because the accepted split ONNX exists.
                if run_profile_id in explicitly_scoped_profiles:
                    continue
                endpoint_id = _generic_endpoint_id(model_id, run_profile_id, case_id)
                if endpoint_id in planned_ids:
                    materialization_evidence.append({
                        **dict(raw),
                        "endpoint_id": endpoint_id,
                        "materialized": True,
                        "evidence_source": str(generation_path),
                    })
            for state in backend_states:
                for run_profile_id in _hailo_state_run_profiles(
                    state, split_profiles,
                ):
                    endpoint_id = _generic_endpoint_id(
                        model_id, run_profile_id, case_id,
                    )
                    if endpoint_id not in planned_ids:
                        continue
                    state_status = _text(state.get("status")).lower()
                    reason = _text(state.get("reason"))
                    evidence = {
                        **dict(state),
                        "endpoint_id": endpoint_id,
                        "terminal_reason_scope": "backend_artifact",
                        "causal_backend": _text(state.get("hw_arch")),
                        "causal_hw_arch": _text(state.get("hw_arch")),
                        "evidence_source": str(generation_path),
                    }
                    if state_status == "ready" and _bool(
                        state.get("available")
                    ) is not False:
                        materialization_evidence.append({
                            **evidence,
                            "materialized": True,
                            "status": "ready",
                        })
                    elif state_status in {
                        "terminal_failed", "terminal_missing"
                    }:
                        terminal_evidence.append({
                            **evidence,
                            "terminal": True,
                            "status": state_status,
                            "terminal_reason": (
                                "backend_terminal_state:"
                                + (reason or state_status)
                            ),
                        })
        rejected = generation.get("rejected_cases")
        for raw in rejected or ():
            if not isinstance(raw, Mapping):
                continue
            case_id = _case_id(raw)
            reason = _concrete_reason(raw)
            if not case_id:
                continue
            backend_states = _backend_terminal_states(raw)
            for state in backend_states:
                for run_profile_id in _hailo_state_run_profiles(
                    state, split_profiles,
                ):
                    endpoint_id = _generic_endpoint_id(
                        model_id, run_profile_id, case_id,
                    )
                    if endpoint_id not in planned_ids:
                        continue
                    state_status = _text(state.get("status")).lower()
                    state_reason = _text(state.get("reason"))
                    if state_status in {
                        "terminal_failed", "terminal_missing"
                    }:
                        terminal_evidence.append({
                            **dict(state),
                            "endpoint_id": endpoint_id,
                            "terminal": True,
                            "status": state_status,
                            "terminal_reason": (
                                "backend_terminal_state:"
                                + (state_reason or state_status)
                            ),
                            "terminal_reason_scope": "backend_artifact",
                            "causal_backend": _text(state.get("hw_arch")),
                            "causal_hw_arch": _text(state.get("hw_arch")),
                            "evidence_source": str(generation_path),
                        })
            # Legacy generators rejected the complete case after one Hailo
            # build failure.  Preserve that case-level terminal fact for all
            # planned split endpoints, but do not falsely claim that ORT,
            # DeepX or the other Hailo backend itself caused the rejection.
            legacy_case_rejection = not backend_states
            for run_profile_id in (
                split_profiles if legacy_case_rejection else ()
            ):
                endpoint_id = _generic_endpoint_id(model_id, run_profile_id, case_id)
                if endpoint_id in planned_ids:
                    terminal_evidence.append({
                        **dict(raw),
                        "endpoint_id": endpoint_id,
                        "terminal": True,
                        # Preserve the absence of a concrete reason.  The core
                        # builder will then leave this endpoint open.
                        "terminal_reason": (
                            f"legacy_case_rejected:{reason}"
                            if legacy_case_rejection and reason else reason
                        ),
                        "terminal_reason_scope": (
                            "case_generation"
                            if legacy_case_rejection else "backend_artifact"
                        ),
                        "causal_backend": _text(
                            raw.get("causal_backend")
                            or raw.get("hw_arch")
                        ),
                        "causal_hw_arch": _text(raw.get("hw_arch")),
                        "builder_backend": _text(raw.get("backend")),
                        "evidence_source": str(generation_path),
                    })

    (
        classified_rows,
        runtime_failure_evidence,
        runtime_failure_diagnostics,
    ) = classify_runtime_failure_measurements(root, normalized_rows)
    terminal_evidence.extend(
        dict(row) for row in runtime_failure_evidence
        if _text(row.get("endpoint_id")) in planned_ids
    )
    projected_measurements: List[Dict[str, Any]] = []
    unprojected_rows: List[Dict[str, Any]] = []
    for raw in classified_rows or ():
        if not isinstance(raw, Mapping):
            unprojected_rows.append({"reason": "not_a_mapping", "row": raw})
            continue
        row = dict(raw)
        model_id = _text(_first(row, ("model_id", "model", "model_name")))
        case_id = _case_id(row)
        run_profile_id = _run_profile_from_result(row)
        if not model_id or not case_id or not run_profile_id:
            unprojected_rows.append({
                "reason": "missing_or_unknown_explicit_endpoint_identity",
                "row": row,
            })
            continue
        endpoint_id = _generic_endpoint_id(model_id, run_profile_id, case_id)
        if endpoint_id not in planned_ids:
            unprojected_rows.append({
                "reason": "result_endpoint_not_in_explicit_plan",
                "derived_endpoint_id": endpoint_id,
                "row": row,
            })
            continue
        projected_measurements.append({
            **row,
            "endpoint_id": endpoint_id,
            "run_profile_id": run_profile_id,
        })

    payload = build_endpoint_lifecycle_ledger(
        planned,
        materialization_evidence=materialization_evidence,
        measurement_evidence=projected_measurements,
        terminal_evidence=terminal_evidence,
    )
    ledger_rows = list(payload.get("rows") or [])
    audit_coverage: Dict[str, Dict[str, Any]] = {}
    for model_id in model_ids:
        planned_audit_ids = {
            _case_id(row)
            for row in audit_candidates_by_model.get(model_id, [])
            if _case_id(row)
        }
        payload_case_ids = {
            _text(row.get("case_id"))
            for row in ledger_rows
            if _text(row.get("model_id")) == model_id
            and _bool(row.get("measurement_payload_present")) is True
            and _text(row.get("case_id")) != "full"
        }
        valid_case_ids = {
            _text(row.get("case_id"))
            for row in ledger_rows
            if _text(row.get("model_id")) == model_id
            and _bool(row.get("valid_measurement")) is True
            and _text(row.get("case_id")) != "full"
        }
        observed = planned_audit_ids & payload_case_ids
        valid_observed = planned_audit_ids & valid_case_ids
        audit_coverage[model_id] = {
            "planned": len(planned_audit_ids),
            # Compatibility name is the historical payload-coverage fact.
            "observed": len(observed),
            "payload_observed": len(observed),
            "valid_observed_any_backend": len(valid_observed),
            "missing_payload_case_ids": sorted(
                planned_audit_ids - payload_case_ids
            ),
            "missing_valid_measurement_case_ids": sorted(
                planned_audit_ids - valid_case_ids
            ),
        }
    payload["audit_coverage"] = audit_coverage
    payload["summary"]["audit_coverage"] = audit_coverage
    expected_total = plan.get("expected_generic_result_rows_total")
    expected_total_int = None
    try:
        expected_total_int = int(expected_total) if expected_total not in (None, "") else None
    except (TypeError, ValueError):
        adapter_errors.append(
            f"invalid_expected_generic_result_rows_total:{expected_total!r}"
        )
    constructed_total = payload["summary"]["planned"]
    expected_count_matches = (
        None if expected_total_int is None
        else constructed_total == expected_total_int
    )
    if expected_count_matches is False:
        adapter_errors.append(
            "planned_endpoint_count_mismatch:"
            f"constructed={constructed_total}:expected={expected_total_int}"
        )
    adapter_status = (
        "ok"
        if not adapter_errors and not plan_diagnostics["unknown_run_profiles"]
        else "partial"
    )
    completeness_reasons: List[str] = []
    if adapter_status != "ok":
        completeness_reasons.append(f"adapter_{adapter_status}")
    if expected_count_matches is not True:
        completeness_reasons.append("planned_count_not_verified")
    if unprojected_rows:
        completeness_reasons.append("normalized_rows_unprojected")
    if int(payload["summary"].get("nonterminal") or 0) > 0:
        completeness_reasons.append("planned_endpoints_nonterminal")
    evidence_completeness_status = (
        "complete" if not completeness_reasons else (
            "incomplete"
            if completeness_reasons == ["planned_endpoints_nonterminal"]
            else "partial"
        )
    )
    valid_measurement_count = int(
        payload["summary"].get("valid_measurement") or 0
    )
    if valid_measurement_count == constructed_total:
        measurement_completeness_status = "complete"
    elif int(payload["summary"].get("nonterminal") or 0) == 0:
        measurement_completeness_status = (
            "incomplete_but_terminally_accounted"
        )
    else:
        measurement_completeness_status = "incomplete_with_open_endpoints"
    payload["adapter"] = {
        "schema": "onnx-splitpoint/evalrun-endpoint-lifecycle-adapter",
        "schema_version": 1,
        "status": adapter_status,
        "evidence_completeness_status": evidence_completeness_status,
        "evidence_completeness_reasons": completeness_reasons,
        "measurement_completeness_status": measurement_completeness_status,
        "read_only": True,
        "run_dir": str(root),
        "effective_execution_plan_path": str(plan_path),
        "candidate_plan_paths": candidate_plan_paths,
        "generation_decision_paths": generation_paths,
        "profile_supplied": isinstance(profile, Mapping),
        **plan_diagnostics,
        "constructed_planned_endpoint_count": constructed_total,
        "expected_generic_result_rows_total": expected_total_int,
        "expected_count_matches": expected_count_matches,
        "projected_normalized_row_count": len(projected_measurements),
        "unprojected_normalized_row_count": len(unprojected_rows),
        "unprojected_normalized_rows": unprojected_rows,
        "runtime_failure_evidence_count": len(runtime_failure_evidence),
        "runtime_failure_evidence": runtime_failure_evidence,
        "runtime_failure_diagnostics": runtime_failure_diagnostics,
        "errors": adapter_errors,
    }
    return payload
