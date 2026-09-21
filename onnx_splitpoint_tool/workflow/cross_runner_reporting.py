from __future__ import annotations

"""Generic-to-Native ranking-transfer reporting.

The Generic Runner measures a broad candidate universe while the Native Runner
provides the hardware-near deployment path.  v60t makes their relationship an
explicit evidence object instead of assuming that absolute timings or rankings
are interchangeable.
"""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..ranking_methods import canonical_backend, canonical_direction, direction_parts
from .artifacts import read_json


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _b(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "ok", "pass", "passed", "claim_ok", "valid", "eligible"}:
        return True
    if text in {"0", "false", "no", "fail", "failed", "invalid", "error", "unsupported"}:
        return False
    return None


def _quality_decision(row: Mapping[str, Any]) -> str:
    if isinstance(row.get("accuracy_assessment"), Mapping):
        return str(row["accuracy_assessment"].get("accuracy_class") or "not_estimable")
    decisions: list[str] = []
    for key in (
        "task_quality_decision", "accuracy_gate_decision",
        "task_quality_status", "quality_gate_status",
    ):
        text = str(row.get(key) or "").strip().lower().replace("-", "_")
        if text in {"pass", "passed", "quality_pass", "quality_passed", "eligible"}:
            decisions.append("pass")
        elif text in {"fail", "failed", "quality_fail", "quality_failed"}:
            decisions.append("fail")
        elif text in {"inconclusive", "quality_inconclusive"}:
            decisions.append("inconclusive")
    if "fail" in decisions:
        return "fail"
    if "inconclusive" in decisions:
        return "inconclusive"
    if "pass" in decisions:
        return "pass"
    if _b(row.get("accuracy_gate_pass")) is True:
        return "pass"
    if _b(row.get("accuracy_gate_pass")) is False:
        return "fail"
    return "unavailable"


def _canon_model(value: Any) -> str:
    return str(value or "").strip()


def _canon_case(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "none"}:
        return ""
    if text == "full":
        return "full"
    if text.startswith("b") and text[1:].isdigit():
        return f"b{int(text[1:]):03d}"
    if text.isdigit():
        return f"b{int(text):03d}"
    return text


def _canon_direction(value: Any, backend: Any = "") -> str:
    # Preserve stage order. The previous substring checks returned the forward
    # direction for every reverse TRT->accelerator row because both tokens were
    # present before the reverse branch was reached.
    return canonical_direction(value or backend or "unknown")


def _quality_identities(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    identities = row.get("quality_request_identities_by_variant")
    if not isinstance(identities, Mapping):
        return []
    variant = str(row.get("variant") or "").strip().lower()
    aliases = {"split": "composed", "pipeline": "composed"}
    variant = aliases.get(variant, variant)
    selected = identities.get(variant)
    if isinstance(selected, Mapping):
        return [selected]
    values = [value for value in identities.values() if isinstance(value, Mapping)]
    # A sole request identity is exact. Multiple variant identities are not
    # interchangeable and therefore stay unresolved.
    return values if len(values) == 1 else []


def _quality_identity(row: Mapping[str, Any]) -> Mapping[str, Any]:
    identities = _quality_identities(row)
    return identities[0] if len(identities) == 1 else {}


def _canonical_scalar(value: Any) -> str:
    return str(value or "").strip().lower()


def _resolved_identity_scalar(
    row: Mapping[str, Any], direct_keys: Sequence[str], identity_key: str,
    *, normalize: Any = _canonical_scalar,
) -> str:
    direct = ""
    for key in direct_keys:
        if row.get(key) not in (None, ""):
            direct = normalize(row.get(key))
            break
    embedded = normalize(_quality_identity(row).get(identity_key))
    return direct or embedded


def _explicit_endpoint_contract_complete(
    row: Mapping[str, Any],
) -> tuple[bool | None, list[str]]:
    """Return only a genuinely explicit row-level completeness declaration.

    ``normalize_benchmark_row`` has historically materialised a missing
    ``endpoint_contract_complete`` field as ``False``.  That compatibility
    value is useful to old consumers, but it is not evidence that the producer
    explicitly declared an incomplete endpoint contract.  New normalized rows
    therefore carry ``endpoint_contract_complete_explicit`` as provenance.

    Rows without the provenance marker keep the former fail-closed behaviour:
    a Boolean value is treated as explicit.  This avoids weakening validation
    for native/archived producers that really did write the field before the
    marker existed.
    """

    value = row.get("endpoint_contract_complete")
    provenance = row.get("endpoint_contract_complete_explicit")
    errors: list[str] = []
    if provenance is False:
        return None, errors
    if provenance is True and not isinstance(value, bool):
        errors.append(
            "endpoint_contract_complete_explicit_without_boolean_value"
        )
        return None, errors
    return (value if isinstance(value, bool) else None), errors


def _identity_resolution_errors(row: Mapping[str, Any]) -> list[str]:
    """Return variant-local explicit/embedded identity conflicts.

    A compact identity is evidence, not a fallback bag of aliases.  Any drift
    between the selected variant (``split`` maps to ``composed``) and explicit
    row fields makes the join ineligible instead of silently preferring one.
    """

    identity = _quality_identity(row)
    errors = [str(value) for value in list(identity.get("identity_errors") or []) if str(value)]
    if identity and identity.get("identity_valid") is False:
        errors.append("embedded_variant_identity_invalid")

    comparisons = (
        (
            "setup_id",
            ("setup_id", "source_setup_id", "measurement_setup_id", "hardware_setup_id"),
            "setup_id", _canonical_scalar,
        ),
        (
            "runtime_precision_identity",
            ("execution_precision", "runtime_precision_identity", "precision"),
            "runtime_precision_identity", _canonical_scalar,
        ),
        (
            "comparison_backend", ("comparison_backend",),
            "comparison_backend", canonical_backend,
        ),
        (
            "producer_backend", ("producer_backend",),
            "producer_backend", canonical_backend,
        ),
        (
            "task", ("task", "benchmark_task_used"), "task", _canonical_scalar,
        ),
        (
            "stage",
            ("stage", "completed_task_stage", "accelerator_output_stage"),
            "stage", _canonical_scalar,
        ),
        (
            "endpoint_contract_hash",
            (
                "endpoint_contract_hash",
                "comparison_endpoint_contract_hash",
                "completed_task_comparison_endpoint_contract_hash",
            ),
            "endpoint_contract_hash",
            lambda value: _canonical_scalar(value).removeprefix("sha256:"),
        ),
    )
    for label, direct_keys, identity_key, normalize in comparisons:
        direct = ""
        for key in direct_keys:
            if row.get(key) not in (None, ""):
                direct = normalize(row.get(key))
                break
        embedded = normalize(identity.get(identity_key))
        if direct and embedded and direct != embedded:
            errors.append(
                f"explicit_embedded_{label}_conflict:{direct}!={embedded}"
            )

    direct_direction = _canon_direction(
        row.get("direction") or row.get("backend") or "",
    )
    embedded_direction = _canon_direction(
        identity.get("direction") or identity.get("source_run_id") or "",
    )
    if (
        any(direction_parts(direct_direction))
        and embedded_direction not in {"", "unknown"}
        and direct_direction != embedded_direction
    ):
        errors.append(
            "explicit_embedded_direction_conflict:"
            f"{direct_direction}!={embedded_direction}"
        )

    direct_complete, completeness_errors = (
        _explicit_endpoint_contract_complete(row)
    )
    errors.extend(completeness_errors)
    embedded_complete = identity.get("endpoint_contract_complete")
    if (
        isinstance(direct_complete, bool)
        and isinstance(embedded_complete, bool)
        and direct_complete is not embedded_complete
    ):
        errors.append("explicit_embedded_endpoint_contract_complete_conflict")

    direct_attestation = row.get("output_endpoint_attestation")
    embedded_attestation = identity.get("output_endpoint_attestation")
    if (
        isinstance(direct_attestation, Mapping) and direct_attestation
        and isinstance(embedded_attestation, Mapping) and embedded_attestation
        and json.dumps(dict(direct_attestation), sort_keys=True, default=str)
        != json.dumps(dict(embedded_attestation), sort_keys=True, default=str)
    ):
        errors.append("explicit_embedded_output_endpoint_attestation_conflict")
    return sorted(set(errors))


def _row_direction(row: Mapping[str, Any]) -> str:
    direct = _canon_direction(row.get("direction"), row.get("backend"))
    if any(direction_parts(direct)):
        return direct
    identity = _quality_identity(row)
    embedded = _canon_direction(
        identity.get("direction") or identity.get("source_run_id") or "unknown",
    )
    return embedded if embedded not in {"", "unknown"} else direct


def _canon_precision(row: Mapping[str, Any]) -> str:
    return _resolved_identity_scalar(
        row,
        ("execution_precision", "runtime_precision_identity", "precision"),
        "runtime_precision_identity",
    )


def _canon_setup(row: Mapping[str, Any]) -> str:
    explicit = _resolved_identity_scalar(
        row,
        ("setup_id", "source_setup_id", "measurement_setup_id", "hardware_setup_id"),
        "setup_id",
    )
    if explicit:
        return explicit
    values = {
        str(value or "").strip().lower()
        for value in list(row.get("quality_source_setup_ids") or [])
        if str(value or "").strip()
    }
    for identity in _quality_identities(row):
        values.update(
            str(value or "").strip().lower()
            for value in list(identity.get("setup_ids") or [])
            if str(value or "").strip()
        )
        setup_id = str(identity.get("setup_id") or "").strip().lower()
        if setup_id:
            values.add(setup_id)
    return next(iter(values)) if len(values) == 1 else ""


def _canon_comparison_backend(row: Mapping[str, Any]) -> str:
    text = _resolved_identity_scalar(
        row, ("comparison_backend",), "comparison_backend",
        normalize=canonical_backend,
    )
    if text:
        return text
    stage1, stage2 = direction_parts(_row_direction(row))
    accelerators = [
        value for value in (stage1, stage2)
        if value.startswith(("hailo", "deepx"))
    ]
    return accelerators[0] if len(accelerators) == 1 else ""


def _endpoint_contract_hash(row: Mapping[str, Any]) -> str:
    direct = str(
        row.get("endpoint_contract_hash")
        or row.get("comparison_endpoint_contract_hash")
        or row.get("completed_task_comparison_endpoint_contract_hash")
        or ""
    ).strip().lower().removeprefix("sha256:")
    embedded = str(
        _quality_identity(row).get("endpoint_contract_hash") or ""
    ).strip().lower().removeprefix("sha256:")
    if direct:
        return direct
    if embedded:
        return embedded
    hashes = {
        str(identity.get("endpoint_contract_hash") or "")
        .strip().lower().removeprefix("sha256:")
        for identity in _quality_identities(row)
        if str(identity.get("endpoint_contract_hash") or "").strip()
    }
    return next(iter(hashes)) if len(hashes) == 1 else ""


def _endpoint_attested(row: Mapping[str, Any], contract_hash: str) -> bool:
    identity = _quality_identity(row)
    direct_attestation = row.get("output_endpoint_attestation")
    attestation = (
        direct_attestation
        if isinstance(direct_attestation, Mapping) and direct_attestation
        else identity.get("output_endpoint_attestation")
    )
    explicit_complete, completeness_errors = (
        _explicit_endpoint_contract_complete(row)
    )
    complete = (
        explicit_complete
        if not completeness_errors and explicit_complete is not None
        else identity.get("endpoint_contract_complete")
        if not completeness_errors
        else None
    )
    return bool(
        complete is True
        and isinstance(attestation, Mapping)
        and attestation.get("attested") is True
        and str(attestation.get("status") or "").strip().lower() == "passed"
        and str(attestation.get("endpoint_contract_hash") or "")
        .strip().lower().removeprefix("sha256:") == contract_hash
    )


def _attested_endpoint_id(row: Mapping[str, Any]) -> str:
    """Return the exact endpoint identity; attestation is gated separately."""
    identity = _quality_identity(row)
    task = str(
        row.get("task") or row.get("benchmark_task_used")
        or identity.get("task") or ""
    ).strip().lower()
    if not task:
        tasks = {
            str(identity.get("task") or "").strip().lower()
            for identity in _quality_identities(row)
            if str(identity.get("task") or "").strip()
        }
        task = next(iter(tasks)) if len(tasks) == 1 else ""
    stage = str(
        row.get("stage") or row.get("completed_task_stage")
        or row.get("accelerator_output_stage") or identity.get("stage") or ""
    ).strip().lower()
    contract_hash = _endpoint_contract_hash(row)
    if (
        not task or not stage
        or len(contract_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in contract_hash)
    ):
        return ""
    return f"{task}:{stage}:{contract_hash}"


_ExactIdentity = tuple[str, str, str, str, str, str, str]
_PlannedIdentity = tuple[str, str, str, str, str]


def _exact_identity(row: Mapping[str, Any]) -> _ExactIdentity:
    return (
        _canon_model(row.get("model_id") or row.get("model")),
        _canon_case(row.get("case_id") or row.get("case")),
        _row_direction(row),
        _canon_precision(row),
        _canon_setup(row),
        _canon_comparison_backend(row),
        _attested_endpoint_id(row),
    )


def _planned_identity(row: Mapping[str, Any]) -> _PlannedIdentity:
    """Return the dimensions frozen before the bounded Native supplement.

    Precision and completed-task endpoint are deliberately absent here.  The
    Native supplement plan selected a model/case/backend/setup intersection;
    the exact runtime precision and endpoint are bound by the request and
    runtime evidence below.  Keeping the two levels separate prevents the much
    larger Generic audit universe from being mistaken for missing Native work.
    """

    return (
        _canon_model(row.get("model_id") or row.get("model")),
        _canon_case(row.get("case_id") or row.get("case")),
        _row_direction(row),
        _canon_setup(row),
        _canon_comparison_backend(row),
    )


def _planned_identity_complete(identity: _PlannedIdentity) -> bool:
    return all(identity)


def _identity_complete(identity: _ExactIdentity) -> bool:
    return all(identity)


def _completed_task_measurement(row: Mapping[str, Any], *, native: bool = False) -> bool:
    """Do not join P2/logits timing with a measured completed task rate."""
    if native and row.get("rate_endpoint_projection_version") == 1:
        value = _f(row.get("completed_task_fps"))
        return value is not None and value > 0 and row.get("performance_endpoint") == "completed_task"
    endpoint = str(row.get("measurement_endpoint") or row.get("performance_endpoint") or "")
    return (endpoint in {"completed_detection", "completed_classification", "prepared_input_to_completed_task", "completed_task"}
            and row.get("postprocess_completion_verified") is True
            and (_f(row.get("postprocess_completed_frames")) or 0) > 0)


def _measured_native_fps(row: Mapping[str, Any]) -> tuple[float | None, str]:
    if row.get('rate_endpoint_projection_version') == 1:
        value = row.get('completed_task_fps')
        return (float(value), 'completed_task_count_time') if value is not None else (None, str(row.get('completed_task_fps_unavailable_reason') or 'completed_task_unavailable'))
    """Return an actually observed Native makespan rate and its source.

    ``paper_fps`` and ``1000 / cycle_ms`` are timing-model rates.  They are
    useful diagnostics, but labelling either one as measured throughput hid a
    large difference in the v2.64 Hailo-8/YOLO smoke run (198.8 measured FPS
    versus a 455.3 FPS stage-cycle rate).  Only fields whose contract denotes
    a measured row/makespan rate are admitted here.
    """
    for key in (
        "native_measured_throughput_fps",
        "fps_makespan",
        "throughput_fps_makespan",
        "pipeline_fps_measured",
        "measured_fps",
        "fps",
    ):
        value = _f(row.get(key))
        if value is not None and value > 0:
            return value, key
    return None, ""


def _native_theoretical_cycle(row: Mapping[str, Any]) -> tuple[float | None, str]:
    """Return a separately-labelled Native timing-model cycle, if present."""
    for key in (
        "paper_equivalent_cycle_ms",
        "pipeline_cycle_ms",
        "cycle_ms",
    ):
        value = _f(row.get(key))
        if value is not None and value > 0:
            return value, key
    paper_fps = _f(row.get("paper_fps") or row.get("paper_equivalent_fps"))
    if paper_fps is not None and paper_fps > 0:
        return 1000.0 / paper_fps, "paper_fps"
    # Older Native summaries expose the two pipeline-thread cycles without a
    # paper-equivalent aggregate.  The pipeline model is bottleneck-limited;
    # selecting p1 alone was the source of the former inflated FPS value.
    thread_cycles = [
        value
        for value in (_f(row.get("p1_thread_ms")), _f(row.get("p2_thread_ms")))
        if value is not None and value > 0
    ]
    if thread_cycles:
        return max(thread_cycles), "max(p1_thread_ms,p2_thread_ms)"
    return None, ""


def _cycle_ms(row: Mapping[str, Any], *, native: bool = False) -> float | None:
    if native:
        measured_fps, _ = _measured_native_fps(row)
        if measured_fps is not None:
            return 1000.0 / measured_fps
        # Legacy Native summaries did not persist a measured makespan FPS.  A
        # cycle can still be used to reproduce their *ordering*, but it must
        # never be re-labelled as measured throughput.  New summaries always
        # take the measured-FPS branch above, even when a stage-cycle value is
        # present as well.
        for key in ("native_measured_cycle_ms", "measured_cycle_ms", "cycle_ms"):
            value = _f(row.get(key))
            if value is not None and value > 0:
                return value
        return None
    # Keep this priority aligned with the Generic scientific-row projection.
    # Full EvaluationRuns feed the reporter normalized benchmark rows, whose
    # canonical values still use the long field names below.  Compact debug
    # packs can instead fall back to archived scientific rows that already
    # expose ``cycle_ms``/``latency_ms``.  Supporting both representations is
    # essential: otherwise the exact same measurements join in the archived
    # replay but disappear as ``generic_ranking_measurement_missing`` in the
    # full-run replay.
    keys = (
        "pipeline_cycle_selected_ms",
        "total_latency_ms",
        "cycle_ms",
        "pipeline_cycle_ms",
        "split_latency_e2e_ms",
        "full_e2e_latency_ms",
        "latency_ms",
        "latency_mean_ms",
    )
    for key in keys:
        value = _f(row.get(key))
        if value is not None and value > 0:
            return value
    fps_keys = (
        "throughput_primary_fps",
        "pipeline_fps_selected",
        "heterogeneous_pipeline_fps",
        "throughput_fps",
        "fps_makespan",
        "fps",
    )
    for key in fps_keys:
        fps = _f(row.get(key))
        if fps is not None and fps > 0:
            return 1000.0 / fps
    return None


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg = (pos + 1 + end) / 2.0
        for offset in range(pos, end):
            ranks[order[offset]] = avg
        pos = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    sx = sum((v - mx) ** 2 for v in x)
    sy = sum((v - my) ** 2 for v in y)
    if sx <= 0 or sy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sx * sy)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return _pearson(_rank(x), _rank(y))


def _kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    concordant = discordant = tie_x = tie_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    return (concordant - discordant) / denom if denom > 0 else None


def _pairwise_counts(x: Sequence[float], y: Sequence[float]) -> tuple[int, int]:
    comparable = correct = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 or dy == 0:
                continue
            comparable += 1
            correct += int(dx == dy)
    return correct, comparable


def _pairwise_concordance(x: Sequence[float], y: Sequence[float]) -> float | None:
    correct, comparable = _pairwise_counts(x, y)
    return correct / comparable if comparable else None


def _median(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.median(vals) if vals else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _load_native_rows(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        run_dir / "reports" / "native_producer_combined_summary.json",
        run_dir / "reports" / "native_producer_summary.json",
    ]
    for path in candidates:
        payload = read_json(path, default={}) or {}
        rows = payload.get("rows") if isinstance(payload, Mapping) else None
        if isinstance(rows, list):
            from ..native_rate_endpoints import report_rate_fields
            return [{**dict(row), **report_rate_fields(row, run_dir)} for row in rows if isinstance(row, Mapping)]
    return []


def _load_native_validation(
    run_dir: Path,
) -> tuple[
    dict[_ExactIdentity, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    path = run_dir / "reports" / "native_validation" / "native_producer_validation_summary.json"
    payload = read_json(path, default={}) or {}
    out: dict[_ExactIdentity, list[dict[str, Any]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    for row in list(payload.get("rows") or []):
        if not isinstance(row, Mapping):
            continue
        identity_errors = _identity_resolution_errors(row)
        key = _exact_identity(row)
        if identity_errors:
            exclusions.append({
                "runner_regime": "native_validation",
                "identity": list(key),
                "reason": "variant_identity_conflict",
                "identity_errors": identity_errors,
            })
            continue
        out[key].append(dict(row))
    return dict(out), exclusions


def _native_plan_direction(backend: Any) -> str:
    text = str(backend or "").strip().lower()
    if not text:
        return "unknown"
    if "_to_" in text:
        return _canon_direction(text)
    return _canon_direction(f"{text}_to_tensorrt")


def _load_planned_native_intersection(
    native_root: Path,
    native_rows: Sequence[Mapping[str, Any]],
) -> tuple[set[_PlannedIdentity], str, list[dict[str, Any]]]:
    """Load the frozen Native denominator, with a legacy observed fallback."""

    matrix_path = native_root / "reports" / "native_expected_matrix.json"
    matrix = read_json(matrix_path, default={}) or {}
    raw_rows: list[Any] = []
    source = "native_observed_fallback"
    if isinstance(matrix, Mapping):
        for field in ("expected_rows", "present_expected_rows"):
            candidate = matrix.get(field)
            if isinstance(candidate, list) and candidate:
                raw_rows = list(candidate)
                source = f"native_expected_matrix.{field}"
                break
    if not raw_rows:
        raw_rows = [dict(row) for row in native_rows]

    grouped: dict[_PlannedIdentity, int] = defaultdict(int)
    exclusions: list[dict[str, Any]] = []
    for source_row in raw_rows:
        if not isinstance(source_row, Mapping):
            exclusions.append({
                "runner_regime": "native_plan",
                "reason": "planned_identity_row_invalid",
            })
            continue
        execution_mode = str(source_row.get("execution_mode") or "").strip().lower()
        if execution_mode and execution_mode != "native_split":
            continue
        row = dict(source_row)
        if not row.get("direction"):
            row["direction"] = _native_plan_direction(
                row.get("backend_key") or row.get("backend")
            )
        if not row.get("comparison_backend"):
            row["comparison_backend"] = (
                row.get("backend_key") or row.get("backend")
            )
        identity = _planned_identity(row)
        if not _planned_identity_complete(identity):
            exclusions.append({
                "runner_regime": "native_plan",
                "identity": list(identity),
                "reason": "planned_identity_incomplete",
            })
            continue
        grouped[identity] += 1

    for identity, count in sorted(grouped.items()):
        if count > 1:
            exclusions.append({
                "runner_regime": "native_plan",
                "identity": list(identity),
                "reason": "duplicate_planned_identity",
                "match_count": count,
            })
    return set(grouped), source, exclusions


def _source_request_payload(
    run_dir: Path,
    source_request: Any,
) -> tuple[dict[str, Any], str]:
    relative = str(source_request or "").strip()
    if not relative:
        return {}, "source_request_not_recorded"
    path = Path(relative)
    if path.is_absolute():
        return {}, "source_request_not_relative"
    root = run_dir.resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return {}, "source_request_outside_run"
    payload = read_json(candidate, default={}) or {}
    if not isinstance(payload, Mapping):
        return {}, "source_request_invalid"
    return dict(payload), "source_request_loaded"


def _central_request_projection(
    run_dir: Path,
    result: Mapping[str, Any],
) -> tuple[_PlannedIdentity, dict[str, Any], list[str]]:
    central = (
        dict(result.get("request_identity") or {})
        if isinstance(result.get("request_identity"), Mapping)
        else {}
    )
    request, request_status = _source_request_payload(
        run_dir, result.get("source_request")
    )
    endpoint = (
        dict(request.get("endpoint_contract") or {})
        if isinstance(request.get("endpoint_contract"), Mapping)
        else {}
    )
    attestation = (
        dict(endpoint.get("output_endpoint_attestation") or {})
        if isinstance(endpoint.get("output_endpoint_attestation"), Mapping)
        else {}
    )

    direction = _canon_direction(
        central.get("source_run_id")
        or central.get("backend")
        or result.get("source_run_id")
        or result.get("backend")
        or "unknown"
    )
    stage1, _stage2 = direction_parts(direction)
    comparison_backend = canonical_backend(stage1) if stage1 else ""
    setup_id = str(
        central.get("setup_id") or result.get("source_setup_id") or ""
    ).strip().lower()
    model = _canon_model(
        central.get("model_id") or result.get("model_id")
    )
    case = _canon_case(
        central.get("case_id") or result.get("case_id")
    )
    task = str(
        central.get("task") or request.get("task") or result.get("task") or ""
    ).strip().lower()
    stage = str(
        central.get("stage") or endpoint.get("stage") or ""
    ).strip().lower()
    endpoint_hash = str(
        central.get("endpoint_contract_hash")
        or request.get("endpoint_contract_hash")
        or endpoint.get("endpoint_contract_hash")
        or result.get("endpoint_contract_hash")
        or ""
    ).strip().lower().removeprefix("sha256:")
    precision = str(
        central.get("runtime_precision_identity")
        or request.get("runtime_precision_identity")
        or result.get("runtime_precision_identity")
        or ""
    ).strip().lower()
    endpoint_complete = (
        central.get("endpoint_contract_complete")
        if isinstance(central.get("endpoint_contract_complete"), bool)
        else endpoint.get("endpoint_contract_complete")
    )

    errors = [
        str(value) for value in list(central.get("identity_errors") or [])
        if str(value)
    ]
    if central.get("identity_valid") is not True:
        errors.append("central_request_identity_invalid")
    if not all((model, case, direction, setup_id, comparison_backend)):
        errors.append("central_request_planned_identity_incomplete")
    if not all((task, stage, precision, endpoint_hash)):
        errors.append("central_request_exact_identity_incomplete")
    if len(endpoint_hash) != 64 or any(
        char not in "0123456789abcdef" for char in endpoint_hash
    ):
        errors.append("central_request_endpoint_hash_invalid")
    if endpoint_complete is not True:
        errors.append("central_request_endpoint_contract_incomplete")

    # The central identity is authoritative.  The referenced request enriches
    # it with endpoint attestation, but any recorded scalar drift is fatal.
    request_checks = (
        ("task", task, _canonical_scalar(request.get("task"))),
        (
            "runtime_precision_identity", precision,
            _canonical_scalar(request.get("runtime_precision_identity")),
        ),
        (
            "endpoint_contract_hash", endpoint_hash,
            _canonical_scalar(
                request.get("endpoint_contract_hash")
                or endpoint.get("endpoint_contract_hash")
            ).removeprefix("sha256:"),
        ),
    )
    for label, expected, observed in request_checks:
        if observed and expected != observed:
            errors.append(
                f"central_composed_request_{label}_conflict:{expected}!={observed}"
            )
    if attestation:
        attested_hash = str(
            attestation.get("endpoint_contract_hash") or ""
        ).strip().lower().removeprefix("sha256:")
        if not (
            attestation.get("attested") is True
            and str(attestation.get("status") or "").strip().lower()
            == "passed"
            and attested_hash == endpoint_hash
        ):
            errors.append("central_composed_request_attestation_invalid")

    planned: _PlannedIdentity = (
        model, case, direction, setup_id, comparison_backend,
    )
    projection = {
        "identity_valid": not errors,
        "identity_errors": sorted(set(errors)),
        "setup_id": setup_id,
        "setup_ids": [setup_id] if setup_id else [],
        "direction": direction,
        "source_run_id": direction,
        "producer_backend": comparison_backend,
        "comparison_backend": comparison_backend,
        "task": task,
        "stage": stage,
        "endpoint_contract_hash": endpoint_hash,
        "endpoint_contract_complete": endpoint_complete is True,
        "runtime_precision_identity": precision,
        "output_endpoint_attestation": attestation,
        "projection_source": (
            "central_quality_request_identity+composed_request"
            if request_status == "source_request_loaded"
            else "central_quality_request_identity"
        ),
        "source_request_status": request_status,
        "source_request": str(result.get("source_request") or ""),
    }
    return planned, projection, sorted(set(errors))


def _load_central_request_identity_index(
    run_dir: Path,
) -> tuple[
    dict[_PlannedIdentity, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    path = run_dir / "quality_management" / "central_quality_summary.json"
    payload = read_json(path, default={}) or {}
    out: dict[_PlannedIdentity, list[dict[str, Any]]] = defaultdict(list)
    exclusions: list[dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        return {}, exclusions
    for result in list(payload.get("results") or []):
        if not isinstance(result, Mapping):
            continue
        if str(result.get("variant") or "").strip().lower() not in {
            "composed", "split", "pipeline",
        }:
            continue
        case = _canon_case(result.get("case_id"))
        if not case or case == "full":
            continue
        planned, projection, errors = _central_request_projection(
            run_dir, result,
        )
        if errors:
            exclusions.append({
                "runner_regime": "generic_request_identity",
                "identity": list(planned),
                "reason": "central_request_identity_projection_invalid",
                "identity_errors": errors,
                "source_request": projection.get("source_request"),
            })
            continue
        out[planned].append(projection)
    return dict(out), exclusions


def _merge_central_request_projection(
    row: dict[str, Any],
    projection: Mapping[str, Any],
    *,
    allow_legacy_normalized_default_migration: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    identities = (
        dict(row.get("quality_request_identities_by_variant") or {})
        if isinstance(row.get("quality_request_identities_by_variant"), Mapping)
        else {}
    )
    existing = (
        dict(identities.get("composed") or {})
        if isinstance(identities.get("composed"), Mapping)
        else {}
    )
    merged = dict(existing)
    errors = [
        str(value) for value in list(existing.get("identity_errors") or [])
        if str(value)
    ]
    scalar_fields = (
        "setup_id", "direction", "source_run_id", "producer_backend",
        "comparison_backend", "task", "stage", "endpoint_contract_hash",
        "runtime_precision_identity",
    )
    for field in scalar_fields:
        normalize = _canonical_scalar
        if field in {"direction", "source_run_id"}:
            normalize = _canon_direction
        elif field in {"producer_backend", "comparison_backend"}:
            normalize = canonical_backend
        old = (
            normalize(existing.get(field))
            if existing.get(field) not in (None, "") else ""
        )
        new = (
            normalize(projection.get(field))
            if projection.get(field) not in (None, "") else ""
        )
        if field == "endpoint_contract_hash":
            old = old.removeprefix("sha256:")
            new = new.removeprefix("sha256:")
        if old and new and old != new:
            errors.append(
                f"archived_central_projection_{field}_conflict:{old}!={new}"
            )
        elif not old and new:
            merged[field] = projection.get(field)
    if "setup_ids" not in merged or merged.get("setup_ids") in (None, "", []):
        merged["setup_ids"] = projection.get("setup_ids")
    old_complete = existing.get("endpoint_contract_complete")
    new_complete = projection.get("endpoint_contract_complete")
    if (
        isinstance(old_complete, bool)
        and isinstance(new_complete, bool)
        and old_complete is not new_complete
    ):
        errors.append(
            "archived_central_projection_endpoint_contract_complete_conflict"
        )
    elif not isinstance(old_complete, bool) and isinstance(new_complete, bool):
        merged["endpoint_contract_complete"] = new_complete
    old_attestation = existing.get("output_endpoint_attestation")
    new_attestation = projection.get("output_endpoint_attestation")
    if isinstance(old_attestation, Mapping) and old_attestation:
        if (
            isinstance(new_attestation, Mapping) and new_attestation
            and json.dumps(dict(old_attestation), sort_keys=True, default=str)
            != json.dumps(dict(new_attestation), sort_keys=True, default=str)
        ):
            errors.append("archived_central_projection_attestation_conflict")
    elif isinstance(new_attestation, Mapping) and new_attestation:
        merged["output_endpoint_attestation"] = dict(new_attestation)

    errors.extend(
        str(value) for value in list(projection.get("identity_errors") or [])
        if str(value)
    )
    merged["identity_errors"] = sorted(set(errors))
    merged["identity_valid"] = bool(
        existing.get("identity_valid", True) is True
        and projection.get("identity_valid") is True
        and not errors
    )
    merged["projection_source"] = projection.get("projection_source")
    merged["source_request_status"] = projection.get("source_request_status")
    merged["source_request"] = projection.get("source_request")
    identities["composed"] = merged
    row["quality_request_identities_by_variant"] = identities
    row["_generic_identity_projection_source"] = merged.get("projection_source")

    # Pre-v2.77 ``normalized_results.json`` files materialised a missing
    # producer declaration as ``endpoint_contract_complete=False`` without a
    # provenance bit.  Offline report replay reads those archived rows directly,
    # so they never pass through the newer normalizer that writes
    # ``endpoint_contract_complete_explicit=False``.  Reconcile only that exact
    # legacy shape, only when the caller attests that these rows came from the
    # archived normalized-results input, and only after an unambiguous, valid
    # central request has supplied the same composed endpoint identity. Modern
    # explicit values keep their Boolean provenance marker and genuine
    # contradictions remain fatal.
    projection_attestation = projection.get("output_endpoint_attestation")
    projection_endpoint_hash = str(
        projection.get("endpoint_contract_hash") or ""
    ).strip().lower().removeprefix("sha256:")
    projection_attested_hash = (
        str(projection_attestation.get("endpoint_contract_hash") or "")
        .strip().lower().removeprefix("sha256:")
        if isinstance(projection_attestation, Mapping)
        else ""
    )
    projection_attestation_valid = bool(
        isinstance(projection_attestation, Mapping)
        and projection_attestation
        and projection_attestation.get("attested") is True
        and str(projection_attestation.get("status") or "").strip().lower()
        == "passed"
        and projection_endpoint_hash
        and projection_attested_hash == projection_endpoint_hash
    )
    if (
        allow_legacy_normalized_default_migration
        and projection_attestation_valid
        and not isinstance(
            row.get("endpoint_contract_complete_explicit"), bool,
        )
        and row.get("endpoint_contract_complete") is False
        and projection.get("endpoint_contract_complete") is True
        and projection.get("identity_valid") is True
        and merged.get("identity_valid") is True
        and not errors
    ):
        row["endpoint_contract_complete_explicit"] = False
        row["_legacy_endpoint_contract_default_migrated"] = True
    return row, sorted(set(errors))


def compute_cross_runner_report(
    run_dir: Path,
    generic_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_candidates: int = 3,
    native_run_dir: Path | None = None,
    generic_input_source: str = "",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    native_root = Path(native_run_dir) if native_run_dir is not None else run_dir
    native_rows = _load_native_rows(native_root)
    validation, validation_exclusions = _load_native_validation(native_root)
    planned, planned_source, plan_exclusions = (
        _load_planned_native_intersection(native_root, native_rows)
    )
    planned_intersection_explicit = planned_source.startswith(
        "native_expected_matrix."
    )
    request_index, request_index_exclusions = (
        _load_central_request_identity_index(run_dir)
    )
    identity_exclusions: list[dict[str, Any]] = [
        *validation_exclusions, *plan_exclusions,
    ]

    native_by_key: dict[_ExactIdentity, list[dict[str, Any]]] = defaultdict(list)
    native_outside_planned_intersection_count = 0
    for source in native_rows:
        row = dict(source)
        case = _canon_case(row.get("case") or row.get("case_id"))
        if not case or case == "full":
            continue
        planned_key = _planned_identity(row)
        if (
            planned_intersection_explicit
            and planned_key not in planned
        ):
            native_outside_planned_intersection_count += 1
            identity_exclusions.append({
                "runner_regime": "native",
                "identity": list(planned_key),
                "reason": "native_row_outside_planned_intersection",
            })
            continue
        cycle = _cycle_ms(row, native=True)
        if cycle is None:
            identity_exclusions.append({
                "runner_regime": "native",
                "identity": list(planned_key),
                "reason": "native_ranking_measurement_missing",
            })
            continue
        measured_fps, measured_fps_source = _measured_native_fps(row)
        theoretical_cycle, theoretical_cycle_source = _native_theoretical_cycle(row)
        identity_errors = _identity_resolution_errors(row)
        key = _exact_identity(row)
        if identity_errors:
            identity_exclusions.append({
                "runner_regime": "native", "identity": list(key),
                "reason": "variant_identity_conflict",
                "identity_errors": identity_errors,
            })
            continue
        if not _identity_complete(key):
            identity_exclusions.append({
                "runner_regime": "native", "identity": list(key),
                "reason": "native_exact_identity_incomplete",
            })
            continue
        row["_cycle_ms"] = cycle
        row["_measured_fps"] = measured_fps
        row["_measured_fps_source"] = measured_fps_source
        row["_theoretical_cycle_ms"] = theoretical_cycle
        row["_theoretical_cycle_source"] = theoretical_cycle_source
        native_by_key[key].append(row)

    # Only request-index errors that affect the frozen Native denominator are
    # Cross-runner exclusions.  Errors in the other Generic audit rows belong
    # to the broad Generic report, not to this bounded transfer experiment.
    identity_exclusions.extend(
        exclusion for exclusion in request_index_exclusions
        if tuple(exclusion.get("identity") or ()) in planned
    )

    generic_by_key: dict[_ExactIdentity, list[dict[str, Any]]] = defaultdict(list)
    generic_outside_planned_intersection_count = 0
    generic_inside_planned_intersection_count = 0
    generic_identity_projection_count = 0
    legacy_endpoint_contract_default_migration_count = 0
    legacy_endpoint_contract_default_migrations: list[dict[str, Any]] = []
    for source in generic_rows:
        row = dict(source)
        if str(row.get("runner_regime") or "generic").lower() != "generic":
            continue
        case = _canon_case(row.get("case_id") or row.get("case"))
        if (
            not case or case == "full"
            or str(row.get("variant") or "split").lower() == "full"
        ):
            continue
        planned_key = _planned_identity(row)
        if (
            planned_intersection_explicit
            and planned_key not in planned
        ):
            generic_outside_planned_intersection_count += 1
            continue
        generic_inside_planned_intersection_count += 1
        projected = request_index.get(planned_key, [])
        if len(projected) == 1:
            row, projection_errors = _merge_central_request_projection(
                row, projected[0],
                allow_legacy_normalized_default_migration=(
                    generic_input_source == "normalized_benchmark_results"
                ),
            )
            generic_identity_projection_count += 1
            legacy_endpoint_contract_default_migration_count += int(
                row.get("_legacy_endpoint_contract_default_migrated") is True
            )
            if row.get("_legacy_endpoint_contract_default_migrated") is True:
                legacy_endpoint_contract_default_migrations.append({
                    "planned_identity": list(planned_key),
                    "exact_identity": list(_exact_identity(row)),
                    "archived_row_value": False,
                    "archived_row_explicitness_marker": "absent",
                    "projected_value": projected[0].get(
                        "endpoint_contract_complete"
                    ),
                    "projection_source": projected[0].get(
                        "projection_source"
                    ),
                    "source_request": projected[0].get("source_request"),
                    "endpoint_contract_hash": projected[0].get(
                        "endpoint_contract_hash"
                    ),
                })
            if projection_errors:
                identity_exclusions.append({
                    "runner_regime": "generic",
                    "identity": list(planned_key),
                    "reason": "archived_central_request_projection_conflict",
                    "identity_errors": projection_errors,
                })
                continue
        elif len(projected) > 1:
            identity_exclusions.append({
                "runner_regime": "generic",
                "identity": list(planned_key),
                "reason": "ambiguous_central_request_identity_projection",
                "match_count": len(projected),
            })
            continue

        cycle = _cycle_ms(row)
        if cycle is None:
            identity_exclusions.append({
                "runner_regime": "generic",
                "identity": list(planned_key),
                "reason": "generic_ranking_measurement_missing",
            })
            continue
        identity_errors = _identity_resolution_errors(row)
        key = _exact_identity(row)
        if identity_errors:
            identity_exclusions.append({
                "runner_regime": "generic", "identity": list(key),
                "reason": "variant_identity_conflict",
                "identity_errors": identity_errors,
            })
            continue
        if not _identity_complete(key):
            identity_exclusions.append({
                "runner_regime": "generic", "identity": list(key),
                "planned_identity": list(planned_key),
                "reason": "generic_exact_identity_incomplete",
                "central_request_projection_match_count": len(projected),
            })
            continue
        row["_cycle_ms"] = cycle
        generic_by_key[key].append(row)

    pairs: list[dict[str, Any]] = []
    for key in sorted(set(generic_by_key) ^ set(native_by_key)):
        identity_exclusions.append({
            "identity": list(key),
            "reason": "missing_exact_counterpart",
            "generic_match_count": len(generic_by_key.get(key, [])),
            "native_match_count": len(native_by_key.get(key, [])),
            "validation_match_count": len(validation.get(key, [])),
        })
    for key in sorted(set(generic_by_key) & set(native_by_key)):
        generic_candidates = generic_by_key[key]
        native_candidates = native_by_key[key]
        validation_candidates = validation.get(key, [])
        if len(generic_candidates) != 1 or len(native_candidates) != 1 or len(validation_candidates) != 1:
            identity_exclusions.append({
                "identity": list(key),
                "reason": "ambiguous_or_missing_exact_identity_join",
                "generic_match_count": len(generic_candidates),
                "native_match_count": len(native_candidates),
                "validation_match_count": len(validation_candidates),
            })
            continue
        # Exact identity joins must never choose the fastest of duplicates.
        generic = generic_candidates[0]
        native = native_candidates[0]
        native_validation = validation_candidates[0]
        generic_contract = _b(generic.get("contract_consistent"))
        native_contract = _b(native_validation.get("contract_consistent"))
        generic_quality = _quality_decision(generic)
        native_quality = _quality_decision(native_validation)
        native_semantic = _b(native_validation.get("semantic_ok"))
        native_quality_evidence_verified = bool(
            _b(native.get("quality_evidence_verified")) is True
            or _b(native_validation.get("central_quality_evidence_verified"))
            is True
        )
        native_precision_quality_verified = bool(
            _b(native.get("precision_quality_verified")) is True
            or _b(native_validation.get("precision_quality_binding_verified"))
            is True
            or (
                _b(native.get("quality_evidence_verified")) is True
                and native.get("precision_quality_verified") in (None, "")
                and native_validation.get("precision_quality_binding_verified")
                in (None, "")
            )
        )
        generic_cycle = float(generic["_cycle_ms"])
        native_cycle = float(native["_cycle_ms"])
        native_measured_fps = _f(native.get("_measured_fps"))
        native_measured_cycle = (
            1000.0 / native_measured_fps
            if native_measured_fps is not None and native_measured_fps > 0
            else None
        )
        native_theoretical_cycle = _f(native.get("_theoretical_cycle_ms"))
        technical_checks = {
            "exact_identity_complete": _identity_complete(key),
            "generic_completed_task_measured": _completed_task_measurement(generic),
            "native_completed_task_measured": _completed_task_measurement(native, native=True),
            "generic_runtime_measurement_present": generic_cycle > 0,
            "generic_measurement_not_explicitly_invalid": (
                _b(generic.get("measurement_valid")) is not False
            ),
            "generic_terminal_failure_absent": (
                _b(generic.get("terminal_failure")) is not True
            ),
            "generic_runtime_not_explicitly_failed": (
                _b(generic.get("runtime_executable")) is not False
            ),
            "native_runtime_ok": _b(native.get("ok")) is True,
            "native_measured_makespan_present": native_measured_fps is not None,
            "native_output_endpoint_match": _b(native.get("output_endpoint_match")) is True,
            "native_comparison_stratum_explicit": _b(native.get("comparison_stratum_explicit")) is True,
        }
        quality_checks = {
            **technical_checks,
            "generic_quality_observation_valid": generic_quality in {"pass", "reference_close", "accuracy_loss", "not_estimable"},
            "native_quality_observation_valid": native_quality in {"pass", "reference_close", "accuracy_loss", "not_estimable"},
            "native_quality_evidence_verified": native_quality_evidence_verified,
            "native_precision_quality_verified": native_precision_quality_verified,
            "validation_semantic_ok": native_semantic is True,
            "validation_task_valid": _b(native_validation.get("task_valid")) is True,
            "validation_accuracy_gate_pass": _b(native_validation.get("accuracy_gate_pass")) is True,
        }
        claim_checks = {
            "generic_contract_consistent": generic_contract is True,
            "generic_quality_pass": generic_quality == "pass",
            "generic_quality_eligible": _b(
                generic.get("eligible_for_ranking")
                if generic.get("eligible_for_ranking") not in (None, "")
                else generic.get("ranking_eligible")
            ) is True,
            **technical_checks,
            "native_performance_claim_eligible": _b(native.get("performance_claim_eligible")) is True,
            "native_precision_quality_verified": native_precision_quality_verified,
            "native_quality_evidence_verified": native_quality_evidence_verified,
            "native_repeat_claim_gate_pass": _b(native.get("repeat_claim_gate_pass")) is True,
            "validation_contract_consistent": native_contract is True,
            "validation_semantic_ok": native_semantic is True,
            "validation_claim_ok": _b(native_validation.get("claim_ok")) is True,
            "validation_task_valid": _b(native_validation.get("task_valid")) is True,
            "validation_accuracy_gate_pass": _b(native_validation.get("accuracy_gate_pass")) is True,
            "validation_eligible_for_ranking": _b(native_validation.get("eligible_for_ranking")) is True,
            "validation_status_claim_ok": str(native_validation.get("status") or "").strip().lower() == "claim_ok",
            "validation_gate_status_eligible": str(native_validation.get("gate_status") or "").strip().lower() == "eligible",
            "generic_endpoint_contract_attested": _endpoint_attested(
                generic, _endpoint_contract_hash(generic),
            ),
            "native_endpoint_contract_attested": _endpoint_attested(
                native, _endpoint_contract_hash(native),
            ),
            "validation_endpoint_contract_attested": _endpoint_attested(
                native_validation, _endpoint_contract_hash(native_validation),
            ),
        }
        technical_eligible = all(technical_checks.values())
        quality_eligible = all(quality_checks.values())
        claim_eligible = all(claim_checks.values())
        exclusion_reasons = [
            name for name, passed in claim_checks.items() if not passed
        ]
        pairs.append({
            "model_id": key[0],
            "case_id": key[1],
            "direction": key[2],
            # ``contract_class`` used to be populated with the precision by
            # accident.  Keep the compatibility column, but bind it to the
            # attested endpoint contract that actually defines comparability.
            "contract_class": key[6],
            "precision": key[3],
            "setup_id": key[4],
            "comparison_backend": key[5],
            "output_endpoint_id": key[6],
            "exact_identity": list(key),
            "generic_cycle_ms": generic_cycle,
            "native_cycle_ms": native_cycle,
            "generic_to_native_ratio": generic_cycle / native_cycle if native_cycle > 0 else None,
            "generic_throughput_fps": 1000.0 / generic_cycle,
            # Compatibility name now carries the measured makespan/row FPS,
            # never a rate inferred from a stage-cycle model.
            "native_throughput_fps": native_measured_fps,
            "native_measured_throughput_fps": native_measured_fps,
            "native_measured_fps_source": native.get("_measured_fps_source"),
            "native_measured_cycle_ms": native_measured_cycle,
            "native_ranking_cycle_ms": native_cycle,
            "native_ranking_cycle_semantics": (
                "measured_makespan_inverse"
                if native_measured_fps is not None
                else "legacy_native_cycle_ordering_only"
            ),
            "native_theoretical_cycle_ms": native_theoretical_cycle,
            "native_theoretical_cycle_rate_fps": (
                1000.0 / native_theoretical_cycle
                if native_theoretical_cycle is not None and native_theoretical_cycle > 0
                else None
            ),
            "native_theoretical_cycle_source": native.get("_theoretical_cycle_source"),
            "generic_accuracy_assessment": generic.get("accuracy_assessment"),
            "native_accuracy_assessment": native_validation.get("accuracy_assessment"),
            "comparison_scope": "completed_task_only",
            "generic_task_quality_status": generic_quality,
            "native_task_quality_status": native_quality,
            "generic_contract_consistent": generic_contract,
            "native_contract_consistent": native_contract,
            "native_semantic_ok": native_semantic,
            "native_status": native.get("status"),
            "planned_native_intersection_member": True,
            "generic_identity_projection_source": generic.get(
                "_generic_identity_projection_source", "row_exact_identity",
            ),
            "eligible_for_technical_transfer": technical_eligible,
            "eligible_for_quality_transfer": quality_eligible,
            "eligible_for_claim_transfer": claim_eligible,
            # Compatibility alias remains the strict, claim-level gate.
            "eligible_for_transfer": claim_eligible,
            "technical_transfer_eligibility_checks": technical_checks,
            "quality_transfer_eligibility_checks": quality_checks,
            "claim_transfer_eligibility_checks": claim_checks,
            "transfer_eligibility_checks": claim_checks,
            "transfer_exclusion_reasons": exclusion_reasons,
            "generic_backend": generic.get("backend"),
            "native_backend": native.get("backend"),
        })

    # Ranking transfer is meaningful only inside one complete comparison
    # stratum.  In particular, rows from different physical setups,
    # comparison backends, precisions or endpoint contracts must never be
    # pooled merely because their model and direction names happen to match.
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(
            str(row["model_id"]),
            str(row["direction"]),
            str(row["precision"]),
            str(row["setup_id"]),
            str(row["comparison_backend"]),
            str(row["output_endpoint_id"]),
        )].append(row)

    groups: list[dict[str, Any]] = []
    for (model, direction, precision, setup_id, comparison_backend, endpoint_id), rows in sorted(grouped.items()):
        technical_rows = [
            row for row in rows
            if row.get("eligible_for_technical_transfer")
        ]
        quality_rows = [
            row for row in rows
            if row.get("eligible_for_quality_transfer")
        ]
        eligible_rows = [
            row for row in rows if row.get("eligible_for_claim_transfer")
        ]
        gx = [float(row["generic_cycle_ms"]) for row in eligible_rows]
        ny = [float(row["native_cycle_ms"]) for row in eligible_rows]
        if eligible_rows:
            granks = _rank(gx); nranks = _rank(ny)
            for row, grank, nrank in zip(eligible_rows, granks, nranks):
                row["generic_rank"] = grank
                row["native_rank"] = nrank
        n = len(eligible_rows)
        technical_gx = [float(row["generic_cycle_ms"]) for row in technical_rows]
        technical_ny = [float(row["native_cycle_ms"]) for row in technical_rows]
        quality_gx = [float(row["generic_cycle_ms"]) for row in quality_rows]
        quality_ny = [float(row["native_cycle_ms"]) for row in quality_rows]
        technical_concordant, technical_comparable = _pairwise_counts(
            technical_gx, technical_ny,
        )
        quality_concordant, quality_comparable = _pairwise_counts(
            quality_gx, quality_ny,
        )
        claim_concordant, claim_comparable = _pairwise_counts(gx, ny)
        if n >= minimum_candidates:
            status = "ok"
            evidence_tier = "claim_eligible"
        elif len(quality_rows) >= minimum_candidates:
            status = "quality_screening_only"
            evidence_tier = "quality_screening"
        elif len(technical_rows) >= minimum_candidates:
            status = "technical_diagnostic_only"
            evidence_tier = "technical_diagnostic"
        else:
            status = "insufficient_candidates"
            evidence_tier = "insufficient_candidates"
        result: dict[str, Any] = {
            "model_id": model,
            "direction": direction,
            "contract_class": endpoint_id,
            "precision": precision,
            "setup_id": setup_id,
            "comparison_backend": comparison_backend,
            "output_endpoint_id": endpoint_id,
            "paired_candidate_count": len(rows),
            "technical_candidate_count": len(technical_rows),
            "quality_candidate_count": len(quality_rows),
            "claim_candidate_count": n,
            "eligible_candidate_count": n,
            "minimum_candidates": minimum_candidates,
            "spearman_rho": _spearman(gx, ny) if n >= minimum_candidates else None,
            "kendall_tau_b": _kendall_tau_b(gx, ny) if n >= minimum_candidates else None,
            "pairwise_concordance": _pairwise_concordance(gx, ny) if n >= minimum_candidates else None,
            "pairwise_concordant_count": claim_concordant,
            "pairwise_comparable_count": claim_comparable,
            "diagnostic_pairwise_concordance": (
                claim_concordant / claim_comparable
                if claim_comparable else None
            ),
            "ratio_median": _median([float(row["generic_to_native_ratio"]) for row in eligible_rows]),
            "ratio_q1": _quantile([float(row["generic_to_native_ratio"]) for row in eligible_rows], 0.25),
            "ratio_q3": _quantile([float(row["generic_to_native_ratio"]) for row in eligible_rows], 0.75),
            "technical_spearman_rho": (
                _spearman(technical_gx, technical_ny)
                if len(technical_rows) >= minimum_candidates else None
            ),
            "technical_kendall_tau_b": (
                _kendall_tau_b(technical_gx, technical_ny)
                if len(technical_rows) >= minimum_candidates else None
            ),
            "technical_pairwise_concordance": (
                _pairwise_concordance(technical_gx, technical_ny)
                if len(technical_rows) >= minimum_candidates else None
            ),
            "technical_pairwise_concordant_count": technical_concordant,
            "technical_pairwise_comparable_count": technical_comparable,
            "technical_diagnostic_pairwise_concordance": (
                technical_concordant / technical_comparable
                if technical_comparable else None
            ),
            "quality_spearman_rho": (
                _spearman(quality_gx, quality_ny)
                if len(quality_rows) >= minimum_candidates else None
            ),
            "quality_kendall_tau_b": (
                _kendall_tau_b(quality_gx, quality_ny)
                if len(quality_rows) >= minimum_candidates else None
            ),
            "quality_pairwise_concordance": (
                _pairwise_concordance(quality_gx, quality_ny)
                if len(quality_rows) >= minimum_candidates else None
            ),
            "quality_pairwise_concordant_count": quality_concordant,
            "quality_pairwise_comparable_count": quality_comparable,
            "quality_diagnostic_pairwise_concordance": (
                quality_concordant / quality_comparable
                if quality_comparable else None
            ),
            "transfer_evidence_tier": evidence_tier,
            "status": status,
        }
        def add_shortlist_metrics(
            cohort_rows: Sequence[Mapping[str, Any]], prefix: str,
        ) -> None:
            for k in (1, 3, 5):
                result[f"{prefix}native_best_hit_at_{k}"] = None
                result[f"{prefix}native_regret_at_{k}"] = None
            if len(cohort_rows) < minimum_candidates:
                return
            native_best = min(
                cohort_rows,
                key=lambda row: float(row["native_cycle_ms"]),
            )
            native_best_cycle = float(native_best["native_cycle_ms"])
            generic_order = sorted(
                cohort_rows,
                key=lambda row: float(row["generic_cycle_ms"]),
            )
            for k in (1, 3, 5):
                if len(cohort_rows) < k:
                    continue
                shortlist = generic_order[:k]
                hit = native_best in shortlist
                best_native_in_shortlist = min(
                    float(row["native_cycle_ms"]) for row in shortlist
                )
                result[f"{prefix}native_best_hit_at_{k}"] = bool(hit)
                result[f"{prefix}native_regret_at_{k}"] = (
                    (best_native_in_shortlist - native_best_cycle)
                    / native_best_cycle
                    if native_best_cycle > 0 else None
                )

        add_shortlist_metrics(technical_rows, "technical_")
        add_shortlist_metrics(quality_rows, "quality_")
        add_shortlist_metrics(eligible_rows, "")
        groups.append(result)

    valid_groups = [row for row in groups if row.get("status") == "ok"]
    quality_groups = [
        row for row in groups
        if int(row.get("quality_candidate_count") or 0) >= minimum_candidates
    ]
    technical_groups = [
        row for row in groups
        if int(row.get("technical_candidate_count") or 0) >= minimum_candidates
    ]

    def technical_micro_summary(
        source_groups: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        concordant = sum(
            int(row.get("technical_pairwise_concordant_count") or 0)
            for row in source_groups
        )
        comparable = sum(
            int(row.get("technical_pairwise_comparable_count") or 0)
            for row in source_groups
        )
        hits = [
            bool(row.get("technical_native_best_hit_at_1"))
            for row in source_groups
            if row.get("technical_native_best_hit_at_1") is not None
        ]
        regrets = [
            float(row["technical_native_regret_at_1"])
            for row in source_groups
            if row.get("technical_native_regret_at_1") is not None
        ]
        return {
            "qualified_group_count": len(source_groups),
            "concordant_pair_count": concordant,
            "comparable_pair_count": comparable,
            "concordance_fraction": (
                concordant / comparable if comparable else None
            ),
            "hit_at_1_count": sum(hits),
            "hit_at_1_comparable_group_count": len(hits),
            "hit_at_1_fraction": sum(hits) / len(hits) if hits else None,
            "regret_at_1_mean": (
                sum(regrets) / len(regrets) if regrets else None
            ),
            "regret_at_1_median": _median(regrets),
        }

    technical_micro = technical_micro_summary(technical_groups)
    technical_by_backend: dict[str, dict[str, Any]] = {}
    for backend in sorted({
        str(row.get("comparison_backend") or "")
        for row in technical_groups
        if str(row.get("comparison_backend") or "")
    }):
        technical_by_backend[backend] = technical_micro_summary([
            row for row in technical_groups
            if str(row.get("comparison_backend") or "") == backend
        ])

    macro = {
        "group_count": len(groups),
        "validated_group_count": len(valid_groups),
        "technical_group_count": len(technical_groups),
        "quality_group_count": len(quality_groups),
        "technical_subminimum_group_count": sum(
            0 < int(row.get("technical_candidate_count") or 0)
            < minimum_candidates
            for row in groups
        ),
        "macro_spearman_rho": _median([row["spearman_rho"] for row in valid_groups if row.get("spearman_rho") is not None]),
        "macro_kendall_tau_b": _median([row["kendall_tau_b"] for row in valid_groups if row.get("kendall_tau_b") is not None]),
        "macro_pairwise_concordance": _median([row["pairwise_concordance"] for row in valid_groups if row.get("pairwise_concordance") is not None]),
        "macro_native_best_hit_at_1": _median([
            float(bool(row["native_best_hit_at_1"]))
            for row in valid_groups
            if row.get("native_best_hit_at_1") is not None
        ]),
        "macro_native_regret_at_1": _median([
            row["native_regret_at_1"] for row in valid_groups
            if row.get("native_regret_at_1") is not None
        ]),
        "macro_technical_native_best_hit_at_1": _median([
            float(bool(row["technical_native_best_hit_at_1"]))
            for row in technical_groups
            if row.get("technical_native_best_hit_at_1") is not None
        ]),
        "macro_technical_native_regret_at_1": _median([
            row["technical_native_regret_at_1"] for row in technical_groups
            if row.get("technical_native_regret_at_1") is not None
        ]),
        "macro_quality_native_best_hit_at_1": _median([
            float(bool(row["quality_native_best_hit_at_1"]))
            for row in quality_groups
            if row.get("quality_native_best_hit_at_1") is not None
        ]),
        "macro_quality_native_regret_at_1": _median([
            row["quality_native_regret_at_1"] for row in quality_groups
            if row.get("quality_native_regret_at_1") is not None
        ]),
        "macro_native_regret_at_5": _median([row["native_regret_at_5"] for row in valid_groups if row.get("native_regret_at_5") is not None]),
        "macro_technical_spearman_rho": _median([
            row["technical_spearman_rho"] for row in technical_groups
            if row.get("technical_spearman_rho") is not None
        ]),
        "macro_quality_spearman_rho": _median([
            row["quality_spearman_rho"] for row in quality_groups
            if row.get("quality_spearman_rho") is not None
        ]),
        "technical_micro_pairwise_concordant_count": technical_micro[
            "concordant_pair_count"
        ],
        "technical_micro_pairwise_comparable_count": technical_micro[
            "comparable_pair_count"
        ],
        "technical_micro_pairwise_concordance": technical_micro[
            "concordance_fraction"
        ],
        "technical_hit_at_1_count": technical_micro["hit_at_1_count"],
        "technical_hit_at_1_comparable_group_count": technical_micro[
            "hit_at_1_comparable_group_count"
        ],
        "technical_hit_at_1_fraction": technical_micro[
            "hit_at_1_fraction"
        ],
        "technical_regret_at_1_mean": technical_micro["regret_at_1_mean"],
        "technical_regret_at_1_median": technical_micro[
            "regret_at_1_median"
        ],
        "technical_metrics_by_backend": technical_by_backend,
        "status": (
            "ok" if valid_groups
            else (
                "quality_screening_only" if quality_groups
                else (
                    "technical_diagnostic_only" if technical_groups
                    else "insufficient_candidates"
                )
            )
        ),
    }
    paired_planned_identities: set[_PlannedIdentity] = {
        (
            str(row.get("model_id") or ""),
            str(row.get("case_id") or ""),
            str(row.get("direction") or ""),
            str(row.get("setup_id") or ""),
            str(row.get("comparison_backend") or ""),
        )
        for row in pairs
    }
    return {
        "schema": "onnx-splitpoint/cross-runner-ranking-transfer",
        "schema_version": 5,
        "planned_native_intersection_source": planned_source,
        "planned_native_intersection_explicit": planned_intersection_explicit,
        "planned_native_intersection_count": len(planned),
        "planned_native_intersection_paired_count": len(
            paired_planned_identities
        ),
        "planned_native_intersection_complete": bool(
            planned_intersection_explicit
            and not plan_exclusions
            and paired_planned_identities == planned
        ),
        "generic_inside_planned_intersection_count": (
            generic_inside_planned_intersection_count
        ),
        "generic_outside_planned_intersection_count": (
            generic_outside_planned_intersection_count
        ),
        "generic_identity_projection_count": generic_identity_projection_count,
        "generic_input_source": generic_input_source,
        "legacy_endpoint_contract_default_migration_count": (
            legacy_endpoint_contract_default_migration_count
        ),
        "legacy_endpoint_contract_default_migrations": (
            legacy_endpoint_contract_default_migrations
        ),
        "native_outside_planned_intersection_count": (
            native_outside_planned_intersection_count
        ),
        "pair_count": len(pairs),
        "technical_pair_count": sum(
            bool(row.get("eligible_for_technical_transfer")) for row in pairs
        ),
        "quality_pair_count": sum(
            bool(row.get("eligible_for_quality_transfer")) for row in pairs
        ),
        "claim_pair_count": sum(
            bool(row.get("eligible_for_claim_transfer")) for row in pairs
        ),
        "eligible_pair_count": sum(bool(row.get("eligible_for_transfer")) for row in pairs),
        "identity_exclusion_count": len(identity_exclusions),
        "identity_exclusion_reason_counts": dict(sorted(Counter(
            str(row.get("reason") or "unspecified")
            for row in identity_exclusions
        ).items())),
        "identity_exclusions": identity_exclusions,
        "groups": groups,
        "macro": macro,
        "pairs": pairs,
        "status": macro["status"],
        "generic_source_run_dir": str(run_dir),
        "native_source_run_dir": str(native_root),
        "native_source_is_separate": native_root != run_dir,
    }


def markdown_for_cross_runner(payload: Mapping[str, Any]) -> str:
    macro = (
        payload.get("macro")
        if isinstance(payload.get("macro"), Mapping)
        else {}
    )
    lines = [
        "# Generic-to-Native ranking transfer",
        "",
        f"Status: `{payload.get('status', 'unavailable')}`  ",
        (
            f"Candidate pairs: **{payload.get('pair_count', 0)}**; "
            f"technical: **{payload.get('technical_pair_count', 0)}**, "
            f"quality: **{payload.get('quality_pair_count', 0)}**, "
            f"claim: **{payload.get('claim_pair_count', payload.get('eligible_pair_count', 0))}**"
        ),
        (
            "Planned Native intersection: "
            f"**{payload.get('planned_native_intersection_paired_count', payload.get('pair_count', 0))}/"
            f"{payload.get('planned_native_intersection_count', payload.get('pair_count', 0))}**; "
            "Generic rows outside this denominator: "
            f"**{payload.get('generic_outside_planned_intersection_count', 0)}**"
        ),
        (
            "Legacy normalized endpoint defaults reconciled by exact central "
            "request projection: "
            f"**{payload.get('legacy_endpoint_contract_default_migration_count', 0)}**"
        ),
        (
            "Qualified technical micro-concordance: "
            f"**{macro.get('technical_micro_pairwise_concordant_count', 0)}/"
            f"{macro.get('technical_micro_pairwise_comparable_count', 0)}** "
            f"({macro.get('technical_micro_pairwise_concordance', '')}); "
            "Hit@1: "
            f"**{macro.get('technical_hit_at_1_count', 0)}/"
            f"{macro.get('technical_hit_at_1_comparable_group_count', 0)}**; "
            f"Regret@1 median: **{macro.get('technical_regret_at_1_median', '')}**"
        ),
        "",
        "| Model | Direction | Contract | technical n | technical pairwise | technical Hit@1 | technical Regret@1 | quality n | quality pairwise | quality Hit@1 | quality Regret@1 | claim n | claim pairwise | claim Hit@1 | claim Regret@1 | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("groups") or []):
        lines.append(
            "| " + " | ".join(
                str(row.get(key, "") if row.get(key) is not None else "")
                for key in (
                    "model_id", "direction", "contract_class",
                    "technical_candidate_count",
                    "technical_pairwise_concordance",
                    "technical_native_best_hit_at_1",
                    "technical_native_regret_at_1",
                    "quality_candidate_count",
                    "quality_pairwise_concordance",
                    "quality_native_best_hit_at_1",
                    "quality_native_regret_at_1",
                    "claim_candidate_count", "pairwise_concordance",
                    "native_best_hit_at_1",
                    "native_regret_at_1", "status",
                )
            ) + " |"
        )
    lines.extend([
        "",
        "Technical, quality and claim cohorts are evaluated separately. A qualified group requires at least three exact pairs in the respective cohort; technical membership does not imply a quality pass. Absolute Native throughput and energy remain Native-runner evidence; this report tests whether the Generic ordering is a usable shortlist surrogate.",
        "",
    ])
    return "\n".join(lines)


__all__ = ["compute_cross_runner_report", "markdown_for_cross_runner"]
