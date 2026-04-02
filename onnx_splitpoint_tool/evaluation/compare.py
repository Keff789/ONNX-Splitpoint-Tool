from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


METRICS = ("full", "part1", "part2", "composed", "transfer")


def compare_predictions_vs_measurements(bundle: dict[str, Any]) -> dict[str, Any]:
    predictions = bundle.get("prediction_rows") or []
    measurements = bundle.get("measurement_rows") or []
    artifacts = bundle.get("artifact_records") or []

    pred_index = _build_prediction_index(predictions)
    rows: list[dict[str, Any]] = []
    for measurement in measurements:
        pred = pred_index.get(str(measurement.get("candidate_id") or ""))
        if pred is None:
            pred = pred_index.get(str(measurement.get("case_id") or ""))
        row = _merge_row(pred or {}, measurement)
        rows.append(row)

    _assign_ranks(rows)
    _assign_pareto_flags(rows)

    return {
        "suite_meta": bundle.get("suite_meta") or {},
        "rows": rows,
        "backend_summaries": _build_backend_summaries(rows),
        "failure_matrix": _build_failure_matrix(rows),
        "highlights": _build_highlights(rows),
        "provenance_notes": _build_provenance_notes(artifacts),
        "artifact_records": artifacts,
    }


def _build_prediction_index(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in predictions:
        if not isinstance(row, dict):
            continue
        for key in (row.get("candidate_id"), row.get("case_id"), row.get("split_id")):
            if key not in (None, "") and str(key) not in index:
                index[str(key)] = row
    return index


def _merge_row(pred: dict[str, Any], measurement: dict[str, Any]) -> dict[str, Any]:
    row = {
        "suite_id": measurement.get("suite_id"),
        "candidate_id": measurement.get("candidate_id") or pred.get("candidate_id"),
        "case_id": measurement.get("case_id") or pred.get("case_id"),
        "split_id": pred.get("split_id"),
        "split_node": pred.get("split_node"),
        "variant_id": measurement.get("variant_id"),
        "run_tag": measurement.get("run_tag"),
        "backend_stage1": measurement.get("backend_stage1"),
        "backend_stage2": measurement.get("backend_stage2"),
        "backend_label": measurement.get("backend_label"),
        "status": measurement.get("status") or "missing",
        "reason_code": measurement.get("reason_code") or "",
        "reason_detail": measurement.get("reason_detail") or "",
        "validation_mode": measurement.get("validation_mode"),
        "validation_pass": measurement.get("validation_pass"),
        "measured": bool(measurement.get("measured")),
        "artifacts_present": measurement.get("artifacts_present"),
        "selection_rank": pred.get("selection_rank"),
        "selection_score": pred.get("selection_score"),
        "selection_reason": pred.get("selection_reason"),
        "predicted_feasible": pred.get("predicted_feasible"),
        "actual_feasible": _actual_feasible(measurement),
        "predicted_pareto": pred.get("predicted_pareto"),
        "variant_records": measurement.get("variant_records") or {},
        "prediction_raw": pred.get("raw") or {},
        "measurement_raw": measurement.get("raw") or {},
    }
    for metric in METRICS:
        pred_value = _finite(pred.get(f"pred_{metric}_ms"))
        meas_value = _finite(measurement.get(f"measured_{metric}_ms"))
        row[f"pred_{metric}_ms"] = pred_value
        row[f"measured_{metric}_ms"] = meas_value
        row[f"abs_error_{metric}_ms"] = _abs_error(pred_value, meas_value)
        row[f"rel_error_{metric}"] = _rel_error(pred_value, meas_value)
        row[f"signed_error_{metric}_ms"] = _signed_error(pred_value, meas_value)
    return row


def _actual_feasible(measurement: dict[str, Any]) -> bool | None:
    status = str(measurement.get("status") or "")
    validation = measurement.get("validation_pass")
    if status in {"error", "missing"}:
        return False
    if validation is False:
        return False
    if measurement.get("measured"):
        return True
    if status == "skipped":
        return False
    return None


def _assign_ranks(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("backend_label") or "unknown")].append(row)
    for group_rows in grouped.values():
        _rank_metric(group_rows, source_key="pred_composed_ms", out_key="predicted_rank", fallback_key="pred_full_ms")
        _rank_metric(group_rows, source_key="measured_composed_ms", out_key="measured_rank", fallback_key="measured_full_ms")
        for row in group_rows:
            pr = row.get("predicted_rank")
            mr = row.get("measured_rank")
            row["rank_shift"] = (mr - pr) if isinstance(pr, int) and isinstance(mr, int) else None


def _rank_metric(rows: list[dict[str, Any]], *, source_key: str, out_key: str, fallback_key: str) -> None:
    ranked = []
    for row in rows:
        value = _finite(row.get(source_key))
        if value is None:
            value = _finite(row.get(fallback_key))
        if value is None:
            row[out_key] = None
            continue
        ranked.append((value, str(row.get("candidate_id") or ""), row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    for idx, (_, _, row) in enumerate(ranked, start=1):
        row[out_key] = idx


def _assign_pareto_flags(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("backend_label") or "unknown")].append(row)
    for group_rows in grouped.values():
        _mark_pareto(group_rows, x_key="pred_composed_ms", y_key="pred_transfer_ms", out_key="predicted_pareto")
        _mark_pareto(group_rows, x_key="measured_composed_ms", y_key="measured_transfer_ms", out_key="actual_pareto")


def _mark_pareto(rows: list[dict[str, Any]], *, x_key: str, y_key: str, out_key: str) -> None:
    points: list[tuple[float, float, dict[str, Any]]] = []
    for row in rows:
        x = _finite(row.get(x_key))
        y = _finite(row.get(y_key))
        if x is None or y is None:
            row.setdefault(out_key, None)
            continue
        points.append((x, y, row))
    for _, _, row in points:
        row[out_key] = True
    for i, (x1, y1, row1) in enumerate(points):
        for j, (x2, y2, row2) in enumerate(points):
            if i == j:
                continue
            if x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1):
                row1[out_key] = False
                break


def _build_backend_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("backend_label") or "unknown")].append(row)

    summaries: list[dict[str, Any]] = []
    for backend, group_rows in sorted(grouped.items()):
        predicted_ranks = []
        measured_ranks = []
        for row in group_rows:
            pr = row.get("predicted_rank")
            mr = row.get("measured_rank")
            if isinstance(pr, int) and isinstance(mr, int):
                predicted_ranks.append(pr)
                measured_ranks.append(mr)
        summary = {
            "backend": backend,
            "row_count": len(group_rows),
            "measured_count": sum(1 for row in group_rows if row.get("measured")),
            "ok_count": sum(1 for row in group_rows if row.get("status") == "ok"),
            "failed_count": sum(1 for row in group_rows if row.get("status") in {"error", "missing"}),
            "feasible_match_rate": _match_rate(group_rows, "predicted_feasible", "actual_feasible"),
            "pareto_match_rate": _match_rate(group_rows, "predicted_pareto", "actual_pareto"),
            "rank_correlation": _pearson(predicted_ranks, measured_ranks),
            "top1_hit_rate": _topk_hit_rate(group_rows, 1),
            "top3_hit_rate": _topk_hit_rate(group_rows, 3),
            "top5_hit_rate": _topk_hit_rate(group_rows, 5),
        }
        for metric in METRICS:
            summary[f"mean_abs_error_{metric}_ms"] = _mean([row.get(f"abs_error_{metric}_ms") for row in group_rows])
            summary[f"mean_rel_error_{metric}"] = _mean([row.get(f"rel_error_{metric}") for row in group_rows])
        summaries.append(summary)
    return summaries


def _match_rate(rows: list[dict[str, Any]], left_key: str, right_key: str) -> float | None:
    pairs = [(row.get(left_key), row.get(right_key)) for row in rows]
    valid = [(a, b) for a, b in pairs if isinstance(a, bool) and isinstance(b, bool)]
    if not valid:
        return None
    return sum(1 for a, b in valid if a == b) / float(len(valid))


def _topk_hit_rate(rows: list[dict[str, Any]], k: int) -> float | None:
    if not rows:
        return None
    pred_sorted = [row for row in sorted(rows, key=_pred_sort_key) if row.get("predicted_rank") is not None]
    meas_sorted = [row for row in sorted(rows, key=_meas_sort_key) if row.get("measured_rank") is not None]
    if not pred_sorted or not meas_sorted:
        return None
    k_eff = min(k, len(pred_sorted), len(meas_sorted))
    pred_ids = {str(row.get("candidate_id")) for row in pred_sorted[:k_eff]}
    meas_ids = {str(row.get("candidate_id")) for row in meas_sorted[:k_eff]}
    if not pred_ids:
        return None
    return len(pred_ids & meas_ids) / float(k_eff)


def _pred_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rank = row.get("predicted_rank")
    return (rank is None, rank or 10**9, str(row.get("candidate_id") or ""))


def _meas_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rank = row.get("measured_rank")
    return (rank is None, rank or 10**9, str(row.get("candidate_id") or ""))


def _build_failure_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        backend = str(row.get("backend_label") or "unknown")
        status = str(row.get("status") or "missing")
        reason = str(row.get("reason_code") or ("none" if status == "ok" else "unknown"))
        counts[(backend, status, reason)] += 1
    return [
        {"backend": backend, "status": status, "reason_code": reason, "count": count}
        for (backend, status, reason), count in sorted(counts.items())
    ]


def _build_highlights(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    valid = [row for row in rows if _finite(row.get("signed_error_composed_ms")) is not None]
    valid.sort(key=lambda row: float(row.get("signed_error_composed_ms") or 0.0))
    under = [_highlight_row(row) for row in valid[:5]]
    over = [_highlight_row(row) for row in valid[-5:]][::-1]
    rank_shifts = [row for row in rows if isinstance(row.get("rank_shift"), int)]
    rank_shifts.sort(key=lambda row: abs(int(row.get("rank_shift") or 0)), reverse=True)
    return {
        "largest_overestimates": over,
        "largest_underestimates": under,
        "largest_rank_shifts": [_highlight_row(row) for row in rank_shifts[:5]],
    }


def _highlight_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id"),
        "case_id": row.get("case_id"),
        "backend": row.get("backend_label"),
        "pred_composed_ms": row.get("pred_composed_ms"),
        "measured_composed_ms": row.get("measured_composed_ms"),
        "signed_error_composed_ms": row.get("signed_error_composed_ms"),
        "predicted_rank": row.get("predicted_rank"),
        "measured_rank": row.get("measured_rank"),
        "rank_shift": row.get("rank_shift"),
        "status": row.get("status"),
        "reason_code": row.get("reason_code"),
    }


def _build_provenance_notes(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for record in artifacts:
        generated = record.get("generated_from_outputs")
        validation = record.get("validation_pass")
        if generated is False or validation is False:
            notes.append(
                {
                    "case_id": record.get("case_id"),
                    "variant_id": record.get("variant_id"),
                    "artifact_type": record.get("artifact_type"),
                    "generated_from_outputs": generated,
                    "validation_pass": validation,
                    "manifest_path": record.get("manifest_path"),
                }
            )
    return notes


def _mean(values: list[Any]) -> float | None:
    clean = [_finite(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return None
    return sum(clean) / float(len(clean))


def _pearson(xs: list[int], ys: list[int]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / float(len(xs))
    y_mean = sum(ys) / float(len(ys))
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def _abs_error(pred: float | None, meas: float | None) -> float | None:
    if pred is None or meas is None:
        return None
    return abs(meas - pred)


def _signed_error(pred: float | None, meas: float | None) -> float | None:
    if pred is None or meas is None:
        return None
    return meas - pred


def _rel_error(pred: float | None, meas: float | None) -> float | None:
    if pred is None or meas is None or pred == 0.0:
        return None
    return (meas - pred) / pred


def _finite(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        num = float(value)
        if math.isfinite(num):
            return num
    except Exception:
        return None
    return None