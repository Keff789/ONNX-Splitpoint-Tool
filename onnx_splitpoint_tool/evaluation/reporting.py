from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

try:
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - plotting is best effort
    Figure = None  # type: ignore[assignment]


CSV_FIELDS = [
    "suite_id",
    "candidate_id",
    "case_id",
    "variant_id",
    "backend_stage1",
    "backend_stage2",
    "backend_label",
    "status",
    "reason_code",
    "validation_mode",
    "validation_pass",
    "measured",
    "predicted_feasible",
    "actual_feasible",
    "predicted_rank",
    "measured_rank",
    "rank_shift",
    "predicted_pareto",
    "actual_pareto",
    "pred_full_ms",
    "measured_full_ms",
    "abs_error_full_ms",
    "rel_error_full",
    "pred_part1_ms",
    "measured_part1_ms",
    "abs_error_part1_ms",
    "rel_error_part1",
    "pred_part2_ms",
    "measured_part2_ms",
    "abs_error_part2_ms",
    "rel_error_part2",
    "pred_composed_ms",
    "measured_composed_ms",
    "abs_error_composed_ms",
    "rel_error_composed",
    "pred_transfer_ms",
    "measured_transfer_ms",
    "abs_error_transfer_ms",
    "rel_error_transfer",
]


def write_evaluation_outputs(
    comparison: dict[str, Any],
    *,
    calibration_bundle: dict[str, Any],
    out_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    suite_meta = comparison.get("suite_meta") or {}
    rows = comparison.get("rows") or []
    if out_dir is None:
        raise ValueError("out_dir is required for evaluation reporting")
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    plots_dir = out_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    prediction_json = out_path / "prediction_vs_measurement.json"
    prediction_csv = out_path / "prediction_vs_measurement.csv"
    failure_json = out_path / "failure_matrix.json"
    failure_csv = out_path / "failure_matrix.csv"
    summary_md = out_path / "comparison_summary.md"
    calibration_json = out_path / "backend_calibration.json"

    _write_json(
        prediction_json,
        {
            "suite_meta": suite_meta,
            "backend_summaries": comparison.get("backend_summaries") or [],
            "rows": rows,
            "highlights": comparison.get("highlights") or {},
            "provenance_notes": comparison.get("provenance_notes") or [],
        },
    )
    _write_csv(prediction_csv, rows, CSV_FIELDS)
    _write_json(failure_json, comparison.get("failure_matrix") or [])
    _write_csv(failure_csv, comparison.get("failure_matrix") or [], ["backend", "status", "reason_code", "count"])
    _write_json(calibration_json, calibration_bundle)

    plot_paths = []
    for metric in ("full", "part1", "part2", "composed"):
        plot_path = _write_scatter_plot(plots_dir, rows, metric)
        if plot_path is not None:
            plot_paths.append(plot_path)
    rank_plot = _write_rank_shift_plot(plots_dir, rows)
    if rank_plot is not None:
        plot_paths.append(rank_plot)

    summary_text = _build_summary_text(
        suite_meta,
        comparison.get("backend_summaries") or [],
        comparison.get("failure_matrix") or [],
        comparison.get("highlights") or {},
        comparison.get("provenance_notes") or [],
        calibration_bundle,
        plot_paths,
    )
    summary_md.write_text(summary_text, encoding="utf-8")

    if log is not None:
        log(f"[evaluation] wrote {prediction_csv.name}, {failure_csv.name}, {summary_md.name}")

    return {
        "ok": True,
        "evaluation_dir": str(out_path),
        "row_count": len(rows),
        "backend_summaries": comparison.get("backend_summaries") or [],
        "summary_markdown": str(summary_md),
        "artifacts": {
            "prediction_vs_measurement_csv": str(prediction_csv),
            "prediction_vs_measurement_json": str(prediction_json),
            "failure_matrix_csv": str(failure_csv),
            "failure_matrix_json": str(failure_json),
            "comparison_summary_md": str(summary_md),
            "backend_calibration_json": str(calibration_json),
            "plots": [str(path) for path in plot_paths],
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_scatter_plot(plots_dir: Path, rows: list[dict[str, Any]], metric: str) -> Path | None:
    if Figure is None:
        return None
    points = []
    for row in rows:
        pred = _finite(row.get(f"pred_{metric}_ms"))
        meas = _finite(row.get(f"measured_{metric}_ms"))
        if pred is None or meas is None:
            continue
        points.append((pred, meas, str(row.get("backend_label") or "unknown")))
    if not points:
        return None
    max_value = max(max(pred, meas) for pred, meas, _ in points)
    fig = Figure(figsize=(5.6, 4.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    xs = [pred for pred, _, _ in points]
    ys = [meas for _, meas, _ in points]
    ax.scatter(xs, ys, s=18)
    ax.plot([0.0, max_value], [0.0, max_value], linestyle="--", linewidth=1)
    ax.set_title(f"Predicted vs measured ({metric})")
    ax.set_xlabel("Predicted ms")
    ax.set_ylabel("Measured ms")
    out_path = plots_dir / f"predicted_vs_measured_{metric}.png"
    fig.savefig(str(out_path), format="png", bbox_inches="tight")
    return out_path


def _write_rank_shift_plot(plots_dir: Path, rows: list[dict[str, Any]]) -> Path | None:
    if Figure is None:
        return None
    points = []
    for row in rows:
        shift = row.get("rank_shift")
        label = str(row.get("candidate_id") or "")
        if isinstance(shift, int):
            points.append((label, shift))
    if not points:
        return None
    points = sorted(points, key=lambda item: abs(item[1]), reverse=True)[:12]
    fig = Figure(figsize=(7.0, 4.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    labels = [label for label, _ in points]
    values = [shift for _, shift in points]
    ax.bar(range(len(values)), values)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Largest rank shifts")
    ax.set_ylabel("Measured rank - predicted rank")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    out_path = plots_dir / "rank_shift.png"
    fig.savefig(str(out_path), format="png", bbox_inches="tight")
    return out_path


def _build_summary_text(
    suite_meta: dict[str, Any],
    backend_summaries: list[dict[str, Any]],
    failure_matrix: list[dict[str, Any]],
    highlights: dict[str, list[dict[str, Any]]],
    provenance_notes: list[dict[str, Any]],
    calibration_bundle: dict[str, Any],
    plot_paths: list[Path],
) -> str:
    lines = ["# Prediction vs Measurement", ""]
    suite_id = suite_meta.get("suite_id") or "unknown"
    hardware_signature = suite_meta.get("hardware_signature") or "unknown"
    lines.append(f"Suite: `{suite_id}`")
    lines.append(f"Hardware signature: `{hardware_signature}`")
    lines.append("")
    if backend_summaries:
        lines.append("## Backend summary")
        lines.append("")
        lines.append("| backend | rows | ok | failed | rank corr | top-3 hit | MAE composed (ms) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for item in backend_summaries:
            lines.append(
                "| {backend} | {rows} | {ok} | {failed} | {rank} | {top3} | {mae} |".format(
                    backend=item.get("backend") or "unknown",
                    rows=item.get("row_count") or 0,
                    ok=item.get("ok_count") or 0,
                    failed=item.get("failed_count") or 0,
                    rank=_fmt(item.get("rank_correlation")),
                    top3=_fmt(item.get("top3_hit_rate")),
                    mae=_fmt(item.get("mean_abs_error_composed_ms")),
                )
            )
        lines.append("")
    if failure_matrix:
        lines.append("## Failure matrix")
        lines.append("")
        for item in failure_matrix[:12]:
            lines.append(
                f"- {item.get('backend')}: {item.get('status')} / {item.get('reason_code')} -> {item.get('count')}"
            )
        lines.append("")
    if highlights:
        lines.append("## Highlights")
        lines.append("")
        for section, entries in (
            ("Largest overestimates", highlights.get("largest_overestimates") or []),
            ("Largest underestimates", highlights.get("largest_underestimates") or []),
            ("Largest rank shifts", highlights.get("largest_rank_shifts") or []),
        ):
            if not entries:
                continue
            lines.append(f"### {section}")
            for entry in entries:
                lines.append(
                    "- {candidate} ({backend}): pred={pred} ms, meas={meas} ms, rank shift={shift}".format(
                        candidate=entry.get("candidate_id") or entry.get("case_id") or "unknown",
                        backend=entry.get("backend") or "unknown",
                        pred=_fmt(entry.get("pred_composed_ms")),
                        meas=_fmt(entry.get("measured_composed_ms")),
                        shift=entry.get("rank_shift") if entry.get("rank_shift") is not None else "n/a",
                    )
                )
            lines.append("")
    if provenance_notes:
        lines.append("## Provenance warnings")
        lines.append("")
        for note in provenance_notes[:10]:
            lines.append(
                f"- {note.get('case_id')} / {note.get('variant_id')}: {note.get('artifact_type')} (actual_outputs={note.get('generated_from_outputs')}, validation_pass={note.get('validation_pass')})"
            )
        lines.append("")
    model_count = len(calibration_bundle.get("models") or {}) if isinstance(calibration_bundle, dict) else 0
    lines.append("## Calibration")
    lines.append("")
    lines.append(f"Backend models available: {model_count}")
    lines.append("")
    if plot_paths:
        lines.append("## Plots")
        lines.append("")
        for path in plot_paths:
            lines.append(f"- {path.name}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    try:
        num = float(value)
        return f"{num:.4g}"
    except Exception:
        return "n/a"


def _finite(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None