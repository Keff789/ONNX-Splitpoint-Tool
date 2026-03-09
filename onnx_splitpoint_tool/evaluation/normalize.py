from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable


METRIC_ALIASES = {
    "full": ("measured_full_ms", "full_mean_ms", "full_ms"),
    "part1": ("measured_part1_ms", "part1_mean_ms", "part1_ms"),
    "part2": ("measured_part2_ms", "part2_mean_ms", "part2_ms"),
    "composed": ("measured_composed_ms", "composed_mean_ms", "composed_ms"),
    "transfer": ("measured_transfer_ms", "measured_overhead_ms", "overhead_ms", "transfer_ms"),
}


def load_evaluation_inputs(
    results_dir: str | Path,
    *,
    suite_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    results_dir = Path(results_dir).expanduser().resolve()
    suite_dir = Path(suite_dir).expanduser().resolve() if suite_dir else None
    run_dir = results_dir.parent if results_dir.name == "results" else results_dir

    benchmark_set_path = _first_existing(results_dir / "benchmark_set.json", suite_dir / "benchmark_set.json" if suite_dir else None)
    benchmark_plan_path = _first_existing(results_dir / "benchmark_plan.json", suite_dir / "benchmark_plan.json" if suite_dir else None)
    run_meta_path = _first_existing(run_dir / "run_meta.json", results_dir / "run_meta.json", suite_dir / "run_meta.json" if suite_dir else None)
    snapshot_path = _first_existing(
        results_dir / "analysis_tables" / "prediction_snapshot.json",
        suite_dir / "analysis_tables" / "prediction_snapshot.json" if suite_dir else None,
    )
    preflight_path = _first_existing(results_dir / "preflight.json", suite_dir / "preflight.json" if suite_dir else None)
    status_matrix_path = _first_existing(
        results_dir / "benchmark_suite_status_matrix.json",
        suite_dir / "benchmark_suite_status_matrix.json" if suite_dir else None,
    )

    benchmark_set = _read_json(benchmark_set_path)
    benchmark_plan = _read_json(benchmark_plan_path)
    run_meta = _read_json(run_meta_path)
    preflight = _read_json(preflight_path)
    status_matrix = _read_json(status_matrix_path, default=[])

    prediction_rows, prediction_meta = _load_prediction_rows(snapshot_path, benchmark_set)
    measurement_rows = _load_measurement_rows(results_dir)
    artifact_records = _load_artifact_manifests(results_dir)

    suite_meta = {
        "suite_id": _first_non_empty(
            prediction_meta.get("suite_id") if isinstance(prediction_meta, dict) else None,
            _dig(benchmark_set, "suite_id"),
            _dig(run_meta, "suite_id"),
            run_dir.name,
        ),
        "analysis_id": _first_non_empty(
            prediction_meta.get("analysis_id") if isinstance(prediction_meta, dict) else None,
            _dig(benchmark_set, "analysis_id"),
            _dig(benchmark_set, "created_at"),
        ),
        "tool_version": _first_non_empty(
            prediction_meta.get("tool_version") if isinstance(prediction_meta, dict) else None,
            _dig(benchmark_set, "tool", "gui"),
            _dig(benchmark_set, "tool", "core"),
        ),
        "hardware_context": _first_non_empty(
            prediction_meta.get("hardware_context") if isinstance(prediction_meta, dict) else None,
            _dig(benchmark_set, "hardware_context"),
            _dig(benchmark_set, "system_spec"),
            _dig(preflight, "hardware"),
            _dig(run_meta, "host"),
        ),
        "input_context": _first_non_empty(
            prediction_meta.get("input_context") if isinstance(prediction_meta, dict) else None,
            _dig(benchmark_set, "input_context"),
            _dig(preflight, "input"),
        ),
        "files": {
            "benchmark_set": _rel_or_none(benchmark_set_path, run_dir),
            "benchmark_plan": _rel_or_none(benchmark_plan_path, run_dir),
            "prediction_snapshot": _rel_or_none(snapshot_path, run_dir),
            "preflight": _rel_or_none(preflight_path, run_dir),
            "status_matrix": _rel_or_none(status_matrix_path, run_dir),
        },
    }
    suite_meta["hardware_signature"] = _hardware_signature(suite_meta.get("hardware_context"), preflight=preflight, run_meta=run_meta)

    bundle = {
        "results_dir": str(results_dir),
        "run_dir": str(run_dir),
        "suite_dir": str(suite_dir) if suite_dir else None,
        "suite_meta": suite_meta,
        "prediction_rows": prediction_rows,
        "measurement_rows": measurement_rows,
        "artifact_records": artifact_records,
        "benchmark_set": benchmark_set,
        "benchmark_plan": benchmark_plan,
        "preflight": preflight,
        "run_meta": run_meta,
        "status_matrix": status_matrix if isinstance(status_matrix, list) else [],
    }

    if log is not None:
        log(
            "[evaluation] loaded "
            f"{len(prediction_rows)} predictions, {len(measurement_rows)} measurements, {len(artifact_records)} artifact records"
        )
    return bundle


def _load_prediction_rows(snapshot_path: Path | None, benchmark_set: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if snapshot_path and snapshot_path.exists():
        payload = _read_json(snapshot_path, default={})
        if isinstance(payload, list):
            return [_normalize_prediction_row(row) for row in payload if isinstance(row, dict)], {}
        if isinstance(payload, dict):
            raw_rows = payload.get("rows") if isinstance(payload.get("rows"), list) else payload.get("predictions")
            rows = [_normalize_prediction_row(row) for row in (raw_rows or []) if isinstance(row, dict)]
            meta = {k: v for k, v in payload.items() if k not in {"rows", "predictions"}}
            return rows, meta
    rows: list[dict[str, Any]] = []
    if isinstance(benchmark_set, dict):
        for index, case in enumerate(benchmark_set.get("cases") or []):
            if not isinstance(case, dict):
                continue
            pred = case.get("prediction_snapshot")
            if not isinstance(pred, dict):
                pred = case.get("predicted") if isinstance(case.get("predicted"), dict) else {}
            row = _normalize_prediction_row({
                "candidate_id": case.get("candidate_id") or case.get("case_id") or case.get("folder") or case.get("boundary_id"),
                "case_id": case.get("case_id") or case.get("folder") or case.get("case_dir"),
                "split_id": case.get("split_id") or case.get("boundary_id") or case.get("boundary"),
                "selection_rank": index + 1,
                **pred,
            })
            rows.append(row)
    return rows, {}


def _load_measurement_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("benchmark_results_*.json")):
        payload = _read_json(path, default=[])
        if isinstance(payload, dict):
            payload_rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        elif isinstance(payload, list):
            payload_rows = payload
        else:
            payload_rows = []
        run_tag = path.stem.replace("benchmark_results_", "", 1)
        for index, record in enumerate(payload_rows):
            if isinstance(record, dict):
                rows.append(_normalize_measurement_row(record, run_tag=run_tag, index=index))
    return rows


def _load_artifact_manifests(results_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*/results_*/artifacts_manifest.json")):
        payload = _read_json(path, default=[])
        case_id = path.parent.parent.name
        result_dir = path.parent.name
        records: list[dict[str, Any]]
        if isinstance(payload, dict):
            if isinstance(payload.get("artifacts"), list):
                records = [r for r in payload.get("artifacts") or [] if isinstance(r, dict)]
            else:
                records = [payload]
        elif isinstance(payload, list):
            records = [r for r in payload if isinstance(r, dict)]
        else:
            records = []
        for record in records:
            item = dict(record)
            item.setdefault("case_id", case_id)
            item.setdefault("result_dir", result_dir)
            item.setdefault("manifest_path", str(path.relative_to(results_dir.parent if results_dir.name == "results" else results_dir)))
            manifests.append(item)
    return manifests


def _normalize_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    candidate_id = _string_id(
        row.get("candidate_id"),
        row.get("case_id"),
        row.get("split_id"),
        row.get("boundary_id"),
        row.get("boundary"),
    )
    case_id = _string_id(row.get("case_id"), row.get("folder"), row.get("case_dir"), candidate_id)
    split_id = _string_id(row.get("split_id"), row.get("boundary_id"), row.get("boundary"), candidate_id)

    pred_full = _first_float(row.get("pred_full_ms"), row.get("predicted_full_ms"), row.get("full_ms"), row.get("latency_full_ms"))
    pred_part1 = _first_float(row.get("pred_part1_ms"), row.get("predicted_part1_ms"), row.get("part1_ms"))
    pred_part2 = _first_float(row.get("pred_part2_ms"), row.get("predicted_part2_ms"), row.get("part2_ms"))
    pred_transfer = _first_float(row.get("pred_transfer_ms"), row.get("predicted_transfer_ms"), row.get("transfer_ms"), row.get("link_latency_ms"))
    pred_composed = _first_float(
        row.get("pred_composed_ms"),
        row.get("predicted_composed_ms"),
        row.get("composed_ms"),
        row.get("latency_total_ms"),
    )
    if pred_composed is None and pred_part1 is not None and pred_part2 is not None:
        pred_composed = pred_part1 + pred_part2 + (pred_transfer or 0.0)

    out = {
        "candidate_id": candidate_id,
        "case_id": case_id,
        "split_id": split_id,
        "split_node": _first_non_empty(row.get("split_node"), row.get("cut_descriptor"), row.get("boundary_name")),
        "selection_rank": _safe_int(row.get("selection_rank")),
        "selection_score": _first_float(row.get("selection_score"), row.get("score")),
        "selection_reason": _first_non_empty(row.get("selection_reason"), row.get("selection_label"), row.get("selection_mode")),
        "pred_full_ms": pred_full,
        "pred_part1_ms": pred_part1,
        "pred_part2_ms": pred_part2,
        "pred_transfer_ms": pred_transfer,
        "pred_composed_ms": pred_composed,
        "pred_speedup_vs_full": _first_float(row.get("pred_speedup_vs_full"), row.get("speedup_vs_full")),
        "pred_activation_bytes": _first_float(row.get("pred_activation_bytes"), row.get("cut_bytes")),
        "pred_peak_activation_memory": _first_float(
            row.get("pred_peak_activation_memory"),
            row.get("peak_act_max_bytes"),
            row.get("peak_activation_memory_bytes"),
        ),
        "pred_hailo_parse_ok": _first_bool(row.get("pred_hailo_parse_ok"), row.get("hailo_parse_ok"), row.get("strict_ok")),
        "pred_hailo_target": _first_non_empty(row.get("pred_hailo_target"), row.get("hailo_target"), row.get("accepted_by")),
        "pred_hailo_hw_arch": _first_non_empty(row.get("pred_hailo_hw_arch"), row.get("pred_hailo_target"), row.get("hw_arch")),
        "predicted_feasible": _first_bool(row.get("predicted_feasible"), row.get("feasible"), row.get("strict_ok")),
        "predicted_pareto": _first_bool(row.get("predicted_pareto"), row.get("is_pareto"), row.get("pareto")),
        "raw": row,
    }
    return out


def _normalize_measurement_row(record: dict[str, Any], *, run_tag: str, index: int) -> dict[str, Any]:
    candidate_id = _string_id(
        record.get("candidate_id"),
        record.get("case_id"),
        record.get("folder"),
        record.get("case_dir"),
        record.get("boundary_id"),
        record.get("boundary"),
        f"{run_tag}:{index}",
    )
    case_id = _string_id(record.get("case_id"), record.get("folder"), record.get("case_dir"), candidate_id)
    backend_stage1 = _first_non_empty(record.get("backend_stage1"), record.get("stage1_provider"), record.get("stage1"), record.get("provider"))
    backend_stage2 = _first_non_empty(record.get("backend_stage2"), record.get("stage2_provider"), record.get("stage2"), record.get("provider"))
    out = {
        "suite_id": _first_non_empty(record.get("suite_id")),
        "candidate_id": candidate_id,
        "case_id": case_id,
        "variant_id": _string_id(record.get("variant_id"), record.get("run_id"), run_tag),
        "run_tag": run_tag,
        "backend_stage1": backend_stage1,
        "backend_stage2": backend_stage2,
        "backend_label": _backend_label(backend_stage1, backend_stage2, fallback=run_tag),
        "status": _first_non_empty(record.get("status"), _derive_status(record), "missing"),
        "reason_code": _first_non_empty(record.get("reason_code"), _derive_reason_code(record), ""),
        "reason_detail": _first_non_empty(record.get("reason_detail"), _derive_reason_detail(record), ""),
        "validation_mode": _first_non_empty(record.get("validation_mode"), record.get("preset"), "auto"),
        "validation_pass": _first_bool(record.get("validation_pass"), record.get("eps_pass"), record.get("ok")),
        "measured": _first_bool(record.get("measured"), _has_any_measurement(record)),
        "artifacts_present": _first_bool(record.get("artifacts_present"), _guess_artifacts_present(record)),
        "variant_records": _normalize_variant_records(record),
        "raw": record,
    }
    for metric_name, aliases in METRIC_ALIASES.items():
        out[f"measured_{metric_name}_ms"] = _first_float(*[record.get(alias) for alias in aliases])
    return out


def _normalize_variant_records(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_records = record.get("variant_records")
    out: dict[str, dict[str, Any]] = {}
    if isinstance(raw_records, dict):
        for name, payload in raw_records.items():
            if isinstance(payload, dict):
                out[str(name)] = {
                    "status": _first_non_empty(payload.get("status"), "missing"),
                    "reason_code": _first_non_empty(payload.get("reason_code"), ""),
                    "reason_detail": _first_non_empty(payload.get("reason_detail"), payload.get("detail"), ""),
                    "measured": _first_bool(payload.get("measured"), False),
                }
        if out:
            return out
    variant_status = record.get("variant_status") if isinstance(record.get("variant_status"), dict) else {}
    variant_errors = record.get("variant_errors") if isinstance(record.get("variant_errors"), dict) else {}
    for name in {"full", "part1", "part2", "composed"}:
        out[name] = {
            "status": _first_non_empty(variant_status.get(name), "missing"),
            "reason_code": _first_non_empty(variant_errors.get(name), ""),
            "reason_detail": _first_non_empty(variant_errors.get(name), ""),
            "measured": bool(_first_float(record.get(f"{name}_mean_ms")) is not None),
        }
    return out


def _derive_status(record: dict[str, Any]) -> str:
    if isinstance(record.get("variant_records"), dict):
        statuses = [str(v.get("status") or "") for v in record.get("variant_records", {}).values() if isinstance(v, dict)]
        return _aggregate_status(statuses)
    if isinstance(record.get("variant_status"), dict):
        statuses = [str(v or "") for v in record.get("variant_status", {}).values()]
        return _aggregate_status(statuses)
    if _first_bool(record.get("ok"), record.get("eps_pass")) is True:
        return "ok"
    if _has_any_measurement(record):
        return "ok"
    return "missing"


def _derive_reason_code(record: dict[str, Any]) -> str | None:
    for key in ("reason_code", "error_code"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    for container_key in ("variant_records", "variant_errors"):
        container = record.get(container_key)
        if isinstance(container, dict):
            for payload in container.values():
                if isinstance(payload, dict):
                    value = payload.get("reason_code") or payload.get("error_code") or payload.get("status")
                else:
                    value = payload
                if value not in (None, "", "ok", "missing"):
                    return str(value)
    return None


def _derive_reason_detail(record: dict[str, Any]) -> str | None:
    for key in ("reason_detail", "error", "message"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    variant_errors = record.get("variant_errors")
    if isinstance(variant_errors, dict):
        for value in variant_errors.values():
            if value not in (None, ""):
                return str(value)
    return None


def _aggregate_status(statuses: list[str]) -> str:
    clean = [s for s in statuses if s]
    if not clean:
        return "missing"
    if any(s == "error" for s in clean):
        return "error"
    if any(s == "ok" for s in clean):
        return "ok"
    if any(s == "skipped" for s in clean):
        return "skipped"
    return clean[0]


def _guess_artifacts_present(record: dict[str, Any]) -> bool:
    for key in ("artifacts_present", "detections_full_png", "detections_composed_png"):
        value = record.get(key)
        if isinstance(value, bool):
            return value
    return False


def _has_any_measurement(record: dict[str, Any]) -> bool:
    for aliases in METRIC_ALIASES.values():
        for alias in aliases:
            value = _safe_float(record.get(alias))
            if value is not None:
                return True
    return False


def _hardware_signature(hardware_context: Any, *, preflight: Any, run_meta: Any) -> str:
    basis = {
        "hardware_context": hardware_context,
        "preflight": preflight if isinstance(preflight, dict) else None,
        "host": _dig(run_meta, "host") if isinstance(run_meta, dict) else None,
    }
    raw = json.dumps(basis, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_json(path: Path | None, *, default: Any = None) -> Any:
    if path is None or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _first_existing(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _string_id(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        out = _safe_float(value)
        if out is not None:
            return out
    return None


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        out = _safe_bool(value)
        if out is not None:
            return out
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        num = float(value)
        if math.isfinite(num):
            return num
    except Exception:
        return None
    return None


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "ok"}:
        return True
    if text in {"0", "false", "no", "missing", "skipped", "error"}:
        return False
    return None


def _dig(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _rel_or_none(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _backend_label(stage1: Any, stage2: Any, *, fallback: str) -> str:
    left = str(stage1 or "").strip()
    right = str(stage2 or "").strip()
    if left and right:
        return left if left == right else f"{left} -> {right}"
    if left:
        return left
    if right:
        return right
    return str(fallback)