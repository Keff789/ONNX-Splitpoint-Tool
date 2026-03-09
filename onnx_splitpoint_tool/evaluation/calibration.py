from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


CALIBRATION_MIN_POINTS = 3
METRICS = ("full", "part1", "part2", "composed", "transfer")


def build_hardware_signature(
    hardware_context: Any,
    *,
    preflight: Any = None,
    run_meta: Any = None,
) -> str:
    basis = {
        "hardware_context": hardware_context,
        "preflight": preflight if isinstance(preflight, dict) else None,
        "host": (run_meta or {}).get("host") if isinstance(run_meta, dict) else None,
    }
    raw = json.dumps(basis, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def fit_backend_calibration(rows: list[dict[str, Any]], *, suite_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    suite_meta = suite_meta or {}
    hardware_context = suite_meta.get("hardware_context")
    input_context = suite_meta.get("input_context")
    bundle = {
        "schema": "onnx-splitpoint/backend-calibration",
        "schema_version": 1,
        "hardware_signature": suite_meta.get("hardware_signature"),
        "picker_hardware_signature": build_hardware_signature(hardware_context),
        "hardware_context": hardware_context,
        "input_context": input_context,
        "suite_id": suite_meta.get("suite_id"),
        "analysis_id": suite_meta.get("analysis_id"),
        "models": {},
        "min_points": CALIBRATION_MIN_POINTS,
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        backend = str(row.get("backend_label") or "unknown")
        grouped.setdefault(backend, []).append(row)

    for backend, backend_rows in sorted(grouped.items()):
        metric_models: dict[str, Any] = {}
        for metric in METRICS:
            pairs = []
            for row in backend_rows:
                if str(row.get("status") or "") != "ok":
                    continue
                pred = _finite(row.get(f"pred_{metric}_ms"))
                meas = _finite(row.get(f"measured_{metric}_ms"))
                if pred is None or meas is None:
                    continue
                pairs.append((pred, meas))
            model = fit_linear_model(pairs)
            if model is not None:
                metric_models[metric] = model
        if metric_models:
            bundle["models"][backend] = metric_models
    return bundle


def fit_linear_model(pairs: list[tuple[float, float]]) -> dict[str, Any] | None:
    if len(pairs) < CALIBRATION_MIN_POINTS:
        return None
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    x_mean = sum(xs) / float(len(xs))
    y_mean = sum(ys) / float(len(ys))
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    a = cov / var_x
    b = y_mean - (a * x_mean)
    preds = [(a * x) + b for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = None if ss_tot <= 0.0 else 1.0 - (ss_res / ss_tot)
    return {
        "a": a,
        "b": b,
        "count": len(pairs),
        "r2": r2,
    }


def apply_linear_model(value: float | None, model: dict[str, Any] | None) -> float | None:
    if value is None or not isinstance(model, dict):
        return value
    a = _finite(model.get("a"))
    b = _finite(model.get("b"))
    count = int(model.get("count") or 0)
    if a is None or b is None or count < CALIBRATION_MIN_POINTS:
        return value
    return (a * value) + b


def load_latest_calibration(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path).expanduser()
    if file_path.is_dir():
        file_path = file_path / "backend_calibration.json"
    if not file_path.exists():
        return None
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        payload.setdefault("_source_path", str(file_path))
    return payload if isinstance(payload, dict) else None


def select_calibration_bundle(
    paths: list[str | Path],
    *,
    hardware_signature: str | None = None,
) -> dict[str, Any] | None:
    candidates = []
    for path in paths:
        payload = load_latest_calibration(path)
        if not isinstance(payload, dict):
            continue
        if hardware_signature:
            known = {
                payload.get("hardware_signature"),
                payload.get("picker_hardware_signature"),
            }
            if hardware_signature not in known and None not in known:
                continue
        candidates.append(payload)
    if not candidates:
        return None
    candidates.sort(key=lambda payload: int(_model_count(payload)), reverse=True)
    return candidates[0]


def _model_count(payload: dict[str, Any]) -> int:
    total = 0
    for metric_models in (payload.get("models") or {}).values():
        if not isinstance(metric_models, dict):
            continue
        for model in metric_models.values():
            if isinstance(model, dict):
                total += int(model.get("count") or 0)
    return total


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
