"""Fit interpretable stage-time and handover models from development rows.

The fitter is intentionally small and auditable.  It uses only development
rows, fits affine FLOP-to-time models per backend and affine handover models per
runner/direction, and writes a profile patch that can be frozen before the
hold-out is opened.  It never fits from rows labelled ``holdout``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .protocol_freeze import is_confirmatory_holdout, normalize_evaluation_role
from .ranking_methods import canonical_backend, canonical_direction, canonical_runner, direction_parts
from .workflow.artifacts import now_iso, sha256_file, sha256_json, write_json, write_text

MODEL_BUNDLE_SCHEMA = "onnx-splitpoint/ranking-model-bundle"
MODEL_BUNDLE_VERSION = 1


def _f(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    value = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        for key in ("rows", "results", "normalized_rows"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def _direction(row: Mapping[str, Any]) -> str:
    raw = _first(row, ("direction", "execution_direction", "backend_pair", "run_id", "backend"))
    if raw:
        direction = canonical_direction(raw)
        if "_to_" in direction:
            return direction
    return canonical_direction(stage1=_first(row, ("stage1", "stage1_backend", "producer_backend")), stage2=_first(row, ("stage2", "stage2_backend", "consumer_backend")))


def _runner(row: Mapping[str, Any]) -> str:
    return canonical_runner(_first(row, ("runner_regime", "runner", "runtime_mode", "variant", "run_id")))


def _role(row: Mapping[str, Any]) -> str:
    return normalize_evaluation_role(
        row.get("evaluation_role") or row.get("role") or ""
    )


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    err = pred - y
    mae = float(np.mean(np.abs(err))) if len(y) else None
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-12)) * 100.0) if len(y) else None
    return {
        "n": int(len(y)),
        "mae_ms": mae,
        "mape_percent": mape,
        "bias_ms": float(np.mean(err)) if len(y) else None,
        "rmse_ms": float(np.sqrt(np.mean(err * err))) if len(y) else None,
    }


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("invalid regression matrix")
    reg = np.eye(x.shape[1], dtype=float) * max(0.0, float(ridge))
    reg[0, 0] = 0.0
    return np.linalg.pinv(x.T @ x + reg) @ x.T @ y


def _fit_stage_model(observations: list[tuple[str, float, float]], ridge: float, minimum_rows: int) -> dict[str, Any]:
    # observation: (model_id, FLOPs, measured_ms)
    if len(observations) < minimum_rows:
        return {"status": "insufficient_rows", "n": len(observations)}
    x_gflop = np.asarray([max(0.0, row[1]) / 1e9 for row in observations], dtype=float)
    y = np.asarray([max(0.0, row[2]) for row in observations], dtype=float)
    x = np.column_stack([np.ones(len(y)), x_gflop])
    beta = _ridge_fit(x, y, ridge)
    intercept = max(0.0, float(beta[0]))
    slope = max(0.0, float(beta[1]))
    pred = intercept + slope * x_gflop
    throughput = (1000.0 / slope) if slope > 0 else None
    return {
        "status": "fitted",
        "mode": "affine_flops",
        "intercept_ms": intercept,
        "per_gflop_ms": slope,
        "throughput_gops": throughput,
        "training_model_ids": sorted({row[0] for row in observations}),
        "metrics": _metrics(y, pred),
    }


def _handover_features(row: Mapping[str, Any]) -> list[float]:
    cut_mib = _f(_first(row, ("cut_mib", "cut_mib_val")))
    if cut_mib is None:
        cut_bytes = _f(_first(row, ("cut_bytes", "crossing_bytes", "cost_bytes")))
        cut_mib = (cut_bytes / (1024.0 ** 2)) if cut_bytes is not None else 0.0
    n_cut = _f(_first(row, ("n_cut_tensors", "crossing_tensor_count", "cut_tensor_count"))) or 0.0
    unknown = _f(_first(row, ("unknown_crossing_tensors", "unknown_count", "unknown_tensor_count"))) or 0.0
    peak = _f(_first(row, ("peak_right_mib", "peak_right_mib_val", "peak_act_right_mib"))) or 0.0
    imbalance = _f(_first(row, ("imbalance", "imbalance_val", "pred_imbalance"))) or 0.0
    return [1.0, max(0.0, cut_mib), max(0.0, n_cut - 1.0), max(0.0, unknown), max(0.0, peak), max(0.0, imbalance)]


def _fit_handover_model(observations: list[tuple[str, Mapping[str, Any], float]], ridge: float, minimum_rows: int) -> dict[str, Any]:
    if len(observations) < minimum_rows:
        return {"status": "insufficient_rows", "n": len(observations)}
    x = np.asarray([_handover_features(row) for _model, row, _time in observations], dtype=float)
    y = np.asarray([max(0.0, time_ms) for _model, _row, time_ms in observations], dtype=float)
    beta = _ridge_fit(x, y, ridge)
    # Negative physical slopes are not portable; clip them and recompute the
    # intercept from the residual mean.  The fit remains deterministic and the
    # report exposes the unclipped coefficients for audit.
    raw = beta.copy()
    slopes = np.maximum(beta[1:], 0.0)
    intercept = max(0.0, float(np.mean(y - x[:, 1:] @ slopes)))
    beta = np.concatenate([[intercept], slopes])
    pred = x @ beta
    names = ["intercept_ms", "per_mib_ms", "per_tensor_ms", "per_unknown_ms", "per_peak_right_mib_ms", "per_imbalance_ms"]
    model = {name: float(value) for name, value in zip(names, beta)}
    model.update({
        "status": "fitted",
        "mode": "affine_features",
        "training_model_ids": sorted({model_id for model_id, _row, _time in observations}),
        "metrics": _metrics(y, pred),
        "raw_unclipped_coefficients": {name: float(value) for name, value in zip(names, raw)},
    })
    return model


def fit_ranking_models(
    rows: Sequence[Mapping[str, Any]],
    *,
    ridge: float = 1e-6,
    minimum_rows: int = 3,
    allow_unlabeled_as_development: bool = False,
) -> dict[str, Any]:
    """Fit models from explicitly declared development rows.

    By default, blank, screening, exploratory and other roles are excluded.  A
    compatibility flag exists for historical data, but bundles created with it
    are intentionally rejected by final-campaign readiness checks.
    """
    development = [
        dict(row)
        for row in rows
        if _role(row) == "development" or (allow_unlabeled_as_development and not _role(row))
    ]
    excluded_holdout = [
        dict(row) for row in rows if is_confirmatory_holdout(_role(row))
    ]
    excluded_other = [
        dict(row)
        for row in rows
        if _role(row) != "development"
        and not is_confirmatory_holdout(_role(row))
        and not (allow_unlabeled_as_development and not _role(row))
    ]

    stage_obs: dict[str, list[tuple[str, float, float]]] = {}
    handover_obs: dict[tuple[str, str], list[tuple[str, Mapping[str, Any], float]]] = {}
    for row in development:
        model_id = str(row.get("model_id") or row.get("model") or "unknown")
        direction = _direction(row)
        stage1, stage2 = direction_parts(direction)
        flops_left = _f(_first(row, ("flops_left", "flops_left_abs", "left_flops")))
        flops_right = _f(_first(row, ("flops_right", "flops_right_abs", "right_flops")))
        t1 = _f(_first(row, ("measured_stage1_ms", "stage1_ms", "part1_ms", "part1_mean_ms", "producer_ms")))
        t2 = _f(_first(row, ("measured_stage2_ms", "stage2_ms", "part2_ms", "part2_mean_ms", "consumer_ms")))
        if stage1 and flops_left is not None and t1 is not None:
            stage_obs.setdefault(canonical_backend(stage1), []).append((model_id, flops_left, t1))
        if stage2 and flops_right is not None and t2 is not None:
            stage_obs.setdefault(canonical_backend(stage2), []).append((model_id, flops_right, t2))
        handover = _f(_first(row, ("measured_handover_ms", "handover_ms", "handoff_ms", "interface_ms", "native_fifo_handoff_ms")))
        if direction and handover is not None:
            handover_obs.setdefault((_runner(row), direction), []).append((model_id, row, handover))

    stage_models = {backend: _fit_stage_model(obs, ridge, minimum_rows) for backend, obs in sorted(stage_obs.items())}
    throughput = {
        backend: model.get("throughput_gops")
        for backend, model in stage_models.items()
        if model.get("status") == "fitted" and _f(model.get("throughput_gops")) is not None
    }
    handover_models: dict[str, dict[str, Any]] = {}
    for (runner, direction), obs in sorted(handover_obs.items()):
        handover_models.setdefault(runner, {})[direction] = _fit_handover_model(obs, ridge, minimum_rows)

    fitted_stage = sum(1 for row in stage_models.values() if row.get("status") == "fitted")
    fitted_handover = sum(1 for block in handover_models.values() for row in block.values() if row.get("status") == "fitted")
    patch = {
        "ranking_validation": {
            "cycle_time_no_handover": {
                "backend_throughput_gops": throughput,
                "stage_time_models": stage_models,
            },
            "cycle_time_with_handover": {
                "backend_throughput_gops": throughput,
                "stage_time_models": stage_models,
                "handover_models": handover_models,
            },
        }
    }
    return {
        "schema": MODEL_BUNDLE_SCHEMA,
        "schema_version": MODEL_BUNDLE_VERSION,
        "created_at": now_iso(),
        "fit_policy": {
            "ridge": float(ridge),
            "minimum_rows": int(minimum_rows),
            "holdout_rows_excluded": True,
            "explicit_development_role_required": not bool(allow_unlabeled_as_development),
            "unlabeled_rows_treated_as_development": bool(allow_unlabeled_as_development),
        },
        "development_row_count": len(development),
        "excluded_holdout_row_count": len(excluded_holdout),
        "excluded_non_development_row_count": len(excluded_other),
        "excluded_non_development_roles": sorted({_role(row) or "unlabeled" for row in excluded_other}),
        "development_model_ids": sorted({str(row.get("model_id") or row.get("model") or "unknown") for row in development}),
        "stage_time_models": stage_models,
        "handover_models": handover_models,
        "fitted_stage_model_count": fitted_stage,
        "fitted_handover_model_count": fitted_handover,
        "profile_patch": patch,
        "status": "fitted" if fitted_stage or fitted_handover else "insufficient_data",
    }


def fit_ranking_models_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    ridge: float = 1e-6,
    minimum_rows: int = 3,
    patch_profile: str | Path | None = None,
    patched_profile_out: str | Path | None = None,
    allow_unlabeled_as_development: bool = False,
) -> Path:
    source = Path(input_path).expanduser().resolve()
    bundle = fit_ranking_models(
        _load_rows(source),
        ridge=ridge,
        minimum_rows=minimum_rows,
        allow_unlabeled_as_development=allow_unlabeled_as_development,
    )
    bundle["training_rows_path"] = str(source)
    bundle["training_rows_sha256"] = sha256_file(source)
    bundle["bundle_payload_sha256"] = sha256_json({k: v for k, v in bundle.items() if k != "bundle_payload_sha256"})
    out = write_json(output_path, bundle)

    if patch_profile:
        if yaml is None:
            raise RuntimeError("PyYAML is required for --patch-profile")
        profile_path = Path(patch_profile).expanduser().resolve()
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        profile = dict(profile or {}) if isinstance(profile, Mapping) else {}
        ranking = dict(profile.get("ranking_validation") or {}) if isinstance(profile.get("ranking_validation"), Mapping) else {}
        patch = bundle["profile_patch"]["ranking_validation"]
        for section in ("cycle_time_no_handover", "cycle_time_with_handover"):
            current = dict(ranking.get(section) or {}) if isinstance(ranking.get(section), Mapping) else {}
            current.update(dict(patch.get(section) or {}))
            ranking[section] = current
        profile["ranking_validation"] = ranking
        campaign = dict(profile.get("campaign") or {}) if isinstance(profile.get("campaign"), Mapping) else {}
        campaign["ranking_model_bundle"] = str(Path(out).resolve())
        campaign["ranking_model_bundle_sha256"] = sha256_file(out)
        profile["campaign"] = campaign
        target = Path(patched_profile_out).expanduser().resolve() if patched_profile_out else profile_path.with_name(profile_path.stem + "_fitted.yaml")
        write_text(target, yaml.safe_dump(profile, sort_keys=False, allow_unicode=True))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fit development-only stage-time and handover models.")
    ap.add_argument("input", help="Canonical development rows as CSV or JSON")
    ap.add_argument("--out", required=True, help="ranking_model_bundle.json")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--minimum-rows", type=int, default=3)
    ap.add_argument("--patch-profile", default="")
    ap.add_argument("--patched-profile-out", default="")
    ap.add_argument(
        "--allow-unlabeled-as-development",
        action="store_true",
        help="Legacy compatibility only. Final-campaign validation rejects bundles fitted with this flag.",
    )
    ns = ap.parse_args(list(argv) if argv is not None else None)
    path = fit_ranking_models_file(
        input_path=ns.input,
        output_path=ns.out,
        ridge=ns.ridge,
        minimum_rows=ns.minimum_rows,
        patch_profile=(ns.patch_profile or None),
        patched_profile_out=(ns.patched_profile_out or None),
        allow_unlabeled_as_development=bool(ns.allow_unlabeled_as_development),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
