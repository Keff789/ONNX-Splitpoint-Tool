from __future__ import annotations

from typing import Any, Mapping


def _num(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _mapping(v: Any) -> dict[str, Any] | None:
    return dict(v) if isinstance(v, Mapping) else None


def _find_result_mapping(data: Mapping[str, Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    """Return the best power_calculations result mapping.

    power_calculations has changed shape a couple of times and serde_saphyr
    serialisation can create subtly different nested dictionaries.  Prefer the
    documented top-level results, then fall back to any nested mapping that has
    both energy and duration-like keys.  This keeps the Splitpoint wrapper from
    reporting postprocess_status=failed simply because the YAML shape changed.
    """
    root = dict(data or {})
    for key in ("firmware_results", "jetson_results", "shelly_results"):
        val = _mapping(root.get(key))
        if val is not None:
            return key, val
    osc = _mapping(root.get("oscilloscope_results"))
    if osc is not None:
        val = _mapping(osc.get("results"))
        if val is not None:
            # ``energy`` and ``duration`` live in the nested ``results``
            # mapping.  Keep the complete field path so provenance validators
            # can bind the declared source field without inventing the
            # non-existent ``oscilloscope_results.energy`` alias.
            return "oscilloscope_results.results", val

    # Recursive fallback: choose the first mapping that carries an energy and a
    # duration value.  Keep the path as source_key for debugging.
    best: tuple[str | None, dict[str, Any] | None] = (None, None)

    def walk(obj: Any, path: str) -> None:
        nonlocal best
        if best[1] is not None:
            return
        if isinstance(obj, Mapping):
            d = dict(obj)
            if _num(d.get("energy")) is not None and _num(d.get("duration")) is not None:
                best = (path or "<root>", d)
                return
            for k, v in d.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")

    walk(root, "")
    return best


def extract_power_calculation_summary(results: Mapping[str, Any] | None) -> dict[str, Any]:
    chosen_key, chosen = _find_result_mapping(results)
    chosen = chosen or {}
    energy = _num(chosen.get("energy"))
    duration = _num(chosen.get("duration"))
    avg_power = (energy / duration) if energy is not None and duration and duration > 0 else None
    return {
        "source_key": chosen_key,
        "energy_total_j": energy,
        "active_duration_s": duration,
        "avg_power_w": avg_power,
        "max_frame_energy_j": _num(chosen.get("max_frame_energy")),
        "idle_frame_energy_j": _num(chosen.get("idle_frame_energy")),
        "start_stop_idx": chosen.get("start_stop_idx"),
    }


def apply_energy_baselines(summary: dict[str, Any], *, idle_baseline_w: float | None = None, accelerator_idle_w: float | None = None, host_only: bool = False) -> dict[str, Any]:
    out = dict(summary)
    e = _num(out.get("energy_total_j"))
    d = _num(out.get("active_duration_s"))
    if e is not None and d is not None and idle_baseline_w is not None:
        out["energy_dynamic_j"] = max(0.0, e - float(idle_baseline_w) * d)
    else:
        out["energy_dynamic_j"] = None
    if e is not None and d is not None and host_only and accelerator_idle_w is not None:
        out["host_normalized_energy_est_j"] = max(0.0, e - float(accelerator_idle_w) * d)
    else:
        out["host_normalized_energy_est_j"] = None
    return out
