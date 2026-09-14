from __future__ import annotations

"""v60z native-full, native-energy, official-quality orchestration helpers.

This module deliberately centralises semantics that used to be spread over the
profile editor, run-mode resolver, native producer runner and scientific
reporter.  Functions are side-effect free unless their name starts with
``apply_``.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping
import json
import re


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "measure"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "none", "plan"}:
        return False
    return default


def run_profile_enabled(row: Mapping[str, Any]) -> bool:
    """Interpret the optional logical-row switch without bool-string leaks."""

    return _bool(row.get("enabled", True), True)


def enabled_run_profiles(value: Any) -> list[Mapping[str, Any]]:
    """Return only enabled mapping rows from a logical run-profile matrix."""

    return [
        row for row in list(value or [])
        if isinstance(row, Mapping) and run_profile_enabled(row)
    ]


def _get(obj: Mapping[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _ensure_map(obj: MutableMapping[str, Any], key: str) -> MutableMapping[str, Any]:
    value = obj.get(key)
    if not isinstance(value, MutableMapping):
        value = {}
        obj[key] = value
    return value


def _set(obj: MutableMapping[str, Any], path: Iterable[str], value: Any) -> None:
    parts = list(path)
    cur = obj
    for key in parts[:-1]:
        cur = _ensure_map(cur, key)
    cur[parts[-1]] = value


def _normalise_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _flatten_selection(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, Mapping):
        for key, selected in value.items():
            if _bool(selected, False):
                out.append(str(key))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, Mapping):
                name = item.get("id") or item.get("name") or item.get("profile") or item.get("backend")
                selected = item.get("enabled", item.get("selected", True))
                if name and _bool(selected, True):
                    out.append(str(name))
            elif item is not None:
                out.append(str(item))
    elif isinstance(value, str):
        out.extend(x.strip() for x in re.split(r"[,;\n]", value) if x.strip())
    return out


def selected_hardware_tokens(profile: Mapping[str, Any]) -> set[str]:
    candidates = [
        profile.get("run_profiles"),
        profile.get("hardware_run_profiles"),
        profile.get("hardware_profiles"),
        profile.get("selected_hardware_profiles"),
        profile.get("targets"),
        _get(profile, "execution", "hardware_run_profiles"),
        _get(profile, "evaluation", "hardware_run_profiles"),
    ]
    tokens: set[str] = set()
    for value in candidates:
        for item in _flatten_selection(value):
            tokens.add(_normalise_token(item))
    # Preserve the explicit Full/stage semantics carried by simplified
    # run_profiles.  `_flatten_selection` intentionally returns the logical ID,
    # but Native Full resolution also needs the `full` backend value.
    for row in enabled_run_profiles(profile.get("run_profiles")):
        rid = _normalise_token(row.get("id") or row.get("run_id") or "")
        if rid:
            tokens.add(rid)
        full = _normalise_token(row.get("full") or "")
        if full:
            tokens.add(full + "_full")
        stage1 = _normalise_token(row.get("stage1") or "")
        stage2 = _normalise_token(row.get("stage2") or "")
        if stage1 and stage2:
            tokens.add(stage1 + "_to_" + stage2)
    return tokens


def native_enabled(profile: Mapping[str, Any]) -> bool:
    values = [
        _get(profile, "native_producers", "enabled"),
        _get(profile, "execution_preset", "overrides", "native_enabled"),
        _get(profile, "native", "enabled"),
    ]
    for value in values:
        if value is not None:
            return _bool(value)
    return False


def resolve_native_energy_enabled(profile: Mapping[str, Any]) -> bool:
    """Resolve native energy with stable modern-to-legacy precedence."""
    values = [
        _get(profile, "native_producers", "energy", "enabled"),
        _get(profile, "energy", "requested_native_energy"),
        _get(profile, "execution_preset", "overrides", "energy_enabled"),
        # Legacy only.  Generic energy is no longer supported, but old profiles
        # used this as a master bit.
        _get(profile, "energy", "enabled"),
    ]
    for value in values:
        if value is not None:
            return _bool(value)
    return False


@dataclass(frozen=True)
class NativeFullPlan:
    enabled: bool
    backends_by_producer: dict[str, tuple[str, ...]]
    selected_full_backends: tuple[str, ...]
    active_producers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backends_by_producer": {k: list(v) for k, v in self.backends_by_producer.items()},
            "selected_full_backends": list(self.selected_full_backends),
            "active_producers": list(self.active_producers),
        }


@dataclass(frozen=True)
class NativeSplitPlan:
    """Native producer-to-TensorRT rows selected by logical run profiles.

    ``native_producers.backends`` describes configured adapter capability and
    predates the simplified run-profile matrix.  It must therefore remain a
    legacy fallback, not become an implicit request for every split adapter.
    """

    enabled: bool
    selected_split_backends: tuple[str, ...]
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "selected_split_backends": list(self.selected_split_backends),
            "source": self.source,
        }


_BACKEND_ALIASES = {
    "hailo8": ("hailo_8", "hailo8"),
    "hailo10h": ("hailo_10h", "hailo10h", "hailo10"),
    "deepx": ("deepx", "dx_m1", "deepx_m1"),
    "tensorrt": ("tensorrt", "trt"),
}

_NATIVE_PRODUCERS = ("hailo8", "hailo10h", "deepx")


def _canonical_backend(value: Any) -> str:
    """Return the execution backend carried by a run-profile stage value."""

    if isinstance(value, Mapping):
        parts = [
            value.get(key)
            for key in (
                "backend", "provider", "target", "hw_arch", "accelerator",
                "id", "name", "type",
            )
            if value.get(key) not in (None, "")
        ]
        token = _normalise_token(" ".join(str(part) for part in parts))
    else:
        token = _normalise_token(value)
    if "hailo10" in token:
        return "hailo10h"
    if "hailo8" in token or "hailo_8" in token:
        return "hailo8"
    if any(alias in token for alias in ("deepx", "dx_m1", "dxm1")):
        return "deepx"
    if "tensorrt" in token or token in {"trt", "tensor_rt"}:
        return "tensorrt"
    if token in {"cpu", "cpu_ort", "ort_cpu"}:
        return "cpu"
    if token in {"cuda", "cuda_ort", "ort_cuda"}:
        return "cuda"
    return token


def _canonical_native_producer(value: Any) -> str:
    backend = _canonical_backend(value)
    if backend in _NATIVE_PRODUCERS:
        return backend
    token = _normalise_token(value)
    if "hailo10" in token:
        return "hailo10h"
    if "hailo8" in token:
        return "hailo8"
    if "deepx" in token or "dx_m1" in token or "dxm1" in token:
        return "deepx"
    return ""


def _split_producer_from_run_profile(row: Mapping[str, Any]) -> str:
    """Resolve only supported forward producer-to-TensorRT logical rows."""

    stage1 = _canonical_backend(row.get("stage1"))
    stage2 = _canonical_backend(row.get("stage2"))
    if stage1 in _NATIVE_PRODUCERS and stage2 == "tensorrt":
        return stage1
    run_id = _normalise_token(
        row.get("id") or row.get("run_id") or row.get("name") or ""
    )
    for producer, aliases in {
        "hailo8": ("hailo8", "hailo_8"),
        "hailo10h": ("hailo10", "hailo10h", "hailo_10h"),
        "deepx": ("deepx", "deepx_m1", "dx_m1", "dxm1"),
    }.items():
        if any(
            run_id.startswith(alias + "_to_tensorrt")
            or run_id.startswith(alias + "_to_trt")
            for alias in aliases
        ):
            return producer
    return ""


def resolve_native_split_plan(
    profile: Mapping[str, Any],
    *,
    explicit_backends: Iterable[Any] | None = None,
) -> NativeSplitPlan:
    """Resolve selected Native split adapters with stable authority rules.

    Explicit CLI selection wins.  Otherwise the presence of ``run_profiles``
    is authoritative, including a valid empty split selection.  Profiles from
    before the logical matrix existed retain their ``native_producers``
    fallback.
    """

    if explicit_backends is not None:
        selected = [
            producer
            for producer in (
                _canonical_native_producer(value)
                for value in explicit_backends
            )
            if producer
        ]
        selected = list(dict.fromkeys(selected))
        return NativeSplitPlan(bool(selected), tuple(selected), "explicit_cli")

    if not native_enabled(profile):
        return NativeSplitPlan(False, (), "native_disabled")

    if "run_profiles" in profile:
        selected: list[str] = []
        rows = profile.get("run_profiles")
        if isinstance(rows, (list, tuple)):
            for raw in enabled_run_profiles(rows):
                producer = _split_producer_from_run_profile(raw)
                if producer and producer not in selected:
                    selected.append(producer)
        return NativeSplitPlan(
            bool(selected), tuple(selected), "evaluation_profile.run_profiles"
        )

    native = (
        profile.get("native_producers")
        if isinstance(profile.get("native_producers"), Mapping)
        else {}
    )
    configured = (
        native.get("split_backends")
        if "split_backends" in native
        else native.get("backends")
    )
    selected = [
        producer
        for producer in (
            _canonical_native_producer(value)
            for value in _flatten_selection(configured)
        )
        if producer
    ]
    selected = list(dict.fromkeys(selected))
    return NativeSplitPlan(bool(selected), tuple(selected), "legacy_native_backends")


def _has_token(tokens: set[str], backend: str, suffix: str | None = None) -> bool:
    aliases = _BACKEND_ALIASES[backend]
    for token in tokens:
        for alias in aliases:
            if alias in token:
                if suffix is None or suffix in token:
                    return True
    return False


def resolve_native_full_plan(profile: Mapping[str, Any]) -> NativeFullPlan:
    """Derive native-full rows from the visible hardware selections.

    TensorRT Full is repeated on every active producer setup so that the native
    split and the GPU-only baseline share the physical system and power scope.
    """
    if not native_enabled(profile):
        return NativeFullPlan(False, {}, (), ())

    tokens = selected_hardware_tokens(profile)
    selected_full: list[str] = []
    for backend in ("hailo8", "hailo10h", "deepx", "tensorrt"):
        if _has_token(tokens, backend, "full"):
            selected_full.append(backend)

    active: list[str] = []
    for producer in ("hailo8", "hailo10h", "deepx"):
        # A producer is active if either its Full row or a producer->TensorRT
        # row was selected.
        if _has_token(tokens, producer) and (
            _has_token(tokens, producer, "full")
            or any((producer in t and ("tensorrt" in t or "trt" in t)) for t in tokens)
        ):
            active.append(producer)

    # If profiles use canonical IDs that the token parser cannot recognise,
    # honour an already resolved producer map as fallback.
    existing = _get(profile, "native_producers", "full_baselines", "backends_by_producer", default={})
    if isinstance(existing, Mapping):
        for key in existing:
            prod = _normalise_token(key)
            if prod in {"hailo8", "hailo10h", "deepx"} and prod not in active:
                active.append(prod)

    by_producer: dict[str, tuple[str, ...]] = {}
    for producer in active:
        rows: list[str] = []
        if producer in selected_full:
            rows.append(producer)
        if "tensorrt" in selected_full:
            rows.append("tensorrt")
        if rows:
            by_producer[producer] = tuple(dict.fromkeys(rows))

    enabled = bool(by_producer)
    return NativeFullPlan(enabled, by_producer, tuple(selected_full), tuple(active))


def apply_native_energy_state(profile: MutableMapping[str, Any], enabled: bool | None = None) -> bool:
    """Persist native-only energy consistently across modern and legacy keys."""
    state = resolve_native_energy_enabled(profile) if enabled is None else bool(enabled)
    if state:
        _set(profile, ("native_producers", "enabled"), True)
        _set(profile, ("execution_preset", "overrides", "native_enabled"), True)
    _set(profile, ("energy", "enabled"), False)          # generic energy remains off
    _set(profile, ("energy", "generic_enabled"), False)
    _set(profile, ("energy", "measurement_path"), "native_only")
    _set(profile, ("energy", "requested_native_energy"), state)
    _set(profile, ("native_producers", "energy", "enabled"), state)
    _set(profile, ("native_producers", "energy", "mode"), "measure" if state else "plan")
    _set(profile, ("native_producers", "energy", "include_split_rows"), state)
    _set(profile, ("native_producers", "energy", "include_full_baselines"), state)
    _set(profile, ("execution_preset", "overrides", "energy_enabled"), state)
    return state


def apply_native_full_plan(profile: MutableMapping[str, Any]) -> NativeFullPlan:
    plan = resolve_native_full_plan(profile)
    full = _ensure_map(_ensure_map(profile, "native_producers"), "full_baselines")
    full["enabled"] = plan.enabled
    full["backends_by_producer"] = {k: list(v) for k, v in plan.backends_by_producer.items()}
    full["backends"] = sorted({b for rows in plan.backends_by_producer.values() for b in rows})
    full["selection_source"] = "visible_hardware_run_profiles"
    profile["resolved_native_full_plan"] = plan.to_dict()
    return plan


def normalise_evaluation_profile(profile: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    apply_native_energy_state(profile)
    apply_native_full_plan(profile)
    return profile


def load_and_normalise_yaml(path: str | Path) -> dict[str, Any]:
    import yaml
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, MutableMapping):
        raise TypeError(f"Evaluation profile must be a mapping: {p}")
    return dict(normalise_evaluation_profile(data))


def save_normalised_yaml(path: str | Path, profile: MutableMapping[str, Any]) -> None:
    import yaml
    p = Path(path)
    normalise_evaluation_profile(profile)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8")
