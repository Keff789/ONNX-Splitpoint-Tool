"""Small productive build boundary shared by workflow starts and dispatch.

Missing artifacts remain buildable. Force is reserved for the separate private
diagnostic builder; it is never inferred from a saved profile or Resume.
"""
from __future__ import annotations

import copy
from functools import wraps
from typing import Any, Callable, Mapping

from .config_values import parse_config_bool, validate_profile_config_booleans

NATIVE_FORCE_KEYS = frozenset({
    "force_rebuild_engines", "native_force_rebuild_engines",
    "force_rebuild_native_engines",
})


class ProductiveForceBuildDisabled(ValueError):
    error_code = "productive_force_build_disabled"


def active_native_force_fields(profile: Mapping[str, Any]) -> list[str]:
    """Inspect effective fields, excluding historical source snapshots."""
    fields: list[str] = []

    def walk(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if key == "execution_preset":
                    continue
                if key in NATIVE_FORCE_KEYS:
                    if parse_config_bool(item, field=path):
                        fields.append(path)
                else:
                    walk(item, path)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{prefix}[{index}]")

    walk(profile)
    return fields


def require_productive_force_off(
    profile: Mapping[str, Any], *, hailo_force_build: bool = False,
) -> None:
    validate_profile_config_booleans(profile)
    fields = [
        f"{family}_build.force_build" for family in ("hailo", "deepx")
        if parse_config_bool(
            (profile.get(f"{family}_build") or {}).get("force_build", False),
            field=f"{family}_build.force_build",
        )
    ]
    if parse_config_bool(hailo_force_build, field="options.hailo_force_build"):
        fields.append("options.hailo_force_build")
    fields.extend(active_native_force_fields(profile))
    if fields:
        raise ProductiveForceBuildDisabled(
            "productive_force_build_disabled: Force muss für normale Starts "
            "und Resume AUS sein: " + ", ".join(fields)
            + ". Passende Artefakte werden wiederverwendet; fehlende dürfen "
            "gebaut werden. Für alte Force-Snapshots ein neues Profil mit "
            "Force AUS verwenden; Archive bleiben unverändert."
        )


def profile_hailo_compute_mapping(profile: Mapping[str, Any]) -> dict[str, Any]:
    from .hailo_compiler_context import normalize_compute_by_family
    block = profile.get("hailo_build") or {}
    return normalize_compute_by_family(block.get("compute_by_family"))


def bind_profile_hailo_builder(
    builder: Callable[..., Any], profile: Mapping[str, Any],
) -> Callable[..., Any]:
    """Carry a detached mapping through scheduler threads and saved calls.

    This is an argument adapter around the existing builder, not a second
    compiler. No parent environment or thread-local context is changed.
    """
    require_productive_force_off(profile)
    mapping = copy.deepcopy(profile_hailo_compute_mapping(profile))

    @wraps(builder)
    def dispatch(source: Any, **kwargs: Any) -> Any:
        if parse_config_bool(kwargs.get("force", False), field="hailo_build.force"):
            raise ProductiveForceBuildDisabled("productive_force_build_disabled:hailo_build.force")
        if "compute_by_family" in kwargs:
            from .hailo_compiler_context import normalize_compute_by_family
            supplied = normalize_compute_by_family(kwargs["compute_by_family"])
            if supplied != mapping:
                raise ValueError("hailo_compute_profile_dispatch_mismatch")
        # Avoid adding an empty override to legacy callers: it must not hide
        # the resolver's explicit legacy-environment source diagnostics.
        if mapping:
            kwargs["compute_by_family"] = copy.deepcopy(mapping)
        return builder(source, **kwargs)

    return dispatch
