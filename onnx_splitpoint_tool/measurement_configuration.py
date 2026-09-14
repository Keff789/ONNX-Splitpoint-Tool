"""Read-only display of resolved configuration, never measurement evidence."""
from __future__ import annotations
import math
from typing import Any, Mapping


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, *, integer: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0 or (integer and int(value) != value):
        return None
    return int(value) if integer else value


def measurement_configuration(profile: Mapping[str, Any], execution_plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Use supplied resolved values without loading today's defaults for old runs.

    Presentation does not enter plan/cache identities or determine admission.
    """
    plan = _mapping(execution_plan)
    snapshot = _mapping(_mapping(profile.get("execution_preset")).get("snapshot"))
    data, quality = _mapping(snapshot.get("data")), _mapping(snapshot.get("quality"))
    statistics = _mapping(_mapping(profile.get("quality_gate")).get("statistics"))
    native = _mapping(profile.get("native_producers"))
    energy = _mapping(native.get("energy"))
    top_energy, snapshot_energy = _mapping(profile.get("energy")), _mapping(snapshot.get("energy"))
    system = _mapping(_mapping(profile.get("measurement_campaign")).get("system_power"))
    duration, duration_source = None, "unavailable"
    for source, value in (
        ("native_producers.energy.duration_s", energy.get("duration_s")),
        ("native_producers.energy.measurement_duration_s", energy.get("measurement_duration_s")),
        ("native_producers.native_energy_duration_s", native.get("native_energy_duration_s")),
    ):
        number = _number(value)
        if number is not None and number > 0:
            duration, duration_source = number, str(energy.get("duration_source") or source)
            break
    repeats, repeats_source = None, "unavailable"
    for source, cfg in (("native_producers.energy", energy), ("energy", top_energy), ("execution_preset.snapshot.energy", snapshot_energy)):
        for key in ("repeat_override", "repeats"):
            number = _number(cfg.get(key), integer=True)
            if number is not None and number > 0:
                repeats, repeats_source = number, source + "." + key
                break
        if repeats is not None:
            break
    validation = _mapping(plan.get("validation_items")) if "validation_items" in plan else _mapping(data.get("validation_items"))
    bootstrap = plan.get("bootstrap_repetitions") if "bootstrap_repetitions" in plan else statistics.get("bootstrap_repetitions", quality.get("bootstrap_repetitions"))
    compute = _mapping(_mapping(profile.get("hailo_build")).get("compute_by_family"))
    def declared_text(*fields: tuple[str, Any]) -> tuple[str | None, str]:
        for source, value in fields:
            if isinstance(value, str) and value.strip():
                return value.strip(), source
        return None, "unavailable"
    scope, scope_source = declared_text(
        ("measurement_campaign.system_power.scope", system.get("scope")),
        ("native_producers.energy.physical_scope", energy.get("physical_scope")),
        ("energy.physical_scope", top_energy.get("physical_scope")),
        ("execution_preset.snapshot.energy.physical_scope", snapshot_energy.get("physical_scope")),
    )
    window, window_source = declared_text(
        ("measurement_campaign.system_power.window", system.get("window")),
        ("measurement_campaign.system_power.measurement_window", system.get("measurement_window")),
        ("native_producers.energy.window_label", energy.get("window_label")),
        ("energy.window_label", top_energy.get("window_label")),
        ("execution_preset.snapshot.energy.window_label", snapshot_energy.get("window_label")),
    )
    return {
        "role": "resolved_configuration_only",
        "validation_items": {task: _number(validation.get(task), integer=True) for task in ("classification", "detection")},
        "bootstrap_repetitions": _number(bootstrap, integer=True),
        "native_energy": {
            "duration_s": duration, "duration_source": duration_source,
            "repetitions": repeats, "repetitions_source": repeats_source,
            "physical_scope": scope, "physical_scope_source": scope_source,
            "window_label": window, "window_label_source": window_source,
            "measured_duration_s": None, "scientific_qualification": "requires_row_evidence",
        },
        "build_preferences": {family: str(_mapping(compute.get(family)).get("device") or "unavailable") for family in ("hailo8", "hailo10h")},
        "gpu_execution": "not_established_by_configuration",
        "profile_label_is_scientific_approval": False,
    }


def measurement_configuration_lines(config: Mapping[str, Any]) -> list[str]:
    energy, validation = _mapping(config.get("native_energy")), _mapping(config.get("validation_items"))
    compute = _mapping(config.get("build_preferences"))
    def show(value: Any) -> str:
        return "unavailable" if value is None else str(value)
    return [
        "Resolved validation CLS/DET=" + show(validation.get("classification")) + "/" + show(validation.get("detection")) + "; bootstrap=" + show(config.get("bootstrap_repetitions")),
        "Native energy request: " + show(energy.get("duration_s")) + " s × " + show(energy.get("repetitions")) + "; scope=" + show(energy.get("physical_scope")) + "; window=" + show(energy.get("window_label")) + "; measured duration and qualification require row evidence",
        "New-build preferences: Hailo8=" + show(compute.get("hailo8")) + "; Hailo10H=" + show(compute.get("hailo10h")) + "; GPU execution requires job evidence; compatible artifacts remain reusable",
        "Profile label is configuration; scientific approval is determined per result.",
    ]
