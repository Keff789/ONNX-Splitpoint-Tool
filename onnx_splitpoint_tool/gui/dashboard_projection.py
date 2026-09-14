"""Text projection of current and legacy result-dashboard contracts."""

from __future__ import annotations

from typing import Any, Mapping


def _count(value: Any) -> int | None:
    # Missing or malformed evidence is not an observed zero.
    return value if type(value) is int and value >= 0 else None


def _count_text(value: Any) -> str:
    count = _count(value)
    return str(count) if count is not None else "unavailable"


def dashboard_summary_line(dashboard: Mapping[str, Any]) -> str:
    """Use the owning summary's model count, never measurement-row counts.

    The scientific compatibility dashboard has ``summary`` and no legacy
    model-health cards. Keep the legacy projection for older saved runs, but
    do not substitute its counters for missing current-contract evidence.
    """

    summary = dashboard.get("summary")
    if isinstance(summary, Mapping):
        return f"models={_count_text(summary.get('model_count'))}"

    overview = dashboard.get("overview")
    if not isinstance(overview, Mapping):
        overview = dashboard
    health = overview.get("model_health_counts")
    if not isinstance(health, Mapping):
        health = {}
    split_models = None
    hailo_models = None
    cards = dashboard.get("models")
    if isinstance(cards, list) and all(isinstance(card, Mapping) for card in cards):
        split_counts = [_count(card.get("complete_split_count")) for card in cards]
        if all(count is not None for count in split_counts):
            split_models = sum(count > 0 for count in split_counts)
        runtime_flags = [
            card["hailo_evidence"].get("runtime_verified")
            if isinstance(card.get("hailo_evidence"), Mapping) else None
            for card in cards
        ]
        if all(type(flag) is bool for flag in runtime_flags):
            hailo_models = sum(runtime_flags)
    fields = [
        ("models", overview.get("model_count")),
        *((status, health.get(status)) for status in ("ok", "warn", "partial", "failed")),
        ("complete_split_models", split_models),
        ("hailo_runtime_models", hailo_models),
    ]
    return ", ".join(f"{name}={_count_text(value)}" for name, value in fields)
