"""Helpers for resolving the Hailo parse-check budget in the GUI.

Historically the GUI defaulted the Hailo parse-check budget to a hard-coded
``25``. That meant users could raise analysis ``Top-k`` to values like ``40``
but would still only see progress like ``x/25`` and, before the regression
fix in v39c, the selection loop could stop early after 25 checked candidates.

This helper keeps the policy lightweight and testable:

* ``auto`` (and empty values) resolve to the current analysis ``Top-k``.
* explicit positive integers remain supported as a manual override.
* persisted legacy default values such as ``25`` can be normalized to
  ``auto`` on load so upgrades behave sensibly without manual cleanup.
"""

from __future__ import annotations

from typing import Any, Tuple


def resolve_hailo_max_checks(raw: Any, *, topk: int) -> Tuple[int, bool]:
    """Resolve the effective Hailo parse-check budget.

    Returns ``(budget, is_auto)`` where ``is_auto`` indicates whether the
    budget followed ``Top-k`` automatically.

    Accepted values for ``raw``:

    * ``"auto"`` / ``""`` / ``None`` -> use ``topk``
    * positive integer strings / ints -> use the explicit value
    """

    try:
        topk_int = int(topk)
    except Exception as exc:  # pragma: no cover - defensive guardrail
        raise ValueError("Top-k must be a positive integer.") from exc

    if topk_int <= 0:
        raise ValueError("Top-k must be a positive integer.")

    text = str(raw or "").strip().lower()
    if text in {"", "auto"}:
        return int(topk_int), True

    try:
        value = int(text)
    except Exception as exc:
        raise ValueError("Hailo max checks must be a positive integer or 'auto'.") from exc

    if value <= 0:
        raise ValueError("Hailo max checks must be a positive integer or 'auto'.")

    return int(value), False


def normalize_persisted_hailo_max_checks(raw: Any) -> str:
    """Normalize persisted GUI state for ``var_hailo_max_checks``.

    Older builds persisted the hard-coded default ``25``. That value behaved
    like a hidden cap rather than an intentional user choice. On upgrade we
    map that legacy default to ``auto`` so the analysis budget follows the
    current ``Top-k`` out of the box.
    """

    text = str(raw or "").strip()
    if not text:
        return "auto"
    if text == "25":
        return "auto"
    return text
