from __future__ import annotations

"""Canonical Hailo hard-timeout parsing.

The compiler is reached through several entry points (profile materialisation,
the legacy BenchmarkSet service, the backend adapter and the GUI).  Keeping a
single parser here prevents one entry point from treating ``unlimited`` as
zero while another silently restores a finite default.
"""

import math
from typing import Any


# The first five spellings are the documented public contract.  The final
# three are retained legacy aliases; every caller accepts the same complete
# set so an old environment override cannot acquire different semantics.
HAILO_UNLIMITED_TIMEOUT_TOKENS = frozenset(
    {"0", "off", "none", "unlimited", "disabled", "disable", "false", "no"}
)


def is_hailo_timeout_unlimited(value: Any) -> bool:
    """Return whether *value* explicitly selects no wall-clock timeout."""

    if isinstance(value, bool):
        return value is False
    return str(value if value is not None else "").strip().lower() in (
        HAILO_UNLIMITED_TIMEOUT_TOKENS
    )


def parse_hailo_timeout_seconds(
    value: Any,
    *,
    default: Any,
    minimum_enabled_s: int = 0,
    label: str = "Hailo hard timeout",
) -> int:
    """Parse a finite timeout or the canonical unlimited value ``0``.

    Empty values use *default*.  Positive values smaller than
    ``minimum_enabled_s`` are raised to that operational floor.  Negative,
    fractional, non-finite and unknown textual values fail closed.
    """

    selected = default if value is None or str(value).strip() == "" else value
    if is_hailo_timeout_unlimited(selected):
        return 0
    if isinstance(selected, bool):
        parsed = int(selected)
    else:
        try:
            number = float(selected)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {label}: {selected!r}") from exc
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"invalid {label}: {selected!r}")
        parsed = int(number)
    if parsed < 0:
        raise ValueError(f"{label} must be >= 0")
    if parsed == 0:
        return 0
    return max(int(minimum_enabled_s), parsed)


def canonical_hailo_disable_tokens() -> list[Any]:
    """Return the stable documented token list used in persisted profiles."""

    return [0, "off", "none", "unlimited", "disabled"]
