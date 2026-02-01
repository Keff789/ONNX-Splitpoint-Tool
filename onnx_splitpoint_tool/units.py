"""Unit helpers (bytes, bandwidth, FLOPs)."""

from __future__ import annotations

from typing import Optional, Dict

# Multipliers for memory sizes (value * UNIT_MULT[unit] -> bytes)

UNIT_MULT = {
    "bytes": 1.0,
    "KB": 1e3,
    "MB": 1e6,
    "GB": 1e9,
    "KiB": 1024.0,
    "MiB": 1024.0**2,
    "GiB": 1024.0**3,
}

# Bandwidth units expressed as BYTES / second
BANDWIDTH_MULT = {
    "B/s": 1.0,
    "KB/s": 1e3,
    "MB/s": 1e6,
    "GB/s": 1e9,
    "TB/s": 1e12,
    "KiB/s": 1024.0,
    "MiB/s": 1024.0**2,
    "GiB/s": 1024.0**3,
    "TiB/s": 1024.0**4,
    # bits/s variants
    "bps": 1.0 / 8.0,
    "Kbps": 1e3 / 8.0,
    "Mbps": 1e6 / 8.0,
    "Gbps": 1e9 / 8.0,
    "Tbps": 1e12 / 8.0,
}

FLOP_UNITS = {"FLOP": 1, "KFLOP": 1e3, "MFLOP": 1e6, "GFLOP": 1e9, "TFLOP": 1e12}


def bandwidth_to_bytes_per_s(value: Optional[float], unit: str) -> Optional[float]:
    """Convert a numeric bandwidth value (e.g., 1000) and unit (e.g., MB/s) to bytes/s."""
    if value is None:
        return None
    unit = (unit or "").strip()
    if unit not in BANDWIDTH_MULT:
        return None
    try:
        return float(value) * float(BANDWIDTH_MULT[unit])
    except Exception:
        return None

