from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)

_DEFAULT: Dict[str, List[Dict[str, Any]]] = {
    "accelerators": [
        {"id": "jetson_xavier_nx_8gb", "name": "Jetson Xavier NX (8GB)", "ram_limit_mb": 8192, "runtime_overhead_mb": 400, "notes": "Host RAM estimate.", "perf": {"tops_int8": 21, "gflops_fp16": 0, "efficiency_factor": 0.7}},
        {"id": "jetson_orin_nx_8gb", "name": "Jetson Orin NX (8GB)", "ram_limit_mb": 8192, "runtime_overhead_mb": 500, "notes": "Shared memory with system processes.", "perf": {"tops_int8": 70, "gflops_fp16": 0, "efficiency_factor": 0.75}},
        {"id": "jetson_orin_nx_16gb", "name": "Jetson Orin NX (16GB)", "ram_limit_mb": 16384, "runtime_overhead_mb": 700, "notes": "Shared memory with system processes.", "perf": {"tops_int8": 100, "gflops_fp16": 0, "efficiency_factor": 0.75}},
        {"id": "hailo_8", "name": "Hailo-8", "ram_limit_mb": 4096, "runtime_overhead_mb": 256, "notes": "Conservative placeholder; verify host/device memory budget.", "perf": {"tops_int8": 26, "gflops_fp16": 0, "efficiency_factor": 0.6}},
        {"id": "hailo_10h", "name": "Hailo-10H", "ram_limit_mb": 6144, "runtime_overhead_mb": 320, "notes": "Conservative placeholder; verify deployment configuration.", "perf": {"tops_int8": 40, "gflops_fp16": 0, "efficiency_factor": 0.6}},
        {"id": "axelera_metis", "name": "Axelera Metis", "ram_limit_mb": 4096, "runtime_overhead_mb": 320, "notes": "Conservative placeholder.", "perf": {"tops_int8": 80, "gflops_fp16": 0, "efficiency_factor": 0.65}},
        {"id": "axelera_metis_max", "name": "Axelera Metis Max", "ram_limit_mb": 8192, "runtime_overhead_mb": 384, "notes": "Conservative placeholder.", "perf": {"tops_int8": 120, "gflops_fp16": 0, "efficiency_factor": 0.65}},
        {"id": "deepx_dx_m1", "name": "DeepX DX-M1", "ram_limit_mb": 4096, "runtime_overhead_mb": 256, "notes": "Conservative placeholder.", "perf": {"tops_int8": 25, "gflops_fp16": 0, "efficiency_factor": 0.6}},
    ],
    "interfaces": [
        {"id": "pcie_gen3_x4", "name": "PCIe Gen3 x4 (M.2 Key M)", "bandwidth_mb_s": 3500, "latency_overhead_ms": 0.1},
        {"id": "pcie_gen4_x4", "name": "PCIe Gen4 x4", "bandwidth_mb_s": 7000, "latency_overhead_ms": 0.08},
        {"id": "ethernet_10g", "name": "Ethernet 10G", "bandwidth_mb_s": 1100, "latency_overhead_ms": 0.25},
    ],
}


# Allow power-users to override the accelerator DB without modifying the repo.
# Example:
#   set SPLITPOINT_ACCEL_DB=C:\path\to\accelerators.json
_ENV_ACCEL_DB = "SPLITPOINT_ACCEL_DB"


def _resource_dir() -> Path:
    return Path(__file__).resolve().parent / "resources"


def _user_db_path() -> Path:
    # Keep consistent with the GUI log/config folder the tool already uses.
    return Path.home() / ".onnx_splitpoint_tool" / "accelerators.json"


def _bundled_db_candidates() -> List[Path]:
    # Prefer richer schemas when present.
    base = _resource_dir()
    return [
        base / "accelerators_updated_v2.json",
        base / "accelerators_updated.json",
        base / "accelerators.json",
    ]


def _iter_candidate_db_paths() -> List[Path]:
    candidates: List[Path] = []

    env = (os.environ.get(_ENV_ACCEL_DB) or "").strip()
    if env:
        candidates.append(Path(env).expanduser())

    candidates.append(_user_db_path())
    candidates.extend(_bundled_db_candidates())
    return candidates


def load_accelerator_specs() -> Dict[str, List[Dict[str, Any]]]:
    """Load accelerator/interface specs.

    Search order:
      1) $SPLITPOINT_ACCEL_DB (if set)
      2) ~/.onnx_splitpoint_tool/accelerators.json (user override)
      3) bundled resources (prefer richest schema)

    Always returns a dict with keys: "accelerators" and "interfaces".
    """

    data: Dict[str, Any] | None = None
    loaded_from: Path | None = None

    for p in _iter_candidate_db_paths():
        try:
            if not p.exists():
                continue
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and isinstance(obj.get("accelerators"), list) and isinstance(obj.get("interfaces"), list):
                data = obj
                loaded_from = p
                break
        except Exception as e:
            LOGGER.warning("Failed to parse accelerator DB at %s: %s", p, e)
            continue

    if not isinstance(data, dict):
        data = dict(_DEFAULT)

    # Ensure minimal structure.
    accels = data.get("accelerators")
    if not isinstance(accels, list) or not accels:
        data["accelerators"] = list(_DEFAULT["accelerators"])
    if not isinstance(data.get("interfaces"), list):
        data["interfaces"] = list(_DEFAULT["interfaces"])

    if loaded_from is not None:
        LOGGER.info("Loaded accelerator DB: %s", loaded_from)
    return data
