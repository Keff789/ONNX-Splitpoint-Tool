"""Validated execution budgets; independent of scientific resampling budgets."""
from __future__ import annotations

import math
import os
from pathlib import Path

DEFAULTS = {"engine": "legacy", "max_active_requests": 1,
            "block_repetitions": 256, "checkpoint_blocks": False,
            "prepared_cache_limit_mib": 512}


def statistics_options(profile):
    quality = (profile or {}).get("quality_gate") or {}
    raw = quality.get("statistics") or {}
    out = {key: raw.get(key, default) for key, default in DEFAULTS.items()}
    if out["engine"] not in {"legacy", "optimized_coco_v1"}:
        raise ValueError("quality_gate.statistics.engine must be legacy or optimized_coco_v1")
    for key, low, high in [("max_active_requests",1,2), ("block_repetitions",1,5000),
                           ("prepared_cache_limit_mib",16,65536)]:
        if type(out[key]) is not int or not low <= out[key] <= high:
            raise ValueError(f"quality_gate.statistics.{key} must be an integer in [{low}, {high}]")
    if type(out["checkpoint_blocks"]) is not bool:
        raise ValueError("quality_gate.statistics.checkpoint_blocks must be boolean")
    if out["engine"] != "legacy":
        known = set(DEFAULTS) | {"method", "confidence_level", "bootstrap_repetitions", "seed",
            "decision", "execution_location", "workers"}
        if set(raw)-known:
            raise ValueError("unknown optimized statistics settings: " + ", ".join(sorted(set(raw)-known)))
        workers = raw.get("workers",4)
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("quality_gate.statistics.workers must be an integer in [1, 64]")
    return out


def reference_threads(profile):
    quality = (profile or {}).get("quality_gate") or {}
    raw = (quality.get("management_reference") or {}).get("intra_op_threads")
    if raw is None:
        # Historical profiles retain their reference generation contract.
        try:
            return max(1,min(64,int((quality.get("statistics") or {}).get("workers") or 4)))
        except (ValueError,TypeError):
            return 4
    if type(raw) is not int or not 1 <= raw <= 64:
        raise ValueError("quality_gate.management_reference.intra_op_threads must be null or an integer in [1, 64]")
    return raw


def resource_budget():
    """Linux affinity and hierarchical cgroup quotas, never compiler tokens."""
    affinity = len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else (os.cpu_count() or 1)
    cpus = float(affinity)
    info = dict((line.split(':')[0], int(line.split()[1])*1024)
                for line in Path('/proc/meminfo').read_text().splitlines())
    available = info['MemAvailable']
    try:
        entry = next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        current = Path('/sys/fs/cgroup') / entry.lstrip('/')
        root = Path('/sys/fs/cgroup')
        while current == root or root in current.parents:
            try:
                quota,period=(current/'cpu.max').read_text().split()
                if quota != 'max': cpus=min(cpus,int(quota)/int(period))
            except (OSError,ValueError): pass
            try:
                limit=(current/'memory.max').read_text().strip()
                if limit != 'max': available=min(available,max(0,int(limit)-int((current/'memory.current').read_text())))
            except (OSError,ValueError): pass
            if current == root: break
            current=current.parent
    except (OSError,StopIteration): pass
    return {"affinity_cpus":affinity,"quota_cpus":cpus,"available_memory_bytes":available,
            "statistics_cpu_slots":max(1,math.floor(cpus)-1),"controller_cpu_reserve":1}
