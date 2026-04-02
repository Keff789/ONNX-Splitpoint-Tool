from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .schema import dumps_json_safe


SCHEMA = "onnx-splitpoint/generation-state"
SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")



def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default



def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(dumps_json_safe(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)



def init_state(
    *,
    model_name: str,
    model_source: str,
    requested_cases: int,
    ranked_candidates: Iterable[int],
    candidate_search_pool: Iterable[int],
    hef_full_policy: str,
    run_mode: str,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "model_name": str(model_name),
        "model_source": str(model_source),
        "requested_cases": int(requested_cases),
        "ranked_candidates": [int(x) for x in ranked_candidates],
        "candidate_search_pool": [int(x) for x in candidate_search_pool],
        "completed_boundaries": [],
        "accepted_boundaries": [],
        "discarded_boundaries": [],
        "generated_cases": 0,
        "discarded_cases": 0,
        "shortfall": max(0, int(requested_cases)),
        "current_boundary": None,
        "hef_full_policy": str(hef_full_policy),
        "run_mode": str(run_mode),
        "suite_full_hefs": {},
    }



def update_state(path: Path, state: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    state = dict(state or {})
    state.update(fields)
    state["updated_at"] = now_iso()
    write_json_atomic(path, state)
    return state



def find_latest_resumable_set(parent_dir: Path, model_base: str) -> Optional[Path]:
    parent_dir = Path(parent_dir)
    prefix = f"{model_base}_benchmark_"
    candidates: list[tuple[float, Path]] = []
    try:
        for child in parent_dir.iterdir():
            if not child.is_dir():
                continue
            if not child.name.startswith(prefix):
                continue
            state_path = child / "generation_state.json"
            if not state_path.exists():
                continue
            state = read_json(state_path, default={}) or {}
            if not isinstance(state, dict):
                continue
            if str(state.get("status") or "").strip().lower() in {"complete", "completed", "ok"}:
                continue
            try:
                mtime = float(state_path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            candidates.append((mtime, child))
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
