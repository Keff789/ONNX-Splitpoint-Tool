from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .artifacts import artifact_exists, read_json


def stage_result_path(run_root: Path, *, stage: str, model_id: str | None = None) -> Path:
    if model_id:
        return Path(run_root) / "models" / str(model_id) / "stages" / f"{stage}.stage_result.json"
    return Path(run_root) / "stages" / f"{stage}.stage_result.json"


def stage_is_complete(
    run_root: Path,
    *,
    stage: str,
    model_id: str | None,
    expected_input_hash: str | None,
    expected_artifacts: Iterable[str],
    forced: bool = False,
) -> tuple[bool, str]:
    if forced:
        return False, "stage was forced"
    p = stage_result_path(run_root, stage=stage, model_id=model_id)
    if not p.is_file():
        return False, "missing stage_result"
    try:
        payload = dict(read_json(p) or {})
    except Exception:
        return False, "unreadable stage_result"
    if str(payload.get("status") or "").lower() not in {"ok", "skipped", "warn"}:
        return False, "previous stage status is not reusable"
    if expected_input_hash and payload.get("input_hash") and str(payload.get("input_hash")) != str(expected_input_hash):
        return False, "input_hash changed"
    arts = list(expected_artifacts or []) or list(payload.get("artifacts") or [])
    missing = [str(a) for a in arts if not artifact_exists(run_root, a)]
    if missing:
        return False, "missing artifacts: " + ", ".join(missing[:5])
    return True, "existing complete stage result matched input hash"
