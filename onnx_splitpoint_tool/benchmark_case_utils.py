from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union


def _maybe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def build_benchmark_case_rejection(
    *,
    boundary: int,
    folder: str,
    reason: str,
    stage: Optional[str] = None,
    hw_arch: Optional[str] = None,
    detail: Optional[str] = None,
    hef_result: Any = None,
) -> Dict[str, Any]:
    """Return a structured rejection record for a benchmark candidate.

    The GUI uses this when a split is intentionally discarded during benchmark-set
    generation (for example because a required Hailo HEF build failed).
    """

    record: Dict[str, Any] = {
        "status": "rejected",
        "boundary": int(boundary),
        "folder": str(folder),
        "reason": str(reason),
    }
    if stage:
        record["stage"] = str(stage)
    if hw_arch:
        record["hw_arch"] = str(hw_arch)

    error_text = detail
    if hef_result is not None:
        result_error = _maybe_attr(hef_result, "error", None)
        if error_text in (None, "") and result_error not in (None, ""):
            error_text = str(result_error)

        record["ok"] = bool(_maybe_attr(hef_result, "ok", False))
        record["skipped"] = bool(_maybe_attr(hef_result, "skipped", False))
        record["timed_out"] = bool(_maybe_attr(hef_result, "timed_out", False))

        for key in (
            "failure_kind",
            "timeout_kind",
            "last_stage",
            "unsupported_reason",
            "debug_log",
            "returncode",
            "backend",
            "net_name",
            "hw_arch",
        ):
            value = _maybe_attr(hef_result, key, None)
            if value not in (None, ""):
                record[key] = value

        details = _maybe_attr(hef_result, "details", None)
        if isinstance(details, dict) and details:
            record["details"] = dict(details)

    if error_text not in (None, ""):
        record["detail"] = str(error_text)

    return record


def archive_benchmark_case(
    case_dir: Union[str, Path],
    suite_dir: Union[str, Path],
    *,
    folder: Optional[str] = None,
    rejected_root_name: str = "_rejected_cases",
) -> Dict[str, Any]:
    """Move a rejected benchmark case under ``<suite>/_rejected_cases/``."""

    case_path = Path(case_dir)
    suite_path = Path(suite_dir)
    target_folder = str(folder or case_path.name)
    info: Dict[str, Any] = {
        "archive_root": str(rejected_root_name),
    }

    if not case_path.exists():
        return info

    try:
        reject_root = suite_path / str(rejected_root_name)
        reject_root.mkdir(parents=True, exist_ok=True)
        target = reject_root / target_folder
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(case_path), str(target))
        info["archive_dir"] = Path(os.path.relpath(target, start=suite_path)).as_posix()
    except Exception as e:  # pragma: no cover - defensive
        info["archive_error"] = f"{type(e).__name__}: {e}"

    return info
