"""Context helpers for LLM validation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ...gui_app import _run_case


def run_case(case_dir: Path, provider: str, image: str, preset: str, warmup: int, runs: int, timeout_s: Optional[int], case_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    return _run_case(case_dir, provider, image, preset, warmup, runs, timeout_s, case_meta)
