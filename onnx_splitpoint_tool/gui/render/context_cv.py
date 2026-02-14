"""Context export helpers for CV-style analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ...gui_app import _write_results_csv, _write_summary_md, _write_table_tex


def export_context(path_base: Path, rows: List[Dict[str, Any]], tag: str, objective: str, topk: int = 10) -> None:
    _write_results_csv(path_base.with_suffix('.csv'), rows)
    _write_table_tex(path_base.with_suffix('.tex'), rows, tag, objective, topk=topk)
    _write_summary_md(path_base.with_suffix('.md'), rows, tag, objective, topk=topk)
