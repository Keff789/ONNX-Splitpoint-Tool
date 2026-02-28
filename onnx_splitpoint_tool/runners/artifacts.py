from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (best effort)."""

    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    txt = json.dumps(data, indent=2, sort_keys=False, default=_json_default)
    tmp.write_text(txt)
    tmp.replace(path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
