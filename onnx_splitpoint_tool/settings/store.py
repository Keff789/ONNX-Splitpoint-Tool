from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _splitpoint_home() -> Path:
    # Keep consistent with other tool state (gui.log, hailo venvs, caches)
    return Path.home() / ".onnx_splitpoint_tool"


def default_settings() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "last_opened_at": None,
        # Persisted tkinter variable values (StringVar/IntVar/BooleanVar…)
        "tk_vars": {},
        # Working directory root (historically called "output_dir")
        "output_dir": None,
        "working_dir": None,
        # Remote benchmarking
        "remote_hosts": [],
        "remote_selected_host_id": None,
    }


@dataclass
class SettingsStore:
    """Load/save persistent settings.

    The store intentionally keeps settings as a plain dict to remain forward
    compatible with new keys across tool versions.
    """

    filename: str = "settings.json"
    home: Path = field(default_factory=_splitpoint_home)

    def path(self) -> Path:
        return self.home / self.filename

    def load(self) -> Dict[str, Any]:
        path = self.path()
        base = default_settings()

        if not path.exists():
            return base

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("settings.json root is not an object")
            # merge defaults (do not delete unknown keys)
            merged = dict(base)
            merged.update(data)

            # Backwards/forwards compatibility: accept either key.
            if (not merged.get("output_dir")) and merged.get("working_dir"):
                merged["output_dir"] = merged.get("working_dir")
            if (not merged.get("working_dir")) and merged.get("output_dir"):
                merged["working_dir"] = merged.get("output_dir")
            return merged
        except Exception:
            # Backup corrupted file
            try:
                ts = time.strftime("%Y%m%d_%H%M%S")
                bak = path.with_name(f"{path.name}.bak.{ts}")
                bak.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    bak.write_bytes(path.read_bytes())
            except Exception:
                # Best-effort backup
                pass
            return base

    def save(self, data: Dict[str, Any]) -> None:
        # Ensure home exists
        self.home.mkdir(parents=True, exist_ok=True)
        path = self.path()
        tmp = path.with_suffix(path.suffix + ".tmp")

        # Shallow copy so we can stamp timestamp without mutating caller
        payload = dict(data or {})
        payload.setdefault("schema_version", 1)
        payload["last_opened_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Keep both keys in sync for compatibility.
        if payload.get("output_dir") and not payload.get("working_dir"):
            payload["working_dir"] = payload.get("output_dir")
        if payload.get("working_dir") and not payload.get("output_dir"):
            payload["output_dir"] = payload.get("working_dir")

        # Atomic write
        txt = json.dumps(payload, indent=2, sort_keys=True)
        tmp.write_text(txt, encoding="utf-8")
        os.replace(tmp, path)

    # Convenience helpers -------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.load().get(key, default)

    def update(self, patch: Dict[str, Any]) -> None:
        data = self.load()
        data.update(patch)
        self.save(data)
