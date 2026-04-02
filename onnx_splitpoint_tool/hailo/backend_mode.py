from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple

VALID_HAILO_BACKENDS: Tuple[str, ...] = ("auto", "subprocess", "local", "wsl", "venv")



def normalize_hailo_backend(mode: str | None) -> str:
    value = str(mode or "auto").strip().lower()
    if value not in VALID_HAILO_BACKENDS:
        return "auto"
    return value



def subprocess_backend_for_platform() -> str:
    return "wsl" if sys.platform == "win32" else "venv"



def auto_prefers_subprocess() -> bool:
    pref = str(os.environ.get("SPLITPOINT_HAILO_AUTO_MODE", "subprocess") or "subprocess").strip().lower()
    return pref not in {"local", "inprocess"}



def backend_display_values() -> Tuple[str, ...]:
    return VALID_HAILO_BACKENDS
