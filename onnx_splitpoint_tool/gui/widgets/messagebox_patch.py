from __future__ import annotations

import tkinter as tk
from tkinter import messagebox as _messagebox
from typing import Any

from .diagnostic_dialog import show_diagnostic_dialog

_INSTALLED = False
_ORIGINALS: dict[str, Any] = {}


def _split_message(message: Any) -> tuple[str, str]:
    text = str(message or "")
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    nonempty = [ln for ln in lines if ln]
    if not text.strip():
        return "", ""
    first = nonempty[0] if nonempty else text[:160]
    # Keep the headline/summary compact.  Full text goes into details.
    if len(first) > 160:
        first = first[:157] + "..."
    if "\n" in text or len(text) > 220 or "Traceback" in text or "Error:" in text or "Exception" in text:
        return first, text
    return text, ""


def install_messagebox_diagnostics() -> None:
    """Route simple messagebox show* calls through the readable diagnostic dialog.

    Tk's stock messagebox can render huge fonts and unreadable walls of text on
    some Linux themes.  Keep askyesno/askokcancel untouched; only informational
    show dialogs are replaced.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    for name in ("showinfo", "showwarning", "showerror"):
        _ORIGINALS[name] = getattr(_messagebox, name)

    def _make(kind: str):
        severity = {"showinfo": "info", "showwarning": "warning", "showerror": "error"}.get(kind, "info")
        original = _ORIGINALS[kind]

        def _wrapped(title: str | None = None, message: Any = None, **options: Any) -> str:
            parent = options.get("parent") or getattr(tk, "_default_root", None)
            if parent is None:
                return original(title or "Message", str(message or ""), **options)
            headline, details = _split_message(message)
            try:
                show_diagnostic_dialog(
                    parent,
                    title=str(title or "Message"),
                    headline=headline or str(title or "Message"),
                    summary="" if details else str(message or ""),
                    details=details,
                    severity=severity,
                )
                return "ok"
            except Exception:
                return original(title or "Message", str(message or ""), **options)

        return _wrapped

    _messagebox.showinfo = _make("showinfo")  # type: ignore[assignment]
    _messagebox.showwarning = _make("showwarning")  # type: ignore[assignment]
    _messagebox.showerror = _make("showerror")  # type: ignore[assignment]
