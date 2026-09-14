from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Any, Optional

from ...log_utils import sanitize_log


_ORIGINAL_MESSAGEBOX_FUNCS: dict[str, Any] = {}


def _fallback_messagebox(level: str, title: str, msg: str) -> None:
    from tkinter import messagebox
    funcs = _ORIGINAL_MESSAGEBOX_FUNCS or {}
    level_l = str(level).lower()
    if level_l in {"error", "failed"}:
        fn = funcs.get("showerror") or messagebox.showerror
    elif level_l in {"warn", "warning"}:
        fn = funcs.get("showwarning") or messagebox.showwarning
    else:
        fn = funcs.get("showinfo") or messagebox.showinfo
    try:
        fn(title, msg)
    except Exception:
        pass


def show_detail_message(
    parent: Any,
    *,
    title: str,
    heading: str,
    message: str = "",
    details: str = "",
    level: str = "info",
    geometry: str = "840x560",
    log_path: Optional[str] = None,
) -> None:
    """Readable diagnostic dialog for long errors/logs."""
    root = getattr(parent, "root", None) or parent
    try:
        dlg = tk.Toplevel(root)
        dlg.title(title)
        try:
            dlg.transient(root)
            dlg.grab_set()
        except Exception:
            pass
        dlg.geometry(geometry)
        dlg.minsize(680, 420)
        outer = ttk.Frame(dlg, padding=14)
        outer.pack(fill="both", expand=True)
        title_font = tkfont.nametofont("TkDefaultFont").copy()
        try:
            base = int(title_font.cget("size"))
        except Exception:
            base = 10
        title_font.configure(size=max(base + 2, 12), weight="bold")
        icon = {"info": "ℹ", "ok": "✓", "success": "✓", "warn": "⚠", "warning": "⚠", "error": "✖", "failed": "✖"}.get(str(level).lower(), "ℹ")
        hdr = ttk.Frame(outer)
        hdr.pack(fill="x")
        ttk.Label(hdr, text=icon, font=title_font, width=3).pack(side="left", anchor="n")
        hf = ttk.Frame(hdr)
        hf.pack(side="left", fill="x", expand=True)
        ttk.Label(hf, text=str(heading or title), font=title_font, wraplength=760, justify="left").pack(anchor="w")
        if message:
            ttk.Label(hf, text=str(message), wraplength=760, justify="left").pack(anchor="w", pady=(6, 0))
        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True, pady=(12, 0))
        tab = ttk.Frame(nb, padding=8)
        nb.add(tab, text="Details")
        text = tk.Text(tab, wrap="word", height=18)
        text.pack(fill="both", expand=True)
        body = str(details or "")
        if log_path:
            body += "\n\nLog file:\n" + str(log_path)
        text.insert("1.0", body or str(message or heading or ""))
        text.configure(state="disabled")
        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(10, 0))
        def _copy() -> None:
            try:
                root.clipboard_clear()
                root.clipboard_append(sanitize_log(body or str(message or heading or "")))
                root.update_idletasks()
            except Exception:
                pass
        ttk.Button(btns, text="Copy details", command=_copy).pack(side="left")
        if log_path:
            def _open() -> None:
                try:
                    opener = getattr(parent, "_open_path", None)
                    if callable(opener):
                        opener(str(log_path))
                except Exception:
                    pass
            ttk.Button(btns, text="Open log", command=_open).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="OK", command=dlg.destroy).pack(side="right")
    except Exception:
        msg = f"{heading}\n\n{message}\n\n{details}"
        _fallback_messagebox(level, title, msg)



def _split_message_for_dialog(title: str, message: str) -> tuple[str, str, str]:
    """Return (heading, summary, details) for patched messagebox calls."""
    raw = str(message or "")
    clean = raw.strip()
    if not clean:
        return str(title or "Message"), "", ""
    parts = [p.strip() for p in clean.split("\n\n", 1)]
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    # Keep the heading compact. If the first paragraph is already long, use the
    # dialog title as the heading and move the text into the details area.
    if len(first) > 140 or "\n" in first:
        return str(title or "Message"), "See details below.", clean
    return first, "", rest or clean


def install_messagebox_replacement(parent: Any) -> None:
    """Replace Tk's huge default info/warn/error boxes with scrollable dialogs.

    This intentionally leaves askyesno/askokcancel untouched, because those need
    return values and are usually short confirmation prompts.
    """
    from tkinter import messagebox

    if getattr(messagebox, "_splitpoint_dialog_patch", False):
        return
    for name in ("showinfo", "showwarning", "showerror"):
        try:
            _ORIGINAL_MESSAGEBOX_FUNCS[name] = getattr(messagebox, name)
        except Exception:
            pass

    def _make(level: str):
        def _wrapped(title: str = "", message: str = "", *args: Any, **kwargs: Any) -> str:
            root = kwargs.get("parent") or parent
            heading, summary, details = _split_message_for_dialog(str(title or "Message"), str(message or ""))
            try:
                show_detail_message(
                    root,
                    title=str(title or "Message"),
                    heading=heading,
                    message=summary,
                    details=details,
                    level=level,
                    geometry="860x540",
                )
            except Exception:
                _fallback_messagebox(level, str(title or "Message"), str(message or ""))
            return "ok"
        return _wrapped

    try:
        messagebox.showinfo = _make("info")  # type: ignore[assignment]
        messagebox.showwarning = _make("warning")  # type: ignore[assignment]
        messagebox.showerror = _make("error")  # type: ignore[assignment]
        messagebox._splitpoint_dialog_patch = True  # type: ignore[attr-defined]
    except Exception:
        pass
