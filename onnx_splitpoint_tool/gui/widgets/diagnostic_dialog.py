from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Iterable, Optional

from ...log_utils import sanitize_log


def show_diagnostic_dialog(
    parent: tk.Misc | None,
    *,
    title: str,
    headline: str | None = None,
    heading: str | None = None,
    summary: str = "",
    message: str = "",
    details: str = "",
    hints: Optional[Iterable[str]] = None,
    severity: str = "info",
    geometry: str = "860x560",
    log_path: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Show a readable diagnostic dialog.

    The stock Tk messagebox renders long messages with an oversized font on
    some Linux themes.  This dialog keeps the top-level message compact and puts
    verbose logs into a scrollable detail area with copy/open-log support.
    """
    root = parent if parent is not None else tk._default_root  # type: ignore[attr-defined]
    if root is None:
        return

    if width or height:
        # Character-ish dimensions are easier to call from older code; map them
        # to a sane pixel geometry.
        w = int(width or 110) * 8
        h = int(height or 32) * 18
        geometry = f"{max(700, min(1200, w))}x{max(420, min(900, h))}"

    heading_text = str(heading or headline or title or "Diagnostics")
    summary_text = str(summary or message or "")
    sev = (severity or "info").lower()
    icon = {"error": "✖", "failed": "✖", "warning": "⚠", "warn": "⚠", "ok": "✓", "success": "✓", "info": "ℹ"}.get(sev, "ℹ")

    dlg = tk.Toplevel(root)
    dlg.title(title)
    try:
        dlg.geometry(geometry)
        dlg.transient(root)
        dlg.grab_set()
    except Exception:
        pass
    try:
        dlg.minsize(700, 420)
    except Exception:
        pass

    outer = ttk.Frame(dlg, padding=12)
    outer.pack(fill=tk.BOTH, expand=True)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(2, weight=1)

    try:
        title_font = tkfont.nametofont("TkDefaultFont").copy()
        base = int(title_font.cget("size") or 10)
        title_font.configure(size=max(11, base + 1), weight="bold")
    except Exception:
        title_font = None

    hdr = ttk.Frame(outer)
    hdr.grid(row=0, column=0, sticky="ew")
    ttk.Label(hdr, text=icon, width=3, font=title_font).pack(side=tk.LEFT, anchor="n")
    htext = ttk.Frame(hdr)
    htext.pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Label(htext, text=heading_text, font=title_font, wraplength=760, justify="left").pack(anchor="w")
    if summary_text:
        ttk.Label(htext, text=summary_text, wraplength=780, justify="left").pack(anchor="w", pady=(6, 0))

    hint_lines = [str(h).strip() for h in list(hints or []) if str(h).strip()]
    if hint_lines:
        hint_box = ttk.LabelFrame(outer, text="Hinweise / nächste Schritte")
        hint_box.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for i, h in enumerate(hint_lines[:8]):
            ttk.Label(hint_box, text="• " + h, wraplength=780, justify="left").grid(row=i, column=0, sticky="w", padx=8, pady=(2, 2))
    else:
        ttk.Frame(outer).grid(row=1, column=0, sticky="ew")

    text_frame = ttk.Frame(outer)
    text_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 8))
    text_frame.rowconfigure(0, weight=1)
    text_frame.columnconfigure(0, weight=1)
    txt = tk.Text(text_frame, wrap="word", height=18)
    scroll = ttk.Scrollbar(text_frame, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=scroll.set)
    txt.grid(row=0, column=0, sticky="nsew")
    scroll.grid(row=0, column=1, sticky="ns")
    body_parts = []
    if details:
        body_parts.append(str(details))
    if log_path:
        body_parts.append("Log file:\n" + str(log_path))
    clean = sanitize_log("\n\n".join(body_parts) or "No additional details.")
    txt.insert("1.0", clean)
    txt.configure(state="disabled")

    btns = ttk.Frame(outer)
    btns.grid(row=3, column=0, sticky="ew", pady=(4, 0))
    btns.columnconfigure(2, weight=1)

    def _copy() -> None:
        try:
            dlg.clipboard_clear()
            dlg.clipboard_append(clean)
            dlg.update_idletasks()
        except Exception:
            pass

    ttk.Button(btns, text="Copy details", command=_copy).grid(row=0, column=0, sticky="w")
    if log_path:
        def _open_log() -> None:
            try:
                import os, subprocess, sys
                if sys.platform.startswith("win"):
                    os.startfile(str(log_path))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(log_path)])
                else:
                    subprocess.Popen(["xdg-open", str(log_path)])
            except Exception:
                pass
        ttk.Button(btns, text="Open log", command=_open_log).grid(row=0, column=1, sticky="w", padx=(8, 0))
    ttk.Button(btns, text="OK", command=dlg.destroy).grid(row=0, column=3, sticky="e")
