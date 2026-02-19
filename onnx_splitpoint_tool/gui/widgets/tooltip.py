"""Lightweight tooltip helper for Tkinter/ttk widgets.

This module intentionally avoids external dependencies (e.g. ttkbootstrap) and
works with plain Tkinter.

Usage:
    from ..widgets.tooltip import attach_tooltip
    attach_tooltip(some_widget, "Hello")
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional


def attach_tooltip(
    widget: tk.Widget,
    text: str,
    *,
    delay_ms: int = 500,
    wraplength: int = 360,
) -> None:
    """Attach a hover tooltip to *widget*.

    Parameters
    ----------
    widget:
        Tk widget to attach tooltip to.
    text:
        Tooltip text.
    delay_ms:
        Delay before showing tooltip.
    wraplength:
        Pixel wrap length for tooltip label.

    Notes
    -----
    - Tooltip is destroyed on <Leave>, <ButtonPress>, and <Destroy>.
    - Safe against widgets being destroyed while the tooltip is scheduled.
    """

    if not text:
        return

    state: dict[str, Optional[object]] = {"after": None, "tip": None}

    def _destroy_tip() -> None:
        tip = state.get("tip")
        if tip is not None:
            try:
                if isinstance(tip, tk.Toplevel) and tip.winfo_exists():
                    tip.destroy()
            except Exception:
                pass
        state["tip"] = None

    def _cancel_after() -> None:
        after_id = state.get("after")
        if after_id is not None:
            try:
                widget.after_cancel(after_id)  # type: ignore[arg-type]
            except Exception:
                pass
        state["after"] = None

    def _hide(_event=None) -> None:
        _cancel_after()
        _destroy_tip()

    def _show() -> None:
        # If widget is gone, do nothing.
        try:
            if not widget.winfo_exists():
                return
        except Exception:
            return

        # Avoid duplicates.
        _destroy_tip()

        # Position below the widget.
        try:
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 8
        except Exception:
            return

        tip = tk.Toplevel(widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")

        # Use a plain tk.Label so background/foreground are guaranteed.
        lbl = tk.Label(
            tip,
            text=text,
            justify=tk.LEFT,
            background="#ffffe0",
            foreground="#000000",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=wraplength,
        )
        lbl.pack(ipadx=6, ipady=4)

        state["tip"] = tip

    def _schedule(_event=None) -> None:
        _cancel_after()
        _destroy_tip()
        try:
            state["after"] = widget.after(delay_ms, _show)
        except Exception:
            state["after"] = None

    # Bind with add="+" to not override existing handlers.
    widget.bind("<Enter>", _schedule, add="+")
    widget.bind("<Leave>", _hide, add="+")
    widget.bind("<ButtonPress>", _hide, add="+")
    widget.bind("<Destroy>", _hide, add="+")
