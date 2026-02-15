"""Lightweight tooltip helpers for Tk widgets."""

from __future__ import annotations

import tkinter as tk
import weakref
from typing import Dict, Optional


class _TooltipController:
    def __init__(self, widget: tk.Misc, text: str, delay_ms: int = 500):
        self.widget = widget
        self.text = str(text or "")
        self.delay_ms = max(0, int(delay_ms))
        self._after_id: Optional[str] = None
        self._tip_window: Optional[tk.Toplevel] = None
        self._label: Optional[tk.Label] = None
        self._bind_ids: Dict[str, str] = {}

        self._bind_ids["<Enter>"] = widget.bind("<Enter>", self._on_enter, add="+")
        self._bind_ids["<Leave>"] = widget.bind("<Leave>", self._on_leave, add="+")
        self._bind_ids["<ButtonPress>"] = widget.bind("<ButtonPress>", self._on_leave, add="+")
        self._bind_ids["<FocusOut>"] = widget.bind("<FocusOut>", self._on_leave, add="+")
        self._bind_ids["<Destroy>"] = widget.bind("<Destroy>", self._on_destroy, add="+")

    def update_text(self, text: str) -> None:
        self.text = str(text or "")
        if self._label is not None:
            self._label.configure(text=self.text)

    def detach(self) -> None:
        self._cancel()
        self._hide()
        for seq, bind_id in list(self._bind_ids.items()):
            try:
                self.widget.unbind(seq, bind_id)
            except Exception:
                pass
        self._bind_ids.clear()

    def _on_enter(self, _event=None) -> None:
        self._cancel()
        if not self.text.strip():
            return
        try:
            self._after_id = self.widget.after(self.delay_ms, self._show)
        except Exception:
            self._after_id = None

    def _on_leave(self, _event=None) -> None:
        self._cancel()
        self._hide()

    def _on_destroy(self, _event=None) -> None:
        self.detach()

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip_window is not None:
            return
        try:
            root = self.widget.winfo_toplevel()
            tw = tk.Toplevel(root)
        except Exception:
            return
        tw.wm_overrideredirect(True)
        try:
            tw.attributes("-topmost", True)
        except Exception:
            pass

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            anchor="w",
            relief=tk.SOLID,
            borderwidth=1,
            background="#fffce8",
            foreground="#222222",
            padx=8,
            pady=4,
            font=("TkDefaultFont", 9),
        )
        label.pack()
        self._tip_window = tw
        self._label = label

        self._position_tip()

    def _position_tip(self) -> None:
        if self._tip_window is None:
            return
        tw = self._tip_window
        try:
            pointer_x = self.widget.winfo_pointerx()
            pointer_y = self.widget.winfo_pointery()
            x = pointer_x + 14
            y = pointer_y + 18
        except Exception:
            x = self.widget.winfo_rootx() + self.widget.winfo_width() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8

        tw.update_idletasks()
        w = tw.winfo_reqwidth()
        h = tw.winfo_reqheight()

        screen_w = tw.winfo_screenwidth()
        screen_h = tw.winfo_screenheight()
        pad = 6
        x = max(pad, min(int(x), int(screen_w - w - pad)))
        y = max(pad, min(int(y), int(screen_h - h - pad)))

        tw.geometry(f"+{x}+{y}")

    def _hide(self) -> None:
        if self._tip_window is not None:
            try:
                self._tip_window.destroy()
            except Exception:
                pass
        self._tip_window = None
        self._label = None


_REGISTRY: "weakref.WeakKeyDictionary[tk.Misc, _TooltipController]" = weakref.WeakKeyDictionary()


def attach_tooltip(widget: tk.Misc, text: str, delay_ms: int = 500) -> None:
    """Attach or update a tooltip for a widget."""
    if widget is None:
        return
    ctrl = _REGISTRY.get(widget)
    if ctrl is None:
        _REGISTRY[widget] = _TooltipController(widget, text=text, delay_ms=delay_ms)
        return
    ctrl.delay_ms = max(0, int(delay_ms))
    ctrl.update_text(text)


def detach_tooltip(widget: tk.Misc) -> None:
    """Detach tooltip and cleanup bindings/window."""
    ctrl = _REGISTRY.pop(widget, None)
    if ctrl is not None:
        ctrl.detach()
