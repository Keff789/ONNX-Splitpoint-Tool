from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Optional

from ...log_utils import sanitize_log


class TextProgressDialog:
    """Reusable text-based progress dialog with log output and copy/cancel controls.

    The dialog itself stays GUI-only, while long-running work should live in services
    or worker functions that call :meth:`append` and :meth:`set_progress` via the Tk
    main loop.
    """

    def __init__(
        self,
        root: tk.Misc,
        *,
        title: str,
        initial_status: str = "Starting…",
        initial_lines: Optional[Iterable[str]] = None,
        progress_mode: str = "determinate",
        progress_maximum: float = 100.0,
        on_cancel: Optional[Callable[[], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
        geometry: str = "760x460",
    ) -> None:
        self.root = root
        self._on_cancel = on_cancel
        self._on_close = on_close
        self._alive = True
        self._finished = False

        dlg = tk.Toplevel(root)
        dlg.title(title)
        dlg.geometry(geometry)
        try:
            dlg.transient(root)
        except Exception:
            pass
        self.window = dlg

        self.status_var = tk.StringVar(value=initial_status)
        ttk.Label(dlg, textvariable=self.status_var).pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))

        pb = ttk.Progressbar(dlg, orient="horizontal", length=400, mode=progress_mode, maximum=progress_maximum)
        pb.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)
        self.progressbar = pb

        txt = tk.Text(dlg, height=18, wrap="word")
        txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.text = txt
        for line in list(initial_lines or []):
            txt.insert("end", str(line).rstrip("\n") + "\n")
        txt.see("end")

        btn_row = ttk.Frame(dlg)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        self.btn_cancel = ttk.Button(btn_row, text="Cancel", command=self._cancel)
        self.btn_cancel.pack(side=tk.LEFT)
        self.btn_copy = ttk.Button(btn_row, text="Copy log", command=self.copy_log)
        self.btn_copy.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_close = ttk.Button(btn_row, text="Close", command=self._close)
        self.btn_close.pack(side=tk.RIGHT)

        try:
            dlg.protocol("WM_DELETE_WINDOW", self._close)
        except Exception:
            pass

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def finished(self) -> bool:
        return self._finished

    def append(self, line: str) -> None:
        if not self._alive:
            return
        try:
            self.text.insert("end", str(line).rstrip("\n") + "\n")
            self.text.see("end")
        except tk.TclError:
            self._alive = False
        except Exception:
            pass

    def set_status(self, text: str) -> None:
        if not self._alive:
            return
        try:
            self.status_var.set(str(text))
        except Exception:
            pass

    def set_progress(self, fraction: float, label: Optional[str] = None) -> None:
        if not self._alive:
            return
        if label is not None:
            self.set_status(label)
        try:
            maxv = float(self.progressbar.cget("maximum") or 100.0)
            self.progressbar["value"] = max(0.0, min(maxv, float(fraction) * maxv))
        except tk.TclError:
            self._alive = False
        except Exception:
            pass

    def set_absolute_progress(self, value: float, label: Optional[str] = None) -> None:
        if not self._alive:
            return
        if label is not None:
            self.set_status(label)
        try:
            maxv = float(self.progressbar.cget("maximum") or 100.0)
            self.progressbar["value"] = max(0.0, min(maxv, float(value)))
        except tk.TclError:
            self._alive = False
        except Exception:
            pass

    def finish(self, *, status_text: Optional[str] = None) -> None:
        self._finished = True
        if status_text:
            self.set_status(status_text)
        try:
            self.btn_cancel.configure(state="disabled")
        except Exception:
            pass
        try:
            self.btn_close.configure(state="normal")
        except Exception:
            pass

    def copy_log(self) -> None:
        if not self._alive:
            return
        try:
            text = sanitize_log(self.text.get("1.0", "end-1c"))
            self.window.clipboard_clear()
            self.window.clipboard_append(text)
            self.window.update_idletasks()
            self.append("[ui] Copied log to clipboard")
        except Exception as exc:
            self.append(f"[ui] Copy log failed: {exc}")

    def _cancel(self) -> None:
        if callable(self._on_cancel):
            try:
                self._on_cancel()
            except Exception:
                pass

    def _close(self) -> None:
        self._alive = False
        if callable(self._on_close):
            try:
                self._on_close()
            except Exception:
                pass
        try:
            self.window.destroy()
        except Exception:
            pass
