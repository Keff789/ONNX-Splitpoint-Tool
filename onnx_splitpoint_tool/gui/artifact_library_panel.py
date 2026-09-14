from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..artifact_store import ArtifactStore, default_artifact_store_root


class ArtifactLibraryPanel(ttk.Frame):
    """Tool-Config view for the unified HEF/DXNN artifact library."""

    def __init__(self, master, *, root_var: tk.StringVar | None = None, **kwargs):
        super().__init__(master, **kwargs)
        self.root_var = root_var or tk.StringVar(value=str(default_artifact_store_root()))
        self.status_var = tk.StringVar(value="Not loaded")
        self._build()
        self.after_idle(self.refresh)

    def _build(self) -> None:
        top = ttk.LabelFrame(self, text="Unified compiler artifact library")
        top.pack(fill="x", padx=8, pady=8)
        ttk.Label(top, text="Library root:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(top, textvariable=self.root_var).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(top, text="Folder…", command=self._choose_root).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Open", command=self._open_root).grid(row=0, column=3, padx=4)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, textvariable=self.status_var).grid(row=1, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 5))

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", padx=8, pady=(0, 6))
        for label, cmd in (
            ("Refresh", self.refresh),
            ("Index existing caches", self._index_existing),
            ("Verify metadata", lambda: self._verify(False)),
            ("Verify SHA-256", lambda: self._verify(True)),
            ("Pin selected", self._pin_selected),
            ("Unpin selected", self._unpin_selected),
            ("Export pinned…", self._export_pinned),
            ("Import pack…", self._import_pack),
            ("Prune preview", self._prune_preview),
        ):
            ttk.Button(buttons, text=label, command=cmd).pack(side="left", padx=3)

        cols = ("id", "kind", "size", "pinned", "status", "contract", "source")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="extended")
        headings = {"id":"ID","kind":"Kind","size":"Size","pinned":"Pinned","status":"Verification","contract":"Contract","source":"Source run"}
        widths = {"id":60,"kind":125,"size":100,"pinned":70,"status":140,"contract":210,"source":170}
        for col in cols:
            self.tree.heading(col, text=headings[col]); self.tree.column(col, width=widths[col], stretch=col in {"contract","source"})
        self.tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _store(self) -> ArtifactStore:
        return ArtifactStore(Path(self.root_var.get()).expanduser())

    def _choose_root(self) -> None:
        path = filedialog.askdirectory(initialdir=str(Path(self.root_var.get()).expanduser()))
        if path:
            self.root_var.set(path); self.refresh()

    def _open_root(self) -> None:
        path = Path(self.root_var.get()).expanduser(); path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("linux"): subprocess.Popen(["xdg-open", str(path)])
            elif sys.platform == "darwin": subprocess.Popen(["open", str(path)])
            elif os.name == "nt": os.startfile(str(path))
        except Exception as exc: messagebox.showerror("Artifact library", str(exc), parent=self)

    @staticmethod
    def _size(value: int) -> str:
        n=float(value)
        for unit in ("B","KiB","MiB","GiB","TiB"):
            if n < 1024 or unit == "TiB": return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TiB"

    def refresh(self) -> None:
        try:
            store=self._store(); stats=store.stats(); records=store.list(limit=5000)
            self.status_var.set(f"{stats['artifact_count']} artifacts · {self._size(stats['logical_bytes'])} · {stats['pinned_count']} pinned · {store.root}")
            for item in self.tree.get_children(): self.tree.delete(item)
            for r in records:
                self.tree.insert("", "end", iid=str(r.artifact_id), values=(r.artifact_id,r.kind,self._size(r.size_bytes),"yes" if r.pinned else "",r.verification_status,r.contract_hash[:18],r.source_run))
        except Exception as exc:
            self.status_var.set(f"ERROR: {exc}")

    def _index_existing(self) -> None:
        try:
            result = self._store().index_existing()
            messagebox.showinfo("Artifact indexing", f"Indexed {result['indexed_count']} existing HEF/DXNN artifacts.\nFailures: {result['failed_count']}", parent=self)
            self.refresh()
        except Exception as exc:
            messagebox.showerror("Artifact indexing", str(exc), parent=self)

    def _verify(self, strict: bool) -> None:
        try:
            result=self._store().verify(strict=strict, quarantine=False)
            messagebox.showinfo("Artifact verification", f"Checked {result['count']} artifacts.\nStatus: {'OK' if result['ok'] else 'problems found'}", parent=self)
            self.refresh()
        except Exception as exc: messagebox.showerror("Artifact verification", str(exc), parent=self)

    def _selected_ids(self) -> list[int]:
        return [int(x) for x in self.tree.selection()]

    def _pin_selected(self) -> None:
        store=self._store()
        for artifact_id in self._selected_ids(): store.pin(artifact_id, label="manual")
        self.refresh()

    def _unpin_selected(self) -> None:
        store=self._store()
        for artifact_id in self._selected_ids(): store.unpin(artifact_id)
        self.refresh()

    def _export_pinned(self) -> None:
        path=filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Artifact packs","*.zip")])
        if not path: return
        try:
            result=self._store().export_pack(path,pinned_only=True)
            messagebox.showinfo("Artifact export", f"Written:\n{result}", parent=self)
        except Exception as exc: messagebox.showerror("Artifact export", str(exc), parent=self)

    def _import_pack(self) -> None:
        path=filedialog.askopenfilename(filetypes=[("Artifact packs","*.zip"),("All files","*")])
        if not path: return
        try:
            result=self._store().import_pack(path)
            messagebox.showinfo("Artifact import", f"Imported {result['imported_count']} artifacts.", parent=self); self.refresh()
        except Exception as exc: messagebox.showerror("Artifact import", str(exc), parent=self)

    def _prune_preview(self) -> None:
        try:
            result=self._store().prune(older_than_days=30,dry_run=True)
            messagebox.showinfo("Prune preview", f"Unpinned candidates older than 30 days: {result['candidate_count']}\nSize: {self._size(result['candidate_bytes'])}\n\nUse the CLI with --apply after reviewing the list.", parent=self)
        except Exception as exc: messagebox.showerror("Prune preview", str(exc), parent=self)
