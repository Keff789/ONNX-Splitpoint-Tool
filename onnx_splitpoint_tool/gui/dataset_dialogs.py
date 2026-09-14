"""GUI helpers for final dataset provisioning and registry binding.

The dialog deliberately keeps dataset access policy separate from benchmark
execution.  Public COCO archives can be provisioned automatically.  ImageNet
remains access-gated: the GUI can install/check the Kaggle CLI and import an
authorised validation archive, but it never embeds credentials or bypasses the
dataset terms.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from ..dataset_provisioning import (
    COCO_DOWNLOAD_POLICIES,
    IMAGENET_EXPORT_KERNEL_SLUG,
    IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG,
    IMAGENET_KAGGLE_DOWNLOAD_MODES,
    IMAGENET_KAGGLE_RULES_URL,
    collect_imagenet_kernel_diagnostics,
    default_dataset_root,
    default_registry_path,
    import_imagenet,
    install_optional_dataset_dependencies,
    kaggle_cli_status,
    load_registry,
    provision_coco2017,
    provision_imagenet_complete,
    provision_imagenet_kaggle,
    register_coco2017,
    register_imagenet,
    registry_status,
    repair_dataset_registry,
    resolve_kaggle_username,
)
from ..validation.official_coco import pycocotools_status


def _set_app_var(app: Any, name: str, value: Any) -> None:
    if app is None:
        return
    var = getattr(app, name, None)
    if var is None:
        if isinstance(value, bool):
            var = tk.BooleanVar(app, value=value)
        else:
            var = tk.StringVar(app, value=str(value or ""))
        setattr(app, name, var)
        return
    try:
        var.set(value)
    except Exception:
        pass


def bind_registry_to_app(app: Any, registry_path: str | Path | None = None) -> dict[str, Any]:
    """Bind canonical dataset/manifest paths from a registry to the main GUI."""

    registry = load_registry(registry_path)
    datasets = dict(registry.get("datasets") or {})
    manifests = dict(registry.get("manifests") or {})
    mapping = {
        "var_final_dataset_registry": str(registry_path or default_registry_path()),
        "var_final_dataset_root": str(registry.get("root") or default_dataset_root()),
        "var_final_imagenet_train": str(
            (datasets.get("imagenet_calibration") or {}).get("root") or ""
        ),
        "var_final_imagenet_val": str(
            (datasets.get("imagenet_validation") or {}).get("root") or ""
        ),
        "var_final_imagenet_labels": str(
            (datasets.get("imagenet_validation") or {}).get("labels")
            or (datasets.get("imagenet_calibration") or {}).get("labels")
            or ""
        ),
        "var_final_coco_train": str(
            (datasets.get("coco2017_calibration") or {}).get("root") or ""
        ),
        "var_final_coco_val": str(
            (datasets.get("coco2017_validation") or {}).get("root") or ""
        ),
        "var_final_coco_annotations": str(
            (datasets.get("coco2017_validation") or {}).get("annotations") or ""
        ),
        "var_manifest_cls_calibration": str(manifests.get("classification_calibration") or ""),
        "var_manifest_cls_validation": str(manifests.get("classification_validation") or ""),
        "var_manifest_det_calibration": str(manifests.get("detection_calibration") or ""),
        "var_manifest_det_validation": str(manifests.get("detection_validation") or ""),
    }
    for name, value in mapping.items():
        _set_app_var(app, name, value)

    if mapping["var_final_coco_annotations"]:
        current = getattr(app, "var_official_coco_annotations", None)
        current_value = ""
        try:
            current_value = str(current.get()) if current is not None else ""
        except Exception:
            pass
        if not current_value.strip():
            _set_app_var(app, "var_official_coco_annotations", mapping["var_final_coco_annotations"])
    return registry


def format_registry_status_line(payload: Mapping[str, Any]) -> str:
    tasks = dict(payload.get("task_readiness") or {})
    cls = dict(tasks.get("classification") or {})
    det = dict(tasks.get("detection") or {})
    cls_text = "ready" if bool(cls.get("ready")) else "incomplete"
    det_text = "ready" if bool(det.get("ready")) else "incomplete"
    global_text = "ready" if bool(payload.get("ready_for_final_profile")) else "incomplete"
    return f"ImageNet: {cls_text} · COCO: {det_text} · global registry: {global_text}"


def _open_path(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


class DatasetProvisioningDialog(tk.Toplevel):
    """Provision and register final campaign datasets without blocking Tk."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        app: Any = None,
        on_updated: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(master)
        self.title("Final datasets & manifests")
        self.geometry("1220x860")
        self.minsize(1020, 720)
        self.transient(master.winfo_toplevel())
        self.app = app
        self.on_updated = on_updated
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._running = False

        default_root_value = str(default_dataset_root())
        default_registry_value = str(default_registry_path())
        try:
            default_root_value = str(app.var_final_dataset_root.get())
        except Exception:
            pass
        try:
            default_registry_value = str(app.var_final_dataset_registry.get())
        except Exception:
            pass

        self.var_root = tk.StringVar(self, value=default_root_value)
        self.var_registry = tk.StringVar(self, value=default_registry_value)
        try:
            initial_registry = load_registry(self.var_registry.get() or None)
            initial_settings = dict(initial_registry.get("settings") or {})
        except Exception:
            initial_registry = {"datasets": {}, "manifests": {}, "settings": {}}
            initial_settings = {}
        initial_datasets = dict(initial_registry.get("datasets") or {})
        initial_manifests = dict(initial_registry.get("manifests") or {})

        # Complete final provisioning is the safe default when the corresponding
        # calibration role is still absent.  Users can explicitly choose
        # validation-only to avoid the additional train download/export.
        coco_calibration_missing = not bool(
            initial_manifests.get("detection_calibration")
            and (initial_datasets.get("coco2017_calibration") or {}).get("root")
        )
        self.var_coco_train = tk.BooleanVar(
            self,
            value=(
                True
                if coco_calibration_missing
                else bool(initial_settings.get("coco_include_train", False))
            ),
        )
        self.var_coco_download_policy = tk.StringVar(
            self, value=str(initial_settings.get("coco_download_policy") or "auto")
        )
        self.var_coco_mirror = tk.StringVar(
            self, value=str(initial_settings.get("coco_mirror_base") or "")
        )
        self.var_coco_calib = tk.IntVar(
            self, value=int(initial_settings.get("detection_calibration_items") or 1000)
        )
        self.var_coco_retain_train_archive = tk.BooleanVar(
            self, value=bool(initial_settings.get("coco_retain_train_archive", False))
        )

        imagenet_calibration_missing = not bool(
            initial_manifests.get("classification_calibration")
            and (initial_datasets.get("imagenet_calibration") or {}).get("root")
        )
        self.var_imagenet_include_calibration = tk.BooleanVar(
            self,
            value=(
                True
                if imagenet_calibration_missing
                else bool(initial_settings.get("imagenet_include_calibration", False))
            ),
        )
        self.var_imagenet_calib = tk.IntVar(
            self, value=int(initial_settings.get("classification_calibration_items") or 1000)
        )
        self.var_imagenet_terms = tk.BooleanVar(self, value=False)
        self.var_imagenet_download_mode = tk.StringVar(
            self, value=str(initial_settings.get("imagenet_download_mode") or "validation_only")
        )
        initial_username = str(initial_settings.get("kaggle_username") or "").strip()
        if not initial_username:
            try:
                initial_username, _ = resolve_kaggle_username()
            except Exception:
                initial_username = ""
        self.var_kaggle_username = tk.StringVar(self, value=initial_username)
        stored_kernel_ref = str(
            initial_settings.get("imagenet_validation_export_kernel_ref") or ""
        ).strip()
        stored_slug = stored_kernel_ref.split("/", 1)[1] if "/" in stored_kernel_ref else ""
        self.var_kaggle_kernel_slug = tk.StringVar(
            self, value=stored_slug or IMAGENET_EXPORT_KERNEL_SLUG
        )
        stored_calibration_kernel_ref = str(
            initial_settings.get("imagenet_calibration_export_kernel_ref") or ""
        ).strip()
        stored_calibration_slug = (
            stored_calibration_kernel_ref.split("/", 1)[1]
            if "/" in stored_calibration_kernel_ref
            else ""
        )
        self.var_kaggle_calibration_kernel_slug = tk.StringVar(
            self,
            value=stored_calibration_slug
            or IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG,
        )
        self.var_imagenet_source = tk.StringVar(self, value="")
        self.var_imagenet_solution = tk.StringVar(self, value="")
        self.var_imagenet_train = tk.StringVar(self, value="")
        self.var_imagenet_val = tk.StringVar(self, value="")
        self.var_imagenet_labels = tk.StringVar(self, value="")

        self.var_coco_val = tk.StringVar(self, value="")
        self.var_coco_train_root = tk.StringVar(self, value="")
        self.var_coco_annotations = tk.StringVar(self, value="")
        self.var_status = tk.StringVar(self, value="Ready")

        self._build()
        self.after(100, self._poll)
        self._refresh_status()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        storage = ttk.LabelFrame(self, text="Storage and registry")
        storage.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        storage.columnconfigure(1, weight=1)
        ttk.Label(storage, text="Dataset root:").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=6
        )
        ttk.Entry(storage, textvariable=self.var_root).grid(
            row=0, column=1, sticky="ew", pady=6
        )
        ttk.Button(
            storage, text="Folder…", command=lambda: self._choose_dir(self.var_root)
        ).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(
            storage,
            text="Open",
            command=lambda: _open_path(Path(self.var_root.get()).expanduser()),
        ).grid(row=0, column=3, padx=(0, 8), pady=6)
        ttk.Label(storage, text="Registry:").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 6)
        )
        ttk.Entry(storage, textvariable=self.var_registry).grid(
            row=1, column=1, sticky="ew", pady=(0, 6)
        )
        ttk.Button(
            storage,
            text="JSON…",
            command=lambda: self._choose_file(self.var_registry, [("JSON", "*.json")]),
        ).grid(row=1, column=2, padx=6, pady=(0, 6))
        ttk.Button(storage, text="Verify", command=self._refresh_status).grid(
            row=1, column=3, padx=(0, 8), pady=(0, 6)
        )

        ttk.Button(
            storage,
            text="Repair/reindex existing assets (no download)",
            command=self._repair_registry,
        ).grid(row=2, column=1, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(
            storage,
            text=(
                "Recovers registry entries from verified manifests and materialised subsets; "
                "train2017.zip is not required."
            ),
            foreground="#555",
        ).grid(row=3, column=1, columnspan=3, sticky="w", pady=(0, 6))

        notebook = ttk.Notebook(self)
        self.notebook = notebook
        notebook.grid(row=1, column=0, sticky="ew", padx=10, pady=6)
        coco = ttk.Frame(notebook)
        imagenet = ttk.Frame(notebook)
        existing = ttk.Frame(notebook)
        status = ttk.Frame(notebook)
        self.existing_tab = existing
        for tab in (coco, imagenet, existing, status):
            tab.columnconfigure(1, weight=1)
        notebook.add(coco, text="COCO 2017")
        notebook.add(imagenet, text="ImageNet validation / Kaggle")
        notebook.add(existing, text="Register/import existing")
        notebook.add(status, text="Status & dependencies")

        self._build_coco_tab(coco)
        self._build_imagenet_tab(imagenet)
        self._build_existing_tab(existing)
        self._build_status_tab(status)

        log_box = ttk.LabelFrame(self, text="Provisioning log")
        log_box.grid(row=2, column=0, sticky="nsew", padx=10, pady=6)
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_box, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        footer = ttk.Frame(self)
        footer.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.var_status).grid(row=0, column=0, sticky="w")
        self.close_btn = ttk.Button(footer, text="Close", command=self.destroy)
        self.close_btn.grid(row=0, column=1, sticky="e")

    def _build_coco_tab(self, tab: ttk.Frame) -> None:
        ttk.Label(
            tab,
            text=(
                "One action can provision both final roles: val2017 + official annotations "
                "for the AP gate, and a deterministic train-derived calibration subset. "
                "When calibration is selected, an already verified materialised subset is reused first. "
                "Only when it is absent or invalid is the ~18 GB train2017 archive downloaded; "
                "then only the requested images are materialised, a filtered annotation "
                "file and calibration manifest are created, and the large archive is removed "
                "unless you explicitly retain it. TLS verification is never disabled."
            ),
            foreground="#555",
            wraplength=1060,
        ).grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(8, 4))
        ttk.Checkbutton(
            tab,
            text=(
                "Provision disjoint train-derived calibration subset "
                "(required for final profile; temporary ~18 GB download)"
            ),
            variable=self.var_coco_train,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=6)
        ttk.Label(tab, text="Calibration items:").grid(
            row=1, column=2, sticky="e", padx=(8, 6)
        )
        ttk.Spinbox(
            tab, from_=1, to=118287, textvariable=self.var_coco_calib, width=10
        ).grid(row=1, column=3, sticky="w", padx=(0, 8))
        ttk.Checkbutton(
            tab,
            text="Keep the verified train2017 ZIP after subset creation",
            variable=self.var_coco_retain_train_archive,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Label(
            tab,
            text="Default: only the selected calibration images remain locally.",
            foreground="#555",
        ).grid(row=2, column=2, columnspan=2, sticky="w", padx=6, pady=4)
        ttk.Label(tab, text="Download policy:").grid(
            row=3, column=0, sticky="w", padx=(8, 6), pady=4
        )
        ttk.Combobox(
            tab,
            textvariable=self.var_coco_download_policy,
            values=list(COCO_DOWNLOAD_POLICIES),
            state="readonly",
            width=18,
        ).grid(row=3, column=1, sticky="w", pady=4)
        ttk.Label(tab, text="Optional mirror base:").grid(
            row=4, column=0, sticky="w", padx=(8, 6), pady=4
        )
        ttk.Entry(tab, textvariable=self.var_coco_mirror).grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=4
        )
        ttk.Button(
            tab,
            text="Provision selected COCO validation + calibration + manifests",
            command=self._provision_coco,
        ).grid(row=5, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 4))
        ttk.Label(
            tab,
            text=(
                "After completion the tool config fields COCO val, COCO train/calib, "
                "DET validation manifest and DET calibration manifest are filled automatically."
            ),
            foreground="#555",
        ).grid(row=6, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))

    def _build_imagenet_tab(self, tab: ttk.Frame) -> None:
        ttk.Label(
            tab,
            text=(
                "One action can provision both final ImageNet roles without downloading the "
                "full training corpus. The validation export packages the official 50,000 "
                "validation images. A second private CPU kernel selects and packages only the "
                "requested deterministic, class-stratified train calibration subset. Existing "
                "verified validation/calibration assets are reused. No Kaggle token is stored "
                "in the tool configuration."
            ),
            foreground="#555",
            wraplength=1060,
        ).grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(8, 4))
        ttk.Checkbutton(
            tab,
            text="I joined the ImageNet competition and accepted its rules on Kaggle",
            variable=self.var_imagenet_terms,
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=6)
        ttk.Checkbutton(
            tab,
            text=(
                "Provision disjoint train-derived calibration subset in the same action "
                "(required for final profile)"
            ),
            variable=self.var_imagenet_include_calibration,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=4)
        ttk.Label(tab, text="Calibration items:").grid(
            row=2, column=2, sticky="e", padx=(8, 6)
        )
        ttk.Spinbox(
            tab, from_=1, to=1281167, textvariable=self.var_imagenet_calib, width=10
        ).grid(row=2, column=3, sticky="w", padx=(0, 8))
        ttk.Label(
            tab,
            text="Recommendation for the final campaign: 1,000 items (approximately one per class).",
            foreground="#555",
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 4))

        ttk.Label(tab, text="Kaggle download mode:").grid(
            row=4, column=0, sticky="w", padx=(8, 6), pady=4
        )
        ttk.Combobox(
            tab,
            textvariable=self.var_imagenet_download_mode,
            values=list(IMAGENET_KAGGLE_DOWNLOAD_MODES),
            state="readonly",
            width=32,
        ).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Button(
            tab,
            text="Install/repair dataset support",
            command=self._install_dependencies,
        ).grid(row=4, column=2, sticky="e", padx=6, pady=4)
        ttk.Button(
            tab,
            text="Check Kaggle auth/rules/access",
            command=self._check_kaggle_setup,
        ).grid(row=4, column=3, sticky="w", padx=(0, 8), pady=4)

        ttk.Label(tab, text="Public Kaggle username:").grid(
            row=5, column=0, sticky="w", padx=(8, 6), pady=4
        )
        ttk.Entry(tab, textvariable=self.var_kaggle_username).grid(
            row=5, column=1, sticky="ew", pady=4
        )
        ttk.Button(
            tab,
            text="Detect",
            command=self._detect_kaggle_username,
        ).grid(row=5, column=2, sticky="e", padx=6, pady=4)
        ttk.Label(
            tab,
            text="Used only as owner of the private kernels; it is not a secret.",
            foreground="#555",
        ).grid(row=5, column=3, sticky="w", padx=(0, 8), pady=4)

        ttk.Label(tab, text="Validation export kernel slug:").grid(
            row=6, column=0, sticky="w", padx=(8, 6), pady=4
        )
        ttk.Entry(tab, textvariable=self.var_kaggle_kernel_slug).grid(
            row=6, column=1, sticky="ew", pady=4
        )
        ttk.Label(tab, text="Calibration export kernel slug:").grid(
            row=6, column=2, sticky="e", padx=(8, 6), pady=4
        )
        ttk.Entry(tab, textvariable=self.var_kaggle_calibration_kernel_slug).grid(
            row=6, column=3, sticky="ew", padx=(0, 8), pady=4
        )

        ttk.Button(
            tab,
            text="Provision selected ImageNet validation + calibration + manifests",
            command=self._provision_imagenet,
        ).grid(row=7, column=0, columnspan=3, sticky="w", padx=8, pady=4)
        ttk.Button(
            tab,
            text="Import existing validation/archive…",
            command=lambda: self.notebook.select(self.existing_tab),
        ).grid(row=7, column=3, sticky="e", padx=8, pady=4)
        ttk.Label(
            tab,
            text=(
                "After completion the tool config fields ImageNet val, ImageNet train/calib, "
                "CLS validation manifest and CLS calibration manifest are filled automatically."
            ),
            foreground="#555",
        ).grid(row=8, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 4))
        ttk.Button(
            tab,
            text="Join / accept ImageNet competition rules",
            command=lambda: webbrowser.open(IMAGENET_KAGGLE_RULES_URL),
        ).grid(row=9, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        ttk.Button(
            tab,
            text="Open Kaggle API credentials",
            command=lambda: webbrowser.open("https://www.kaggle.com/settings/api"),
        ).grid(row=9, column=2, columnspan=2, sticky="e", padx=8, pady=(0, 4))
        ttk.Button(
            tab,
            text="Download latest validation-export diagnostics",
            command=self._collect_kaggle_export_diagnostics,
        ).grid(row=10, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        ttk.Label(
            tab,
            text="Kernel errors are also downloaded automatically after a failed provision run.",
            foreground="#555",
        ).grid(row=10, column=2, columnspan=2, sticky="e", padx=8, pady=(0, 8))

    def _build_existing_tab(self, tab: ttk.Frame) -> None:
        ttk.Label(
            tab,
            text=(
                "Import an authorised ImageNet archive/root or register already installed, "
                "class-organised ImageNet and COCO directories."
            ),
            foreground="#555",
            wraplength=1060,
        ).grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(8, 4))

        ttk.Label(tab, text="ImageNet source/archive:").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=4
        )
        ttk.Entry(tab, textvariable=self.var_imagenet_source).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        ttk.Button(tab, text="Select…", command=self._choose_imagenet_source).grid(
            row=1, column=2, padx=6, pady=4
        )
        ttk.Button(tab, text="Import", command=self._import_imagenet).grid(
            row=1, column=3, padx=(0, 8), pady=4
        )

        ttk.Label(tab, text="Validation solution CSV:").grid(
            row=2, column=0, sticky="w", padx=(8, 6), pady=4
        )
        ttk.Entry(tab, textvariable=self.var_imagenet_solution).grid(
            row=2, column=1, sticky="ew", pady=4
        )
        ttk.Button(
            tab,
            text="Select…",
            command=lambda: self._choose_file(
                self.var_imagenet_solution, [("CSV", "*.csv"), ("All", "*")]
            ),
        ).grid(row=2, column=2, padx=6, pady=4)
        ttk.Label(
            tab,
            text=(
                "For a raw ILSVRC2012_img_val.tar, provide LOC_val_solution.csv so the "
                "50,000 flat images can be assigned to the 1,000 classes."
            ),
            foreground="#555",
            wraplength=920,
        ).grid(row=3, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 4))

        imagenet_rows = [
            (4, "ImageNet train/calibration:", self.var_imagenet_train, False),
            (5, "ImageNet val:", self.var_imagenet_val, False),
            (6, "Synset labels:", self.var_imagenet_labels, True),
        ]
        for row, label, var, is_file in imagenet_rows:
            ttk.Label(tab, text=label).grid(
                row=row, column=0, sticky="w", padx=(8, 6), pady=4
            )
            ttk.Entry(tab, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
            # Avoid the conditional-lambda binding trap by assigning explicitly.
            if is_file:
                button_command = lambda v=var: self._choose_file(
                    v, [("Text", "*.txt"), ("All", "*")]
                )
            else:
                button_command = lambda v=var: self._choose_dir(v)
            ttk.Button(tab, text="Select…", command=button_command).grid(
                row=row, column=2, padx=6, pady=4
            )
        ttk.Button(
            tab,
            text="Register ImageNet + manifests",
            command=self._register_imagenet,
        ).grid(row=6, column=3, padx=(0, 8), pady=4)

        coco_rows = [
            (7, "COCO train2017:", self.var_coco_train_root),
            (8, "COCO val2017:", self.var_coco_val),
            (9, "COCO annotations:", self.var_coco_annotations),
        ]
        for row, label, var in coco_rows:
            ttk.Label(tab, text=label).grid(
                row=row, column=0, sticky="w", padx=(8, 6), pady=4
            )
            ttk.Entry(tab, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
            if row == 9:
                button_command = lambda v=var: self._choose_coco_annotations(v)
            else:
                button_command = lambda v=var: self._choose_dir(v)
            ttk.Button(tab, text="Select…", command=button_command).grid(
                row=row, column=2, padx=6, pady=4
            )
        ttk.Button(
            tab,
            text="Register COCO + manifests",
            command=self._register_coco,
        ).grid(row=9, column=3, padx=(0, 8), pady=(4, 8))

    def _build_status_tab(self, tab: ttk.Frame) -> None:
        self.status_text = scrolledtext.ScrolledText(tab, height=12, wrap="word")
        self.status_text.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=8, pady=8)
        tab.rowconfigure(0, weight=1)
        ttk.Button(tab, text="Refresh", command=self._refresh_status).grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 8)
        )
        ttk.Button(
            tab,
            text="Install/repair dataset support",
            command=self._install_dependencies,
        ).grid(row=1, column=1, sticky="w", padx=6, pady=(0, 8))
        ttk.Button(
            tab,
            text="Check Kaggle auth/rules/access",
            command=self._check_kaggle_setup,
        ).grid(row=1, column=2, sticky="w", padx=6, pady=(0, 8))

    def _choose_dir(self, var: tk.StringVar) -> None:
        initial = str(Path(var.get() or self.var_root.get()).expanduser())
        value = filedialog.askdirectory(parent=self, initialdir=initial)
        if value:
            var.set(value)

    def _choose_file(self, var: tk.StringVar, types: list[tuple[str, str]]) -> None:
        value = filedialog.askopenfilename(parent=self, filetypes=types)
        if value:
            var.set(value)

    def _choose_imagenet_source(self) -> None:
        if messagebox.askyesno(
            "ImageNet source",
            "Select a directory? Choose No to select an archive.",
            parent=self,
        ):
            self._choose_dir(self.var_imagenet_source)
        else:
            self._choose_file(
                self.var_imagenet_source,
                [("Archives", "*.zip *.tar *.tgz *.tar.gz"), ("All", "*")],
            )

    def _choose_coco_annotations(self, var: tk.StringVar) -> None:
        if messagebox.askyesno(
            "COCO annotations",
            "Select the annotations directory? Choose No to select instances_val2017.json directly.",
            parent=self,
        ):
            self._choose_dir(var)
        else:
            self._choose_file(var, [("COCO JSON", "*.json"), ("All", "*")])

    def _append_log(self, line: str) -> None:
        self.log_text.insert("end", str(line).rstrip() + "\n")
        self.log_text.see("end")

    def _run(
        self,
        title: str,
        fn: Callable[[Callable[[str], None]], Mapping[str, Any]],
    ) -> None:
        if self._running:
            messagebox.showwarning(
                "Datasets", "A provisioning job is already running.", parent=self
            )
            return
        self._running = True
        self.var_status.set(title)
        self.close_btn.configure(state="disabled")
        self._append_log(f"\n=== {title} ===")

        def worker() -> None:
            try:
                result = fn(lambda line: self._events.put(("log", line)))
                self._events.put(("done", dict(result)))
            except Exception as exc:
                self._events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    self._append_log(json.dumps(payload, indent=2, ensure_ascii=False))
                    self._finish(True)
                elif kind == "error":
                    self._append_log(f"ERROR: {payload}")
                    self._finish(False)
                    messagebox.showerror("Datasets", str(payload), parent=self)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(120, self._poll)

    def _finish(self, ok: bool) -> None:
        self._running = False
        self.close_btn.configure(state="normal")
        self.var_status.set("Completed" if ok else "Failed")
        if not ok:
            return
        bind_registry_to_app(self.app, self.var_registry.get() or None)
        if callable(self.on_updated):
            try:
                self.on_updated()
            except Exception:
                pass
        self._refresh_status()

    def _provision_coco(self) -> None:
        self._run(
            "Provision COCO 2017",
            lambda log: provision_coco2017(
                root=self.var_root.get(),
                include_train=bool(self.var_coco_train.get()),
                calibration_items=int(self.var_coco_calib.get()),
                registry_path=self.var_registry.get(),
                download_policy=str(self.var_coco_download_policy.get() or "auto"),
                mirror_base=str(self.var_coco_mirror.get() or "").strip() or None,
                materialize_calibration_subset=True,
                retain_train_archive=bool(self.var_coco_retain_train_archive.get()),
                log=log,
            ),
        )

    def _provision_imagenet(self) -> None:
        if not self.var_imagenet_terms.get():
            messagebox.showwarning(
                "ImageNet", "Confirm the ImageNet/Kaggle terms first.", parent=self
            )
            return
        mode = str(self.var_imagenet_download_mode.get() or "validation_only")
        allow_large = False
        if mode == "full_competition":
            allow_large = messagebox.askyesno(
                "Large ImageNet download",
                "The full Kaggle competition bundle is roughly 160+ GB before extraction. "
                "Continue with the full download?",
                parent=self,
            )
            if not allow_large:
                return

        status = kaggle_cli_status()
        if not bool(status.get("available")):
            install = messagebox.askyesno(
                "Kaggle CLI missing",
                "The Kaggle CLI is not installed in the Python environment running this GUI. "
                "Install/repair dataset support now? Configure Kaggle authentication afterwards "
                "and click Provision again.",
                parent=self,
            )
            if install:
                self._install_dependencies(confirm=False)
            return
        if not bool(status.get("authentication_configured_hint")):
            try_anyway = messagebox.askyesno(
                "Kaggle authentication not detected",
                "The Kaggle CLI is installed, but no standard token file or environment token "
                "was detected. A newer CLI may use another credential store. Try the API request "
                "anyway?\n\nOtherwise create/configure an API token in Kaggle settings, accept "
                "the ImageNet competition rules, and use Check Kaggle auth/rules/access.",
                parent=self,
            )
            if not try_anyway:
                return

        if (
            mode in {"validation_only", "validation_via_private_kernel"}
            or bool(self.var_imagenet_include_calibration.get())
        ):
            if not self.var_kaggle_username.get().strip():
                detected, _ = resolve_kaggle_username()
                if detected:
                    self.var_kaggle_username.set(detected)
            if not self.var_kaggle_username.get().strip():
                messagebox.showwarning(
                    "Kaggle username required",
                    "Kaggle access is valid, but access_token does not contain the public "
                    "account name. Enter the username from your Kaggle profile URL. It is "
                    "used only as owner of the private validation-export kernel.",
                    parent=self,
                )
                return

        self._run(
            "Provision ImageNet validation + calibration",
            lambda log: provision_imagenet_complete(
                root=self.var_root.get(),
                accept_terms=True,
                include_calibration=bool(self.var_imagenet_include_calibration.get()),
                calibration_items=int(self.var_imagenet_calib.get()),
                registry_path=self.var_registry.get(),
                download_mode=mode,
                allow_large_download=allow_large,
                kaggle_username=self.var_kaggle_username.get().strip() or None,
                validation_kernel_slug=self.var_kaggle_kernel_slug.get().strip()
                or IMAGENET_EXPORT_KERNEL_SLUG,
                calibration_kernel_slug=(
                    self.var_kaggle_calibration_kernel_slug.get().strip()
                    or IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG
                ),
                log=log,
            ),
        )

    def _collect_kaggle_export_diagnostics(self) -> None:
        username = self.var_kaggle_username.get().strip()
        if not username:
            try:
                username, _ = resolve_kaggle_username()
            except Exception:
                username = ""
        if not username:
            messagebox.showwarning(
                "Kaggle username required",
                "Enter the public Kaggle username used to own the private export kernel.",
                parent=self,
            )
            return
        self.var_kaggle_username.set(username)
        self._run(
            "Download ImageNet export diagnostics",
            lambda log: collect_imagenet_kernel_diagnostics(
                root=self.var_root.get(),
                kaggle_username=username,
                kernel_slug=self.var_kaggle_kernel_slug.get().strip()
                or IMAGENET_EXPORT_KERNEL_SLUG,
                log=log,
            ),
        )

    def _detect_kaggle_username(self) -> None:
        try:
            username, source = resolve_kaggle_username()
        except Exception as exc:
            messagebox.showerror("Kaggle username", str(exc), parent=self)
            return
        if username:
            self.var_kaggle_username.set(username)
            self.var_status.set(f"Kaggle username detected from {source}")
        else:
            messagebox.showinfo(
                "Kaggle username",
                "The access-token file authenticates correctly but does not contain the public "
                "username. Enter the first path component of your Kaggle profile URL, for "
                "example 'kmika' from kaggle.com/kmika.",
                parent=self,
            )

    def _import_imagenet(self) -> None:
        if not self.var_imagenet_source.get().strip():
            messagebox.showwarning(
                "ImageNet", "Select an ImageNet source or archive first.", parent=self
            )
            return
        self._run(
            "Import ImageNet",
            lambda log: import_imagenet(
                source=self.var_imagenet_source.get(),
                root=self.var_root.get(),
                calibration_items=int(self.var_imagenet_calib.get()),
                registry_path=self.var_registry.get(),
                validation_solution=self.var_imagenet_solution.get().strip() or None,
                labels=self.var_imagenet_labels.get().strip() or None,
                log=log,
            ),
        )

    def _register_imagenet(self) -> None:
        if not self.var_imagenet_val.get().strip():
            messagebox.showwarning(
                "ImageNet", "Select the class-organised ImageNet validation directory.", parent=self
            )
            return
        self._run(
            "Register ImageNet",
            lambda log: register_imagenet(
                train_root=self.var_imagenet_train.get() or None,
                val_root=self.var_imagenet_val.get(),
                labels=self.var_imagenet_labels.get() or None,
                calibration_items=int(self.var_imagenet_calib.get()),
                registry_path=self.var_registry.get(),
                log=log,
            ),
        )

    def _register_coco(self) -> None:
        self._run(
            "Register COCO",
            lambda log: register_coco2017(
                val_root=self.var_coco_val.get(),
                train_root=self.var_coco_train_root.get() or None,
                annotations_root=self.var_coco_annotations.get(),
                calibration_items=int(self.var_coco_calib.get()),
                registry_path=self.var_registry.get(),
                log=log,
            ),
        )

    def _repair_registry(self) -> None:
        self._run(
            "Repair/reindex existing dataset registry",
            lambda log: repair_dataset_registry(
                root=self.var_root.get(),
                registry_path=self.var_registry.get(),
                verify_manifests=True,
                log=log,
            ),
        )

    def _refresh_status(self) -> None:
        try:
            payload = registry_status(
                self.var_registry.get() or None, verify_manifests=False
            )
            payload["pycocotools"] = pycocotools_status()
            payload["kaggle_cli"] = kaggle_cli_status()
            self.status_text.configure(state="normal")
            self.status_text.delete("1.0", "end")
            self.status_text.insert(
                "1.0", json.dumps(payload, indent=2, ensure_ascii=False)
            )
            self.status_text.configure(state="disabled")
            self.var_status.set(format_registry_status_line(payload))
        except Exception as exc:
            self.var_status.set(f"Status failed: {exc}")

    def _install_dependencies(self, *, confirm: bool = True) -> None:
        command = (
            f'"{sys.executable}" -m pip install --upgrade '
            '"pycocotools>=2.0.7" "kaggle>=1.6"'
        )
        if confirm and not messagebox.askyesno(
            "Install dataset support",
            "Install or update pycocotools and the Kaggle CLI in the exact Python "
            "environment that runs this GUI?\n\n" + command,
            parent=self,
        ):
            return
        self._run(
            "Install/repair dataset support",
            lambda log: install_optional_dataset_dependencies(log=log),
        )

    def _check_kaggle_setup(self) -> None:
        self._run(
            "Check Kaggle CLI, credentials, rules and ImageNet access",
            lambda log: kaggle_cli_status(check_access=True, log=log),
        )


def open_dataset_provisioning_dialog(
    master: tk.Misc,
    *,
    app: Any = None,
    on_updated: Optional[Callable[[], None]] = None,
) -> DatasetProvisioningDialog:
    return DatasetProvisioningDialog(master, app=app, on_updated=on_updated)

# v60m: development profiles keep screening validation unless final validation
# was explicitly selected; calibration manifests may still auto-bind.
from onnx_splitpoint_tool.v60m_policy import install_dataset_binding_guards as _v60m_install_dataset_binding_guards
_v60m_install_dataset_binding_guards(globals())
