from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict, Mapping


def _copy_to_clipboard(root: Any, text: str) -> None:
    try:
        root.clipboard_clear()
        root.clipboard_append(str(text or ""))
        root.update_idletasks()
    except Exception as exc:  # pragma: no cover - Tk/clipboard edge cases
        messagebox.showerror("Copy", f"Failed to copy to clipboard:\n\n{exc}")


def _open_path(app: Any, path: str) -> None:
    target = str(path or "").strip()
    if not target:
        return
    opener = getattr(app, "_open_path", None)
    if callable(opener):
        opener(target)
        return
    _copy_to_clipboard(app, target)
    messagebox.showinfo("Open path", f"Path copied to clipboard:\n\n{target}")


def _readonly_entry(parent: tk.Widget, value: str) -> ttk.Entry:
    ent = ttk.Entry(parent)
    ent.insert(0, str(value or ""))
    ent.configure(state="readonly")
    return ent


def _summary_header(summary: Mapping[str, Any]) -> tuple[str, str]:
    status = str(summary.get("final_status") or "warn").strip().lower()
    accepted = int(summary.get("accepted_count") or 0)
    requested = int(summary.get("requested_cases") or 0)
    shortfall = int(summary.get("shortfall") or 0)
    rejected = int(summary.get("rejected_count") or 0)
    auto_skipped = int(summary.get("benign_count") or 0)
    preflight = summary.get("hailo_full_model_preflight") if isinstance(summary.get("hailo_full_model_preflight"), Mapping) else None

    if status == "ok":
        title = "Benchmark set created successfully"
        subtitle = f"{accepted}/{requested} requested cases were accepted."
    elif status == "cancelled":
        title = "Benchmark-set generation cancelled"
        subtitle = f"{accepted}/{requested} requested cases were accepted before cancellation."
    else:
        title = "Benchmark set created with warnings"
        subtitle = f"{accepted}/{requested} requested cases were accepted."

    details = []
    if shortfall > 0:
        details.append(f"shortfall {shortfall}")
    if rejected > 0:
        details.append(f"{rejected} rejected")
    if auto_skipped > 0:
        details.append(f"{auto_skipped} auto-skipped")
    if preflight and bool(preflight.get("aborted")):
        details.append("Hailo parser preflight blocked generation")
    elif preflight and bool(preflight.get("plan_adjusted")):
        details.append("Hailo parser preflight adjusted the plan")
    if details:
        subtitle += "  " + " · ".join(details)
    return title, subtitle


def show_benchmark_completion_dialog(
    app: Any,
    *,
    title: str,
    summary_data: Mapping[str, Any],
    fallback_text: str = "",
) -> None:
    parent = getattr(app, "root", None) or app
    dlg = tk.Toplevel(parent)
    dlg.title(title)
    dlg.transient(parent)
    dlg.grab_set()
    dlg.geometry("980x720")
    dlg.minsize(900, 620)

    title_font = tkfont.nametofont("TkDefaultFont").copy()
    try:
        base_size = int(title_font.cget("size"))
    except Exception:
        base_size = 10
    title_font.configure(size=max(base_size + 3, 13), weight="bold")

    outer = ttk.Frame(dlg, padding=12)
    outer.pack(fill="both", expand=True)

    bold_font = tkfont.nametofont("TkDefaultFont").copy()
    bold_font.configure(weight="bold")

    header_title, header_subtitle = _summary_header(summary_data)
    ttk.Label(outer, text=header_title, font=title_font).pack(anchor="w")
    ttk.Label(outer, text=header_subtitle, wraplength=920, justify="left").pack(anchor="w", pady=(4, 10))

    notebook = ttk.Notebook(outer)
    notebook.pack(fill="both", expand=True)

    tab_overview = ttk.Frame(notebook, padding=12)
    notebook.add(tab_overview, text="Overview")
    tab_raw = ttk.Frame(notebook, padding=12)
    notebook.add(tab_raw, text="Raw details")

    # --- Overview / Key figures -------------------------------------------------
    metrics = ttk.LabelFrame(tab_overview, text="Key figures", padding=10)
    metrics.pack(fill="x")
    for col in range(4):
        metrics.grid_columnconfigure(col, weight=1)

    metric_pairs = [
        ("Accepted", f"{int(summary_data.get('accepted_count') or 0)}/{int(summary_data.get('requested_cases') or 0)}"),
        ("Preferred shortlist", str(int(summary_data.get("preferred_shortlist_count") or 0))),
        ("Candidate pool", str(int(summary_data.get("candidate_search_pool_count") or 0))),
        ("Auto-skipped", str(int(summary_data.get("benign_count") or 0))),
        ("Rejected", str(int(summary_data.get("rejected_count") or 0))),
        ("Other warnings", str(int(summary_data.get("extra_warning_count") or 0))),
        ("Shortfall", str(int(summary_data.get("shortfall") or 0))),
        ("Backfilled", str(int(summary_data.get("backfilled_cases_count") or 0))),
    ]
    for idx, (label, value) in enumerate(metric_pairs):
        row = idx // 2
        col = (idx % 2) * 2
        ttk.Label(metrics, text=label + ":", font=bold_font).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=3)
        ttk.Label(metrics, text=value).grid(row=row, column=col + 1, sticky="w", pady=3)

    # --- Hailo outlook ---------------------------------------------------------
    if bool(summary_data.get("hailo_selected")):
        outlook_frame = ttk.LabelFrame(tab_overview, text="Hailo outlook", padding=10)
        outlook_frame.pack(fill="x", pady=(10, 0))
        outlook = summary_data.get("hailo_outlook") if isinstance(summary_data.get("hailo_outlook"), Mapping) else {}
        top_boundary = outlook.get("top_boundary")
        ttk.Label(
            outlook_frame,
            text=(
                f"Top candidate: b{top_boundary if top_boundary is not None else '?'}   "
                f"Likely single-context: {int(outlook.get('likely_single_context_count') or 0)}/{int(outlook.get('candidate_count') or 0)}"
            ),
        ).pack(anchor="w")
        ttk.Label(
            outlook_frame,
            text=(
                "Risk bands low/med/high: "
                f"{int(outlook.get('low_risk_count') or 0)}/"
                f"{int(outlook.get('medium_risk_count') or 0)}/"
                f"{int(outlook.get('high_risk_count') or 0)}"
            ),
        ).pack(anchor="w", pady=(4, 0))
        top_candidates = [f"b{int(x)}" for x in list(summary_data.get("top_hailo_boundaries") or [])[:6]]
        if top_candidates:
            ttk.Label(outlook_frame, text="Top Hailo candidates: " + ", ".join(top_candidates)).pack(anchor="w", pady=(4, 0))

    preflight = summary_data.get("hailo_full_model_preflight") if isinstance(summary_data.get("hailo_full_model_preflight"), Mapping) else None
    if preflight and bool(preflight.get("checked")):
        preflight_frame = ttk.LabelFrame(tab_overview, text="Hailo parser preflight", padding=10)
        preflight_frame.pack(fill="x", pady=(10, 0))

        result_count = int(preflight.get("result_count") or 0)
        ok_count = int(preflight.get("ok_count") or 0)
        failed_count = int(preflight.get("failed_count") or 0)
        unsupported_count = int(preflight.get("unsupported_failure_count") or 0)

        if bool(preflight.get("aborted")):
            headline = "Status: FAILED — benchmark generation was stopped before the candidate loop."
        elif bool(preflight.get("plan_adjusted")):
            headline = "Status: WARN — plan-aware preflight adjustment was applied and generation continued."
        elif failed_count > 0:
            headline = "Status: WARN — some Hailo targets failed this preflight, generation continued."
        else:
            headline = "Status: OK"
        ttk.Label(preflight_frame, text=headline, wraplength=900, justify="left").pack(anchor="w")
        ttk.Label(
            preflight_frame,
            text=(
                f"Targets ok/total: {ok_count}/{result_count}"
                f"   Failed: {failed_count}"
                f"   Unsupported-op failures: {unsupported_count}"
            ),
        ).pack(anchor="w", pady=(4, 0))

        for entry in list(preflight.get("results") or [])[:6]:
            if not isinstance(entry, Mapping):
                continue
            hw_arch = str(entry.get("hw_arch") or "?")
            status_txt = "OK" if bool(entry.get("ok")) else "FAILED"
            detail_parts = []
            unsupported_ops = [str(x).strip() for x in list(entry.get("unsupported_ops") or []) if str(x).strip()]
            unsupported_nodes = [str(x).strip() for x in list(entry.get("unsupported_nodes") or []) if str(x).strip()]
            unsupported_scope = str(entry.get("unsupported_scope") or "").strip()
            if unsupported_scope:
                detail_parts.append("scope=" + unsupported_scope)
            if unsupported_ops:
                detail_parts.append("ops=" + ", ".join(unsupported_ops[:4]))
            if unsupported_nodes:
                preview = ", ".join(unsupported_nodes[:3])
                if len(unsupported_nodes) > 3:
                    preview += ", …"
                detail_parts.append("nodes=" + preview)
            error_text = str(entry.get("error") or "").strip()
            if not detail_parts and error_text:
                compact = " ".join(error_text.split())
                if len(compact) > 180:
                    compact = compact[:179].rstrip() + "…"
                detail_parts.append(compact)
            detail = f" | {'; '.join(detail_parts)}" if detail_parts else ""
            ttk.Label(preflight_frame, text=f"{hw_arch}: {status_txt}{detail}", wraplength=900, justify="left").pack(anchor="w", pady=(4, 0))

    # --- Paths ----------------------------------------------------------------
    paths = ttk.LabelFrame(tab_overview, text="Paths", padding=10)
    paths.pack(fill="x", pady=(10, 0))
    paths.grid_columnconfigure(1, weight=1)

    path_rows = [
        ("Benchmark folder", str(summary_data.get("out_dir") or "")),
        ("Harness", str(summary_data.get("harness_path") or "")),
        ("Generation log", str(summary_data.get("bench_log_path") or "")),
    ]
    for row, (label, value) in enumerate(path_rows):
        ttk.Label(paths, text=label + ":", font=bold_font).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ent = _readonly_entry(paths, value)
        ent.grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(paths, text="Open", command=lambda p=value: _open_path(app, p)).grid(row=row, column=2, padx=(8, 0), pady=4)
        ttk.Button(paths, text="Copy", command=lambda p=value: _copy_to_clipboard(parent, p)).grid(row=row, column=3, padx=(6, 0), pady=4)

    # --- Issue groups ----------------------------------------------------------
    issues = ttk.LabelFrame(tab_overview, text="Main issues", padding=10)
    issues.pack(fill="both", expand=True, pady=(10, 0))

    issue_groups = [dict(g) for g in list(summary_data.get("issue_groups") or []) if isinstance(g, Mapping)]
    if issue_groups:
        tree_frame = ttk.Frame(issues)
        tree_frame.pack(fill="both", expand=True)

        cols = ("kind", "count", "examples")
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8)
        tree.heading("kind", text="Category")
        tree.heading("count", text="Count")
        tree.heading("examples", text="Examples")
        tree.column("kind", width=380, anchor="w")
        tree.column("count", width=80, anchor="center")
        tree.column("examples", width=280, anchor="w")
        ybar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ybar.set)
        tree.pack(side="left", fill="both", expand=True)
        ybar.pack(side="right", fill="y")

        detail_box = scrolledtext.ScrolledText(issues, height=8, wrap="word")
        detail_box.pack(fill="x", expand=False, pady=(10, 0))
        detail_box.configure(state="disabled")

        iid_to_group: Dict[str, Dict[str, Any]] = {}
        for idx, group in enumerate(issue_groups, start=1):
            iid = f"issue_{idx}"
            iid_to_group[iid] = group
            tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    f"{str(group.get('kind_label') or '').strip()} · {str(group.get('title') or '').strip()}".strip(" ·"),
                    int(group.get("count") or 0),
                    str(group.get("examples_text") or "—"),
                ),
            )

        def _update_issue_detail(*_args: Any) -> None:
            sel = tree.selection()
            if not sel:
                return
            group = iid_to_group.get(str(sel[0])) or {}
            lines = [str(group.get("title") or "Issue")]
            kind_label = str(group.get("kind_label") or "").strip()
            if kind_label:
                lines.insert(0, kind_label)
            lines.append(f"Count: {int(group.get('count') or 0)}")
            examples_text = str(group.get("examples_text") or "").strip()
            if examples_text and examples_text != "—":
                lines.append(f"Examples: {examples_text}")
            sample_detail = str(group.get("sample_detail") or "").strip()
            if sample_detail:
                lines.append("")
                lines.append(sample_detail)
            detail_box.configure(state="normal")
            detail_box.delete("1.0", "end")
            detail_box.insert("1.0", "\n".join(lines).strip() + "\n")
            detail_box.configure(state="disabled")

        tree.bind("<<TreeviewSelect>>", _update_issue_detail)
        first = tree.get_children()
        if first:
            tree.selection_set(first[0])
            _update_issue_detail()
    else:
        ttk.Label(issues, text="No warnings or rejections were recorded for this benchmark generation.").pack(anchor="w")

    # --- Raw details -----------------------------------------------------------
    raw_text = str(summary_data.get("raw_text") or fallback_text or "").strip()
    raw_box = scrolledtext.ScrolledText(tab_raw, wrap="word")
    raw_box.pack(fill="both", expand=True)
    raw_box.insert("1.0", (raw_text + "\n") if raw_text else "No additional detail available.\n")
    raw_box.configure(state="disabled")

    # --- Buttons ---------------------------------------------------------------
    btns = ttk.Frame(outer)
    btns.pack(fill="x", pady=(10, 0))
    ttk.Button(btns, text="Open benchmark folder", command=lambda: _open_path(app, str(summary_data.get("out_dir") or ""))).pack(side="left")
    ttk.Button(btns, text="Open log", command=lambda: _open_path(app, str(summary_data.get("bench_log_path") or ""))).pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Copy summary", command=lambda: _copy_to_clipboard(parent, raw_text or fallback_text)).pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="OK", command=dlg.destroy).pack(side="right")

    dlg.bind("<Escape>", lambda _e: dlg.destroy())
    dlg.update_idletasks()
    try:
        px = int(parent.winfo_rootx()) + 40
        py = int(parent.winfo_rooty()) + 40
        dlg.geometry(f"+{px}+{py}")
    except Exception:
        pass
    dlg.wait_window()
