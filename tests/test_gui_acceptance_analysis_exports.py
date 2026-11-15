"""Focused regressions found during the normal 2.95.5 GUI acceptance.

The end-to-end Tk session is recorded outside the source tree. These tests
exercise the actual table callbacks with only the file/message dialogs replaced.
"""
from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


def _gui_module():
    from onnx_splitpoint_tool import gui_app

    return gui_app


def _table_app(module, current_path, legacy_path):
    app = SimpleNamespace(
        gui_state=SimpleNamespace(current_model_path=current_path),
        model_path=legacy_path,
        analysis={
            "costs_bytes": [1024.0],
            "crossing_counts_all": [1],
            "unknown_crossing_counts": [0],
            "flops_left_prefix": [1e9],
            "total_flops": 2e9,
            "semantic_labels_by_boundary": ["Conv → Relu"],
        },
        current_picks=[0],
    )
    app._make_tex_table = MethodType(module.SplitPointAnalyserGUI._make_tex_table, app)
    return app


@pytest.mark.parametrize(
    ("selected", "legacy", "expected_name"),
    [
        ("/selected/selected_classifier.onnx", "/stale/old_detector.onnx", "selected_classifier"),
        ("/selected/selected_classifier.onnx", None, "selected_classifier"),
        (None, "/legacy/fallback_classifier.onnx", "fallback_classifier"),
    ],
)
def test_tex_export_uses_current_gui_model_and_real_table(
    monkeypatch, tmp_path, selected, legacy, expected_name
):
    module = _gui_module()
    app = _table_app(module, selected, legacy)
    target = tmp_path / "table with spaces.tex"
    dialogs = []
    messages = []

    def save_dialog(**kwargs):
        dialogs.append(kwargs)
        return str(target)

    monkeypatch.setattr(module.filedialog, "asksaveasfilename", save_dialog)
    monkeypatch.setattr(module.messagebox, "showinfo", lambda *args: messages.append(args))
    monkeypatch.setattr(
        module.messagebox, "showerror",
        lambda *args: pytest.fail(f"Unexpected export failure: {args}"),
    )

    module.SplitPointAnalyserGUI._export_tex_table(app)

    assert dialogs[0]["initialfile"] == f"split_candidates_{expected_name}.tex"
    text = target.read_text(encoding="utf-8")
    assert "\\begin{table}" in text
    assert expected_name.replace("_", "\\_") in text
    assert "old\\_detector" not in text
    assert "0 & Conv" in text
    assert messages[0][0] == "Export complete"


def test_tex_export_cancel_leaves_existing_file_unchanged(monkeypatch, tmp_path):
    module = _gui_module()
    app = _table_app(module, "/selected/model.onnx", "/stale/model.onnx")
    target = tmp_path / "existing.tex"
    target.write_text("preserved user content", encoding="utf-8")
    monkeypatch.setattr(module.filedialog, "asksaveasfilename", lambda **kwargs: "")
    monkeypatch.setattr(
        module.messagebox, "showinfo", lambda *args: pytest.fail("Cancelled export reported success")
    )

    module.SplitPointAnalyserGUI._export_tex_table(app)

    assert target.read_text(encoding="utf-8") == "preserved user content"
    assert list(tmp_path.iterdir()) == [target]


def test_tex_export_without_analysis_reports_guidance(monkeypatch):
    module = _gui_module()
    app = SimpleNamespace(analysis=None, current_picks=[])
    messages = []
    monkeypatch.setattr(module.messagebox, "showinfo", lambda *args: messages.append(args))
    monkeypatch.setattr(
        module.filedialog, "asksaveasfilename",
        lambda **kwargs: pytest.fail("Export without analysis opened a save dialog"),
    )

    module.SplitPointAnalyserGUI._export_tex_table(app)

    assert messages == [("Nothing to export", "Run an analysis first.")]


@pytest.mark.parametrize(
    ("params", "error_text"),
    [
        (ValueError("Top-k must be a positive integer."), "Top-k"),
        (SimpleNamespace(prune_skip_block=True, skip_min_span=-1, skip_allow_last_n=0), "Min skip span"),
        (SimpleNamespace(prune_skip_block=True, skip_min_span=0, skip_allow_last_n=-1), "Allow last N"),
    ],
)
def test_manual_split_invalid_parameters_do_not_prompt_write_or_start_worker(
    monkeypatch, tmp_path, params, error_text
):
    module = _gui_module()
    messages = []

    def read_params():
        if isinstance(params, ValueError):
            raise params
        return params

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid split parameters must stop before dialogs, writes or workers")

    app = SimpleNamespace(
        gui_state=SimpleNamespace(current_model_path=str(tmp_path / "selected.onnx")),
        model_path=str(tmp_path / "stale.onnx"),
        analysis={"order": [0, 1], "nodes": [], "model": object()},
        _selected_boundary_index=lambda: 0,
        _read_params=read_params,
    )
    monkeypatch.setattr(module.asc, "cut_tensors_for_boundary", lambda *args: ["cut"])
    monkeypatch.setattr(module.simpledialog, "askstring", forbidden)
    monkeypatch.setattr(module.filedialog, "askdirectory", forbidden)
    monkeypatch.setattr(module.os, "makedirs", forbidden)
    monkeypatch.setattr(module.threading, "Thread", forbidden)
    monkeypatch.setattr(module.messagebox, "showerror", lambda *args: messages.append(args))

    module.SplitPointAnalyserGUI._split_selected_boundary(app)

    assert len(messages) == 1
    assert messages[0][0] == "Invalid parameters"
    assert error_text in messages[0][1]
    assert not list(tmp_path.iterdir())


# Reuse the real ONNX/manual-worker fixture from the preceding repair; only its
# dialogs, compiler dispatch and rendering are test doubles.
from test_manual_split_callback import manual_gui  # noqa: E402,F401


def test_manual_export_preserves_zero_context_hops_in_renderer_and_manifest(
    manual_gui, monkeypatch, tmp_path
):
    from test_manual_split_callback import _manual_export

    real_callback = manual_gui.gui.SplitPointAnalyserGUI._split_selected_boundary
    real_export_context = manual_gui.legacy.asc.export_boundary_graphviz_context
    renderer_hops = []

    def configure_zero_hops(app):
        app.var_split_ctx_hops.set("0")
        return real_callback(app)

    def observe_context(*args, **kwargs):
        renderer_hops.append(kwargs["hops"])
        return real_export_context(*args, **kwargs)

    monkeypatch.setattr(
        manual_gui.gui.SplitPointAnalyserGUI, "_split_selected_boundary", configure_zero_hops
    )
    monkeypatch.setattr(
        manual_gui.legacy.asc, "export_boundary_graphviz_context", observe_context
    )

    _selected, out_dir, manifest, _timeouts = _manual_export(
        manual_gui, monkeypatch, tmp_path, boundary=2
    )

    assert manifest["split_context_hops"] == 0
    assert renderer_hops == [0, 0]
    assert (out_dir / manifest["split_context"]["dot"]).is_file()
    assert (out_dir / manifest["split_context_cutflow"]["dot"]).is_file()


@pytest.fixture
def export_preview(tmp_path):
    """Actual Tk widgets, isolated from persisted settings and the full app."""
    import tkinter as tk
    from onnx_splitpoint_tool.gui.panels import panel_export

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    app = SimpleNamespace(
        gui_state=SimpleNamespace(current_model_path=str(tmp_path / "selected_model.onnx")),
        model_path=str(tmp_path / "stale_model.onnx"),
        default_output_dir=str(tmp_path / "export space"),
        _selected_boundary_index=lambda: 27,
        var_split_runner=tk.BooleanVar(root, value=True),
        var_split_validate=tk.BooleanVar(root, value=False),
        var_split_ctx_full=tk.BooleanVar(root, value=False),
        var_split_ctx_cutflow=tk.BooleanVar(root, value=False),
        var_split_ctx_hops=tk.StringVar(root, value="0"),
        var_split_folder=tk.BooleanVar(root, value=True),
    )
    frame = panel_export.build_panel(root, app)

    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from descendants(child)

    widgets = list(descendants(frame))
    preview = next(widget for widget in widgets if isinstance(widget, tk.Text))
    checks = {
        str(widget.cget("text")): widget
        for widget in widgets if widget.winfo_class() == "TCheckbutton"
    }
    try:
        yield app, preview, checks
    finally:
        root.destroy()


def test_export_preview_validation_changes_immediately(export_preview):
    app, preview, _checks = export_preview
    assert "validation_core.json" not in preview.get("1.0", "end")

    app.var_split_validate.set(True)

    text = preview.get("1.0", "end")
    assert "validation_core.json" in text
    assert "validation_report.json" not in text
    app.var_split_validate.set(False)
    assert "validation_core.json" not in preview.get("1.0", "end")


def test_export_preview_uses_selected_source_and_always_included_files(export_preview):
    _app, preview, checks = export_preview
    text = preview.get("1.0", "end")
    assert "selected_model.onnx (full model copy)" in text
    assert "selected_model_part1_b27.onnx" in text
    assert "selected_model_part2_b27.onnx" in text
    assert "split_manifest.json" in text
    assert "stale_model" not in text
    for label in ("Models", "Context", "Reports"):
        assert "disabled" in checks[label].state()
        checks[label].invoke()
    assert preview.get("1.0", "end") == text


def test_export_preview_matches_folder_and_direct_context_policy(export_preview):
    app, preview, checks = export_preview
    text = preview.get("1.0", "end")
    assert "split_context_b27.dot" in text
    assert "split_context_b27_cutflow.dot" in text
    assert "disabled" in checks["Full"].state()
    assert "disabled" in checks["Cut-flow"].state()

    app.var_split_folder.set(False)
    app.var_split_ctx_cutflow.set(True)

    text = preview.get("1.0", "end")
    assert "split_context_b27.dot" not in text
    assert "split_context_b27_cutflow.dot" in text
    assert "cut-flow hops=0" in text
    assert "disabled" not in checks["Full"].state()
    assert "disabled" not in checks["Cut-flow"].state()
    app.var_split_ctx_cutflow.set(False)
    assert "split_context_b27_cutflow.dot" in preview.get("1.0", "end")


def test_manual_export_existing_regular_file_reports_failure_and_preserves_it(
    monkeypatch, tmp_path
):
    module = _gui_module()
    from test_manual_split_callback import _Value

    occupied = tmp_path / "selected_split_b0"
    occupied.write_text("preserve existing user file", encoding="utf-8")
    messages = []
    app = SimpleNamespace(
        gui_state=SimpleNamespace(current_model_path=str(tmp_path / "selected.onnx")),
        model_path=None,
        analysis={"order": [0, 1], "nodes": [], "model": object()},
        _selected_boundary_index=lambda: 0,
        _read_params=lambda: SimpleNamespace(
            prune_skip_block=False, skip_min_span=0, skip_allow_last_n=0
        ),
        default_output_dir=None,
        var_split_folder=_Value(True),
    )
    monkeypatch.setattr(module.asc, "cut_tensors_for_boundary", lambda *args: ["cut"])
    monkeypatch.setattr(module.simpledialog, "askstring", lambda *args, **kwargs: "")
    monkeypatch.setattr(module.filedialog, "askdirectory", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(module.messagebox, "showerror", lambda *args: messages.append(args))
    monkeypatch.setattr(
        module.threading, "Thread", lambda **kwargs: pytest.fail("Invalid export started a worker")
    )

    module.SplitPointAnalyserGUI._split_selected_boundary(app)

    assert len(messages) == 1
    assert messages[0][0] == "Export failed"
    assert "selected_split_b0" in messages[0][1]
    assert occupied.read_text(encoding="utf-8") == "preserve existing user file"
    assert list(tmp_path.iterdir()) == [occupied]


@pytest.mark.parametrize("artifact", ["full_context", "cutflow_context", "runner", "validation", "validation_report"])
def test_selected_artifact_write_failure_does_not_publish_success_manifest(
    manual_gui, monkeypatch, tmp_path, artifact
):
    from test_manual_split_callback import _manual_export

    module = manual_gui.legacy
    real_callback = manual_gui.gui.SplitPointAnalyserGUI._split_selected_boundary
    real_context = module.asc.export_boundary_graphviz_context
    detail = f"owned test output cannot write {artifact}"

    def configure_requested_artifacts(app):
        if artifact == "runner":
            app.var_split_runner.set(True)
        return real_callback(app)

    def context_with_write_failure(*args, **kwargs):
        is_cutflow = bool(kwargs.get("cut_flow_only"))
        if (artifact == "full_context" and not is_cutflow) or (
            artifact == "cutflow_context" and is_cutflow
        ):
            raise OSError(detail)
        return real_context(*args, **kwargs)

    def runner_with_write_failure(*args, **kwargs):
        raise OSError(detail)

    monkeypatch.setattr(
        manual_gui.gui.SplitPointAnalyserGUI, "_split_selected_boundary",
        configure_requested_artifacts,
    )
    monkeypatch.setattr(module.asc, "export_boundary_graphviz_context", context_with_write_failure)
    if artifact == "runner":
        monkeypatch.setattr(module.asc, "write_runner_skeleton_onnxruntime", runner_with_write_failure)

    if artifact == "validation":
        monkeypatch.setattr(module.asc, "validate_split_onnxruntime", runner_with_write_failure)
    elif artifact == "validation_report":
        real_json_writer = module.write_benchmark_json_atomic

        def report_write_failure(path, payload):
            if path.name == "validation_core.json":
                raise OSError(detail)
            return real_json_writer(path, payload)

        monkeypatch.setattr(module, "write_benchmark_json_atomic", report_write_failure)

    # The shared positive-path helper asserts a successful completion. This
    # negative case must fail that assertion, then prove the actual failure
    # dialog and absence of a manifest for the incomplete export.
    with pytest.raises(AssertionError):
        _manual_export(manual_gui, monkeypatch, tmp_path, boundary=2)

    assert manual_gui.messages[-1][:2] == ("showerror", "Split failed")
    assert detail in manual_gui.messages[-1][2]
    out_dir = tmp_path / "manual export"
    assert not (out_dir / "split_manifest.json").exists()
    assert (out_dir / "chosen_part1_b2.onnx").is_file()
    assert (out_dir / "chosen_part2_b2.onnx").is_file()


def test_numeric_validation_fail_remains_reported_without_technical_relabeling(
    manual_gui, monkeypatch, tmp_path
):
    from test_manual_split_callback import _manual_export
    import json

    monkeypatch.setattr(
        manual_gui.legacy.asc, "validate_split_onnxruntime",
        lambda **kwargs: {"ok": False, "reason": "bounded numeric mismatch fixture"},
    )

    _selected, out_dir, manifest, _timeouts = _manual_export(
        manual_gui, monkeypatch, tmp_path, boundary=2
    )

    assert "ORT validation: FAIL" in manual_gui.messages[-1][2]
    assert (out_dir / "split_manifest.json").is_file()
    assert json.loads((out_dir / "validation_core.json").read_text())["ok"] is False
    assert manifest["boundary"] == 2
