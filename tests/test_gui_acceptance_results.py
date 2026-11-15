"""Regression checks for result loading through the real Tk panel callbacks."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def result_panel(tmp_path, monkeypatch):
    import tkinter as tk

    from onnx_splitpoint_tool.gui.panels import panel_benchmark_analysis as panel

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    app = SimpleNamespace(default_output_dir=str(tmp_path / "workdir"))
    messages = []
    errors = []
    monkeypatch.setattr(panel.messagebox, "showerror", lambda title, msg: messages.append((title, msg)))
    root.report_callback_exception = lambda typ, value, tb: errors.append(value)

    class QueuedWorker:
        """Complete the worker before Tk dispatches its queued callbacks."""

        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(panel.threading, "Thread", QueuedWorker)
    frame = panel.build_panel(root, app)
    frame.pack()
    try:
        yield root, frame, app, messages, errors
    finally:
        root.destroy()


def _buttons(widget):
    if widget.winfo_class() == "TButton":
        yield widget
    for child in widget.winfo_children():
        yield from _buttons(child)


@pytest.mark.parametrize("source_kind", ["missing", "empty", "invalid_json"])
def test_analysis_failure_keeps_original_diagnostic_after_worker_exit(result_panel, tmp_path, source_kind):
    root, frame, app, messages, errors = result_panel
    source = tmp_path / f"{source_kind} results with spaces"
    if source_kind != "missing":
        source.mkdir()
    if source_kind == "invalid_json":
        (source / "benchmark_set.json").write_text("{not valid json", encoding="utf-8")
        (source / "benchmark_results_cpu.csv").write_text("boundary,full_mean_ms\n", encoding="utf-8")

    app.var_benchmark_analysis_source.set(str(source))
    analyze = next(button for button in _buttons(frame) if button.cget("text") == "Analyse starten")
    analyze.invoke()
    root.update()

    assert not errors, f"Tk callback failed instead of displaying the source error: {errors}"
    assert len(messages) == 1
    title, diagnostic = messages[0]
    assert title == "Benchmark-Analyse"
    if source_kind == "invalid_json":
        assert "Expecting property name" in diagnostic
    elif source_kind == "empty":
        assert "Could not locate a benchmark results directory" in diagnostic
    else:
        assert str(source) in diagnostic
    assert app.var_benchmark_analysis_status.get() == f"Analyse fehlgeschlagen: {diagnostic}"
    assert str(analyze.cget("state")) == "normal"


@pytest.mark.parametrize("data", ["", "1,,,False\n", "1,3.5,2.0,True\n"])
def test_loaded_results_render_decision_tables_without_callback_errors(result_panel, tmp_path, data):
    """A real successful reader must reach the decision-table renderer."""
    import json

    root, frame, app, messages, errors = result_panel
    source = tmp_path / "real result schema"
    source.mkdir()
    (source / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"boundary": 1}], "summary": {"generated_cases": 1}}),
        encoding="utf-8",
    )
    (source / "benchmark_results_cpu.csv").write_text(
        "boundary,full_mean_ms,composed_mean_ms,final_pass_all\n" + data,
        encoding="utf-8",
    )
    app.var_benchmark_analysis_source.set(str(source))
    analyze = next(button for button in _buttons(frame) if button.cget("text") == "Analyse starten")
    analyze.invoke()
    root.update()

    assert not errors, f"Tk result rendering failed: {errors}"
    assert not messages
    assert app.var_benchmark_analysis_status.get().startswith("Analyse geladen:")
    assert str(analyze.cget("state")) == "normal"


def test_analysis_reader_accepts_large_quoted_json_and_preserves_missing_metrics(tmp_path):
    import csv
    import json

    from onnx_splitpoint_tool.benchmark.analysis import load_benchmark_analysis

    source = tmp_path / "large canonical result"
    source.mkdir()
    (source / "benchmark_set.json").write_text(
        json.dumps({"cases": [{"boundary": 1}], "summary": {"generated_cases": 1}}),
        encoding="utf-8",
    )
    evidence = json.dumps({"diagnostic": 'quoted "JSON", with newlines\n' * 10000})
    assert len(evidence) > 131072
    with (source / "benchmark_results_cpu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["boundary", "final_pass_all", "composed_mean_ms", "evidence"])
        writer.writeheader()
        writer.writerow({"boundary": 1, "final_pass_all": False, "composed_mean_ms": "", "evidence": evidence})

    previous_limit = csv.field_size_limit()
    report = load_benchmark_analysis(source, tmp_path / "cache")

    assert csv.field_size_limit() == previous_limit
    assert report.provider_rows["cpu"][0]["evidence"] == evidence
    assert report.providers[0].full_baseline_ms is None
    assert report.providers[0].best_composed_ms is None
    assert report.issues


def test_analysis_reader_restores_csv_limit_after_read_error(tmp_path):
    import csv

    from onnx_splitpoint_tool.benchmark.analysis import _read_csv_rows

    source = tmp_path / "invalid.csv"
    source.write_bytes(b"boundary,evidence\n1,\xff\n")
    previous_limit = csv.field_size_limit()
    with pytest.raises(UnicodeDecodeError):
        _read_csv_rows(source)
    assert csv.field_size_limit() == previous_limit


def test_closed_subplot_releases_only_its_parent_close_callback(result_panel, tmp_path, monkeypatch):
    """Rerendering a plot must not close its already destroyed subplot again."""
    import json

    from matplotlib.backend_bases import CloseEvent, FigureCanvasBase
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    from matplotlib.figure import Figure

    root, frame, app, messages, errors = result_panel
    source = tmp_path / "results"
    source.mkdir()
    (source / "benchmark_set.json").write_text(json.dumps({"cases": []}), encoding="utf-8")
    (source / "benchmark_results_cpu.csv").write_text("boundary,full_mean_ms\n", encoding="utf-8")
    app.var_benchmark_analysis_source.set(str(source))
    next(button for button in _buttons(frame) if button.cget("text") == "Analyse starten").invoke()
    root.update()
    toolbar = frame.benchmark_analysis_state["toolbars"]["decision_fps_latency"]
    child_canvas = FigureCanvasBase(Figure())
    child_manager_calls = []
    independent_calls = []
    independent_id = toolbar.canvas.mpl_connect("close_event", lambda event: independent_calls.append(event))

    def create_subplot(self, *args):
        self.canvas.mpl_connect("close_event", lambda event: child_manager_calls.append(event))
        self.subplot_tool = SimpleNamespace(figure=child_canvas.figure)
        return self.subplot_tool

    monkeypatch.setattr(NavigationToolbar2Tk, "configure_subplots", create_subplot)
    toolbar._buttons["Subplots"].invoke()
    parent_event = CloseEvent("close_event", toolbar.canvas)
    toolbar.canvas.callbacks.process("close_event", parent_event)
    assert len(child_manager_calls) == 1

    child_canvas.callbacks.process("close_event", CloseEvent("close_event", child_canvas))
    toolbar.canvas.callbacks.process("close_event", parent_event)

    assert len(child_manager_calls) == 1
    assert len(independent_calls) == 2
    assert independent_id in toolbar.canvas.callbacks.callbacks["close_event"]
    assert not errors
