"""Run the ordinary manual split callback and worker without Tk or compilers."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from onnx_splitpoint_tool.core_analysis import analyze_model


class _Value:
    def __init__(self, value="", **_kwargs):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Widget:
    """Only the progress-dialog surface; no application logic is replaced."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __getattr__(self, name):
        if name == "index":
            return lambda *_args: "1.0"
        return lambda *_args, **_kwargs: None


def _write_model(path: Path, *, external: bool, weight: float):
    path.parent.mkdir(parents=True)
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Add", ["x", "left_weight"], ["left"]),
                helper.make_node("Sub", ["x", "left_weight"], ["right"]),
                helper.make_node("Concat", ["left", "right"], ["joined"], axis=1),
                helper.make_node("Add", ["joined", "right_weight"], ["y"]),
            ],
            "manual_split_regression",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
            initializer=[
                numpy_helper.from_array(np.full((1, 2), weight, dtype=np.float32), "left_weight"),
                numpy_helper.from_array(np.full((1, 4), weight, dtype=np.float32), "right_weight"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=9,
    )
    helper.set_model_props(model, {"task": "classification"})
    if external:
        onnx.save_model(
            model, path, save_as_external_data=True, all_tensors_to_one_file=True,
            location="weights.data", size_threshold=0,
        )
    else:
        onnx.save_model(model, path)


@pytest.fixture
def manual_gui(monkeypatch):
    # Match the existing GUI tests: import the actual GUI without selecting a
    # display backend, then replace widgets rather than the manual callback.
    import matplotlib

    monkeypatch.setattr(matplotlib, "use", lambda *_args, **_kwargs: None)
    gui = importlib.import_module("onnx_splitpoint_tool.gui.app")
    legacy = importlib.import_module("onnx_splitpoint_tool.gui_app")
    for name in ("BooleanVar", "StringVar"):
        monkeypatch.setattr(legacy.tk, name, _Value)
    for name in ("Toplevel", "Text"):
        monkeypatch.setattr(legacy.tk, name, _Widget)
    for name in ("Label", "Progressbar", "Frame", "Scrollbar"):
        monkeypatch.setattr(legacy.ttk, name, _Widget)
    monkeypatch.setattr(legacy.simpledialog, "askstring", lambda *_args, **_kwargs: "")

    messages = []
    for kind in ("showinfo", "showerror"):
        monkeypatch.setattr(
            legacy.messagebox, kind,
            lambda title, body, _kind=kind, **_kwargs: messages.append((_kind, title, body)),
        )

    # Exercise the real context exporter too, but keep this regression
    # independent of Graphviz, matplotlib rendering and subprocesses.
    export_context = legacy.asc.export_boundary_graphviz_context

    def context_without_render(*args, **kwargs):
        kwargs["render"] = False
        return export_context(*args, **kwargs)

    monkeypatch.setattr(legacy.asc, "export_boundary_graphviz_context", context_without_render)
    external_data_calls = []
    ensure_external_data = legacy.asc.ensure_external_data_files

    def record_external_data(model, **kwargs):
        external_data_calls.append(kwargs)
        return ensure_external_data(model, **kwargs)

    monkeypatch.setattr(legacy.asc, "ensure_external_data_files", record_external_data)
    validation_calls = []

    def record_validation(**kwargs):
        validation_calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(legacy.asc, "validate_split_onnxruntime", record_validation)
    from onnx_splitpoint_tool import hailo_backend

    monkeypatch.setattr(
        hailo_backend, "hailo_build_hef_auto",
        lambda *_args, **_kwargs: pytest.fail("unexpected Hailo compiler dispatch"),
    )
    return SimpleNamespace(
        gui=gui, legacy=legacy, messages=messages, validation_calls=validation_calls,
        external_data_calls=external_data_calls,
    )


def _manual_export(manual_gui, monkeypatch, tmp_path, *, boundary, external=False,
                   fallback=False, hailo=False, hailo_targets=True):
    selected = tmp_path / "selected model" / "chosen.onnx"
    stale = tmp_path / "old model" / "old.onnx"
    _write_model(selected, external=external, weight=0.25)
    _write_model(stale, external=external, weight=50.0)
    analysis = analyze_model(str(selected))
    out_dir = tmp_path / "manual export"
    out_dir.mkdir()
    monkeypatch.setattr(
        manual_gui.legacy.filedialog, "askdirectory", lambda **_kwargs: str(out_dir)
    )

    timeout_calls = []

    def timeout():
        # A pure ONNX export must never read the Hailo timeout. A requested
        # build captures it on the GUI thread before the worker starts.
        assert hailo and hailo_targets, "ONNX-only export read a Hailo timeout"
        assert not worker_started
        timeout_calls.append(2473)
        return 2473

    params = SimpleNamespace(
        prune_skip_block=False, skip_min_span=0, skip_allow_last_n=0,
        batch_override=1, llm_enable=False,
    )
    app = SimpleNamespace(
        gui_state=SimpleNamespace(current_model_path=None if fallback else str(selected)),
        model_path=str(selected if fallback else stale), analysis=analysis,
        default_output_dir=None, memory_by_boundary={},
        _selected_boundary_index=lambda: boundary, _read_params=lambda: params,
        _accel_by_name=lambda _name: {}, _hailo_hef_timeout_seconds=timeout,
        _hailo_publish_gui_diagnostics=lambda *_args, **_kwargs: None,
        configure=lambda **_kwargs: None, update_idletasks=lambda: None,
        after=lambda _delay, callback: callback(),
    )
    values = {
        "var_split_folder": False, "var_strict_boundary": True,
        "var_split_validate": True, "var_split_runner": False,
        "var_runner_target": "auto", "var_split_ctx_full": True,
        "var_split_ctx_cutflow": True, "var_split_ctx_hops": "1",
        "var_split_eps": "1e-4", "var_memf_left_accel": "",
        "var_memf_right_accel": "", "var_llm_mode": "decode",
        "var_llm_decode": "0", "var_hailo_hef_full": hailo,
        "var_hailo_hef_part1": hailo, "var_hailo_hef_part2": hailo,
        "var_hailo_hef_hailo8_enable": hailo_targets,
        "var_hailo_hef_hailo10_enable": hailo and hailo_targets,
    }
    for name, value in values.items():
        setattr(app, name, _Value(value))

    worker_started = False

    class InlineThread:
        def __init__(self, *, target, daemon):
            assert daemon
            self.target = target

        def start(self):
            nonlocal worker_started
            worker_started = True
            # Simulate the user changing both path fields before a scheduled
            # worker runs. Every file operation must use its captured source.
            app.model_path = str(stale)
            app.gui_state.current_model_path = str(stale)
            self.target()

    monkeypatch.setattr(manual_gui.legacy.threading, "Thread", InlineThread)
    manual_gui.gui.SplitPointAnalyserGUI._split_selected_boundary(app)
    assert manual_gui.messages, "manual callback never reported completion"
    assert manual_gui.messages[-1][:2] == ("showinfo", "Split complete"), manual_gui.messages
    manifest_path = out_dir / "split_manifest.json"
    assert manifest_path.is_file(), "saved ONNX parts alone are not a complete export"
    return selected, out_dir, json.loads(manifest_path.read_text()), timeout_calls


@pytest.mark.parametrize("boundary,cut", [(1, ["left", "right"]), (2, ["joined"])])
@pytest.mark.parametrize("external", [False, True], ids=["embedded", "external"])
def test_manual_callback_completes_with_selected_source(
    manual_gui, monkeypatch, tmp_path, boundary, cut, external
):
    selected, out_dir, manifest, timeout_calls = _manual_export(
        manual_gui, monkeypatch, tmp_path, boundary=boundary, external=external
    )
    assert timeout_calls == []
    assert manifest["cut_tensors"] == cut
    assert manifest["cut"]["full_names"] == cut
    assert manifest["io"]["part2_external_inputs"] == cut
    assert manifest["pipeline"]["links"] == [{"from": name, "to": name} for name in cut]
    assert manifest["full_model_source"] == str(selected)
    assert manifest["models"]["full"] == {"path": selected.name, "source": str(selected)}
    assert (out_dir / manifest["full_model"]).read_bytes() == selected.read_bytes()
    assert [call["source_model_path"] for call in manual_gui.external_data_calls] == [str(selected)] * 2
    assert len(manual_gui.validation_calls) == 1
    assert manual_gui.validation_calls[0]["full_model_path"] == str(selected)
    assert json.loads((out_dir / "validation_core.json").read_text()) == {"ok": True}

    for key in ("split_context", "split_context_cutflow"):
        assert (out_dir / manifest[key]["dot"]).read_text().startswith("digraph")
    for key in ("part1", "part2"):
        model = onnx.load(out_dir / manifest[key])
        onnx.checker.check_model(model)
        assert model.graph.initializer, "fixture must exercise external data on each side"
        for initializer in model.graph.initializer:
            np.testing.assert_array_equal(numpy_helper.to_array(initializer), 0.25)
    if external:
        assert (out_dir / "weights.data").read_bytes() == (selected.parent / "weights.data").read_bytes()


def test_manual_callback_retains_model_path_fallback(manual_gui, monkeypatch, tmp_path):
    selected, out_dir, manifest, _ = _manual_export(
        manual_gui, monkeypatch, tmp_path, boundary=2, fallback=True
    )
    assert manifest["full_model_source"] == str(selected)
    assert (out_dir / manifest["full_model"]).read_bytes() == selected.read_bytes()
    assert manual_gui.validation_calls[0]["full_model_path"] == str(selected)


def test_manual_callback_captures_timeout_for_all_mocked_hailo_builds(
    manual_gui, monkeypatch, tmp_path
):
    from onnx_splitpoint_tool import hailo_backend

    calls = []

    def fake_build(model_path, **kwargs):
        calls.append((model_path, kwargs))
        path = Path(kwargs["outdir"]) / "mock.hef"
        path.write_bytes(b"test double; no compiler or device")
        return SimpleNamespace(ok=True, hef_path=str(path))

    monkeypatch.setattr(hailo_backend, "hailo_build_hef_auto", fake_build)
    selected, out_dir, manifest, timeout_calls = _manual_export(
        manual_gui, monkeypatch, tmp_path, boundary=1, hailo=True
    )
    assert timeout_calls == [2473]
    assert manifest["hailo"]["config"]["timeout_s"] == 2473
    assert len(calls) == 6  # Full, Part-1 and Part-2 for both selected targets.
    assert [path for path, _kwargs in calls] == [
        str(selected), str(out_dir / manifest["part1"]), str(out_dir / manifest["part2"])
    ] * 2
    assert {kwargs["hw_arch"] for _path, kwargs in calls} == {"hailo8", "hailo10h"}
    for _path, kwargs in calls:
        assert kwargs["wsl_timeout_s"] == 2473
        assert kwargs["task"] == "classification"
        assert kwargs["build_evidence_context"]["full_source_onnx_path"] == str(selected)
    for target in manifest["hailo"]["hefs"].values():
        assert set(target) == {"full", "part1", "part2"}
        assert all((out_dir / path).is_file() for path in target.values())


def test_manual_callback_without_hailo_target_does_not_read_timeout(
    manual_gui, monkeypatch, tmp_path
):
    _selected, _out_dir, manifest, timeout_calls = _manual_export(
        manual_gui, monkeypatch, tmp_path, boundary=2, hailo=True, hailo_targets=False
    )
    assert timeout_calls == []
    assert "hailo" not in manifest
