"""Real editor save/load transitions and read-only overlay validation; no GPU."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool import run_modes as rm
from onnx_splitpoint_tool import hailo_dependency_plan as hp
from onnx_splitpoint_tool.gui import run_mode_editor as gui
from test_v27934_hailo8_dependency_plan import h8, stage


MANIFEST_PATH = "build.hailo.compute_by_family.hailo8.dependency_manifest"


def _dialog(config, on_saved):
    """Use real Tcl variables and real save/coerce; widget drawing is external."""
    interpreter = tk.Tcl()
    dialog = SimpleNamespace(mode_id="standard", config=copy.deepcopy(config),
                             mode=copy.deepcopy(config["modes"]["standard"]),
                             vars={}, _manifest_explicit_vars={}, _manifest_status_vars={},
                             on_saved=on_saved, destroyed=False)
    dialog.status = tk.StringVar(interpreter)
    dialog.destroy = lambda: setattr(dialog, "destroyed", True)
    dialog._coerce = lambda spec: gui.RunModeEditDialog._coerce(dialog, spec)
    for _, specs in gui._SECTIONS:
        for _, path, kind, default, _, _ in specs:
            value = gui._get_path(dialog.mode, path, default)
            if kind == "bool":
                variable = tk.BooleanVar(interpreter, value=value)
            else:
                if kind.startswith("list"):
                    value = ", ".join(str(item) for item in value)
                variable = tk.StringVar(interpreter, value=str(value if value is not None else ""))
            dialog.vars[path] = variable
            if kind == "optional_manifest":
                dialog._manifest_explicit_vars[path] = tk.BooleanVar(
                    interpreter, value=gui._get_path(dialog.mode, path, None) is not None)
                dialog._manifest_status_vars[path] = tk.StringVar(interpreter)
    return dialog


@pytest.mark.parametrize("value", [None, "", "~/private/overlay_manifest.json", "relative/overlay.json"])
def test_gui_save_close_fresh_process_preserves_manifest_intent(tmp_path, monkeypatch, value):
    config = rm.default_run_modes_config()
    family = config["modes"]["standard"]["build"]["hailo"]["compute_by_family"]["hailo8"]
    family["device"] = "gpu"
    if value is not None:
        family["dependency_manifest"] = value
    path = tmp_path / "run_modes.yaml"
    def save(result):
        rm.save_run_modes_config(path, result, expected_revision="")
    dialog = _dialog(config, save)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *_a, **_k: pytest.fail("GUI save rejected"))
    gui.RunModeEditDialog._save(dialog)
    assert dialog.destroyed
    program = """
import json, sys
sys.path.insert(0, sys.argv[1])
from onnx_splitpoint_tool.run_modes import load_run_modes_config
from onnx_splitpoint_tool.hailo_compiler_context import resolve_compute_selection
c = load_run_modes_config(sys.argv[2])
f = c['modes']['standard']['build']['hailo']['compute_by_family']
print(json.dumps({'family':f['hailo8'], 'selection':resolve_compute_selection('hailo8',compute_by_family=f,env={'ONNX_SPLITPOINT_HAILO8_DEPENDENCY_MANIFEST':'/old/env.json'})}))
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program,
                             str(Path(__file__).parents[1]), str(path)],
                            capture_output=True, text=True, timeout=30, check=True)
    saved = json.loads(result.stdout)
    assert ("dependency_manifest" in saved["family"]) is (value is not None)
    expected = "/old/env.json" if value is None else value
    assert saved["selection"]["dependency_manifest"] == expected
    assert saved["selection"]["dependency_manifest_source"] == (
        "environment:ONNX_SPLITPOINT_HAILO8_DEPENDENCY_MANIFEST" if value is None
        else "compute_by_family.hailo8")


def test_gui_explicit_empty_and_return_to_environment_are_distinct(monkeypatch):
    saved = []
    dialog = _dialog(rm.default_run_modes_config(), lambda value: saved.append(copy.deepcopy(value)))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *_a, **_k: pytest.fail("save rejected"))
    dialog._manifest_explicit_vars[MANIFEST_PATH].set(True)
    gui.RunModeEditDialog._save(dialog)
    assert gui._get_path(saved[-1]["modes"]["standard"], MANIFEST_PATH, None) == ""
    reopened = _dialog(saved[-1], lambda value: saved.append(copy.deepcopy(value)))
    reopened._manifest_explicit_vars[MANIFEST_PATH].set(False)
    gui.RunModeEditDialog._save(reopened)
    assert gui._get_path(saved[-1]["modes"]["standard"], MANIFEST_PATH, None) is None


def test_gui_stale_save_keeps_editor_open_and_current_registry(tmp_path, monkeypatch):
    path = tmp_path / "run_modes.yaml"
    config = rm.default_run_modes_config()
    rm.save_run_modes_config(path, config)
    baseline = rm.load_run_modes_config(path)
    revision = rm.run_modes_revision(baseline)
    current = copy.deepcopy(baseline)
    gui._set_path(current["modes"]["standard"], MANIFEST_PATH, "/newer/overlay.json")
    rm.save_run_modes_config(path, current, expected_revision=revision)
    before = path.read_bytes()
    errors = []
    def save(value):
        rm.save_run_modes_config(path, value, expected_revision=revision, baseline=baseline)
    dialog = _dialog(baseline, save)
    dialog._manifest_explicit_vars[MANIFEST_PATH].set(True)
    dialog.vars[MANIFEST_PATH].set("/stale/overlay.json")
    monkeypatch.setattr(gui.messagebox, "showerror", lambda _title, message, **_kw: errors.append(message))
    gui.RunModeEditDialog._save(dialog)
    assert not dialog.destroyed and errors
    assert path.read_bytes() == before


def test_readonly_validation_uses_actual_overlay_and_preserves_environment(h8):
    manifest = stage(h8)
    config = rm.default_run_modes_config()["modes"]["standard"]
    gui._set_path(config, "build.hailo.compute_by_family.hailo8.device", "gpu")
    gui._set_path(config, MANIFEST_PATH, str(manifest))
    originals = {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()}
    before = dict(os.environ)
    result = gui.validate_hailo8_overlay_selection(config, selected_python=h8[0] / "bin/python", env={})
    assert result["status"] == "metadata_valid"
    assert result["gpu_compute_test"] == "not_run"
    assert result["selected_python"] == str(h8[0] / "bin/python")
    assert dict(os.environ) == before
    assert originals == {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()}


def test_cpu_validation_ignores_saved_missing_manifest_without_venv_probe(monkeypatch):
    config = rm.default_run_modes_config()["modes"]["standard"]
    gui._set_path(config, MANIFEST_PATH, "/missing/overlay.json")
    monkeypatch.setattr(hp, "validated_overlay_components", lambda **_kw: pytest.fail("CPU inspected overlay"))
    result = gui.validate_hailo8_overlay_selection(config, env={})
    assert result["status"] == "not_used_for_cpu"


def test_invalid_overlay_fails_readonly_validation_and_does_not_mutate_config(tmp_path):
    config = rm.default_run_modes_config()["modes"]["standard"]
    gui._set_path(config, "build.hailo.compute_by_family.hailo8.device", "gpu")
    gui._set_path(config, MANIFEST_PATH, str(tmp_path / "missing.json"))
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="manifest_invalid"):
        gui.validate_hailo8_overlay_selection(config, selected_python=tmp_path / "venv/bin/python", env={})
    assert config == before


def test_browse_selects_path_without_install_or_validation(monkeypatch):
    dialog = _dialog(rm.default_run_modes_config(), lambda value: None)
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **_kw: "/existing/overlay.json")
    gui.RunModeEditDialog._browse_manifest(dialog, MANIFEST_PATH)
    assert dialog.vars[MANIFEST_PATH].get() == "/existing/overlay.json"
    assert dialog._manifest_explicit_vars[MANIFEST_PATH].get()


def test_overlay_summary_is_visible_in_mode_and_profile_brief():
    config = rm.default_run_modes_config()
    gui._set_path(config["modes"]["standard"], MANIFEST_PATH, "/selected/overlay.json")
    profile, _ = rm.apply_run_mode({}, mode_id="standard", config=config)
    for text in (rm.mode_summary("standard", config), rm.run_mode_profile_brief(profile)):
        assert "/selected/overlay.json" in text
        assert "compute_by_family.hailo8" in text


@pytest.mark.parametrize("value", [None, "", "/user/overlay_manifest.json"])
def test_real_profile_editor_bound_snapshot_roundtrip_keeps_manifest(monkeypatch, value):
    from test_profile_editor_campaign_roundtrip_v261e import _headless_editor, FINAL_PROFILE
    source = yaml.safe_load(FINAL_PROFILE.read_text())
    mode = rm.default_run_modes_config()["modes"]["final"]
    mode["build"]["hailo"]["compute_by_family"]["hailo8"]["device"] = "gpu"
    if value is not None:
        gui._set_path(mode, MANIFEST_PATH, value)
    source["execution_preset"] = {"id": "final", "follow_tool_config": False, "snapshot": mode}
    source["hailo_build"] = copy.deepcopy(mode["build"]["hailo"])
    original = copy.deepcopy(source)
    editor = _headless_editor(monkeypatch)
    editor._apply_payload(source, path=str(FINAL_PROFILE))
    result = editor._build_payload()
    assert source == original
    for family in (result["hailo_build"]["compute_by_family"]["hailo8"],
                   result["execution_preset"]["snapshot"]["build"]["hailo"]["compute_by_family"]["hailo8"]):
        assert ("dependency_manifest" in family) is (value is not None)
        if value is not None:
            assert family["dependency_manifest"] == value
