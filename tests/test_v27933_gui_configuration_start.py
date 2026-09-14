"""Synthetic configuration/UI cases; no hardware, no historical-cause claim."""
from __future__ import annotations

import copy
import importlib
import tkinter as tk
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from onnx_splitpoint_tool.gui import run_mode_editor as editor_module
from onnx_splitpoint_tool.gui.panels import panel_evaluation_workflow as panel
from onnx_splitpoint_tool.run_modes import (
    apply_run_mode, default_run_modes_config, load_run_modes_config,
    run_modes_revision, save_run_modes_config,
)


class Value:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


def _panel_state(path: Path):
    loaded = load_run_modes_config(path)
    state = SimpleNamespace(
        path=path, config=loaded, _baseline=copy.deepcopy(loaded),
        _loaded_revision=run_modes_revision(loaded), app=None,
        mode_id=Value("final"), _refresh=lambda: None,
    )
    for method in ("_persist", "_reload", "_edit"):
        setattr(state, method, MethodType(getattr(editor_module.RunModesPanel, method), state))
    return state


def _registry(path: Path, *, force=True):
    cfg = default_run_modes_config()
    cfg["modes"]["final"]["build"]["hailo"]["force_build"] = force
    cfg["modes"]["final"]["build"]["deepx"]["force_build"] = force
    save_run_modes_config(path, cfg)
    return cfg


@pytest.mark.parametrize("bad", ["false", "true", None, 0, 1, [], "unexpected"])
def test_t33_10_gui_bool_boundary_rejects_non_boolean_with_field(bad):
    spec = next(spec for _, specs in editor_module._SECTIONS for spec in specs
                if spec[1] == "build.hailo.force_build")
    editor = object.__new__(editor_module.RunModeEditDialog)
    editor.mode_id = "final"
    editor.vars = {spec[1]: Value(bad)}
    with pytest.raises(ValueError, match=r"config_boolean_invalid:modes.final.build.hailo.force_build"):
        editor._coerce(spec)


@pytest.mark.parametrize("value", [False, True])
def test_t33_10_real_tcl_boolean_variable_preserves_value(value):
    spec = next(spec for _, specs in editor_module._SECTIONS for spec in specs
                if spec[1] == "build.deepx.force_build")
    editor = object.__new__(editor_module.RunModeEditDialog)
    editor.mode_id = "final"
    editor.vars = {spec[1]: tk.BooleanVar(tk.Tcl(), value=value)}
    assert editor._coerce(spec) is value


def test_t33_10_invalid_registry_rejected_before_window_creation(monkeypatch):
    cfg = default_run_modes_config()
    cfg["modes"]["final"]["build"]["hailo"]["force_build"] = "false"
    monkeypatch.setattr(tk.Toplevel, "__init__", lambda *a, **kw: pytest.fail("window built before validation"))
    with pytest.raises(ValueError, match=r"config_boolean_invalid:modes.final.build.hailo.force_build"):
        editor_module.RunModeEditDialog(None, mode_id="final", config=cfg, on_saved=lambda cfg: None)


def test_t33_11_13_stale_panel_conflict_reloads_and_preserves_foreign_values(tmp_path, monkeypatch):
    path = tmp_path / "run_modes.yaml"
    _registry(path)
    state = _panel_state(path)
    newer = load_run_modes_config(path)
    newer["modes"]["final"]["build"]["hailo"]["force_build"] = False
    newer["modes"]["custom-night"] = {"user_note": "retain exact user extension"}
    newer["profile_references"] = {"night": "/user/night.yaml"}
    save_run_modes_config(path, newer)
    before = path.read_bytes()
    messages = []
    monkeypatch.setattr(editor_module.messagebox, "askyesno", lambda title, text, **kw: messages.append(text) or True)
    assert state._persist(state.config) is False
    assert path.read_bytes() == before
    assert "modes.final.build.hailo.force_build" in messages[0]
    assert "neu laden" in messages[0]
    assert state.config["modes"]["final"]["build"]["hailo"]["force_build"] is False
    assert state.config["modes"]["custom-night"]["user_note"] == "retain exact user extension"
    assert state.config["profile_references"]["night"] == "/user/night.yaml"
    assert state._loaded_revision == run_modes_revision(load_run_modes_config(path))


def test_t33_15_successful_panel_save_updates_loaded_basis(tmp_path):
    path = tmp_path / "run_modes.yaml"
    _registry(path)
    state = _panel_state(path)
    pending = copy.deepcopy(state.config)
    pending["modes"]["final"]["build"]["hailo"]["force_build"] = False
    old_revision = state._loaded_revision
    assert state._persist(pending) is True
    current = load_run_modes_config(path)
    assert state._baseline == current == state.config
    assert state._loaded_revision == run_modes_revision(current) != old_revision
    pending = copy.deepcopy(state.config)
    pending["modes"]["final"]["build"]["deepx"]["force_build"] = False
    assert state._persist(pending) is True


def test_t33_11_dialog_keeps_its_revision_when_parent_reload_occurs(tmp_path, monkeypatch):
    path = tmp_path / "run_modes.yaml"
    _registry(path)
    state = _panel_state(path)
    captured = {}
    monkeypatch.setattr(editor_module, "RunModeEditDialog", lambda master, **kw: captured.update(kw))
    monkeypatch.setattr(editor_module.messagebox, "askyesno", lambda *a, **kw: False)
    state._edit()
    newer = load_run_modes_config(path)
    newer["modes"]["final"]["build"]["hailo"]["force_build"] = False
    save_run_modes_config(path, newer)
    state._reload()
    before = path.read_bytes()
    assert captured["on_saved"](captured["config"]) is False
    assert path.read_bytes() == before


def _force_profile(tmp_path, *, follow=True):
    path = tmp_path / "run_modes.yaml"
    config = _registry(path)
    source = {
        "name": "synthetic_force_gui", "deepx_build": {"classification_preprocessing": "current_scale_only"},
        "model_suite": {"primary": [{"id": "resnet50", "task": "classification", "enabled": True}]},
        "run_profiles": [{"id": "hailo8", "enabled": True}],
        "execution_preset": {"id": "final", "follow_tool_config": follow,
                             "config_path": str(path), "snapshot": config["modes"]["final"],
                             "overrides": {"native_enabled": False, "energy_enabled": False}},
    }
    resolved, _ = apply_run_mode(source, config=config, config_path=path)
    return source, resolved


@pytest.mark.parametrize("follow", [False, True])
def test_t33_16_17_summary_shows_effective_force_source_and_explicit_legacy(tmp_path, monkeypatch, follow):
    source, resolved = _force_profile(tmp_path, follow=follow)
    monkeypatch.setattr(panel, "load_evaluation_profile", lambda *a, **kw: SimpleNamespace(
        raw_profile=resolved, source_profile=source, profile_id=source["name"], profile_path="/synthetic/profile.yaml", source="file"))
    lines, visible = panel._profile_summary_payload("synthetic_force_gui")
    text = "\n".join(lines)
    assert visible, text
    assert "Hailo Force: AN – kompatible Cachetreffer" in text
    assert "DeepX Force: AN – kompatible Cachetreffer" in text
    assert "Hailo-Integrität: relaxed" in text
    assert "DeepX Classification: current_scale_only (Quelle: Evaluationsprofil)" in text
    if follow:
        assert f"Wertequelle: Tool Config / final / {tmp_path / 'run_modes.yaml'}" in text
    else:
        assert "Wertequelle: gebundener Profilsnapshot / final" in text
    assert "strict final freeze" not in text


def _options(tmp_path, profile):
    return SimpleNamespace(
        profile="synthetic.yaml", out=str(tmp_path / "not-created-until-approved"), run_id="synthetic",
        resume=False, force_stage=[], models_root="", skip_benchmarks=True, no_remote=True,
        profile_start_snapshot={"resolved_profile": profile}, hailo_force_build=False,
        force_build_confirmed_backends=("hailo", "deepx"), force_build_confirmation_source="old_gui_click",
    )


def test_t33_18_rejected_actual_queue_callback_allocates_no_job_or_compiler(tmp_path, monkeypatch):
    # v34 productive starts reject Force outright; the unchanged v33 baseline
    # separately verified its historical consent dialogue.
    import matplotlib
    monkeypatch.setattr(matplotlib, "use", lambda *a, **kw: None)
    gui = importlib.import_module("onnx_splitpoint_tool.gui.app")
    _, profile = _force_profile(tmp_path)
    opts = _options(tmp_path, profile)
    app = SimpleNamespace(_eval_workflow_snapshot_options=lambda **kw: opts, var_eval_workflow_status=Value(""))
    monkeypatch.setattr(gui, "build_effective_execution_plan", lambda _: {})
    calls = []
    monkeypatch.setattr(panel.messagebox, "askyesno", lambda *a, **kw: pytest.fail("obsolete Force confirmation offered"))
    monkeypatch.setattr(panel.messagebox, "showerror", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(gui, "_new_evaluation_workflow_job_id", lambda: pytest.fail("job allocated after declined Force"))
    monkeypatch.setattr(gui, "EvaluationWorkflowRunner", lambda *a, **kw: pytest.fail("runner dispatched after declined Force"))
    assert gui.SplitPointAnalyserGUI._queue_evaluation_workflow(app) is None
    assert app.var_eval_workflow_status.get() == "Start blockiert: Force muss AUS sein."
    assert not Path(opts.out).exists()
    assert opts.force_build_confirmed_backends == ()
    assert "hailo_build.force_build" in calls[0][1]
    assert "deepx_build.force_build" in calls[0][1]


def test_t33_18_confirmation_grants_only_requested_backend_and_never_replays(tmp_path, monkeypatch):
    opts = _options(tmp_path, {"hailo_build": {"force_build": False}, "deepx_build": {"force_build": True}})
    answers = iter([True, False])
    monkeypatch.setattr(panel.messagebox, "askyesno", lambda *a, **kw: next(answers))
    assert panel.confirm_force_build_start(opts, parent=None) is True
    assert opts.force_build_confirmed_backends == ("deepx",)
    assert opts.force_build_confirmation_source == "gui_confirmation"
    assert panel.confirm_force_build_start(opts, parent=None) is False
    assert opts.force_build_confirmed_backends == ()


def test_t33_20_resume_confirms_archived_force_not_current_profile(tmp_path, monkeypatch):
    opts = _options(tmp_path, {"hailo_build": {"force_build": False}})
    opts.resume = True
    archived = tmp_path / "archived_profile.yaml"
    archived.write_text("hailo_build:\n  force_build: true\n", encoding="utf-8")
    before = archived.read_bytes()
    text = []
    monkeypatch.setattr(panel.messagebox, "askyesno", lambda title, message, **kw: text.append(message) or False)
    assert panel.confirm_force_build_start(opts, parent=None, preview=lambda _: {
        "backends": ("hailo",), "profile_source": "archived_profile_snapshot", "profile_path": str(archived),
    }) is False
    assert "Resume mit dem gebundenen Profilsnapshot" in text[0]
    assert str(archived) in text[0]
    assert "Force AN: Hailo" in text[0]
    assert archived.read_bytes() == before


def test_t33_21_22_real_editor_state_and_layout_keep_legacy_notice_separate(monkeypatch):
    from test_profile_editor_campaign_roundtrip_v261e import _headless_editor
    from onnx_splitpoint_tool.gui import profile_editor
    editor = _headless_editor(monkeypatch)
    editor.var_deepx_classification_preprocessing.set("current_scale_only")
    class Widget:
        def __init__(self, parent=None, **kwargs):
            self.parent, self.options, self.grid_options = parent, kwargs, {}
        def grid(self, **kwargs):
            self.grid_options = kwargs
        def columnconfigure(self, *args, **kwargs):
            pass
    for cls in ("LabelFrame", "Label", "Spinbox", "Combobox", "Entry", "Checkbutton"):
        monkeypatch.setattr(profile_editor.ttk, cls, Widget)
    monkeypatch.setattr(profile_editor, "attach_tooltip", lambda *a, **kw: None)
    editor._build_targets_tab_simple(Widget())
    notice = editor.deepx_legacy_notice_label
    explanation = editor.hardware_profile_explanation_label
    assert notice.parent is explanation.parent
    assert notice.grid_options["row"] != explanation.grid_options["row"]
    assert notice.options["textvariable"].get().startswith("Legacy/A-B-Diagnose: current_scale_only")
    assert "Remote Setup" in explanation.options["text"]
    assert editor.var_deepx_classification_preprocessing.get() == "current_scale_only"


def test_t33_17_profile_editor_roundtrip_keeps_bound_snapshot(tmp_path, monkeypatch):
    from test_profile_editor_campaign_roundtrip_v261e import _headless_editor
    from onnx_splitpoint_tool.gui import profile_editor
    source, resolved = _force_profile(tmp_path, follow=False)
    before = copy.deepcopy(resolved)
    editor = _headless_editor(monkeypatch)
    editor._apply_payload(resolved)
    monkeypatch.setattr(profile_editor, "load_run_modes_config", lambda *a, **kw: pytest.fail("snapshot editor consulted current registry"))
    rebuilt = editor._build_payload()
    assert rebuilt["execution_preset"]["follow_tool_config"] is False
    assert rebuilt["execution_preset"]["snapshot"] == resolved["execution_preset"]["snapshot"]
    assert rebuilt["hailo_build"]["force_build"] is True
    assert rebuilt["deepx_build"]["force_build"] is True
    assert rebuilt["deepx_build"]["classification_preprocessing"] == "current_scale_only"
    assert rebuilt["execution_preset"]["build_provenance"]["deepx_classification_source"] == "evaluation_profile"
    assert resolved == before


@pytest.mark.parametrize("field", ["hailo_build", "deepx_build"])
def test_t33_10_profile_editor_rejects_bad_force_before_changing_any_widget(field):
    from onnx_splitpoint_tool.gui.profile_editor import EvaluationProfileEditor
    editor = object.__new__(EvaluationProfileEditor)
    with pytest.raises(ValueError, match=f"config_boolean_invalid:{field}.force_build"):
        editor._apply_payload({field: {"force_build": "false"}})
    assert "_loading_profile" not in editor.__dict__


def test_t33_20_resume_dialog_discloses_existing_contract_mismatch(tmp_path, monkeypatch):
    opts = _options(tmp_path, {})
    opts.resume = True
    messages = []
    monkeypatch.setattr(panel.messagebox, "askyesno", lambda title, text, **kw: messages.append(text) or False)
    assert panel.confirm_force_build_start(opts, parent=None, preview=lambda _: {
        "backends": ("hailo",), "profile_source": "archived_run_snapshot",
        "profile_path": "/synthetic/old/profile.yaml", "profile_mismatch": True,
    }) is False
    assert "aktuelle Profil weicht vom archivierten Vertrag ab" in messages[0]
    assert "Die Resume-Prüfung bleibt aktiv" in messages[0]
