"""Real Tk regression for bounded profile scope and Native Energy save/load."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tkinter as tk

import pytest
import yaml

from onnx_splitpoint_tool.gui import profile_editor
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("energy_enabled", [False, True])
def test_real_editor_retains_forced_scope_checkpoint_and_energy_duration(tmp_path, monkeypatch, energy_enabled):
    # Probe an owned child because stale forwarded X sockets can block Tk().
    try:
        probe = subprocess.run([sys.executable, "-B", "-c", "import tkinter as tk; r=tk.Tk(); r.destroy()"],
                               capture_output=True, text=True, timeout=6)
    except subprocess.TimeoutExpired:
        pytest.skip("real Tk display connection did not answer within 6 seconds")
    if probe.returncode:
        pytest.skip("real Tk display unavailable: " + probe.stderr.strip())
    source = yaml.safe_load((ROOT / "onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml").read_text())
    source["model_suite"]["primary"] = [source["model_suite"]["primary"][0]]
    source["selection_policy"]["forced_cases"] = {"resnet50": ["b010"]}
    source["selection_policy"]["max_accepted_cases_per_model"] = 1
    source["workflow_execution"] = {"native_release_mode": "per_case", "setup_queue_mode": "per_setup"}
    source["campaign"] = {"mode": "development"}
    mode = deepcopy(default_run_modes_config()["modes"]["standard"])
    mode["data"]["validation_items"] = {"classification": 32, "detection": 32}
    mode["quality"]["bootstrap_repetitions"] = 100
    mode["energy"].update(native_duration_s=60, repeats=3)
    mode["runtime"]["native"].update(frames=100, warmup=10, repetitions=1)
    source["execution_preset"] = {"id": "standard", "follow_tool_config": False, "snapshot": mode,
        "overrides": {"native_enabled": True, "energy_enabled": energy_enabled}}
    checkpoint = {"scope": "bounded_gui_acceptance", "required_row_count": 3}
    source["native_producers"] = {"enabled": True, "energy": {"enabled": energy_enabled, "duration_s": 60},
                                  "native_performance_checkpoint": checkpoint, "build_missing_engines": False}
    source, _ = apply_run_mode(source, follow_tool_config=False)
    original = tmp_path / "bounded.yaml"
    original.write_text(yaml.safe_dump(source))
    original_bytes = original.read_bytes()
    saved = tmp_path / "bounded_saved.yaml"
    errors = []
    monkeypatch.setattr(profile_editor.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))
    monkeypatch.setattr(profile_editor.filedialog, "asksaveasfilename", lambda **kwargs: str(saved))
    root = tk.Tk()
    root.withdraw()
    editor = None
    try:
        editor = profile_editor.EvaluationProfileEditor(root, profile_var=tk.StringVar(root, str(original)))
        editor.withdraw()
        assert editor._load_profile(str(original)), errors
        assert editor.var_native_energy_duration_s.get() == 60
        assert editor.var_native_energy_mode.get() == ("measure" if energy_enabled else "plan")
        assert editor.var_native_energy_enabled.get() is energy_enabled
        root.update()
        editor._save(use_after=False)
        assert saved.is_file() and errors == [], errors
        saved_profile, _ = load_runtime_profile_snapshot(str(saved))
        assert saved_profile["selection_policy"]["forced_cases"] == {"resnet50": ["b010"]}
        assert saved_profile["native_producers"]["native_performance_checkpoint"] == checkpoint
        assert saved_profile["native_producers"]["energy"]["duration_s"] == 60
        assert saved_profile["native_producers"]["energy"]["enabled"] is energy_enabled
        assert saved_profile["validation_execution"]["max_items"] == {"classification": 32, "detection": 32}
        assert saved_profile["quality_gate"]["statistics"]["bootstrap_repetitions"] == 100
        assert saved_profile["workflow_execution"] == source["workflow_execution"]
        assert editor._load_profile(str(saved)), errors
        assert editor.var_native_energy_duration_s.get() == 60
        assert editor.var_native_energy_mode.get() == ("measure" if energy_enabled else "plan")
        assert original.read_bytes() == original_bytes
    finally:
        if editor is not None:
            editor.destroy()
        root.destroy()
