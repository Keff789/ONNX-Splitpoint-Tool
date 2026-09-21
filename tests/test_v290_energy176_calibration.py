"""Real Tk/config roundtrip; external pytest temp only, no device operations."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk

import pytest

from onnx_splitpoint_tool.energy import config
from onnx_splitpoint_tool.energy.comparison import verify_accelerator_idle_calibration_binding
from onnx_splitpoint_tool.gui.panels import panel_hardware as panel
from test_v27912_accelerator_env_registry_merge import _merge_energy_only
from test_v27914_simple_idle_calibration_comparison import _write_simple_evidence


def registry(tmp_path):
    value = 0.9667883102777299
    evidence, _ = _write_simple_evidence(tmp_path, accelerator_idle_power_w=value,
        m2_off={"avg_power_w": 11.50135158092831}, m2_on={"avg_power_w": 12.46813989120604})
    row = {"id": "orin_nx_hailo8_01", "accelerator": "hailo8",
        "host": {"address": "192.0.2.20", "user": "nx", "port": 22},
        "energy": {"enabled": True, "urecs_address": "192.0.2.10",
            "accelerator_idle_w": value, "accelerator_idle_calibration_evidence": str(evidence),
            "accelerator_idle_calibrated_at": "2026-09-03T12:02:00+00:00"}}
    path = tmp_path / 'hardware.yaml'
    config.save_hardware_registry({"hardware_setups": [row]}, path)
    return path, row


def merge(row, value):
    panel._merge_accelerator_env_card_fields(row, accelerator=row['accelerator'],
        host_address=row['host']['address'], host_user='nx', host_port=22,
        remote_base_dir='~/splitpoint_runs', remote_venv='venv', provider='hailo8',
        energy_enabled=True, urecs_address=row['energy']['urecs_address'],
        idle_baseline_w=None, accelerator_idle_w=value)


def test_real_tk_platform_refresh_and_config_roundtrip(tmp_path, monkeypatch):
    path, row = registry(tmp_path)
    root = tk.Tk(); root.withdraw()
    try:
        app = SimpleNamespace(root=root, _hardware_setups_path=lambda: path)
        var = panel._str_var(app, 'var_hwsetup_orin_nx_hailo8_01_accel_idle_w', str(row['energy']['accelerator_idle_w']))
        # Exclude only the scheduled network probe, not the real GUI/config code.
        monkeypatch.setattr(panel, '_schedule_initial_platform_power_refresh', lambda *a: None)
        panel._build_platform_power_ui(ttk.Frame(root), app)
        app._platform_power_reload_cards_callback()
        loaded = config.load_hardware_registry(path)
        target = next(x for x in loaded['hardware_setups'] if x['id'] == row['id'])
        merge(target, float(var.get()))
        config.save_hardware_registry(loaded, path)
        setup = config.get_setup_energy(row['id'], path)
        assert setup.accelerator_idle_w == row['energy']['accelerator_idle_w']
        assert verify_accelerator_idle_calibration_binding(setup)['accelerator_idle_calibration_verified']
        snapshot = tmp_path / 'snapshot.json'
        snapshot.write_text(json.dumps(config.load_hardware_registry(path)))
        assert config.get_setup_energy(row['id'], snapshot).accelerator_idle_w == setup.accelerator_idle_w
    finally:
        root.destroy()


@pytest.mark.parametrize('value', [1.5, None, float('nan')])
def test_changed_or_invalid_value_cannot_retain_calibration_binding(tmp_path, value):
    path, row = registry(tmp_path)
    merge(row, value)
    assert not row['energy'].get('accelerator_idle_calibration_evidence')
    assert not row['energy'].get('accelerator_idle_calibrated_at')
    config.save_hardware_registry({'hardware_setups': [row]}, path)
    assert not verify_accelerator_idle_calibration_binding(config.get_setup_energy(row['id'], path))['accelerator_idle_calibration_verified']


@pytest.mark.parametrize('damage', ['setup', 'source', 'evidence'])
def test_foreign_binding_stays_rejected(tmp_path, damage):
    path, row = registry(tmp_path)
    setup = config.get_setup_energy(row['id'], path)
    if damage == 'setup': setup.setup_id = 'foreign'
    elif damage == 'source': setup.urecs_address = '192.0.2.99'
    else: setup.accelerator_idle_calibration_evidence = str(tmp_path / 'absent.json')
    assert not verify_accelerator_idle_calibration_binding(setup)['accelerator_idle_calibration_verified']
