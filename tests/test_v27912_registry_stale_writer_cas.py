from __future__ import annotations

from pathlib import Path
import threading

import pytest
import yaml

from onnx_splitpoint_tool.energy import config as energy_config
from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI


SETUP_ID = "orin_nx_hailo8_01"


def _setup(registry: dict, setup_id: str = SETUP_ID) -> dict:
    return next(
        row
        for row in registry.get("hardware_setups") or []
        if isinstance(row, dict) and str(row.get("id") or "") == setup_id
    )


def _registry() -> dict:
    return {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "energy_defaults": {"mode": "fast_firmware", "run_count": 3},
        "hardware_setups": [
            {
                "id": SETUP_ID,
                "label": "Orin NX + Hailo-8",
                "accelerator": "hailo8",
                "host": {
                    "address": "192.0.2.104",
                    "user": "nx",
                    "port": 22,
                },
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.197",
                    "accelerator_idle_w": 1.0,
                    "calibration_manifest": "/old/method.json",
                    "calibration_sha256": "1" * 64,
                },
                "power_control": {"enabled": True, "m2_command": "m.2"},
            }
        ],
    }


class _FakeGui:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path

    def _hardware_setups_path(self) -> Path:
        return self.registry_path

    def _hardware_registry_load(self) -> dict:
        return SplitPointAnalyserGUI._hardware_registry_load(self)


def test_stale_gui_full_registry_save_is_rejected_without_losing_calibration(
    tmp_path: Path,
) -> None:
    """A GUI snapshot must not erase a calibration committed after its load."""

    path = tmp_path / "hardware_setups.yaml"
    energy_config.save_hardware_registry(_registry(), path)
    gui = _FakeGui(path)

    stale_gui_payload = SplitPointAnalyserGUI._hardware_registry_load(gui)
    stale_revision = stale_gui_payload.get(
        energy_config.HARDWARE_REGISTRY_REVISION_KEY
    )
    assert stale_revision == energy_config.hardware_registry_file_sha256(path)

    # Simulate the atomic commit made by a calibration worker while the GUI
    # still owns a complete, now-stale registry payload.
    commit_errors: list[BaseException] = []

    def _calibration_commit() -> None:
        try:
            current = energy_config.load_hardware_registry(path)
            energy = _setup(current)["energy"]
            energy.update(
                {
                    "accelerator_idle_w": 2.375,
                    "accelerator_idle_calibrated_at": "2026-09-02T18:00:00+00:00",
                    "accelerator_idle_calibration_evidence": "/evidence/current",
                    "accelerator_idle_calibration_binding_path": "/evidence/current/binding.json",
                    "accelerator_idle_calibration_binding_sha256": "2" * 64,
                    "calibration_manifest": "/methods/current.json",
                    "calibration_sha256": "3" * 64,
                }
            )
            energy_config.save_hardware_registry(current, path)
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            commit_errors.append(exc)

    worker = threading.Thread(target=_calibration_commit)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert commit_errors == []

    with pytest.raises(
        energy_config.HardwareRegistryConflictError,
        match="stale save rejected",
    ):
        SplitPointAnalyserGUI._hardware_registry_save(gui, stale_gui_payload)

    stored = energy_config.load_hardware_registry(path)
    stored_energy = _setup(stored)["energy"]
    assert stored_energy["accelerator_idle_w"] == pytest.approx(2.375)
    assert stored_energy["accelerator_idle_calibration_binding_path"] == (
        "/evidence/current/binding.json"
    )
    assert stored_energy["accelerator_idle_calibration_binding_sha256"] == "2" * 64
    assert stored_energy["calibration_manifest"] == "/methods/current.json"
    assert stored_energy["calibration_sha256"] == "3" * 64

    # The app eagerly reloads the winning revision for the card/conflict UI.
    reloaded = gui._hardware_registry_conflict_reload
    assert _setup(reloaded)["energy"]["accelerator_idle_w"] == pytest.approx(2.375)

    fresh_gui_payload = SplitPointAnalyserGUI._hardware_registry_load(gui)
    fresh_gui_payload["operator_note"] = "guarded GUI save"
    SplitPointAnalyserGUI._hardware_registry_save(gui, fresh_gui_payload)

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["operator_note"] == "guarded GUI save"
    assert energy_config.HARDWARE_REGISTRY_REVISION_KEY not in raw


def test_energy_defaults_fresh_merge_preserves_calibration_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "hardware_setups.yaml"
    energy_path = tmp_path / "energy_config.yaml"
    current = _registry()
    current_energy = _setup(current)["energy"]
    current_energy.update(
        {
            "accelerator_idle_w": 4.5,
            "accelerator_idle_calibration_binding_path": "/evidence/binding.json",
            "accelerator_idle_calibration_binding_sha256": "4" * 64,
        }
    )
    energy_config.save_hardware_registry(current, registry_path)
    monkeypatch.setattr(energy_config, "default_registry_path", lambda: registry_path)

    energy_config.save_energy_defaults(
        energy_config.EnergyDefaults(enabled=True, run_count=7),
        energy_path,
    )

    stored = energy_config.load_hardware_registry(registry_path)
    assert stored["energy_defaults"]["enabled"] is True
    assert stored["energy_defaults"]["run_count"] == 7
    stored_energy = _setup(stored)["energy"]
    assert stored_energy["accelerator_idle_w"] == pytest.approx(4.5)
    assert stored_energy["accelerator_idle_calibration_binding_path"] == (
        "/evidence/binding.json"
    )
    assert stored_energy["accelerator_idle_calibration_binding_sha256"] == "4" * 64
