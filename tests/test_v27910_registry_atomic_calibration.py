from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from onnx_splitpoint_tool import platform_power as pp
from onnx_splitpoint_tool.energy import config as energy_config


def _setup(registry: dict, setup_id: str) -> dict:
    return next(
        row
        for row in registry["hardware_setups"]
        if isinstance(row, dict) and row.get("id") == setup_id
    )


def _registry(*, accelerator_idle_w: float = 0.75) -> dict:
    return {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "operator_note": "initial",
        "energy_defaults": {
            "mode": "fast_firmware",
            "run_count": 3,
        },
        "hardware_setups": [
            {
                "id": "orin_nx_hailo8_01",
                "label": "Jetson + Hailo-8",
                "accelerator": "hailo8",
                "host": {
                    "address": "192.0.2.20",
                    "user": "nx",
                    "port": 22,
                },
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.10",
                    "accelerator_idle_w": accelerator_idle_w,
                },
                "power_control": {
                    "enabled": True,
                    "udp_port": 3000,
                    "boot_timeout_s": 240,
                },
                "tags": ["initial"],
            },
            {
                "id": "lab_peer",
                "label": "Peer setup",
                "accelerator": "deepx_m1",
                "host": {"address": "192.0.2.30", "user": "nx", "port": 22},
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.11",
                    "accelerator_idle_w": 2.0,
                },
            },
        ],
    }


def test_hardware_registry_save_uses_sibling_replace_and_preserves_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "hardware_setups.yaml"
    path.write_text("sentinel: old\n", encoding="utf-8")
    path.chmod(0o640)

    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(source, destination) -> None:
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(energy_config.os, "replace", recording_replace)
    energy_config.save_hardware_registry(_registry(), path)

    assert len(calls) == 1
    temporary, destination = calls[0]
    assert destination == path
    assert temporary.parent == path.parent
    assert temporary != path
    assert not temporary.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o640

    stored = energy_config.load_hardware_registry(path)
    target = _setup(stored, "orin_nx_hailo8_01")
    assert target["energy"]["accelerator_idle_w"] == pytest.approx(0.75)
    assert target["power_control"]["boot_timeout_s"] == pytest.approx(240)
    assert stored["energy_defaults"]["mode"] == "fast_firmware"


def test_failed_atomic_replace_leaves_original_bytes_and_mode_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "hardware_setups.yaml"
    energy_config.save_hardware_registry(_registry(accelerator_idle_w=0.75), path)
    path.chmod(0o600)
    original = path.read_bytes()

    def fail_replace(_source, _destination) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(energy_config.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        energy_config.save_hardware_registry(_registry(accelerator_idle_w=9.5), path)

    assert path.read_bytes() == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert list(tmp_path.glob(".hardware_setups.yaml.*.tmp")) == []
    stored = energy_config.load_hardware_registry(path)
    assert _setup(stored, "orin_nx_hailo8_01")["energy"][
        "accelerator_idle_w"
    ] == pytest.approx(0.75)


def test_calibration_commit_merges_into_fresh_registry_without_lost_updates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hardware_setups.yaml"
    energy_config.save_hardware_registry(_registry(), path)

    # This is the snapshot held by a calibration that has already started.
    stale_snapshot = energy_config.load_hardware_registry(path)
    expected_projection = pp._calibration_configuration_projection(
        stale_snapshot, "orin_nx_hailo8_01"
    )

    # Simulate unrelated Tool Config changes while both power captures run.
    current = energy_config.load_hardware_registry(path)
    current["operator_note"] = "changed while calibration was running"
    target = _setup(current, "orin_nx_hailo8_01")
    target["tags"] = ["edited-during-calibration"]
    peer = _setup(current, "lab_peer")
    peer["energy"]["accelerator_idle_w"] = 4.25
    peer["host"]["address"] = "203.0.113.88"
    energy_config.save_hardware_registry(current, path)

    pp._save_accelerator_idle_power(
        stale_snapshot,
        "orin_nx_hailo8_01",
        1.60,
        registry_path=path,
        calibration_record={
            "finished_at": "2026-09-02T12:00:00+00:00",
            "output_dir": "/tmp/calibration-evidence",
        },
        expected_configuration_projection=expected_projection,
    )

    stored = energy_config.load_hardware_registry(path)
    assert stored["operator_note"] == "changed while calibration was running"
    assert stored["energy_defaults"]["mode"] == "fast_firmware"
    assert stored["energy_defaults"]["run_count"] == 3

    target = _setup(stored, "orin_nx_hailo8_01")
    assert target["host"]["address"] == "192.0.2.20"
    assert target["energy"]["enabled"] is True
    assert target["power_control"]["boot_timeout_s"] == pytest.approx(240)
    assert target["tags"] == ["edited-during-calibration"]
    assert target["energy"]["accelerator_idle_w"] == pytest.approx(1.60)
    assert (
        target["energy"]["accelerator_idle_calibrated_at"]
        == "2026-09-02T12:00:00+00:00"
    )
    assert (
        target["energy"]["accelerator_idle_calibration_evidence"]
        == "/tmp/calibration-evidence"
    )

    peer = _setup(stored, "lab_peer")
    assert peer["energy"]["accelerator_idle_w"] == pytest.approx(4.25)
    assert peer["host"]["address"] == "203.0.113.88"


def test_calibration_commit_rejects_relevant_configuration_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hardware_setups.yaml"
    energy_config.save_hardware_registry(_registry(accelerator_idle_w=0.75), path)
    start = energy_config.load_hardware_registry(path)
    expected_projection = pp._calibration_configuration_projection(
        start, "orin_nx_hailo8_01"
    )

    changed = energy_config.load_hardware_registry(path)
    _setup(changed, "orin_nx_hailo8_01")["host"]["address"] = (
        "203.0.113.77"
    )
    energy_config.save_hardware_registry(changed, path)

    with pytest.raises(
        pp.PlatformStateError,
        match="configuration_changed_during_calibration",
    ):
        pp._save_accelerator_idle_power(
            start,
            "orin_nx_hailo8_01",
            1.60,
            registry_path=path,
            calibration_record={
                "finished_at": "2026-09-02T12:00:00+00:00",
                "output_dir": "/tmp/calibration-evidence",
                "binding_path": "/tmp/new-binding.json",
                "binding_sha256": "a" * 64,
            },
            expected_configuration_projection=expected_projection,
        )

    stored = energy_config.load_hardware_registry(path)
    target = _setup(stored, "orin_nx_hailo8_01")
    assert target["host"]["address"] == "203.0.113.77"
    assert target["energy"]["accelerator_idle_w"] == pytest.approx(0.75)
    assert "accelerator_idle_calibration_binding_path" not in target["energy"]


@pytest.mark.parametrize("source", ["host", "remote"])
def test_calibration_commit_rejects_ssh_extra_args_change_without_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    path = tmp_path / "hardware_setups.yaml"
    initial = _registry(accelerator_idle_w=0.75)
    target = _setup(initial, "orin_nx_hailo8_01")
    target.setdefault(source, {})["ssh_extra_args"] = "-o BatchMode=yes"
    energy_config.save_hardware_registry(initial, path)
    start = energy_config.load_hardware_registry(path)
    expected_projection = pp._calibration_configuration_projection(
        start, "orin_nx_hailo8_01"
    )

    changed = energy_config.load_hardware_registry(path)
    changed_target = _setup(changed, "orin_nx_hailo8_01")
    changed_target[source]["ssh_extra_args"] = (
        "-o BatchMode=yes -o ConnectTimeout=9"
    )
    energy_config.save_hardware_registry(changed, path)
    writes: list[Path] = []
    real_writer = energy_config._write_hardware_registry_atomic

    def record_write(registry, destination):
        writes.append(Path(destination))
        return real_writer(registry, destination)

    monkeypatch.setattr(
        energy_config, "_write_hardware_registry_atomic", record_write
    )
    with pytest.raises(
        pp.PlatformStateError,
        match="configuration_changed_during_calibration",
    ):
        pp._save_accelerator_idle_power(
            start,
            "orin_nx_hailo8_01",
            1.60,
            registry_path=path,
            calibration_record={
                "finished_at": "2026-09-02T12:00:00+00:00",
                "output_dir": "/tmp/calibration-evidence",
                "binding_path": "/tmp/new-binding.json",
                "binding_sha256": "a" * 64,
            },
            expected_configuration_projection=expected_projection,
        )

    assert writes == []
    stored = energy_config.load_hardware_registry(path)
    stored_target = _setup(stored, "orin_nx_hailo8_01")
    assert stored_target["energy"]["accelerator_idle_w"] == pytest.approx(0.75)
    assert stored_target[source]["ssh_extra_args"].endswith("ConnectTimeout=9")
    assert "accelerator_idle_calibration_binding_path" not in stored_target["energy"]
