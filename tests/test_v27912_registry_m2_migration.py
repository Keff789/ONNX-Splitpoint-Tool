from __future__ import annotations

from pathlib import Path
import stat

import yaml

from onnx_splitpoint_tool.energy import config as energy_config
from onnx_splitpoint_tool.workflow import hardware_matrix


def _setup(setup_id: str, command=...) -> dict:
    power = {"udp_port": 3000}
    if command is not ...:
        power["m2_command"] = command
    return {
        "id": setup_id,
        "label": setup_id,
        "accelerator": (
            "hailo8" if "hailo8" in setup_id else
            "hailo10" if "hailo10" in setup_id else
            "deepx_m1"
        ),
        "host": {"address": "192.0.2.20", "user": "nx", "port": 22},
        "energy": {
            "enabled": True,
            "urecs_address": "192.0.2.10",
            "accelerator_idle_w": 1.25,
        },
        "power_control": power,
    }


def _registry() -> dict:
    return {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "hardware_setups": [
            _setup("orin_nx_hailo8_01", "m2"),
            _setup("orin_nx_hailo10_01"),
            _setup("orin_nx_deepx_m1_01", " custom-m2-token "),
        ],
    }


def _row(payload: dict, setup_id: str) -> dict:
    return next(
        row for row in payload["hardware_setups"]
        if row.get("id") == setup_id
    )


def test_load_persists_only_known_m2_default_migration_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hardware_setups.yaml"
    path.write_text(yaml.safe_dump(_registry(), sort_keys=False), encoding="utf-8")
    path.chmod(0o640)

    loaded = energy_config.load_hardware_registry(path)

    assert _row(loaded, "orin_nx_hailo8_01")["power_control"]["m2_command"] == "m.2"
    assert _row(loaded, "orin_nx_hailo10_01")["power_control"]["m2_command"] == "m.2"
    assert (
        _row(loaded, "orin_nx_deepx_m1_01")["power_control"]["m2_command"]
        == " custom-m2-token "
    )

    # Loading the central registry is an upgrade path, not merely an in-memory
    # compatibility shim. The same narrowly scoped values are on disk and file
    # permissions survive the sibling-file atomic replacement.
    persisted = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert _row(persisted, "orin_nx_hailo8_01")["power_control"]["m2_command"] == "m.2"
    assert _row(persisted, "orin_nx_hailo10_01")["power_control"]["m2_command"] == "m.2"
    assert (
        _row(persisted, "orin_nx_deepx_m1_01")["power_control"]["m2_command"]
        == " custom-m2-token "
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o640

    # Additive energy-method fields are available without replacing existing
    # setup energy values.
    h8_energy = _row(loaded, "orin_nx_hailo8_01")["energy"]
    assert h8_energy["accelerator_idle_w"] == 1.25
    assert h8_energy["calibration_manifest"] == ""
    assert h8_energy["calibration_sha256"] == ""


def test_save_migrates_legacy_and_preserves_custom_command(tmp_path: Path) -> None:
    path = tmp_path / "hardware_setups.yaml"
    energy_config.save_hardware_registry(_registry(), path)
    stored = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert _row(stored, "orin_nx_hailo8_01")["power_control"]["m2_command"] == "m.2"
    assert _row(stored, "orin_nx_hailo10_01")["power_control"]["m2_command"] == "m.2"
    assert (
        _row(stored, "orin_nx_deepx_m1_01")["power_control"]["m2_command"]
        == " custom-m2-token "
    )


def test_ensure_writes_correct_default_and_energy_method_keys(
    tmp_path: Path,
) -> None:
    path = hardware_matrix.ensure_hardware_setups_file(
        tmp_path / "hardware_setups.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(payload["hardware_setups"]) == 3
    for row in payload["hardware_setups"]:
        assert row["power_control"]["m2_command"] == "m.2"
        assert row["energy"]["calibration_manifest"] == ""
        assert row["energy"]["calibration_sha256"] == ""


def test_energy_setup_reads_manifest_identity_without_rewriting_custom_token(
    tmp_path: Path,
) -> None:
    payload = _registry()
    h8 = _row(payload, "orin_nx_hailo8_01")
    h8["energy"]["calibration_manifest"] = "/tmp/fs-method.json"
    h8["energy"]["calibration_sha256"] = "A" * 64
    path = tmp_path / "hardware_setups.yaml"
    energy_config.save_hardware_registry(payload, path)

    setup = energy_config.get_setup_energy("orin_nx_hailo8_01", path)
    assert setup.calibration_manifest == "/tmp/fs-method.json"
    assert setup.calibration_sha256 == "a" * 64


def test_load_inside_existing_write_lock_defers_persistence_without_blocking(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hardware_setups.yaml"
    path.write_text(yaml.safe_dump(_registry(), sort_keys=False), encoding="utf-8")

    with energy_config._hardware_registry_write_lock(path):
        loaded = energy_config.load_hardware_registry(path)
        assert _row(loaded, "orin_nx_hailo8_01")["power_control"]["m2_command"] == "m.2"
        # The outer writer owns the commit; load must not perform an unlocked
        # replacement or deadlock trying to reacquire its own lock.
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert _row(raw, "orin_nx_hailo8_01")["power_control"]["m2_command"] == "m2"

