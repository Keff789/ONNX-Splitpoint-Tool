from __future__ import annotations

import copy
from pathlib import Path

from onnx_splitpoint_tool.gui.panels.panel_hardware import (
    _merge_accelerator_env_card_fields,
)
from onnx_splitpoint_tool.platform_power import host_config_from_setup


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "onnx_splitpoint_tool" / "gui" / "panels" / "panel_hardware.py"


def _merge_energy_only(setup: dict) -> None:
    host = host_config_from_setup(setup)
    runtime = dict(setup.get("runtime") or {})
    remote = dict(setup.get("remote") or {})
    _merge_accelerator_env_card_fields(
        setup,
        accelerator=str(setup.get("accelerator") or "hailo8"),
        host_address=host.host,
        host_user=host.user,
        host_port=host.port,
        remote_base_dir=host.remote_base_dir,
        remote_venv=str(
            runtime.get("activate")
            or runtime.get("venv")
            or remote.get("remote_venv")
            or "source ~/hailo_py/bin/activate"
        ),
        provider=str(
            runtime.get("provider") or remote.get("provider") or "hailo8"
        ),
        energy_enabled=True,
        urecs_address=str(dict(setup.get("energy") or {}).get("urecs_address") or ""),
        idle_baseline_w=7.25,
        accelerator_idle_w=1.5,
    )


def test_modern_host_hidden_ssh_and_target_fields_survive_energy_card_save() -> None:
    setup = {
        "id": "orin_nx_hailo8_01",
        "label": "Orin NX + Hailo-8",
        "accelerator": "hailo8",
        "host": {
            "address": "192.168.0.104",
            "user": "nx",
            "port": 22,
            "base_dir": "~/splitpoint_runs",
            "ssh_extra_args": "-J nx@192.168.0.2 -o IdentitiesOnly=yes",
            "target_identity": "jetson-h8-primary",
            "host_key_alias": "splitpoint-h8",
        },
        "runtime": {
            "kind": "hailort",
            "provider": "hailo8",
            "activate": "source ~/hailo_py/bin/activate",
            "venv": "source ~/hailo_py/bin/activate",
            "device_selector": "0000:01:00.0",
        },
        "remote": {
            "remote_base_dir": "~/splitpoint_runs",
            "remote_venv": "source ~/hailo_py/bin/activate",
            "provider": "hailo8",
            # If both modern host and legacy remote aliases are present they
            # must describe the same effective SSH target.  Keep an exposed
            # duplicate here so this fixture exercises preservation rather
            # than an intentionally conflicting registry.
            "ssh_extra_args": "-J nx@192.168.0.2 -o IdentitiesOnly=yes",
            "custom_target": {"rail": "slot-a", "device": "hailo0"},
        },
        "energy": {
            "enabled": False,
            "urecs_address": "192.168.0.197",
            "idle_baseline_w": 6.0,
            "accelerator_idle_w": 1.0,
            "calibration_manifest": "/evidence/fs-method.json",
            "calibration_sha256": "a" * 64,
        },
    }
    before = copy.deepcopy(setup)
    identity_before = host_config_from_setup(before).to_dict()

    _merge_energy_only(setup)

    assert setup["host"]["ssh_extra_args"] == before["host"]["ssh_extra_args"]
    assert setup["host"]["target_identity"] == "jetson-h8-primary"
    assert setup["host"]["host_key_alias"] == "splitpoint-h8"
    assert setup["remote"]["ssh_extra_args"] == before["remote"]["ssh_extra_args"]
    assert setup["remote"]["custom_target"] == {
        "rail": "slot-a",
        "device": "hailo0",
    }
    assert setup["runtime"]["device_selector"] == "0000:01:00.0"
    assert setup["energy"]["calibration_manifest"] == "/evidence/fs-method.json"
    assert setup["energy"]["calibration_sha256"] == "a" * 64
    assert setup["energy"]["idle_baseline_w"] == 7.25
    assert setup["energy"]["accelerator_idle_w"] == 1.5
    assert host_config_from_setup(setup).to_dict() == identity_before


def test_legacy_remote_hidden_ssh_fields_survive_energy_card_save() -> None:
    setup = {
        "id": "orin_nx_hailo10_01",
        "label": "Orin NX + Hailo-10",
        "accelerator": "hailo10h",
        "remote": {
            "host": "192.168.0.145",
            "user": "nx",
            "port": 2222,
            "remote_base_dir": "~/splitpoint_runs",
            "remote_venv": "source ~/venvs/hailo10/bin/activate",
            "provider": "hailo10h",
            "ssh_extra_args": "-J gateway -o HostKeyAlias=orin-h10",
            "target_id": "h10-lab-slot",
        },
        "runtime": {
            "kind": "hailort",
            "provider": "hailo10h",
            "activate": "source ~/venvs/hailo10/bin/activate",
            "compiler_target": "hailo10h",
        },
        "energy": {
            "enabled": False,
            "urecs_address": "192.168.0.176",
            "calibration_manifest": "/evidence/fs-method.json",
            "calibration_sha256": "b" * 64,
        },
    }
    before = copy.deepcopy(setup)
    identity_before = host_config_from_setup(before).to_dict()

    _merge_energy_only(setup)

    assert setup["remote"]["ssh_extra_args"] == before["remote"]["ssh_extra_args"]
    assert setup["remote"]["target_id"] == "h10-lab-slot"
    assert setup["remote"]["host"] == "192.168.0.145"
    assert setup["remote"]["port"] == 2222
    assert setup["runtime"]["compiler_target"] == "hailo10h"
    assert setup["energy"]["calibration_manifest"] == "/evidence/fs-method.json"
    assert setup["energy"]["calibration_sha256"] == "b" * 64
    assert host_config_from_setup(setup).to_dict() == identity_before


def test_remote_execution_extras_are_imported_before_modern_remote_alias() -> None:
    setup = {
        "id": "legacy-deepx",
        "accelerator": "deepx_m1",
        "remote_execution": {
            "host": "192.168.0.102",
            "user": "nx",
            "port": 22,
            "ssh_extra_args": "-o HostKeyAlias=deepx-legacy",
            "remote_base_dir": "~/legacy-runs",
            "remote_venv": "source ~/venvs/deepx-runtime/bin/activate",
            "provider": "deepx_m1",
            "target_serial": "dxm1-001",
        },
        "energy": {"urecs_address": "192.168.0.185"},
    }

    _merge_accelerator_env_card_fields(
        setup,
        accelerator="deepx_m1",
        host_address="192.168.0.102",
        host_user="nx",
        host_port=22,
        remote_base_dir="~/legacy-runs",
        remote_venv="source ~/venvs/deepx-runtime/bin/activate",
        provider="deepx_m1",
        energy_enabled=True,
        urecs_address="192.168.0.185",
        idle_baseline_w=8.0,
        accelerator_idle_w=2.0,
    )

    assert setup["remote"]["ssh_extra_args"] == "-o HostKeyAlias=deepx-legacy"
    assert setup["remote"]["target_serial"] == "dxm1-001"
    assert setup["remote_execution"]["target_serial"] == "dxm1-001"
    assert host_config_from_setup(setup).ssh_extra_args == (
        "-o HostKeyAlias=deepx-legacy"
    )


def test_host_edit_synchronizes_existing_endpoint_aliases_and_keeps_extras() -> None:
    setup = {
        "id": "orin_nx_hailo8_01",
        "accelerator": "hailo8",
        "host": {
            "address": "192.168.0.104",
            "host": "192.168.0.104",
            "user": "nx",
            "port": 22,
            "ssh_extra_args": "-o BatchMode=yes",
            "target_identity": "h8-slot",
        },
        "remote": {
            "host": "192.168.0.104",
            "address": "192.168.0.104",
            "user": "nx",
            "port": 22,
            "ssh_extra_args": "-o BatchMode=yes",
            "custom_target": {"rail": "slot-a"},
        },
        "remote_execution": {
            "host": "192.168.0.104",
            "user": "nx",
            "port": 22,
            "ssh_extra_args": "-o BatchMode=yes",
            "legacy_target": "jetson-a",
        },
        "runtime": {
            "provider": "hailo8",
            "host": "192.168.0.104",
            "user": "nx",
            "port": 22,
            "device_selector": "hailo0",
        },
        "energy": {"urecs_address": "192.168.0.197"},
    }

    _merge_accelerator_env_card_fields(
        setup,
        accelerator="hailo8",
        host_address="192.168.0.204",
        host_user="operator",
        host_port=2222,
        remote_base_dir="~/new-runs",
        remote_venv="source ~/hailo_py/bin/activate",
        provider="hailo8",
        energy_enabled=True,
        urecs_address="192.168.0.197",
        idle_baseline_w=7.0,
        accelerator_idle_w=1.25,
    )

    identity = host_config_from_setup(setup)
    assert (identity.host, identity.user, identity.port) == (
        "192.168.0.204",
        "operator",
        2222,
    )
    for owner in ("host", "remote", "remote_execution", "runtime"):
        assert setup[owner].get("host") == "192.168.0.204"
        assert setup[owner].get("user") == "operator"
        assert setup[owner].get("port") == 2222
    assert setup["host"]["target_identity"] == "h8-slot"
    assert setup["remote"]["custom_target"] == {"rail": "slot-a"}
    assert setup["remote_execution"]["legacy_target"] == "jetson-a"
    assert setup["runtime"]["device_selector"] == "hailo0"


def test_accelerator_env_save_uses_merge_helper_not_mapping_replacement() -> None:
    source = PANEL_PATH.read_text(encoding="utf-8")
    assert source.count("_merge_accelerator_env_card_fields(") >= 2
    assert 'found["host"] = {' not in source
    assert 'found["runtime"] = {' not in source
    assert 'found["remote"] = {' not in source
