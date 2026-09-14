from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.gui.panels.panel_hardware import (
    _fs_energy_method_badge_state,
    _platform_power_operation_error_text,
)


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "onnx_splitpoint_tool" / "gui" / "panels" / "panel_hardware.py"


def _configured_energy() -> dict[str, str]:
    return {
        "calibration_manifest": "/evidence/fs_energy_method.json",
        "calibration_sha256": "a" * 64,
    }


def _verified_source_integrity() -> dict[str, object]:
    return {"ok": True, "status": "verified", "errors": []}


def test_fs_energy_method_badge_is_conservative_and_runtime_verified() -> None:
    missing = _fs_energy_method_badge_state("setup", {}, verifier=None)
    assert missing == {
        "status": "missing",
        "text": "FS energy method: missing",
        "level": "warn",
        "detail": (
            "Missing setup energy field(s): calibration_manifest, "
            "calibration_sha256"
        ),
        "path": "",
    }

    verified = _fs_energy_method_badge_state(
        "setup",
        _configured_energy(),
        verifier=lambda setup_id: {
            "configured": True,
            "verified": setup_id == "setup",
            "status": "inherited_validated_method_verified",
            "path": "/evidence/fs_energy_method.json",
            "runtime_binding_id": "binding-setup",
            "runtime_binding_errors": [],
            "source_integrity_verification": _verified_source_integrity(),
        },
    )
    assert verified["status"] == "verified"
    assert verified["text"] == "FS energy method: verified"
    assert verified["level"] == "ok"
    assert "current runtime binding" in verified["detail"]


def test_fs_energy_method_rejects_generic_or_direct_calibration_verification() -> None:
    for generic in (
        {
            "configured": True,
            "verified": True,
            "status": "verified",
            "runtime_binding_id": "binding-present",
        },
        {
            "configured": True,
            "verified": True,
            "status": "direct_calibration_verified",
            "runtime_binding_id": "binding-present",
        },
        {
            "configured": True,
            "verified": True,
            "status": "inherited_validated_method_verified",
            "runtime_binding_id": "",
        },
    ):
        state = _fs_energy_method_badge_state(
            "setup",
            _configured_energy(),
            verifier=lambda _setup_id, value=generic: value,
        )
        assert state["status"] == "invalid"
        assert state["text"] == "FS energy method: invalid"
        assert state["level"] == "error"


@pytest.mark.parametrize("field", ["configured", "verified"])
@pytest.mark.parametrize("truthy_non_bool", ["false", 1, {"value": True}])
def test_fs_energy_method_badge_never_trusts_truthy_nonboolean_gate_fields(
    field: str,
    truthy_non_bool: object,
) -> None:
    result: dict[str, object] = {
        "configured": True,
        "verified": True,
        "status": "inherited_validated_method_verified",
        "runtime_binding_id": "binding-present",
        "source_integrity_verification": _verified_source_integrity(),
    }
    result[field] = truthy_non_bool

    state = _fs_energy_method_badge_state(
        "setup",
        _configured_energy(),
        verifier=lambda _setup_id: result,
    )

    assert state["status"] != "verified"
    assert state["level"] != "ok"


def test_fs_energy_method_badge_rejects_failed_installed_source_integrity() -> None:
    state = _fs_energy_method_badge_state(
        "setup",
        _configured_energy(),
        verifier=lambda _setup_id: {
            "configured": True,
            "verified": True,
            "status": "inherited_validated_method_verified",
            "runtime_binding_id": "binding-setup",
            "source_integrity_verification": {
                "ok": False,
                "status": "source_integrity_binding_failed",
                "errors": ["current_installed_source_integrity_failed"],
            },
        },
    )

    assert state["status"] == "invalid"
    assert state["level"] == "error"
    assert "current_installed_source_integrity_failed" in state["detail"]


@pytest.mark.parametrize("raw_data_port", ["3000", True, 0, 65536])
def test_fs_method_badge_rejects_malformed_raw_energy_data_port(
    tmp_path: Path,
    raw_data_port: object,
) -> None:
    setup_id = "orin_nx_hailo8_01"
    energy = {
        "urecs_address": "192.168.0.197",
        "calibration_manifest": str(tmp_path / "must-not-be-read.json"),
        "calibration_sha256": "c" * 64,
    }
    registry_path = tmp_path / "hardware_setups.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/hardware-setups",
                "schema_version": 2,
                "energy_defaults": {"data_port": raw_data_port},
                "hardware_setups": [
                    {
                        "id": setup_id,
                        "accelerator": "hailo8",
                        "energy": energy,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = _fs_energy_method_badge_state(
        setup_id,
        energy,
        registry_path=registry_path,
    )

    assert state["status"] == "invalid"
    assert state["text"] == "FS energy method: invalid"
    assert state["level"] == "error"
    assert "invalid_energy_defaults_data_port" in state["detail"]


@pytest.mark.parametrize(
    "mutation",
    [
        "enabled_false",
        "enabled_string",
        "scope_mb",
        "window_alias",
        "address_conflict",
        "address_alias_malformed",
    ],
)
def test_fs_method_badge_rejects_method_identity_fuzz(
    tmp_path: Path,
    mutation: str,
) -> None:
    setup_id = "orin_nx_hailo8_01"
    defaults: dict[str, object] = {
        "data_port": 3000,
        "channel": 0,
        "sample_rate": 2000,
        "physical_scope": "FS",
        "window_label": "command",
    }
    energy: dict[str, object] = {
        "enabled": True,
        "urecs_address": "192.168.0.197",
        "calibration_manifest": str(tmp_path / "must-not-be-read.json"),
        "calibration_sha256": "c" * 64,
    }
    if mutation == "enabled_false":
        energy["enabled"] = False
    elif mutation == "enabled_string":
        energy["enabled"] = "true"
    elif mutation == "scope_mb":
        defaults["physical_scope"] = "MB"
    elif mutation == "window_alias":
        defaults["window_label"] = "command_window"
    elif mutation == "address_conflict":
        energy["address"] = "192.168.0.250"
    elif mutation == "address_alias_malformed":
        energy["address"] = " 192.168.0.197"
    else:  # pragma: no cover
        raise AssertionError(mutation)

    registry_path = tmp_path / "hardware_setups.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/hardware-setups",
                "schema_version": 2,
                "energy_defaults": defaults,
                "hardware_setups": [
                    {
                        "id": setup_id,
                        "accelerator": "hailo8",
                        "energy": energy,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = _fs_energy_method_badge_state(
        setup_id,
        energy,
        registry_path=registry_path,
    )
    assert state["status"] == "invalid"
    assert state["level"] == "error"
    assert "invalid_energy_method_registry_contract" in state["detail"]


def test_fs_energy_method_badge_exposes_invalid_binding_reasons() -> None:
    invalid = _fs_energy_method_badge_state(
        "setup",
        _configured_energy(),
        verifier=lambda _setup_id: {
            "configured": True,
            "verified": False,
            "status": "runtime_binding_invalid",
            "path": "/evidence/fs_energy_method.json",
            "runtime_binding_errors": [
                "collector_binary_sha256_mismatch",
                "setup_binding_missing",
            ],
        },
    )
    assert invalid["status"] == "invalid"
    assert invalid["text"] == "FS energy method: invalid"
    assert invalid["level"] == "error"
    assert "collector_binary_sha256_mismatch" in invalid["detail"]
    assert "/evidence/fs_energy_method.json" in invalid["detail"]


def test_fs_energy_method_uses_the_selected_non_default_registry_path(
    tmp_path: Path,
) -> None:
    selected_registry = tmp_path / "custom-hardware-setups.yaml"
    captured: dict[str, object] = {}

    def verifier(setup_id: str, *, registry_path=None):
        captured["setup_id"] = setup_id
        captured["registry_path"] = registry_path
        return {
            "configured": True,
            "verified": True,
            "status": "inherited_validated_method_verified",
            "path": "/evidence/fs_energy_method.json",
            "verification": {"runtime_binding_id": "nested-binding-id"},
            "runtime_binding_errors": [],
            "source_integrity_verification": _verified_source_integrity(),
        }

    state = _fs_energy_method_badge_state(
        "custom-setup",
        _configured_energy(),
        registry_path=selected_registry,
        verifier=verifier,
    )
    assert state["status"] == "verified"
    assert captured == {
        "setup_id": "custom-setup",
        "registry_path": selected_registry,
    }

    # Older optional shims used by import-only tests accepted only setup_id.
    fallback = _fs_energy_method_badge_state(
        "custom-setup",
        _configured_energy(),
        registry_path=selected_registry,
        verifier=lambda _setup_id: {
            "configured": True,
            "verified": True,
            "status": "inherited_validated_method_verified",
            "path": "/evidence/fs_energy_method.json",
            "runtime_binding_id": "legacy-shim-binding-id",
            "source_integrity_verification": _verified_source_integrity(),
        },
    )
    assert fallback["status"] == "verified"


def test_fs_energy_method_never_calls_configuration_only_verified() -> None:
    state = _fs_energy_method_badge_state(
        "setup",
        {
            "calibration_manifest": "/does/not/exist.json",
            "calibration_sha256": "b" * 64,
        },
        verifier=None,
    )
    assert state["status"] == "invalid"
    assert state["level"] == "error"
    assert "readable file" in state["detail"]


def test_operation_error_dialog_text_preserves_gate_reasons_and_evidence() -> None:
    class GateError(RuntimeError):
        pass

    error = GateError("u.RECS measurement returned no average power")
    error.evidence_path = "/evidence/calibration/m2_idle_power_calibration.json"
    error.result = {
        "status": "final_energy_gate_failed",
        "final_energy_gate_failures": [
            {
                "run_index": 0,
                "reasons": ["full_system_calibration_not_locally_verified"],
            }
        ],
        "output_dir": "/evidence/calibration/m2_off",
        "energy_calibration_verification": {
            "runtime_binding_errors": ["setup_binding_missing"]
        },
    }
    text = _platform_power_operation_error_text(error)
    assert "GateError: u.RECS measurement returned no average power" in text
    assert (
        "Final energy gate reasons: "
        "full_system_calibration_not_locally_verified" in text
    )
    assert "Evidence:" in text
    assert (
        "evidence_path: /evidence/calibration/m2_idle_power_calibration.json"
        in text
    )
    assert "output_dir: /evidence/calibration/m2_off" in text


def test_operation_error_text_parses_new_backend_message_without_attributes() -> None:
    error = RuntimeError(
        "status=final_energy_gate_failed; "
        "final_energy_gate_reasons=method_sha256_mismatch, setup_binding_missing; "
        "evidence_dir=/evidence/calibration/m2_on; "
        "calibration_evidence=/evidence/calibration/m2_idle_power_calibration.json"
    )
    text = _platform_power_operation_error_text(error)
    assert "method_sha256_mismatch" in text
    assert "setup_binding_missing" in text
    assert "evidence_dir: /evidence/calibration/m2_on" in text
    assert (
        "calibration_evidence: "
        "/evidence/calibration/m2_idle_power_calibration.json" in text
    )


def test_three_card_ui_contains_method_badge_and_uses_structured_error_text() -> None:
    source = PANEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    labels = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "FS energy method: not checked" in labels
    assert "FS energy method: verified" in labels
    assert "FS energy method: missing" in labels
    assert "FS energy method: invalid" in labels
    assert 'path_resolver = getattr(app, "_hardware_setups_path", None)' in source
    assert "setup_id, registry_path=registry_path" in source
    assert "setup_id, energy, registry_path=registry_path" in source
    assert 'registry_path=state.get("registry_path")' in source
    assert "_platform_power_operation_error_text(error)" in source
    assert "messagebox.showerror(\n                            f\"Platform power — {setup_id}\", error_text" in source
