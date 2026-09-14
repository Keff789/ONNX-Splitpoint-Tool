from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy.collector import (
    HOST_NORMALIZATION_ROLE_NONE,
    HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
    apply_configured_energy_baselines,
    normalize_host_normalization_role,
    run_fast_firmware_measurement,
    select_host_normalization_role,
)
from onnx_splitpoint_tool.energy.config import EnergySetup
from onnx_splitpoint_tool.platform_power import measure_idle_power
from onnx_splitpoint_tool.benchmark.remote_run import (
    _energy_phase_payload_from_aggregate,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "run_id",
    (
        "ort_tensorrt",
        "tensorrt",
        "native_full_tensorrt",
        "tensorrt_full",
        "trt_full",
    ),
)
def test_only_exact_tensorrt_full_identities_select_host_normalization(
    run_id: str,
) -> None:
    assert (
        select_host_normalization_role(run_id=run_id, target_variant="full")
        == HOST_NORMALIZATION_ROLE_TENSORRT_FULL
    )


@pytest.mark.parametrize(
    ("run_id", "variant"),
    (
        ("hailo8", "full"),
        ("native_full_hailo8", "full"),
        ("hailo10", "full"),
        ("native_full_hailo10h", "full"),
        ("deepx_m1_full", "full"),
        ("native_full_deepx", "full"),
        ("hailo8_to_trt", "composed"),
        ("hailo10_to_tensorrt", "composed"),
        ("deepx_m1_to_tensorrt", "composed"),
        ("tensorrt_to_hailo8", "composed"),
        ("ort_tensorrt", "composed"),
        ("ort_tensorrt", "part1"),
        ("ort_tensorrt", "part2"),
        ("native_full_tensorrt", "split"),
        ("m2_idle_calibration_m2_off", "full"),
        ("", "full"),
        ("ort_tensorrt", ""),
    ),
)
def test_hailo_deepx_split_and_calibration_roles_never_select_correction(
    run_id: str, variant: str
) -> None:
    assert (
        select_host_normalization_role(
            run_id=run_id, target_variant=variant
        )
        == HOST_NORMALIZATION_ROLE_NONE
    )


def test_explicit_role_parser_rejects_unknown_or_fuzzy_values() -> None:
    assert normalize_host_normalization_role(None) == HOST_NORMALIZATION_ROLE_NONE
    assert normalize_host_normalization_role("") == HOST_NORMALIZATION_ROLE_NONE
    assert (
        normalize_host_normalization_role("tensorrt_full")
        == HOST_NORMALIZATION_ROLE_TENSORRT_FULL
    )
    for invalid in ("tensorrt", "trt_full", "TensorRT-Full", "host_only", True):
        with pytest.raises(ValueError, match="Unsupported host_normalization_role"):
            normalize_host_normalization_role(invalid)


def test_collector_rejects_unknown_role_before_creating_measurement_output(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="Unsupported host_normalization_role"):
        run_fast_firmware_measurement(
            "true",
            out_dir,
            setup=EnergySetup(
                setup_id="test",
                enabled=True,
                urecs_address="192.0.2.10",
            ),
            host_normalization_role="tensorrt-ish",
        )
    assert not out_dir.exists()


def test_accelerator_idle_is_subtracted_only_for_selected_tensorrt_full() -> None:
    summary = {"energy_total_j": 100.0, "active_duration_s": 10.0}
    trt_full = apply_configured_energy_baselines(
        summary,
        idle_baseline_w=3.0,
        accelerator_idle_w=2.0,
        host_normalization_role=HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
        accelerator_idle_calibration={
            "accelerator_idle_calibration_verified": True,
            "accelerator_idle_calibration_status": "verified",
            "accelerator_idle_calibration_binding_sha256": "a" * 64,
        },
        host_normalization_source_run_id="native_full_tensorrt",
        host_normalization_target_variant="full",
    )
    assert trt_full["energy_dynamic_j"] == pytest.approx(70.0)
    assert trt_full["host_normalized_energy_est_j"] == pytest.approx(80.0)
    assert trt_full["accelerator_idle_correction_requested"] is True
    assert trt_full["accelerator_idle_correction_applied"] is True
    assert trt_full["accelerator_idle_correction_status"] == "applied"

    unselected = apply_configured_energy_baselines(
        summary,
        idle_baseline_w=3.0,
        accelerator_idle_w=2.0,
        host_normalization_role=HOST_NORMALIZATION_ROLE_NONE,
    )
    assert unselected["energy_dynamic_j"] == pytest.approx(70.0)
    assert unselected["host_normalized_energy_est_j"] is None
    assert unselected["accelerator_idle_correction_requested"] is False
    assert unselected["accelerator_idle_correction_applied"] is False
    assert unselected["accelerator_idle_correction_status"] == "not_requested"


def test_selected_tensorrt_full_without_calibration_is_explicitly_unavailable() -> None:
    out = apply_configured_energy_baselines(
        {"energy_total_j": 100.0, "active_duration_s": 10.0},
        idle_baseline_w=None,
        accelerator_idle_w=None,
        host_normalization_role=HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
        host_normalization_source_run_id="native_full_tensorrt",
        host_normalization_target_variant="full",
    )
    assert out["host_normalized_energy_est_j"] is None
    assert out["accelerator_idle_correction_requested"] is True
    assert out["accelerator_idle_correction_applied"] is False
    assert (
        out["accelerator_idle_correction_status"]
        == "unavailable_missing_accelerator_idle_w"
    )


@pytest.mark.parametrize("invalid", (-0.1, float("nan"), float("inf"), True))
def test_invalid_calibration_value_is_never_applied(invalid: object) -> None:
    out = apply_configured_energy_baselines(
        {"energy_total_j": 100.0, "active_duration_s": 10.0},
        idle_baseline_w=None,
        accelerator_idle_w=invalid,
        host_normalization_role=HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
        host_normalization_source_run_id="native_full_tensorrt",
        host_normalization_target_variant="full",
    )
    assert out["host_normalized_energy_est_j"] is None
    assert out["accelerator_idle_correction_applied"] is False
    assert (
        out["accelerator_idle_correction_status"]
        == "unavailable_invalid_accelerator_idle_w"
    )


def test_idle_calibration_measurement_never_requests_its_own_correction(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    manifest = tmp_path / "verified_energy_method.json"
    manifest.write_text('{"test":"method"}\n', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    method = {
        "path": str(manifest.resolve()),
        "sha256": digest,
        "verification_status": "inherited_validated_method_verified",
        "runtime_binding_id": "orin_nx_hailo8_01",
        "verified": True,
    }

    def runner(_command, _output_dir, **kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "status": "ok",
            "avg_power_w": 12.5,
            "energy_calibration_manifest": str(manifest.resolve()),
            "energy_calibration_sha256": digest,
            "energy_calibration_verification": {
                "verified": True,
                "status": "inherited_validated_method_verified",
                "runtime_binding_id": "orin_nx_hailo8_01",
            },
        }

    registry = {
        "energy_defaults": {
            "enabled": True,
            "data_port": 3000,
            "channel": 0,
            "sample_rate": 2000,
            "physical_scope": "FS",
            "window_label": "command",
        },
        "hardware_setups": [
            {
                "id": "orin_nx_hailo8_01",
                "accelerator": "hailo8",
                "host": {
                    "address": "192.0.2.20",
                    "user": "nx",
                    "port": 22,
                },
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.10",
                    "calibration_manifest": str(manifest.resolve()),
                    "calibration_sha256": digest,
                    "accelerator_idle_w": 9.9,
                },
            }
        ],
    }
    value, _evidence = measure_idle_power(
        "orin_nx_hailo8_01",
        registry=registry,
        setup={"id": "orin_nx_hailo8_01", "accelerator": "hailo8"},
        cfg={
            "energy": {
                "enabled": True,
                "urecs_address": "192.0.2.10",
                "accelerator_idle_w": 9.9,
            }
        },
        state_label="m2_off",
        duration_s=5,
        output_dir=tmp_path,
        measurement_runner=runner,
        energy_method=method,
    )
    assert value == pytest.approx(12.5)
    assert captured["setup"].accelerator_idle_w is None
    assert "host_normalization_role" not in captured


def test_remote_benchmark_row_projects_tensorrt_full_correction(
    tmp_path: Path,
) -> None:
    aggregate_path = tmp_path / "energy_aggregate.json"
    payload = _energy_phase_payload_from_aggregate(
        {
            "energy_measurement_scope": "command_energy",
            "phases": [
                {
                    "phase": "latency",
                    "avg_energy_total_j": 100.0,
                    "avg_energy_dynamic_j": 70.0,
                    "avg_host_normalized_energy_est_j": 80.0,
                    "host_normalization_role": "tensorrt_full",
                    "accelerator_idle_correction_requested": True,
                    "accelerator_idle_correction_applied": True,
                    "accelerator_idle_correction_statuses": ["applied"],
                }
            ],
        },
        aggregate_path,
        {
            "energy_target_case": "full",
            "energy_target_variant": "full",
            "energy_applies_to_all_cases": True,
        },
    )
    assert payload["energy_dynamic_j"] == pytest.approx(70.0)
    assert payload["host_normalized_energy_est_j"] == pytest.approx(80.0)
    assert payload["host_normalization_role"] == "tensorrt_full"
    assert payload["accelerator_idle_correction_requested"] is True
    assert payload["accelerator_idle_correction_applied"] is True
    assert payload["accelerator_idle_correction_statuses"] == ["applied"]


def _measurement_call_keywords(relative: str) -> list[set[str]]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    calls: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "run_fast_firmware_measurement":
            calls.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return calls


def test_only_gui_row_target_and_benchmark_remote_row_target_wire_selector() -> None:
    gui_calls = _measurement_call_keywords("onnx_splitpoint_tool/gui/app.py")
    remote_calls = _measurement_call_keywords(
        "onnx_splitpoint_tool/benchmark/remote_run.py"
    )
    assert sum("host_normalization_role" in keys for keys in gui_calls) == 1
    assert sum("host_normalization_role" in keys for keys in remote_calls) == 1
    assert len(gui_calls) >= 1
    assert len(remote_calls) >= 2
