from __future__ import annotations

from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.run_modes import (
    apply_run_mode,
    default_run_modes_config,
)
from onnx_splitpoint_tool.workflow.hailo_remote_binding import (
    _hailo_build_targets_from_options,
)
from onnx_splitpoint_tool.workflow.profile_options import (
    workflow_options_from_profile_snapshot,
)
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot,
)


def _phase5_profile(*, hw_arch: str, targets: list[str]) -> dict:
    mode = default_run_modes_config()["modes"]["standard"]
    return {
        "name": f"phase5_{hw_arch}_{len(targets)}",
        "purpose": "v2.77.7 Phase-5 physical Hailo projection regression",
        "models_root_hint": "/home/kmika/Models",
        "model_suite": {
            "primary": [
                {
                    "id": "mobilenet_v3_large",
                    "task": "classification",
                    "enabled": True,
                    "evaluation_role": "development",
                }
            ],
            "reserve": [],
        },
        "run_profiles": [
            {
                "id": "ort_tensorrt",
                "type": "same_backend_reference",
                "full": "tensorrt",
                "stage1": "tensorrt",
                "stage2": "tensorrt",
                "enabled": True,
            }
        ],
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": False,
            "snapshot": mode,
            "overrides": {
                "native_enabled": False,
                "energy_enabled": False,
            },
        },
        "hailo_build": {
            "hw_arch": hw_arch,
            "targets": targets,
            # This is deliberately different from Standard.  Only the physical
            # axis above may survive; run-mode effort remains authoritative.
            "timeout_s": 123,
        },
    }


@pytest.mark.parametrize(
    ("hw_arch", "targets"),
    [
        ("hailo8", ["hailo8"]),
        ("hailo10", ["hailo10"]),
        ("hailo8", []),
    ],
)
def test_apply_run_mode_preserves_only_explicit_physical_hailo_axis(
    hw_arch: str,
    targets: list[str],
) -> None:
    source = _phase5_profile(hw_arch=hw_arch, targets=targets)
    resolved, _audit = apply_run_mode(
        source, config=default_run_modes_config()
    )

    assert resolved["hailo_build"]["hw_arch"] == hw_arch
    assert resolved["hailo_build"]["targets"] == targets
    assert resolved["hailo_build"]["timeout_s"] == 3600

    # Saved materialised profiles are projected again when loaded.  The
    # physical contract must therefore also be idempotent across that second
    # pass.
    reprojected, _audit = apply_run_mode(
        resolved, config=default_run_modes_config()
    )
    assert reprojected["hailo_build"]["hw_arch"] == hw_arch
    assert reprojected["hailo_build"]["targets"] == targets
    assert reprojected["hailo_build"]["timeout_s"] == 3600


def test_missing_targets_still_receive_hailo_arch_default() -> None:
    source = _phase5_profile(hw_arch="hailo10", targets=["hailo10"])
    del source["hailo_build"]["targets"]
    resolved, _audit = apply_run_mode(
        source, config=default_run_modes_config()
    )

    # A genuinely absent target key keeps the historical hw_arch-derived
    # default; only an explicit empty list disables Hailo.
    assert resolved["hailo_build"]["hw_arch"] == "hailo10"
    assert resolved["hailo_build"]["targets"] == ["hailo10"]


@pytest.mark.parametrize(
    ("hw_arch", "targets"),
    [("hailo10", ["hailo10"]), ("hailo8", [])],
)
def test_repeated_run_mode_projection_is_idempotent_for_physical_axis(
    hw_arch: str,
    targets: list[str],
) -> None:
    first, _audit = apply_run_mode(
        _phase5_profile(hw_arch=hw_arch, targets=targets),
        config=default_run_modes_config(),
    )
    second, _audit = apply_run_mode(
        first, config=default_run_modes_config()
    )

    assert second["hailo_build"]["hw_arch"] == hw_arch
    assert second["hailo_build"]["targets"] == targets
    assert second["hailo_build"]["timeout_s"] == 3600


@pytest.mark.parametrize(
    ("hw_arch", "targets"),
    [
        ("hailo10", ["hailo10"]),
        ("hailo8", []),
    ],
)
def test_start_snapshot_options_keep_explicit_targets_including_empty(
    tmp_path,
    hw_arch: str,
    targets: list[str],
) -> None:
    source = _phase5_profile(hw_arch=hw_arch, targets=targets)
    resolved, _audit = apply_run_mode(
        source, config=default_run_modes_config()
    )
    snapshot = build_profile_start_snapshot(
        profile_request="phase5-v2777",
        source_profile=source,
        resolved_profile=resolved,
        profile_id=source["name"],
        profile_path="phase5-v2777.yaml",
        profile_source="file",
        runtime_bindings={},
        schema_version=2,
    )
    options = workflow_options_from_profile_snapshot(
        profile_request="phase5-v2777",
        out_root=str(tmp_path),
        start_snapshot=snapshot,
        required_run_mode="standard",
    )

    assert options.hailo_hw_arch == hw_arch
    assert options.hailo_build_targets == targets


def test_profile_options_defaults_only_when_targets_key_is_absent(
    tmp_path,
) -> None:
    source = _phase5_profile(hw_arch="hailo10", targets=["hailo10"])
    resolved, _audit = apply_run_mode(
        source, config=default_run_modes_config()
    )
    del resolved["hailo_build"]["targets"]
    snapshot = build_profile_start_snapshot(
        profile_request="phase5-v2777-missing-targets",
        source_profile=resolved,
        resolved_profile=resolved,
        profile_id=source["name"],
        profile_path="phase5-v2777-missing-targets.yaml",
        profile_source="file",
        runtime_bindings={},
        schema_version=2,
    )
    options = workflow_options_from_profile_snapshot(
        profile_request="phase5-v2777-missing-targets",
        out_root=str(tmp_path),
        start_snapshot=snapshot,
        required_run_mode="standard",
    )

    assert options.hailo_build_targets == ["hailo10"]


def test_remote_binding_target_projection_distinguishes_empty_from_missing() -> None:
    assert _hailo_build_targets_from_options(
        SimpleNamespace(hailo_hw_arch="hailo10")
    ) == ["hailo10"]
    assert _hailo_build_targets_from_options(
        SimpleNamespace(hailo_hw_arch="hailo8", hailo_build_targets=[])
    ) == []
    assert _hailo_build_targets_from_options(
        SimpleNamespace(
            hailo_hw_arch="hailo10",
            hailo_build_targets=["hailo10"],
        )
    ) == ["hailo10"]
