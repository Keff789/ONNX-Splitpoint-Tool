from __future__ import annotations

from pathlib import Path

import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.run_modes import infer_run_mode
from onnx_splitpoint_tool.workflow.profile_options import (
    workflow_options_from_profile_snapshot,
)
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/resnet50_v2772_hailo_parallel_build_canary.yaml"
PROFILE_ID = "resnet50_v2772_hailo_parallel_build_canary"


def _source_profile() -> dict:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_canary_legacy_profile_preserves_explicit_control_blocks() -> None:
    source = _source_profile()
    loaded = load_evaluation_profile(PROFILE, validate=True)

    assert loaded is not None and not isinstance(loaded, tuple)
    assert loaded.profile_id == PROFILE_ID
    assert "execution_preset" not in source
    for block in (
        "workflow",
        "benchmark_execution",
        "remote_execution",
        "model_preparation",
        "hailo_build",
        "build_scheduler",
        "artifact_store",
        "hardware_smoke",
        "validation",
    ):
        assert loaded.raw_profile[block] == source[block]
    assert infer_run_mode(loaded.raw_profile) == "smoke"


def test_canary_profile_options_keep_exact_narrow_build_scope(tmp_path: Path) -> None:
    source = _source_profile()
    snapshot = build_profile_start_snapshot(
        profile_request=str(PROFILE),
        source_profile=source,
        resolved_profile=source,
        profile_id=PROFILE_ID,
        profile_path=str(PROFILE),
        profile_source="file",
        runtime_bindings={},
        schema_version=2,
    )
    options = workflow_options_from_profile_snapshot(
        profile_request=str(PROFILE),
        out_root=str(tmp_path),
        start_snapshot=snapshot,
        models_root="/home/kmika/Models",
        required_run_mode="smoke",
        require_fresh_run=False,
    )

    assert options.execution_mode == "generate_benchmarksets"
    assert options.skip_benchmarks is True
    assert options.no_remote is True
    assert options.stop_after == "build_backend_artifacts"
    assert options.only_model == "resnet50"
    assert options.max_models == 1
    assert options.hailo_build_targets == ["hailo8", "hailo10"]
    assert options.hailo_build_backend == "auto"
    assert options.hailo_force_build is False
    assert options.hailo_build_full is False
    assert options.hailo_build_part1 is True
    assert options.hailo_build_part2 is False
    assert options.hailo_calib_count == 8
    assert options.benchmark_runs == 1
    assert options.required_run_mode == "smoke"
