from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import subprocess

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.full_only_quality_canary import (
    resolve_full_only_quality_canary,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "scripts/create_v27523_yolov7_quality_canary_profile.py"
)


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "v27523_canary_profile_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_canary() -> dict:
    return {
        "enabled": True,
        "execution_scope": "full_only",
        "full_run_ids": [
            {
                "id": "hailo8_full",
                "run_id": "hailo8",
                "setup_id": "orin_nx_hailo8_01",
                "backend": "hailo8",
                "variant": "full",
                "execution_role": "full_quality_only",
            },
            {
                "id": "hailo10h_full",
                "run_id": "hailo10",
                "setup_id": "orin_nx_hailo10_01",
                "backend": "hailo10h",
                "variants": ["full"],
                "execution_role": "full_quality_only",
            },
        ],
        "setup_local_tensorrt_companions": [
            {
                "id": "tensorrt_at_hailo8_full",
                "run_id": "ort_tensorrt",
                "setup_id": "orin_nx_hailo8_01",
                "backend": "tensorrt",
                "variant": "full",
                "execution_role": "full_quality_only",
            },
            {
                "id": "tensorrt_at_hailo10h_full",
                "run_id": "ort_tensorrt",
                "setup_id": "orin_nx_hailo10_01",
                "backend": "tensorrt",
                "variants": ["full"],
                "execution_role": "full_quality_only",
            },
        ],
    }


def test_run_mode_materialization_preserves_canary_and_full_variants() -> None:
    config = default_run_modes_config()
    profile = {
        "name": "canary-preservation",
        "model_suite": {
            "primary": [{"id": "yolov7_paper", "task": "detection"}]
        },
        "run_profiles": [
            {
                "id": "ort_tensorrt", "type": "same_backend_reference",
                "full": "tensorrt", "stage1": "tensorrt",
                "stage2": "tensorrt",
            },
            {
                "id": "hailo8", "type": "same_backend_reference",
                "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8",
            },
            {
                "id": "hailo10", "type": "same_backend_reference",
                "full": "hailo10h", "stage1": "hailo10h",
                "stage2": "hailo10h",
            },
        ],
        "execution_preset": {
            "id": "standard",
            "follow_tool_config": False,
            "overrides": {"native_enabled": False, "energy_enabled": False},
            "snapshot": copy.deepcopy(config["modes"]["standard"]),
        },
        "native_producers": {
            "enabled": False,
            "energy": {"enabled": False},
        },
        "energy": {"requested_native_energy": False},
        "quality_canary": _minimal_canary(),
    }

    once, _ = apply_run_mode(profile, config=config)
    twice, _ = apply_run_mode(once, config=config)

    assert once["quality_canary"] == profile["quality_canary"]
    assert twice["quality_canary"] == profile["quality_canary"]
    contract = resolve_full_only_quality_canary(
        twice, plan_rows=twice["run_profiles"]
    )
    assert contract["ok"] is True, contract["errors"]
    assert len(contract["expected_full_quality_identities"]) == 4


def test_generator_writes_reloadable_exact_full_only_profile(
    tmp_path: Path, monkeypatch,
) -> None:
    run_modes = tmp_path / "run_modes.yaml"
    monkeypatch.setenv("ONNX_SPLITPOINT_RUN_MODES_FILE", str(run_modes))
    models = tmp_path / "Models"
    models.mkdir()
    (models / "yolov7_paper.onnx").write_bytes(b"test-model")
    destination = tmp_path / "profiles" / "canary.yaml"
    module = _generator_module()

    module.write_profile(
        destination=destination,
        models_root=models.resolve(),
        profile_id="yolov7_full_only_quality_canary_v27523_test",
    )

    source = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert source["quality_canary"]["enabled"] is True
    loaded = load_evaluation_profile(str(destination), validate=True)
    assert loaded is not None and not isinstance(loaded, tuple)
    profile = dict(loaded.raw_profile or {})
    plan = module.verify_profile(profile)
    assert plan["generic_rows_total"] == 0
    assert plan["expected_full_quality_results_total"] == 4
    assert plan["remote_run_invocations_total"] == 2
    assert plan["expected_full_quality_identities"] == (
        module._expected_full_quality_identities()
    )
    assert plan["setup_groups"] == {
        module.H8_SETUP: ["hailo8", "ort_tensorrt"],
        module.H10_SETUP: ["hailo10", "ort_tensorrt"],
    }
    assert plan["effective_generic_run_ids"] == []
    assert plan["management_reference_profiles"] == ["ort_cpu"]
    assert profile["hailo_build"]["preset"] == "final"
    assert profile["hailo_build"]["optimization_level"] == 2
    assert profile["hailo_build"]["cache_integrity"] == "strict"
    assert profile["hailo_build"]["calib_count"] == 500
    assert plan["calibration_items"]["detection"] == 500
    assert profile["validation_execution"]["max_items"]["detection"] == 5000
    assert plan["validation_items"]["detection"] == 5000
    assert (
        profile["quality_gate"]["statistics"]["bootstrap_repetitions"]
        == 5000
    )
    assert profile["quality_gate"]["statistics"] == {
        "method": "paired_bootstrap",
        "confidence_level": 0.95,
        "bootstrap_repetitions": 5000,
        "seed": 20260710,
        "decision": "lower_one_sided_bound",
        "execution_location": "central_management",
        "workers": 4,
    }
    assert profile["native_producers"]["enabled"] is False
    assert profile["native_producers"]["energy"]["enabled"] is False
    assert profile["energy"]["enabled"] is False
    assert profile["energy"]["generic_enabled"] is False
    assert profile["energy"]["requested_native_energy"] is False

    identity_drift = copy.deepcopy(profile)
    identity_drift["quality_canary"]["full_run_ids"][1]["setup_id"] = (
        "orin_nx_hailo10_unapproved"
    )
    identity_drift["quality_canary"][
        "setup_local_tensorrt_companions"
    ][1]["setup_id"] = "orin_nx_hailo10_unapproved"
    with pytest.raises(ValueError, match="identity contract mismatch"):
        module.verify_profile(identity_drift)


def test_launchers_are_shell_valid_and_fail_closed() -> None:
    online = ROOT / "scripts/run_v27523_yolov7_quality_canary.sh"
    replay = ROOT / "scripts/run_v27523_yolov7_pack_replay.sh"
    for script in (online, replay):
        subprocess.run(["bash", "-n", str(script)], check=True)

    online_text = online.read_text(encoding="utf-8")
    assert "ONNX_SPLITPOINT_CANARY_MIN_FREE_GIB:-15" in online_text
    assert "ONNX_SPLITPOINT_CANARY_MIN_FREE_INODES:-50000" in online_text
    assert "--require-run-mode standard" in online_text
    assert "--require-fresh-run" in online_text
    assert "create_evaluation_debug_pack.py" in online_text
    assert "central_quality_replay_inputs" in online_text
    assert "OFFLINE_REPLAY=READY_FROM_DEBUG_PACK" in online_text
    assert "OFFLINE_REPLAY=NOT_READY" in online_text
    assert 'int(row.get("n") or 0) == 5000' in online_text
    assert 'int(seed_schema.get("repetitions") or 0) == 5000' in online_text
    assert 'float(component.get("margin")) == 0.01' in online_text
    assert "rm -rf -- \"$RUN_DIR\"" not in online_text

    replay_text = replay.read_text(encoding="utf-8")
    assert "--full-only" in replay_text
    assert "central_quality_replay_v27522.json" in replay_text
    assert "OFFLINE_REPLAY=PASS" in replay_text
    assert 'int(row.get("n") or 0) == 5000' in replay_text
    assert 'int(seed_schema.get("repetitions") or 0) == 5000' in replay_text
    assert 'float(component.get("margin")) == 0.01' in replay_text
