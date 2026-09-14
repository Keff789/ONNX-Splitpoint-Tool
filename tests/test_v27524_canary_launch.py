from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.contracts import StageResult
from onnx_splitpoint_tool.workflow.full_only_quality_canary import (
    resolve_full_only_quality_canary,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _required_hailo_full_preupload_gate,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    ROOT / "scripts/create_v27524_yolov7_quality_canary_profile.py"
)


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "v27524_canary_profile_generator", GENERATOR_PATH
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
        profile_id="yolov7_full_only_quality_canary_v27524_test",
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
    assert profile["hailo_build"]["mode"] == "cache_verify_only"
    assert profile["hailo_build"]["preset"] == "balanced"
    assert profile["hailo_build"]["optimization_level"] == 1
    assert profile["hailo_build"]["cache_integrity"] == "relaxed"
    assert profile["hailo_build"]["calib_count"] == 500
    assert profile["hailo_build"]["calib_batch_size"] == 8
    assert profile["hailo_build"]["calibration_storage"] == "memmap"
    assert profile["hailo_build"]["timeout_s"] == 3600
    assert (
        profile["hailo_build"]["full_baseline_cold_build_policy"]
        == "cache_or_defer"
    )
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


def _receipt_attested_promotion(backend: str) -> dict:
    return {
        "backend": backend,
        "artifact_path": f"hailo/{backend}/full/compiled.hef",
        "artifact_sha256": "a" * 64,
        "hailo_build_receipt_file_sha256": "b" * 64,
        "hailo_build_receipt_identity_sha256": "c" * 64,
    }


def _preupload_profile() -> dict:
    return {
        "quality_canary": _minimal_canary(),
        "hailo_build": {
            "mode": "cache_verify_only",
            "preset": "balanced",
            "optimization_level": 1,
            "calib_count": 500,
            "calib_batch_size": 8,
            "calibration_storage": "memmap",
            "cache_integrity": "relaxed",
        },
    }


def _deepx_only_preupload_profile() -> dict:
    return {
        "quality_canary": {
            "enabled": True,
            "execution_scope": "full_only",
            "full_run_ids": [
                {
                    "id": "deepx_m1_full",
                    "run_id": "deepx_m1_full",
                    "setup_id": "orin_nx_deepx_m1_01",
                    "backend": "deepx_m1",
                    "variant": "full",
                    "execution_role": "full_quality_only",
                },
            ],
            "setup_local_tensorrt_companions": [
                {
                    "id": "tensorrt_at_deepx_m1_full",
                    "run_id": "ort_tensorrt",
                    "setup_id": "orin_nx_deepx_m1_01",
                    "backend": "tensorrt",
                    "variant": "full",
                    "execution_role": "full_quality_only",
                },
            ],
        },
    }


def test_preupload_gate_is_all_or_nothing_and_accepts_hailo10_alias() -> None:
    profile = _preupload_profile()
    one_hit = _required_hailo_full_preupload_gate(
        profile,
        {
            "recorded_hailo_full_contracts": [
                _receipt_attested_promotion("hailo8"),
            ],
        },
    )
    assert one_hit["enabled"] is True
    assert one_hit["ok"] is False
    assert one_hit["ready_backends"] == ["hailo8"]
    assert one_hit["missing_backends"] == ["hailo10h"]
    assert one_hit["hailo_axes"]["preset"] == "balanced"
    assert one_hit["hailo_axes"]["optimization_level"] == 1
    assert one_hit["hailo_axes"]["cache_integrity"] == "relaxed"
    assert one_hit["hailo_axes"]["calib_count"] == 500

    both_hits = _required_hailo_full_preupload_gate(
        profile,
        {
            "recorded_hailo_full_contracts": [
                _receipt_attested_promotion("hailo8"),
                _receipt_attested_promotion("hailo10"),
            ],
        },
    )
    assert both_hits["ok"] is True
    assert both_hits["ready_backends"] == ["hailo8", "hailo10h"]
    assert both_hits["missing_backends"] == []

    naked_h10 = _receipt_attested_promotion("hailo10h")
    naked_h10.pop("hailo_build_receipt_identity_sha256")
    missing_receipt = _required_hailo_full_preupload_gate(
        profile,
        {
            "recorded_hailo_full_contracts": [
                _receipt_attested_promotion("hailo8"), naked_h10,
            ],
        },
    )
    assert missing_receipt["ok"] is False
    assert missing_receipt["missing_backends"] == ["hailo10h"]

    normal_standard = _required_hailo_full_preupload_gate(
        {"execution_preset": {"id": "standard"}}, {},
    )
    assert normal_standard["enabled"] is False
    assert normal_standard["ok"] is True


def test_preupload_gate_is_disabled_for_deepx_only_full_canary() -> None:
    gate = _required_hailo_full_preupload_gate(
        _deepx_only_preupload_profile(),
        {"recorded_hailo_full_contracts": []},
    )

    assert gate == {
        "enabled": False,
        "ok": True,
        "requested_backends": [],
        "ready_backends": [],
        "missing_backends": [],
        "ambiguous_backends": [],
        "observed_promotions": [],
        "hailo_axes": {},
    }


def test_deepx_only_canary_reaches_remote_executor_without_hailo_receipts(
    tmp_path: Path,
) -> None:
    model_id = "resnet50"
    benchmark_set = tmp_path / "models" / model_id / "benchmark_set"
    benchmark_set.mkdir(parents=True)
    (benchmark_set / "backend_artifact_decisions.json").write_text(
        json.dumps({"recorded_hailo_full_contracts": []}),
        encoding="utf-8",
    )
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.profile_payload = _deepx_only_preupload_profile()
    workflow.log = mock.Mock()
    workflow.options = SimpleNamespace()
    workflow._cancel_event = mock.Mock()
    workflow._process_registry = None
    workflow._remote_process_registry = None
    workflow.session_id = "deepx-only-canary-test"
    workflow._schedule_management_cpu_reference = mock.Mock()

    executor_reached = RuntimeError("deepx remote executor reached")
    with (
        mock.patch(
            "onnx_splitpoint_tool.workflow.runner."
            "benchmark_set_postcondition_v60v",
            return_value={
                "valid": True,
                "selected_suite_dir": str(benchmark_set),
            },
        ),
        mock.patch(
            "onnx_splitpoint_tool.workflow.runner.finalize_suite_for_runtime",
            return_value={"benchmark_plan": {}},
        ),
        mock.patch(
            "onnx_splitpoint_tool.workflow.runner."
            "execute_benchmark_suite_if_requested",
            side_effect=executor_reached,
        ) as executor,
        pytest.raises(RuntimeError, match="deepx remote executor reached"),
    ):
        workflow._stage_run_benchmarks(
            model_id, {"id": model_id, "task": "classification"},
        )

    executor.assert_called_once()
    workflow._schedule_management_cpu_reference.assert_called_once()


def test_missing_required_hef_blocks_before_executor_call(
    tmp_path: Path,
) -> None:
    model_id = "yolov7_paper"
    benchmark_set = tmp_path / "models" / model_id / "benchmark_set"
    benchmark_set.mkdir(parents=True)
    (benchmark_set / "backend_artifact_decisions.json").write_text(
        json.dumps({
            "recorded_hailo_full_contracts": [
                _receipt_attested_promotion("hailo8"),
            ],
        }),
        encoding="utf-8",
    )
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.profile_payload = _preupload_profile()
    workflow.log = mock.Mock()

    with mock.patch(
        "onnx_splitpoint_tool.workflow.runner."
        "execute_benchmark_suite_if_requested",
        side_effect=AssertionError("remote executor must not be called"),
    ) as executor:
        artifacts, metrics, message, status = (
            workflow._stage_run_benchmarks(model_id, {"id": model_id})
        )

    executor.assert_not_called()
    assert artifacts == {}
    assert status == "failed"
    assert metrics["skip_reason"] == (
        "required_hailo_full_preupload_gate_failed"
    )
    assert metrics["measured_result_count"] == 0
    assert "STOP before remote upload" in message
    assert "hailo10h" in message


def test_failed_artifact_gate_skips_every_upload_dependent_stage(
    tmp_path: Path,
) -> None:
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow._stop_requested = False
    workflow.run_dir = tmp_path / "run"
    workflow.manifest = {}
    workflow.jobs = None
    workflow.options = SimpleNamespace(
        skip_analysis=False, skip_benchmarks=False, no_remote=False,
    )
    called: list[tuple[str, str]] = []

    def fake_run_stage(model_id, stage, fn):
        payload = fn()
        status = str(payload[3])
        called.append((stage, status))
        return StageResult(
            stage=stage,
            model_id=model_id,
            status=status,
            started_at="x",
            finished_at="x",
        )

    workflow._run_stage = fake_run_stage
    for name in (
        "_stage_resolve_model",
        "_stage_check_validation_assets",
        "_stage_prepare_model",
        "_stage_analyze_model",
        "_stage_select_split_candidates",
        "_stage_prepare_full_baselines",
        "_stage_generate_benchmark_set",
        "_stage_validate_outputs",
        "_stage_hardware_smoke",
    ):
        setattr(workflow, name, lambda *args, **kwargs: ({}, {}, "ok", "ok"))
    workflow._stage_build_backend_artifacts = lambda *args, **kwargs: (
        {}, {}, "STOP before remote upload", "failed",
    )
    workflow._stage_run_benchmarks = mock.Mock(
        side_effect=AssertionError("upload stage must be skipped"),
    )

    workflow._run_model({"id": "yolov7_paper"})

    workflow._stage_run_benchmarks.assert_not_called()
    statuses = dict(called)
    assert statuses["build_backend_artifacts"] == "failed"
    assert statuses["run_benchmarks"] == "skipped"
    assert statuses["validate_outputs"] == "skipped"
    assert statuses["hardware_smoke"] == "skipped"


def test_launchers_are_shell_valid_and_fail_closed() -> None:
    online = ROOT / "scripts/run_v27524_yolov7_quality_canary.sh"
    replay = ROOT / "scripts/run_v27524_yolov7_pack_replay.sh"
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
