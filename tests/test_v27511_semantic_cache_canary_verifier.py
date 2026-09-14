from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
VARIANT = "v27510_resnet50_b052_hailo8_smoke_cache_verify"
NAMESPACE = "v000_" + VARIANT
VARIANT_ROOT = Path("native_producers/variants") / NAMESPACE / "hailo8"
BINDING_REL = (
    VARIANT_ROOT
    / "cache_verify_native_split/orin_nx_hailo8_01"
    / "native_split_quality_binding_set.json"
)
RUNNER_REL = VARIANT_ROOT / "analysis_tables/native_fifo_eval_runner.json"
RESULT_REL = (
    VARIANT_ROOT
    / "resnet50/benchmark_set/native_pipeline/b052/hailo_to_trt"
    / "float32_layout_fp16/native_fifo_results.json"
)


def _checker():
    path = ROOT / "scripts/verify_cache_canary_result.py"
    name = f"v27511_semantic_cache_canary_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _binding(run_id: str) -> dict[str, Any]:
    value = json.loads((
        ROOT
        / "tests/fixtures/v27510_cache_canary"
        / "resnet50_b052_hailo8_outer_binding.json"
    ).read_text(encoding="utf-8"))
    # Deliberately do not reseal the historical binding.  These stale hashes
    # prove that the terminal PASS decision now uses semantic compatibility.
    value["eval_run_id"] = run_id
    value["source_run_id"] = "hailo8_to_trt"
    value["cache_verify_replay"] = {
        "artifact_policy": "cache_verify_only",
        "compiler_dispatched": False,
        "source_binding_sha256": "1" * 64,
        "local_validation_status": "diagnostic_hashes_not_authoritative",
    }
    return value


def _manifest(run_id: str) -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run_id,
        "profile_start_snapshot": {
            "cache_verify_attestation": {
                "schema": "onnx-splitpoint/cache-verify-attestation",
                "schema_version": 1,
                "status": "verified",
                "mode": "cache_verify_only",
                "compiler_dispatch_allowed": False,
                "attestation_sha256": "2" * 64,
                "actual_plan": {
                    "hailo_build_mode": "cache_verify_only",
                    "hailo_force_build": False,
                    "deepx_build_mode": "cache_verify_only",
                    "deepx_force_build": False,
                    "native_build_missing_engines": False,
                    "native_force_rebuild_variants": [],
                    "deepx_prefetch_enabled": False,
                    "generic_runtime_enabled": False,
                    "models": ["resnet50"],
                    "native_backends": ["hailo8"],
                    "native_full_backends": [],
                    "native_energy_enabled": False,
                    "native_frames": 10,
                    "native_warmup": 2,
                    "native_repetitions": 1,
                    "native_rows": [{
                        "model_id": "resnet50",
                        "case_id": "b052",
                        "backend": "hailo8",
                        "setup_id": "orin_nx_hailo8_01",
                        "variant_id": VARIANT,
                    }],
                },
            },
        },
    }


def _repetition(runtime_id: str, fps: float) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "ok",
        "mode": "native_hailort_tensorrt_fifo",
        "frames": 10,
        "completed_frames": 10,
        "requested_frames": 10,
        "completed_work_units": 10,
        "duration_s": 0.0,
        "warmup": 2,
        "queue_depth": 2,
        "task": "classification",
        "fps_makespan": fps,
        "runtime_instance_id": runtime_id,
        "repetition_id": "hailo8:semantic-canary-repetition-1",
        "process_local_repetition_index": 1,
        "repetition_index": 1,
        "repetition_runtime_scope": "fresh_process_per_repetition",
        "workload_contract_sha256": "3" * 64,
    }


def _payloads(run_dir: Path) -> dict[Path, dict[str, Any]]:
    run_id = run_dir.name
    binding = _binding(run_id)
    cache_attestation = {
        "schema": "onnx-splitpoint/native-split-cache-verify-attestation",
        "schema_version": 1,
        "status": "verified",
        "artifact_policy": "cache_verify_only",
        "compiler_dispatch_allowed": False,
        "compiler_dispatched": False,
        "eval_run_id": run_id,
        "model_id": "resnet50",
        "case_id": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "backend": "hailo8_to_trt",
        "selected_cache_root": "/remote/cache/resnet50-compatible",
        "source_binding_sha256": "4" * 64,
        "source_binding_artifact_set_sha256": "5" * 64,
        "attestation_sha256": "6" * 64,
    }
    binding_set = {
        "schema": "onnx-splitpoint/native-split-quality-binding-set",
        "schema_version": 2,
        "mode": "cache_verify_only",
        "diagnostic_only": True,
        "claim_eligible": False,
        "eval_run_id": run_id,
        "setup_id": "orin_nx_hailo8_01",
        "cache_verify_attestation": cache_attestation,
        "cache_verify_attestation_sha256": "7" * 64,
        "bindings_by_model_case_backend": {
            "resnet50|b052|hailo8_to_trt": binding,
        },
        "binding_set_sha256": "8" * 64,
    }

    fps = 499.424413
    runtime_id = "fresh_process:" + "1" * 64
    repetition = _repetition(runtime_id, fps)
    result = {
        "ok": True,
        "mode": "native_hailort_tensorrt_fifo",
        "frames": 10,
        "completed_frames": 10,
        "requested_frames": 10,
        "duration_s": 0.0,
        "warmup": 2,
        "queue_depth": 2,
        "task": "classification",
        "precision": "float32_layout_fp16",
        "hw_arch": "hailo8",
        "case": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "eval_run_id": run_id,
        "source_run_id": "hailo8_to_trt",
        "fps_makespan": fps,
        "repetitions_requested": 1,
        "repetitions_completed": 1,
        "repetition_count_requested": 1,
        "repetition_count_attempted": 1,
        "repetition_count_valid": 1,
        "repetition_status": "complete",
        "repetition_aggregation": "median_never_best_of",
        "repetition_runtime_scope": "fresh_process_per_repetition",
        "repetition_independence_verified": True,
        "runtime_instance_id": runtime_id,
        "repetition_records": [copy.deepcopy(repetition)],
        "repetition_evidence": [copy.deepcopy(repetition)],
        "native_split_quality_runtime_boundary_verified": True,
        "hailo_runtime_output_count": 1,
        "hailo_runtime_output_name": "resnet50_part1_b52/conv24",
        "hailo_runtime_output_frame_bytes": 1_605_632,
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": "9" * 64,
        "native_command_contract": {
            "schema": "diagnostic-command",
            "contract_sha256": "a" * 64,
            "runtime_options": {"frames": 999},
        },
        "native_command_contract_sha256": "b" * 64,
        "workload_contract_sha256": "c" * 64,
        "native_split_quality_consumer_attestation": {
            "status": "diagnostic-only",
            "attestation_sha256": "d" * 64,
        },
    }
    runner = {
        "ok": True,
        "orchestration_status": "ok",
        "evidence_status": "complete",
        "row_count": 1,
        "ok_count": 1,
        "failed_count": 0,
        "rows": [{
            "model": "resnet50",
            "case_id": "b052",
            "setup_id": "orin_nx_hailo8_01",
            "comparison_backend": "hailo8",
            "status": "ok",
            "result_ok": True,
            "returncode": 0,
            "timed_out": False,
            "child_result_fresh": True,
            "eval_run_id": run_id,
            "source_run_id": "hailo8_to_trt",
            "fps_makespan": fps,
            "native_fifo_result": "/remote/native_fifo_results.json",
            "native_fifo_result_sha256": "0" * 64,
            "native_fifo_result_size_bytes": -1,
            "native_command_contract_sha256": "e" * 64,
        }],
    }
    # This is the exact class of incomplete fallback projection observed in
    # the v2.75.10 field run.  It is intentionally diagnostic-only.
    summary = {
        "schema": "onnx-splitpoint/native-producer-combined-summary",
        "schema_version": 7,
        "row_count": 1,
        "ok_count": 1,
        "evidence_status": "complete",
        "rows": [{
            "model": "resnet50",
            "case": "b052",
            "backend": "hailo8_to_trt",
            "setup_id": "orin_nx_hailo8_01",
            "ok": True,
            "status": "ok",
            # returncode, freshness, result hash and command fields absent
        }],
    }
    return {
        Path("run_manifest.json"): _manifest(run_id),
        BINDING_REL: binding_set,
        RUNNER_REL: runner,
        RESULT_REL: result,
        Path("reports/native_producer_summary.json"): summary,
    }


def _write_run(run_dir: Path) -> dict[Path, dict[str, Any]]:
    payloads = _payloads(run_dir)
    for relative, payload in payloads.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return payloads


def test_semantic_canary_accepts_real_writer_without_top_level_status(
    tmp_path: Path,
) -> None:
    checker = _checker()
    run_dir = tmp_path / "cache_verify_semantic_pass"
    _write_run(run_dir)

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result["errors"]
    assert result["status"] == "semantic_cache_canary_pass"
    assert result["diagnostics"]["collector_summary"]["authoritative"] is False
    assert result["diagnostics"]["native_result_file"]["sha256_match"] is False
    assert result["diagnostics"]["hash_receipt_command"]["authoritative"] is False


@pytest.mark.parametrize("status", ("failed", "error"))
def test_semantic_canary_rejects_contradictory_optional_result_status(
    tmp_path: Path, status: str,
) -> None:
    checker = _checker()
    run_dir = tmp_path / f"cache_verify_result_status_{status}"
    payloads = _payloads(run_dir)
    payloads[RESULT_REL]["status"] = status
    for relative, payload in payloads.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is False
    assert f"native_result.status:{status!r}!='ok'" in result["errors"]


def test_diagnostic_hash_receipt_command_and_collector_fields_may_be_absent(
    tmp_path: Path,
) -> None:
    checker = _checker()
    run_dir = tmp_path / "cache_verify_without_diagnostic_proofs"
    payloads = _payloads(run_dir)
    binding_set = payloads[BINDING_REL]
    attestation = binding_set["cache_verify_attestation"]
    binding = binding_set["bindings_by_model_case_backend"][
        "resnet50|b052|hailo8_to_trt"
    ]
    result_payload = payloads[RESULT_REL]
    runner_row = payloads[RUNNER_REL]["rows"][0]
    for field in (
        "binding_set_sha256", "cache_verify_attestation_sha256",
    ):
        binding_set.pop(field, None)
    for field in (
        "selected_cache_root", "source_binding_sha256",
        "source_binding_artifact_set_sha256", "attestation_sha256",
    ):
        attestation.pop(field, None)
    for field in (
        "binding_sha256", "engine_build_receipt",
    ):
        binding.pop(field, None)
    binding["artifacts"].pop("engine_build_receipt", None)
    binding["artifacts"].pop("trtexec", None)
    for field in (
        "native_command_contract", "native_command_contract_sha256",
        "workload_contract_sha256",
        "native_split_quality_binding_sha256",
        "native_split_quality_consumer_attestation",
    ):
        result_payload.pop(field, None)
    for field in (
        "native_fifo_result", "native_fifo_result_sha256",
        "native_fifo_result_size_bytes", "native_command_contract_sha256",
    ):
        runner_row.pop(field, None)
    payloads.pop(Path("reports/native_producer_summary.json"))
    for relative, payload in payloads.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result["errors"]
    assert result["diagnostics"]["collector_summary"][
        "read_status"
    ] == "missing"


@pytest.mark.parametrize(
    "mutation",
    (
        "compiler_allowed",
        "compiler_dispatched",
        "semantic_model_drift",
        "runner_nonzero",
        "runner_not_fresh",
        "runner_stale_eval",
        "result_stale_eval",
        "completed_frames_short",
        "warmup_drift",
        "queue_depth_drift",
        "completed_work_units_short",
        "runtime_identity_drift",
        "nonfinite_fps",
    ),
)
def test_semantic_canary_rejects_only_hard_gate_drift(
    tmp_path: Path, mutation: str,
) -> None:
    checker = _checker()
    run_dir = tmp_path / f"cache_verify_{mutation}"
    payloads = _payloads(run_dir)
    manifest = payloads[Path("run_manifest.json")]
    binding_set = payloads[BINDING_REL]
    runner_row = payloads[RUNNER_REL]["rows"][0]
    result_payload = payloads[RESULT_REL]
    if mutation == "compiler_allowed":
        manifest["profile_start_snapshot"]["cache_verify_attestation"][
            "compiler_dispatch_allowed"
        ] = True
    elif mutation == "compiler_dispatched":
        binding_set["cache_verify_attestation"]["compiler_dispatched"] = True
    elif mutation == "semantic_model_drift":
        binding_set["bindings_by_model_case_backend"][
            "resnet50|b052|hailo8_to_trt"
        ]["preselection"]["model_id"] = "resnet18"
    elif mutation == "runner_nonzero":
        runner_row["returncode"] = 1
    elif mutation == "runner_not_fresh":
        runner_row["child_result_fresh"] = False
    elif mutation == "runner_stale_eval":
        runner_row["eval_run_id"] = "old-run"
    elif mutation == "result_stale_eval":
        result_payload["eval_run_id"] = "old-run"
    elif mutation == "completed_frames_short":
        result_payload["completed_frames"] = 9
    elif mutation == "warmup_drift":
        result_payload["warmup"] = 3
    elif mutation == "queue_depth_drift":
        result_payload["queue_depth"] = 3
    elif mutation == "completed_work_units_short":
        result_payload["repetition_records"][0]["completed_work_units"] = 9
    elif mutation == "runtime_identity_drift":
        result_payload["repetition_records"][0]["runtime_instance_id"] = (
            "fresh_process:" + "2" * 64
        )
    elif mutation == "nonfinite_fps":
        result_payload["fps_makespan"] = float("nan")
    _write_run(run_dir)
    for relative, payload in payloads.items():
        (run_dir / relative).write_text(json.dumps(payload), encoding="utf-8")

    verified = checker.verify_cache_canary(run_dir)

    assert verified["ok"] is False, mutation
    assert verified["errors"], mutation


def test_collector_success_cannot_mask_failed_canonical_runner(
    tmp_path: Path,
) -> None:
    checker = _checker()
    run_dir = tmp_path / "cache_verify_runner_failed"
    payloads = _payloads(run_dir)
    payloads[RUNNER_REL]["ok"] = False
    payloads[RUNNER_REL]["rows"][0].update({
        "status": "failed",
        "result_ok": False,
        "returncode": 9,
    })
    for relative, payload in payloads.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is False
    assert any("native_runner" in error for error in result["errors"])


def test_redundant_runner_and_result_mirrors_are_diagnostic_only(
    tmp_path: Path,
) -> None:
    checker = _checker()
    run_dir = tmp_path / "cache_verify_redundant_mirror_drift"
    payloads = _payloads(run_dir)
    runner = payloads[RUNNER_REL]
    result_payload = payloads[RESULT_REL]
    runner.update({
        "evidence_status": "stale-mirror",
        "row_count": 99,
        "ok_count": 0,
        "failed_count": 99,
    })
    runner["rows"][0]["fps_makespan"] = 498.0
    result_payload.pop("repetition_count_requested")
    result_payload.pop("repetition_count_attempted")
    result_payload.pop("repetition_count_valid")
    result_payload["repetition_evidence"] = [{
        "status": "stale-diagnostic-mirror",
    }]
    for relative, payload in payloads.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result["errors"]
    assert result["diagnostics"]["redundant_execution_mirrors"][
        "authoritative"
    ] is False


def test_authoritative_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    checker = _checker()
    run_dir = tmp_path / "cache_verify_duplicate_key"
    _write_run(run_dir)
    (run_dir / RUNNER_REL).write_text(
        '{"ok":true,"ok":false}', encoding="utf-8",
    )

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is False
    assert any("duplicate_json_key:ok" in error for error in result["errors"])


def test_authoritative_symlink_is_rejected(tmp_path: Path) -> None:
    checker = _checker()
    run_dir = tmp_path / "cache_verify_symlink"
    _write_run(run_dir)
    result_path = run_dir / RESULT_REL
    outside = tmp_path / "outside-native-result.json"
    outside.write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8")
    result_path.unlink()
    result_path.symlink_to(outside)

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is False
    assert any("native_result_path_invalid" in error for error in result["errors"])


def test_diagnostic_summary_symlink_is_not_followed_or_authoritative(
    tmp_path: Path,
) -> None:
    checker = _checker()
    run_dir = tmp_path / "cache_verify_summary_symlink"
    _write_run(run_dir)
    summary_path = run_dir / "reports/native_producer_summary.json"
    outside = tmp_path / "outside-summary.json"
    outside.write_text('{"ok":false}', encoding="utf-8")
    summary_path.unlink()
    summary_path.symlink_to(outside)

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result["errors"]
    assert result["diagnostics"]["collector_summary"][
        "read_status"
    ].startswith("unsafe_path:")
