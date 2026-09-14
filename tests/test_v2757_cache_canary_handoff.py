from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from onnx_splitpoint_tool.native_command_contract import (
    seal_native_command_contract,
)
from onnx_splitpoint_tool.cache_verify_policy import CACHE_VERIFY_ONLY
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot,
)
from tests.test_v2757_cache_verify_only import PROFILE, _resolved_profile


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SOURCE_BINDING_SHA256 = (
    "a07c35d0f3c23432e93d37751fe58dda930bd0674c85bde3a3c51d7aaa8f2542"
)
HISTORICAL_ARTIFACT_SET_SHA256 = (
    "f1cbf4f2d5c5fce0fc328059b1202fbbf86d9fbb2deb70bf53c2e922e2552c44"
)
VARIANT_ROOT_REL = (
    "native_producers/variants/"
    "v000_v27510_resnet50_b052_hailo8_smoke_cache_verify/hailo8/"
)
NATIVE_RUNNER_REL = (
    VARIANT_ROOT_REL + "analysis_tables/native_fifo_eval_runner.json"
)
NATIVE_RESULT_REL = (
    VARIANT_ROOT_REL + "resnet50/benchmark_set/native_pipeline/"
    "b052/hailo_to_trt/float32_layout_fp16/native_fifo_results.json"
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _load_script(name: str):
    path = ROOT / "scripts" / name
    module_name = f"v27510_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _canary_payloads(run_dir: Path) -> dict[str, dict[str, Any]]:
    variant = "v27510_resnet50_b052_hailo8_smoke_cache_verify"
    matrix_row = {
        "execution_mode": "native_split",
        "backend_key": "hailo8",
        "backend": "hailo8",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "model": "resnet50",
        "case": "b052",
        "variant": variant,
        "actual_ok": True,
        "actual_status": "ok",
    }
    summary_row = {
        "backend": "hailo8_to_trt",
        "comparison_backend": "hailo8",
        "model": "resnet50",
        "case": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "ok": True,
        "status": "ok",
    }
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run_dir.name,
        "profile_start_snapshot": {
            "cache_verify_attestation": {
                "status": "verified",
                "mode": "cache_verify_only",
                "compiler_dispatch_allowed": False,
                "expected_plan": {
                    "run_mode": "smoke",
                    "calibration_items": {
                        "classification": 8,
                        "detection": 8,
                    },
                    "hailo_preset": "smoke",
                    "hailo_optimization_level": 0,
                    "hailo_calib_count": 8,
                    "hailo_calib_batch_size": 8,
                    "hailo_calibration_storage": "memory",
                    "hailo_calibration_memory_cap_mb": 256,
                    "hailo_cache_integrity": "relaxed",
                    "logical_run_profiles": ["hailo8_to_trt"],
                    "native_full_backends": [],
                    "native_rows": [{
                        "model_id": "resnet50",
                        "case_id": "b052",
                        "backend": "hailo8",
                        "setup_id": "orin_nx_hailo8_01",
                        "variant_id": variant,
                    }],
                },
                "actual_plan": {
                    "run_mode": "smoke",
                    "calibration_items": {
                        "classification": 8,
                        "detection": 8,
                    },
                    "hailo_preset": "smoke",
                    "hailo_optimization_level": 0,
                    "hailo_calib_count": 8,
                    "hailo_calib_batch_size": 8,
                    "hailo_calibration_storage": "memory",
                    "hailo_calibration_memory_cap_mb": 256,
                    "hailo_cache_integrity": "relaxed",
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
                        "variant_id": variant,
                    }],
                },
            }
        },
        "root_stages": {
            "run_native_producers": {
                "stage": "run_native_producers",
                "status": "ok",
                "state": "completed",
                "complete": True,
                "details": {
                    "row_count": 1,
                    "ok_count": 1,
                    "orchestration_status": "ok",
                },
            }
        },
    }
    matrix = {
        "schema": "onnx-splitpoint/native-expected-matrix",
        "schema_version": 2,
        "expected_row_count": 1,
        "present_expected_row_count": 1,
        "successful_expected_row_count": 1,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 0,
        "duplicate_expected_identity_count": 0,
        "duplicate_actual_identity_count": 0,
        "invalid_actual_identity_count": 0,
        "unexpected_actual_row_count": 0,
        "row_presence_complete": True,
        "execution_success_complete": True,
        "setup_distribution_match": True,
        "present_expected_rows": [matrix_row],
        "successful_expected_rows": [matrix_row],
        "failed_expected_rows": [],
        "missing_expected_rows": [],
        "invalid_actual_rows": [],
        "unexpected_actual_rows": [],
    }
    summary = {
        "schema": "onnx-splitpoint/native-producer-combined-summary",
        "schema_version": 7,
        "row_count": 1,
        "ok_count": 1,
        "failed_or_unsupported_count": 0,
        "native_full_row_count": 0,
        "evidence_status": "complete",
        "rows": [summary_row],
    }
    stage = {
        "schema": "onnx-splitpoint/native-producer-variant-stage",
        "schema_version": 2,
        "run_id": run_dir.name,
        "status": "ok",
        "state": "completed",
        "complete": True,
        "diagnostic_only": True,
        "claim_eligible": False,
        "variant_results": [{
            "id": variant,
            "case_map": {"resnet50": ["b052"]},
            "performance_repetitions": 1,
            "rc": 0,
            "ok": True,
        }],
        "final_report": {"rc": 0},
        "native_performance_checkpoint": {
            "checkpoint_terminal_valid": True,
            "performance_matrix_complete": True,
            "required_campaign_row_count": 1,
        },
        "native_split_quality_first": {
            "required": False,
            "status": "not_applicable_cache_verify_only",
        },
        "native_validation": {
            "enabled": False,
            "requested": False,
            "status": "not_applicable",
            "rc": 0,
            "complete": True,
        },
        "native_energy": {
            "enabled": False,
            "requested": False,
            "status": "not_applicable",
            "ok": True,
            "complete": True,
        },
        "summary": {
            "variant_count": 1,
            "variant_ok_count": 1,
            "native_report_rc": 0,
        },
    }
    historical_contract = json.loads((
        ROOT / "tests/fixtures/v27510_cache_canary"
        / "resnet50_b052_hailo8_command_contract.json"
    ).read_text(encoding="utf-8"))
    binding = json.loads((
        ROOT / "tests/fixtures/v27510_cache_canary"
        / "resnet50_b052_hailo8_outer_binding.json"
    ).read_text(encoding="utf-8"))
    binding.pop("binding_sha256")
    source_binding_sha = HISTORICAL_SOURCE_BINDING_SHA256
    for field in (
        "producer_binding_sha256",
        "source_request_sha256",
        "central_result_sha256",
        "central_quality_selection",
        "central_quality_selection_sha256",
    ):
        binding.pop(field, None)
    binding.update({
        "eval_run_id": run_dir.name,
        "source_run_id": "hailo8_to_trt",
        "cache_verify_replay": {
            "artifact_policy": "cache_verify_only",
            "source_binding_sha256": source_binding_sha,
            "local_validation_status": (
                "local_files_rehashed_and_exact_cross_links_verified"
            ),
            "compiler_dispatched": False,
        },
    })
    binding["binding_sha256"] = _canonical_sha256(binding)
    artifacts = binding["artifacts"]
    artifact_set_sha = _canonical_sha256({
        role: {
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for role, row in sorted(artifacts.items())
    })
    assert artifact_set_sha == HISTORICAL_ARTIFACT_SET_SHA256
    inner_equivalence_key_sha = _canonical_sha256({
        "binding_sha256": source_binding_sha,
        "artifact_set_sha256": HISTORICAL_ARTIFACT_SET_SHA256,
    })
    cache_root = "/home/nx/splitpoint_runs/_onnx_splitpoint_cache/tensorrt/resnet50-old-smoke"
    inner_binding_path = (
        cache_root
        + "/native_split_quality/orin_nx_hailo8_01/resnet50/b052/"
        "hailo8_to_trt/source/native_split_quality_binding.json"
    )
    attestation = {
        "schema": "onnx-splitpoint/native-split-cache-verify-attestation",
        "schema_version": 1,
        "status": "verified",
        "artifact_policy": "cache_verify_only",
        "compiler_dispatch_allowed": False,
        "compiler_dispatched": False,
        "eval_run_id": run_dir.name,
        "model_id": "resnet50",
        "case_id": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "backend": "hailo8_to_trt",
        "engine_cache_root": str(Path(cache_root).parent),
        "selected_cache_root": cache_root,
        "selection_rule": (
            "historical_binding_and_artifact_set_then_lexicographic_"
            "cache_root"
        ),
        "expected_source_binding_sha256": source_binding_sha,
        "expected_source_binding_artifact_set_sha256": artifact_set_sha,
        "exact_hit_count": 1,
        "distinct_source_binding_sha256_count": 1,
        "distinct_equivalence_key_count": 1,
        "equivalent_cache_roots": [cache_root],
        "searched_cache_roots": [cache_root],
        "cache_root_diagnostics": [{
            "cache_root": cache_root,
            "status": "exact_hit",
            "reason": "verified",
            "mismatch_axes": [],
            "source_binding_sha256": source_binding_sha,
            "artifact_set_sha256": artifact_set_sha,
            "inner_equivalence_key_sha256": inner_equivalence_key_sha,
            "inner_exact_binding_count": 1,
            "inner_equivalent_binding_paths": [inner_binding_path],
            "selected": True,
        }],
        "source_binding_sha256": source_binding_sha,
        "source_binding_artifact_set_sha256": artifact_set_sha,
        "replay_binding_sha256": binding["binding_sha256"],
        "artifact_sha256_by_role": {
            role: row["sha256"] for role, row in artifacts.items()
        },
    }
    attestation["attestation_sha256"] = _canonical_sha256(attestation)
    binding_set = {
        "schema": "onnx-splitpoint/native-split-quality-binding-set",
        "schema_version": 2,
        "mode": "cache_verify_only",
        "diagnostic_only": True,
        "claim_eligible": False,
        "eval_run_id": run_dir.name,
        "setup_id": "orin_nx_hailo8_01",
        "cache_verify_attestation": attestation,
        "cache_verify_attestation_sha256": attestation[
            "attestation_sha256"
        ],
        "bindings_by_model_case_backend": {
            "resnet50|b052|hailo8_to_trt": binding,
        },
    }
    binding_set["binding_set_sha256"] = _canonical_sha256(binding_set)
    command = copy.deepcopy(historical_contract)
    command.pop("contract_sha256")
    for field in (
        "source_request_sha256",
        "native_split_quality_source_request_sha256",
        "native_split_quality_central_result_sha256",
        "native_split_quality_selection_sha256",
    ):
        command.pop(field, None)
    replay_sha = _canonical_sha256(binding["cache_verify_replay"])
    command["runtime_options"].update({
        "frames": 10,
        "warmup": 2,
        "repetitions": 1,
        "queue_depth": 2,
        "duration_s": 0.0,
        "task": "classification",
        "producer_impl": "hailo8_cpp_vstreams_fifo",
    })
    command.update({
        "eval_run_id": run_dir.name,
        "source_run_id": "hailo8_to_trt",
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "native_split_quality_eval_run_id": run_dir.name,
        "native_split_quality_source_run_id": "hailo8_to_trt",
        "native_split_quality_local_verification": copy.deepcopy(
            binding["local_artifact_verification"]
        ),
        "native_split_quality_cache_verify_source_binding_sha256": (
            source_binding_sha
        ),
        "native_split_quality_cache_verify_replay_sha256": replay_sha,
    })
    command = seal_native_command_contract(command)
    consumer_attestation = {
        "schema": (
            "onnx-splitpoint/native-split-quality-consumer-attestation"
        ),
        "schema_version": 1,
        "status": "local_files_rehashed_and_exact_command_join_verified",
        "binding_sha256": binding["binding_sha256"],
        "command_contract_sha256": command["contract_sha256"],
        "eval_run_id": run_dir.name,
        "source_run_id": "hailo8_to_trt",
        "backend": "hailo8_to_trt",
        "model_id": "resnet50",
        "case_id": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "task": "classification",
        "precision": "float32_layout_fp16",
        "local_artifact_verification_sha256": _canonical_sha256(
            binding["local_artifact_verification"]
        ),
        "semantic_output_manifest_sha256": command["artifacts"][
            "semantic_output_manifest"
        ]["sha256"],
        "semantic_boundary_manifest_sha256": command["artifacts"][
            "semantic_boundary_manifest"
        ]["sha256"],
        "native_split_quality_cache_verify_source_binding_sha256": (
            source_binding_sha
        ),
        "native_split_quality_cache_verify_replay_sha256": replay_sha,
    }
    consumer_attestation["attestation_sha256"] = _canonical_sha256(
        consumer_attestation
    )
    fps_makespan = 321.25
    repetition_record = {
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
        "fps_makespan": fps_makespan,
        "runtime_instance_id": "fresh_process:" + "1" * 64,
        "repetition_id": "hailo8:" + "2" * 32,
        "process_local_repetition_index": 1,
        "repetition_index": 1,
        "repetition_runtime_scope": "fresh_process_per_repetition",
        "workload_contract_sha256": command["contract_sha256"],
    }
    native_result = {
        "ok": True,
        "status": "ok",
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
        "eval_run_id": run_dir.name,
        "source_run_id": "hailo8_to_trt",
        "fps_makespan": fps_makespan,
        "repetitions_requested": 1,
        "repetitions_completed": 1,
        "repetition_count_requested": 1,
        "repetition_count_attempted": 1,
        "repetition_count_valid": 1,
        "repetition_status": "complete",
        "repetition_aggregation": "median_never_best_of",
        "repetition_runtime_scope": "fresh_process_per_repetition",
        "repetition_independence_verified": True,
        "runtime_instance_id": repetition_record["runtime_instance_id"],
        "repetition_records": [copy.deepcopy(repetition_record)],
        "repetition_evidence": [copy.deepcopy(repetition_record)],
        "native_command_contract": copy.deepcopy(command),
        "native_command_contract_sha256": command["contract_sha256"],
        "workload_contract_sha256": command["contract_sha256"],
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "native_split_quality_eval_run_id": run_dir.name,
        "native_split_quality_source_run_id": "hailo8_to_trt",
        "native_split_quality_cache_verify_source_binding_sha256": (
            source_binding_sha
        ),
        "native_split_quality_cache_verify_replay_sha256": replay_sha,
        "native_split_quality_consumer_attestation": copy.deepcopy(
            consumer_attestation
        ),
        "native_split_quality_consumer_status": (
            "exact_quality_native_engine_command_and_boundary_match"
        ),
        "native_split_quality_runtime_boundary_verified": True,
        "hailo_runtime_output_count": 1,
        "hailo_runtime_output_name": "resnet50_part1_b52/conv24",
        "hailo_runtime_output_frame_bytes": 1_605_632,
    }
    native_result_rel = NATIVE_RESULT_REL
    native_result_path = run_dir / native_result_rel
    native_result_bytes = json.dumps(native_result).encode("utf-8")
    native_result_sha = hashlib.sha256(native_result_bytes).hexdigest()
    summary_row.update({
        "task": "classification",
        "precision": "float32_layout_fp16",
        "eval_run_id": run_dir.name,
        "source_run_id": "hailo8_to_trt",
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "native_split_quality_eval_run_id": run_dir.name,
        "native_split_quality_source_run_id": "hailo8_to_trt",
        "native_split_quality_cache_verify_source_binding_sha256": (
            source_binding_sha
        ),
        "native_split_quality_cache_verify_replay_sha256": (
            replay_sha
        ),
        "native_command_contract": command,
        "native_command_contract_sha256": command["contract_sha256"],
        "workload_contract_sha256": command["contract_sha256"],
        "native_split_quality_consumer_attestation": (
            consumer_attestation
        ),
        "native_split_quality_consumer_status": (
            "exact_quality_native_engine_command_and_boundary_match"
        ),
        "native_split_quality_required": True,
        "performance_claims_emitted": False,
        "execution_role": "cache_verify_diagnostic_replay",
        "frames": 10,
        "warmup": 2,
        "fps_makespan": fps_makespan,
        "repetition_count_requested": 1,
        "repetition_count_attempted": 1,
        "repetition_count_valid": 1,
        "repetition_status": "complete",
        "repetition_aggregation": (
            "median_with_deterministic_percentile_bootstrap_ci95"
        ),
        "repetition_records": [copy.deepcopy(repetition_record)],
        "performance_repetitions": [copy.deepcopy(repetition_record)],
        "report": str(native_result_path),
        "native_fifo_result_sha256": native_result_sha,
        "native_fifo_result_size_bytes": len(native_result_bytes),
        "child_result_fresh": True,
        "returncode": 0,
        "timed_out": False,
    })
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
            "eval_run_id": run_dir.name,
            "source_run_id": "hailo8_to_trt",
            "fps_makespan": fps_makespan,
            "native_fifo_result": str(native_result_path),
            "native_fifo_result_sha256": native_result_sha,
            "native_fifo_result_size_bytes": len(native_result_bytes),
        }],
    }
    return {
        "run_manifest.json": manifest,
        "reports/native_expected_matrix.json": matrix,
        "reports/native_producer_summary.json": summary,
        "reports/native_producer_stage.json": stage,
        "native_producers/variants/"
        "v000_v27510_resnet50_b052_hailo8_smoke_cache_verify/"
        "hailo8/cache_verify_native_split/orin_nx_hailo8_01/"
        "native_split_quality_binding_set.json": binding_set,
        NATIVE_RUNNER_REL: runner,
        native_result_rel: native_result,
    }


def _refresh_native_result_reference(
    payloads: dict[str, dict[str, Any]], run_dir: Path,
) -> None:
    native_result = payloads[NATIVE_RESULT_REL]
    native_result_bytes = json.dumps(native_result).encode("utf-8")
    row = payloads["reports/native_producer_summary.json"]["rows"][0]
    row["report"] = str(run_dir / NATIVE_RESULT_REL)
    row["native_fifo_result_sha256"] = hashlib.sha256(
        native_result_bytes
    ).hexdigest()
    row["native_fifo_result_size_bytes"] = len(native_result_bytes)


def _write_canary(run_dir: Path, payloads: dict[str, dict[str, Any]]) -> None:
    for relative, payload in payloads.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_exact_canary_postcondition_accepts_current_writer_shapes(
    tmp_path: Path,
) -> None:
    checker = _load_script("verify_cache_canary_result.py")
    run_dir = tmp_path / "cache_verify_exact"
    _write_canary(run_dir, _canary_payloads(run_dir))

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result["errors"]
    assert result["status"] == "semantic_cache_canary_pass"


def test_exact_canary_accepts_actual_wrapper_and_collector_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _load_script("verify_cache_canary_result.py")
    wrapper = _load_script("native_fifo_eval_runner.py")
    collector = _load_script("native_producer_final_report.py")
    run_dir = tmp_path / "cache_verify_projected_writer_chain"
    payloads = _canary_payloads(run_dir)
    _write_canary(run_dir, payloads)
    raw = payloads[NATIVE_RESULT_REL]
    result_path = run_dir / NATIVE_RESULT_REL
    output_manifest = (
        result_path.parent
        / "native_fifo_outputs/native_fifo_outputs_manifest.json"
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps({
        "task": "classification",
    }), encoding="utf-8")
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY,
    )
    matrix_case = {
        "case_id": "b052",
        "status": "ok",
        "result_ok": True,
        "returncode": 0,
        "timed_out": False,
        "child_result_fresh": True,
        "native_fifo_result": str(result_path),
        "native_fifo_result_sha256": hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest(),
        "native_fifo_result_size_bytes": result_path.stat().st_size,
        "native_split_quality_required": True,
        "performance_claims_emitted": False,
        "execution_role": "cache_verify_diagnostic_replay",
        "setup_id": "orin_nx_hailo8_01",
        "eval_run_id": run_dir.name,
        "source_run_id": "hailo8_to_trt",
        "native_split_quality_binding": copy.deepcopy(
            raw["native_split_quality_binding"]
        ),
        "native_split_quality_binding_sha256": raw[
            "native_split_quality_binding_sha256"
        ],
        "native_split_quality_eval_run_id": run_dir.name,
        "native_split_quality_source_run_id": "hailo8_to_trt",
        "native_split_quality_cache_verify_source_binding_sha256": raw[
            "native_split_quality_cache_verify_source_binding_sha256"
        ],
        "native_split_quality_cache_verify_replay_sha256": raw[
            "native_split_quality_cache_verify_replay_sha256"
        ],
        "native_split_quality_consumer_attestation": copy.deepcopy(
            raw["native_split_quality_consumer_attestation"]
        ),
        "native_split_quality_consumer_status": raw[
            "native_split_quality_consumer_status"
        ],
    }
    benchmark_set = (
        run_dir / "native_producers/hailo8/resnet50/benchmark_set"
    )
    eval_rows = wrapper._extract_rows(
        "resnet50", benchmark_set, {"cases": [matrix_case]},
        setup_id="orin_nx_hailo8_01",
    )
    analysis = (
        run_dir / "native_producers/hailo8/analysis_tables/"
        "native_fifo_eval_runner__cache.json"
    )
    analysis.parent.mkdir(parents=True, exist_ok=True)
    analysis.write_text(json.dumps({
        "precision": "float32_layout_fp16",
        "rows": eval_rows,
    }), encoding="utf-8")
    collected = collector._rows_from_native_fifo_runner(
        run_dir / "native_producers/hailo8"
    )
    final_rows = collector._aggregate_repetitions(collected)
    assert len(final_rows) == 1 and final_rows[0]["ok"] is True
    summary = {
        "schema": "onnx-splitpoint/native-producer-combined-summary",
        "schema_version": 7,
        "row_count": 1,
        "ok_count": 1,
        "failed_or_unsupported_count": 0,
        "native_full_row_count": 0,
        "evidence_status": "complete",
        "rows": final_rows,
    }
    (run_dir / "reports/native_producer_summary.json").write_text(
        json.dumps(summary), encoding="utf-8",
    )

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result["errors"]


def test_exact_canary_postcondition_accepts_two_equivalent_cache_roots(
    tmp_path: Path,
) -> None:
    checker = _load_script("verify_cache_canary_result.py")
    run_dir = tmp_path / "cache_verify_two_equivalent_roots"
    payloads = _canary_payloads(run_dir)
    binding_set_key = next(
        key for key in payloads
        if key.endswith("native_split_quality_binding_set.json")
    )
    binding_set = payloads[binding_set_key]
    attestation = binding_set["cache_verify_attestation"]
    first = attestation["cache_root_diagnostics"][0]
    second_root = (
        "/home/nx/splitpoint_runs/_onnx_splitpoint_cache/tensorrt/"
        "resnet50-z-smoke-copy"
    )
    second_inner = (
        second_root
        + "/native_split_quality/orin_nx_hailo8_01/resnet50/b052/"
        "hailo8_to_trt/source/native_split_quality_binding.json"
    )
    second = copy.deepcopy(first)
    second.update({
        "cache_root": second_root,
        "inner_exact_binding_count": 1,
        "inner_equivalent_binding_paths": [second_inner],
        "selected": False,
    })
    attestation["searched_cache_roots"].append(second_root)
    attestation["cache_root_diagnostics"].append(second)
    attestation["equivalent_cache_roots"].append(second_root)
    attestation["exact_hit_count"] = 2
    attestation.pop("attestation_sha256")
    attestation["attestation_sha256"] = _canonical_sha256(attestation)
    binding_set["cache_verify_attestation_sha256"] = attestation[
        "attestation_sha256"
    ]
    binding_set.pop("binding_set_sha256")
    binding_set["binding_set_sha256"] = _canonical_sha256(binding_set)
    _write_canary(run_dir, payloads)

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result


def test_exact_canary_accepts_historical_root_with_distinct_root_excluded(
    tmp_path: Path,
) -> None:
    checker = _load_script("verify_cache_canary_result.py")
    run_dir = tmp_path / "cache_verify_372_exact_907_distinct"
    payloads = _canary_payloads(run_dir)
    binding_set_key = next(
        key for key in payloads
        if key.endswith("native_split_quality_binding_set.json")
    )
    binding_set = payloads[binding_set_key]
    attestation = binding_set["cache_verify_attestation"]
    second_root = (
        "/home/nx/splitpoint_runs/_onnx_splitpoint_cache/tensorrt/"
        "resnet50-9074d00f2ca3fcc1"
    )
    attestation["searched_cache_roots"].append(second_root)
    attestation["cache_root_diagnostics"].append({
        "cache_root": second_root,
        "status": "no_exact_hit",
        "reason": "historical_cache_identity_mismatch",
        "mismatch_axes": [
            "source_binding_sha256", "artifact_set_sha256",
        ],
    })
    attestation.pop("attestation_sha256")
    attestation["attestation_sha256"] = _canonical_sha256(attestation)
    binding_set["cache_verify_attestation_sha256"] = attestation[
        "attestation_sha256"
    ]
    binding_set.pop("binding_set_sha256")
    binding_set["binding_set_sha256"] = _canonical_sha256(binding_set)
    _write_canary(run_dir, payloads)

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_key",
        "wrong_key",
        "zero_count",
        "empty_path",
        "duplicate_paths",
        "unsorted_paths",
        "outside_cache_root",
    ),
)
def test_inner_equivalence_receipt_drift_is_diagnostic_only(
    tmp_path: Path,
    mutation: str,
) -> None:
    checker = _load_script("verify_cache_canary_result.py")
    run_dir = tmp_path / ("inner-" + mutation)
    payloads = _canary_payloads(run_dir)
    binding_set_path = (
        "native_producers/variants/"
        "v000_v27510_resnet50_b052_hailo8_smoke_cache_verify/"
        "hailo8/cache_verify_native_split/orin_nx_hailo8_01/"
        "native_split_quality_binding_set.json"
    )
    binding_set = payloads[binding_set_path]
    attestation = binding_set["cache_verify_attestation"]
    row = attestation["cache_root_diagnostics"][0]
    if mutation == "missing_key":
        row.pop("inner_equivalence_key_sha256")
    elif mutation == "wrong_key":
        row["inner_equivalence_key_sha256"] = "f" * 64
    elif mutation == "zero_count":
        row["inner_exact_binding_count"] = 0
    elif mutation == "empty_path":
        row["inner_equivalent_binding_paths"] = [""]
    elif mutation == "duplicate_paths":
        path = row["inner_equivalent_binding_paths"][0]
        row["inner_exact_binding_count"] = 2
        row["inner_equivalent_binding_paths"] = [path, path]
    elif mutation == "unsorted_paths":
        path = row["inner_equivalent_binding_paths"][0]
        row["inner_exact_binding_count"] = 2
        row["inner_equivalent_binding_paths"] = [path + "/z", path + "/a"]
    elif mutation == "outside_cache_root":
        row["inner_equivalent_binding_paths"] = [
            "/tmp/unrelated-root/forged-binding.json"
        ]
    attestation.pop("attestation_sha256")
    attestation["attestation_sha256"] = _canonical_sha256(attestation)
    binding_set["cache_verify_attestation_sha256"] = attestation[
        "attestation_sha256"
    ]
    binding_set.pop("binding_set_sha256")
    binding_set["binding_set_sha256"] = _canonical_sha256(binding_set)
    _write_canary(run_dir, payloads)

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is True, result["errors"]
    assert result["diagnostics"]["cache_selection"][
        "authoritative"
    ] is False


def test_exact_canary_postcondition_rejects_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    checker = _load_script("verify_cache_canary_result.py")
    run_dir = tmp_path / "duplicate-key"
    _write_canary(run_dir, _canary_payloads(run_dir))
    runner_path = run_dir / NATIVE_RUNNER_REL
    runner_path.write_text(
        '{"ok":true,"ok":false}',
        encoding="utf-8",
    )

    result = checker.verify_cache_canary(run_dir)

    assert result["ok"] is False
    assert any("duplicate_json_key:ok" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("stage_partial", ""),
        ("matrix_failed", ""),
        ("extra_summary", ""),
        ("wrong_case", ""),
        ("energy_enabled", ""),
        ("quality_required", ""),
        ("checkpoint_incomplete", ""),
        (
            "manifest_unverified",
            "manifest_cache_verify_or_compiler_fence_invalid",
        ),
        ("standard_cache_plan", "manifest.actual_plan.hailo_build_mode"),
        ("missing_replay_binding_set", "cache_binding_set_path_invalid"),
        (
            "cache_compiler_dispatched",
            "cache_binding_attestation_or_compiler_fence_invalid",
        ),
        ("cache_multiple_exact_roots", ""),
        ("cache_source_binding_drift", ""),
        ("summary_replay_binding_drift", ""),
        (
            "cache_binding_portable_invalid",
            "cache_binding.preselection",
        ),
        ("summary_consumer_command_drift", ""),
        ("summary_metrics_missing", ""),
        ("raw_result_command_options_drift", ""),
        ("raw_result_repetition_stale_command", ""),
        ("raw_result_file_hash_drift", ""),
        ("child_result_not_fresh", ""),
        (
            "raw_result_metrics_missing",
            "native_result.completed_frames",
        ),
        ("raw_result_outside_run", ""),
    ),
)
def test_semantic_canary_separates_hard_gates_from_diagnostics(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    checker = _load_script("verify_cache_canary_result.py")
    run_dir = tmp_path / mutation
    payloads = _canary_payloads(run_dir)
    if mutation == "stage_partial":
        payloads["reports/native_producer_stage.json"]["status"] = "partial"
    elif mutation == "matrix_failed":
        matrix = payloads["reports/native_expected_matrix.json"]
        matrix["successful_expected_row_count"] = 0
        matrix["failed_expected_row_count"] = 1
    elif mutation == "extra_summary":
        summary = payloads["reports/native_producer_summary.json"]
        summary["rows"].append(copy.deepcopy(summary["rows"][0]))
        summary["row_count"] = 2
        summary["ok_count"] = 2
    elif mutation == "wrong_case":
        payloads["reports/native_producer_summary.json"]["rows"][0]["case"] = "b053"
    elif mutation == "energy_enabled":
        energy = payloads["reports/native_producer_stage.json"]["native_energy"]
        energy.update({"enabled": True, "status": "ok"})
    elif mutation == "quality_required":
        quality = payloads["reports/native_producer_stage.json"]["native_split_quality_first"]
        quality.update({"required": True, "status": "required"})
    elif mutation == "checkpoint_incomplete":
        checkpoint = payloads["reports/native_producer_stage.json"]["native_performance_checkpoint"]
        checkpoint["performance_matrix_complete"] = False
    elif mutation == "manifest_unverified":
        attestation = payloads["run_manifest.json"]["profile_start_snapshot"]["cache_verify_attestation"]
        attestation["status"] = "invalid"
    elif mutation == "standard_cache_plan":
        attestation = payloads["run_manifest.json"]["profile_start_snapshot"]["cache_verify_attestation"]
        attestation["actual_plan"]["hailo_build_mode"] = "standard"
    elif mutation == "missing_replay_binding_set":
        payloads.pop(
            "native_producers/variants/"
            "v000_v27510_resnet50_b052_hailo8_smoke_cache_verify/"
            "hailo8/cache_verify_native_split/orin_nx_hailo8_01/"
            "native_split_quality_binding_set.json"
        )
    elif mutation == "cache_compiler_dispatched":
        binding_set = payloads[
            "native_producers/variants/"
            "v000_v27510_resnet50_b052_hailo8_smoke_cache_verify/"
            "hailo8/cache_verify_native_split/orin_nx_hailo8_01/"
            "native_split_quality_binding_set.json"
        ]
        binding_set["cache_verify_attestation"]["compiler_dispatched"] = True
    elif mutation == "cache_multiple_exact_roots":
        binding_set = payloads[
            "native_producers/variants/"
            "v000_v27510_resnet50_b052_hailo8_smoke_cache_verify/"
            "hailo8/cache_verify_native_split/orin_nx_hailo8_01/"
            "native_split_quality_binding_set.json"
        ]
        attestation = binding_set["cache_verify_attestation"]
        second = "/home/nx/splitpoint_runs/_onnx_splitpoint_cache/tensorrt/resnet50-smoke-copy"
        attestation["searched_cache_roots"].append(second)
        attestation["cache_root_diagnostics"].append({
            "cache_root": second,
            "status": "exact_hit",
            "reason": "verified",
            "mismatch_axes": [],
        })
    elif mutation == "cache_source_binding_drift":
        binding_set = payloads[
            "native_producers/variants/"
            "v000_v27510_resnet50_b052_hailo8_smoke_cache_verify/"
            "hailo8/cache_verify_native_split/orin_nx_hailo8_01/"
            "native_split_quality_binding_set.json"
        ]
        binding = binding_set["bindings_by_model_case_backend"][
            "resnet50|b052|hailo8_to_trt"
        ]
        binding["cache_verify_replay"]["source_binding_sha256"] = "f" * 64
    elif mutation == "summary_replay_binding_drift":
        row = payloads["reports/native_producer_summary.json"]["rows"][0]
        row["native_split_quality_binding_sha256"] = "f" * 64
    elif mutation == "cache_binding_portable_invalid":
        binding_set_path = (
            "native_producers/variants/"
            "v000_v27510_resnet50_b052_hailo8_smoke_cache_verify/"
            "hailo8/cache_verify_native_split/orin_nx_hailo8_01/"
            "native_split_quality_binding_set.json"
        )
        binding_set = payloads[binding_set_path]
        binding = binding_set["bindings_by_model_case_backend"][
            "resnet50|b052|hailo8_to_trt"
        ]
        binding.pop("preselection")
        binding.pop("binding_sha256")
        binding["binding_sha256"] = _canonical_sha256(binding)
        attestation = binding_set["cache_verify_attestation"]
        attestation["replay_binding_sha256"] = binding["binding_sha256"]
        attestation.pop("attestation_sha256")
        attestation["attestation_sha256"] = _canonical_sha256(attestation)
        binding_set["cache_verify_attestation_sha256"] = attestation[
            "attestation_sha256"
        ]
        binding_set.pop("binding_set_sha256")
        binding_set["binding_set_sha256"] = _canonical_sha256(binding_set)
        summary_row = payloads[
            "reports/native_producer_summary.json"
        ]["rows"][0]
        summary_row["native_split_quality_binding"] = copy.deepcopy(binding)
        summary_row["native_split_quality_binding_sha256"] = binding[
            "binding_sha256"
        ]
    elif mutation == "summary_consumer_command_drift":
        row = payloads["reports/native_producer_summary.json"]["rows"][0]
        command = copy.deepcopy(row["native_command_contract"])
        command.pop("contract_sha256")
        command["artifacts"]["engine"]["sha256"] = "f" * 64
        command = seal_native_command_contract(command)
        row["native_command_contract"] = command
        row["native_command_contract_sha256"] = command["contract_sha256"]
        consumer = row["native_split_quality_consumer_attestation"]
        consumer.pop("attestation_sha256")
        consumer["command_contract_sha256"] = command["contract_sha256"]
        consumer["attestation_sha256"] = _canonical_sha256(consumer)
    elif mutation == "summary_metrics_missing":
        row = payloads["reports/native_producer_summary.json"]["rows"][0]
        row.pop("fps_makespan")
    elif mutation == "raw_result_command_options_drift":
        result_payload = payloads[NATIVE_RESULT_REL]
        command = copy.deepcopy(result_payload["native_command_contract"])
        command.pop("contract_sha256")
        command["runtime_options"]["frames"] = 11
        command = seal_native_command_contract(command)
        result_payload["native_command_contract"] = copy.deepcopy(command)
        result_payload["native_command_contract_sha256"] = command[
            "contract_sha256"
        ]
        result_payload["workload_contract_sha256"] = command[
            "contract_sha256"
        ]
        for records_key in ("repetition_records", "repetition_evidence"):
            result_payload[records_key][0]["workload_contract_sha256"] = (
                command["contract_sha256"]
            )
        consumer = result_payload[
            "native_split_quality_consumer_attestation"
        ]
        consumer.pop("attestation_sha256")
        consumer["command_contract_sha256"] = command["contract_sha256"]
        consumer["attestation_sha256"] = _canonical_sha256(consumer)
        row = payloads["reports/native_producer_summary.json"]["rows"][0]
        row["native_command_contract"] = copy.deepcopy(command)
        row["native_command_contract_sha256"] = command["contract_sha256"]
        row["workload_contract_sha256"] = command["contract_sha256"]
        row["native_split_quality_consumer_attestation"] = copy.deepcopy(
            consumer
        )
        row["repetition_records"][0]["workload_contract_sha256"] = (
            command["contract_sha256"]
        )
        _refresh_native_result_reference(payloads, run_dir)
    elif mutation == "raw_result_repetition_stale_command":
        result_payload = payloads[NATIVE_RESULT_REL]
        for records_key in ("repetition_records", "repetition_evidence"):
            result_payload[records_key][0]["workload_contract_sha256"] = (
                "f" * 64
            )
        row = payloads["reports/native_producer_summary.json"]["rows"][0]
        row["repetition_records"][0]["workload_contract_sha256"] = "f" * 64
        _refresh_native_result_reference(payloads, run_dir)
    elif mutation == "raw_result_file_hash_drift":
        row = payloads["reports/native_producer_summary.json"]["rows"][0]
        row["native_fifo_result_sha256"] = "f" * 64
    elif mutation == "child_result_not_fresh":
        row = payloads["reports/native_producer_summary.json"]["rows"][0]
        row["child_result_fresh"] = False
    elif mutation == "raw_result_metrics_missing":
        payloads[NATIVE_RESULT_REL].pop("completed_frames")
        _refresh_native_result_reference(payloads, run_dir)
    elif mutation == "raw_result_outside_run":
        row = payloads["reports/native_producer_summary.json"]["rows"][0]
        row["report"] = "/tmp/forged-native_fifo_results.json"
    _write_canary(run_dir, payloads)

    result = checker.verify_cache_canary(run_dir)

    if expected_error:
        assert result["ok"] is False
        assert any(
            expected_error in error for error in result["errors"]
        ), result
    else:
        assert result["ok"] is True, result["errors"]


def test_cache_canary_uses_attested_single_row_not_mode_default_matrix() -> None:
    variants = _load_script("run_evalrun_native_producer_variants.py")
    cfg = {
        "cache_verify_only": True,
        "cache_verify_expected_plan": {
            "native_rows": [{
                "model_id": "resnet50",
                "case_id": "b052",
                "backend": "hailo8",
            }]
        },
        "_workflow_context": {"execution_preset": {"id": "smoke"}},
    }

    assert variants._native_performance_required_campaign_rows(cfg) == 1

    cfg["native_performance_checkpoint"] = {"required_row_count": 63}
    with pytest.raises(ValueError, match="drifts"):
        variants._native_performance_required_campaign_rows(cfg)


def _manual_args(*, dump_outputs: bool, native_validation: bool) -> Namespace:
    return Namespace(
        artifact_policy="normal",
        hailo8_ssh="", hailo10_ssh="", deepx_ssh="",
        hailo8_env="", hailo10_env="", deepx_env="",
        hailo8_setup_id="", hailo10_setup_id="", deepx_setup_id="",
        backends="hailo8", case_policy="all_accepted", precision="fp16",
        frames=10, warmup=2, repetitions=1, queue_depth=2, inflight=4,
        hailo_format="float32", native_letterbox_pad_value=0,
        native_force_rebuild_engines=False, native_dequant_scale=0.0,
        native_dequant_zero_point=0.0, native_boundary_layout="as_input",
        remote_root="/remote", remote_tool_dir="/tool",
        dump_outputs=dump_outputs, native_boundary_debug=False,
        no_build_missing_engines=True, engine_build_python="auto", no_copy=False,
        native_full_baselines=False, native_full_backends="",
        native_full_backends_by_producer="", trt_quality_producer_sets="",
        native_energy=False, native_energy_mode="", native_energy_timeout=900,
        native_energy_runs=1, native_energy_allow_unpaired=False,
        native_energy_duration_s=0.0, native_validation=native_validation,
        native_validation_mode="dump_and_visual", native_validation_topk=5,
        case_map="", native_telemetry_label="cache",
        native_split_quality_required=False,
        native_split_quality_binding_sets="",
        smoke_diagnostic_quality_continue=False,
        native_execution_contract_json="",
    )


def test_dump_outputs_does_not_enable_native_validation() -> None:
    updater = _load_script("update_evalset_native_producers.py")

    cfg = updater._native_cfg_from_args(
        _manual_args(dump_outputs=True, native_validation=False)
    )

    assert cfg["dump_outputs"] is True
    assert cfg["dump_boundary"] is False
    assert "validation" not in cfg


def test_explicit_fresh_run_id_is_allowed_only_for_cache_canary() -> None:
    options = WorkflowOptions(
        profile=str(PROFILE),
        out="/unused",
        run_id="cache_verify_explicit",
        execution_mode="generate_and_run",
        artifact_policy=CACHE_VERIFY_ONLY,
        require_fresh_run=True,
        required_run_mode="smoke",
    )
    runner = EvaluationWorkflowRunner(options)
    runner.profile_payload = _resolved_profile()

    runner._validate_requested_start_contract()


def test_profile_driven_cli_forwards_exact_cache_canary_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from onnx_splitpoint_tool.workflow import run_evaluation

    run_id = "cache_verify_resnet50_b052_hailo8_exact"
    captured: dict[str, WorkflowOptions] = {}

    class _Runner:
        def __init__(self, options: WorkflowOptions, **_kwargs: object) -> None:
            captured["options"] = options

        def run(self):
            return type(
                "Result",
                (),
                {
                    "ok": True,
                    "status": "ok",
                    "run_dir": tmp_path / run_id,
                    "manifest_path": tmp_path / run_id / "run_manifest.json",
                    "to_dict": lambda self: {"ok": True, "status": "ok"},
                },
            )()

    monkeypatch.setattr(run_evaluation, "EvaluationWorkflowRunner", _Runner)

    rc = run_evaluation.main(
        [
            "--profile",
            str(PROFILE),
            "--out",
            str(tmp_path),
            "--run-id",
            run_id,
            "--profile-driven",
            "--require-run-mode",
            "smoke",
            "--require-fresh-run",
            "--execution-mode",
            "generate_and_run",
        ]
    )

    assert rc == 0, capsys.readouterr().out
    options = captured["options"]
    assert options.run_id == run_id
    assert options.artifact_policy == CACHE_VERIFY_ONLY
    assert options.require_fresh_run is True


def test_profile_driven_run_id_override_is_rejected_for_normal_profile() -> None:
    from onnx_splitpoint_tool.workflow.profile_options import (
        workflow_options_from_profile_snapshot,
    )

    profile = _resolved_profile()
    profile.pop("execution_guard", None)
    snapshot = build_profile_start_snapshot(
        profile_request="normal.yaml",
        source_profile=profile,
        resolved_profile=profile,
        profile_id="normal",
        profile_path="normal.yaml",
        profile_source="test",
        runtime_bindings={},
    )

    with pytest.raises(ValueError, match="reserved for an attested"):
        workflow_options_from_profile_snapshot(
            profile_request="normal.yaml",
            out_root="/tmp/runs",
            start_snapshot=snapshot,
            run_id_override="must_not_be_accepted",
        )


def test_profile_driven_cache_canary_rejects_empty_run_id_override() -> None:
    from onnx_splitpoint_tool.workflow.profile_options import (
        load_runtime_profile_snapshot,
        workflow_options_from_profile_snapshot,
    )

    _profile, snapshot = load_runtime_profile_snapshot(str(PROFILE))

    with pytest.raises(ValueError, match="must not be empty"):
        workflow_options_from_profile_snapshot(
            profile_request=str(PROFILE),
            out_root="/tmp/runs",
            start_snapshot=snapshot,
            run_id_override="",
        )


def test_cache_canary_never_schedules_management_ort(
    tmp_path: Path,
) -> None:
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile=str(PROFILE), out=str(tmp_path))
    )
    runner.profile_payload = _resolved_profile()

    runner._schedule_management_cpu_reference("resnet50", tmp_path)

    assert runner._management_reference_executor is None
    assert runner._management_reference_futures == {}


def test_remote_cache_replay_materializes_one_diagnostic_binding_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_script(
        "materialize_cache_verify_native_split_binding.py"
    )
    benchmark_set = tmp_path / "benchmark_set"
    (benchmark_set / "b052").mkdir(parents=True)
    cache_parent = tmp_path / "engine-cache"
    (cache_parent / "resnet-cache/native_split_quality").mkdir(
        parents=True
    )
    replay = {
        "artifact_policy": CACHE_VERIFY_ONLY,
        "source_binding_sha256": HISTORICAL_SOURCE_BINDING_SHA256,
        "local_validation_status": (
            "local_files_rehashed_and_exact_cross_links_verified"
        ),
        "compiler_dispatched": False,
    }
    binding = {
        "binding_sha256": "2" * 64,
        "eval_run_id": "cache-run",
        "cache_verify_replay": replay,
        "artifacts": {
            "part1_runtime": {"sha256": "1" * 64, "size_bytes": 1},
            "boundary_metadata": {"sha256": "2" * 64, "size_bytes": 2},
            "source_part2_onnx": {"sha256": "3" * 64, "size_bytes": 3},
            "build_part2_onnx": {"sha256": "4" * 64, "size_bytes": 4},
            "engine": {"sha256": "5" * 64, "size_bytes": 5},
            "native_trt_meta": {"sha256": "6" * 64, "size_bytes": 6},
            "engine_build_receipt": {
                "sha256": "7" * 64, "size_bytes": 7,
            },
            "trtexec": {"sha256": "8" * 64, "size_bytes": 8},
        },
    }
    calls: list[Path] = []

    def fake_prepare(**kwargs: Any) -> dict[str, Any]:
        cache_root = Path(kwargs["cache_root"])
        calls.append(cache_root)
        replay_binding = copy.deepcopy(binding)
        artifact_set_sha = helper._artifact_set_sha256(replay_binding)
        source_binding_sha = replay_binding["cache_verify_replay"][
            "source_binding_sha256"
        ]
        persistent_binding_path = str(
            cache_root
            / "native_split_quality/source/native_split_quality_binding.json"
        )
        return {
            "binding": replay_binding,
            "persistent_binding_path": persistent_binding_path,
            "cache_verify_source_binding_sha256": source_binding_sha,
            "cache_verify_source_artifact_set_sha256": artifact_set_sha,
            "cache_verify_equivalence_key_sha256": _canonical_sha256({
                "binding_sha256": source_binding_sha,
                "artifact_set_sha256": artifact_set_sha,
            }),
            "cache_verify_exact_binding_count": 1,
            "cache_verify_equivalent_binding_paths": [
                persistent_binding_path
            ],
        }

    monkeypatch.setattr(
        helper, "prepare_native_split_quality_binding", fake_prepare
    )
    monkeypatch.setattr(
        helper,
        "validate_native_split_quality_binding",
        lambda value, **_kwargs: (copy.deepcopy(value), "local_ok"),
    )
    output = tmp_path / "run/binding-set.json"

    result = helper.materialize(
        benchmark_set=benchmark_set,
        model_id="resnet50",
        case_id="b052",
        setup_id="orin_nx_hailo8_01",
        backend="hailo8_to_trt",
        eval_run_id="cache-run",
        engine_cache_root=cache_parent,
        output=output,
    )

    assert calls == [(cache_parent / "resnet-cache").resolve()]
    assert result["mode"] == CACHE_VERIFY_ONLY
    assert result["diagnostic_only"] is True
    assert result["claim_eligible"] is False
    assert list(result["bindings_by_model_case_backend"]) == [
        "resnet50|b052|hailo8_to_trt"
    ]
    assert result["cache_verify_attestation"][
        "compiler_dispatch_allowed"
    ] is False
    assert json.loads(output.read_text(encoding="utf-8")) == result

    matrix = _load_script("native_fifo_smoke_matrix.py")
    monkeypatch.setattr(
        matrix,
        "validate_native_split_quality_binding",
        lambda value, **_kwargs: (value, "portable_ok"),
    )
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY
    )
    assert matrix._valid_quality_first_binding_set(
        result, setup_id="orin_nx_hailo8_01"
    )


def test_cache_replay_wrapper_requires_cache_mirrors_not_central_shas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _load_script("native_fifo_eval_runner.py")
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_ARTIFACT_POLICY", CACHE_VERIFY_ONLY
    )
    result_path = tmp_path / "native_fifo_results.json"
    binding = {"binding_sha256": "a" * 64}
    consumer = {"attestation_sha256": "b" * 64}
    shared = {
        "eval_run_id": "cache-run",
        "source_run_id": "hailo8_to_trt",
        "native_split_quality_binding_sha256": "a" * 64,
        "native_split_quality_eval_run_id": "cache-run",
        "native_split_quality_source_run_id": "hailo8_to_trt",
        "native_split_quality_cache_verify_source_binding_sha256": "c" * 64,
        "native_split_quality_cache_verify_replay_sha256": "d" * 64,
        "native_split_quality_binding": binding,
        "native_split_quality_consumer_attestation": consumer,
        "native_split_quality_consumer_status": (
            "exact_quality_native_engine_command_and_boundary_match"
        ),
    }
    result_path.write_text(
        json.dumps({"ok": True, **shared}), encoding="utf-8"
    )
    matrix = {
        "cases": [{
            "case_id": "b052",
            "status": "ok",
            "result_ok": True,
            "returncode": 0,
            "timed_out": False,
            "child_result_fresh": True,
            "native_fifo_result": str(result_path),
            "native_fifo_result_sha256": hashlib.sha256(
                result_path.read_bytes()
            ).hexdigest(),
            "native_fifo_result_size_bytes": result_path.stat().st_size,
            "native_split_quality_required": True,
            "performance_claims_emitted": False,
            "execution_role": "cache_verify_diagnostic_replay",
            **shared,
        }]
    }

    rows = wrapper._extract_rows(
        "resnet50", tmp_path, matrix,
        setup_id="orin_nx_hailo8_01",
    )

    assert len(rows) == 1
    assert rows[0]["result_ok"] is True
    assert rows[0]["status"] == "ok"
    assert rows[0]["performance_claims_emitted"] is False
    assert rows[0]["execution_role"] == (
        "cache_verify_diagnostic_replay"
    )
    final_report = _load_script("native_producer_final_report.py")
    projected = final_report._split_quality_fields(rows[0])
    assert projected["native_split_quality_binding"] == binding
    assert projected["native_split_quality_binding_sha256"] == "a" * 64
    assert projected[
        "native_split_quality_cache_verify_source_binding_sha256"
    ] == "c" * 64
    assert projected[
        "native_split_quality_cache_verify_replay_sha256"
    ] == "d" * 64
    assert projected["native_split_quality_required"] is True
    assert projected["performance_claims_emitted"] is False
    assert projected[
        "native_split_quality_consumer_attestation"
    ] == consumer
    assert projected["native_split_quality_consumer_status"] == (
        "exact_quality_native_engine_command_and_boundary_match"
    )
    assert projected["execution_role"] == (
        "cache_verify_diagnostic_replay"
    )


def test_hailo_writer_rebinds_both_repetition_mirrors_to_final_command() -> None:
    writer = _load_script("native_hailo_trt_fifo_from_benchmarkset.py")
    payload = {
        "repetition_records": [{"workload_contract_sha256": "a" * 64}],
        "repetition_evidence": [{"workload_contract_sha256": "b" * 64}],
    }

    writer._bind_repetition_records_to_workload(payload, "c" * 64)

    assert payload["repetition_records"][0][
        "workload_contract_sha256"
    ] == "c" * 64
    assert payload["repetition_evidence"][0][
        "workload_contract_sha256"
    ] == "c" * 64


@pytest.mark.parametrize("hash_matches", (True, False))
def test_final_collector_rehashes_transferred_cache_replay_result(
    tmp_path: Path, hash_matches: bool,
) -> None:
    final_report = _load_script("native_producer_final_report.py")
    root = tmp_path / "hailo8"
    result_path = (
        root / "resnet50/benchmark_set/native_pipeline/b052/hailo_to_trt/"
        "float32_layout_fp16/native_fifo_results.json"
    )
    result_path.parent.mkdir(parents=True)
    result_payload = {
        "ok": True,
        "fps_makespan": 123.0,
        "frames": 10,
        "warmup": 2,
    }
    result_path.write_text(json.dumps(result_payload), encoding="utf-8")
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    analysis = root / "analysis_tables/native_fifo_eval_runner__cache.json"
    analysis.parent.mkdir(parents=True)
    analysis.write_text(json.dumps({
        "precision": "float32_layout_fp16",
        "rows": [{
            "model": "resnet50",
            "case_id": "b052",
            "precision": "float32_layout_fp16",
            "status": "ok",
            "result_ok": True,
            "returncode": 0,
            "timed_out": False,
            "child_result_fresh": True,
            "native_fifo_result": "/remote/path/native_fifo_results.json",
            "native_fifo_result_sha256": (
                result_sha if hash_matches else "f" * 64
            ),
            "native_fifo_result_size_bytes": result_path.stat().st_size,
            "native_split_quality_required": True,
        }],
    }), encoding="utf-8")

    rows = final_report._rows_from_native_fifo_runner(root)

    assert len(rows) == 1
    assert rows[0]["ok"] is hash_matches
    assert rows[0]["report"] == str(result_path)
    assert rows[0]["native_fifo_result_sha256"] == result_sha
    assert rows[0]["native_fifo_result_size_bytes"] == result_path.stat().st_size
    assert rows[0]["child_result_fresh"] is True


def test_variant_remote_binding_uses_frozen_hardware_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnx_splitpoint_tool.workflow.runner as runner_module

    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner._profile_with_cli_hardware_overrides = lambda: {}
    monkeypatch.setattr(
        runner_module,
        "normalize_hardware_targets",
        lambda _profile: [{
            "id": "orin_nx_hailo8_01",
            "accelerator": "hailo8",
            "enabled": True,
            "runtime": {
                "host": "192.0.2.8",
                "user": "nx",
                "port": 22,
                "remote_base_dir": "/srv/splitpoint",
                "remote_venv": "~/venvs/h8/bin/activate",
            },
            "remote": {
                "host": "192.0.2.8",
                "user": "nx",
                "port": 22,
                "remote_base_dir": "/srv/splitpoint",
            },
        }],
    )

    remotes = runner._materialize_variant_native_remotes({
        "backends": ["hailo8"],
        "variants": [{"id": "cache", "backends": ["hailo8"]}],
    })

    assert remotes["hailo8"]["ssh"] == "nx@192.0.2.8"
    assert remotes["hailo8"]["setup_id"] == "orin_nx_hailo8_01"
    assert remotes["hailo8"]["remote_base_dir"] == "/srv/splitpoint"
    assert remotes["hailo8"]["env"] == (
        "source ~/venvs/h8/bin/activate"
    )


def test_cache_coordinator_forwards_remote_engine_cache_base(
    tmp_path: Path,
) -> None:
    coordinator = _load_script(
        "run_evalrun_native_producer_variants.py"
    )
    (tmp_path / "models/resnet50/benchmark_set/legacy_suite/b052").mkdir(
        parents=True
    )
    resolved = _resolved_profile()
    base = copy.deepcopy(resolved["native_producers"])
    base["_workflow_context"] = {
        "execution_preset": copy.deepcopy(resolved["execution_preset"]),
    }
    from onnx_splitpoint_tool.native_execution_contract import (
        resolve_native_execution_contract,
    )
    base["_native_execution_contract"] = (
        resolve_native_execution_contract(resolved)
    )
    base["remotes"] = {
        "hailo8": {
            "ssh": "nx@host",
            "setup_id": "orin_nx_hailo8_01",
            "remote_base_dir": "/srv/splitpoint",
        }
    }

    cmd = coordinator._build_update_cmd(
        tmp_path,
        base,
        copy.deepcopy(base["variants"][0]),
        refresh_suites=False,
        timeout_s=30,
    )

    index = cmd.index("--hailo8-remote-base-dir")
    assert cmd[index + 1] == "/srv/splitpoint"
