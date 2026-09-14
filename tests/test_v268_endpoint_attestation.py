from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from onnx_splitpoint_tool.native_output_endpoint import (
    DECODED_NMS_ATTESTATION_SOURCE,
    attest_decoded_nms,
    load_authoritative_output_contract,
    runtime_output_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"v268_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decoded_rows(count: int = 8) -> np.ndarray:
    rows = np.zeros((1, count, 6), dtype=np.float32)
    rows[0, :, 0] = np.arange(count, dtype=np.float32)
    rows[0, :, 1] = np.arange(count, dtype=np.float32) * 0.5
    rows[0, :, 2] = rows[0, :, 0] + 10.0
    rows[0, :, 3] = rows[0, :, 1] + 12.0
    rows[0, :, 4] = np.linspace(0.1, 0.9, count, dtype=np.float32)
    rows[0, :, 5] = np.arange(count, dtype=np.float32) % 3
    return rows


def _one_class_raw_bn6(count: int = 32) -> np.ndarray:
    """Representative xywh/objectness/class-score raw head with [B,N,6]."""
    rows = np.zeros((1, count, 6), dtype=np.float32)
    rows[0, :, 0] = np.linspace(100.0, 500.0, count)  # centre x
    rows[0, :, 1] = np.linspace(80.0, 400.0, count)   # centre y
    rows[0, :, 2] = np.linspace(10.0, 80.0, count)    # width, not x2
    rows[0, :, 3] = np.linspace(8.0, 70.0, count)     # height, not y2
    rows[0, :, 4] = np.linspace(0.01, 0.99, count)    # objectness
    rows[0, :, 5] = np.linspace(0.02, 0.98, count)    # class probability
    return rows


def test_explicit_nms_binding_and_plausible_values_are_attested() -> None:
    result = attest_decoded_nms({"efficient_nms": _decoded_rows()})
    assert result["attested"] is True
    assert result["status"] == "passed"
    assert result["contract_source"] == DECODED_NMS_ATTESTATION_SOURCE
    contract = runtime_output_contract(
        "detection", {"efficient_nms": _decoded_rows()}, raw_fallback=False,
    )
    assert contract["contract_family"] == "decoded_nms"
    assert contract["output_endpoint_attestation"]["attested"] is True
    assert contract["endpoint_contract_complete"] is True
    assert len(contract["endpoint_contract_hash"]) == 64


def test_decoded_pre_nms_bn6_cannot_be_attested_from_values_alone() -> None:
    rows = _decoded_rows(8400)
    result = attest_decoded_nms({"output0": rows})
    assert result["attested"] is False
    assert "explicit_nms_declaration_required" in result["reason"]
    contract = runtime_output_contract(
        "detection", {"output0": rows}, raw_fallback=False,
    )
    assert contract["contract_family"] == "unknown"
    assert contract["endpoint_contract_complete"] is False


def test_loader_bound_yolo_matrix_remains_decoded_pre_nms(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "pre_nms"
    suite.mkdir()
    (suite / "output_contracts.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/output-contracts",
            "schema_version": 1,
            "model_id": "yolo11l",
            "task": "detection",
            "contracts": [{
                "schema": "onnx-splitpoint/output-contract",
                "schema_version": 1,
                "model_id": "yolo11l",
                "task": "detection",
                "backend": "cuda_ort",
                "variant": "full",
                "endpoint_mode": "decoded_pre_nms",
                "stage": "decoded_pre_nms",
                "contract_family": "decoded_pre_nms",
                "output_format": "ultralytics_decoded",
                "contract_status": "recorded",
                "host_tail_required": True,
                "postprocessing_required": True,
                "requires_external_postprocess": True,
            }],
        }, sort_keys=True),
        encoding="utf-8",
    )
    declaration = load_authoritative_output_contract(
        suite,
        backend="tensorrt",
        model_id="yolo11l",
        variant="full",
        task="detection",
    )
    assert declaration["contract_resolution_status"] == "attested"
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    contract = runtime_output_contract(
        "detection",
        {"output0": output},
        raw_fallback=False,
        declared_contract=declaration,
    )
    assert contract["endpoint_contract_complete"] is True
    assert contract["stage"] == "decoded_pre_nms"
    assert contract["contract_family"] == "decoded_pre_nms"
    assert contract["output_format"] == "ultralytics_decoded"
    assert len(contract["endpoint_contract_hash"]) == 64
    assert contract["output_endpoint_attestation"]["attested"] is True


def test_one_class_raw_bn6_is_not_mislabelled_as_decoded_nms() -> None:
    result = attest_decoded_nms({"raw_head": _one_class_raw_bn6()})
    assert result["attested"] is False
    assert "class_column_not_integer_nonnegative" in result["reason"]
    contract = runtime_output_contract(
        "detection", {"raw_head": _one_class_raw_bn6()}, raw_fallback=False,
    )
    assert contract["contract_family"] == "unknown"
    assert contract["output_format"] == "tensor_outputs"


def test_multioutput_and_nonfinite_bn6_fail_closed() -> None:
    multi = attest_decoded_nms({"a": _decoded_rows(), "b": _decoded_rows()})
    assert multi["attested"] is False
    assert multi["reason"] == "exactly_one_runtime_output_required"
    nonfinite = _decoded_rows()
    nonfinite[0, 0, 4] = np.nan
    invalid = attest_decoded_nms({"a": nonfinite})
    assert invalid["attested"] is False
    assert invalid["reason"] == "nonfinite_runtime_values"


def test_full_dump_contract_uses_value_attestation_and_raw_fallback() -> None:
    full = _load_script("native_full_semantic_dump.py")
    decoded = full._contract("detection", {"efficient_nms": _decoded_rows()})
    assert decoded["contract_family"] == "decoded_nms"
    assert decoded["contract_source"] == DECODED_NMS_ATTESTATION_SOURCE
    assert decoded["claim_eligible_e2e"] is True
    raw = full._contract("detection", {"raw": _one_class_raw_bn6()})
    assert raw["contract_family"] == "unknown"
    assert raw["claim_eligible_e2e"] is False


def test_split_contracts_use_the_same_fail_closed_attestation() -> None:
    hailo10 = _load_script("native_hailo10_trt_e2e_from_benchmarkset.py")
    deepx = _load_script("native_deepx_trt_e2e_from_benchmarkset.py")
    for module in (hailo10, deepx):
        assert module._native_output_contract(
            "detection", {"raw": _one_class_raw_bn6()},
        )["contract_family"] == "unknown"
        assert module._native_output_contract(
            "detection", {"efficient_nms": _decoded_rows()},
        )["contract_family"] == "decoded_nms"


def test_hailo8_manifest_annotation_reads_values_not_only_shape(tmp_path: Path) -> None:
    hailo8 = _load_script("native_hailo_trt_fifo_from_benchmarkset.py")
    benchmark_set = tmp_path / "benchmark_set"
    benchmark_set.mkdir()
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({"benchmark_task": "detection"}), encoding="utf-8",
    )
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    raw = _one_class_raw_bn6()
    (dump_dir / "output.bin").write_bytes(raw.tobytes())
    manifest_path = dump_dir / "native_outputs_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "onnx-splitpoint/runner-output-dump",
        "schema_version": 3,
        "outputs": [{
            "name": "raw", "file": "output.bin", "dtype": str(raw.dtype),
            "shape": list(raw.shape), "bytes": raw.nbytes,
        }],
    }), encoding="utf-8")
    hailo8._annotate_output_contract(manifest_path, benchmark_set)
    annotated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert annotated["contract_family"] == "unknown"
    assert annotated["output_endpoint_attestation"]["attested"] is False


def test_final_report_rejects_shape_only_legacy_decoded_nms() -> None:
    final_report = _load_script("native_producer_final_report.py")
    legacy = {
        "task": "detection",
        "contract_family": "decoded_nms",
        "output_format": "bn6_detections",
        "contract_source": "benchmark_task_and_runtime_output_shape",
    }
    assert final_report._explicit_output_endpoint(legacy) == ""
    attestation = attest_decoded_nms({"efficient_nms": _decoded_rows()})
    modern = {
        **legacy,
        "contract_source": DECODED_NMS_ATTESTATION_SOURCE,
        "stage": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": attestation["endpoint_contract_hash"],
        "output_endpoint_attestation": attestation,
    }
    modern["stage"] = "decoded_nms"
    modern["endpoint_contract_complete"] = True
    modern["endpoint_contract_hash"] = attestation["endpoint_contract_hash"]
    endpoint_id = final_report._explicit_output_endpoint(modern)
    assert endpoint_id.endswith(attestation["endpoint_contract_hash"])
    modern.pop("output_endpoint_attestation")
    assert final_report._explicit_output_endpoint(modern) == ""


def test_semantic_validator_rejects_shape_only_legacy_manifest(tmp_path: Path) -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    manifest_path = tmp_path / "native_outputs_manifest.json"
    legacy = {
        "task": "detection",
        "contract_family": "decoded_nms",
        "output_format": "bn6_detections",
        "contract_source": "benchmark_task_and_runtime_output_shape",
        "outputs": [{"name": "out", "shape": [1, 100, 6]}],
    }
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert validator._expected_detection_contract(manifest_path) == (
        "unknown", "metadata_unavailable",
    )
    attestation = attest_decoded_nms({"efficient_nms": _decoded_rows()})
    modern = {
        **legacy,
        "contract_source": DECODED_NMS_ATTESTATION_SOURCE,
        "stage": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": attestation["endpoint_contract_hash"],
        "output_endpoint_attestation": attestation,
    }
    manifest_path.write_text(json.dumps(modern), encoding="utf-8")
    assert validator._expected_detection_contract(manifest_path) == (
        "decoded_nms", "explicit_endpoint_declaration_and_runtime_value_attestation",
    )


def test_full_endpoint_gate_is_scoped_to_setup_and_comparison_pair() -> None:
    final_report = _load_script("native_producer_final_report.py")
    endpoint_hash = "a" * 64
    attestation = {
        "attested": True,
        "status": "passed",
        "task": "classification",
        "stage": "classification_logits",
        "endpoint": "classification_logits",
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_id": (
            f"classification:classification_logits:{endpoint_hash}"
        ),
    }

    def decoded(backend: str, setup: str, comparison: str, **extra):
        samples = [10.0, 11.0, 12.0]
        return {
            "backend": backend, "model": "yolo", "case": "full",
            "execution_mode": "native_full_baseline", "setup_id": setup,
            "comparison_backend": comparison, "execution_precision": "fp16",
            "ok": True, "repetition_count_requested": 3,
            "repetition_count_attempted": 3, "repetition_count_valid": 3,
            "repetition_status": "complete",
            "repetition_aggregation": "median_with_deterministic_percentile_bootstrap_ci95",
            "repetition_runtime_scope": "fresh_runtime_per_repetition",
            "repetition_independence_verified": True,
            "repetition_records": [
                    {"repetition_index": index, "runtime_instance_id": f"{backend}-{setup}-{index}", "ok": True, "status": "ok", "fps_makespan": fps, "completed_work_units": 100, "workload_contract_sha256": "c" * 64}
                for index, fps in enumerate(samples, 1)
            ],
            "fps_repetition_samples": samples,
            "fps_median": 11.0, "fps_ci95_low": 10.0, "fps_ci95_high": 12.0,
            "quality_evidence_verified": True,
            "task_quality_observation_valid": True,
            "quality_accuracy_gate_pass": True,
            "task": "classification",
            "contract_family": "classification_logits",
            "output_format": "classification_logits",
            "contract_source": DECODED_NMS_ATTESTATION_SOURCE,
            "output_endpoint_attestation": attestation,
            "stage": "classification_logits",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": endpoint_hash,
            **extra,
        }

    rows = [
        decoded(
            "native_full_deepx", "deepx-host", "deepx_m1",
            outer_makespan_verified=True,
        ),
        decoded("native_full_tensorrt", "deepx-host", "deepx_m1"),
        {
            **decoded("native_full_hailo10h", "hailo-host", "hailo10h"),
            "contract_family": "raw_head",
            "output_format": "raw_detection_tensors",
            "contract_source": "runtime_value_attestation_failed",
            "output_endpoint_attestation": {
                "attested": False, "status": "failed",
                "contract_source": DECODED_NMS_ATTESTATION_SOURCE,
            },
        },
        decoded("native_full_tensorrt", "hailo-host", "hailo10h"),
    ]
    gated = final_report._apply_comparison_claim_gates(rows)
    deepx_pair = [row for row in gated if row["setup_id"] == "deepx-host"]
    hailo_pair = [row for row in gated if row["setup_id"] == "hailo-host"]
    assert len(deepx_pair) == 2
    assert all(row["output_endpoint_match"] for row in deepx_pair)
    assert all(row["performance_claim_eligible"] for row in deepx_pair)
    assert len(hailo_pair) == 2
    assert not any(row["output_endpoint_match"] for row in hailo_pair)
    assert not any(row["performance_claim_eligible"] for row in hailo_pair)
    assert all(
        "output_endpoint_not_common_across_backends"
        in row["performance_claim_exclusion_reasons"]
        for row in hailo_pair
    )

    missing_identity = final_report._apply_comparison_claim_gates([
        decoded("native_full_hailo8", "", ""),
        decoded("native_full_tensorrt", "", ""),
    ])
    assert not any(row["output_endpoint_match"] for row in missing_identity)
    assert all(
        "comparison_stratum_identity_missing"
        in row["performance_claim_exclusion_reasons"]
        for row in missing_identity
    )


def test_classification_topk_integer_and_multioutput_fail_closed() -> None:
    topk = runtime_output_contract(
        "classification", {"topk_indices": np.arange(5, dtype=np.int64)},
        raw_fallback=False,
    )
    assert topk["contract_family"] == "unknown"
    assert topk["endpoint_contract_complete"] is False
    multi = runtime_output_contract(
        "classification",
        {
            "scores": np.ones((1, 1000), dtype=np.float32),
            "indices": np.arange(5, dtype=np.int64),
        },
        raw_fallback=False,
    )
    assert multi["contract_family"] == "unknown"
    logits = runtime_output_contract(
        "classification", {"logits": np.linspace(-2, 2, 1000, dtype=np.float32)[None]},
        raw_fallback=False,
    )
    assert logits["contract_family"] == "unknown"
    assert logits["endpoint_contract_complete"] is False
    assert logits["output_endpoint_attestation"]["reason"] == (
        "authoritative_suite_declaration_required"
    )
