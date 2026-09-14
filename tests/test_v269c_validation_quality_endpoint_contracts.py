from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np

from onnx_splitpoint_tool.benchmark.suite_refresh import (
    _normalize_suite_validation_payloads,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    runtime_output_contract,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _authoritative_classification(root: Path, stage: str) -> dict:
    root.mkdir(parents=True)
    (root / "output_contracts.json").write_text(json.dumps({
        "model_id": "resnet50", "task": "classification",
        "contracts": [{
            "model_id": "resnet50", "backend": "cuda_ort",
            "variant": "full", "task": "classification",
            "endpoint_mode": stage, "contract_status": "recorded",
            "host_tail_required": False, "postprocessing_required": False,
        }],
    }), encoding="utf-8")
    return load_authoritative_output_contract(
        root, backend="tensorrt", model_id="resnet50",
        variant="full", task="classification",
    )


def _classification_source(root: Path, count: int = 8) -> Path:
    source = root / "val_by_wnid"
    samples = []
    for index in range(count):
        wnid = f"n{index % 4:08d}"
        image = source / "images" / wnid / f"image_{index:03d}.JPEG"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"classification-{index}".encode())
        samples.append({
            "image": image.relative_to(source).as_posix(),
            "sample_id": f"sample-{index}",
            "class_name": wnid,
            "label_id": index % 4,
        })
    (source / "manifest.json").write_text(
        json.dumps({"samples": samples}), encoding="utf-8",
    )
    return source


def test_classification_suite_refresh_is_idempotent_and_binds_dataset_root(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = _classification_source(tmp_path)
    run = {
        "id": "ort_tensorrt",
        "benchmark_task": "classification",
        "validation_images": str(source),
        "validation_items_requested": 4,
        "validation_budget_authoritative": True,
        "validation_max_images": 4,
        "mini_classification_eval": True,
    }
    plan = {"runs": [run]}
    (suite / "benchmark_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (suite / "benchmark_set.json").write_text(
        json.dumps({"model_name": "resnet50", "plan": plan}), encoding="utf-8",
    )

    kwargs = dict(
        benchmark_set_json=suite / "benchmark_set.json",
        validation_images=str(source), validation_max_images=4,
        validation_reference_mode="auto", mini_coco_ap50=False,
        benchmark_task="classification", mini_classification_eval=True,
        log=None,
    )
    _normalize_suite_validation_payloads(suite, **kwargs)
    first = json.loads((suite / "benchmark_plan.json").read_text(encoding="utf-8"))
    first_source = first["runs"][0]["validation_images"]
    assert first_source.endswith("_n4_s20260710")
    assert not first_source.endswith("manifest.json")
    assert (suite / first_source / "manifest.json").is_file()

    # Simulate the independent refresh performed by another setup worker.  The
    # source must remain byte-for-byte stable instead of growing another suffix.
    _normalize_suite_validation_payloads(suite, **kwargs)
    second = json.loads((suite / "benchmark_plan.json").read_text(encoding="utf-8"))
    assert second["runs"][0]["validation_images"] == first_source
    assert first_source.count("_n4_s20260710") == 1


def test_endpoint_hash_is_precision_neutral_but_stage_sensitive(
    tmp_path: Path,
) -> None:
    logits_declaration = _authoritative_classification(
        tmp_path / "logits", "classification_logits",
    )
    probabilities_declaration = _authoritative_classification(
        tmp_path / "probabilities", "classification_probabilities",
    )
    fp32 = runtime_output_contract(
        "classification", {"logits": np.zeros((1, 1000), dtype=np.float32)},
        raw_fallback=False,
        declared_contract=logits_declaration,
    )
    int8 = runtime_output_contract(
        "classification", {"logits": np.zeros((1, 1000), dtype=np.int8)},
        raw_fallback=False,
        declared_contract=logits_declaration,
    )
    probabilities = runtime_output_contract(
        "classification",
        {"probabilities": np.full((1, 1000), 0.001, dtype=np.float32)},
        raw_fallback=False,
        declared_contract=probabilities_declaration,
    )
    assert fp32["endpoint_contract_hash"] == int8["endpoint_contract_hash"]
    assert fp32["endpoint_contract_hash"] != probabilities["endpoint_contract_hash"]


def test_new_central_request_identity_requires_endpoint_and_precision(
    tmp_path: Path,
) -> None:
    request = tmp_path / "quality_inputs" / "setup-a" / "results" / "b001" / "results_ort_tensorrt" / "task_quality_inputs" / "full_request.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps({
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "task": "classification", "variant": "full",
    }), encoding="utf-8")
    identity = EvaluationWorkflowRunner._quality_request_identity(
        request, model_id="resnet50", variant="full",
    )
    assert identity["identity_valid"] is False
    assert "missing_endpoint_contract_hash" in identity["identity_errors"]
    assert "missing_runtime_precision_identity" in identity["identity_errors"]
    assert "missing_quality_contract_sha256" in identity["identity_errors"]
    assert "missing_preprocessing_contract_sha256" in identity["identity_errors"]


def test_native_central_quality_binding_is_exact_and_precision_fail_closed() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "native_producer_validate_visualize.py"
    spec = importlib.util.spec_from_file_location("v269c_native_quality_binding", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    policy = AccuracyGatePolicy()
    endpoint_hash = "a" * 64
    request_sha = "b" * 64
    quality_contract_sha = "c" * 64
    preprocessing_sha = "d" * 64
    model_sha = "e" * 64
    dataset_sha = "f" * 64
    row = {
        "backend": "hailo8_to_trt", "model": "resnet50", "case": "b052",
        "setup_id": "orin_nx_hailo8_01", "task": "classification",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "runtime_precision_identity": "float32_layout_fp16",
    }
    result = {
        "model_id": "resnet50", "task": "classification", "case_id": "b052",
        "source_run_id": "hailo8_to_trt", "source_setup_id": "orin_nx_hailo8_01",
        "variant": "composed", "status": "completed", "technical_status": "completed",
        "decision": "pass", "policy_sha256": policy.sha256(),
        "source_request_sha256": request_sha,
        "model_sha256": model_sha,
        "validation_dataset_sha256": dataset_sha,
        "endpoint_contract_hash": endpoint_hash,
        "runtime_precision_identity": "float32_layout_fp16",
        "quality_contract_sha256": quality_contract_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "request_identity": {
            "schema_version": 3, "identity_valid": True,
            "model_id": "resnet50", "task": "classification", "case_id": "b052",
            "source_run_id": "hailo8_to_trt", "setup_id": "orin_nx_hailo8_01",
            "variant": "composed", "source_request_sha256": request_sha,
            "model_sha256": model_sha,
            "validation_dataset_sha256": dataset_sha,
            "endpoint_contract_hash": endpoint_hash,
            "runtime_precision_identity": "float32_layout_fp16",
            "quality_contract_sha256": quality_contract_sha,
            "preprocessing_contract_sha256": preprocessing_sha,
        },
        "primary": {"metric": "top1_accuracy", "delta": 0.0, "ci_low": 0.0, "margin": 0.01},
    }
    module._bind_central_quality_evidence(row, [result], policy)
    assert row["central_quality_evidence_verified"] is True
    assert row["precision_quality_verified"] is True
    assert row["precision_quality_binding_verified"] is True
    assert row["task_quality_observation_valid"] is True
    assert row["source_request_sha256"] == request_sha
    assert row["quality_contract_sha256"] == quality_contract_sha
    assert row["preprocessing_contract_sha256"] == preprocessing_sha

    mismatch = dict(row, runtime_precision_identity="uint8_dequant_fp16")
    module._bind_central_quality_evidence(mismatch, [result], policy)
    assert mismatch["central_quality_evidence_verified"] is False
    assert mismatch["precision_quality_verified"] is False

    negative = {
        key: value
        for key, value in row.items()
        if not key.startswith("quality_")
        and key
        not in {
            "central_quality_evidence_verified",
            "precision_quality_verified",
            "precision_quality_binding_verified",
            "task_quality_observation_valid",
        }
    }
    failed_result = dict(result, decision="fail")
    module._bind_central_quality_evidence(
        negative,
        [failed_result],
        policy,
    )
    assert negative["central_quality_evidence_verified"] is True
    assert negative["precision_quality_binding_verified"] is True
    assert negative["task_quality_observation_valid"] is True
