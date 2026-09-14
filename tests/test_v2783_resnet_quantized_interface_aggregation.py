from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

from onnx_splitpoint_tool.workflow.results import (
    _validation_report_to_row,
    normalize_benchmark_row,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"


def _runner_functions(*names: str) -> dict[str, Any]:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in set(names)
    ]
    assert {node.name for node in selected} == set(names)
    namespace: dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "np": np,
        "re": re,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(RUNNER), "exec"),
        namespace,
    )
    return namespace


def _b119_interface(*, structural_pass: bool = True) -> dict[str, Any]:
    return {
        "pass": False,
        "status": "failed",
        "checks": {
            "stage1_vs_ort": {
                "required": True,
                "status": "ok",
                "pass": False,
                "structural_pass": structural_pass,
                "mapping_pass": structural_pass,
                "shape_pass": structural_pass,
                "numeric_pass": False,
                "max_abs": 0.38953375816345215,
                "per_tensor": [{
                    "ref_name": "mean",
                    "out_name": "mean" if structural_pass else "wrong_mean",
                    "match": "name" if structural_pass else "shape",
                    "ref_shape": [1, 2048, 1, 1],
                    "out_shape": [1, 2048, 1, 1],
                    "shape_match": True,
                    "max_abs": 0.38953375816345215,
                    "pass": False,
                }],
            }
        },
    }


def _b119_proxy() -> dict[str, Any]:
    top5 = [421, 919, 716, 675, 696]
    return {
        "pass": True,
        "n_classes": 1000,
        "reference_shape": [1, 1000],
        "candidate_shape": [1, 1000],
        "shape_match": True,
        "top1_ref": 421,
        "top1_out": 421,
        "top1_match": True,
        "top5_ref": list(top5),
        "top5_out": list(top5),
        "top5_overlap": 5,
        "top5_agreement": 1.0,
        "cosine_similarity": 0.9942717115277478,
        "max_abs": 0.40528106689453125,
        "mean_abs": 0.03988013416528702,
    }


def _b119_comparison() -> dict[str, Any]:
    return {
        "status": "ok",
        "passed": True,
        "passed_fidelity": True,
        "validation_mode_used": "proxy_classification",
        "reference_source": "cpu_full_proxy",
        "reference_variant": "full",
        "reference_provider": "cpu",
        "classification_proxy": _b119_proxy(),
    }


def _b119_drift() -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "ok",
        "mode": "primary_vs_cpu_full",
        "variant": "composed",
        "reference_source": "cpu_full_proxy",
        "reference_variant": "full",
        "reference_provider": "cpu",
        "single_sample": {"classification_proxy": _b119_proxy()},
    }


def _b119_binding() -> dict[str, Any]:
    return {
        "exact": True,
        "binding_sha256": "f0519c3ea392c355c294100a62de07e97fcbcd7ba2d6e4cac1bfb964dab129e4",
        "no_build": True,
        "no_fallback": True,
    }


def _decision(**overrides: Any) -> dict[str, Any]:
    fn = _runner_functions(
        "_quantized_classification_interface_gate_decision"
    )["_quantized_classification_interface_gate_decision"]
    payload = {
        "interface_checks": _b119_interface(),
        "primary_comparison": _b119_comparison(),
        "backend_drift": _b119_drift(),
        "stage2_binding": _b119_binding(),
        "primary_variant": "composed",
        "requested_primary_variant": "composed",
        "minimum_cosine": 0.98,
    }
    payload.update(overrides)
    return fn(**payload)


def test_b119_real_proxy_values_override_numeric_only_interface_drift() -> None:
    decision = _decision()
    assert decision == {
        "decision": "pass",
        "reason_codes": [
            "quantized_interface_drift_primary_cpu_full_proxy_passed"
        ],
        "minimum_cosine": 0.98,
        "top1": 421,
        "top5_overlap": 5,
        "cosine_similarity": 0.9942717115277478,
        "binding_sha256": _b119_binding()["binding_sha256"],
    }


def test_structural_mapping_or_shape_mismatch_is_always_a_hard_failure() -> None:
    decision = _decision(interface_checks=_b119_interface(structural_pass=False))
    assert decision["decision"] == "fail"
    assert "interface_mapping_or_shape_mismatch" in decision["reason_codes"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("top1", "primary_top1_mismatch"),
        ("top5", "primary_top5_incomplete"),
        ("cosine", "primary_cosine_below_contract"),
        ("shape", "primary_output_shape_or_class_count_mismatch"),
    ],
)
def test_explicit_primary_proxy_disagreement_is_a_hard_failure(
    mutation: str, reason: str,
) -> None:
    comparison = _b119_comparison()
    drift = _b119_drift()
    for container in (
        comparison["classification_proxy"],
        drift["single_sample"]["classification_proxy"],
    ):
        if mutation == "top1":
            container["top1_out"] = 422
            container["top1_match"] = False
            container["pass"] = False
        elif mutation == "top5":
            container["top5_out"][-1] = 999
            container["top5_overlap"] = 4
        elif mutation == "cosine":
            container["cosine_similarity"] = 0.979999
        elif mutation == "shape":
            container["candidate_shape"] = [1000]
            container["shape_match"] = False
            container["pass"] = False
    decision = _decision(
        primary_comparison=comparison,
        backend_drift=drift,
    )
    assert decision["decision"] == "fail"
    assert reason in decision["reason_codes"]


@pytest.mark.parametrize("missing", ["binding", "reference", "proxy"])
def test_missing_exact_evidence_is_inconclusive_not_false(missing: str) -> None:
    kwargs: dict[str, Any] = {}
    if missing == "binding":
        kwargs["stage2_binding"] = {
            "exact": False,
            "reason_codes": ["native_stage2_fallback_not_disabled"],
        }
    elif missing == "reference":
        comparison = _b119_comparison()
        comparison["reference_source"] = "full_same_backend"
        kwargs["primary_comparison"] = comparison
    else:
        comparison = _b119_comparison()
        comparison["classification_proxy"] = None
        kwargs["primary_comparison"] = comparison
    decision = _decision(**kwargs)
    assert decision["decision"] == "inconclusive"
    assert decision["reason_codes"]


def test_native_stage2_binding_requires_receipt_no_fallback_and_exact_names() -> None:
    fn = _runner_functions(
        "_strict_native_split_stage2_binding_evidence"
    )["_strict_native_split_stage2_binding_evidence"]
    base = {
        "binding": {"binding_sha256": "a" * 64},
        "receipt": {"status": "verified_exact_native_full_engine_receipt"},
        "session_build_info": {
            "runtime": "native_tensorrt",
            "explicit_engine": True,
            "fallback_disabled": True,
            "build_disabled": True,
            "artifact_binding_status": (
                "verified_exact_native_tensorrt_engine_receipt"
            ),
        },
        "stage2_provider": "tensorrt",
        "native_no_fallback": True,
        "native_build_enabled": False,
        "expected_inputs": ["mean"],
        "input_mapping": {"mean": "mean"},
        "stage1_output_names": ["mean"],
        "expected_input_shapes": {"mean": [1, 2048, 1, 1]},
        "stage1_output_shapes": {"mean": [1, 2048, 1, 1]},
        "boundary_input_name": "mean",
    }
    assert fn(**base)["exact"] is True

    fallback = dict(base, native_no_fallback=False)
    assert fn(**fallback)["exact"] is False
    assert "native_stage2_fallback_not_disabled" in fn(**fallback)["reason_codes"]

    mapped_by_shape = dict(base, input_mapping={"mean": "other"})
    assert fn(**mapped_by_shape)["exact"] is False
    assert "native_stage2_input_mapping_not_exact" in fn(**mapped_by_shape)["reason_codes"]

    wrong_shape = dict(
        base,
        expected_input_shapes={"mean": [1, 2048]},
    )
    assert fn(**wrong_shape)["exact"] is False
    assert "native_stage2_input_shape_not_exact" in fn(**wrong_shape)["reason_codes"]


def test_tensor_comparison_separates_structural_and_numeric_verdicts() -> None:
    fn = _runner_functions(
        "_compare_tensor_maps_by_shape_or_name"
    )["_compare_tensor_maps_by_shape_or_name"]
    reference = {"mean": np.zeros((1, 4, 1, 1), dtype=np.float32)}
    quantized = {"mean": np.full((1, 4, 1, 1), 0.389533758, dtype=np.float32)}
    result = fn(reference, quantized, eps=0.05)
    assert result["structural_pass"] is True
    assert result["numeric_pass"] is False
    assert result["pass"] is False

    shape_only = {"other": quantized["mean"]}
    result = fn(reference, shape_only, eps=0.05)
    assert result["shape_pass"] is True
    assert result["mapping_pass"] is False
    assert result["structural_pass"] is False
    assert result["pass"] is False


def test_candidate_only_rows_keep_accuracy_but_not_fake_zero_agreement() -> None:
    fn = _runner_functions(
        "_aggregate_classification_dataset_rows"
    )["_aggregate_classification_dataset_rows"]
    rows = [
        {
            "candidate_metric_status": (
                "available_reference_comparison_unavailable"
            ),
            "gt": {
                "label_id": index,
                "top1_hit": True,
                "top5_hit": True,
            },
        }
        for index in range(5)
    ]
    metrics, passed = fn(rows, pass_ratio_thr=0.8)
    assert passed is None
    assert metrics["n_images"] == 5
    assert metrics["reference_compared_images"] == 0
    assert metrics["image_pass_ratio"] is None
    assert metrics["top1_agreement"] is None
    assert metrics["top5_agreement"] is None
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top5_accuracy"] == 1.0

    # A real local comparison error is still a measured failure.  Only the
    # explicit central-management candidate-only status is removed from the
    # comparison denominator.
    failed_metrics, failed = fn([{"error": "reference execution failed"}])
    assert failed is False
    assert failed_metrics["reference_compared_images"] == 1
    assert failed_metrics["image_pass_ratio"] == 0.0
    assert failed_metrics["top1_agreement"] == 0.0


@pytest.mark.parametrize(
    ("decision", "final_value", "expected"),
    [
        ("pass", True, True),
        ("fail", False, False),
        ("inconclusive", None, None),
    ],
)
def test_result_normalizer_preserves_runtime_contract_tristate(
    tmp_path: Path, decision: str, final_value: Optional[bool],
    expected: Optional[bool],
) -> None:
    report = {
        "case_id": "b119",
        "run_cfg": {
            "provider": "tensorrt",
            "stage1_provider": "hailo8",
            "stage2_provider": "tensorrt",
        },
        "primary_variant": "composed",
        "timings": {"composed": {"mean_ms": 6.845}},
        "runtime_contract_decision": decision,
        "runtime_contract_reason_codes": ["test_reason"],
        "final_pass": final_value,
        "final_pass_all": final_value,
        "interface_checks": {
            "pass": False,
            "status": "warning_primary_cpu_full_proxy_override",
            "gate_override": "primary_cpu_full_proxy_passed",
            "gate_blocking": False,
            "checks": {
                "stage1_vs_ort": {
                    "status": "ok",
                    "pass": False,
                    "mapping_pass": True,
                    "shape_pass": True,
                    "structural_pass": True,
                    "numeric_pass": False,
                }
            },
        },
    }
    row = _validation_report_to_row(
        report, tmp_path / "b119/results_hailo8_to_trt/validation_report.json",
    )
    assert row["validation_ok"] is expected
    assert row["runtime_contract_decision"] == decision
    assert row["runtime_contract_reason_codes"] == ["test_reason"]
    # The strict raw interface verdict remains visible even when the aggregate
    # technical contract is promoted by the semantic proxy.
    assert row["interface_checks"]["pass"] is False
    normalized = normalize_benchmark_row(
        row,
        model_id="resnet50",
        source_path=tmp_path / "validation_report.json",
    )
    assert normalized["validation_ok"] is expected
    assert normalized["runtime_contract_decision"] == decision
    assert normalized["runtime_contract_reason_codes"] == ["test_reason"]
    assert normalized["strict_boundary_numeric_pass"] is False
    if decision == "pass":
        assert normalized["interface_contract_pass"] is True
        assert normalized["contract_consistent"] is True
        assert normalized["validation_claim_level"] == (
            "semantic_with_boundary_numeric_warning"
        )


def test_runner_main_preserves_inconclusive_in_report_and_eligibility() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'runtime_contract_decision = "inconclusive"' in source
    assert 'final_pass_all=(\n            bool(final_pass_all) if final_pass_all is not None else None' in source
    assert 'final_pass=(bool(final_pass) if final_pass is not None else None)' in source
    assert '"runtime_contract_decision": runtime_contract_decision' in source
    assert '"ranking_eligible": _ranking_eligible' in source
    assert "if final_pass is False:" in source
    assert "elif final_pass is None:" in source
    assert "validation inconclusive" in source
    assert "not interface_hard_failure" in source
