from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from onnx_splitpoint_tool.runners.harness.yolo import _get_ultralytics_regcls_pairs
from onnx_splitpoint_tool.workflow.results import (
    _deployment_contract_for_variant,
    _validation_report_to_row,
    normalize_benchmark_row,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"


def _functions(path: Path, names: Sequence[str]) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    ns: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "np": np,
        "re": __import__("re"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _generic_hailo_heads(reg_channels: int, class_channels: int) -> tuple[list[str], list[np.ndarray]]:
    names: list[str] = []
    outputs: list[np.ndarray] = []
    conv = 61
    for size in (80, 40, 20):
        names.extend([f"yolo26s_full/conv{conv}", f"yolo26s_full/conv{conv + 3}"])
        outputs.extend([
            np.zeros((size, size, reg_channels), dtype=np.float32),
            np.zeros((size, size, class_channels), dtype=np.float32),
        ])
        conv += 16
    return names, outputs


def test_yolo26_hailo_c4_c80_heads_pair_regression_before_classes_in_both_decoders() -> None:
    names, outputs = _generic_hailo_heads(4, 80)
    expected = [(1, 0, 1), (2, 2, 3), (3, 4, 5)]
    assert _get_ultralytics_regcls_pairs(names, outputs) == expected

    template = _functions(RUNNER, ["_try_parse_suffix_int", "_get_yolo_ultralytics_regcls_pairs"])
    assert template["_get_yolo_ultralytics_regcls_pairs"](names, outputs) == expected


def test_four_class_model_keeps_c64_dfl_as_regression_counter_regression() -> None:
    names, outputs = _generic_hailo_heads(64, 4)
    expected = [(1, 0, 1), (2, 2, 3), (3, 4, 5)]
    assert _get_ultralytics_regcls_pairs(names, outputs) == expected

    template = _functions(RUNNER, ["_try_parse_suffix_int", "_get_yolo_ultralytics_regcls_pairs"])
    assert template["_get_yolo_ultralytics_regcls_pairs"](names, outputs) == expected


def test_one2one_cv2_cv3_names_override_ambiguous_channel_heuristics() -> None:
    names: list[str] = []
    outputs: list[np.ndarray] = []
    for level, size in enumerate((80, 40, 20)):
        # Deliberately put class first; the semantic names must still pair cv2
        # as regression and cv3 as classification.
        names.extend([
            f"/model.23/one2one_cv3.{level}/one2one_cv3.{level}.2/Conv",
            f"/model.23/one2one_cv2.{level}/one2one_cv2.{level}.2/Conv",
        ])
        outputs.extend([
            np.zeros((size, size, 80), dtype=np.float32),
            np.zeros((size, size, 4), dtype=np.float32),
        ])
    expected = [(0, 1, 0), (1, 3, 2), (2, 5, 4)]
    assert _get_ultralytics_regcls_pairs(names, outputs) == expected


def test_variant_contract_does_not_leak_hailo_full_raw_head_to_decoded_rows() -> None:
    ns = _functions(
        RUNNER,
        ["_is_hailo_provider", "_deployment_contract_summary_for_variant"],
    )
    contract = ns["_deployment_contract_summary_for_variant"]
    common = {
        "task": "detection",
        "hailo_full_endpoint_mode": "raw_detection_head",
        "hailo_full_raw_detection_head": True,
        "hailo_part2_raw_detection_head": False,
        "hailo_part2_host_tail_required": False,
        "host_tail_available": False,
        "host_tail_model": None,
        "stage2_input_contract_kind": "image_or_decoded",
        "stage2_calibration_gate": {},
    }

    cpu = contract(variant="full", full_provider="cpu", stage2_provider="cpu", **common)
    trt = contract(variant="full", full_provider="tensorrt", stage2_provider="tensorrt", **common)
    composed = contract(
        variant="composed", full_provider="hailo8", stage2_provider="tensorrt", **common
    )
    hailo_full = contract(
        variant="full", full_provider="hailo8", stage2_provider="hailo8", **common
    )

    for decoded in (cpu, trt, composed):
        assert decoded["endpoint_mode"] == "decoded_or_native"
        assert decoded["raw_head_contract_present"] is False
        assert decoded["host_tail_required"] is False
        assert decoded["host_tail_available"] is False
    assert hailo_full["raw_head_contract_present"] is True
    assert hailo_full["host_tail_required"] is True
    assert hailo_full["host_tail_available"] is False
    assert hailo_full["raw_head_contract_status"] == "missing_host_tail"


def test_validation_report_ingestion_uses_primary_variant_contract(tmp_path: Path) -> None:
    report = {
        "case_id": "b038",
        "primary_variant": "composed",
        "run_cfg": {
            "provider": "tensorrt",
            "stage1_provider": "hailo8",
            "stage2_provider": "tensorrt",
        },
        "timings": {"composed": {"mean_ms": 10.0}, "full": {"mean_ms": 8.0}},
        "variant_status": {"composed": "ok", "full": "ok"},
        # Keep the old global value deliberately wrong to prove the scoped map
        # is authoritative during fallback validation-report ingestion.
        "deployment_contract_summary": {
            "variant": "full",
            "raw_head_contract_present": True,
            "raw_head_contract_status": "missing_host_tail",
            "host_tail_required": True,
            "host_tail_available": False,
        },
        "deployment_contracts_by_variant": {
            "full": {
                "variant": "full",
                "endpoint_mode": "decoded_or_native",
                "raw_head_contract_present": False,
                "raw_head_contract_status": "none",
                "host_tail_required": False,
                "host_tail_available": False,
            },
            "composed": {
                "variant": "composed",
                "endpoint_mode": "decoded_or_native",
                "raw_head_contract_present": False,
                "raw_head_contract_status": "none",
                "host_tail_required": False,
                "host_tail_available": False,
            },
        },
    }
    row = _validation_report_to_row(
        report,
        tmp_path / "b038" / "results_hailo8_to_tensorrt" / "validation_report.json",
    )
    assert row["raw_head_contract_present"] is False
    assert row["host_tail_required"] is False
    assert row["endpoint_mode"] == "decoded_or_native"
    assert _deployment_contract_for_variant(row, "composed")["raw_head_contract_present"] is False


def test_normalized_rows_only_gate_the_genuine_hailo_raw_head(tmp_path: Path) -> None:
    decoded = normalize_benchmark_row(
        {
            "case_id": "b038",
            "backend": "hailo8_to_tensorrt",
            "provider": "tensorrt",
            "variant": "split",
            "primary_variant": "composed",
            "stage1_provider": "hailo8",
            "stage2_provider": "tensorrt",
            "total_latency_ms": 10.0,
            "composed_mean_ms": 10.0,
            "runtime_ok": True,
            "validation_ok": True,
            "task": "detection",
            "deployment_contract_summary": {
                "variant": "full",
                "raw_head_contract_present": True,
                "host_tail_required": True,
                "host_tail_available": False,
            },
            "deployment_contracts_by_variant": {
                "composed": {
                    "variant": "composed",
                    "endpoint_mode": "decoded_or_native",
                    "raw_head_contract_present": False,
                    "raw_head_contract_status": "none",
                    "host_tail_required": False,
                    "host_tail_available": False,
                },
            },
        },
        model_id="yolo26s",
        source_path=tmp_path / "decoded.json",
    )
    assert decoded["raw_head_contract_present"] is False
    assert decoded["contract_gate_reason"] != "raw_head_host_tail_missing"

    hailo_raw = normalize_benchmark_row(
        {
            "case_id": "full",
            "backend": "hailo8",
            "provider": "hailo8",
            "variant": "full",
            "primary_variant": "full",
            "full_latency_ms": 15.0,
            "total_latency_ms": 15.0,
            "runtime_ok": True,
            "validation_ok": True,
            "task": "detection",
            "deployment_contracts_by_variant": {
                "full": {
                    "variant": "full",
                    "endpoint_mode": "raw_detection_head",
                    "raw_head_contract_present": True,
                    "raw_head_contract_status": "missing_host_tail",
                    "host_tail_required": True,
                    "host_tail_available": False,
                },
            },
        },
        model_id="yolo26s",
        source_path=tmp_path / "hailo_raw.json",
    )
    assert hailo_raw["raw_head_contract_present"] is True
    assert hailo_raw["contract_gate_reason"] == "raw_head_host_tail_missing"
