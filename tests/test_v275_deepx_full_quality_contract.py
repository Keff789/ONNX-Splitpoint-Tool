from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"


def _functions(names: Sequence[str]) -> Dict[str, Any]:
    tree = ast.parse(SUITE.read_text(encoding="utf-8"), filename=str(SUITE))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "np": np,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SUITE), "exec"),
        namespace,
    )
    return namespace


def test_bn6_normalization_matches_completed_class_aware_postfilter() -> None:
    ns = _functions(["_deepx_yolo_nms", "_deepx_bn6_decode"])
    output = np.asarray(
        [[
            [10.0, 100.0, 110.0, 200.0, 0.90, 1.0],
            [12.0, 102.0, 108.0, 198.0, 0.80, 1.0],
            [12.0, 102.0, 108.0, 198.0, 0.70, 2.0],
            [20.0, 110.0, 30.0, 120.0, 0.10, 3.0],
        ]],
        dtype=np.float32,
    )

    detections = ns["_deepx_bn6_decode"](
        output,
        orig_shape=(480, 640, 3),
        scale=1.0,
        pad_x=0,
        pad_y=80,
        conf=0.25,
    )

    assert [item["class_id"] for item in detections] == [1, 2]
    assert [item["confidence"] for item in detections] == pytest.approx([0.9, 0.7])
    assert detections[0]["box_xyxy"] == pytest.approx([10.0, 20.0, 110.0, 120.0])


def test_bn6_decoder_attests_same_thresholds_as_completed_v2(tmp_path: Path) -> None:
    ns = _functions([
        "_deepx_yolo_nms",
        "_deepx_yolo_decode",
        "_deepx_yolov7_decode",
        "_deepx_bn6_decode",
        "_deepx_model_family",
        "_deepx_contract_model_id",
        "_deepx_detection_decode",
        "_deepx_endpoint_json_sha256",
        "_deepx_decoded_bn6_attestation",
        "_deepx_bn6_runtime_semantic_attestation",
        "_deepx_input_geometry",
    ])
    authoritative = {
        "endpoint_mode": "decoded",
        "host_tail_required": False,
        "postprocessing_required": False,
    }
    contract = {
        **authoritative,
        "endpoint_contract_binding_status": "attested",
        "postprocessing": {"host_required": False, "nms_on_host": False},
        "endpoint_semantic_attestation": {
            "pass": True,
            **authoritative,
            "stage": "decoded_nms",
            "output_format": "bn6_detections",
            "declared_output_semantics": "xyxy_score_class",
            "authoritative_contract": authoritative,
            "authoritative_contract_sha256": ns["_deepx_endpoint_json_sha256"](
                authoritative
            ),
        },
    }
    output = np.zeros((1, 1, 6), dtype=np.float32)
    output[0, 0] = [10.0, 10.0, 20.0, 20.0, 0.9, 1.0]

    detections, decoder = ns["_deepx_detection_decode"](
        root=tmp_path,
        run={"model_id": "yolo26s"},
        contract=contract,
        outputs=[output],
        orig_shape=(640, 640, 3),
        scale=1.0,
        pad_x=0,
        pad_y=0,
    )

    assert detections
    assert decoder["pass"] is True
    assert decoder["decoder_id"] == "bn6_classaware_postfilter_inverse_letterbox_v1"
    assert decoder["source_endpoint_has_integrated_nms"] is True
    assert decoder["host_decoder_applied"] is False
    assert decoder["host_nms_applied"] is True
    assert decoder["confidence_threshold"] == 0.25
    assert decoder["nms_iou_threshold"] == 0.45
    assert decoder["nms_max_detections"] == 300


def test_bn6_runtime_attestation_rejects_one_invalid_row_in_twenty() -> None:
    ns = _functions([
        "_deepx_endpoint_json_sha256",
        "_deepx_decoded_bn6_attestation",
        "_deepx_bn6_runtime_semantic_attestation",
    ])
    authoritative = {
        "endpoint_mode": "decoded",
        "host_tail_required": False,
        "postprocessing_required": False,
    }
    contract = {
        **authoritative,
        "endpoint_contract_binding_status": "attested",
        "postprocessing": {"host_required": False, "nms_on_host": False},
        "endpoint_semantic_attestation": {
            "pass": True,
            **authoritative,
            "stage": "decoded_nms",
            "output_format": "bn6_detections",
            "declared_output_semantics": "xyxy_score_class",
            "authoritative_contract": authoritative,
            "authoritative_contract_sha256": ns[
                "_deepx_endpoint_json_sha256"
            ](authoritative),
        },
    }
    output = np.zeros((1, 20, 6), dtype=np.float32)
    output[0, -1] = [10.0, 10.0, 20.0, 20.0, 0.9, 1.5]

    attestation = ns["_deepx_bn6_runtime_semantic_attestation"](
        output,
        contract,
    )

    assert attestation["pass"] is False
    assert attestation["attested"] is False
    assert "integer_nonnegative_class_fraction" in attestation["reason"]
    assert attestation["fraction_threshold"] == 1.0


@pytest.mark.parametrize("bad_class", [1.5, -1.0, float("nan")])
def test_bn6_decoder_rejects_invalid_retained_class_id(
    bad_class: float,
) -> None:
    ns = _functions(["_deepx_yolo_nms", "_deepx_bn6_decode"])
    output = np.asarray(
        [[[10.0, 10.0, 20.0, 20.0, 0.9, bad_class]]],
        dtype=np.float32,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "deepx_bn6_detection_(record_nonfinite|class_id_invalid)"
        ),
    ):
        ns["_deepx_bn6_decode"](
            output,
            orig_shape=(640, 640, 3),
            scale=1.0,
            pad_x=0,
            pad_y=0,
            conf=0.25,
        )


def test_quality_record_reader_accepts_each_schema_and_rejects_mixing() -> None:
    normalize = _functions(["_deepx_detection_record_components"])[
        "_deepx_detection_record_components"
    ]

    assert normalize(
        {"box_xyxy": [1, 2, 3, 4], "confidence": 0.75, "class_id": 5},
        image_id="legacy.jpg",
    ) == ([1.0, 2.0, 3.0, 4.0], 0.75, 5)
    assert normalize(
        {"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.8, "class_id": 6},
        image_id="completed.jpg",
    ) == ([1.0, 2.0, 3.0, 4.0], 0.8, 6)

    with pytest.raises(RuntimeError, match="deepx_quality_detection_schema_mixed"):
        normalize(
            {
                "box_xyxy": [1, 2, 3, 4],
                "confidence": 0.75,
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "score": 0.75,
                "class_id": 5,
            },
            image_id="mixed.jpg",
        )
    with pytest.raises(RuntimeError, match="deepx_quality_detection_box_invalid"):
        normalize(
            {"x1": 1, "y1": 2, "x2": 3, "score": 0.8, "class_id": 6},
            image_id="partial.jpg",
        )
    with pytest.raises(RuntimeError, match="deepx_quality_detection_numeric_invalid"):
        normalize(
            {"x1": 1, "y1": 2, "x2": 3, "y2": 4, "score": 0.8, "class_id": 6.5},
            image_id="fractional-class.jpg",
        )
