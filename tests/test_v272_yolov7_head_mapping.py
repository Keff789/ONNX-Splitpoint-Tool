from __future__ import annotations

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenPostprocessError,
    build_detection_completion_runtime,
    build_yolov7_head_mapping,
    tensor_signature,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)


def _heads_640() -> dict[str, np.ndarray]:
    return {
        "vendor_small": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "vendor_medium": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "vendor_large": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }


def _raw_source(
    outputs: dict[str, np.ndarray],
    *,
    endpoint_hash: str,
) -> dict:
    signature = tensor_signature(outputs)
    return {
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "model_sha256": YOLOV7_PAPER_ONNX_SHA256,
        "tensor_signature": signature,
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "stage": "raw_head",
            "endpoint": "raw_head",
            "endpoint_contract_hash": endpoint_hash,
        },
    }


def test_yolov7_head_mapping_is_name_and_permutation_invariant() -> None:
    first = _heads_640()
    second = {
        "backend_out_17": first["vendor_large"],
        "backend_out_3": first["vendor_small"],
        "backend_out_9": first["vendor_medium"],
    }

    first_mapping = build_yolov7_head_mapping(first, input_hw=[640, 640])
    second_mapping = build_yolov7_head_mapping(second, input_hw=[640, 640])

    assert first_mapping == second_mapping
    assert [head["stride"] for head in first_mapping["heads"]] == [
        8, 16, 32,
    ]
    assert [head["grid_hw"] for head in first_mapping["heads"]] == [
        [80, 80], [40, 40], [20, 20],
    ]
    assert "vendor_small" not in str(first_mapping)
    assert "backend_out_17" not in str(second_mapping)


def test_yolov7_head_mapping_rejects_duplicate_or_missing_stride() -> None:
    duplicate = _heads_640()
    duplicate["vendor_large"] = np.full(
        (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
    )

    with pytest.raises(
        FrozenPostprocessError,
        match="duplicate_stride",
    ):
        build_yolov7_head_mapping(duplicate, input_hw=[640, 640])

    missing = _heads_640()
    missing.pop("vendor_large")
    with pytest.raises(
        FrozenPostprocessError,
        match="exactly_three_outputs",
    ):
        build_yolov7_head_mapping(missing, input_hw=[640, 640])


def test_yolov7_head_mapping_rejects_ambiguous_physical_layout() -> None:
    # For input 96, the stride-32 grid is 3x3.  [3,3,3,85] can mean either
    # [anchors,H,W,C] or [H,W,anchors,C], so choosing one would be name/order
    # dependent and must fail closed.
    ambiguous = {
        "p3": np.zeros((1, 3, 12, 12, 85), dtype=np.float32),
        "p4": np.zeros((1, 3, 6, 6, 85), dtype=np.float32),
        "p5": np.zeros((3, 3, 3, 85), dtype=np.float32),
    }
    with pytest.raises(
        FrozenPostprocessError,
        match="layout_ambiguous",
    ):
        build_yolov7_head_mapping(ambiguous, input_hw=[96, 96])


def test_permuted_physical_heads_share_completed_content_and_comparison() -> None:
    first = _heads_640()
    strong = first["vendor_small"][0, 0, 0, 0]
    strong[:4] = 0.0
    strong[4] = 20.0
    strong[7] = 20.0
    second = {
        "out_c": first["vendor_large"],
        "out_a": first["vendor_small"],
        "out_b": first["vendor_medium"],
    }
    first_runtime = build_detection_completion_runtime(
        model_id="yolov7_paper",
        outputs=first,
        input_hw=[640, 640],
        original_wh=[80, 60],
        source_endpoint_contract=_raw_source(
            first, endpoint_hash="a" * 64,
        ),
    )
    second_runtime = build_detection_completion_runtime(
        model_id="yolov7_paper",
        outputs=second,
        input_hw=[640, 640],
        original_wh=[80, 60],
        source_endpoint_contract=_raw_source(
            second, endpoint_hash="b" * 64,
        ),
    )

    first_result = first_runtime.process(first)
    second_result = second_runtime.process(second)

    assert first_result["content_sha256"] == second_result["content_sha256"]
    assert (
        first_runtime.execution_contract["comparison_endpoint_contract"][
            "output_endpoint_id"
        ]
        == second_runtime.execution_contract["comparison_endpoint_contract"][
            "output_endpoint_id"
        ]
    )
    assert (
        first_runtime.execution_contract["processor_contract"][
            "raw_output_tensor_signature"
        ]
        == second_runtime.execution_contract["processor_contract"][
            "raw_output_tensor_signature"
        ]
    )
