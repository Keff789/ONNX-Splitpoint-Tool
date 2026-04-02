from __future__ import annotations

from pathlib import Path

import numpy as np

from onnx_splitpoint_tool.runners._types import Stage, StagePlan
from onnx_splitpoint_tool.runners.interface_transfer import (
    DefaultTransfer,
    TransferMeta,
    adapt_tensor_verbose,
    build_tensor_contract_entry,
    infer_tensor_layout,
)


def test_stage_alias_is_kept_for_template_backcompat() -> None:
    assert Stage is StagePlan


def test_default_transfer_roundtrip_and_meta() -> None:
    tr = DefaultTransfer()
    tensors = {
        "cut0": np.zeros((1, 32, 20, 20), dtype=np.float32),
        "cut1": np.zeros((20, 20, 64), dtype=np.float32),
    }
    blob, meta = tr.serialize(tensors)
    assert isinstance(meta, TransferMeta)
    assert meta.total_bytes == len(blob)
    assert [s.key for s in meta.specs] == ["cut0", "cut1"]
    roundtrip = tr.deserialize(blob, meta)
    assert set(roundtrip.keys()) == {"cut0", "cut1"}
    assert roundtrip["cut0"].shape == (1, 32, 20, 20)
    assert roundtrip["cut1"].shape == (20, 20, 64)


def test_layout_guess_and_verbose_adaptation() -> None:
    src = np.zeros((1, 256, 40, 40), dtype=np.float32)
    adapted, op = adapt_tensor_verbose(src, (1, 40, 40, 256))
    assert op == "transpose_nchw_to_nhwc"
    assert adapted.shape == (1, 40, 40, 256)
    assert infer_tensor_layout(src.shape) == "NCHW"
    assert infer_tensor_layout(adapted.shape) == "NHWC"


def test_contract_entry_captures_source_canonical_and_consumer() -> None:
    src = np.zeros((1, 256, 40, 40), dtype=np.float32)
    canonical, op = adapt_tensor_verbose(src, (40, 40, 256))
    consumer = np.ascontiguousarray(canonical)

    contract = build_tensor_contract_entry(
        index=0,
        expected_name="input_layer1",
        source_name="o1",
        source_arr=src,
        canonical_arr=canonical,
        consumer_arr=consumer,
        target_shape=(40, 40, 256),
        producer_backend="tensorrt",
        consumer_backend="hailo8",
        source_kind="stage1_output",
        adapt_op=op,
        quantized_hint=False,
    )

    assert contract["expected_name"] == "input_layer1"
    assert contract["source_name"] == "o1"
    assert contract["producer_backend"] == "tensorrt"
    assert contract["consumer_backend"] == "hailo8"
    assert contract["canonical"]["physical_layout"] == "HWC"
    assert contract["canonical"]["adapt_op"] == "squeeze_batch_transpose_chw_to_hwc"
