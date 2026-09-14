"""Replay observed Hailo10 buffers through the production boundary path.

The continuation is an identity ONNX so all boundary values are observable.
The recorded HEF / TensorRT artefacts are neither rebuilt nor quality-tuned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

from onnx_splitpoint_tool.runners.backends.hailo_backend import (
    _HailoInferModelSession,
    _HailoSession,
    _adapt_tensor,
)
from onnx_splitpoint_tool.runners.native_split_quality_runtime import (
    prepare_quality_first_boundary_input,
)
from scripts import native_trt_from_benchmarkset as bridge_builder


FIXTURE = Path(__file__).parent / "fixtures" / "v281_hailo10_nwc"
MANIFEST = json.loads((FIXTURE / "manifest.json").read_text())
SAMPLES = MANIFEST["samples"]
TARGET = (1, 84, 8400)


def _raw(sample):
    blob = FIXTURE / "physical_outputs.npz"
    assert hashlib.sha256(blob.read_bytes()).hexdigest() == MANIFEST["fixture_sha256"]
    with np.load(blob, allow_pickle=False) as archive:
        raw = archive[sample["key"]]
    assert hashlib.sha256(raw.tobytes()).hexdigest() == sample["physical_bytes_sha256"]
    return raw


def _session(kind, sample):
    # Hardware is the captured buffer boundary. These are the unmodified
    # production methods used for async slots and the InferModel sync path.
    session = object.__new__(kind)
    session._hef_output_names = [sample["physical_name"]]
    session._output_name_hef_to_canonical = {
        sample["physical_name"]: sample["canonical_name"],
    }
    session.output_shapes = {sample["canonical_name"]: TARGET}
    session.attested_source_output_shapes = {}
    session.copy_outputs = True
    return session


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: sample["key"])
@pytest.mark.parametrize("kind", [_HailoSession, _HailoInferModelSession])
def test_exact_captured_buffers_are_reordered_to_canonical(sample, kind):
    raw = _raw(sample)
    before = raw.tobytes()
    session = _session(kind, sample)
    mapped = session._canonical_outputs_from_slot_buffers({sample["physical_name"]: raw})
    assert list(mapped) == [sample["canonical_name"]]
    actual = mapped[sample["canonical_name"]]
    np.testing.assert_array_equal(actual, raw.transpose(0, 2, 1))
    assert actual.shape == TARGET and actual.dtype == np.uint8
    assert raw.tobytes() == before
    # Guard specifically against the previous same-numel reshape bug.
    assert not np.array_equal(actual, raw.reshape(TARGET))


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: sample["key"])
def test_physical_canonical_engine_bridge_roundtrip_on_actual_buffers(tmp_path, sample):
    raw = _raw(sample)
    mapped = _session(_HailoInferModelSession, sample)._canonical_outputs_from_slot_buffers(
        {sample["physical_name"]: raw},
    )[sample["canonical_name"]]
    name = sample["canonical_name"]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", [name], ["observed_boundary"])],
        "observable_boundary_continuation",
        [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, TARGET)],
        [onnx.helper.make_tensor_value_info("observed_boundary", onnx.TensorProto.FLOAT, TARGET)],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    source = tmp_path / "part2.onnx"
    onnx.save(model, source)
    scale = sample["quantization"]["scale"]
    zero_point = sample["quantization"]["zero_point"]
    bridge, metadata = bridge_builder._make_uint8_dequant_bridge_onnx(
        source, tmp_path / "bridge", root=tmp_path, case=sample["case"],
        scale=scale, zero_point=zero_point, boundary_layout="memory_nwc_to_ncw",
    )
    feed = prepare_quality_first_boundary_input(
        mapped, target_shape=TARGET, target_dtype=np.uint8,
        boundary_layout=metadata["boundary_layout"],
        boundary_transform="uint8_dequant_then_layout",
        dequant_scale=scale, dequant_zero_point=zero_point,
    )
    assert feed.flags.c_contiguous
    # Same bytes as the engine input captured by the read-only hardware probe;
    # therefore its existing layout/dequant bridge requires no artifact rebuild.
    assert hashlib.sha256(feed.tobytes()).hexdigest() == sample["observed_engine_feed_bytes_sha256"]
    np.testing.assert_array_equal(feed, raw.reshape(TARGET))
    actual = ReferenceEvaluator(str(bridge)).run(None, {name: feed})[0]
    expected = ((raw.astype(np.float32) - zero_point) * scale).transpose(0, 2, 1)
    np.testing.assert_array_equal(actual, expected)
    # A layout correction cannot recover the scores already lost in the HEF.
    assert np.count_nonzero(actual[:, 4:, :]) == 0
    assert sample["observed_scores_max"] == 0.0


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_nwc_singleton_mapping_and_inverse_are_exact(dtype):
    raw = np.arange(70, dtype=dtype).reshape(10, 7)
    canonical = _adapt_tensor(raw, (1, 7, 10))
    np.testing.assert_array_equal(canonical, raw.T[None])
    np.testing.assert_array_equal(_adapt_tensor(raw[None], (1, 7, 10)), canonical)
    np.testing.assert_array_equal(_adapt_tensor(canonical, raw.shape), raw)
    np.testing.assert_array_equal(_adapt_tensor(canonical, raw[None].shape), raw[None])
    assert canonical.dtype == raw.dtype


@pytest.mark.parametrize("source_shape,target_shape,axes,add_batch,drop_batch", [
    ((2, 3, 5), (3, 5, 2), (1, 2, 0), False, False),  # CHW -> HWC
    ((3, 5, 2), (2, 3, 5), (2, 0, 1), False, False),  # HWC -> CHW
    ((1, 2, 3, 5), (1, 3, 5, 2), (0, 2, 3, 1), False, False),
    ((1, 3, 5, 2), (1, 2, 3, 5), (0, 3, 1, 2), False, False),
    ((3, 5, 2), (1, 2, 3, 5), (2, 0, 1), True, False),
    ((1, 2, 3, 5), (3, 5, 2), (1, 2, 0), False, True),
])
def test_other_declared_image_layouts_keep_value_order(source_shape, target_shape, axes, add_batch, drop_batch):
    raw = np.arange(np.prod(source_shape), dtype=np.float32).reshape(source_shape)
    expected = raw[0] if drop_batch else raw
    expected = expected.transpose(axes)
    if add_batch:
        expected = expected[None]
    np.testing.assert_array_equal(_adapt_tensor(raw, target_shape), expected)


@pytest.mark.parametrize("source_shape,target_shape", [
    ((7, 10), (1, 7, 10)),  # Same physical/canonical order, add batch only.
    ((1, 7, 10), (7, 10)),
    ((7, 7), (1, 7, 7)),  # No axis guess for equal shape.
    ((1, 7, 7), (7, 7)),
    ((1, 10, 7), (1, 10, 7)),
    ((2, 35), (1, 7, 10)),  # Legacy fallback outside exact reversed axes.
])
def test_unrelated_singleton_and_legacy_fallback_behavior_is_preserved(source_shape, target_shape):
    raw = np.arange(np.prod(source_shape)).reshape(source_shape)
    np.testing.assert_array_equal(_adapt_tensor(raw, target_shape), raw.reshape(target_shape))


def test_invalid_element_count_still_rejected():
    with pytest.raises(ValueError, match="Cannot adapt tensor"):
        _adapt_tensor(np.zeros((10, 7), np.uint8), (1, 8, 10))
