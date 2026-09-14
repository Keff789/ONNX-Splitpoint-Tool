"""Tight Float32 edge normalization, real YOLO NMS, no accelerator required."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as pp
from onnx_splitpoint_tool import native_three_stage as three_stage
from onnx_splitpoint_tool.runners.harness import yolo
from tests.test_v27923_decoded_completion import _runtime, _outputs
from tests.test_v27926_deepx_full_decoded_pre_nms import (
    case, _suite, _install_runtime, _decoded_outputs,
)
from tests.test_v27927_deepx_value_diagnostics import _probe_case

EPS = pp.DECODED_PRE_NMS_SCORE_EPSILON


def contract(values):
    return pp.build_frozen_postprocess_contract(
        model_id="yolo11l", outputs={"output0": values},
        input_hw=[640, 640], original_wh=[640, 480],
        source_contract_family="decoded_pre_nms",
    )


def seal(value):
    value = copy.deepcopy(value)
    value["invariant_identity"] = pp.frozen_postprocess_invariant_identity(value)
    value["invariant_contract_sha256"] = pp.canonical_json_sha256(value["invariant_identity"])
    value.pop("contract_sha256", None)
    value["contract_sha256"] = pp.canonical_json_sha256(value)
    return value


def channels(values):
    return values[0] if values.shape[1] == 84 else values[0].T


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("value", [-EPS, -(2.0**-24), -1e-9, 1.0 + EPS])
def test_allowed_edges_copy_only_scores_and_never_sigmoid(monkeypatch, transpose, value):
    values = _decoded_outputs()
    values[0, 8, 17] = value
    if transpose:
        values = values.transpose(0, 2, 1)
    before = values.tobytes()
    values.setflags(write=False)
    monkeypatch.setattr(yolo, "_sigmoid", lambda *a: pytest.fail("declared probabilities must not be sigmoided"))
    c = contract(values)
    prepared, observation = pp.prepare_decoded_pre_nms_outputs(
        {"output0": values}, score_policy=c["decoded_pre_nms_score_policy"],
    )
    assert not np.shares_memory(prepared["output0"], values)
    assert observation["corrected_score_count"] == 1
    assert observation["maximum_absolute_correction"] <= EPS
    assert observation["source_modified"] is False
    original = channels(values)
    fixed = channels(prepared["output0"])
    assert fixed[8, 17] == (0.0 if value < 0 else 1.0)
    valid = (original >= 0) & (original <= 1)
    np.testing.assert_array_equal(fixed[valid], original[valid])
    np.testing.assert_array_equal(fixed[:4], original[:4])
    frozen = pp.FrozenDetectionPostprocessor(c)
    result = frozen.process({"output0": values})
    assert result["decoded_pre_nms_score_normalization"] == observation
    assert frozen.completed_count == 1
    assert values.tobytes() == before


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_in_range_uses_same_buffer_and_other_dtypes_do_not_get_a_policy(dtype):
    values = _decoded_outputs().astype(dtype)
    c = contract(values)
    assert ("decoded_pre_nms_score_policy" in c) is (dtype is np.float32)
    prepared, obs = pp.prepare_decoded_pre_nms_outputs(
        {"output0": values}, score_policy=c.get("decoded_pre_nms_score_policy"),
    )
    assert prepared["output0"] is values
    assert obs["corrected_score_count"] == 0
    assert obs["processing_copy_created"] is False
    pp.FrozenDetectionPostprocessor(c).process(prepared)


@pytest.mark.parametrize("channel,value", [
    (4, np.nextafter(np.float32(-EPS), np.float32(-np.inf))),
    (4, np.nextafter(np.float32(1 + EPS), np.float32(np.inf))),
    (4, -1e-6), (4, 1.00001), (2, -1e-9), (3, -1e-9),
    (0, np.nan), (1, np.inf), (4, -np.inf),
])
@pytest.mark.parametrize("transpose", [False, True])
def test_rejected_values_never_mutate_data_or_advance_count(channel, value, transpose):
    values = _decoded_outputs()
    if transpose:
        values = values.transpose(0, 2, 1)
    runtime = pp.FrozenDetectionPostprocessor(contract(values))
    runtime.process({"output0": values})
    invalid = values.copy()
    channels(invalid)[channel, 17] = value
    before = invalid.tobytes()
    with pytest.raises(pp.FrozenPostprocessError):
        runtime.process({"output0": invalid})
    assert runtime.completed_count == 1
    assert invalid.tobytes() == before


@pytest.mark.parametrize("dtype,value", [
    (np.float16, -2.0**-24), (np.float64, -1e-16),
    (np.float64, np.nextafter(np.float64(1), np.float64(2))),
])
def test_other_dtypes_stay_strict_without_narrowing(dtype, value):
    values = _decoded_outputs().astype(dtype)
    values[0, 8, 17] = value
    with pytest.raises(pp.FrozenPostprocessError, match="decoded_pre_nms_values_invalid"):
        contract(values)


@pytest.mark.parametrize("change", ["epsilon", "dtype", "action", "remove", "null", "historical_hash"])
def test_policy_is_bound_not_an_unchecked_runtime_override(change):
    c = contract(_decoded_outputs())
    if change == "epsilon":
        c["decoded_pre_nms_score_policy"]["absolute_tolerance"] = 1e-4
    elif change == "dtype":
        c["decoded_pre_nms_score_policy"]["dtype"] = "float16"
    elif change == "action":
        c["decoded_pre_nms_score_policy"]["action"] = "sigmoid"
    elif change == "remove":
        c.pop("decoded_pre_nms_score_policy")
    elif change == "null":
        c["decoded_pre_nms_score_policy"] = None
    else:
        c["implementation_artifacts"]["native_detection_postprocess"]["sha256"] = pp._V27927_NATIVE_DETECTION_POSTPROCESS_SHA256
    with pytest.raises(pp.FrozenPostprocessError, match="score_policy"):
        pp.verify_frozen_postprocess_contract(seal(c))


@pytest.mark.parametrize("sha", [pp._V27926_NATIVE_DETECTION_POSTPROCESS_SHA256,
                                  pp._V27927_NATIVE_DETECTION_POSTPROCESS_SHA256])
def test_archived_contract_remains_strict_and_is_not_resealed(sha):
    values = _decoded_outputs()
    c = contract(values)
    c.pop("decoded_pre_nms_score_policy")
    c["implementation_artifacts"]["native_detection_postprocess"]["sha256"] = sha
    c = seal(c)
    original = copy.deepcopy(c)
    frozen = pp.FrozenDetectionPostprocessor(c)
    assert frozen.process({"output0": values})["detection_count"] == 2
    values[0, 8, 17] = -2.0**-24
    with pytest.raises(pp.FrozenPostprocessError, match="decoded_pre_nms_values_invalid"):
        frozen.process({"output0": values})
    assert c == original


@pytest.mark.parametrize("transpose", [False, True])
def test_fast_tail_and_execution_oracle_share_the_same_rule(tmp_path, transpose):
    outputs = _outputs("bnc" if transpose else "bcn")
    channels(outputs["output0"])[8, 17] = -2.0**-24
    before = outputs["output0"].tobytes()
    reference = _runtime(tmp_path, outputs)
    fast = three_stage.FastDetectionCompletionRuntime(reference.execution_contract)
    result = fast.process(outputs)
    exact = reference.process(outputs)
    assert result["detections"] == exact["detections"]
    assert result["decoded_pre_nms_score_normalization"] == exact["decoded_pre_nms_score_normalization"]
    assert result["decoded_pre_nms_score_normalization"]["corrected_score_count"] == 1
    pp.verify_detection_completion_execution_attestation(
        reference.attestation(), execution_contract=reference.execution_contract,
    )
    assert outputs["output0"].tobytes() == before


@pytest.fixture
def recorded_values():
    path = os.environ.get("V27928_REPLAY_NPZ", "")
    if not path:
        pytest.skip("recorded DeepX tensor is shipped in the delivery bundle; set V27928_REPLAY_NPZ")
    with np.load(path, allow_pickle=False) as archive:
        assert archive.files == ["tensor_000"]
        values = archive["tensor_000"]
    assert values.shape == (1, 84, 8400) and values.dtype == np.float32
    assert np.count_nonzero(values[0, 4:] < 0) == 21
    assert float(np.min(values[0, 4:])) == -(2.0**-24)
    return values


@pytest.mark.parametrize("transpose", [False, True])
def test_recorded_tensor_actual_harness_nms_exact_replay(recorded_values, monkeypatch, transpose):
    values = recorded_values.transpose(0, 2, 1) if transpose else recorded_values
    before = values.tobytes()
    outputs = {"output0": values}
    c = contract(values)
    clean = values.copy()
    scores = channels(clean)[4:]
    scores[scores < 0] = 0
    monkeypatch.setattr(yolo, "_sigmoid", lambda *a: pytest.fail("unexpected sigmoid"))
    result = pp.FrozenDetectionPostprocessor(c).process(outputs)
    baseline = pp.FrozenDetectionPostprocessor(contract(clean)).process({"output0": clean})
    assert result["detections"] == baseline["detections"]
    assert result["detections_sha256"] == baseline["detections_sha256"]
    assert result["decoded_pre_nms_score_normalization"]["corrected_score_count"] == 21
    assert result["detection_count"] > 0
    assert np.array_equal(channels(values)[4:].argmax(axis=0), channels(clean)[4:].argmax(axis=0))
    assert np.array_equal(channels(values)[4:].max(axis=0), channels(clean)[4:].max(axis=0))
    assert values.tobytes() == before


@pytest.mark.parametrize("mode", ["probe", "performance", "energy", "quality"])
def test_recorded_tensor_whole_suite_paths(case, monkeypatch, recorded_values, mode):
    suite, calls, run, args, results = _probe_case(case, monkeypatch, recorded_values)
    before = recorded_values.tobytes()
    if mode == "quality":
        bound = suite._deepx_bind_authoritative_endpoint_contract(case.root, run, case.contract)
        detections, detail = suite._deepx_detection_decode(
            root=case.root, run=run, contract=bound, outputs=[recorded_values],
            orig_shape=(640, 640, 3), scale=1., pad_x=0, pad_y=0,
        )
        assert detail["pass"] is True
        result = detail["frozen_postprocess_result"]
        assert result["decoded_pre_nms_score_normalization"]["corrected_score_count"] == 21
        assert detections
    elif mode == "probe":
        result = suite.run_deepx_output_value_probe(case.root, case.cached, run, args, results)
        assert result["status"] == "diagnostic_pass", result
        assert result["engine_call_count"] == 2
        assert result["decoded_pre_nms_score_normalization"]["corrected_score_count"] == 21
        with np.load(results / result["raw_output_snapshot"]["file"], allow_pickle=False) as raw:
            assert raw["tensor_000"].tobytes() == before
    else:
        args.energy_measurement_only = mode == "energy"
        result = suite._run_deepx_prepared_feed_benchmark(case.root, case.cached, run, args, results)
        assert result["status"] == "ok", result
        assert result["postprocess_completed_frames"] == result["completed_frames"]
        assert result["completed_frames"] > 0
        detail = result["frozen_host_postprocess_result"]
        assert detail["decoded_pre_nms_score_normalization"]["corrected_score_count"] == 21
    assert recorded_values.tobytes() == before
