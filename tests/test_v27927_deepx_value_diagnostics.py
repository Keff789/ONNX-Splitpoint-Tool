"""Bounded numeric diagnostics without changing v26 acceptance predicates."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import onnx_splitpoint_tool.native_detection_postprocess as postprocess
from tests.test_v27926_deepx_full_decoded_pre_nms import (
    case, _suite, _decoded_outputs, _install_runtime,
)


def _contract(values):
    return postprocess.build_frozen_postprocess_contract(
        model_id="yolo11l", outputs={"actual_runtime_output": values},
        input_hw=[640, 640], original_wh=[640, 640],
        source_contract_family="decoded_pre_nms",
    )


@pytest.mark.parametrize("channel,value,violation", [
    (2, -1e-9, "negative_width"), (2, -27., "negative_width"),
    (3, -1e-9, "negative_height"), (3, -90., "negative_height"),
    (4, -1e-9, "scores_below_zero"), (4, -200., "scores_below_zero"),
    (4, np.nextafter(np.float32(1), np.float32(2)), "scores_above_one"),
    (4, 255., "scores_above_one"),
])
@pytest.mark.parametrize("transposed", [False, True])
def test_v26_range_predicate_stays_strict_with_reviewable_detail(channel, value, violation, transposed):
    values = _decoded_outputs()
    values[0, channel, 7] = value
    if transposed:
        values = values.transpose(0, 2, 1)
    original = values.copy()
    with pytest.raises(postprocess.FrozenPostprocessError) as caught:
        # Direct legacy/no-policy validation remains strict in v28.
        postprocess._verify_decoded_pre_nms_values({"actual_runtime_output": values})
    assert str(caught.value) == "decoded_pre_nms_values_invalid"
    diagnostic = caught.value.diagnostics
    tensor = diagnostic["tensors"][0]
    assert tensor["name"] == "actual_runtime_output"
    assert tensor["dtype"] == "float32"
    assert tensor["shape"] == list(values.shape)
    sample = tensor["violations"][violation]
    assert sample["count"] == 1
    assert sample["examples"][0]["index"] == ([0, 7, channel] if transposed else [0, channel, 7])
    assert sample["examples"][0]["value"] == float(np.float32(value))
    assert diagnostic["values_modified"] is False
    np.testing.assert_array_equal(values, original)
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("value,violation,serialized", [
    (np.nan, "nan", "NaN"), (np.inf, "positive_inf", "+Inf"),
    (-np.inf, "negative_inf", "-Inf"),
])
def test_nonfinite_retains_v26_reason_and_json_safe_examples(value, violation, serialized):
    values = _decoded_outputs()
    values[0, 5, 7] = value
    with pytest.raises(postprocess.FrozenPostprocessError) as caught:
        _contract(values)
    assert str(caught.value) == "decoded_pre_nms_nonfinite_values"
    row = caught.value.diagnostics["tensors"][0]
    assert row["all_finite"] is False
    assert row["violations"][violation]["count"] == 1
    assert row["violations"][violation]["examples"][0]["value"] == serialized
    json.dumps(caught.value.diagnostics, allow_nan=False)


def test_channel_extrema_and_example_limit_are_exact_and_bounded():
    values = _decoded_outputs()
    values[0, 0] = -1000.  # x/y may lie outside the image; not an invalid width.
    values[0, 1] = 1000.
    values[0, 2] = -2.
    values[0, 3] = -3.
    values[0, 4:] = 2.
    row = postprocess.inspect_decoded_pre_nms_values({"output": values}, sample_limit=3)["tensors"][0]
    assert {k: v["finite_min"] for k, v in row["xywh"].items()} == {
        "x": -1000., "y": 1000., "width": -2., "height": -3.,
    }
    assert row["class_scores"]["finite_min"] == row["class_scores"]["finite_max"] == 2.
    assert row["violations"]["scores_above_one"]["count"] == 80 * 8400
    assert all(len(v["examples"]) <= 3 for v in row["violations"].values())
    assert len(json.dumps(row)) < 12000


def test_valid_measured_contract_never_collects_statistics(monkeypatch):
    values = _decoded_outputs()
    values[0, 0] = -1000.
    values[0, 1] = 1000.
    values[0, 2:4] = -0.
    values[0, 4, 7] = 1.
    def unexpected(*args, **kwargs):
        raise AssertionError("valid measured path must not collect diagnostics")
    monkeypatch.setattr(postprocess, "inspect_decoded_pre_nms_values", unexpected)
    contract = _contract(values)
    postprocess.verify_frozen_postprocess_contract(contract, outputs={"actual_runtime_output": values})


def _probe_case(case, monkeypatch, output):
    suite = _suite()
    calls = _install_runtime(monkeypatch, output)
    image = case.root / "probe_image.png"
    Image.new("RGB", (640, 640), "black").save(image)
    monkeypatch.setattr(suite, "_deepx_find_prepared_feed_image", lambda *a: (image, "fixture"))
    run = {"benchmark_task": "detection", "model_id": "yolo11l", "setup_id": "test_deepx"}
    args = SimpleNamespace(runs=3, warmup=1, energy_measurement_only=False,
                           prepared_input_manifest="", quality_evidence_model_id="yolo11l")
    results = case.root / "results"
    results.mkdir()
    return suite, calls, run, args, results


def test_prepared_failure_preserves_original_error_values_and_input(case, monkeypatch):
    output = _decoded_outputs()
    output[0, 4, 7] = -0.00001
    suite, calls, run, args, results = _probe_case(case, monkeypatch, output)
    result = suite._run_deepx_prepared_feed_benchmark(case.root, case.cached, run, args, results)
    assert result["status"] == "runtime_failed"
    assert result["error"] == "FrozenPostprocessError: decoded_pre_nms_values_invalid"
    assert result["diagnostic_phase"] == "structural_probe"
    assert len(calls) == result["engine_call_count"] == 1
    assert result["prepared_input_binding_verified"] is True
    assert Path(result["prepared_input_manifest"]).is_file()
    assert result["output_value_diagnostics"]["tensors"][0]["violations"]["scores_below_zero"]["count"] == 1
    assert not list(results.glob("*.npz"))  # ordinary failed runs only attach bounded detail
    saved = json.loads((results / "deepx_prepared_feed_benchmark.json").read_text())
    assert saved == result


@pytest.mark.parametrize("invalid", [False, True])
def test_short_probe_uses_exact_feed_and_preserves_unmodified_snapshot(case, monkeypatch, invalid):
    output = _decoded_outputs()
    if invalid:
        output[0, 4, 7] = -0.00001
    suite, calls, run, args, results = _probe_case(case, monkeypatch, output)
    result = suite.run_deepx_output_value_probe(case.root, case.cached, run, args, results)
    assert result["status"] == ("runtime_failed" if invalid else "diagnostic_pass")
    assert result["diagnostic_only"] is True
    assert result["counts_as_benchmark"] is False
    assert len(calls) == result["engine_call_count"] == (1 if invalid else 2)
    assert not {"fps_makespan", "mean_ms", "makespan_s", "completed_frames"}.intersection(result)
    assert args.runs == 3 and args.warmup == 1 and args.prepared_input_manifest == ""
    snapshot = result["raw_output_snapshot"]
    assert snapshot["status"] == "saved"
    with np.load(results / snapshot["file"], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["tensor_000"], output)
    saved = json.loads((results / "deepx_output_value_probe.json").read_text())
    assert result == saved
    if not invalid:
        assert result["postprocess_completion_verified"] is True
        assert result["completed_detection_count"] == 2


def test_quality_decoder_failure_keeps_same_numeric_diagnostics(case, monkeypatch):
    output = _decoded_outputs()
    output[0, 3, 7] = -2.
    suite, *_ = _probe_case(case, monkeypatch, output)
    bound = suite._deepx_bind_authoritative_endpoint_contract(case.root, {"benchmark_task": "detection"}, case.contract)
    detections, result = suite._deepx_detection_decode(
        root=case.root, run={"model_id": "yolo11l"}, contract=bound,
        outputs=[output], orig_shape=(640, 640, 3), scale=1., pad_x=0, pad_y=0,
    )
    assert not detections and result["pass"] is False
    assert result["error"] == "FrozenPostprocessError: decoded_pre_nms_values_invalid"
    assert result["output_value_diagnostics"]["tensors"][0]["violations"]["negative_height"]["count"] == 1


def test_second_probe_failure_snapshot_matches_failing_tensor(case, monkeypatch):
    import sys
    output = _decoded_outputs()
    suite, calls, run, args, results = _probe_case(case, monkeypatch, output)
    base = sys.modules["dx_engine"].InferenceEngine
    class ChangingEngine(base):
        def run(self, feeds):
            values = super().run(feeds)
            if len(calls) == 2:
                values[0][0, 2, 9] = -24.
            return values
    monkeypatch.setattr(sys.modules["dx_engine"], "InferenceEngine", ChangingEngine)
    result = suite.run_deepx_output_value_probe(case.root, case.cached, run, args, results)
    assert result["status"] == "runtime_failed"
    assert result["diagnostic_phase"] == "untimed_completion_probe"
    assert result["output_diagnostic_observation"] == "failing_tensor"
    assert result["engine_call_count"] == 2
    row = result["output_value_diagnostics"]["tensors"][0]
    assert row["xywh"]["width"]["finite_min"] == -24.
    snapshot = result["raw_output_snapshot"]
    assert snapshot["phase"] == "untimed_completion_probe"
    assert len(list(results.glob("*.npz"))) == 1
    with np.load(results / snapshot["file"], allow_pickle=False) as archive:
        assert archive["tensor_000"][0, 2, 9] == -24.


def test_probe_snapshot_size_bound_does_not_weaken_geometry_check(case, monkeypatch):
    output = np.zeros((1, 84, 60000), dtype=np.float32)
    suite, calls, run, args, results = _probe_case(case, monkeypatch, output)
    result = suite.run_deepx_output_value_probe(case.root, case.cached, run, args, results)
    assert result["status"] == "runtime_failed"
    assert result["error"] == "FrozenPostprocessError: decoded_pre_nms_tensor_geometry_invalid"
    assert len(calls) == 1
    assert result["raw_output_snapshot"]["status"] == "size_limit_exceeded"
    assert not list(results.glob("*.npz"))


@pytest.mark.parametrize("success", [False, True])
def test_full_scientific_latency_never_comes_from_run_model(tmp_path, monkeypatch, success):
    suite = _suite()
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"mock artifact")
    monkeypatch.delenv("ONNX_SPLITPOINT_DEEPX_ALLOW_DIAGNOSTIC_FALLBACK", raising=False)
    monkeypatch.setattr(suite, "subprocess", SimpleNamespace(
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="mock", stderr=""),
        TimeoutExpired=TimeoutError, PIPE=-1,
    ))
    monkeypatch.setattr(suite, "_parse_deepx_run_model_output", lambda *a: {"latency_ms": 79.937})
    prepared = {
        "status": "ok" if success else "runtime_failed",
        "mean_ms": 5.0 if success else None,
        "preprocessing_contract_audit": {"pass": True},
        "postprocess_included": success,
        "postprocess_completion_verified": success,
    }
    monkeypatch.setattr(suite, "_run_deepx_prepared_feed_benchmark", lambda *a: prepared)
    monkeypatch.setattr(suite, "_run_deepx_semantic_validation", lambda *a: {})
    row = suite._run_deepx_full_run(
        tmp_path, {"id": "deepx_m1_full", "dxnn_path": str(dxnn),
                   "benchmark_task": "detection", "setup_id": "deepx_setup"},
        SimpleNamespace(runs=1, timeout=30, energy_measurement_only=False, validation_images=""),
        expected_endpoint_identity={"model_id": "yolo11l", "backend": "deepx_m1", "variant": "full"},
    )[0]
    assert row["full_e2e_latency_ms"] == row["full_e2e_mean_ms"] == (5.0 if success else None)
    assert row["diagnostic_full_e2e_latency_ms"] == 79.937
    assert row["full_measurement_endpoint"] == row["measurement_endpoint"] == ("completed_detection" if success else "")
    assert row["setup_id"] == "deepx_setup"


@pytest.mark.parametrize("implementation_hash,expected_error", [
    ("00f0ad986c8021b864ddff3f7697dfc50e93fda8e9adf4e243fcd2f6b2164eed", None),
    ("0" * 64, "frozen_postprocess_implementation_sha256_mismatch"),
    (postprocess._V27922_NATIVE_DETECTION_POSTPROCESS_SHA256, "decoded_pre_nms_implementation_not_supported"),
])
def test_only_exact_unchanged_v26_decoded_contract_is_compatible(implementation_hash, expected_error):
    values = _decoded_outputs()
    contract = _contract(values)
    contract.pop("decoded_pre_nms_score_policy", None)  # archived v26 semantics
    contract["implementation_artifacts"]["native_detection_postprocess"]["sha256"] = implementation_hash
    contract["invariant_identity"] = postprocess.frozen_postprocess_invariant_identity(contract)
    contract["invariant_contract_sha256"] = postprocess.canonical_json_sha256(contract["invariant_identity"])
    contract.pop("contract_sha256")
    contract["contract_sha256"] = postprocess.canonical_json_sha256(contract)
    if expected_error:
        with pytest.raises(postprocess.FrozenPostprocessError, match=expected_error):
            postprocess.verify_frozen_postprocess_contract(contract)
    else:
        original_hash = contract["contract_sha256"]
        verified = postprocess.verify_frozen_postprocess_contract(contract)
        assert verified["contract_sha256"] == original_hash
        assert verified["implementation_artifacts"]["native_detection_postprocess"]["sha256"] == implementation_hash
        assert implementation_hash not in postprocess._LEGACY_UNBOUND_DECODER_NATIVE_SHA256
        values[0, 4, 7] = -1e-9
        with pytest.raises(postprocess.FrozenPostprocessError, match="decoded_pre_nms_values_invalid"):
            postprocess.verify_frozen_postprocess_contract(contract, outputs={"actual_runtime_output": values})
