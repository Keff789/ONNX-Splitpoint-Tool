"""Untimed live boundary and interpreter provenance; synthetic SDK metadata only."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from tests.test_generic_quality_completion import make_config, capture


def observed_boundary():
    source = (Path(__file__).parents[1] / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text()
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.FunctionDef) and node.name == "_generic_boundary_runtime")
    p1 = {"outputs": {"cut_a": {"shape": [1, 3], "dtype": "float32"},
                       "cut_b": {"shape": [1, 2], "dtype": "uint8"}}}
    p2 = {"inputs": {"trt_a": {"shape": [1, 3], "dtype": "float16"},
                      "trt_b": {"shape": [1, 2], "dtype": "uint8"},
                      "image": {"shape": [1, 3, 4, 4], "dtype": "float32"}}}
    namespace = {"hailo_p1": None,
        "deepx_p1": SimpleNamespace(output_names=list(p1["outputs"]), describe_io=lambda: p1),
        "native_p2_sess": SimpleNamespace(input_names=list(p2["inputs"]), describe_io=lambda: p2),
        "p2_in_map": {"trt_a": "cut_a", "trt_b": "cut_b", "image": "image"},
        "feeds_full": {"unrelated_global_feed": object()}, "_boundary_mode": "canonical_float32",
        "part1_cut_names_manifest": ["cut_a", "cut_b"], "part2_cut_names_manifest": ["trt_a", "trt_b"],
        "manifest_cut_name_map_p2_to_p1": {"trt_a": "cut_a", "trt_b": "cut_b"}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual generated observer>", "exec"), namespace)
    return namespace


def test_actual_generated_observer_uses_resolved_map_and_current_feed_without_inference():
    namespace = observed_boundary()
    value = namespace["_generic_boundary_runtime"]({"image": object()})
    assert value["part2_to_part1"] == {"trt_a": "cut_a", "trt_b": "cut_b"}
    assert value["passthrough_inputs"] == {"image": "image"}
    assert value["part1_io"]["outputs"]["cut_b"]["dtype"] == "uint8"
    assert value["transfer_contract"]["hailo_exact_dequantization"] is False
    assert namespace["_generic_boundary_runtime"]()["passthrough_inputs"] == {}
    namespace["p2_in_map"] = None
    with pytest.raises(RuntimeError, match="resolved actual"):
        namespace["_generic_boundary_runtime"]()


def test_context_captures_actual_python_and_only_normal_execution_environment(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("NOT_AN_EXECUTION_ENV_SECRET", "must-not-be-captured")
    context = quality.capture_runtime_context(observed_boundary()["_generic_boundary_runtime"]())
    environment = context["environment"]
    assert environment["executable"] == str(Path(sys.executable).absolute())
    assert environment["executable_sha256"] == quality._sha(Path(sys.executable).resolve())
    assert environment["environment"]["OMP_NUM_THREADS"] == "3"
    assert "NOT_AN_EXECUTION_ENV_SECRET" not in environment["environment"]


def test_context_is_bound_into_actual_quality_evidence_and_detects_tampering(tmp_path):
    fixture = make_config(tmp_path)
    runtime, results = capture(fixture)
    context = quality.capture_runtime_context(observed_boundary()["_generic_boundary_runtime"]())
    with pytest.raises(ValueError, match="every image"):
        runtime.evidence(results, runtime_context=context)
    for _ in results:
        runtime.observe_boundary_runtime(context["boundary_runtime"])
        values = {"cut_a": np.ones((1, 3), dtype=np.float32)}
        with runtime.observe_boundary_transfer(lambda tensors: tensors) as transfer:
            assert transfer(values) is values
    evidence = runtime.evidence(results, runtime_context=context)
    records = [{"image_id": key, "ground_truth": [], "candidate": results[key]} for key in sorted(results)]
    assert quality._validate_evidence(evidence, records, fixture["config"]["completion_contract"],
        fixture["config"]["expected_runtime_artifacts"])["runtime_context"] == {
            **context, "boundary_observation_count": len(results),
            "logical_boundary": {"part1_outputs": {"cut_a": {"shape": [1, 3], "dtype": "float32"}},
                                 "part2_inputs": {"cut_a": {"shape": [1, 3], "dtype": "float32"}}}}
    context["boundary_runtime"]["part2_to_part1"]["trt_b"] = "cut_a"
    with pytest.raises(ValueError, match="changed between images"):
        runtime.observe_boundary_runtime(context["boundary_runtime"])
    assert evidence["runtime_context"] != context
    evidence["runtime_context"] = context
    with pytest.raises(ValueError, match="evidence hash"):
        quality._validate_evidence(evidence, records, fixture["config"]["completion_contract"],
            fixture["config"]["expected_runtime_artifacts"])


def test_transfer_observer_preserves_array_identity_and_forwarded_arguments(tmp_path):
    runtime, _ = capture(make_config(tmp_path))
    incoming = {"cut_a": np.ones((1, 3), dtype=np.float32)}
    outgoing = {"trt_a": np.ones((1, 3), dtype=np.float16)}
    calls = []
    def original(tensors, *, feeds_full_override):
        assert tensors is incoming
        calls.append(feeds_full_override)
        return outgoing
    with runtime.observe_boundary_transfer(original) as transfer:
        assert transfer(incoming, feeds_full_override={"image": "sentinel"}) is outgoing
    assert calls == [{"image": "sentinel"}]
    assert runtime._logical_boundary == {
        "part1_outputs": {"cut_a": {"shape": [1, 3], "dtype": "float32"}},
        "part2_inputs": {"trt_a": {"shape": [1, 3], "dtype": "float16"}}}
    with pytest.raises(ValueError, match="shapes/dtypes changed"):
        with runtime.observe_boundary_transfer(lambda value: value) as transfer:
            transfer({"cut_a": np.ones((1, 4), dtype=np.float32)})
    with pytest.raises(ValueError, match="was not observed"):
        with runtime.observe_boundary_transfer(original):
            pass


@pytest.mark.parametrize("fault", ["no_mapping", "no_io", "no_python", "no_hash", "relative_python"])
def test_missing_observed_runtime_authority_rejected(fault):
    value = quality.capture_runtime_context(observed_boundary()["_generic_boundary_runtime"]())
    if fault == "no_mapping":
        value["boundary_runtime"]["part2_to_part1"] = {}
    elif fault == "no_io":
        value["boundary_runtime"]["part1_io"] = {}
    elif fault == "no_python":
        value["environment"].pop("executable")
    elif fault == "no_hash":
        value["environment"]["executable_sha256"] = ""
    else:
        value["environment"]["executable"] = "python3"
    with pytest.raises(ValueError):
        quality.validate_runtime_context(value)
