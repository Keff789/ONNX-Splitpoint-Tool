"""H8 untimed Quality uses the unchanged normal RGB/255 adapter; no devices."""
import ast
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from onnx_splitpoint_tool.runners import generic_quality_accounting as accounting
from onnx_splitpoint_tool.benchmark.evaluation_profiles import validate_evaluation_profile_payload
from onnx_splitpoint_tool.workflow.benchmark_binding import _benchmark_runs_from_profile
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import _apply_declared_run_variants
from onnx_splitpoint_tool.workflow.runner import _performance_plan_v263
from tests.test_generic_quality_completion import make_config, descriptor, preprocessing_contract
from tests.test_generic_quality_profile_transport import profile

RUNNER = Path(__file__).parents[1]/"onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"


def normal_adapter():
    names = {"_completion_hailo_inputs", "_completion_hailo_quantized_inputs"}
    tree = ast.parse(RUNNER.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == 2
    namespace = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["_completion_hailo_inputs"]


def h8_config(tmp_path, *, raw=False):
    fixture = make_config(tmp_path)
    cfg = fixture["config"]
    cfg["identity"].update(comparison_backend="hailo8", setup_id="synthetic_h8")
    fixture["context"].update(setup_id="synthetic_h8", stage1_backend="hailo8")
    cfg["runtime_source_files"].append(descriptor(RUNNER))
    values = np.arange(fixture["inputs"]["images"].size, dtype=np.uint32).astype(np.uint8).reshape(fixture["inputs"]["images"].shape)
    fixture["inputs"] = {"images": values}
    p = tmp_path/("prepared.bin" if raw else "prepared.npy")
    if raw: p.write_bytes(values.tobytes())
    else: np.save(p, values, allow_pickle=False)
    cfg["representative_input"] = {"image_id":fixture["images"][0].name,
        "source_image_sha256":quality._sha(fixture["images"][0]), "original_wh":[80,60],
        "runtime_input_name":"images", "runtime_input_adapter":deepcopy(quality.H8_INPUT_ADAPTER),
        "prepared_input":{**descriptor(p), "shape":list(values.shape), "dtype":"uint8",
                          "format":"raw" if raw else "npy", "input_name":"images"}}
    fixture["path"].write_text(json.dumps(cfg))
    return fixture


def h8_capture(fixture, *, actual=None, adapter=None):
    cfg = fixture["config"]; policy = cfg["completion_contract"]["processor_contract"]
    runtime = quality.GenericQualityCompletion(fixture["path"], producer_context=fixture["context"],
        runner_source=RUNNER, input_hw=policy["input_hw"], expected_image_ids=cfg["expected_image_ids"],
        runtime_precision_identity=cfg["identity"]["precision"], confidence_threshold=.25,
        iou_threshold=.45, max_detections=300, letterbox=True,
        preprocessing_contract=preprocessing_contract(policy["input_hw"]), class_names=["class0"],
        timed_input_adapter=normal_adapter() if adapter is None else adapter)
    class Session:
        def infer(self, feeds): return fixture["outputs"]
    session = Session(); records = {}
    feeds = normal_adapter()({"task":"detection", "input_binding":{"runtime_input_format":
        cfg["representative_input"]["runtime_input_adapter"]["runtime_input_format"]}},
                             "hailo8", fixture["inputs"]) if actual is None else actual
    for image in fixture["images"]:
        with runtime.observe_stage1_inputs(session, prepared_inputs=fixture["inputs"]):
            out = session.infer(feeds)
        with Image.open(image) as handle: wh = list(handle.size)
        records[image.name] = runtime.process(out, image_path=image, original_wh=wh, prepared_inputs=fixture["inputs"])
    evidence = runtime.evidence(records)
    rows = [{"image_id":name,"ground_truth":[],"candidate":records[name]} for name in sorted(records)]
    quality._validate_evidence(evidence, rows, cfg["completion_contract"], cfg["expected_runtime_artifacts"])
    return evidence, rows


@pytest.mark.parametrize("raw", [False, True])
def test_real_normal_h8_adapter_and_portable_completed_input_relation(tmp_path, raw):
    fixture = h8_config(tmp_path, raw=raw)
    before = fixture["inputs"]["images"].copy()
    evidence, _ = h8_capture(fixture)
    assert np.array_equal(fixture["inputs"]["images"], before)
    comparison = evidence["per_image"][0]["representative_input_comparison"]
    assert comparison["prepared_tensor"]["dtype"] == "uint8"
    assert comparison["actual_runtime_input"]["images"]["dtype"] == "float32"
    assert comparison["prepared_tensor"]["tensor_sha256"] != comparison["actual_runtime_input"]["images"]["tensor_sha256"]


@pytest.mark.parametrize("fault", ["raw_uint8", "float64", "wrong_scale"])
def test_wrong_observed_sdk_dtype_or_scale_rejects(tmp_path, fault):
    fixture = h8_config(tmp_path); value = fixture["inputs"]["images"]
    actual = value if fault == "raw_uint8" else value.astype(np.float64)/255 if fault == "float64" else value.astype(np.float32)/256
    with pytest.raises(ValueError, match="normal timed input relation"):
        h8_capture(fixture, actual={"images":actual})


@pytest.mark.parametrize("fault", ["missing", "scale", "other_backend", "missing_producer_backend"])
def test_h8_relation_explicit_scope_fails_closed(tmp_path, fault):
    fixture = h8_config(tmp_path)
    if fault == "missing": fixture["config"]["representative_input"].pop("runtime_input_adapter")
    elif fault == "scale": fixture["config"]["representative_input"]["runtime_input_adapter"]["runtime_input_format"] = "uint8"
    elif fault == "other_backend": fixture["config"]["identity"]["comparison_backend"] = "hailo10h"
    else: fixture["context"].pop("stage1_backend")
    fixture["path"].write_text(json.dumps(fixture["config"]))
    with pytest.raises(ValueError): h8_capture(fixture)


@pytest.mark.parametrize("fault", ["file", "expected_and_observed_scale", "drop_relation", "drop_representative"])
def test_portable_resealed_relation_cannot_claim_wrong_conversion(tmp_path, fault):
    fixture = h8_config(tmp_path); evidence, records = h8_capture(fixture)
    comp = evidence["per_image"][0]["representative_input_comparison"]
    if fault == "file": comp["runtime_input_adapter_proof"]["prepared_file_base64"] = "AAAA"
    elif fault == "expected_and_observed_scale":
        wrong = quality._input_snapshot({"images":fixture["inputs"]["images"].astype(np.float32)/256})
        comp["runtime_input_adapter_proof"]["expected_runtime_input"] = wrong
        comp["actual_runtime_input"] = wrong; evidence["per_image"][0]["runtime_inputs"] = wrong
    elif fault == "drop_relation": comp.pop("runtime_input_adapter_proof")
    else: evidence["representative_input"] = None
    evidence.pop("evidence_sha256"); evidence["evidence_sha256"] = quality._hash(evidence)
    with pytest.raises(ValueError):
        quality._validate_evidence(evidence, records, fixture["config"]["completion_contract"], fixture["config"]["expected_runtime_artifacts"])


def test_normal_profile_projection_and_no_fps_accounting_accept_exact_h8_opt_in():
    payload = profile(); row = payload["run_profiles"][0]
    row.update(id="hailo8_to_trt",stage1="hailo8",hardware_setup_id="synthetic_h8")
    payload["quality_gate"].setdefault("statistics",{})["execution_location"] = "central_management"
    payload = validate_evaluation_profile_payload(payload, source="synthetic H8 explicit Quality")
    formal = _benchmark_runs_from_profile(payload,["hailo8","tensorrt"])
    run = next(item for item in formal if item["id"] == row["id"])
    legacy = _apply_declared_run_variants([{"id":row["id"],"type":"matrix","stage1":"hailo8","stage2":"tensorrt"}],payload)
    assert accounting.authorized_runs({"runs":legacy},payload)
    projected = _performance_plan_v263({"runs":[run]},payload)
    assert projected["runs"] == [] and projected["performance_matrix_applicable"] is False


def test_h8_callback_must_be_actual_frozen_runner_function(tmp_path):
    fixture = h8_config(tmp_path)
    with pytest.raises(ValueError, match="actual frozen normal adapter"):
        h8_capture(fixture, adapter=lambda *args:fixture["inputs"])
