"""Observation-only Quality loop wall clock; synthetic outputs, no inference."""
from copy import deepcopy
import json

from PIL import Image
import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from tests.test_generic_quality_completion import make_config, preprocessing_contract, descriptor


def test_quality_capture_clock_lifecycle_and_exact_observed_scope(tmp_path, monkeypatch):
    fixture=make_config(tmp_path);cfg=fixture["config"];policy=cfg["completion_contract"]["processor_contract"]
    cfg["runtime_source_files"].append(descriptor(__file__));fixture["path"].write_text(json.dumps(cfg))
    runtime=quality.GenericQualityCompletion(fixture["path"],producer_context=fixture["context"],runner_source=__file__,
        input_hw=policy["input_hw"],expected_image_ids=cfg["expected_image_ids"],runtime_precision_identity=cfg["identity"]["precision"],
        confidence_threshold=.25,iou_threshold=.45,max_detections=300,letterbox=True,
        preprocessing_contract=preprocessing_contract(policy["input_hw"]),class_names=["class0"])
    clock=iter([1_000_000_000,6_000_000_000,7_000_000_000])
    monkeypatch.setattr(quality.time,"perf_counter_ns",lambda:next(clock))
    runtime.begin_capture()
    with pytest.raises(ValueError,match="already started"):runtime.begin_capture()
    records={}
    for image in fixture["images"]:
        with Image.open(image) as handle:wh=list(handle.size)
        records[image.name]=runtime.process(fixture["outputs"],image_path=image,original_wh=wh,
            prepared_inputs=fixture["inputs"],runtime_inputs=fixture["inputs"])
    timing=runtime.finish_capture()
    assert timing["image_count"]==2 and timing["wall_seconds"]==5.
    assert timing["performance_measurement_executed"] is False
    assert "central_CPU_reference_comparison_and_bootstrap" in timing["excludes"]
    with pytest.raises(ValueError,match="lifecycle"):runtime.finish_capture()
    proof=runtime.evidence(records);assert proof["capture_timing"]==timing
    rows=[{"image_id":key,"candidate":records[key],"ground_truth":[]} for key in sorted(records)]
    quality._validate_evidence(proof,rows,cfg["completion_contract"],cfg["expected_runtime_artifacts"])
    for key,value in [("wall_seconds",4.),("image_count",1),("scope","timed_throughput"),("performance_measurement_executed",True)]:
        damaged=deepcopy(proof);damaged["capture_timing"][key]=value
        damaged.pop("evidence_sha256");damaged["evidence_sha256"]=quality._hash(damaged)
        with pytest.raises(ValueError,match="timing"):
            quality._validate_evidence(damaged,rows,cfg["completion_contract"],cfg["expected_runtime_artifacts"])
