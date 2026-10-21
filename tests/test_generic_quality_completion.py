"""Real normal completion arithmetic, synthetic outputs/files only; no devices."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool.runners import generic_quality_completion as quality


def descriptor(path):
    return {"path": str(Path(path).resolve()), "sha256": quality._sha(path), "size_bytes": Path(path).stat().st_size}


def make_config(tmp_path, model="yolo26s"):
    """Reusable actual-loader fixture for the independent suite guard tests."""
    product = quality._product()
    if model == "yolo11l":
        from test_v27923_decoded_completion import _outputs, _source
        outputs = _outputs(); source = _source(tmp_path, outputs); hw = [640, 640]
    elif model == "yolov7_paper":
        from test_v272_yolov7_head_mapping import _heads_640, _raw_source
        outputs = _heads_640()
        for value in outputs.values():
            value.fill(-20)
        outputs["vendor_small"][0, 0, 30, 20, :4] = 0
        outputs["vendor_small"][0, 0, 30, 20, 4:6] = 20
        source = _raw_source(outputs, endpoint_hash="b"*64)
        # These synthetic outputs have no actual full ONNX file authority.
        # The normal registered decoder still binds the named YOLOv7 model.
        source.pop("model_sha256")
        hw = [640, 640]
    else:
        from test_v272_detection_completion_runtime import _decoded_outputs, _decoded_source
        outputs = _decoded_outputs(); source = _decoded_source(outputs); hw = [64, 64]
    preprocess = {"mode": "letterbox", "rgb": True, "pad_value": 114}
    contract = product.build_detection_completion_execution_contract(model_id=model, outputs=outputs,
        input_hw=hw, original_wh=[80, 60], source_endpoint_contract=source, preprocess=preprocess)
    artifacts = {}
    for role in quality.ARTIFACT_ROLES:
        path = tmp_path/(role+".test-only")
        path.write_bytes(("synthetic compiled/source fixture "+role).encode())
        artifacts[role] = descriptor(path)
    images = []
    for name, wh in (("a.png", [80, 60]), ("b.png", [60, 80])):
        path = tmp_path/name; Image.new("RGB", wh, color=(23, 53, 73)).save(path); images.append(path)
    modules = [Path(quality.__file__), Path(product.__file__), Path(quality._three_stage().__file__),
               Path(product._yolo_module.__file__), Path(quality._endpoint.__file__), Path(__file__)]
    config = {"schema": quality.CONFIG_SCHEMA, "schema_version": 1,
        "identity": {"model_id": model, "case_id": "b039", "setup_id": "test_only_h10", "precision": "float32_layout_fp16"},
        "completion_contract": contract, "source_endpoint_contract": source,
        "preprocess": preprocess, "expected_runtime_artifacts": artifacts,
        "generic_artifact_binding": {"artifacts": deepcopy(artifacts)},
        "runtime_source_files": [descriptor(path) for path in modules],
        "expected_image_ids": [p.name for p in images]}
    path = tmp_path/"quality.json"; path.write_text(json.dumps(config))
    context = {**config["identity"], "part1_model_path": artifacts["source_part1_onnx"]["path"],
        "part2_model_path": artifacts["source_part2_onnx"]["path"],
        "runtime_artifacts": [{"role": "stage1_runtime", "path": artifacts["part1_runtime"]["path"]},
                              {"role": "stage2_runtime", "path": artifacts["engine"]["path"]}]}
    return {"path": path, "config": config, "outputs": outputs, "context": context, "images": images,
            "inputs": {"images": np.full((*hw, 3), 114, dtype=np.uint8)}}


def capture(fixture):
    policy = fixture["config"]["completion_contract"]["processor_contract"]
    runtime = quality.GenericQualityCompletion(fixture["path"], producer_context=fixture["context"],
        runner_source=__file__, input_hw=policy["input_hw"], expected_image_ids=fixture["config"]["expected_image_ids"],
        runtime_precision_identity=fixture["config"]["identity"]["precision"],
        confidence_threshold=.25, iou_threshold=.45, max_detections=300, letterbox=True,
        preprocessing_contract=preprocessing_contract(policy["input_hw"]),
        class_names=["synthetic_class_"+str(index) for index in range(80)])
    records = {}
    class Session:
        def infer(self, feeds):
            return fixture["outputs"]
    session = Session(); original = session.infer
    for image in fixture["images"]:
        with runtime.observe_stage1_inputs(session):
            outputs = session.infer(fixture["inputs"])
        assert session.infer == original
        with Image.open(image) as source:
            wh = list(source.size)
        records[image.name] = runtime.process(outputs, image_path=image, original_wh=wh,
                                               prepared_inputs=fixture["inputs"])
    return runtime, records


def preprocessing_contract(input_hw):
    return {"schema": "onnx-splitpoint/image-preprocessing-contract", "schema_version": 2,
        "contract_scope": "prepared_rgb_uint8_semantics", "task": "detection",
        "preprocess_mode": "letterbox", "spatial_transform": "centered_letterbox",
        "target_hw": list(input_hw), "color_space": "RGB", "input_domain": "uint8_0_255",
        "resize_interpolation": "bilinear", "resize_rounding": "python_round_ties_to_even",
        "placement": "floor_top_left_remainder_bottom_right", "pad_value": 114,
        "letterbox_pad_value": 114, "image_scale": "norm", "letterbox": True}


def envelope(fixture, tmp_path):
    runtime, detections = capture(fixture)
    raw = runtime.evidence(detections)
    records = [{"image_id": image, "ground_truth": [], "candidate": detections[image]} for image in sorted(detections)]
    artifacts = fixture["config"]["expected_runtime_artifacts"]
    producer = {**fixture["config"]["identity"], "execution_role": "generic_composed_quality_candidate",
        "implementation_runner_sha256": raw["implementation_runner"]["sha256"],
        "implementation_runner_size_bytes": raw["implementation_runner"]["size_bytes"],
        "runtime_precision_identity": fixture["config"]["identity"]["precision"],
        "runtime_artifacts": [{"role": "stage1_runtime", **artifacts["part1_runtime"]},
                              {"role": "stage2_runtime", **artifacts["engine"]}],
        "model": {"part1_onnx_sha256": artifacts["source_part1_onnx"]["sha256"],
                  "part2_onnx_sha256": artifacts["source_part2_onnx"]["sha256"]}}
    projection = fixture["config"].get("source_endpoint_projection")
    if projection is not None:
        producer["model"]["source_onnx_sha256"] = projection["full_model"]["sha256"]
    producer, completion = quality.bind_completed_quality_producer(producer, {}, raw)
    extra = quality.completed_quality_export_fields(raw, records, producer)
    candidate = {"task": "detection", "variant": "composed", "producer_identity": producer,
                 "record_count": len(records), "records": records, **extra}
    candidate_path = tmp_path/"candidate.json"; candidate_path.write_text(json.dumps(candidate))
    request = {"task": "detection", "variant": "composed", "candidate": {**descriptor(candidate_path), "path": candidate_path.name},
        "record_count": len(records), "expected_image_ids": [r["image_id"] for r in records],
        "producer_identity": producer, **extra}
    request_path = tmp_path/"request.json"; request_path.write_text(json.dumps(request))
    result = {"task": "detection", "variant": "composed", "status": "completed", "technical_status": "completed",
              "source_request_sha256": quality._sha(request_path),
              "producer_identity_sha256": producer["producer_identity_sha256"]}
    return request, result, request_path, candidate_path


@pytest.mark.parametrize("model", ["yolo11l", "yolov7_paper", "yolo26s"])
def test_real_default_factory_geometry_and_every_record_proven(model, tmp_path):
    fixture = make_config(tmp_path, model)
    request, result, request_path, _ = envelope(fixture, tmp_path)
    proof = quality.validate_completed_quality_binding(request, result,
        fixture["config"]["completion_contract"], fixture["config"]["expected_runtime_artifacts"],
        request_path=request_path, reference_descriptor={"expected_image_ids": ["a.png", "b.png"]})
    assert proof["image_count"] == 2 and proof["all_actual_completed_records_verified"]
    evidence = request[quality.FIELD]
    assert [r["original_wh"] for r in evidence["per_image"]] == [[80, 60], [60, 80]]
    assert evidence["per_image"][0]["completion_contract"]["comparison_endpoint_contract"] == evidence["per_image"][1]["completion_contract"]["comparison_endpoint_contract"]
    if model == "yolo26s":
        assert len(evidence["per_image"][0]["attestation"]["last_result"]["detections"]) == 2  # no second NMS


@pytest.mark.parametrize("fault", ["contract", "artifact", "source", "missing_role", "alias", "image_order", "preprocess"])
def test_preparatory_config_rejects_changed_policy_bytes_and_scope(tmp_path, fault):
    fixture = make_config(tmp_path); config = fixture["config"]
    if fault == "contract":
        config["completion_contract"]["processor_contract"]["confidence_threshold"] = .1
    elif fault in ("artifact", "source"):
        p = config["expected_runtime_artifacts"]["part1_runtime"]["path"] if fault == "artifact" else config["runtime_source_files"][-1]["path"]
        if fault == "source":
            config["runtime_source_files"][-1]["sha256"] = "e"*64
        else:
            Path(p).write_bytes(b"changed synthetic test artifact")
    elif fault == "missing_role":
        config["expected_runtime_artifacts"].pop("part1_build_receipt")
    elif fault == "alias":
        config["generic_artifact_binding"]["artifacts"]["engine"]["path"] = config["expected_runtime_artifacts"]["part1_runtime"]["path"]
    elif fault == "image_order":
        config["expected_image_ids"].reverse()
    else:
        config["preprocess"]["pad_value"] = 0
    fixture["path"].write_text(json.dumps(config))
    with pytest.raises(ValueError):
        quality.load_quality_completion_config(fixture["path"])


@pytest.mark.parametrize("fault", ["strip", "candidate", "geometry", "counts", "source", "input", "producer", "result", "label", "preprocessing", "runner_closure", "runner_producer"])
def test_persisted_proof_rejects_tampered_or_legacy_raw_only_quality(tmp_path, fault):
    fixture = make_config(tmp_path)
    request, result, request_path, candidate_path = envelope(fixture, tmp_path)
    candidate = json.loads(candidate_path.read_text()); proof = request[quality.FIELD]
    if fault == "strip":
        request.pop(quality.FIELD)
    elif fault == "candidate":
        candidate["records"][0]["candidate"] = []
    elif fault == "geometry":
        proof["per_image"][0]["original_wh"] = [60, 80]
    elif fault == "counts":
        proof["per_image"][0]["attestation"]["completed_work_units"] = 2
    elif fault == "source":
        proof["runtime_source_files"][0]["sha256"] = "a"*64
    elif fault == "input":
        proof["per_image"][0]["runtime_inputs"] = {}
    elif fault == "producer":
        request["producer_identity"]["candidate_execution_completion_contract"]["completion_mode"] = "old_proxy_adapter"
    elif fault == "label":
        candidate["records"][0]["candidate"][0]["class_name"] = "wrong_class"
    elif fault == "preprocessing":
        proof["image_preprocessing_contract"]["resize_interpolation"] = "nearest"
        proof["image_preprocessing_contract_sha256"] = quality._hash(proof["image_preprocessing_contract"])
    elif fault == "runner_closure":
        proof["runtime_source_files"] = [item for item in proof["runtime_source_files"]
            if item["path"] != proof["implementation_runner"]["path"]]
        source_map = quality._sources(proof["runtime_source_files"], check_files=False)
        proof["source_binding"] = {"before": source_map, "after": source_map}
    elif fault == "runner_producer":
        request["producer_identity"]["implementation_runner_sha256"] = "e"*64
        request["producer_identity"].pop("producer_identity_sha256")
        request["producer_identity"]["producer_identity_sha256"] = quality._hash(request["producer_identity"])
    else:
        result["source_request_sha256"] = "a"*64
    if fault not in ("strip", "candidate", "producer", "result", "label"):
        proof.pop("evidence_sha256"); proof["evidence_sha256"] = quality._hash(proof)
        candidate[quality.FIELD] = deepcopy(proof)
    candidate_path.write_text(json.dumps(candidate)); request["candidate"]["sha256"] = quality._sha(candidate_path)
    request_path.write_text(json.dumps(request))
    if fault != "result":
        result["source_request_sha256"] = quality._sha(request_path)
    with pytest.raises(ValueError):
        quality.validate_completed_quality_binding(request, result, fixture["config"]["completion_contract"],
            fixture["config"]["expected_runtime_artifacts"], request_path=request_path, reference_descriptor={})


def test_actual_input_observer_forwards_once_and_restores_on_failure(tmp_path):
    fixture = make_config(tmp_path); runtime, _ = capture(fixture)
    class Session:
        def infer(self, inputs):
            inputs["images"].fill(0)
            raise RuntimeError("test-only failed inference")
    session = Session(); original = session.infer; values = {"images": np.ones((2, 2, 3), np.uint8)}
    with pytest.raises(RuntimeError, match="failed inference"):
        with runtime.observe_stage1_inputs(session):
            session.infer(values)
    assert session.infer == original
    assert np.all(runtime._runtime_inputs["images"] == 1)


def test_representative_proves_actual_uint8_backend_input_not_float_pre_adapter(tmp_path):
    fixture = make_config(tmp_path); image = fixture["images"][0]
    npy = tmp_path/"input.npy"; np.save(npy, fixture["inputs"]["images"])
    representative = {"image_id": image.name, "source_image_sha256": quality._sha(image), "original_wh": [80, 60],
        "prepared_input": {**descriptor(npy), "shape": list(fixture["inputs"]["images"].shape), "dtype": "uint8", "format": "npy"},
        "runtime_input_name": "images"}
    fixture["config"]["representative_input"] = representative
    fixture["path"].write_text(json.dumps(fixture["config"]))
    runtime, _ = capture(fixture)
    assert runtime.entries[image.name]["representative_input_comparison"]
    fixture["inputs"] = {"images": fixture["inputs"]["images"].astype(np.float32)/255}
    with pytest.raises(ValueError, match="representative actual runtime input differs"):
        capture(fixture)


def test_generated_bundle_imports_quality_module_from_deployed_package(tmp_path):
    from onnx_splitpoint_tool.workflow.generic_completed_task import prepare_runner_bundle
    bundle = tmp_path/"generated"; prepare_runner_bundle(bundle)
    program = "import sys;sys.path.insert(0,sys.argv[1]);from splitpoint_runners import generic_quality_completion as q;assert q._product().__name__=='splitpoint_runners.native_detection_postprocess';assert q._three_stage().__name__=='splitpoint_runners.native_three_stage'"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(bundle)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("field", ["native_source", "native_command_contract", "native_uid", "original_run_cfg"])
def test_preparatory_quality_has_no_native_authority(tmp_path, field):
    fixture = make_config(tmp_path)
    fixture["config"][field] = {"native_split_quality_binding": "test-only-claim"} if field == "original_run_cfg" else "test-only-claim"
    fixture["path"].write_text(json.dumps(fixture["config"]))
    with pytest.raises(ValueError, match="Native"):
        quality.load_quality_completion_config(fixture["path"])


def test_suite_relative_paths_resolve_normally_and_escape_rejected(tmp_path):
    fixture = make_config(tmp_path)
    directory = tmp_path/"quality_runtime"; directory.mkdir(); path = directory/"b039.json"
    config = deepcopy(fixture["config"]); config["path_root"] = "suite_root"
    for descriptor in config["expected_runtime_artifacts"].values():
        descriptor["path"] = Path(descriptor["path"]).name
    config["generic_artifact_binding"]["artifacts"] = deepcopy(config["expected_runtime_artifacts"])
    path.write_text(json.dumps(config))
    checked = quality.load_quality_completion_config(path)
    assert checked["expected_runtime_artifacts"]["engine"]["path"] == str(tmp_path/"engine.test-only")
    config["expected_runtime_artifacts"]["engine"]["path"] = "../escape.engine"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="escapes suite"):
        quality.load_quality_completion_config(path)


def projected_fixture(tmp_path):
    fixture = make_config(tmp_path)
    full = tmp_path/"full-source.test-only.onnx"; full.write_bytes(b"synthetic independent full model")
    artifacts = fixture["config"]["expected_runtime_artifacts"]
    authority = {**fixture["config"]["source_endpoint_contract"],
        "full_model_sha256": quality._sha(full),
        "terminal_model_sha256": artifacts["source_part2_onnx"]["sha256"]}
    fields = quality._endpoint.project_generic_quality_endpoint(authority,
        full_model=descriptor(full), part2_source=artifacts["source_part2_onnx"])
    fixture["config"].update(fields)
    policy = fixture["config"]["completion_contract"]["processor_contract"]
    fixture["config"]["completion_contract"] = quality._product().build_detection_completion_execution_contract(
        model_id="yolo26s", outputs=fixture["outputs"], input_hw=policy["input_hw"], original_wh=[80, 60],
        source_endpoint_contract=fields["source_endpoint_contract"], preprocess=fixture["config"]["preprocess"])
    fixture["context"]["source_model_path"] = str(full)
    fixture["path"].write_text(json.dumps(fixture["config"]))
    return fixture


def test_completed_quality_preserves_distinct_full_and_p2_source_authority(tmp_path):
    fixture = projected_fixture(tmp_path)
    request, result, request_path, _ = envelope(fixture, tmp_path)
    proof = quality.validate_completed_quality_binding(request, result, fixture["config"]["completion_contract"],
        fixture["config"]["expected_runtime_artifacts"], request_path=request_path, reference_descriptor={})
    assert proof["all_actual_completed_records_verified"]
    evidence = request[quality.FIELD]
    assert evidence["source_endpoint_authority"] == fixture["config"]["source_endpoint_authority"]
    assert "terminal_model_sha256" not in evidence["source_endpoint_contract"]
    assert evidence["source_endpoint_projection"]["part2_source"]["sha256"] != evidence["source_endpoint_projection"]["full_model"]["sha256"]


@pytest.mark.parametrize("fault", ["actual_full_model", "exported_full_model"])
def test_completed_quality_rejects_wrong_actual_or_exported_full_model(tmp_path, fault):
    fixture = projected_fixture(tmp_path)
    if fault == "actual_full_model":
        fixture["context"]["source_model_path"] = fixture["config"]["expected_runtime_artifacts"]["source_part2_onnx"]["path"]
        with pytest.raises(ValueError, match="actual full model"):
            capture(fixture)
    else:
        runtime, detections = capture(fixture)
        raw = runtime.evidence(detections)
        request, _, _, _ = envelope(fixture, tmp_path)
        producer = deepcopy(request["producer_identity"])
        producer["model"]["source_onnx_sha256"] = "b"*64
        producer.pop("producer_identity_sha256")
        producer["producer_identity_sha256"] = quality._hash(producer)
        records = [{"image_id": image, "ground_truth": [], "candidate": detections[image]} for image in sorted(detections)]
        with pytest.raises(ValueError, match="exporter full source"):
            quality.completed_quality_export_fields(raw, records, producer)
