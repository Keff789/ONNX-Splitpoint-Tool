"""Bounded real CPU completions plus the normal N5000 portable protocol.

The production population guard remains 5000.  A test-only loader wrapper limits
the actual fake-hardware FIFO loop to two image geometries after that guard has
passed.  The separate envelope phase replays compact fixture observations to
5000 IDs and mocks only their completion-verifier boundary.  No hardware or
bootstrap runs here, and replayed rows are not hardware completion evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest

from onnx_splitpoint_tool import native_dataset_quality as capture
from onnx_splitpoint_tool import native_dataset_quality_export as exporter
from onnx_splitpoint_tool import native_detection_postprocess as postprocess
from onnx_splitpoint_tool.native_dataset_quality_h8 import _load_native_runtime
from onnx_splitpoint_tool.native_output_endpoint import load_authoritative_output_contract, runtime_output_contract
from onnx_splitpoint_tool.native_split_quality import seal_native_split_quality_binding
from onnx_splitpoint_tool.quality_cache import image_ids_fingerprint
from onnx_splitpoint_tool.quality_service import quality_request_from_manifest
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from test_v269d_trt_central_quality_producer import _detection_quality_contract
from test_v269f_native_split_receipt_validation import _fixture_payload
from test_v272_detection_completion_runtime import _decoded_outputs


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: Any) -> dict[str, Any]:
    path.write_text(json.dumps(payload, sort_keys=True))
    return capture.file_descriptor(path)


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    prior_payload, _ = _fixture_payload(tmp_path / "prior")
    prior = seal_native_split_quality_binding(prior_payload)
    prior_file = _write(tmp_path / "prior.json", prior)
    dataset = tmp_path / "images"
    dataset.mkdir()
    images = [dataset / "image-00000.png", dataset / "image-00001.png"]
    for image, wh in zip(images, [(80, 60), (60, 80)]):
        Image.new("RGB", wh, (21, 47, 83)).save(image)
    rows = [{"image_id": f"image-{index:05d}.png", "image_sha256": capture.file_descriptor(images[index % 2])["sha256"],
             "ground_truth": []} for index in range(5000)]
    ground_truth = _write(tmp_path / "ground_truth.json", {"records": rows})
    quality = _detection_quality_contract()
    quality["preprocessing"]["identity"]["input_hw"] = [64, 64]
    quality["preprocessing"]["sha256"] = exporter.canonical_json_sha256(quality["preprocessing"]["identity"])
    quality["dataset"].update(image_count=5000, image_ids_sha256=image_ids_fingerprint([row["image_id"] for row in rows]),
        ground_truth_sha256=exporter.canonical_json_sha256([{"image_id": row["image_id"], "ground_truth": []} for row in rows]))
    quality.pop("quality_contract_sha256")
    quality["quality_contract_sha256"] = exporter.canonical_json_sha256(quality)
    benchmark = tmp_path / "benchmark_set"
    benchmark.mkdir()
    _write(benchmark / "output_contracts.json", {"schema": "onnx-splitpoint/output-contracts", "schema_version": 1,
        "model_id": "yolo26s", "task": "detection", "contracts": [{"schema": "onnx-splitpoint/output-contract", "schema_version": 1,
        "model_id": "yolo26s", "task": "detection", "backend": "tensorrt", "variant": "full", "contract_status": "recorded",
        "endpoint_mode": "decoded_nms", "host_tail_required": False, "postprocessing_required": False,
        "source_coordinate_space": "model_input_letterbox_xyxy_pixels"}]})
    outputs = _decoded_outputs()
    declaration = load_authoritative_output_contract(benchmark, backend="tensorrt", model_id="yolo26s", variant="full", task="detection")
    source = runtime_output_contract("detection", outputs, raw_fallback=False, declared_contract=declaration)
    timed_contract = postprocess.build_detection_completion_execution_contract(
        model_id="yolo26s", outputs=outputs, input_hw=[64, 64], original_wh=[80, 60], source_endpoint_contract=source,
        preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114})
    timed_file = _write(tmp_path / "timed.json", {"model_id": "yolo26s", "case": "b038", "fps_makespan": 10.0,
        "setup_id": "hailo8_setup", "precision": "uint8_dequant_fp16", "completion_runtime_mode": "strict",
        "completion_execution_contract": timed_contract, "native_split_quality_binding": prior, "completed_frames": 1000})
    config = {"schema": "onnx-splitpoint/native-dataset-quality-config", "schema_version": 1,
        "performance_claims_emitted": False,
        "identity": {"eval_run_id": "fresh-capture", "model_id": "yolo26s", "case_id": "b038", "setup_id": "hailo8_setup",
                     "source_run_id": "hailo8_to_trt", "backend": "hailo8_to_trt", "runtime_precision_identity": "uint8_dequant_fp16"},
        "native": {"engine_build_forbidden": True, "task": "detection", "benchmark_set": str(benchmark),
                   "native_split_quality_binding": prior_file["path"], "preprocess_mode": "letterbox", "letterbox_pad_value": 114},
        "ground_truth": ground_truth, "dataset_root": str(dataset), "prior_artifact_selection_file": prior_file,
        "prior_artifact_selection": prior, "quality_contract": quality, "quality_contract_sha256": quality["quality_contract_sha256"],
        "source_model_sha256": quality["model"]["sha256"], "timed_fps_authority": timed_file,
        "timed_completion_contract": timed_contract, "completion_runtime_mode": "strict", "policy_sha256": "9" * 64,
        "metric_gate_config": {"metric": "map_50_95", "non_inferiority_margin": 0.01},
        "statistics": {"method": "paired_bootstrap", "bootstrap_repetitions": 1000, "seed": 20260710,
                       "confidence_level": 0.95, "decision": "lower_one_sided_bound"}}
    config_path = tmp_path / "config.json"
    _write(config_path, config)
    state = SimpleNamespace(events=[], observed=[], fault=None, export_calls=0, verifier_calls=0, sources_calls=0, attestation_calls=0)
    actual_attestation = postprocess.DetectionCompletionRuntime.attestation

    def count_actual_attestation(runtime: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        state.attestation_calls += 1
        return actual_attestation(runtime, *args, **kwargs)

    monkeypatch.setattr(postprocess.DetectionCompletionRuntime, "attestation", count_actual_attestation)
    native = _load_native_runtime()
    prepared = SimpleNamespace(input_names=["images"], handle=SimpleNamespace(runtime_input_shapes={"images": (64, 64, 3)}))

    class Backend:
        def run(self, _prepared: Any, inputs: Any) -> Any:
            state.events.append("hailo_run")
            return SimpleNamespace(outputs={"cut": np.zeros((2, 2, 2), np.uint8)})

    class TRT:
        inputs = ["cut"]
        shapes = {"cut": (1, 2, 2, 2)}
        dtypes = {"cut": np.dtype("uint8")}

        def run(self, feeds: Any) -> dict[str, np.ndarray]:
            state.events.append("trt_run")
            return outputs

    class Native:
        output_signature = {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in outputs.items()}
        source_paths = (native.__file__,)

        def __init__(self, _config: Any, _work: Path):
            pass

        def __enter__(self) -> Native:
            state.events.append("open")
            return self

        def __exit__(self, *args: Any) -> None:
            state.events.append("close")

        def run_image(self, image: Path, completion: Any) -> dict[str, Any]:
            if state.fault == "missing_completion":
                return {"native_completed_count": 0}
            timing: dict[str, Any] = {}
            inputs = native._hailo8_python_input(prepared, image, quantized=True, preprocess_mode="letterbox",
                letterbox_pad_value=114, task="detection", timing_input_binding=timing)
            result = native._hailo8_python_fifo_run(Backend(), prepared, inputs, TRT(), frames=1, warmup=0,
                duration_s=0.0, queue_depth=3, completion_runtime=completion, warmup_completion_runtime=None,
                expected_boundary_name="cut", expected_boundary_shape=[2, 2, 2], expected_boundary_dtype="uint8")
            state.observed.append(deepcopy(result))
            return {"native_completed_count": result["completed_work_units"], "completion_execution_attestation": result["completion_execution_attestation"],
                "completion_execution_contract": result["completion_execution_contract"], "prepared_input_snapshot": [{"name": "images",
                    "shape": [64, 64, 3], "dtype": "uint8", "sha256": hashlib.sha256(inputs["images"].tobytes()).hexdigest()}],
                "timing_input_binding": timing, "runtime_boundary_evidence": result["strict_quality_boundary"]}

    actual_load = capture.load_capture_config

    def bounded_load(path: Path) -> dict[str, Any]:
        loaded = actual_load(path)
        assert len(loaded["_ground_truth_records"]) == capture.IMAGE_COUNT == 5000
        loaded["_ground_truth_records"] = loaded["_ground_truth_records"][:2]
        return loaded

    monkeypatch.setattr(capture, "load_capture_config", bounded_load)
    actual_sources = capture._sources

    def unit_sources(_paths: Any) -> list[dict[str, Any]]:
        # The test Native class has no Product source path.  The unit closure
        # truthfully lists only actual capture/export/completion/FIFO sources.
        state.sources_calls += 1
        from onnx_splitpoint_tool.native_dataset_quality_consumer import required_native_dataset_source_paths
        source_paths = [ROOT / relative for relative in required_native_dataset_source_paths("hailo8_to_trt")]
        result = actual_sources(source_paths)
        if state.fault == "source_change" and state.sources_calls > 1:
            result[0]["sha256"] = "0" * 64
        return result

    monkeypatch.setattr(capture, "_sources", unit_sources)
    actual_export = exporter.export_native_quality_request

    def replay_export(config: Any, records: list[Any], completions: list[Any], sources: list[Any], output: Path):
        state.export_calls += 1
        assert len(records) == len(completions) == 2
        state.captured_records = deepcopy(records)
        state.captured_completions = deepcopy(completions)
        expanded_records, expanded_completions = [], []
        state.fixture_attestations = {}
        for ordinal, expected in enumerate(rows):
            index = ordinal % 2
            replay_contract = completions[index]["execution_contract"]
            compact_attestation = {"unit_fixture_verified_attestation_sha256": completions[index]["execution_attestation"]["attestation_sha256"],
                "completion_count": 1, "completed_work_units": 1, "observation_relation": "same_hotloop_sentinel",
                "last_result": {"detections": records[index]["candidate"]}}
            state.fixture_attestations[compact_attestation["unit_fixture_verified_attestation_sha256"]] = completions[index]
            expanded_records.append({**records[index], "image_id": expected["image_id"]})
            expanded_completions.append({**completions[index], "image_id": expected["image_id"],
                "execution_contract": replay_contract, "execution_attestation": compact_attestation})
        return actual_export(config, expanded_records, expanded_completions, sources, output)

    monkeypatch.setattr(capture, "export_native_quality_request", replay_export)
    return SimpleNamespace(path=config_path, config=config, rows=rows, images=images, state=state, factory=Native,
                           output=tmp_path / "capture", actual_load=actual_load)


def test_bounded_real_native_capture_then_normal_N5000_export_and_intake(fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = capture.capture_native_quality(fixture.path, fixture.output, runtime_factory=fixture.factory)
    state = fixture.state
    assert state.events.count("open") == state.events.count("close") == 1
    assert state.events.count("hailo_run") == state.events.count("trt_run") == 2
    assert state.export_calls == 1 and len(state.observed) == state.attestation_calls == 2
    first, second = state.captured_completions
    assert first["execution_contract"]["processor_contract"]["original_wh"] == [80, 60]
    assert second["execution_contract"]["processor_contract"]["original_wh"] == [60, 80]
    assert first["execution_contract"]["contract_sha256"] != second["execution_contract"]["contract_sha256"]
    assert first["execution_attestation"]["last_result"]["detections"] != second["execution_attestation"]["last_result"]["detections"]
    for observed in state.observed:
        assert observed["completion_execution_attestation"]["completion_count"] == 1
        assert len(observed["completion_execution_attestation"]["last_result"]["detections"]) == 2
        assert all("class_name" not in row for row in observed["completion_execution_attestation"]["last_result"]["detections"])
        assert observed["completion_execution_contract"]["processor_contract"]["host_nms_applied"] is False
    request_path = Path(descriptor["path"])
    request = json.loads(request_path.read_text())
    candidate = json.loads((request_path.parent / request["candidate"]["path"]).read_text())

    def verify_compact_unit_replay(raw: dict[str, Any], *, execution_contract: dict[str, Any], expected_observation_relation: str) -> dict[str, Any]:
        # Only the compact replay phase is mocked.  Both original observations
        # above already passed the actual Native FIFO and normal verifier.
        state.verifier_calls += 1
        original = state.fixture_attestations[raw["unit_fixture_verified_attestation_sha256"]]
        assert execution_contract == original["execution_contract"]
        assert raw["last_result"]["detections"] == original["execution_attestation"]["last_result"]["detections"]
        assert expected_observation_relation == "same_hotloop_sentinel"
        return original["execution_attestation"]

    monkeypatch.setattr(postprocess, "verify_detection_completion_execution_attestation", verify_compact_unit_replay)
    reference = {"schema": "onnx-splitpoint/task-quality-reference-input", "schema_version": 1,
        "task": "detection", "variant": "full", "pairing_key": "image_id", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "provenance_required": True, "quality_contract": fixture.config["quality_contract"],
        "quality_contract_sha256": fixture.config["quality_contract_sha256"], "record_count": 5000,
        "records": [{"image_id": row["image_id"], "ground_truth": [], "reference": row["candidate"]} for row in candidate["records"]]}
    reference_path = fixture.output.parent / "cpu_reference.json"
    reference_descriptor = _write(reference_path, reference)
    loaded = quality_request_from_manifest(request_path, reference_artifact=reference_descriptor, reference_artifact_bytes=reference_path.read_bytes())
    assert len(loaded.candidate_records) == len(loaded.reference_records) == state.verifier_calls == 5000
    assert loaded.repetitions == 1000 and loaded.seed == 20260710
    assert loaded.artifact_provenance_binding_status == "verified" and loaded.artifact_provenance_claim_eligible is True
    identity = EvaluationWorkflowRunner._quality_request_identity(request_path, model_id="yolo26s", variant="composed", manifest=request)
    assert identity["identity_valid"] is True and identity["producer_identity_validated"] is True, identity


@pytest.mark.parametrize("fault,expected", [("changed_image", "source image changed"),
    ("missing_completion", "image did not complete exactly once"), ("source_change", "implementation changed during capture")])
def test_capture_fails_closed_before_request_publication(fixture: SimpleNamespace, fault: str, expected: str) -> None:
    fixture.state.fault = fault
    if fault == "changed_image":
        Image.new("RGB", (60, 80), (0, 0, 0)).save(fixture.images[1])
    with pytest.raises(ValueError, match=expected):
        capture.capture_native_quality(fixture.path, fixture.output, runtime_factory=fixture.factory)
    assert fixture.state.events[-1] == "close"
    assert fixture.state.export_calls == 0
    assert not (fixture.output / "composed_request.json").exists()
    failure = json.loads((fixture.output / "native_quality_capture_failure.json").read_text())
    assert failure["valid_capture"] is False and failure["performance_claims_emitted"] is False


def test_production_N5000_guard_still_rejects_short_population(fixture: SimpleNamespace) -> None:
    short = {"records": fixture.rows[:-1]}
    ground_truth_path = Path(fixture.config["ground_truth"]["path"])
    fixture.config["ground_truth"] = _write(ground_truth_path, short)
    _write(fixture.path, fixture.config)
    with pytest.raises(ValueError, match="exactly 5000 frozen images"):
        fixture.actual_load(fixture.path)
    assert capture.IMAGE_COUNT == 5000


def test_file_descriptor_works_without_python311_file_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    payload = b"native-quality-streaming-hash\x00" * 310000
    assert len(payload) > 8 * 1024 * 1024
    path = tmp_path / "multi_chunk.bin"
    path.write_bytes(payload)
    assert capture.file_descriptor(path) == {"path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


@pytest.mark.parametrize("fault,message", [("contract", "timed completion authority differs"),
    ("artifacts", "timed artifact authority differs"), ("missing_count", "timed completed source missing")])
def test_original_timed_authority_must_bind_actual_contract_artifacts_and_completion(fixture: SimpleNamespace, fault: str, message: str) -> None:
    path = Path(fixture.config["timed_fps_authority"]["path"])
    timed = json.loads(path.read_text())
    if fault == "contract":
        timed["completion_execution_contract"]["contract_sha256"] = "0" * 64
    elif fault == "artifacts":
        timed["native_split_quality_binding"]["artifacts"]["engine"]["sha256"] = "0" * 64
    else:
        timed.pop("completed_frames")
    fixture.config["timed_fps_authority"] = _write(path, timed)
    _write(fixture.path, fixture.config)
    with pytest.raises(ValueError, match=message):
        fixture.actual_load(fixture.path)
    assert not fixture.state.events
