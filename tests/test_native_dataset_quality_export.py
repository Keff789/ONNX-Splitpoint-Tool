"""Pure portable-envelope checks with 5000 synthetic records; no inference."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_dataset_quality_consumer import validate_native_dataset_quality_producer
from onnx_splitpoint_tool.native_dataset_quality_export import export_native_quality_request
from onnx_splitpoint_tool.native_split_quality import seal_native_split_quality_binding
from onnx_splitpoint_tool.quality_cache import image_ids_fingerprint
from onnx_splitpoint_tool.quality_service import QualityArtifactIntegrityError, _load_quality_artifact
from test_v269d_trt_central_quality_producer import _detection_quality_contract
from test_v269f_native_split_receipt_validation import _fixture_payload


@pytest.fixture
def capture(tmp_path: Path) -> SimpleNamespace:
    payload, _ = _fixture_payload(tmp_path / "prior")
    prior = seal_native_split_quality_binding(payload)
    records = [{"image_id": f"image-{index:05d}", "candidate": [], "ground_truth": []} for index in range(5000)]
    quality = _detection_quality_contract()
    quality["dataset"].update(image_count=5000, image_ids_sha256=image_ids_fingerprint([row["image_id"] for row in records]),
        ground_truth_sha256=canonical_json_sha256([{"image_id": row["image_id"], "ground_truth": []} for row in records]))
    quality.pop("quality_contract_sha256")
    quality["quality_contract_sha256"] = canonical_json_sha256(quality)
    endpoint = {"endpoint_contract_hash": "e" * 64, "endpoint_contract_complete": True, "stage": "decoded_nms"}
    base = {"schema": "synthetic-normal-completion-contract", "schema_version": 1, "completed_endpoint_contract": endpoint}
    base["contract_sha256"] = canonical_json_sha256(base)
    contract = {"contract_sha256": "a" * 64}
    attestation = {"completion_count": 1, "completed_work_units": 1, "last_result": {"detections": []}}
    completions = [{"image_id": row["image_id"], "image_sha256": "b" * 64,
        "execution_contract": contract, "execution_attestation": attestation, "native_completed_count": 1,
        "prepared_input_snapshot": [{"name": "images", "shape": [1, 3, 8, 8], "dtype": "uint8", "sha256": "c" * 64}],
        "timing_input_binding": {"input_image_sha256": "b" * 64}, "boundary_proof": {"status": "exact_runtime_boundary_verified"}}
        for row in records]
    descriptor = {"path": "/prior/native_fps.json", "sha256": "d" * 64, "size_bytes": 123}
    from onnx_splitpoint_tool.native_dataset_quality_consumer import required_native_dataset_source_paths
    root = Path(__file__).resolve().parents[1]
    sources = [{"path": str(root / relative), "relative_path": relative,
                "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                "size_bytes": (root / relative).stat().st_size}
               for relative in sorted(required_native_dataset_source_paths("hailo8_to_trt"))]
    config = {"identity": {"eval_run_id": "fresh-native-dataset-capture", "model_id": "yolo26s", "case_id": "b038",
        "setup_id": "hailo8_setup", "source_run_id": "hailo8_to_trt", "backend": "hailo8_to_trt",
        "runtime_precision_identity": "uint8_dequant_fp16"}, "quality_contract": quality,
        "quality_contract_sha256": quality["quality_contract_sha256"], "metric_gate_config": {"metric": "map_50_95", "non_inferiority_margin": 0.01},
        "policy_sha256": "9" * 64, "statistics": {"method": "paired_bootstrap", "bootstrap_repetitions": 1000,
            "seed": 20260710, "confidence_level": 0.95, "decision": "lower_one_sided_bound"},
        "prior_artifact_selection": prior, "timed_fps_authority": descriptor, "base_completion_contract": base,
        "source_model_sha256": quality["model"]["sha256"]}
    return SimpleNamespace(config=config, records=records, completions=completions, sources=sources, output=tmp_path / "export")


def _export(capture: SimpleNamespace) -> dict[str, Any]:
    return export_native_quality_request(capture.config, capture.records, capture.completions, capture.sources, capture.output)


def test_exports_standard_normal_request_and_one_bound_sidecar(capture: SimpleNamespace) -> None:
    before = deepcopy(capture.config)
    descriptor = _export(capture)
    request_path = Path(descriptor["path"])
    assert request_path.name == "composed_request.json" and request_path.is_absolute()
    assert descriptor["sha256"] == hashlib.sha256(request_path.read_bytes()).hexdigest()
    assert descriptor["size_bytes"] == request_path.stat().st_size
    request = json.loads(request_path.read_text())
    candidate, _, _ = _load_quality_artifact(request["candidate"], request_path, verify=True)
    producer = request["producer_identity"]
    sidecar, _, _ = _load_quality_artifact(producer["completion_evidence"], request_path, verify=True)
    assert {path.name for path in capture.output.iterdir()} == {"composed_request.json", "composed_candidate.json", "native_completion_evidence.json"}
    assert request["candidate"]["path"] == "composed_candidate.json"
    assert producer["completion_evidence"]["path"] == "native_completion_evidence.json"
    assert request["schema"] == "onnx-splitpoint/central-quality-evaluation-request"
    assert candidate["schema"] == "onnx-splitpoint/task-quality-candidate-input"
    assert sidecar["schema"] == "onnx-splitpoint/native-dataset-quality-completion-evidence"
    assert sidecar["record_count"] == producer["completed_frames"] == request["record_count"] == candidate["record_count"] == 5000
    assert sidecar["additional_nms_applied"] is False
    assert len(sidecar["records"]) == 5000
    assert set(sidecar["records"][0]) == {"image_id", "image_sha256", "execution_contract", "execution_attestation",
                                              "prepared_input_snapshot", "timing_input_binding", "boundary_proof"}
    assert sidecar["records"][0]["prepared_input_snapshot"] == {"inputs": capture.completions[0]["prepared_input_snapshot"]}
    assert candidate["records"] == capture.records
    assert producer == candidate["producer_identity"]
    assert producer["prior_artifact_selection"] == capture.config["prior_artifact_selection"]
    assert producer["preselection"] == capture.config["prior_artifact_selection"]["preselection"]
    assert producer["artifacts"] == capture.config["prior_artifact_selection"]["artifacts"]
    assert len(producer["artifacts"]) == 8
    assert producer["candidate_records_sha256"] == canonical_json_sha256(capture.records)
    validated, sha = validate_native_dataset_quality_producer(producer, role="export test", task="detection")
    assert validated == producer and sha == producer["producer_identity_sha256"]
    assert request["producer_identity_sha256"] == candidate["producer_identity_sha256"] == sha
    assert request["reference"] == {"source": "management_cpu_reference", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "required": True, "expected_image_ids": [row["image_id"] for row in capture.records],
        "expected_image_ids_sha256": image_ids_fingerprint([row["image_id"] for row in capture.records]), "record_count": 5000,
        "quality_contract_sha256": capture.config["quality_contract_sha256"]}
    assert request["execution_location"] == "management_node" and request["pairing_key"] == "image_id"
    assert request["metric_gate_config"] == capture.config["metric_gate_config"]
    assert request["statistics"] == capture.config["statistics"]
    assert request["candidate_execution_completion_contract"] == capture.config["base_completion_contract"]
    assert request["endpoint_contract"] == capture.config["base_completion_contract"]["completed_endpoint_contract"]
    for envelope in (request, candidate):
        assert envelope["producer_provenance_required"] is True
        assert envelope["performance_claims_emitted"] is False
        assert envelope["producer_kind"] == "native_dataset_completed_quality"
        assert envelope["quality_contract"] == capture.config["quality_contract"]
        for field in ("quality_contract_sha256", "preprocessing_contract_sha256", "decoder_contract_sha256",
                      "nms_contract_sha256", "quality_record_endpoint_contract_sha256"):
            assert envelope[field] == producer[field]
        assert "native_split_quality_binding" not in envelope
    assert capture.config == before


@pytest.mark.parametrize("part", ["records", "completions"])
def test_rejects_non_5000_counts_before_writing(capture: SimpleNamespace, part: str) -> None:
    setattr(capture, part, getattr(capture, part)[:-1])
    with pytest.raises(ValueError, match="exactly_5000"):
        _export(capture)
    assert not capture.output.exists()


def test_duplicate_ids_are_rejected_before_writing(capture: SimpleNamespace) -> None:
    capture.records[-1]["image_id"] = capture.records[0]["image_id"]
    capture.completions[-1]["image_id"] = capture.records[0]["image_id"]
    with pytest.raises(ValueError, match="duplicate_image_ids"):
        _export(capture)
    assert not capture.output.exists()


def test_completion_ids_must_align_in_typed_domain(capture: SimpleNamespace) -> None:
    capture.completions[1]["image_id"] = capture.completions[0]["image_id"]
    with pytest.raises(ValueError, match="completion_image_id_alignment_mismatch"):
        _export(capture)
    assert not capture.output.exists()


@pytest.mark.parametrize("count", [0, 2, True])
def test_exactly_one_observed_native_completion_is_required(capture: SimpleNamespace, count: Any) -> None:
    capture.completions[0]["native_completed_count"] = count
    with pytest.raises(ValueError, match="exactly_one_native_completion_per_image_required"):
        _export(capture)


def test_cannot_change_completed_detections(capture: SimpleNamespace) -> None:
    capture.records[0]["candidate"] = [{"score": 0.9}]
    with pytest.raises(ValueError, match="candidate_differs_from_completed_detections"):
        _export(capture)


@pytest.mark.parametrize("field,value", [("fps", 100), ("fps_makespan", 100), ("performance_claims_emitted", True)])
def test_performance_claims_and_rates_are_rejected(capture: SimpleNamespace, field: str, value: Any) -> None:
    capture.config[field] = value
    with pytest.raises(ValueError, match="performance_.*forbidden"):
        _export(capture)


@pytest.mark.parametrize("field,value", [("bootstrap_repetitions", 999), ("seed", 20260711)])
def test_scientific_statistics_are_not_changed(capture: SimpleNamespace, field: str, value: int) -> None:
    capture.config["statistics"][field] = value
    with pytest.raises(ValueError, match="B1000_seed20260710_required"):
        _export(capture)


def test_old_quality_run_cannot_be_relabelled_as_fresh(capture: SimpleNamespace) -> None:
    capture.config["identity"]["eval_run_id"] = capture.config["prior_artifact_selection"]["eval_run_id"]
    with pytest.raises(ValueError, match="fresh_evaluation_id_required"):
        _export(capture)


def test_normal_sidecar_descriptor_rejects_modified_bytes(capture: SimpleNamespace) -> None:
    result = _export(capture)
    request_path = Path(result["path"])
    request = json.loads(request_path.read_text())
    descriptor = request["producer_identity"]["completion_evidence"]
    path = request_path.parent / descriptor["path"]
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b'"record_count":5000', b'"record_count":4000', 1))
    with pytest.raises(QualityArtifactIntegrityError, match="SHA-256 mismatch"):
        _load_quality_artifact(descriptor, request_path, verify=True)


def test_fresh_atomic_exports_never_overwrite_existing_files(capture: SimpleNamespace) -> None:
    _export(capture)
    before = {path.name: path.read_bytes() for path in capture.output.iterdir()}
    with pytest.raises(ValueError, match="fresh_output_files_required"):
        _export(capture)
    assert before == {path.name: path.read_bytes() for path in capture.output.iterdir()}
    assert len(before) == 3
