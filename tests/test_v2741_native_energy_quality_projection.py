from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256,
    verify_native_split_part2_input_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v2741_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_test_module(name: str):
    path = ROOT / "tests" / name
    spec = importlib.util.spec_from_file_location(
        f"v2741_fixture_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reseal(contract: dict) -> dict:
    contract = copy.deepcopy(contract)
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def _technical_split_contract() -> dict:
    path = (
        ROOT / "tests" / "fixtures" / "v2725_hailo8_resume"
        / (
            "hailo8_to_trt__resnet50__b052__"
            "orin_nx_hailo8_01__8eb40a232207.command_contract.json"
        )
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    verified, status = verify_native_split_part2_input_contract(contract)
    assert verified is not None, status
    return contract


@pytest.mark.parametrize(
    ("producer", "drop_metadata"),
    (("hailo10h_to_trt", False), ("deepx_to_trt", True)),
)
def test_energy_part2_crosslinks_repair_the_two_real_contract_shapes(
    producer: str, drop_metadata: bool,
) -> None:
    contract = _technical_split_contract()
    contract.pop("engine", None)
    contract.pop("engine_sha256", None)
    if drop_metadata:
        contract["boundary_contract"].pop("metadata_path", None)
        contract["boundary_contract"].pop("metadata_sha256", None)
    contract = _reseal(contract)

    verified, status = verify_native_split_part2_input_contract(contract)
    assert verified is None
    assert status == "native_energy_part2_command_artifact_duplicate_mismatch"

    artifacts = contract["artifacts"]
    contract["engine"] = artifacts["engine"]["path"]
    contract["engine_sha256"] = artifacts["engine"]["sha256"]
    contract["boundary_contract"]["metadata_path"] = (
        artifacts["native_trt_meta"]["path"]
    )
    contract["boundary_contract"]["metadata_sha256"] = (
        artifacts["native_trt_meta"]["sha256"]
    )
    contract = _reseal(contract)

    verified, status = verify_native_split_part2_input_contract(contract)
    assert verified is not None, f"{producer}: {status}"
    assert status == "verified_single_static_part2_input_technical_proof"
    assert len(verified["inputs"]) == 1


@pytest.mark.parametrize(
    ("container", "field"),
    (
        ("contract", "engine"),
        ("contract", "engine_sha256"),
        ("boundary", "metadata_path"),
        ("boundary", "metadata_sha256"),
    ),
)
def test_energy_part2_crosslink_tamper_remains_fail_closed(
    container: str, field: str,
) -> None:
    contract = _technical_split_contract()
    target = contract if container == "contract" else contract["boundary_contract"]
    target[field] = "f" * 64 if field.endswith("sha256") else "/tampered/path"
    contract = _reseal(contract)
    verified, status = verify_native_split_part2_input_contract(contract)
    assert verified is None
    assert status == "native_energy_part2_command_artifact_duplicate_mismatch"


@pytest.mark.parametrize(
    "name",
    (
        "native_hailo10_trt_e2e_from_benchmarkset.py",
        "native_deepx_trt_e2e_from_benchmarkset.py",
        "native_producer_validate_visualize.py",
    ),
)
def test_changed_remote_runner_mirrors_are_byte_identical(name: str) -> None:
    assert (ROOT / "scripts" / name).read_bytes() == (
        ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name
    ).read_bytes()


def test_energy_producers_project_crosslinks_after_quality_override() -> None:
    hailo = (ROOT / "scripts/native_hailo10_trt_e2e_from_benchmarkset.py").read_text(
        encoding="utf-8",
    )
    deepx = (ROOT / "scripts/native_deepx_trt_e2e_from_benchmarkset.py").read_text(
        encoding="utf-8",
    )
    for source in (hailo, deepx):
        assert "engine_artifact = contract_payload['artifacts'].get('engine')" in source
        assert "contract_payload['engine'] = engine_path_bound" in source
        assert "contract_payload['engine_sha256'] = engine_sha_bound" in source
        assert source.index("contract_payload['engine'] = engine_path_bound") < source.index(
            "seal_native_command_contract(contract_payload)",
        )
    assert "metadata_artifact = contract_payload['artifacts'].get('native_trt_meta')" in deepx
    assert "contract_payload['boundary_contract']['metadata_path']" in deepx
    assert "contract_payload['boundary_contract']['metadata_sha256']" in deepx


def test_native_trt_detection_joins_physical_endpoint_and_verifies_completed_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    fixture = _load_test_module("test_v269d_quality_first_native_final_chain.py")
    row, result, producer, policy = fixture._quality_first_fixture()
    row["task"] = "detection"
    result["task"] = "detection"
    result["request_identity"]["task"] = "detection"
    decoder_sha = "6" * 64
    nms_sha = "7" * 64
    for target in (result, result["request_identity"]):
        target["decoder_contract_sha256"] = decoder_sha
        target["nms_contract_sha256"] = nms_sha

    physical_hash = row["endpoint_contract_hash"]
    comparison = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "model_id": row["model"],
        "class_aware": True,
        "score_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "input_hw": [640, 640],
        "canonical_completion_policy_id": (
            "decoded_nms_xyxy_original_classaware_postfilter_v2"
        ),
        "nms_semantics_id": "class_aware_nms_xyxy_v1",
        "endpoint_contract_hash": "8" * 64,
        "output_endpoint_id": "completed:decoded_nms",
    }
    row.update({
        "completed_task_completion_mode": "detection_completion_execution_v1",
        "completed_task_comparison_endpoint_contract": copy.deepcopy(comparison),
        "completed_task_comparison_endpoint_contract_hash": comparison[
            "endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": comparison[
            "output_endpoint_id"
        ],
    })

    producer = copy.deepcopy(producer)
    producer["task"] = "detection"
    producer["model_id"] = row["model"]
    producer["endpoint_contract_hash"] = physical_hash
    producer["endpoint"] = {
        "identity": {"stage": "raw_head"}, "sha256": physical_hash,
    }
    preprocessing_sha = "5" * 64
    quality_sha = "4" * 64
    record_sha = "3" * 64
    producer["quality_contract"] = {
        "task": "detection",
        "canonical_record_endpoint": "decoded_xyxy_score_class_detections",
        "source_endpoint_is_raw": True,
        "model": {"sha256": producer["source_onnx"]["sha256"]},
        "dataset": copy.deepcopy(producer["dataset"]),
        "preprocessing": {
            "identity": {"target_hw": [640, 640]},
            "sha256": preprocessing_sha,
        },
        "decoder": {
            "identity": {
                "canonical_record_endpoint": (
                    "decoded_xyxy_score_class_detections"
                ),
                "source_output_format": "multiscale_head",
                "source_endpoint_semantics": "raw_multiscale_head",
                "source_endpoint_has_integrated_nms": False,
                "confidence_threshold": 0.25,
            },
            "sha256": decoder_sha,
        },
        "nms": {
            "identity": {"iou_threshold": 0.45, "max_detections": 300},
            "sha256": nms_sha,
        },
        "quality_contract_sha256": quality_sha,
    }
    producer.update({
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "quality_record_endpoint_contract_sha256": record_sha,
        "quality_record_endpoint": {
            "identity": {
                "canonical_record_endpoint": (
                    "decoded_xyxy_score_class_detections"
                ),
                "decoder_contract_sha256": decoder_sha,
                "nms_contract_sha256": nms_sha,
            },
            "sha256": record_sha,
        },
    })
    for target in (result, result["request_identity"]):
        target["quality_contract_sha256"] = quality_sha
        target["preprocessing_contract_sha256"] = preprocessing_sha
        target["decoder_contract_sha256"] = decoder_sha
        target["nms_contract_sha256"] = nms_sha
        target["quality_record_endpoint_contract_sha256"] = record_sha
    result["quality_contract"] = copy.deepcopy(producer["quality_contract"])
    producer_sha = "2" * 64
    monkeypatch.setattr(
        validator,
        "_native_trt_quality_first_binding",
        lambda _row, *, task: (copy.deepcopy(producer), producer_sha, []),
    )
    monkeypatch.setattr(
        validator,
        "_central_trt_quality_producer",
        lambda _result, *, task: (copy.deepcopy(producer), producer_sha, []),
    )

    def verify_comparison(value):
        if dict(value or {}) != comparison:
            raise ValueError("comparison mismatch")
        return copy.deepcopy(comparison)

    def verify_completed(value):
        if (
            value.get("completed_task_comparison_endpoint_contract_hash")
            != comparison["endpoint_contract_hash"]
        ):
            raise ValueError("completed projection mismatch")
        return (
            "detection_completion_execution_v1",
            {"source": "physical"},
            copy.deepcopy(comparison),
        )

    monkeypatch.setattr(
        validator,
        "_verify_completed_detection_comparison_endpoint_contract",
        verify_comparison,
    )
    monkeypatch.setattr(validator, "_verified_completed_v2_contract", verify_completed)

    pristine_row = copy.deepcopy(row)
    validator._bind_central_quality_evidence(row, [result], policy)
    assert row["central_quality_evidence_verified"] is True
    assert row["central_quality_binding_status"] == "exact_identity_match"
    assert row["endpoint_contract_hash"] == physical_hash
    quality_request = row["task_quality_gate"]["quality_input_request"]
    assert quality_request["endpoint_contract_hash"] == comparison[
        "endpoint_contract_hash"
    ]
    assert quality_request["source_endpoint_contract_hash"] == physical_hash
    assert quality_request["quality_join_endpoint"] == "completed_task_decoded_nms"
    assert row["completed_task_quality_projection_verified"] is True
    assert row["completed_task_quality_projection_contract_hash"] == comparison[
        "endpoint_contract_hash"
    ]

    # Exercise the second physical Detection form through the same complete
    # bind path without adding a separate collected test.  Direct BN6 output
    # uses the dedicated duplicate fields, but must project to the identical
    # canonical Completed-v2 endpoint.
    bn6_producer = copy.deepcopy(producer)
    bn6_quality = bn6_producer["quality_contract"]
    bn6_decoder = bn6_quality["decoder"]["identity"]
    bn6_decoder.update({
        "source_output_format": "bn6_detections",
        "source_endpoint_semantics": "bn6_detections",
    })
    bn6_decoder.pop("source_endpoint_has_integrated_nms", None)
    bn6_nms = bn6_quality["nms"]["identity"]
    bn6_nms.update({
        "detr_or_bn6_confidence_threshold": 0.25,
        "detr_or_bn6_iou_threshold": 0.45,
        "detr_or_bn6_max_detections": 300,
    })
    bn6_quality["source_endpoint_is_raw"] = False
    bn6_producer["endpoint"]["identity"]["stage"] = "decoded_nms"
    bn6_decoder_sha = validator._canonical_json_sha256(bn6_decoder)
    bn6_nms_sha = validator._canonical_json_sha256(bn6_nms)
    bn6_quality["decoder"]["sha256"] = bn6_decoder_sha
    bn6_quality["nms"]["sha256"] = bn6_nms_sha
    bn6_producer["decoder_contract_sha256"] = bn6_decoder_sha
    bn6_producer["nms_contract_sha256"] = bn6_nms_sha
    bn6_record_identity = bn6_producer["quality_record_endpoint"]["identity"]
    bn6_record_identity["decoder_contract_sha256"] = bn6_decoder_sha
    bn6_record_identity["nms_contract_sha256"] = bn6_nms_sha
    bn6_record_sha = validator._canonical_json_sha256(bn6_record_identity)
    bn6_producer["quality_record_endpoint"]["sha256"] = bn6_record_sha
    bn6_producer["quality_record_endpoint_contract_sha256"] = bn6_record_sha
    bn6_quality.pop("quality_contract_sha256", None)
    bn6_quality_sha = validator._canonical_json_sha256(bn6_quality)
    bn6_quality["quality_contract_sha256"] = bn6_quality_sha
    bn6_producer["quality_contract_sha256"] = bn6_quality_sha
    bn6_result = copy.deepcopy(result)
    bn6_result["quality_contract"] = copy.deepcopy(bn6_quality)
    for target in (bn6_result, bn6_result["request_identity"]):
        target["quality_contract_sha256"] = bn6_quality_sha
        target["decoder_contract_sha256"] = bn6_decoder_sha
        target["nms_contract_sha256"] = bn6_nms_sha
        target["quality_record_endpoint_contract_sha256"] = bn6_record_sha
    monkeypatch.setattr(
        validator,
        "_native_trt_quality_first_binding",
        lambda _row, *, task: (copy.deepcopy(bn6_producer), producer_sha, []),
    )
    monkeypatch.setattr(
        validator,
        "_central_trt_quality_producer",
        lambda _result, *, task: (copy.deepcopy(bn6_producer), producer_sha, []),
    )
    bn6_row = copy.deepcopy(pristine_row)
    validator._bind_central_quality_evidence(bn6_row, [bn6_result], policy)
    assert bn6_row["central_quality_evidence_verified"] is True
    assert bn6_row["central_quality_binding_status"] == "exact_identity_match"
    assert bn6_row["endpoint_contract_hash"] == physical_hash
    assert bn6_row["task_quality_gate"]["quality_input_request"][
        "endpoint_contract_hash"
    ] == comparison["endpoint_contract_hash"]

    monkeypatch.setattr(
        validator,
        "_native_trt_quality_first_binding",
        lambda _row, *, task: (copy.deepcopy(producer), producer_sha, []),
    )
    monkeypatch.setattr(
        validator,
        "_central_trt_quality_producer",
        lambda _result, *, task: (copy.deepcopy(producer), producer_sha, []),
    )

    tampered = copy.deepcopy(row)
    tampered["completed_task_comparison_endpoint_contract_hash"] = "9" * 64
    validator._bind_central_quality_evidence(tampered, [result], policy)
    assert tampered["central_quality_evidence_verified"] is False
    assert tampered["central_quality_binding_status"] == (
        "native_completed_task_projection_invalid"
    )

    for field, bad_value in (("max_detections", 200), ("iou_threshold", 0.50)):
        bad_producer = copy.deepcopy(producer)
        bad_producer["quality_contract"]["nms"]["identity"][field] = bad_value
        monkeypatch.setattr(
            validator,
            "_native_trt_quality_first_binding",
            lambda _row, *, task, value=bad_producer: (
                copy.deepcopy(value), producer_sha, [],
            ),
        )
        rejected = copy.deepcopy(pristine_row)
        validator._bind_central_quality_evidence(rejected, [result], policy)
        assert rejected["central_quality_evidence_verified"] is False
        assert rejected["central_quality_binding_status"] == (
            "native_trt_completed_quality_semantics_mismatch"
        )

    monkeypatch.setattr(
        validator,
        "_native_trt_quality_first_binding",
        lambda _row, *, task: (copy.deepcopy(producer), producer_sha, []),
    )
    partial_projection = copy.deepcopy(result)
    partial_projection["completed_task_endpoint_contract_hash"] = (
        comparison["endpoint_contract_hash"]
    )
    rejected = copy.deepcopy(pristine_row)
    validator._bind_central_quality_evidence(
        rejected, [partial_projection], policy,
    )
    assert rejected["central_quality_evidence_verified"] is False
    assert rejected["central_quality_binding_status"] == "no_exact_identity_match"


def test_declared_completed_v2_skips_legacy_raw_head_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    input_manifest = tmp_path / "native_input.json"
    output_manifest = tmp_path / "native_output.json"
    full_onnx = tmp_path / "full.onnx"
    for path in (input_manifest, output_manifest, full_onnx):
        path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        validator,
        "_find_self_reference_input_manifest",
        lambda *args, **kwargs: (
            input_manifest, "native_split_input_manifest", "found",
        ),
    )

    def legacy_probe_forbidden(*args, **kwargs):
        raise AssertionError("legacy raw-head probe must not run")

    monkeypatch.setattr(
        validator, "_precomputed_full_self_reference_detection",
        legacy_probe_forbidden,
    )
    monkeypatch.setattr(
        validator, "_auto_run_yolo_full_self_reference_probe",
        legacy_probe_forbidden,
    )
    monkeypatch.setattr(
        validator, "_find_full_onnx_for_native_manifest",
        lambda *args, **kwargs: full_onnx,
    )
    monkeypatch.setattr(
        validator, "_native_input_dump_feed",
        lambda *args, **kwargs: np.zeros((1, 3, 4, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        validator, "load_dump",
        lambda *args, **kwargs: ({"output0": np.zeros((1, 1, 6))}, {}),
    )
    detection = {
        "class_id": 1, "score": 0.9,
        "x1": 1.0, "y1": 1.0, "x2": 3.0, "y2": 3.0,
    }
    monkeypatch.setattr(
        validator,
        "_completed_v2_self_reference_detection",
        lambda *args, **kwargs: {
            "available": True,
            "completed_v2_verified": True,
            "full_mode": "completed_v2:full_bn6_normalized",
            "native_mode": "completed_v2:native_decoded_nms",
            "expected_contract_family": "decoded_nms",
            "expected_contract_source": (
                "verified_completed_task_comparison_endpoint_v2"
            ),
            "reference_detections": [copy.deepcopy(detection)],
            "native_detections": [copy.deepcopy(detection)],
        },
    )

    class FakeSession:
        def __init__(self, *args, **kwargs):
            self._input = SimpleNamespace(
                name="images", shape=[1, 3, 4, 4], type="tensor(float)",
            )

        def get_inputs(self):
            return [self._input]

        def get_outputs(self):
            return [SimpleNamespace(name="output0")]

        def run(self, *args, **kwargs):
            return [np.zeros((1, 1, 6), dtype=np.float32)]

    monkeypatch.setitem(
        sys.modules, "onnxruntime",
        SimpleNamespace(InferenceSession=FakeSession),
    )
    result = validator._full_onnx_self_reference_detection(
        output_manifest,
        tmp_path,
        endpoint_evidence={
            "completed_task_completion_mode": "detection_completion_execution_v1",
        },
    )
    assert result["completed_v2_verified"] is True
    assert result["semantic_available"] is True
    assert result["semantic_ok"] is True
    assert result["expected_contract_family"] == "decoded_nms"
