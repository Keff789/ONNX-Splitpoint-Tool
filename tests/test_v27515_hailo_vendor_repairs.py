from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import canonical_json_bytes
from onnx_splitpoint_tool.native_output_endpoint import _endpoint_hash, _tensor_signature
from onnx_splitpoint_tool.runners.backends.hailo_backend import (
    _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
    _adapt_attested_hailo_output_tensor,
    _build_attested_hailo_output_map,
    _order_attested_output_mapping,
    _reconcile_attested_source_output_metadata,
    _validated_attested_source_output_shapes,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from scripts import native_producer_validate_visualize as validator
from scripts import smoke_hailo10_hef_runner as hailo_runner


ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_CLAIM_FIELDS = (
    "source_request_sha256",
    "model_sha256",
    "validation_dataset_sha256",
    "validation_dataset_image_ids_sha256",
    "validation_dataset_ground_truth_sha256",
    "task_quality_policy_sha256",
    "quality_contract_sha256",
    "preprocessing_contract_sha256",
    "decoder_contract_sha256",
    "nms_contract_sha256",
    "quality_record_endpoint_contract_sha256",
    "central_quality_result_sha256",
    "prepared_input_evidence_sha256",
)


def _vendor_quality_fixture(
    backend: str = "native_full_hailo8",
) -> tuple[dict, dict, dict]:
    precision = {"schema": "runtime-precision", "mode": "int8"}
    preprocessing = {"schema": "preprocessing", "mode": "letterbox"}
    numeric = {"schema": "runtime-numeric-input", "layout": "HWC"}
    components = {field: _digest(field) for field in _CLAIM_FIELDS}
    binding = {
        **components,
        "binding_sha256": _digest("binding"),
        "source_run_id": "run-1",
        "source_case_id": "full",
        "runtime_precision_identity": precision,
        "preprocessing_contract_sha256": components[
            "preprocessing_contract_sha256"
        ],
        "preprocessing_contract": {
            "identity": preprocessing,
            "sha256": components["preprocessing_contract_sha256"],
        },
    }
    runtime_binding = {
        "identity": preprocessing,
        "sha256": components["preprocessing_contract_sha256"],
        "runtime_numeric_input_identity": numeric,
        "runtime_numeric_input_sha256": _digest("numeric"),
    }
    full_contract = {
        "quality_request_binding": copy.deepcopy(binding),
        "quality_request_binding_sha256": binding["binding_sha256"],
        "quality_request_binding_set_sha256": _digest("set"),
        "runtime_preprocessing_binding": runtime_binding,
    }
    row = {
        "backend": backend,
        "quality_request_binding": binding,
        "quality_request_binding_status": "verified_exact",
        "quality_request_binding_sha256": binding["binding_sha256"],
        "quality_request_binding_set_sha256": _digest("set"),
        "quality_source_run_id": "run-1",
        "quality_source_case_id": "full",
        "preprocessing_contract": preprocessing,
        "runtime_precision_identity": precision,
        "runtime_preprocessing_identity": preprocessing,
        "runtime_preprocessing_sha256": components[
            "preprocessing_contract_sha256"
        ],
        "runtime_numeric_input_identity": numeric,
        "runtime_numeric_input_sha256": _digest("numeric"),
        "full_command_contract": full_contract,
        **components,
    }
    target = {
        "backend": backend,
        # This validator-derived mirror must agree; it is not overwritten.
        "runtime_precision_identity": copy.deepcopy(precision),
    }
    return target, row, copy.deepcopy(row)


@pytest.mark.parametrize(
    "backend",
    ["native_full_hailo8", "native_full_hailo10h", "native_full_deepx"],
)
def test_vendor_full_quality_binding_is_projected_without_loss(backend: str) -> None:
    target, row, native_json = _vendor_quality_fixture(backend)
    validator._merge_vendor_full_quality_evidence(target, row, native_json)

    assert target["quality_request_binding"] == row["quality_request_binding"]
    assert target["quality_request_binding"] is not row["quality_request_binding"]
    assert target["quality_request_binding_status"] == "verified_exact"
    assert target["quality_request_binding_sha256"] == _digest("binding")
    assert target["quality_request_binding_set_sha256"] == _digest("set")
    for field in _CLAIM_FIELDS:
        assert target[field] == row[field]
    for field in (
        "runtime_precision_identity",
        "runtime_preprocessing_identity",
        "runtime_preprocessing_sha256",
        "runtime_numeric_input_identity",
        "runtime_numeric_input_sha256",
    ):
        assert target[field] == row[field]
    assert target.get("vendor_full_quality_provenance_conflict") is not True


@pytest.mark.parametrize(
    ("mutate", "expected_field"),
    [
        (
            lambda target, row, native: target.update(
                runtime_precision_identity={"schema": "runtime-precision", "mode": "fp16"}
            ),
            "runtime_precision_identity",
        ),
        (
            lambda target, row, native: row["full_command_contract"].update(
                quality_request_binding_set_sha256=_digest("other-set")
            ),
            "quality_request_binding_set_sha256",
        ),
        (
            lambda target, row, native: target.update(
                full_command_contract={
                    **copy.deepcopy(row["full_command_contract"]),
                    "quality_request_binding_set_sha256": _digest(
                        "target-other-set"
                    ),
                }
            ),
            "quality_request_binding_set_sha256",
        ),
        (
            lambda target, row, native: native["quality_request_binding"].update(
                decoder_contract_sha256=_digest("other-decoder")
            ),
            "decoder_contract_sha256",
        ),
        (
            lambda target, row, native: native.update(
                runtime_numeric_input_identity={
                    "schema": "runtime-numeric-input", "layout": "NCHW"
                }
            ),
            "runtime_numeric_input_identity",
        ),
    ],
)
def test_vendor_full_quality_duplicate_conflicts_fail_closed(
    mutate, expected_field: str,
) -> None:
    target, row, native_json = _vendor_quality_fixture()
    mutate(target, row, native_json)
    validator._merge_vendor_full_quality_evidence(target, row, native_json)

    assert target["vendor_full_quality_provenance_conflict"] is True
    assert expected_field in target["vendor_full_quality_provenance_conflict_fields"]
    assert target["quality_request_binding_status"] == "conflicting_duplicate_evidence"
    assert (
        f"vendor_full_quality_provenance_conflict:{expected_field}"
        in target["quality_request_binding_errors"]
    )


def test_projected_vendor_binding_passes_unchanged_strict_consumer() -> None:
    """Replay the production projection immediately into the existing join."""

    policy = AccuracyGatePolicy()
    endpoint_sha = _digest("classification-endpoint")
    precision = "hailo8_int8_full"
    preprocessing = {"schema": "preprocessing", "mode": "resize"}
    field_values = {
        "source_request_sha256": _digest("central-request"),
        "model_sha256": _digest("resnet50"),
        "validation_dataset_sha256": _digest("imagenet"),
        "validation_dataset_image_ids_sha256": _digest("image-ids"),
        "validation_dataset_ground_truth_sha256": _digest("ground-truth"),
        "task_quality_policy_sha256": policy.sha256(),
        "quality_contract_sha256": _digest("quality-contract"),
        "preprocessing_contract_sha256": _digest("preprocessing-contract"),
        "quality_record_endpoint_contract_sha256": _digest("quality-endpoint"),
        "prepared_input_evidence_sha256": _digest("prepared-input"),
    }
    identity = {
        "schema_version": 4,
        "identity_valid": True,
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "full",
        "source_run_id": "hailo8",
        "setup_id": "orin_nx_hailo8_01",
        "variant": "full",
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": precision,
        **field_values,
    }
    result = {
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "full",
        "source_run_id": "hailo8",
        "source_setup_id": "orin_nx_hailo8_01",
        "variant": "full",
        "status": "completed",
        "technical_status": "completed",
        "decision": "pass",
        "policy_sha256": policy.sha256(),
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": precision,
        "request_identity": identity,
        "primary": {
            "metric": "top1_accuracy",
            "delta": 0.0,
            "ci_low": 0.0,
            "margin": 0.01,
        },
        **field_values,
    }
    central_sha = validator._canonical_json_sha256(result)
    binding = {
        "schema": "onnx-splitpoint/native-full-quality-request-binding",
        "schema_version": 1,
        "eval_run_id": "eval-1",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "backend": "native_full_hailo8",
        "model_id": "resnet50",
        "source_run_id": "hailo8",
        "source_case_id": "full",
        "task": "classification",
        "variant": "full",
        "runtime_precision_identity": precision,
        "endpoint_contract_hash": endpoint_sha,
        "central_quality_result_sha256": central_sha,
        "preprocessing_contract": {
            "identity": preprocessing,
            "sha256": field_values["preprocessing_contract_sha256"],
        },
        **field_values,
    }
    binding["binding_sha256"] = validator._canonical_json_sha256(binding)
    binding_set_sha = _digest("binding-set")
    full_contract = {
        "input_case": "full",
        "quality_request_binding": copy.deepcopy(binding),
        "quality_request_binding_sha256": binding["binding_sha256"],
        "quality_request_binding_set_sha256": binding_set_sha,
        "runtime_preprocessing_binding": {
            "identity": copy.deepcopy(preprocessing),
            "sha256": field_values["preprocessing_contract_sha256"],
        },
    }
    producer_row = {
        "backend": "native_full_hailo8",
        "quality_request_binding": copy.deepcopy(binding),
        "quality_request_binding_status": "verified_exact",
        "quality_request_binding_sha256": binding["binding_sha256"],
        "quality_request_binding_set_sha256": binding_set_sha,
        "preprocessing_contract": copy.deepcopy(preprocessing),
        "runtime_preprocessing_identity": copy.deepcopy(preprocessing),
        "runtime_preprocessing_sha256": field_values[
            "preprocessing_contract_sha256"
        ],
        "full_command_contract": copy.deepcopy(full_contract),
        **field_values,
    }
    validation_row = {
        "backend": "native_full_hailo8",
        "model": "resnet50",
        "case": "full",
        "setup_id": "orin_nx_hailo8_01",
        "task": "classification",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": precision,
        "full_command_contract": copy.deepcopy(full_contract),
    }

    validator._merge_vendor_full_quality_evidence(
        validation_row, producer_row, copy.deepcopy(producer_row)
    )
    validator._bind_central_quality_evidence(
        validation_row, [result], policy
    )

    assert validation_row["central_quality_evidence_verified"] is True
    assert validation_row["central_quality_binding_status"] == "exact_identity_match"
    assert validation_row["quality_first_binding_status"] == (
        "central_vendor_full_exact_request_binding_match"
    )
    assert validation_row["quality_request_binding"] == binding
    assert validation_row["runtime_quality_gate_policy_sha256"] == policy.sha256()


def _source_output_contract(source_path: Path) -> dict:
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    compiler_sha = _digest("compiler-onnx")
    body = {
        "schema": "onnx-splitpoint/hailo-source-raw-head-attestation",
        "schema_version": 1,
        "raw_endpoint_origin": "source_onnx_graph_outputs",
        "source_onnx_sha256": source_sha,
        "compiler_onnx_sha256": compiler_sha,
        "outputs": [
            {"name": name, "element_type": 1, "rank": 5, "shape": list(shape)}
            for name, shape in _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES.items()
        ],
    }
    attestation = copy.deepcopy(body)
    attestation["attestation_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return {
        "contract_resolution_status": "attested",
        "authoritative_output_contract": True,
        "backend": "hailo8",
        "model_id": "yolov7_paper",
        "variant": "full",
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_mode": "raw_head",
        "compiled_artifact_raw_head": True,
        "host_tail_required": True,
        "postprocessing_required": True,
        "requires_external_postprocess": True,
        "source_onnx_multiscale_raw_head": True,
        "raw_endpoint_origin": "source_onnx_graph_outputs",
        "full_end_node_names": [],
        "source_onnx_sha256": source_sha,
        "compiler_onnx_sha256": compiler_sha,
        "source_onnx_raw_head_attestation": attestation,
        "source_onnx_raw_head_attestation_sha256": attestation[
            "attestation_sha256"
        ],
    }


def _verify_source_contract(contract: dict, source_path: Path) -> dict[str, list[int]]:
    return hailo_runner._verified_hailo8_yolov7_source_output_shapes(
        contract,
        source_onnx=source_path,
        hw_arch="hailo8",
        backend_label="native_full_hailo8",
        model="yolov7_paper",
        task="detection",
    )


def test_hailo8_yolov7_source_attestation_projects_exact_shapes(tmp_path: Path) -> None:
    source = tmp_path / "yolov7.onnx"
    source.write_bytes(b"source-onnx-bytes")
    contract = _source_output_contract(source)

    assert _verify_source_contract(contract, source) == {
        name: list(shape)
        for name, shape in _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES.items()
    }
    # No exception is exposed to adjacent backends/models.
    assert hailo_runner._verified_hailo8_yolov7_source_output_shapes(
        None,
        source_onnx="",
        hw_arch="hailo10h",
        backend_label="native_full_hailo10h",
        model="yolov7_paper",
        task="detection",
    ) == {}


def test_hailo8_yolov7_suite_alias_survives_real_loader_checker_chain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "yolov7.onnx"
    source.write_bytes(b"source-onnx-bytes")
    hef = tmp_path / "yolov7.hef"
    hef.write_bytes(b"hef-bytes")

    recorded = _source_output_contract(source)
    recorded.update({
        "contract_status": "recorded",
        # This is the historical suite spelling.  The authoritative loader
        # canonicalizes it before the smoke checker consumes the declaration.
        "endpoint_mode": "raw_detection_head",
        "output_format": "raw_detection_tensors",
        "recorded_artifact_path": hef.name,
        "recorded_artifact_sha256": hashlib.sha256(hef.read_bytes()).hexdigest(),
        "recorded_artifact_size_bytes": hef.stat().st_size,
    })
    declaration_path = tmp_path / "output_contracts.json"
    declaration_path.write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "yolov7_paper",
        "task": "detection",
        "contracts": [recorded],
    }), encoding="utf-8")

    resolved = hailo_runner._resolve_declared_output_contract(
        declaration_path,
        hw_arch="hailo8",
        model="yolov7_paper",
        task="detection",
    )

    assert resolved["contract_resolution_status"] == "attested"
    assert resolved["authoritative_output_contract"] is True
    assert resolved["stage"] == "raw_head"
    assert resolved["contract_family"] == "raw_head"
    assert resolved["endpoint_mode"] == "raw_head"
    assert _verify_source_contract(resolved, source) == {
        name: list(shape)
        for name, shape in _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES.items()
    }


@pytest.mark.parametrize(
    "tamper",
    [
        lambda contract: contract.update(authoritative_output_contract=False),
        lambda contract: contract.update(
            source_onnx_raw_head_attestation_sha256=_digest("wrong-attestation")
        ),
        lambda contract: contract.update(compiler_onnx_sha256=_digest("other-compiler")),
        lambda contract: contract.update(full_end_node_names=["conv71"]),
    ],
)
def test_hailo8_yolov7_source_attestation_rejects_contract_tampering(
    tmp_path: Path, tamper,
) -> None:
    source = tmp_path / "yolov7.onnx"
    source.write_bytes(b"source-onnx-bytes")
    contract = _source_output_contract(source)
    tamper(contract)
    with pytest.raises(RuntimeError):
        _verify_source_contract(contract, source)


@pytest.mark.parametrize(
    ("present", "value"),
    [
        pytest.param(False, None, id="missing"),
        pytest.param(True, None, id="null"),
        pytest.param(True, "", id="string"),
        pytest.param(True, {}, id="object"),
        pytest.param(True, 0, id="number"),
    ],
)
def test_hailo8_yolov7_source_attestation_requires_explicit_empty_end_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: bool, value: object,
) -> None:
    source = tmp_path / "yolov7.onnx"
    source.write_bytes(b"source-onnx-bytes")
    contract = _source_output_contract(source)
    if present:
        contract["full_end_node_names"] = value
    else:
        contract.pop("full_end_node_names")

    with pytest.raises(RuntimeError, match="not authoritative"):
        _verify_source_contract(contract, source)

    prepare_calls: list[object] = []

    class _BackendThatMustNotPrepare:
        def __init__(self, **_options: object) -> None:
            pass

        def prepare(self, config: object, artifacts: object) -> object:
            prepare_calls.append((config, artifacts))
            raise AssertionError("invalid attestation reached HailoBackend.prepare")

    hef = tmp_path / "model.hef"
    hef.write_bytes(b"hef")
    monkeypatch.setattr(
        hailo_runner,
        "_resolve_declared_output_contract",
        lambda *_args, **_kwargs: copy.deepcopy(contract),
    )
    monkeypatch.setattr(hailo_runner, "HailoBackend", _BackendThatMustNotPrepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_hailo10_hef_runner.py",
            "--hef",
            str(hef),
            "--onnx",
            str(source),
            "--hw-arch",
            "hailo8",
            "--backend-label",
            "native_full_hailo8",
            "--model",
            "yolov7_paper",
            "--task",
            "detection",
            "--declared-output-contract-json",
            str(tmp_path / "declaration.json"),
            "--artifacts-dir",
            str(tmp_path / "artifacts"),
        ],
    )
    with pytest.raises(RuntimeError, match="not authoritative"):
        hailo_runner.main()
    assert prepare_calls == []


def test_hailo8_yolov7_source_attestation_rejects_resealed_wrong_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "yolov7.onnx"
    source.write_bytes(b"source-onnx-bytes")
    contract = _source_output_contract(source)
    attestation = contract["source_onnx_raw_head_attestation"]
    attestation["outputs"][0]["shape"] = [1, 3, 81, 81, 85]
    body = {key: value for key, value in attestation.items() if key != "attestation_sha256"}
    attestation["attestation_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    contract["source_onnx_raw_head_attestation_sha256"] = attestation[
        "attestation_sha256"
    ]
    with pytest.raises(RuntimeError, match="output signature mismatch"):
        _verify_source_contract(contract, source)


def test_hailo8_yolov7_source_attestation_rejects_changed_source_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "yolov7.onnx"
    source.write_bytes(b"source-onnx-bytes")
    contract = _source_output_contract(source)
    source.write_bytes(b"changed-source-onnx-bytes")
    with pytest.raises(RuntimeError, match="byte binding mismatch"):
        _verify_source_contract(contract, source)


def test_attested_backend_mapper_uses_shapes_not_names_or_positions() -> None:
    names = ["yolov7/conv95", "yolov7/conv71", "yolov7/conv84"]
    physical_shapes = {
        "yolov7/conv95": (20, 20, 255),
        "yolov7/conv71": (80, 80, 255),
        "yolov7/conv84": (40, 40, 255),
    }
    assert _build_attested_hailo_output_map(
        names, _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES, physical_shapes
    ) == {
        "yolov7/conv95": "clone_2",
        "yolov7/conv71": "output",
        "yolov7/conv84": "clone_1",
    }
    unordered = {
        "clone_2": np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
        "output": np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
        "clone_1": np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
    }
    assert list(_order_attested_output_mapping(
        unordered, _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES
    )) == ["output", "clone_1", "clone_2"]


def test_attested_backend_adapter_only_accepts_rank5_or_packed_hwc_nhwc() -> None:
    target = _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES["clone_2"]
    canonical = np.arange(np.prod(target), dtype=np.float32).reshape(target)
    packed_nhwc = canonical.transpose(0, 2, 3, 1, 4).reshape(1, 20, 20, 255)
    packed_hwc = packed_nhwc[0]

    np.testing.assert_array_equal(
        _adapt_attested_hailo_output_tensor(canonical, target), canonical
    )
    np.testing.assert_array_equal(
        _adapt_attested_hailo_output_tensor(packed_hwc, target), canonical
    )
    np.testing.assert_array_equal(
        _adapt_attested_hailo_output_tensor(packed_nhwc, target), canonical
    )
    for forbidden in (
        canonical[0],  # rank-4 anchor-first, same element count
        packed_hwc.transpose(2, 0, 1),  # packed CHW, same element count
    ):
        with pytest.raises(RuntimeError, match="shape mismatch"):
            _adapt_attested_hailo_output_tensor(forbidden, target)


def test_attested_backend_mapper_rejects_missing_extra_and_ambiguous_streams() -> None:
    shapes = {
        "conv71": (80, 80, 255),
        "conv84": (40, 40, 255),
        "conv95": (20, 20, 255),
    }
    with pytest.raises(RuntimeError):
        _build_attested_hailo_output_map(
            ["conv71", "conv84"],
            _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
            {"conv71": shapes["conv71"], "conv84": shapes["conv84"]},
        )
    ambiguous = {**shapes, "conv95": (40, 40, 255)}
    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        _build_attested_hailo_output_map(
            list(ambiguous), _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES, ambiguous
        )
    with pytest.raises(RuntimeError):
        _build_attested_hailo_output_map(
            [*shapes, "extra"],
            _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
            {**shapes, "extra": (10, 10, 255)},
        )


def test_attested_backend_metadata_falls_back_only_from_empty_loader_state() -> None:
    expected_names = list(_HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES)
    names, shapes = _reconcile_attested_source_output_metadata(
        [], {}, _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES
    )
    assert names == expected_names
    assert shapes == _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES
    assert _reconcile_attested_source_output_metadata(
        expected_names,
        _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
        _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
    ) == (expected_names, _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES)
    with pytest.raises(RuntimeError, match="names conflict"):
        _reconcile_attested_source_output_metadata(
            ["output", "clone_2", "clone_1"],
            _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
            _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
        )
    with pytest.raises(RuntimeError, match="shapes conflict"):
        _reconcile_attested_source_output_metadata(
            expected_names,
            {**_HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES, "clone_2": (1, 3, 21, 21, 85)},
            _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
        )


def test_canonicalization_restores_exact_existing_endpoint_hash() -> None:
    canonical = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES.items()
    }
    physical = {
        "yolov7_paper_full/conv71": np.zeros((80, 80, 255), dtype=np.float32),
        "yolov7_paper_full/conv84": np.zeros((40, 40, 255), dtype=np.float32),
        "yolov7_paper_full/conv95": np.zeros((20, 20, 255), dtype=np.float32),
    }
    canonical_hash = _endpoint_hash(
        task="detection",
        stage="raw_head",
        output_format="raw_detection_tensors",
        signature=_tensor_signature(canonical),
    )
    physical_hash = _endpoint_hash(
        task="detection",
        stage="raw_head",
        output_format="raw_detection_tensors",
        signature=_tensor_signature(physical),
    )
    mapping = _build_attested_hailo_output_map(
        list(physical),
        _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
        {name: tuple(tensor.shape) for name, tensor in physical.items()},
    )
    adapted = _order_attested_output_mapping(
        {
            mapping[name]: _adapt_attested_hailo_output_tensor(
                tensor, _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES[mapping[name]]
            )
            for name, tensor in physical.items()
        },
        _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES,
    )
    adapted_hash = _endpoint_hash(
        task="detection",
        stage="raw_head",
        output_format="raw_detection_tensors",
        signature=_tensor_signature(adapted),
    )
    assert canonical_hash == "051f0d9e489355d312ad656532acf3ed8a2719774d8ef9843278818f5bd15adc"
    assert physical_hash == "79881fc083be708db1c6ebe85792263726fa193beb6057ef4b8a18d7956704aa"
    assert adapted_hash == canonical_hash
    assert canonical_hash != physical_hash


def test_remote_script_mirrors_are_byte_identical() -> None:
    pairs = (
        (
            ROOT / "scripts/native_producer_validate_visualize.py",
            ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_producer_validate_visualize.py",
        ),
        (
            ROOT / "scripts/smoke_hailo10_hef_runner.py",
            ROOT / "onnx_splitpoint_tool/resources/remote_scripts/smoke_hailo10_hef_runner.py",
        ),
        (
            ROOT / "scripts/native_producer_final_report.py",
            ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_producer_final_report.py",
        ),
        (
            ROOT / "scripts/native_full_baseline_eval_runner.py",
            ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_full_baseline_eval_runner.py",
        ),
    )
    for local, mirror in pairs:
        assert local.read_bytes() == mirror.read_bytes()


def test_attested_backend_option_is_wired_from_smoke_to_prepare() -> None:
    smoke_source = (ROOT / "scripts/smoke_hailo10_hef_runner.py").read_text(
        encoding="utf-8"
    )
    backend_source = (
        ROOT / "onnx_splitpoint_tool/runners/backends/hailo_backend.py"
    ).read_text(encoding="utf-8")
    assert 'options["attested_source_output_shapes"]' in smoke_source
    assert 'options.get(\n                "attested_source_output_shapes"' in backend_source
    assert "_build_attested_hailo_output_map(" in backend_source
    assert "_adapt_attested_hailo_output_tensor(" in backend_source


def test_attested_backend_contract_is_exact_not_generic() -> None:
    assert _validated_attested_source_output_shapes(
        {
            name: list(shape)
            for name, shape in _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES.items()
        }
    ) == _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES
    with pytest.raises(RuntimeError, match="sealed Hailo-8 YOLOv7 contract"):
        _validated_attested_source_output_shapes(
            {"output": [1, 3, 80, 80, 85]}
        )
    with pytest.raises(RuntimeError, match="sealed Hailo-8 YOLOv7 contract"):
        _validated_attested_source_output_shapes(
            {
                "clone_2": [1, 3, 20, 20, 85],
                "clone_1": [1, 3, 40, 40, 85],
                "output": [1, 3, 80, 80, 85],
            }
        )
