from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.quality_cache import (
    image_ids_fingerprint,
    json_fingerprint,
)
from onnx_splitpoint_tool.quality_service import (
    QualityArtifactIntegrityError,
    quality_request_from_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_TEMPLATE = (
    ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
)
CANONICAL_ENDPOINT = ROOT / "onnx_splitpoint_tool/native_output_endpoint.py"
CANONICAL_DETECTION_POSTPROCESS = (
    ROOT / "onnx_splitpoint_tool/native_detection_postprocess.py"
)
RUNNER_LIBRARY = ROOT / "onnx_splitpoint_tool/runners"


def _save_signature_model(
    path: Path, *, graph_name: str, outputs: Sequence[tuple[str, Sequence[int]]],
) -> None:
    # The endpoint binder needs the exact ONNX output signature, not an
    # executable graph.  Keeping the synthetic graph node-free makes the real
    # YOLO tensor shapes cheap to construct while preserving their names/ranks.
    graph = helper.make_graph(
        [],
        graph_name,
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))
            for name, shape in outputs
        ],
    )
    onnx.save(helper.make_model(graph), path)


def _write_contract(
    suite: Path, *, model_id: str, endpoint_mode: str,
) -> None:
    needs_postprocess = endpoint_mode in {
        "raw_detection_head", "decoded_pre_nms",
    }
    output_format = (
        "ultralytics_decoded"
        if endpoint_mode == "decoded_pre_nms" else None
    )
    (suite / "output_contracts.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": model_id,
        "task": "detection",
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "schema_version": 1,
            "model_id": model_id,
            "task": "detection",
            "backend": "cuda_ort",
            "variant": "full",
            "endpoint_mode": endpoint_mode,
            **(
                {"output_format": output_format}
                if output_format is not None else {}
            ),
            "contract_status": "recorded",
            "host_tail_required": needs_postprocess,
            "postprocessing_required": needs_postprocess,
            "requires_external_postprocess": needs_postprocess,
        }],
    }, sort_keys=True), encoding="utf-8")
    (suite / "benchmark_set.json").write_text(
        json.dumps({"model_name": model_id, "benchmark_task": "detection"}),
        encoding="utf-8",
    )


def _write_hailo_raw_contract(suite: Path, *, model_id: str) -> Path:
    hef = suite / f"{model_id}_full.hef"
    hef.write_bytes(b"synthetic-hailo-raw-head-artifact")
    (suite / "output_contracts.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": model_id,
        "task": "detection",
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "schema_version": 1,
            "model_id": model_id,
            "task": "detection",
            "backend": "hailo8",
            "variant": "full",
            "endpoint_mode": "raw_detection_head",
            "contract_status": "recorded",
            "host_tail_required": True,
            "postprocessing_required": True,
            "requires_external_postprocess": True,
            "recorded_artifact_path": hef.name,
            "recorded_artifact_sha256": hashlib.sha256(
                hef.read_bytes()
            ).hexdigest(),
            "recorded_artifact_size_bytes": hef.stat().st_size,
        }],
    }, sort_keys=True), encoding="utf-8")
    return hef


def _load_self_contained_runner(
    tmp_path: Path, *, model_id: str, endpoint_mode: str,
    outputs: Sequence[tuple[str, Sequence[int]]],
) -> tuple[Any, Path, Path, Path]:
    suite = tmp_path / model_id / "suite"
    case = suite / "b038"
    case.mkdir(parents=True)
    shutil.copytree(RUNNER_LIBRARY, suite / "splitpoint_runners")
    shutil.copy2(
        CANONICAL_ENDPOINT,
        suite / "splitpoint_runners/native_output_endpoint.py",
    )
    shutil.copy2(
        CANONICAL_DETECTION_POSTPROCESS,
        suite / "splitpoint_runners/native_detection_postprocess.py",
    )
    runner_path = case / "run_split_onnxruntime.py"
    endpoint_sha = hashlib.sha256(CANONICAL_ENDPOINT.read_bytes()).hexdigest()
    runner_source = RUNNER_TEMPLATE.read_text(encoding="utf-8").replace(
        "__VENDORED_ENDPOINT_ATTESTOR_SHA256__",
        endpoint_sha,
    )
    runner_path.write_text(runner_source, encoding="utf-8")

    manifest_full = suite / f"{model_id}.onnx"
    _save_signature_model(
        manifest_full, graph_name=f"{model_id}_graph", outputs=outputs,
    )
    persistent_full = tmp_path / model_id / "persistent_trt_cache/source.onnx"
    persistent_full.parent.mkdir(parents=True)
    shutil.copy2(manifest_full, persistent_full)
    part2 = case / f"{model_id}_part2.onnx"
    shutil.copy2(manifest_full, part2)
    (case / "split_manifest.json").write_text(json.dumps({
        "full_model": f"../{manifest_full.name}",
        "part2_model": part2.name,
    }), encoding="utf-8")
    _write_contract(suite, model_id=model_id, endpoint_mode=endpoint_mode)

    module_name = f"_v269e_generated_runner_{model_id}_{id(tmp_path)}"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # The release-test environment intentionally carries ONNX but not the much
    # larger onnxruntime wheel.  These regressions exercise only the generated
    # runner's contract/export helpers, so a module placeholder is sufficient;
    # no inference API is invoked.
    missing = object()
    previous_onnxruntime = sys.modules.get("onnxruntime", missing)
    if previous_onnxruntime is missing:
        sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_onnxruntime is missing:
            sys.modules.pop("onnxruntime", None)
        else:
            sys.modules["onnxruntime"] = previous_onnxruntime

    # This proves that the generated runner loaded the leaf file from its own
    # bundle.  A globally importable onnx_splitpoint_tool package cannot mask a
    # broken self-contained import in this regression.
    assert module.runtime_output_contract.__module__ == (
        "_onnx_splitpoint_suite_native_output_endpoint"
    )
    assert module._runtime_output_contract_import_error == ""
    assert module._endpoint_attestor_sha256 == endpoint_sha
    return module, case, manifest_full, persistent_full


def _full_detection_request(
    module: Any, *, case: Path, persistent_full: Path,
    names: list[str], arrays: list[np.ndarray], diagnostics: dict[str, Any],
    producer_context: dict[str, Any] | None = None,
    quality_contract: dict[str, Any] | None = None,
    completed_task_evidence: dict[str, Any] | None = None,
    full_only_quality_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration = module._recorded_suite_endpoint_declaration(
        base_dir=case,
        full_model=persistent_full,
        terminal_model=persistent_full,
        variant="full",
        provider="tensorrt",
        task="detection",
        output_names=names,
        outputs=arrays,
        diagnostics=diagnostics,
    )
    endpoint = module._central_quality_endpoint_contract(
        task="detection",
        output_names=names,
        outputs=arrays,
        detected_output_format=module._detect_output_format(names, arrays),
        declared_endpoint_contract=declaration,
        diagnostics=diagnostics,
    )
    policy = {
        "statistics": {
            "execution_location": "central_management",
            "bootstrap_repetitions": 25,
        },
        "detection": {"non_inferiority_margin": 0.01},
    }
    policy["policy_sha256"] = module._quality_contract_sha256(policy)
    request = module._export_central_quality_inputs(
        out_dir=case / "quality",
        task="detection",
        variant="full",
        policy=policy,
        gt_by_image={"000000000001.jpg": []},
        candidate_by_image={"000000000001.jpg": []},
        reference_by_image={},
        endpoint_contract=endpoint,
        runtime_precision_identity="fp16",
        producer_context=producer_context,
        quality_contract=quality_contract,
        completed_task_evidence=completed_task_evidence,
        full_only_quality_identity=full_only_quality_identity,
        diagnostics=diagnostics,
    )
    return endpoint, request


def _quality_contract_and_producer_context(
    module: Any, *, root: Path, source_onnx: Path,
    source_output_format: str = "bn6_detections",
    model_id: str = "yolo26s",
) -> tuple[dict[str, Any], dict[str, Any]]:
    cache = root / "producer"
    cache.mkdir(parents=True)
    engine = cache / "full_fp16.engine"
    trtexec = cache / "trtexec"
    engine.write_bytes(b"setup-local-yolo26-fp16-engine")
    trtexec.write_bytes(b"setup-local-trtexec")

    def artifact(path: Path) -> tuple[str, int]:
        return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size

    source_sha, source_size = artifact(source_onnx)
    engine_sha, engine_size = artifact(engine)
    trtexec_sha, trtexec_size = artifact(trtexec)
    receipt_payload = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "command": [
            str(trtexec.resolve()),
            f"--onnx={source_onnx.resolve()}",
            f"--saveEngine={engine.resolve()}",
            "--fp16",
        ],
        "source_onnx": str(source_onnx.resolve()),
        "source_onnx_sha256": source_sha,
        "engine": str(engine.resolve()),
        "engine_sha256": engine_sha,
        "trtexec": str(trtexec.resolve()),
        "trtexec_sha256": trtexec_sha,
    }
    receipt = {
        **receipt_payload,
        "receipt_sha256": module._native_trt_canonical_json_sha256(
            receipt_payload
        ),
    }
    receipt_path = cache / "engine_build_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    preprocessing_identity = module._canonical_image_preprocessing_contract(
        "detection", (640, 640)
    )
    runner_sha = module._quality_file_sha256(Path(module.__file__).resolve())
    attestor_sha = str(module._endpoint_attestor_sha256)
    decoder_identity = {
        "schema": "onnx-splitpoint/detection-decoder-contract",
        "schema_version": 1,
        "adapter_id": "vendored_yolo_harness_with_local_fallback_v1",
        "canonical_record_endpoint": "decoded_xyxy_score_class_detections",
        "source_output_format": source_output_format,
        "source_endpoint_semantics": (
            "raw_multiscale_head"
            if source_output_format == "multiscale_head"
            else source_output_format
        ),
        "source_endpoint_has_integrated_nms": (
            False
            if source_output_format in {
                "multiscale_head", "ultralytics_decoded",
            }
            else None
        ),
        "confidence_threshold": 0.25,
        "implementation_runner_sha256": runner_sha,
    }
    nms_identity = {
        "schema": "onnx-splitpoint/detection-nms-contract",
        "schema_version": 1,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "detr_or_bn6_confidence_threshold": 0.25,
        "detr_or_bn6_iou_threshold": 0.45,
        "detr_or_bn6_max_detections": 300,
        "implementation_runner_sha256": runner_sha,
    }
    decoder_sha = module._quality_contract_sha256(decoder_identity)
    nms_sha = module._quality_contract_sha256(nms_identity)
    quality_record_endpoint_identity = {
        "schema": (
            "onnx-splitpoint/"
            "detection-quality-record-endpoint-contract"
        ),
        "schema_version": 1,
        "canonical_record_endpoint": (
            "decoded_xyxy_score_class_detections"
        ),
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "implementation_runner_sha256": runner_sha,
        "vendored_endpoint_attestor_sha256": attestor_sha,
    }
    quality_record_endpoint_sha = module._quality_contract_sha256(
        quality_record_endpoint_identity
    )
    quality = {
        "schema": "onnx-splitpoint/central-detection-quality-contract",
        "schema_version": 1,
        "task": "detection",
        "model": {"sha256": source_sha},
        "dataset": {
            "manifest_sha256": hashlib.sha256(b"coco-smoke-manifest").hexdigest(),
            "image_ids_sha256": hashlib.sha256(
                b"coco-smoke-image-ids"
            ).hexdigest(),
            "ground_truth_sha256": hashlib.sha256(
                b"coco-smoke-ground-truth"
            ).hexdigest(),
            "image_count": 1,
        },
        "preprocessing": {
            "identity": preprocessing_identity,
            "sha256": module._quality_contract_sha256(preprocessing_identity),
        },
        "decoder": {
            "identity": decoder_identity,
            "sha256": decoder_sha,
        },
        "nms": {
            "identity": nms_identity,
            "sha256": nms_sha,
        },
        "quality_record_endpoint": {
            "identity": quality_record_endpoint_identity,
            "sha256": quality_record_endpoint_sha,
        },
        "quality_record_endpoint_contract_sha256": (
            quality_record_endpoint_sha
        ),
        "canonical_record_endpoint": "decoded_xyxy_score_class_detections",
        "source_endpoint_is_raw": source_output_format == "multiscale_head",
    }
    quality["quality_contract_sha256"] = module._quality_contract_sha256(quality)
    receipt_sha = module._native_trt_canonical_json_sha256(receipt)
    receipt_size = len(module._native_trt_canonical_json_bytes(receipt))
    context = {
        "execution_role": "full_quality_only",
        "eval_run_id": "eval-smoke-v269e",
        "model_id": model_id,
        "setup_id": "orin_nx_hailo8_01",
        "source_run_id": "hailo8_to_trt",
        "originating_plan_run_id": "hailo8_to_tensorrt",
        "source_onnx": str(source_onnx.resolve()),
        "source_onnx_sha256": source_sha,
        "source_onnx_size_bytes": source_size,
        "build_onnx": str(source_onnx.resolve()),
        "build_onnx_sha256": source_sha,
        "build_onnx_size_bytes": source_size,
        "engine": str(engine.resolve()),
        "engine_sha256": engine_sha,
        "engine_size_bytes": engine_size,
        "trtexec": str(trtexec.resolve()),
        "trtexec_sha256": trtexec_sha,
        "trtexec_size_bytes": trtexec_size,
        "engine_build_receipt": receipt,
        "engine_build_receipt_path": str(receipt_path.resolve()),
        "engine_build_receipt_sha256": receipt_sha,
        "engine_build_receipt_size_bytes": receipt_size,
        "engine_build_receipt_file_sha256": hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest(),
    }
    return quality, context


def _full_only_trt_export_bundle(
    tmp_path: Path,
) -> tuple[Any, Path, Path, Path, dict[str, Any]]:
    """Export the real signed TRT request used by the Full-only canary."""

    module, case, _manifest_full, persistent_full = (
        _load_self_contained_runner(
            tmp_path,
            model_id="yolo26s",
            endpoint_mode="decoded",
            outputs=[("output0", [1, 300, 6])],
        )
    )
    quality_contract, producer_context = (
        _quality_contract_and_producer_context(
            module,
            root=tmp_path / "full-only-trt",
            source_onnx=persistent_full,
        )
    )
    image_id = "000000000001.jpg"
    quality_contract["dataset"].update({
        "image_ids_sha256": image_ids_fingerprint([image_id]),
        "ground_truth_sha256": json_fingerprint([{
            "image_id": image_id,
            "ground_truth": [],
        }]),
        "image_count": 1,
    })
    quality_contract["contract_scope"] = (
        "canonical_quality_record_semantics"
    )
    quality_contract["source_endpoint_role"] = (
        "canonical_reference_model_output"
    )
    quality_contract.pop("quality_contract_sha256", None)
    quality_contract["quality_contract_sha256"] = json_fingerprint(
        quality_contract
    )
    producer_context.update({
        "eval_run_id": "eval-full-only-trt-alias",
        "setup_id": "orin_nx_deepx_m1_01",
        "source_run_id": "native_full_tensorrt",
        "originating_plan_run_id": "ort_tensorrt",
    })
    full_only_identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": "tensorrt_at_deepx_m1_full",
        "eval_run_id": producer_context["eval_run_id"],
        "model_id": producer_context["model_id"],
        "setup_id": producer_context["setup_id"],
        "source_run_id": producer_context["source_run_id"],
        "backend": "tensorrt",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    _endpoint, exported = _full_detection_request(
        module,
        case=case,
        persistent_full=persistent_full,
        names=["output0"],
        arrays=[np.zeros((1, 300, 6), dtype=np.float32)],
        diagnostics={},
        producer_context=producer_context,
        quality_contract=quality_contract,
        full_only_quality_identity=full_only_identity,
    )
    request_path = Path(exported["request"]["path"])
    candidate_path = request_path.parent / exported["candidate"]["path"]
    reference_path = case / "quality/full_only_reference.json"
    module._write_stable_quality_json(reference_path, {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "detection",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "provenance_required": True,
        "quality_contract": quality_contract,
        "quality_contract_sha256": quality_contract[
            "quality_contract_sha256"
        ],
        "records": [{
            "image_id": image_id,
            "ground_truth": [],
            "reference": [],
        }],
    })
    return (
        module, request_path, candidate_path, reference_path,
        full_only_identity,
    )


def test_full_only_trt_export_keeps_physical_backend_and_loads_strictly(
    tmp_path: Path,
) -> None:
    (
        _module, request_path, candidate_path, reference_path,
        full_only_identity,
    ) = _full_only_trt_export_bundle(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    for payload in (request, candidate):
        assert payload["backend"] == "native_tensorrt"
        assert payload["producer_identity"]["backend"] == "native_tensorrt"
        assert payload["full_only_plan_identity_required"] is True
        assert payload["full_only_plan_identity"] == full_only_identity
        assert payload["full_only_plan_identity"]["backend"] == "tensorrt"
        assert payload["full_only_plan_identity_sha256"] == json_fingerprint(
            full_only_identity
        )

    loaded = quality_request_from_manifest(
        request_path, reference_artifact=reference_path,
    )
    assert len(loaded.candidate_records) == 1
    assert loaded.metric_gate_config["producer_identity_sha256"] == request[
        "producer_identity_sha256"
    ]


@pytest.mark.parametrize(
    ("tampered_artifact", "expected_error"),
    [
        (
            "request",
            "request top-level backend differs from TensorRT producer identity",
        ),
        (
            "candidate",
            "candidate top-level backend differs from TensorRT producer identity",
        ),
    ],
)
def test_full_only_trt_strict_loader_rejects_logical_backend_as_producer_duplicate(
    tmp_path: Path, tampered_artifact: str, expected_error: str,
) -> None:
    module, request_path, candidate_path, reference_path, _identity = (
        _full_only_trt_export_bundle(tmp_path)
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if tampered_artifact == "request":
        request["backend"] = "tensorrt"
    else:
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["backend"] = "tensorrt"
        candidate_descriptor = module._write_stable_quality_json(
            candidate_path, candidate,
        )
        candidate_descriptor["path"] = candidate_path.name
        request["candidate"] = candidate_descriptor
    module._write_stable_quality_json(request_path, request)

    with pytest.raises(QualityArtifactIntegrityError, match=expected_error):
        quality_request_from_manifest(
            request_path, reference_artifact=reference_path,
        )


def test_vendored_yolo26_persistent_full_copy_emits_decoded_nms_request(
    tmp_path: Path,
) -> None:
    outputs = [("output0", [1, 300, 6])]
    module, case, manifest_full, persistent_full = _load_self_contained_runner(
        tmp_path, model_id="yolo26s", endpoint_mode="decoded", outputs=outputs,
    )
    arrays = [np.zeros((1, 300, 6), dtype=np.float32)]
    diagnostics: dict[str, Any] = {}
    quality_contract, producer_context = _quality_contract_and_producer_context(
        module, root=tmp_path / "yolo26s", source_onnx=persistent_full,
    )

    endpoint, request = _full_detection_request(
        module, case=case, persistent_full=persistent_full,
        names=["output0"], arrays=arrays, diagnostics=diagnostics,
        producer_context=producer_context,
        quality_contract=quality_contract,
    )

    assert manifest_full.resolve() != persistent_full.resolve()
    assert diagnostics["declaration"]["status"] == "attested"
    assert endpoint["endpoint_contract_complete"] is True
    assert endpoint["stage"] == "decoded_nms"
    assert request["variant"] == "full"
    assert request["record_count"] == 1
    assert Path(request["request"]["path"]).is_file()
    assert request["producer_identity"]["execution_role"] == "full_quality_only"
    assert request["producer_identity"]["model_id"] == "yolo26s"
    assert request["producer_identity"]["engine"]["sha256"] == (
        producer_context["engine_sha256"]
    )
    assert request["producer_identity"]["vendored_endpoint_attestor_sha256"] == (
        hashlib.sha256(CANONICAL_ENDPOINT.read_bytes()).hexdigest()
    )
    assert request["producer_identity"]["endpoint_attestor"]["identity"][
        "source"
    ] == "suite_vendored"
    assert request["producer_identity"]["quality_contract"]["decoder"][
        "identity"
    ]["source_output_format"] == "bn6_detections"
    assert request["producer_identity"]["quality_contract"]["nms"][
        "identity"
    ]["detr_or_bn6_confidence_threshold"] == 0.25
    assert request["producer_identity"]["quality_contract"]["nms"][
        "identity"
    ]["detr_or_bn6_iou_threshold"] == 0.45
    assert request["producer_identity"]["quality_contract"]["nms"][
        "identity"
    ]["detr_or_bn6_max_detections"] == 300
    assert len(request["producer_identity_sha256"]) == 64
    assert diagnostics["producer_identity"]["status"] == "attested"
    assert diagnostics["export"]["status"] == "emitted"
    # Changing the attestor after import cannot silently reuse the already
    # loaded functions: producer construction re-hashes the live suite file.
    Path(module._endpoint_attestor_path).write_text(
        "# modified endpoint attestor\n",
        encoding="utf-8",
    )
    tampered_diagnostics: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="attestor bytes"):
        _full_detection_request(
            module,
            case=case,
            persistent_full=persistent_full,
            names=["output0"],
            arrays=arrays,
            diagnostics=tampered_diagnostics,
            producer_context=producer_context,
            quality_contract=quality_contract,
        )
    assert tampered_diagnostics["producer_identity"]["status"] == "failed_closed"
    shutil.copy2(CANONICAL_ENDPOINT, Path(module._endpoint_attestor_path))

    # The same logical context with changed receipt bytes must fail before a
    # request can be emitted, and the diagnostic must name the producer gate.
    receipt_path = Path(producer_context["engine_build_receipt_path"])
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    tamper_diagnostics: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="receipt binding is inconsistent"):
        _full_detection_request(
            module, case=case, persistent_full=persistent_full,
            names=["output0"], arrays=arrays,
            diagnostics=tamper_diagnostics,
            producer_context=producer_context,
            quality_contract=quality_contract,
        )
    assert tamper_diagnostics["producer_identity"] == {
        "status": "failed_closed",
        "reason": "native_full_tensorrt_producer_identity_invalid",
        "error": (
            "RuntimeError: quality-only TRT producer receipt binding is inconsistent"
        ),
    }


def test_generic_quality_export_requires_exact_suite_vendored_attestor(
    tmp_path: Path,
) -> None:
    module, case, _manifest_full, persistent_full = (
        _load_self_contained_runner(
            tmp_path,
            model_id="yolo26s",
            endpoint_mode="decoded",
            outputs=[("output0", [1, 300, 6])],
        )
    )
    quality_contract, _producer_context = (
        _quality_contract_and_producer_context(
            module, root=tmp_path / "generic-yolo26s",
            source_onnx=persistent_full,
        )
    )
    expected = hashlib.sha256(CANONICAL_ENDPOINT.read_bytes()).hexdigest()
    assert (
        module._verified_suite_vendored_endpoint_attestor_sha256()
        == expected
    )
    quality_root = case / "quality" / "task_quality_inputs"

    def export_generic() -> tuple[dict[str, Any], dict[str, Any]]:
        return _full_detection_request(
            module,
            case=case,
            persistent_full=persistent_full,
            names=["output0"],
            arrays=[np.zeros((1, 300, 6), dtype=np.float32)],
            diagnostics={},
            quality_contract=quality_contract,
        )

    module._endpoint_attestor_source = "installed_package_fallback"
    with pytest.raises(RuntimeError, match="attestor bytes"):
        export_generic()
    assert not quality_root.exists()
    module._endpoint_attestor_source = "suite_vendored"

    module._endpoint_attestor_sha256 = "b" * 64
    with pytest.raises(RuntimeError, match="attestor bytes"):
        export_generic()
    assert not quality_root.exists()
    module._endpoint_attestor_sha256 = expected

    Path(module._endpoint_attestor_path).write_text(
        "# modified endpoint attestor\n", encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="attestor bytes"):
        export_generic()
    assert not quality_root.exists()

    shutil.copy2(CANONICAL_ENDPOINT, Path(module._endpoint_attestor_path))
    _endpoint, request = export_generic()
    assert Path(request["request"]["path"]).is_file()
    assert quality_contract["quality_record_endpoint"]["identity"][
        "vendored_endpoint_attestor_sha256"
    ] == expected


def test_generic_yolo26_full_request_carries_verified_completed_endpoint(
    tmp_path: Path,
) -> None:
    module, case, _manifest_full, persistent_full = (
        _load_self_contained_runner(
            tmp_path,
            model_id="yolo26s",
            endpoint_mode="decoded",
            outputs=[("output0", [1, 300, 6])],
        )
    )
    raw_outputs: dict[str, np.ndarray] = {}
    conv = 61
    for size in (80, 40, 20):
        raw_outputs[f"yolo26s_full/conv{conv}"] = np.zeros(
            (size, size, 4), dtype=np.float32,
        )
        raw_outputs[f"yolo26s_full/conv{conv + 3}"] = np.full(
            (size, size, 80), -20.0, dtype=np.float32,
        )
        conv += 16
    frozen = build_frozen_postprocess_contract(
        model_id="yolo26s",
        outputs=raw_outputs,
        input_hw=[640, 640],
        original_wh=[640, 640],
    )
    processor = FrozenDetectionPostprocessor(frozen)
    frozen_result = processor.process(
        raw_outputs, original_wh=[640, 640],
    )
    attestation = build_completed_detection_endpoint_attestation(
        frozen,
        frozen_result,
        completed_frames=3,
        postprocess_completed_frames=3,
    )
    completed = {
        "completed_task_comparison_endpoint_contract": attestation[
            "completed_task_comparison_endpoint_contract"
        ],
        "completed_task_comparison_endpoint_contract_hash": attestation[
            "completed_task_comparison_endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": attestation[
            "completed_task_comparison_output_endpoint_id"
        ],
        "completed_task_endpoint_attestation": attestation,
    }
    endpoint, request = _full_detection_request(
        module,
        case=case,
        persistent_full=persistent_full,
        names=["output0"],
        arrays=[np.zeros((1, 300, 6), dtype=np.float32)],
        diagnostics={},
        completed_task_evidence=completed,
    )
    assert endpoint["endpoint_contract_hash"] != request[
        "completed_task_endpoint_contract_hash"
    ]
    assert request["quality_join_endpoint"] == (
        "completed_task_decoded_nms"
    )
    assert request["completed_task_endpoint_contract_hash"] == (
        attestation[
            "completed_task_comparison_endpoint_contract_hash"
        ]
    )

    request_path = Path(request["request"]["path"])
    candidate_path = request_path.parent / request["candidate"]["path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert candidate["completed_task_endpoint_contract_hash"] == request[
        "completed_task_endpoint_contract_hash"
    ]
    reference = case / "quality/reference.json"
    reference.write_text(json.dumps({
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "detection",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "records": [{
            "image_id": "000000000001.jpg",
            "ground_truth": [],
            "reference": [],
        }],
    }), encoding="utf-8")
    loaded = quality_request_from_manifest(
        request_path, reference_artifact=reference,
    )
    assert len(loaded.candidate_records) == 1

    candidate["completed_task_endpoint_contract_hash"] = "f" * 64
    candidate_descriptor = module._write_stable_quality_json(
        candidate_path, candidate,
    )
    candidate_descriptor["path"] = candidate_path.name
    request_payload = json.loads(
        request_path.read_text(encoding="utf-8"),
    )
    request_payload["candidate"] = candidate_descriptor
    module._write_stable_quality_json(request_path, request_payload)
    with pytest.raises(
        QualityArtifactIntegrityError,
        match="different completed-task endpoint bindings",
    ):
        quality_request_from_manifest(
            request_path, reference_artifact=reference,
        )


def test_vendored_yolov7_persistent_full_copy_emits_raw_head_request(
    tmp_path: Path,
) -> None:
    outputs = [
        ("output", [1, 3, 80, 80, 85]),
        ("clone_1", [1, 3, 40, 40, 85]),
        ("clone_2", [1, 3, 20, 20, 85]),
    ]
    module, case, _manifest_full, persistent_full = _load_self_contained_runner(
        tmp_path, model_id="yolov7_paper",
        endpoint_mode="raw_detection_head", outputs=outputs,
    )
    arrays = [np.zeros(tuple(shape), dtype=np.float32) for _name, shape in outputs]
    names = [name for name, _shape in outputs]
    diagnostics: dict[str, Any] = {}
    quality_contract, producer_context = _quality_contract_and_producer_context(
        module, root=tmp_path / "yolov7_paper", source_onnx=persistent_full,
        source_output_format="multiscale_head",
    )
    producer_context["model_id"] = "yolov7_paper"

    endpoint, request = _full_detection_request(
        module, case=case, persistent_full=persistent_full,
        names=names, arrays=arrays, diagnostics=diagnostics,
        producer_context=producer_context,
        quality_contract=quality_contract,
    )

    assert module._detect_output_format(names, arrays) == "multiscale_head"
    assert endpoint["endpoint_contract_complete"] is True
    assert endpoint["stage"] == "raw_head"
    assert endpoint["output_format"] == "raw_detection_tensors"
    assert endpoint["output_endpoint_attestation"][
        "endpoint_contract_hash"
    ] == endpoint["endpoint_contract_hash"]
    assert request["variant"] == "full"
    assert request["record_count"] == 1
    assert request["producer_identity"]["endpoint_authority"]["identity"][
        "output_endpoint_attestation"
    ]["endpoint_contract_hash"] == endpoint["endpoint_contract_hash"]
    assert request["producer_identity"]["quality_contract"]["decoder"][
        "identity"
    ]["confidence_threshold"] == 0.25
    assert request["producer_identity"]["quality_contract"]["nms"][
        "identity"
    ]["iou_threshold"] == 0.45
    assert request["producer_identity"]["quality_contract"]["nms"][
        "identity"
    ]["max_detections"] == 300
    assert diagnostics["export"]["status"] == "emitted"

    noncanonical_quality = json.loads(json.dumps(quality_contract))
    noncanonical_quality["nms"]["identity"]["max_detections"] = 200
    noncanonical_quality["nms"]["sha256"] = module._quality_contract_sha256(
        noncanonical_quality["nms"]["identity"]
    )
    noncanonical_quality.pop("quality_contract_sha256")
    noncanonical_quality["quality_contract_sha256"] = (
        module._quality_contract_sha256(noncanonical_quality)
    )
    with pytest.raises(
        RuntimeError,
        match="detection semantics differ from the canonical completed endpoint",
    ):
        module._build_native_full_trt_quality_producer_identity(
            context=producer_context,
            task="detection",
            variant="full",
            policy={
                "policy_sha256": module._quality_contract_sha256(
                    {"name": "noncanonical-quality-semantics"}
                ),
            },
            quality_contract=noncanonical_quality,
            endpoint_contract=endpoint,
            runtime_precision_identity="fp16",
        )

    missing_hash = json.loads(json.dumps(endpoint))
    missing_hash["output_endpoint_attestation"].pop("endpoint_contract_hash")
    with pytest.raises(
        RuntimeError,
        match="endpoint authority is incomplete or inconsistent",
    ):
        module._build_native_full_trt_quality_producer_identity(
            context=producer_context,
            task="detection",
            variant="full",
            policy={
                "policy_sha256": module._quality_contract_sha256(
                    {"name": "raw-head-negative"}
                ),
            },
            quality_contract=quality_contract,
            endpoint_contract=missing_hash,
            runtime_precision_identity="fp16",
        )


def test_hailo_yolo26_compiled_raw_heads_bind_beside_decoded_terminal_onnx(
    tmp_path: Path,
) -> None:
    module, case, _manifest_full, persistent_full = _load_self_contained_runner(
        tmp_path,
        model_id="yolo26s",
        endpoint_mode="decoded",
        outputs=[("output0", [1, 300, 6])],
    )
    _write_hailo_raw_contract(case.parent, model_id="yolo26s")
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for level, size in enumerate((80, 40, 20)):
        names.extend([
            f"/model.23/one2one_cv2.{level}/one2one_cv2.{level}.2/Conv",
            f"/model.23/one2one_cv3.{level}/one2one_cv3.{level}.2/Conv",
        ])
        arrays.extend([
            np.zeros((size, size, 4), dtype=np.float32),
            np.zeros((size, size, 80), dtype=np.float32),
        ])

    diagnostics: dict[str, Any] = {}
    declaration = module._recorded_suite_endpoint_declaration(
        base_dir=case,
        full_model=persistent_full,
        terminal_model=persistent_full,
        variant="full",
        provider="hailo8",
        task="detection",
        output_names=names,
        outputs=arrays,
        diagnostics=diagnostics,
    )
    actual_format = module._detect_output_format(names, arrays)
    endpoint = module._central_quality_endpoint_contract(
        task="detection",
        output_names=names,
        outputs=arrays,
        detected_output_format=actual_format,
        declared_endpoint_contract=declaration,
        diagnostics=diagnostics,
    )

    assert actual_format == "ultralytics_regcls"
    assert declaration["stage"] == "raw_head"
    assert declaration["graph_binding_source"] == (
        "authoritative_suite_hailo_artifact_raw_endpoint_plus_exact_source_onnx:v1"
    )
    assert endpoint["endpoint_contract_complete"] is True
    assert endpoint["stage"] == "raw_head"
    assert endpoint["output_format"] == "raw_detection_tensors"
    assert diagnostics["declaration"]["reason"] == (
        "authoritative_hailo_artifact_contract_plus_raw_runtime_signature"
    )
    assert diagnostics["runtime_endpoint"]["status"] == "attested"


def test_yolo11_full_trt_binds_decoded_pre_nms_producer(
    tmp_path: Path,
) -> None:
    module, case, _manifest_full, persistent_full = (
        _load_self_contained_runner(
            tmp_path,
            model_id="yolo11l",
            endpoint_mode="decoded_pre_nms",
            outputs=[("output0", [1, 84, 8400])],
        )
    )
    names = ["output0"]
    arrays = [np.zeros((1, 84, 8400), dtype=np.float32)]
    diagnostics: dict[str, Any] = {}
    declaration = module._recorded_suite_endpoint_declaration(
        base_dir=case,
        full_model=persistent_full,
        terminal_model=persistent_full,
        variant="full",
        provider="tensorrt",
        task="detection",
        output_names=names,
        outputs=arrays,
        diagnostics=diagnostics,
    )
    detected = module._detect_output_format(names, arrays)
    endpoint = module._central_quality_endpoint_contract(
        task="detection",
        output_names=names,
        outputs=arrays,
        detected_output_format=detected,
        declared_endpoint_contract=declaration,
        diagnostics=diagnostics,
    )

    assert detected == "ultralytics_decoded"
    assert declaration["stage"] == "decoded_pre_nms"
    assert endpoint["endpoint_contract_complete"] is True
    assert endpoint["stage"] == "decoded_pre_nms"
    assert endpoint["output_format"] == "ultralytics_decoded"
    assert diagnostics["runtime_endpoint"]["status"] == "attested"

    quality, context = _quality_contract_and_producer_context(
        module,
        root=tmp_path / "yolo11-producer",
        source_onnx=persistent_full,
        source_output_format="ultralytics_decoded",
        model_id="yolo11l",
    )
    image_id = "000000000001.jpg"
    quality["dataset"].update({
        "image_ids_sha256": image_ids_fingerprint([image_id]),
        "ground_truth_sha256": json_fingerprint([{
            "image_id": image_id,
            "ground_truth": [],
        }]),
        "image_count": 1,
    })
    quality["contract_scope"] = "canonical_quality_record_semantics"
    quality["source_endpoint_role"] = "canonical_reference_model_output"
    quality.pop("quality_contract_sha256", None)
    quality["quality_contract_sha256"] = json_fingerprint(quality)
    policy = {"name": "yolo11-pre-nms-producer"}
    policy["policy_sha256"] = module._quality_contract_sha256(policy)
    producer = module._build_native_full_trt_quality_producer_identity(
        context=context,
        task="detection",
        variant="full",
        policy=policy,
        quality_contract=quality,
        endpoint_contract=endpoint,
        runtime_precision_identity="fp16",
    )
    assert producer["endpoint"]["identity"]["stage"] == "decoded_pre_nms"
    assert producer["quality_contract"]["decoder"]["identity"][
        "source_output_format"
    ] == "ultralytics_decoded"

    emitted_endpoint, emitted = _full_detection_request(
        module,
        case=case,
        persistent_full=persistent_full,
        names=names,
        arrays=arrays,
        diagnostics={},
        producer_context=context,
        quality_contract=quality,
    )
    assert emitted_endpoint["stage"] == "decoded_pre_nms"
    assert emitted["producer_identity"]["endpoint"]["identity"][
        "stage"
    ] == "decoded_pre_nms"
    assert emitted["producer_identity"]["quality_contract"]["decoder"][
        "identity"
    ]["source_output_format"] == "ultralytics_decoded"
    request_path = Path(emitted["request"]["path"])
    assert request_path.is_file()
    reference_path = case / "quality" / "yolo11_reference.json"
    module._write_stable_quality_json(reference_path, {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "detection",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "provenance_required": True,
        "quality_contract": quality,
        "quality_contract_sha256": quality["quality_contract_sha256"],
        "records": [{
            "image_id": image_id,
            "ground_truth": [],
            "reference": [],
        }],
    })
    loaded = quality_request_from_manifest(
        request_path,
        reference_artifact=reference_path,
    )
    assert len(loaded.candidate_records) == 1


def test_hailo_full_quality_uses_timed_decoder_invariant_per_image(
    tmp_path: Path,
) -> None:
    module, _case, _manifest_full, _persistent_full = (
        _load_self_contained_runner(
            tmp_path,
            model_id="yolo26s",
            endpoint_mode="decoded",
            outputs=[("output0", [1, 300, 6])],
        )
    )
    outputs: dict[str, np.ndarray] = {}
    for level, size in enumerate((80, 40, 20)):
        outputs[
            f"/model.23/one2one_cv2.{level}/one2one_cv2.{level}.2/Conv"
        ] = np.zeros((size, size, 4), dtype=np.float32)
        outputs[
            f"/model.23/one2one_cv3.{level}/one2one_cv3.{level}.2/Conv"
        ] = np.zeros((size, size, 80), dtype=np.float32)
    timed = build_frozen_postprocess_contract(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[640, 480],
    )
    presentation_sorted = {
        name: outputs[name] for name in sorted(outputs)
    }
    assert list(presentation_sorted) != list(outputs)

    first_detections, first = module._frozen_hailo_full_quality_candidate(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[640, 480],
        timed_contract=timed,
    )
    second_detections, second = module._frozen_hailo_full_quality_candidate(
        model_id="yolo26s",
        outputs=presentation_sorted,
        input_hw=[640, 640],
        original_wh=[1280, 720],
        timed_contract=timed,
    )

    assert first_detections == []
    assert second_detections == []
    assert first["contract_sha256"] != second["contract_sha256"]
    assert first["invariant_contract_sha256"] == (
        timed["invariant_contract_sha256"]
    )
    assert second["invariant_contract_sha256"] == (
        timed["invariant_contract_sha256"]
    )

    missing = dict(presentation_sorted)
    missing.pop(next(iter(missing)))
    with pytest.raises(
        RuntimeError,
        match="outputs differ from timed decoder tensors",
    ):
        module._frozen_hailo_full_quality_candidate(
            model_id="yolo26s",
            outputs=missing,
            input_hw=[640, 640],
            original_wh=[640, 480],
            timed_contract=timed,
        )

    tampered = dict(timed)
    tampered["invariant_contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="differs from timed decoder"):
        module._frozen_hailo_full_quality_candidate(
            model_id="yolo26s",
            outputs=outputs,
            input_hw=[640, 640],
            original_wh=[640, 480],
            timed_contract=tampered,
        )


def test_incomplete_detection_candidate_reports_primary_per_image_error(
    tmp_path: Path,
) -> None:
    module, _case, _manifest_full, _persistent_full = (
        _load_self_contained_runner(
            tmp_path,
            model_id="yolo26s",
            endpoint_mode="decoded",
            outputs=[("output0", [1, 300, 6])],
        )
    )

    with pytest.raises(RuntimeError) as raised:
        module._require_complete_detection_candidate_adapter(
            expected_image_ids=["a.jpg", "b.jpg"],
            candidate_by_image={},
            per_image_rows=[
                {
                    "image": "a.jpg",
                    "error": (
                        "RuntimeError: Hailo Full quality decoder invariant "
                        "differs from timed decoder"
                    ),
                },
                {
                    "image": "b.jpg",
                    "error": (
                        "RuntimeError: Hailo Full quality decoder invariant "
                        "differs from timed decoder"
                    ),
                },
            ],
        )

    message = str(raised.value)
    assert "candidate adapter incomplete" in message
    assert "expected=2 observed=0 missing=2 extra=0" in message
    assert '"RuntimeError": 2' in message
    assert "differs from timed decoder" in message


def test_quality_first_hash_mismatch_and_composed_path_mismatch_are_explicit(
    tmp_path: Path,
) -> None:
    outputs = [("output0", [1, 300, 6])]
    module, case, _manifest_full, persistent_full = _load_self_contained_runner(
        tmp_path, model_id="yolo26s", endpoint_mode="decoded", outputs=outputs,
    )
    different = persistent_full.parent / "different.onnx"
    _save_signature_model(
        different, graph_name="different_bytes_same_signature", outputs=outputs,
    )
    arrays = [np.zeros((1, 300, 6), dtype=np.float32)]
    diagnostics: dict[str, Any] = {}

    assert module._recorded_suite_endpoint_declaration(
        base_dir=case, full_model=different, terminal_model=different,
        variant="full", provider="tensorrt", task="detection",
        output_names=["output0"], outputs=arrays, diagnostics=diagnostics,
    ) == {}
    assert diagnostics["declaration"]["reason"] == (
        "executed_full_model_sha256_mismatch"
    )
    # The generic downstream export failure must not hide the exact first
    # attestation failure.  Conversely, successful phases have reason strings
    # too and therefore must never be selected as failures.
    diagnostics["runtime_endpoint"] = {
        "status": "attested",
        "reason": "authoritative_declaration_and_runtime_tensor_verified",
    }
    diagnostics["producer_identity"] = {
        "status": "attested",
        "reason": "native_full_tensorrt_producer_identity_verified",
    }
    diagnostics["export"] = {
        "status": "failed_closed",
        "reason": "endpoint_contract_incomplete",
    }
    error = module._quality_first_no_request_error(diagnostics)
    assert "fail_closed_reason=executed_full_model_sha256_mismatch" in str(error)

    producer_error = module._quality_first_no_request_error({
        "declaration": {"status": "attested", "reason": "declaration_ok"},
        "runtime_endpoint": {"status": "attested", "reason": "endpoint_ok"},
        "producer_identity": {
            "status": "failed_closed",
            "reason": "native_full_tensorrt_producer_identity_invalid",
        },
        "export": {
            "status": "failed_closed",
            "reason": "central_quality_request_not_emitted",
        },
    })
    assert (
        "fail_closed_reason=native_full_tensorrt_producer_identity_invalid"
        in str(producer_error)
    )

    wrong_part2 = case / "byte_identical_but_not_manifest_part2.onnx"
    shutil.copy2(case / "yolo26s_part2.onnx", wrong_part2)
    diagnostics = {}
    assert module._recorded_suite_endpoint_declaration(
        base_dir=case, full_model=persistent_full, terminal_model=wrong_part2,
        variant="composed", provider="tensorrt", task="detection",
        output_names=["output0"], outputs=arrays, diagnostics=diagnostics,
    ) == {}
    assert diagnostics["declaration"]["reason"] == (
        "composed_terminal_manifest_path_mismatch"
    )
