"""Normal Native dataset quality intake; scheduling belongs to the caller.

The prior Native selection proves the reused artifacts.  The fresh producer
and per-image normal completion attestations prove the newly captured data.
CPU reference records are read through the existing sealed reference reader.
No inference, detector postprocessing, pool creation, or scientific policy is
implemented here.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import ast
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

NATIVE_DATASET_PRODUCER_SCHEMA = "onnx-splitpoint/native-dataset-quality-producer"
_PUBLISHED_PRODUCER_ROOT = ContextVar("native_dataset_published_producer_root", default=None)


def _capture_execution_ast(path):
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"load_capture_config", "_timed_model_identity"}:
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "native_command_contract":
            node.names = [name for name in node.names if name.name != "verify_native_command_contract"]
        body.append(node)
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


@contextmanager
def published_native_producer_source_scope(source_freeze=None):
    """Read source proofs from a caller-bound published producer, never a request root.

    Only the separately reviewed model-alias loader may differ. Capture,
    postprocessing, Native hotloops, metrics and statistical policy remain the
    same implementation. This changes evidence lookup, not executable imports.
    """
    current = Path(__file__).resolve().parents[1]
    if source_freeze is None:
        yield None
        return
    from .quality_service import _load_quality_artifact
    from .source_integrity import verify_installed_source_integrity
    _descriptor(source_freeze, "caller published Native source freeze")
    path = Path(source_freeze["path"])
    release, _, _ = _load_quality_artifact(source_freeze, path, verify=True)
    _need(release.get("pushed") is True and isinstance(release.get("commit"), str)
          and len(release["commit"]) == 40, "Native producer source is not a published release")
    root = Path(str(release.get("source_export_root") or "")).resolve()
    manifest = release.get("source_manifest")
    _descriptor(manifest, "Native producer source manifest")
    _need(Path(manifest["path"]).resolve() == root / "SOURCE_MANIFEST.json", "Native producer manifest root differs")
    _load_quality_artifact(manifest, path, verify=True)
    integrity = verify_installed_source_integrity(root)
    _need(integrity.get("ok") is True, "Native producer installed source verification failed")
    critical = (required_native_dataset_source_paths("hailo8_to_trt") | required_native_dataset_source_paths("hailo10_to_trt")) - {
        "onnx_splitpoint_tool/native_dataset_quality.py"}
    critical |= {"onnx_splitpoint_tool/" + name for name in (
        "quality_service.py", "quality_metrics.py", "quality_cache.py", "quality_statistics_config.py",
        "accuracy_reporting.py", "quality_result_enrichment.py", "quality_lifecycle.py", "preprocessing_contract.py",
        "native_command_contract.py", "native_split_quality.py", "native_output_endpoint.py")}
    critical.add("onnx_splitpoint_tool/workflow/runner.py")
    for relative in critical:
        _need((root / relative).is_file(), "Native producer compatible source missing: " + relative)
        _need((root / relative).read_bytes() == (current / relative).read_bytes(),
              "Native producer/consumer scientific or runtime source differs: " + relative)
    capture = "onnx_splitpoint_tool/native_dataset_quality.py"
    _need(_capture_execution_ast(root / capture) == _capture_execution_ast(current / capture),
          "Native producer/consumer capture execution differs beyond model metadata")
    proof = {"source_freeze": dict(source_freeze), "commit": release["commit"], "source_manifest": manifest,
             "installed_source_verified": True, "scientific_and_runtime_sources_identical": sorted(critical),
             "capture_execution_unchanged_metadata_alias_only": True}
    token = _PUBLISHED_PRODUCER_ROOT.set(root)
    try:
        yield proof
    finally:
        _PUBLISHED_PRODUCER_ROOT.reset(token)


def required_native_dataset_source_paths(backend):
    """The finite source paths actually called by the Native capture adapters."""
    from .native_split_quality import canonical_native_split_backend
    selected = canonical_native_split_backend(backend)
    common = {
        "onnx_splitpoint_tool/native_dataset_quality.py",
        "onnx_splitpoint_tool/native_dataset_quality_export.py",
        "onnx_splitpoint_tool/native_detection_postprocess.py",
        "onnx_splitpoint_tool/native_three_stage.py",
        "onnx_splitpoint_tool/runners/backends/hailo_backend.py",
    }
    if selected == "hailo8_to_trt":
        return common | {"onnx_splitpoint_tool/native_dataset_quality_h8.py",
            "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_fifo_from_benchmarkset.py",
            "onnx_splitpoint_tool/resources/remote_scripts/native_hailo10_trt_e2e_from_benchmarkset.py"}
    _need(selected == "hailo10h_to_trt", "Native dataset backend unsupported")
    return common | {"onnx_splitpoint_tool/native_dataset_quality_h10.py", "scripts/native_hailo10_trt_e2e_from_benchmarkset.py"}


def _need(value: Any, message: str) -> None:
    if not value:
        from .quality_service import QualityArtifactIntegrityError
        raise QualityArtifactIntegrityError(message)


def _descriptor(value: Any, role: str) -> None:
    from .quality_service import _require_contract_sha256
    _need(isinstance(value, Mapping), f"{role} descriptor missing")
    _require_contract_sha256(value.get("sha256"), role + ".sha256")
    _need(type(value.get("size_bytes")) is int and value["size_bytes"] > 0,
          f"{role} size invalid")
    _need(isinstance(value.get("path"), str) and value["path"].strip(), f"{role} path missing")


def validate_native_dataset_quality_producer(value: Any, *, role: str, task: str):
    """Validate a fresh Native producer without re-labelling its prior Q seal."""
    from .native_split_quality import validate_native_split_quality_binding
    from .quality_cache import json_fingerprint
    from .quality_service import _require_contract_sha256, _validate_detection_quality_contract
    _need(isinstance(value, Mapping), f"{role} Native producer missing")
    producer = deepcopy(dict(value))
    _need(producer.get("schema") == NATIVE_DATASET_PRODUCER_SCHEMA
          and type(producer.get("schema_version")) is int and producer["schema_version"] == 1,
          f"{role} Native producer schema invalid")
    declared = _require_contract_sha256(producer.get("producer_identity_sha256"), role + ".producer_identity_sha256")
    unhashed = dict(producer)
    unhashed.pop("producer_identity_sha256", None)
    _need(json_fingerprint(unhashed) == declared, f"{role} Native producer SHA mismatch")
    _need(task == "detection" and producer.get("task") == task
          and producer.get("variant") == "composed"
          and producer.get("producer_kind") == "native_dataset_completed_quality"
          and producer.get("execution_role") == "native_dataset_quality_candidate"
          and producer.get("performance_claims_emitted") is False
          and producer.get("additional_nms_applied") is False,
          f"{role} Native producer role invalid")
    for name in ("eval_run_id", "model_id", "case_id", "setup_id", "source_run_id", "backend", "runtime_precision_identity"):
        _need(isinstance(producer.get(name), str) and producer[name].strip() == producer[name] and bool(producer[name]),
              f"{role} Native producer {name} invalid")
    _need(producer["source_run_id"] == producer["backend"], f"{role} Native backend differs")
    prior, status = validate_native_split_quality_binding(producer.get("prior_artifact_selection"), expected_identity={
        "model": producer["model_id"], "case": producer["case_id"], "setup_id": producer["setup_id"],
        "backend": producer["backend"], "task": task, "precision": producer["runtime_precision_identity"],
    })
    _need(prior is not None, f"{role} prior Native artifact selection invalid: {status}")
    _need(producer["eval_run_id"] != prior["eval_run_id"], f"{role} fresh Native capture uses prior evaluation id")
    _need(producer.get("preselection") == prior["preselection"], f"{role} Native preselection changed")
    artifacts = producer.get("artifacts")
    _need(isinstance(artifacts, Mapping) and set(artifacts) == set(prior["artifacts"]), f"{role} Native artifact roles changed")
    for name, artifact in artifacts.items():
        _descriptor(artifact, role + ".artifacts." + name)
        _need(all(artifact.get(key) == prior["artifacts"][name].get(key) for key in ("sha256", "size_bytes")),
              f"{role} Native artifact changed: {name}")
    quality, quality_sha = _validate_detection_quality_contract(producer.get("quality_contract"), role=role)
    expected_hashes = {
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": quality["preprocessing"]["sha256"],
        "decoder_contract_sha256": quality["decoder"]["sha256"],
        "nms_contract_sha256": quality["nms"]["sha256"],
        "quality_record_endpoint_contract_sha256": quality["quality_record_endpoint"]["sha256"],
    }
    for name, expected in expected_hashes.items():
        _need(_require_contract_sha256(producer.get(name), role + "." + name) == expected,
              f"{role} Native {name} differs")
    endpoint = producer.get("endpoint_contract")
    endpoint_sha = _require_contract_sha256(producer.get("endpoint_contract_hash"), role + ".endpoint_contract_hash")
    _need(isinstance(endpoint, Mapping) and endpoint.get("endpoint_contract_hash") == endpoint_sha,
          f"{role} Native endpoint binding differs")
    completion = producer.get("candidate_execution_completion_contract")
    _need(isinstance(completion, Mapping), f"{role} Native base completion missing")
    body = dict(completion)
    completion_sha = _require_contract_sha256(body.pop("contract_sha256", ""), role + ".completion.contract_sha256")
    _need(json_fingerprint(body) == completion_sha
          and producer.get("candidate_execution_completion_contract_sha256") == completion_sha
          and completion.get("completed_endpoint_contract") == endpoint,
          f"{role} Native base completion binding differs")
    _need(producer.get("dataset") == quality["dataset"]
          and producer.get("preprocessing") == quality["preprocessing"], f"{role} Native record semantics differ")
    model = producer.get("model") or {}
    _need(model.get("source_onnx_sha256") == quality["model"]["sha256"]
          and producer.get("source_model_sha256") == quality["model"]["sha256"], f"{role} Native source model differs")
    count = producer.get("record_count")
    _need(type(count) is int and count > 0 and producer.get("completed_frames") == count
          and quality["dataset"].get("image_count") == count, f"{role} Native completion count differs")
    _require_contract_sha256(producer.get("candidate_records_sha256"), role + ".candidate_records_sha256")
    _descriptor(producer.get("completion_evidence"), role + ".completion_evidence")
    _descriptor(producer.get("timed_fps_authority"), role + ".timed_fps_authority")
    sources = producer.get("implementation_source_files")
    _need(isinstance(sources, list) and bool(sources), f"{role} Native source closure missing")
    source_paths = set()
    for index, source in enumerate(sources):
        _descriptor(source, f"{role}.implementation_source_files[{index}]")
        relative = Path(str(source.get("relative_path") or ""))
        _need(not relative.is_absolute() and relative.parts and relative.parts[0] in {"onnx_splitpoint_tool", "scripts"}
              and ".." not in relative.parts, f"{role} Native source relative path invalid")
        _need(relative.as_posix() not in source_paths, f"{role} duplicate Native source path")
        source_paths.add(relative.as_posix())
        root = _PUBLISHED_PRODUCER_ROOT.get() or Path(__file__).resolve().parents[1]
        actual = root / relative
        _need(actual.resolve().is_relative_to(root) and actual.is_file(), f"{role} Native source missing")
        encoded = actual.read_bytes()
        _need(len(encoded) == source["size_bytes"] and hashlib.sha256(encoded).hexdigest() == source["sha256"],
              f"{role} Native source differs from the consuming frozen Product")
    _need(required_native_dataset_source_paths(producer["backend"]) <= source_paths,
          f"{role} Native executed source closure incomplete")
    return producer, declared


def validate_native_dataset_completion_evidence(producer, request_path, candidate_records):
    """Check existing FIFO completions against the exported detections, no NMS."""
    from .native_detection_postprocess import verify_detection_completion_execution_attestation, verify_detection_completion_execution_contract
    from .native_three_stage import verify_fast_completion_attestation
    from .quality_cache import indexed_prediction_records, json_fingerprint
    from .quality_service import _load_quality_artifact, _require_contract_sha256
    _need(json_fingerprint(candidate_records) == producer["candidate_records_sha256"], "Native candidate records SHA differs")
    evidence, _, _ = _load_quality_artifact(producer["completion_evidence"], request_path, verify=True)
    _need(evidence.get("schema") == "onnx-splitpoint/native-dataset-quality-completion-evidence"
          and evidence.get("schema_version") == 1 and evidence.get("additional_nms_applied") is False,
          "Native completion evidence schema/role invalid")
    rows = evidence.get("records")
    _need(isinstance(rows, list) and evidence.get("record_count") == len(rows) == producer["record_count"],
          "Native completion evidence count differs")
    indexed = indexed_prediction_records(rows)
    candidates = indexed_prediction_records(candidate_records)
    _need(indexed.keys() == candidates.keys(), "Native completion evidence image IDs differ")
    base = verify_detection_completion_execution_contract(producer["candidate_execution_completion_contract"])
    expected_preprocessing = producer["quality_contract"]["preprocessing"]["identity"]
    selection = producer["prior_artifact_selection"]["preselection"]
    for key, row in indexed.items():
        image_sha = _require_contract_sha256(row.get("image_sha256"), "Native image_sha256")
        for name in ("prepared_input_snapshot", "timing_input_binding", "boundary_proof"):
            _need(isinstance(row.get(name), Mapping) and bool(row[name]), "Native " + name + " missing")
        timing = row["timing_input_binding"]
        _need(timing.get("input_image_sha256") == image_sha, "Native prepared input belongs to another image")
        _need(isinstance(timing.get("preprocessing_contract"), Mapping)
              and json_fingerprint(timing["preprocessing_contract"]) == timing.get("preprocessing_contract_sha256"),
              "Native executed preprocessing contract SHA differs")
        inputs = row["prepared_input_snapshot"].get("inputs")
        _need(isinstance(inputs, list) and bool(inputs), "Native prepared input snapshot missing")
        for prepared in inputs:
            _need(isinstance(prepared, Mapping) and bool(prepared.get("name")) and bool(prepared.get("dtype"))
                  and isinstance(prepared.get("shape"), list) and bool(prepared["shape"])
                  and all(type(dim) is int and dim > 0 for dim in prepared["shape"]), "Native prepared input snapshot invalid")
            _require_contract_sha256(prepared.get("sha256"), "Native prepared input SHA")
        contract, attestation = row.get("execution_contract"), row.get("execution_attestation")
        _need(isinstance(contract, Mapping) and isinstance(attestation, Mapping), "Native completion attestation missing")
        contract = verify_detection_completion_execution_contract(contract)
        _need(contract.get("source_endpoint") == base.get("source_endpoint"), "Native completion source endpoint differs from timed authority")
        for name in ("completion_mode", "model_id", "model_family", "execution_policy"):
            _need(contract.get(name) == base.get(name), "Native completion policy differs: " + name)
        processor = contract["processor_contract"]
        for name in ("confidence_threshold", "iou_threshold", "max_detections", "input_hw", "source_output_format",
                     "output_coordinate_space", "host_nms_applied", "source_nms_attested"):
            _need(processor.get(name) == base["processor_contract"].get(name), "Native processor policy differs: " + name)
        executed = timing["preprocessing_contract"]
        expected_hw = expected_preprocessing.get("target_hw", expected_preprocessing.get("input_hw"))
        _need(executed.get("target_hw") == processor.get("input_hw") == expected_hw,
              "Native executed preprocessing target differs")
        mode = str(selection.get("preprocess_mode") or "").removesuffix("_rgb_uint8")
        _need(executed.get("preprocess_mode") == mode and executed.get("color_space") == "RGB",
              "Native executed preprocessing mode differs")
        if mode == "letterbox":
            _need(executed.get("pad_value") == selection.get("letterbox_pad_value"), "Native executed preprocessing pad differs")
        for name in ("resize_interpolation", "resize_rounding", "placement", "spatial_transform", "image_scale", "letterbox"):
            if name in expected_preprocessing:
                _need(executed.get(name) == expected_preprocessing[name], "Native/reference spatial preprocessing differs: " + name)
        if attestation.get("observation_relation") == "same_hotloop_sentinel":
            verified = verify_detection_completion_execution_attestation(
                attestation, execution_contract=contract, expected_observation_relation="same_hotloop_sentinel")
        else:
            verified = verify_fast_completion_attestation(attestation, execution_contract=contract)
        _need(verified.get("completed_work_units") == 1 and verified.get("completion_count") == 1,
              "Native per-image completion count differs")
        _need((verified.get("last_result") or {}).get("detections") == candidates[key].get("candidate"),
              "Native candidate differs from actual completion detections")


@dataclass(frozen=True)
class NativeQualityRequest:
    request: Any
    request_path: Path
    run_dir: Path
    model_id: str
    manifest: Mapping[str, Any]
    identity: Mapping[str, Any]
    reference_status: Mapping[str, Any]
    reference_store_manifest: str
    producer_source: Mapping[str, Any] | None = None


def load_native_quality_request(request_path, *, run_dir, model_id, reference_run_dir,
                                reference_artifact, reference_status,
                                expected_reference_source_contract_sha256,
                                expected_metric_gate_config, expected_policy_sha256,
                                expected_seed=20260710, expected_confidence_level=.95,
                                producer_source_freeze=None):
    """Load a new Native capture with a genuinely reused canonical CPU artifact."""
    from .quality_service import quality_request_from_manifest
    from .workflow.runner import EvaluationWorkflowRunner as Runner
    from .workflow.artifacts import sha256_file
    root, path = Path(run_dir).resolve(), Path(request_path).resolve()
    _need(path.is_relative_to(root), "Native request escapes its actual collection run")
    manifest = json.loads(path.read_text())
    _need(manifest.get("model_id") == model_id, "Native request model id differs")
    _need((manifest.get("producer_identity") or {}).get("schema") == NATIVE_DATASET_PRODUCER_SCHEMA,
          "Native consumer requires the actual Native producer")
    _need(manifest.get("metric_gate_config") == expected_metric_gate_config
          and manifest.get("policy_sha256") == expected_policy_sha256, "Native request differs from original quality policy")
    records, status, encoded = Runner._read_management_reference_status(
        run_dir=Path(reference_run_dir), model_id=model_id, raw_status=reference_status,
        expected_source_contract=expected_reference_source_contract_sha256)
    bound = {"path": status["reference_path"], "sha256": status["reference_sha256"], "size_bytes": status["reference_size_bytes"]}
    _need(all(reference_artifact.get(name) == value for name, value in bound.items()), "Native CPU reference descriptor differs from normal status")
    with published_native_producer_source_scope(producer_source_freeze) as producer_source:
        request = quality_request_from_manifest(path, reference_artifact=bound, reference_artifact_bytes=encoded)
        identity = Runner._quality_request_identity(path, model_id=model_id, variant=manifest.get("variant", ""), manifest=manifest)
    _need(len(request.candidate_records) == 5000 and request.repetitions == 1000,
          "Native dataset quality requires N5000/B1000")
    _need(request.seed == expected_seed and request.confidence_level == expected_confidence_level,
          "Native request seed/confidence differs from original quality policy")
    _need(identity.get("identity_valid") is True and identity.get("producer_identity_validated") is True,
          "Native request identity invalid: " + str(identity.get("identity_errors")))
    context = SimpleNamespace(run_dir=root, _central_quality_reference_store=None, dataset_registry_binding={})
    reference_identity, stored = Runner._central_reference_contract(context, model_id, records, status)
    request = replace(request, reference_identity=reference_identity.fingerprint(),
                      request_id=f"{model_id}:{request.request_id}:{sha256_file(path)}")
    return NativeQualityRequest(request, path, root, model_id, manifest, identity, status, str(stored.get("manifest_path") or ""), producer_source)


def evaluate_native_quality_request(loaded: NativeQualityRequest, *, service):
    """Evaluate through the caller's one shared service and normal enrichment."""
    from .quality_lifecycle import completed_outcome
    from .quality_result_enrichment import enrich_completed_quality_result
    from .workflow.runner import EvaluationWorkflowRunner as Runner
    result = completed_outcome(service.evaluate(loaded.request))
    case_id, source_run = Runner._quality_request_scope(loaded.request_path)
    result = enrich_completed_quality_result(result, request_identity=loaded.identity,
        manifest=loaded.manifest, model_id=loaded.model_id, request_path=loaded.request_path,
        run_dir=loaded.run_dir, collection_eval_run_id=loaded.run_dir.name, case_id=case_id,
        run_id=source_run, reference_status=loaded.reference_status,
        reference_store_manifest=loaded.reference_store_manifest)
    if loaded.producer_source is not None:
        result["native_dataset_producer_source"] = deepcopy(dict(loaded.producer_source))
    return result
