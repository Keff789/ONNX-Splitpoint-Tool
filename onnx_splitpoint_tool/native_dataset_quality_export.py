"""Portable normal Quality envelopes for a fresh Native dataset capture.

Inference, completion and central evaluation belong to their existing runners.
This exporter binds their already completed records without emitting performance
measurements or running another detector postprocessor.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .native_command_contract import canonical_json_sha256
from .native_split_quality import validate_native_split_quality_binding
from .quality_cache import canonical_image_id, image_ids_fingerprint
from .quality_service import _validate_detection_quality_contract


_COUNT = 5000
_HASH_FIELDS = {
    "preprocessing_contract_sha256": "preprocessing",
    "decoder_contract_sha256": "decoder",
    "nms_contract_sha256": "nms",
    "quality_record_endpoint_contract_sha256": "quality_record_endpoint",
}
_PERFORMANCE_FIELDS = {"fps", "fps_makespan", "paper_equivalent_fps", "e2e_fps", "fps_mean",
                       "p1_ms", "p2_ms", "p2_run_ms", "energy_j", "power_w"}


def _need(value: Any, message: str) -> None:
    if not value:
        raise ValueError("native_dataset_quality_export:" + message)


def _sha256(value: Any, name: str) -> str:
    _need(isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), name + "_sha256_invalid")
    return value


def _descriptor(value: Any, name: str) -> dict[str, Any]:
    _need(isinstance(value, Mapping), name + "_descriptor_missing")
    _need(isinstance(value.get("path"), str) and bool(value["path"].strip()), name + "_path_missing")
    _sha256(value.get("sha256"), name)
    _need(type(value.get("size_bytes")) is int and value["size_bytes"] > 0, name + "_size_invalid")
    return deepcopy(dict(value))


def _no_performance(value: Any) -> None:
    if isinstance(value, Mapping):
        _need(not (_PERFORMANCE_FIELDS & set(value)), "performance_metrics_forbidden")
        _need(value.get("performance_claims_emitted") is not True, "performance_claims_forbidden")
        for name, item in value.items():
            # These are unchanged descriptions of the prior authoritative run.
            if name not in {"prior_artifact_selection", "timed_fps_authority"}:
                _no_performance(item)
    elif isinstance(value, list):
        for item in value:
            _no_performance(item)


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _file_descriptor(path: str, encoded: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": hashlib.sha256(encoded).hexdigest(), "size_bytes": len(encoded)}


def _write_fresh(path: Path, encoded: bytes) -> None:
    """Publish a complete file atomically, with no overwrite even under a race."""
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_native_quality_request(
    config: dict[str, Any], records: list[Any], completion_records: list[Any],
    implementation_source_files: list[Any], out_dir: Path,
) -> dict[str, Any]:
    """Export one N5000/B1000 Native candidate; return its request descriptor."""
    _need(isinstance(config, Mapping), "config_missing")
    _no_performance(config)
    _no_performance(records)
    _no_performance(completion_records)
    _need(isinstance(records, list) and len(records) == _COUNT, "exactly_5000_candidate_records_required")
    _need(isinstance(completion_records, list) and len(completion_records) == _COUNT, "exactly_5000_completion_records_required")
    identity_raw = config.get("identity")
    _need(isinstance(identity_raw, Mapping), "identity_missing")
    identity: dict[str, str] = {}
    for name in ("eval_run_id", "model_id", "case_id", "setup_id", "source_run_id", "backend", "runtime_precision_identity"):
        value = identity_raw.get(name)
        _need(isinstance(value, str) and bool(value) and value.strip() == value, "identity_" + name + "_invalid")
        identity[name] = value
    _need(identity["source_run_id"] == identity["backend"], "backend_identity_mismatch")
    quality, quality_sha = _validate_detection_quality_contract(config.get("quality_contract"), role="Native export")
    if "quality_contract_sha256" in config:
        _need(config["quality_contract_sha256"] == quality_sha, "quality_contract_hash_mismatch")
    _need(quality["dataset"].get("image_count") == _COUNT, "quality_contract_image_count_mismatch")
    source_model_sha = _sha256(config.get("source_model_sha256"), "source_model")
    _need(source_model_sha == quality["model"]["sha256"], "source_model_mismatch")
    prior_raw = config.get("prior_artifact_selection")
    prior, status = validate_native_split_quality_binding(prior_raw, expected_identity={
        "model": identity["model_id"], "case": identity["case_id"], "setup_id": identity["setup_id"],
        "backend": identity["backend"], "task": "detection", "precision": identity["runtime_precision_identity"],
    })
    _need(prior is not None, "prior_artifact_selection_invalid:" + status)
    _need(identity["eval_run_id"] != prior["eval_run_id"], "fresh_evaluation_id_required")
    _need(len(prior["artifacts"]) == 8, "exactly_eight_prior_artifacts_required")
    statistics = config.get("statistics")
    _need(isinstance(statistics, Mapping) and type(statistics.get("bootstrap_repetitions")) is int
          and statistics["bootstrap_repetitions"] == 1000 and type(statistics.get("seed")) is int
          and statistics["seed"] == 20260710, "B1000_seed20260710_required")
    _need(isinstance(config.get("metric_gate_config"), Mapping), "metric_gate_config_missing")
    _sha256(config.get("policy_sha256"), "policy")
    timed_authority = _descriptor(config.get("timed_fps_authority"), "timed_fps_authority")
    _need(isinstance(implementation_source_files, list) and bool(implementation_source_files), "implementation_sources_missing")
    sources = [_descriptor(value, "implementation_source") for value in implementation_source_files]
    base = config.get("base_completion_contract")
    _need(isinstance(base, Mapping), "base_completion_contract_missing")
    base_identity = {key: value for key, value in base.items() if key != "contract_sha256"}
    base_sha = _sha256(base.get("contract_sha256"), "base_completion_contract")
    _need(canonical_json_sha256(base_identity) == base_sha, "base_completion_contract_hash_mismatch")
    endpoint = base.get("completed_endpoint_contract")
    _need(isinstance(endpoint, Mapping), "completed_endpoint_contract_missing")
    endpoint_sha = _sha256(endpoint.get("endpoint_contract_hash"), "completed_endpoint_contract")

    ids: list[Any] = []
    tokens: list[str] = []
    completion_rows: list[dict[str, Any]] = []
    for candidate, completed in zip(records, completion_records):
        _need(isinstance(candidate, Mapping) and set(candidate) == {"image_id", "candidate", "ground_truth"}, "candidate_record_fields_invalid")
        _need(isinstance(candidate["candidate"], list) and isinstance(candidate["ground_truth"], list), "detection_records_must_be_arrays")
        token = canonical_image_id(candidate["image_id"])
        _need(isinstance(completed, Mapping) and canonical_image_id(completed.get("image_id")) == token, "completion_image_id_alignment_mismatch")
        count = completed.get("native_completed_count", completed.get("completed_native"))
        _need(type(count) is int and count == 1, "exactly_one_native_completion_per_image_required")
        _sha256(completed.get("image_sha256"), "image")
        contract, attestation = completed.get("execution_contract"), completed.get("execution_attestation")
        _need(isinstance(contract, Mapping) and isinstance(attestation, Mapping), "completion_contract_attestation_missing")
        _need(type(attestation.get("completion_count")) is int and attestation["completion_count"] == 1
              and type(attestation.get("completed_work_units")) is int and attestation["completed_work_units"] == 1,
              "completion_attestation_count_mismatch")
        _need((attestation.get("last_result") or {}).get("detections") == candidate["candidate"], "candidate_differs_from_completed_detections")
        snapshot = completed.get("prepared_input_snapshot")
        if isinstance(snapshot, list):
            snapshot = {"inputs": snapshot}
        _need(isinstance(snapshot, Mapping) and bool(snapshot), "prepared_input_snapshot_missing")
        for name in ("timing_input_binding", "boundary_proof"):
            _need(isinstance(completed.get(name), Mapping) and bool(completed[name]), name + "_missing")
        completion_rows.append({name: deepcopy(completed[name]) for name in (
            "image_id", "image_sha256", "execution_contract", "execution_attestation", "timing_input_binding", "boundary_proof",
        )} | {"prepared_input_snapshot": deepcopy(dict(snapshot))})
        ids.append(candidate["image_id"])
        tokens.append(token)
    _need(len(set(tokens)) == _COUNT, "duplicate_image_ids")
    ids_sha = image_ids_fingerprint(ids)
    _need(ids_sha == quality["dataset"]["image_ids_sha256"], "quality_contract_image_ids_mismatch")
    gt = [{"image_id": row["image_id"], "ground_truth": row["ground_truth"]} for row in sorted(records, key=lambda row: canonical_image_id(row["image_id"]))]
    _need(canonical_json_sha256(gt) == quality["dataset"]["ground_truth_sha256"], "quality_contract_ground_truth_mismatch")

    evidence = {"schema": "onnx-splitpoint/native-dataset-quality-completion-evidence", "schema_version": 1,
                "record_count": _COUNT, "records": completion_rows, "additional_nms_applied": False}
    evidence_bytes = _encoded(evidence)
    evidence_descriptor = _file_descriptor("native_completion_evidence.json", evidence_bytes)
    hashes = {name: quality[component]["sha256"] for name, component in _HASH_FIELDS.items()}
    hashes["quality_contract_sha256"] = quality_sha
    producer = {"schema": "onnx-splitpoint/native-dataset-quality-producer", "schema_version": 1,
        "producer_kind": "native_dataset_completed_quality", "task": "detection", "variant": "composed",
        "execution_role": "native_dataset_quality_candidate", "performance_claims_emitted": False,
        **identity, "source_model_sha256": source_model_sha, "model": {"source_onnx_sha256": source_model_sha},
        "dataset": deepcopy(quality["dataset"]), "preprocessing": deepcopy(quality["preprocessing"]),
        "quality_contract": deepcopy(quality), **hashes, "prior_artifact_selection": deepcopy(prior_raw),
        "preselection": deepcopy(prior["preselection"]), "artifacts": deepcopy(prior["artifacts"]),
        "completed_frames": _COUNT, "record_count": _COUNT, "candidate_records_sha256": canonical_json_sha256(records),
        "timed_fps_authority": timed_authority, "implementation_source_files": sources,
        "completion_evidence": evidence_descriptor, "additional_nms_applied": False,
        "endpoint_contract": deepcopy(endpoint), "endpoint_contract_hash": endpoint_sha,
        "candidate_execution_completion_contract": deepcopy(base), "candidate_execution_completion_contract_sha256": base_sha}
    producer["producer_identity_sha256"] = canonical_json_sha256(producer)
    common = {**identity, **hashes, "task": "detection", "variant": "composed", "pairing_key": "image_id",
        "record_count": _COUNT, "execution_role": "native_dataset_quality_candidate", "performance_claims_emitted": False,
        "producer_kind": producer["producer_kind"], "producer_provenance_required": True,
        "producer_identity": producer, "producer_identity_sha256": producer["producer_identity_sha256"],
        "provenance_required": True, "quality_contract": deepcopy(quality),
        "source_model_sha256": source_model_sha, "endpoint_contract": deepcopy(endpoint), "endpoint_contract_hash": endpoint_sha,
        "candidate_execution_completion_contract": deepcopy(base), "candidate_execution_completion_contract_sha256": base_sha,
        "additional_nms_applied": False}
    candidate_bytes = _encoded({"schema": "onnx-splitpoint/task-quality-candidate-input", "schema_version": 1,
                                **common, "records": records})
    candidate_descriptor = _file_descriptor("composed_candidate.json", candidate_bytes)
    reference = {"source": "management_cpu_reference", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "required": True, "expected_image_ids": ids,
        "expected_image_ids_sha256": ids_sha, "record_count": _COUNT, "quality_contract_sha256": quality_sha}
    request_bytes = _encoded({"schema": "onnx-splitpoint/central-quality-evaluation-request", "schema_version": 1,
        **common, "status": "pending_central_evaluation", "execution_location": "management_node",
        "requested_by": "central_management", "reference": reference, "candidate": candidate_descriptor,
        "reference_record_count": _COUNT, "expected_image_ids": ids, "expected_image_ids_sha256": ids_sha,
        "metric_gate_config": deepcopy(dict(config["metric_gate_config"])), "policy_sha256": config["policy_sha256"],
        "statistics": deepcopy(dict(statistics))})
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in ("native_completion_evidence.json", "composed_candidate.json", "composed_request.json")]
    _need(not any(path.exists() or path.is_symlink() for path in paths), "fresh_output_files_required")
    for path, encoded in zip(paths, (evidence_bytes, candidate_bytes, request_bytes)):
        _write_fresh(path, encoded)
    return _file_descriptor(str(paths[-1]), request_bytes)
