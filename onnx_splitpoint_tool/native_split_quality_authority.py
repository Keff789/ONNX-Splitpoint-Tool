"""Authoritative Quality-FIRST policy for Native Split claim boundaries.

Native result rows and Central Quality result rows are evidence, not policy
authorities.  In particular, removing a self-declared Quality-FIRST marker
from both rows must never make a current run eligible for the historical
legacy join.  The immutable EvaluationRun manifest and the Native stage plan
decide whether the Quality-FIRST chain is mandatory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW = (
    "v2.69f-hardware-smoke-native-energy-repair"
)
SELECTION_FINGERPRINT_WORKFLOW = (
    "v2.70d-targeted-console-smoke-native-authority-repair"
)

CANONICAL_NATIVE_SPLIT_BACKENDS = frozenset({
    "hailo8_to_trt",
    "hailo10h_to_trt",
    "deepx_to_trt",
})
NATIVE_SPLIT_BACKEND_ALIASES = {
    "hailo8_to_trt": "hailo8_to_trt",
    "hailo8_to_tensorrt": "hailo8_to_trt",
    "hailo10_to_trt": "hailo10h_to_trt",
    "hailo10h_to_trt": "hailo10h_to_trt",
    "hailo10_to_tensorrt": "hailo10h_to_trt",
    "hailo10h_to_tensorrt": "hailo10h_to_trt",
    "deepx_to_trt": "deepx_to_trt",
    "deepx_to_tensorrt": "deepx_to_trt",
    "deepx_m1_to_trt": "deepx_to_trt",
    "deepx_m1_to_tensorrt": "deepx_to_trt",
}
NATIVE_SPLIT_BACKENDS = frozenset(NATIVE_SPLIT_BACKEND_ALIASES)

_HISTORICAL_WORKFLOW_RE = re.compile(
    r"^(?:"
    r"v(?:5[0-9]|6[01])[-a-z0-9._+]*"
    r"|v2\.(?:6[0-8](?:[a-z])?|69[abcde])[-a-z0-9._+]*"
    r"|benchmark_suite"
    r")$",
    re.IGNORECASE,
)
_CURRENT_OR_LATER_WORKFLOW_RE = re.compile(
    r"^v2\.(?:(?:69f)|(?:69[g-z])|(?:[7-9][0-9]))[-a-z0-9._+]*$",
    re.IGNORECASE,
)


def canonical_native_split_backend(value: Any) -> str:
    """Return the one canonical backend token for every supported alias."""
    token = str(value or "").strip().lower().replace("-", "_")
    return NATIVE_SPLIT_BACKEND_ALIASES.get(token, token)


def is_native_split_backend(value: Any) -> bool:
    """Return whether *value* names a managed Native Split backend."""
    return canonical_native_split_backend(value) in CANONICAL_NATIVE_SPLIT_BACKENDS


def _strict_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    duplicate = False

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except Exception as exc:
        return None, f"{path.name}_unreadable:{type(exc).__name__}"
    if duplicate:
        return None, f"{path.name}_duplicate_json_key"
    if not isinstance(value, dict):
        return None, f"{path.name}_root_not_object"
    return value, "ok"


def _effective_manifest_workflow(manifest: Mapping[str, Any]) -> tuple[str, list[str]]:
    errors: list[str] = []
    sessions = [
        row for row in list(manifest.get("execution_sessions") or [])
        if isinstance(row, Mapping)
    ]
    session_workflow = str(
        sessions[-1].get("workflow_version") if sessions else ""
    ).strip()
    current_workflow = str(manifest.get("current_workflow_version") or "").strip()
    creation_workflow = str(manifest.get("workflow_version") or "").strip()
    if current_workflow and session_workflow and current_workflow != session_workflow:
        errors.append("manifest_current_and_session_workflow_mismatch")
    effective = current_workflow or session_workflow or creation_workflow
    if not effective:
        errors.append("manifest_workflow_version_missing")
    return effective, errors


def _workflow_mode(workflow_version: str) -> str:
    normalized = str(workflow_version or "").strip()
    if normalized == CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW:
        return "required"
    if _CURRENT_OR_LATER_WORKFLOW_RE.fullmatch(normalized):
        return "required"
    if _HISTORICAL_WORKFLOW_RE.fullmatch(normalized):
        return "legacy"
    return "invalid"


def _requires_explicit_selection_fingerprint(workflow_version: str) -> bool:
    """Return whether the Native stage must copy the canonical fingerprint.

    v2.70c already records the fingerprint in the immutable start snapshot,
    but its Native stage only copied the requested-selection snapshot SHA.
    Keep that archived run readable while requiring the explicit field from
    v2.70d onward.
    """
    normalized = str(workflow_version or "").strip().lower()
    if normalized == SELECTION_FINGERPRINT_WORKFLOW:
        return True
    match = re.match(r"^v2\.(\d+)([a-z]?)", normalized)
    if not match:
        return False
    minor = int(match.group(1))
    suffix = match.group(2)
    return minor > 70 or (minor == 70 and suffix >= "d")


def _sha256_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""


def resolve_native_split_quality_authority(
    *,
    run_manifest_path: Path | str,
    stage_path: Path | str,
) -> dict[str, Any]:
    """Resolve the run-wide Native Split Quality-FIRST requirement.

    Unknown or malformed context deliberately resolves to ``required`` with an
    invalid authority status.  Callers therefore fail closed for split rows;
    only an explicitly version-bound historical manifest can enter legacy
    compatibility mode.
    """
    manifest_path = Path(run_manifest_path).expanduser().resolve()
    native_stage_path = Path(stage_path).expanduser().resolve()
    errors: list[str] = []

    manifest, manifest_status = _strict_json_object(manifest_path)
    if manifest is None:
        errors.append(manifest_status)
        workflow_version = ""
        run_id = ""
        mode = "invalid"
    else:
        if manifest.get("schema") != "onnx-splitpoint/evaluation-run-manifest":
            errors.append("run_manifest_schema_invalid")
        workflow_version, workflow_errors = _effective_manifest_workflow(manifest)
        errors.extend(workflow_errors)
        run_id = str(manifest.get("run_id") or "").strip()
        if not run_id:
            errors.append("run_manifest_run_id_missing")
        mode = _workflow_mode(workflow_version)
        if mode == "invalid":
            errors.append("run_manifest_workflow_version_unknown")
    manifest_snapshot = (
        manifest.get("profile_start_snapshot")
        if isinstance(manifest, Mapping)
        and isinstance(manifest.get("profile_start_snapshot"), Mapping)
        else {}
    )
    profile_start_snapshot_sha256 = _sha256_token(
        manifest_snapshot.get("snapshot_sha256")
    )
    requested_selection = (
        manifest_snapshot.get("requested_selection")
        if isinstance(manifest_snapshot.get("requested_selection"), Mapping)
        else {}
    )
    resolved_selection = (
        manifest_snapshot.get("resolved_selection")
        if isinstance(manifest_snapshot.get("resolved_selection"), Mapping)
        else {}
    )
    requested_selection_sha = _sha256_token(
        requested_selection.get("selection_snapshot_sha256")
        or requested_selection.get("snapshot_sha256")
    )
    resolved_selection_sha = _sha256_token(
        resolved_selection.get("selection_snapshot_sha256")
        or resolved_selection.get("snapshot_sha256")
    )
    profile_selection_fingerprint = _sha256_token(
        manifest_snapshot.get("selection_fingerprint")
    )
    selection_identity_mode = (
        "canonical_selection_fingerprint"
        if profile_selection_fingerprint
        else "legacy_equal_selection_snapshots"
    )
    if mode == "required":
        if not profile_start_snapshot_sha256:
            errors.append("run_manifest_profile_snapshot_sha256_missing_or_invalid")
        if not requested_selection_sha or not resolved_selection_sha:
            errors.append("run_manifest_selection_snapshot_sha256_missing_or_invalid")
        elif (
            not profile_selection_fingerprint
            and requested_selection_sha != resolved_selection_sha
        ):
            errors.append("run_manifest_selection_snapshot_sha256_mismatch")
        if (
            _requires_explicit_selection_fingerprint(workflow_version)
            and not profile_selection_fingerprint
        ):
            errors.append("run_manifest_selection_fingerprint_missing_or_invalid")
    tool_version = str(
        (manifest or {}).get("current_tool_version")
        or (manifest or {}).get("tool_version")
        or ""
    ).strip() if isinstance(manifest, Mapping) else ""

    stage, stage_status = _strict_json_object(native_stage_path)
    stage_required: bool | None = None
    stage_applicable: bool | None = None
    stage_run_id = ""
    if stage is None:
        if mode == "required":
            errors.append(stage_status)
    else:
        if stage.get("schema") not in {
            "onnx-splitpoint/native-producer-stage",
            "onnx-splitpoint/native-producer-variant-stage",
        }:
            errors.append("native_stage_schema_invalid")
        stage_run_id = str(stage.get("run_id") or "").strip()
        if not stage_run_id:
            eval_run_dir = str(stage.get("eval_run_dir") or "").strip()
            stage_run_id = Path(eval_run_dir).name if eval_run_dir else ""
        if run_id and stage_run_id and stage_run_id != run_id:
            errors.append("native_stage_run_id_mismatch")
        stage_workflow = str(stage.get("workflow_version") or "").strip()
        if mode == "required" and not stage_workflow:
            errors.append("native_stage_workflow_version_missing")
        elif stage_workflow and workflow_version and stage_workflow != workflow_version:
            errors.append("native_stage_workflow_version_mismatch")
        manifest_snapshot_sha = profile_start_snapshot_sha256
        stage_snapshot_sha = _sha256_token(
            stage.get("profile_start_snapshot_sha256")
        )
        if mode == "required" and manifest_snapshot_sha:
            if not stage_snapshot_sha:
                errors.append("native_stage_profile_snapshot_sha256_missing")
            elif stage_snapshot_sha != manifest_snapshot_sha:
                errors.append("native_stage_profile_snapshot_sha256_mismatch")
        stage_selection_sha = _sha256_token(
            stage.get("profile_selection_snapshot_sha256")
        )
        if mode == "required" and requested_selection_sha:
            if not stage_selection_sha:
                errors.append("native_stage_selection_snapshot_sha256_missing")
            elif stage_selection_sha != requested_selection_sha:
                errors.append("native_stage_selection_snapshot_sha256_mismatch")
        stage_selection_fingerprint = _sha256_token(
            stage.get("profile_selection_fingerprint")
        )
        if mode == "required" and profile_selection_fingerprint:
            if not stage_selection_fingerprint:
                if _requires_explicit_selection_fingerprint(workflow_version):
                    errors.append("native_stage_selection_fingerprint_missing")
            elif stage_selection_fingerprint != profile_selection_fingerprint:
                errors.append("native_stage_selection_fingerprint_mismatch")
        split_plan = stage.get("native_split_quality_first")
        if isinstance(split_plan, Mapping):
            required_value = split_plan.get("required")
            if isinstance(required_value, bool):
                stage_required = required_value
            else:
                errors.append("native_stage_split_quality_required_not_boolean")
            # ``required`` is the workflow policy.  ``applicable`` records
            # whether the sealed Native matrix actually contains Split rows.
            # Full-only campaigns keep the policy enabled, but have nothing
            # to bind and therefore must not be reported as an invalid legacy
            # or missing-Quality run.
            applicable_value = split_plan.get("applicable", True)
            if isinstance(applicable_value, bool):
                stage_applicable = applicable_value
            else:
                errors.append(
                    "native_stage_split_quality_applicable_not_boolean"
                )
        elif mode == "required":
            errors.append("native_stage_split_quality_plan_missing")

    # v2.69f and every later workflow requires QF for every managed split row.
    # The stage plan is an independent cross-check, never a downgrade switch.
    if mode == "required" and stage_required is not True:
        errors.append("native_stage_split_quality_not_required")

    required = mode != "legacy"
    valid = bool(mode in {"required", "legacy"} and not errors)
    return {
        "schema": "onnx-splitpoint/native-split-quality-authority",
        "schema_version": 1,
        "mode": mode,
        "valid": valid,
        "native_split_quality_required": required,
        "workflow_version": workflow_version,
        "tool_version": tool_version,
        "run_id": run_id,
        "profile_start_snapshot_sha256": profile_start_snapshot_sha256,
        # Compatibility name retained as the requested-selection snapshot SHA.
        # Requested and resolved SHAs legitimately differ when follow_tool_config
        # resolves a newer central run-mode definition.
        "profile_selection_snapshot_sha256": requested_selection_sha,
        "profile_requested_selection_snapshot_sha256": requested_selection_sha,
        "profile_resolved_selection_snapshot_sha256": resolved_selection_sha,
        "profile_selection_fingerprint": profile_selection_fingerprint,
        "selection_identity_mode": selection_identity_mode,
        "stage_run_id": stage_run_id,
        "stage_required": stage_required,
        "stage_applicable": stage_applicable,
        "native_split_quality_applicable": stage_applicable is not False,
        "run_manifest": str(manifest_path),
        "native_stage": str(native_stage_path),
        "errors": list(dict.fromkeys(errors)),
    }


def native_split_quality_required_for_row(
    row: Mapping[str, Any], authority: Mapping[str, Any] | None,
) -> bool:
    """Return the authoritative requirement for one row.

    A missing authority is unknown, not legacy.  Only a valid ``legacy``
    authority may disable the current Quality-FIRST chain.
    """
    if not is_native_split_backend(row.get("backend")):
        return False
    if not isinstance(authority, Mapping):
        return True
    return not bool(
        authority.get("valid") is True
        and str(authority.get("mode") or "") == "legacy"
        and authority.get("native_split_quality_required") is False
    )


def apply_native_split_quality_authority(
    row: dict[str, Any], authority: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach authority diagnostics and force the split requirement if needed."""
    if not is_native_split_backend(row.get("backend")):
        return row
    required = native_split_quality_required_for_row(row, authority)
    context = dict(authority or {})
    row["native_split_quality_authority"] = context
    row["native_split_quality_authority_valid"] = bool(context.get("valid"))
    row["native_split_quality_authority_mode"] = str(
        context.get("mode") or "invalid"
    )
    row["native_split_quality_authority_workflow_version"] = str(
        context.get("workflow_version") or ""
    )
    errors = list(context.get("errors") or [])
    if not context:
        errors = ["native_split_quality_authority_missing"]
    previous = list(row.get("native_split_quality_authority_errors") or [])
    if previous and previous != errors:
        row["native_split_quality_authority_previous_errors"] = previous
    row["native_split_quality_authority_errors"] = errors
    if required:
        row["native_split_quality_required"] = True
        row["native_split_quality_binding_required"] = True
    elif context.get("valid") is True and context.get("mode") == "legacy":
        row["native_split_quality_legacy_status"] = "historical_diagnostic_only"
        row["execution_role"] = str(
            row.get("execution_role") or "legacy_manual_diagnostic"
        )
        row["performance_claims_emitted"] = False
        row["status"] = "historical_diagnostic_only"
        row["ranking_eligible"] = False
        row["performance_eligible"] = False
        row["energy_eligible"] = False
        row["thesis_valid"] = False
    return row


NATIVE_DATASET_ENERGY_SCHEMA = "onnx-splitpoint/native-dataset-energy-quality-authority"


class _VerifiedNativeDatasetQuality(dict):
    """In-process result of the verifier; a JSON row cannot declare this type."""


def is_verified_native_dataset_quality(value: Any) -> bool:
    return isinstance(value, _VerifiedNativeDatasetQuality)


def _dataset_bound_json(descriptor: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    """Read an existing published artifact; never create an authority file."""
    import hashlib
    path = Path(str(descriptor.get("path") or "")).expanduser().resolve(strict=True)
    encoded = path.read_bytes()
    if (not _sha256_token(descriptor.get("sha256"))
            or hashlib.sha256(encoded).hexdigest() != descriptor["sha256"]
            or descriptor.get("size_bytes", len(encoded)) != len(encoded)):
        raise ValueError("native_dataset_energy_artifact_changed")
    value, status = _strict_json_object(path)
    if value is None:
        raise ValueError(status)
    return value, path


def _dataset_descriptor(path: Path | str) -> dict[str, Any]:
    import hashlib
    path = Path(path).expanduser().resolve(strict=True)
    encoded = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(), "size_bytes": len(encoded)}


def native_dataset_energy_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (canonical_native_split_backend(row.get("backend")),
            str(row.get("model_id") or row.get("model") or ""),
            str(row.get("case") or row.get("case_id") or ""),
            str(row.get("setup_id") or row.get("source_setup_id") or ""),
            str(row.get("precision") or row.get("runtime_precision_identity") or ""))


def native_dataset_energy_inputs(paths: list[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    """Index actual normal completed-case receipts without a legacy manifest."""
    indexed = {}
    for path in paths:
        descriptor = _dataset_descriptor(path)
        case, _ = _dataset_bound_json(descriptor)
        _native_dataset_case_closed(case)
        result, _ = _dataset_bound_json(case["normal_result"])
        if (case.get("normal_native_admission") != "completed_verified_actual_native_producer"
                or result.get("technical_status") != "completed"
                or result.get("status") != "completed"
                or result.get("producer_binding_eligible") is not True):
            raise ValueError("native_dataset_energy_quality_not_completed")
        identity = native_dataset_energy_identity(result)
        if not all(identity) or identity in indexed:
            raise ValueError("native_dataset_energy_identity_missing_or_ambiguous")
        indexed[identity] = descriptor
    return indexed


def _native_dataset_case_closed(case: Mapping[str, Any]) -> None:
    shutdown, _ = _dataset_bound_json(case["normal_service_shutdown"])
    terminal_descriptor = case.get("normal_source_epoch_terminal") or case.get("normal_terminal")
    if not isinstance(terminal_descriptor, Mapping):
        raise ValueError("native_dataset_energy_service_terminal_missing")
    terminal, _ = _dataset_bound_json(terminal_descriptor)
    completed = terminal.get("completed_case_keys", terminal.get("completed_cases", []))
    if (case.get("shared_pool_terminal_pending") is not False
            or shutdown.get("finished") is not True or shutdown.get("admission_closed") is not True
            or shutdown.get("manager_alive") is not False or shutdown.get("dispatcher_alive") is not False
            or shutdown.get("queue_size") != 0 or shutdown.get("inflight_keys") != [] or shutdown.get("group_keys") != []
            or not shutdown.get("workers") or any(worker.get("alive") is not False or worker.get("exitcode") != 0
                for worker in shutdown["workers"])
            or case.get("case_key") not in completed):
        raise ValueError("native_dataset_energy_service_not_closed_for_this_case")


def _dataset_completion_same_operation(actual: Mapping[str, Any], timed: Mapping[str, Any]) -> None:
    from .native_detection_postprocess import verify_detection_completion_execution_contract
    actual = verify_detection_completion_execution_contract(actual)
    timed = verify_detection_completion_execution_contract(timed)
    for name in ("task", "model_id", "model_family", "completion_mode", "execution_policy", "yolov7_head_mapping"):
        if actual.get(name) != timed.get(name):
            raise ValueError("native_dataset_energy_completion_policy_drift:" + name)
    geometry = {"contract_sha256", "implementation_artifacts", "original_wh", "letterbox_geometry_contract",
                "letterbox_geometry_contract_sha256", "source_output_endpoint_attestation_sha256"}
    semantic = lambda value: {key: item for key, item in value.items() if key not in geometry}
    endpoint = lambda value: {key: item for key, item in value.items() if key != "output_endpoint_attestation_sha256"}
    if (semantic(actual["processor_contract"]) != semantic(timed["processor_contract"])
            or endpoint(actual["source_endpoint"]) != endpoint(timed["source_endpoint"])):
        raise ValueError("native_dataset_energy_processor_or_source_endpoint_drift")


def resolve_native_dataset_energy_quality(
    descriptor: Mapping[str, Any], row: Mapping[str, Any], command: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate fresh Native N5000 data and bind it to the measured command.

    Existing request/producer/completion validators own scientific provenance.
    No inference, AP/bootstrap evaluation, or legacy workflow fabrication occurs.
    """
    from copy import deepcopy
    from .native_command_contract import canonical_json_sha256, verify_native_energy_command_contract
    from .quality_service import quality_request_from_manifest, _quality_artifact_path
    from .native_dataset_quality_consumer import published_native_producer_source_scope
    from .workflow.runner import EvaluationWorkflowRunner
    case, _ = _dataset_bound_json(descriptor)
    _native_dataset_case_closed(case)
    result, _ = _dataset_bound_json(case["normal_result"])
    manifest, request_path = _dataset_bound_json(case["source_request"])
    for role, declared in (("candidate", manifest["candidate"]),
                           ("completion_evidence", manifest["producer_identity"]["completion_evidence"])):
        if (case[role]["sha256"] != declared["sha256"]
                or case[role]["size_bytes"] != declared["size_bytes"]
                or Path(case[role]["path"]).resolve() != _quality_artifact_path(declared, request_path).resolve()):
            raise ValueError("native_dataset_energy_capture_descriptor_drift:" + role)
    for role in ("producer_source_freeze", "consumer_source_freeze"):
        _dataset_bound_json(case[role])
    if (case.get("normal_native_admission") != "completed_verified_actual_native_producer"
            or result.get("status") != "completed" or result.get("technical_status") != "completed"
            or result.get("producer_binding_eligible") is not True
            or result.get("execution_location") != "central_management"
            or result.get("decision") not in {"pass", "fail", "inconclusive", "reference_close", "accuracy_loss", "not_estimable"}):
        raise ValueError("native_dataset_energy_quality_not_completed")
    reference = result.get("management_cpu_reference") or {}
    reference_descriptor = {"path": reference.get("reference_path"), "sha256": reference.get("reference_sha256"),
                            "size_bytes": reference.get("reference_size_bytes")}
    with published_native_producer_source_scope(case["producer_source_freeze"]) as producer_source:
        if producer_source != result.get("native_dataset_producer_source"):
            raise ValueError("native_dataset_energy_published_producer_source_drift")
        request = quality_request_from_manifest(request_path, reference_artifact=reference_descriptor)
        identity = EvaluationWorkflowRunner._quality_request_identity(request_path,
            model_id=manifest["model_id"], variant="composed", manifest=manifest)
    if (len(request.candidate_records) != 5000 or request.repetitions != 1000
            or request.seed != 20260710 or request.confidence_level != .95):
        raise ValueError("native_dataset_energy_quality_population_or_statistics_drift")
    producer = manifest.get("producer_identity") or {}
    handoff, _ = _dataset_bound_json(case["actual_capture_handoff"])
    capture_config, _ = _dataset_bound_json(handoff["config"])
    terminal, _ = _dataset_bound_json(handoff["terminal"])
    if (handoff.get("request") != case["source_request"]
            or handoff.get("completed_images") != 5000
            or terminal.get("status") != "completed" or terminal.get("capture_completed") is not True
            or terminal.get("capture_returncode") != 0 or terminal.get("collection_returncode") != 0
            or terminal.get("completed_images") != 5000
            or terminal.get("request") != case["source_request"]
            or terminal.get("producer_identity_sha256") != producer.get("producer_identity_sha256")):
        raise ValueError("native_dataset_energy_capture_not_closed")
    recorded = result.get("request_identity") or {}
    if (identity.get("identity_valid") is not True or identity.get("producer_identity_validated") is not True
            or producer.get("schema") != "onnx-splitpoint/native-dataset-quality-producer"
            or result.get("producer_identity") != producer
            or recorded.get("producer_identity") != producer
            or recorded.get("identity_valid") is not True or recorded.get("producer_identity_validated") is not True
            or recorded.get("producer_binding_eligible") is not True
            or native_dataset_energy_identity(producer) != native_dataset_energy_identity(row)
            or native_dataset_energy_identity(result) != native_dataset_energy_identity(row)):
        raise ValueError("native_dataset_energy_producer_or_identity_drift")
    for field in ("source_request_sha256", "eval_run_id", "model_id", "case_id", "setup_id", "runtime_precision_identity",
                  "endpoint_contract_hash", "quality_contract_sha256", "preprocessing_contract_sha256",
                  "decoder_contract_sha256", "nms_contract_sha256", "source_model_sha256"):
        if not identity.get(field) or recorded.get(field) != identity[field]:
            raise ValueError("native_dataset_energy_result_identity_drift:" + field)
    if (_sha256_token(result.get("source_request_sha256")) != case["source_request"]["sha256"]
            or result.get("source_request_sha256") != identity["source_request_sha256"]
            or result.get("schema") != "onnx-splitpoint/management-paired-quality-result"
            or result.get("schema_version") != 1
            or manifest["candidate"]["sha256"] != case["candidate"]["sha256"]
            or producer["completion_evidence"]["sha256"] != case["completion_evidence"]["sha256"]
            or result.get("policy_sha256") != manifest.get("policy_sha256")
            or result.get("metric_gate_config") != manifest.get("metric_gate_config")
            or result.get("primary", {}).get("n") != 5000
            or result.get("primary", {}).get("bootstrap_repetitions") != 1000):
        raise ValueError("native_dataset_energy_normal_result_request_drift")
    import math
    primary = result["primary"]
    if any(type(primary.get(key)) not in (float, int) or not math.isfinite(primary[key])
           for key in ("candidate", "reference", "ci_low", "ci_high")):
        raise ValueError("native_dataset_energy_primary_observation_incomplete")
    # The old seal selects hardware artifacts only; it never supplies the new Q.
    declared_timed = producer["timed_fps_authority"]
    local_timed = capture_config["timed_fps_authority"]
    if any(local_timed.get(key) != declared_timed.get(key) for key in ("sha256", "size_bytes")):
        raise ValueError("native_dataset_energy_timed_source_alias_drift")
    timed, timed_path = _dataset_bound_json(local_timed)
    samples = row.get("fps_repetition_samples") or []
    if (row.get("repetition_count_valid") != 3 or len(samples) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples)):
        raise ValueError("native_dataset_energy_three_admitted_FPS_observations_required")
    verified, status = verify_native_energy_command_contract(command, expected_identity={
        "backend": row["backend"], "model": row.get("model") or row.get("model_id"),
        "case": row.get("case") or row.get("case_id"), "setup_id": row["setup_id"], "precision": row["precision"]})
    if (verified is None or timed.get("native_command_contract") != verified
            or row.get("native_command_contract") != verified
            or timed.get("native_command_contract_sha256") != verified.get("contract_sha256")
            or row.get("native_command_contract_sha256") != verified.get("contract_sha256")
            or row.get("native_split_quality_binding") != producer.get("prior_artifact_selection")
            or verified.get("native_split_quality_binding") != producer.get("prior_artifact_selection")
            or timed.get("native_split_quality_binding") != producer.get("prior_artifact_selection")
            or timed.get("ok") is not True or not timed.get("completed_frames")):
        raise ValueError("native_dataset_energy_timed_command_or_artifact_drift:" + str(status))
    sources = [str(row.get("report") or ""), *list(row.get("source_reports") or [])]
    runtime_id = timed.get("runtime_instance_id")
    same_repetition = any(record.get("runtime_instance_id") == runtime_id
        and record.get("fps_makespan") == timed.get("fps_makespan")
        for record in row.get("repetition_records", []) if isinstance(record, Mapping))
    if (str(timed_path) not in sources and not (runtime_id and same_repetition)):
        raise ValueError("native_dataset_energy_timed_source_not_in_selected_FPS")
    _dataset_completion_same_operation(producer["candidate_execution_completion_contract"], timed["completion_execution_contract"])
    local_decision = result["decision"]
    passed = local_decision in {"pass", "reference_close"}
    policy = manifest["policy_sha256"]
    validation = {key: deepcopy(value) for key, value in result.items()}
    validation.update(backend=row["backend"], model=row.get("model") or row.get("model_id"),
        case=row.get("case") or row.get("case_id"), setup_id=row["setup_id"], precision=row["precision"],
        comparison_backend=row["comparison_backend"], task="detection", ok=True, error="", semantic_ok=True,
        contract_consistent=True, claim_ok=passed, accuracy_gate_pass=passed,
        central_quality_evidence_verified=True, precision_quality_binding_verified=True,
        precision_quality_verified=True, task_quality_observation_valid=True, quality_provenance_complete=True,
        quality_claim_result_verified=True, endpoint_contract_complete=True,
        accuracy_gate_policy_sha256=policy, task_quality_policy_sha256=policy, runtime_quality_gate_policy_sha256=policy,
        validation_dataset_image_ids_sha256=result["validation_image_ids_sha256"],
        validation_dataset_ground_truth_sha256=result["validation_ground_truth_sha256"],
        task_quality_gate_decision=local_decision)
    # Physical endpoint facts come from the exact measured Native source;
    # the new dataset-completion endpoint retains its own distinct namespace.
    validation["native_dataset_quality_endpoint_contract_hash"] = result["endpoint_contract_hash"]
    validation["endpoint_contract_hash"] = timed["endpoint_contract_hash"]
    validation["output_endpoint_id"] = timed.get("output_endpoint_id", "")
    validation["stage"] = row.get("stage") or row.get("contract_family") or ""
    evidence = {"schema": NATIVE_DATASET_ENERGY_SCHEMA, "schema_version": 1,
        "authority_kind": "native_dataset_quality", "case_record": dict(descriptor),
        "native_split_quality_required": True, "native_split_energy_binding_valid": True,
        "native_split_energy_binding_status": "closed_native_dataset_and_normal_central_result_verified",
        "native_command_contract_sha256": verified["contract_sha256"],
        "identity": list(native_dataset_energy_identity(row)),
        "source_request_sha256": _sha256_token(identity["source_request_sha256"]),
        "source_request_file_sha256": case["source_request"]["sha256"],
        "normal_result_sha256": case["normal_result"]["sha256"],
        "fresh_eval_run_id": producer["eval_run_id"],
        "timed_fps_declared_source": dict(declared_timed), "timed_fps_local_source": dict(local_timed),
        "legacy_task_quality_decision": primary.get("legacy_decision", ""),
        "prior_artifact_selection_sha256": producer["prior_artifact_selection"]["binding_sha256"],
        "local_task_quality_decision": local_decision, "accuracy_assessment": result.get("accuracy_assessment")}
    evidence["evidence_sha256"] = canonical_json_sha256(evidence)
    validation["native_dataset_quality_authority_sha256"] = evidence["evidence_sha256"]
    return _VerifiedNativeDatasetQuality(validation=validation, evidence=evidence)


def verify_native_dataset_energy_binding(row: Mapping[str, Any]) -> tuple[str, str]:
    """Shared executor entry: recheck the same published sources before energy."""
    from .native_command_contract import canonical_json_sha256
    raw = row.get("native_split_energy_quality_binding")
    if not isinstance(raw, Mapping) or raw.get("schema") != NATIVE_DATASET_ENERGY_SCHEMA:
        raise ValueError("native_dataset_energy_quality_binding_missing")
    body = dict(raw); declared = body.pop("evidence_sha256", "")
    if (canonical_json_sha256(body) != declared
            or row.get("native_split_energy_quality_binding_sha256") != declared
            or row.get("source_request_sha256") != raw.get("source_request_sha256")
            or row.get("successful_command_contract_sha256") != raw.get("native_command_contract_sha256")):
        raise ValueError("native_dataset_energy_quality_binding_drift")
    # Planner preserves the selected FPS row explicitly, without replacing any
    # old source metadata by a fabricated fresh Quality-FIRST workflow.
    selected = row.get("native_dataset_quality_FPS_source")
    if not isinstance(selected, Mapping):
        raise ValueError("native_dataset_energy_FPS_source_missing")
    current = resolve_native_dataset_energy_quality(raw["case_record"], selected, row["native_command_contract"])
    if current["evidence"] != raw or native_dataset_energy_identity(row) != native_dataset_energy_identity(selected):
        raise ValueError("native_dataset_energy_quality_evidence_changed")
    return raw["native_command_contract_sha256"], "fresh_native_dataset_quality_binding_verified"
