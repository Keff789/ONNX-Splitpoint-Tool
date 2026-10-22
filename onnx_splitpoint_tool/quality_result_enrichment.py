"""Normal completed-service metadata projection shared by live and recovered admission."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from .workflow.artifacts import relpath


def enrich_completed_quality_result(result: Mapping[str, Any], *, request_identity: Mapping[str, Any],
                                    manifest: Mapping[str, Any], model_id: str, request_path: Path,
                                    run_dir: Path, collection_eval_run_id: str, case_id: str,
                                    run_id: str, reference_status: Mapping[str, Any],
                                    reference_store_manifest: str = "", producer_flat_field_names=()) -> dict[str, Any]:
    """Project verified actual bindings; this function performs no evaluation."""
    result = copy.deepcopy(dict(result))
    request_identity = copy.deepcopy(dict(request_identity))
    result_completed = (
        str(result.get("status") or "completed") == "completed"
        and str(result.get("technical_status") or "completed") == "completed"
    )
    request_identity["producer_binding_eligible"] = bool(
        (
            request_identity.get("producer_identity_validated")
            or request_identity.get(
                "native_split_quality_binding_required"
            ) is True
        )
        and result_completed
    )
    result.update({
        "status": str(result.get("status") or "completed"),
        # Collection ownership is known locally even for legacy Full
        # requests that do not attest a remote evaluation run ID.
        # Do not fabricate that missing producer attestation.
        "collection_eval_run_id": collection_eval_run_id,
        "eval_run_binding_status": (
            "producer_run_bound" if request_identity.get("eval_run_id")
            else "workflow_request_collection"
        ),
        "task": str(manifest.get("task") or ""),
        "variant": str(manifest.get("variant") or ""),
        "case_id": str(request_identity.get("case_id") or case_id),
        "run_id": str(request_identity.get("source_run_id") or run_id),
        "model_id": model_id,
        "eval_run_id": str(request_identity.get("eval_run_id") or ""),
        "policy_sha256": str(manifest.get("policy_sha256") or ""),
        "metric_gate_config": dict(manifest.get("metric_gate_config") or {}),
        "source_request": relpath(request_path, run_dir),
        "source_request_sha256": str(request_identity.get("source_request_sha256") or ""),
        "source_setup_id": str(request_identity.get("setup_id") or ""),
        "source_run_id": str(request_identity.get("source_run_id") or ""),
        "backend": str(request_identity.get("backend") or ""),
        "execution_role": str(
            request_identity.get("execution_role") or ""
        ),
        "performance_claims_emitted": (
            request_identity.get("performance_claims_emitted")
        ),
        "endpoint_contract_hash": str(request_identity.get("endpoint_contract_hash") or ""),
        "completed_task_endpoint_contract": copy.deepcopy(
            request_identity.get(
                "completed_task_endpoint_contract"
            ) or {}
        ),
        "completed_task_endpoint_contract_hash": str(
            request_identity.get(
                "completed_task_endpoint_contract_hash"
            ) or ""
        ),
        "completed_task_output_endpoint_id": str(
            request_identity.get(
                "completed_task_output_endpoint_id"
            ) or ""
        ),
        "quality_join_endpoint": str(
            request_identity.get("quality_join_endpoint") or ""
        ),
        "runtime_precision_identity": str(request_identity.get("runtime_precision_identity") or ""),
        "quality_contract_sha256": str(request_identity.get("quality_contract_sha256") or ""),
        "preprocessing_contract_sha256": str(request_identity.get("preprocessing_contract_sha256") or ""),
        "decoder_contract_sha256": str(request_identity.get("decoder_contract_sha256") or ""),
        "nms_contract_sha256": str(request_identity.get("nms_contract_sha256") or ""),
        "quality_record_endpoint_contract_sha256": str(
            request_identity.get("quality_record_endpoint_contract_sha256") or ""
        ),
        "source_model_sha256": str(request_identity.get("source_model_sha256") or ""),
        "model_sha256": str(request_identity.get("source_model_sha256") or ""),
        "validation_dataset_manifest_sha256": str(
            request_identity.get("validation_dataset_manifest_sha256") or ""
        ),
        "validation_dataset_sha256": str(
            request_identity.get("validation_dataset_manifest_sha256") or ""
        ),
        "validation_image_ids_sha256": str(
            request_identity.get("validation_image_ids_sha256") or ""
        ),
        "validation_ground_truth_sha256": str(
            request_identity.get("validation_ground_truth_sha256") or ""
        ),
        "task_quality_policy_sha256": str(
            request_identity.get("task_quality_policy_sha256") or ""
        ),
        "producer_binding_eligible": bool(
            request_identity.get("producer_binding_eligible")
        ),
        "artifact_provenance_binding_status": str(
            request_identity.get(
                "artifact_provenance_binding_status"
            ) or "not_required"
        ),
        "artifact_provenance_claim_eligible": bool(
            request_identity.get(
                "artifact_provenance_claim_eligible"
            )
            and request_identity.get("producer_binding_eligible")
        ),
        "native_split_quality_binding_required": bool(
            request_identity.get(
                "native_split_quality_binding_required"
            ) is True
        ),
        "native_split_quality_binding": copy.deepcopy(
            request_identity.get("native_split_quality_binding") or {}
        ),
        "native_split_quality_binding_sha256": str(
            request_identity.get(
                "native_split_quality_binding_sha256"
            ) or ""
        ),
        "request_identity": request_identity,
        "execution_location": "central_management",
        "management_cpu_reference": dict(reference_status),
        "cpu_reference_store_manifest": str(reference_store_manifest or ""),
        "scientific_status": str(result.get("decision") or "inconclusive"),
        "technical_status": "completed" if result_completed else "failed",
    })
    if request_identity.get("producer_identity_validated") is True:
        result["producer_identity"] = copy.deepcopy(
            request_identity.get("producer_identity") or {}
        )
        for field_name in producer_flat_field_names:
            if field_name in request_identity:
                result[field_name] = copy.deepcopy(request_identity[field_name])
        for field_name in (
            "part1_model_sha256", "part2_model_sha256",
            "runtime_artifact_set_sha256", "runtime_artifacts",
            "implementation_runner_sha256",
            "candidate_execution_completion_contract_sha256",
        ):
            if field_name in request_identity:
                result[field_name] = copy.deepcopy(
                    request_identity[field_name]
                )
    if request_identity.get(
        "full_only_plan_identity_validated"
    ) is True:
        result.update({
            "full_only_plan_identity_validated": True,
            "full_only_plan_identity_required": True,
            "full_only_plan_identity": copy.deepcopy(
                request_identity.get("full_only_plan_identity") or {}
            ),
            "full_only_plan_identity_sha256": str(
                request_identity.get(
                    "full_only_plan_identity_sha256"
                ) or ""
            ),
            "quality_canary_id": str(
                request_identity.get("quality_canary_id") or ""
            ),
            "dispatch_run_id": str(
                request_identity.get("dispatch_run_id") or ""
            ),
            "producer_backend": str(
                request_identity.get("producer_backend") or ""
            ),
            "backend": str(request_identity.get("backend") or ""),
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        })
    return result
