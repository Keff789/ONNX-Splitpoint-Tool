"""Canonical Host-Postprocessing evidence normalisation.

Native Full artefacts describe the timed host decoder/NMS with the canonical
``host_postprocess_*`` and completed-task attestation fields.  Older gate code
looked only for the compatibility name ``host_tail_available`` and therefore
rejected fully attested rows.  This module resolves both generations without
weakening the contract: a legacy boolean is accepted only when no newer,
partially populated evidence block is present.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, MutableMapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        return None
    token = str(value or "").strip().lower()
    if token in {"1", "true", "yes", "on", "ok", "pass", "passed"}:
        return True
    if token in {"0", "false", "no", "off", "fail", "failed"}:
        return False
    return None


def _sha256(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token.split(":", 1)[1]
    return token if _SHA256_RE.fullmatch(token) else ""


def _first(
    containers: tuple[Mapping[str, Any], ...],
    *names: str,
) -> Any:
    for container in containers:
        for name in names:
            if name in container and container.get(name) not in (None, ""):
                return container.get(name)
    return None


def _host_postprocess_not_required(
    row: Mapping[str, Any],
    external_summary: Mapping[str, Any],
    nested_summary: Mapping[str, Any],
    *,
    frozen: bool | None,
) -> bool:
    """Recognise rows whose completed task does not need a host tail.

    Explicit raw-head/host-tail requirements always win.  This keeps malformed
    or tampered raw-head evidence fail-closed while allowing classification and
    accelerator-integrated decoded-NMS rows to report an honest N/A instead of
    an incomplete-host-evidence failure.
    """
    contract_containers = (row, external_summary, nested_summary)
    requirement_values = [
        parsed
        for container in contract_containers
        for name in (
            "host_postprocess_required",
            "host_tail_required",
            "requires_host_decode_nms",
            "requires_external_postprocess",
        )
        if name in container
        for parsed in [_bool_or_none(container.get(name))]
        if parsed is not None
    ]
    declared_tasks = {
        str(container.get(name) or "").strip().lower()
        for container in contract_containers
        for name in ("task", "benchmark_task")
        if str(container.get(name) or "").strip()
    }
    if len(declared_tasks) > 1:
        return False
    task = next(iter(declared_tasks), "")
    physical_stage = str(
        _first(
            contract_containers,
            "accelerator_output_stage",
            "stage",
        )
        or ""
    ).strip().lower()
    physical_family = str(
        _first(
            contract_containers,
            "accelerator_output_contract_family",
            "contract_family",
            "output_format",
        )
        or ""
    ).strip().lower()
    raw_head = bool(
        "raw_head" in physical_stage
        or "raw_head" in physical_family
        or physical_family in {
            "raw_detection_head",
            "raw_detection_tensors",
            "raw_head_only",
        }
    )
    explicitly_not_required = bool(
        frozen is False
        or (
            requirement_values
            and not any(requirement_values)
        )
    )

    # A positive requirement, a physical raw-head endpoint, or an explicitly
    # frozen host tail must never be downgraded to N/A.
    if any(requirement_values) or raw_head or frozen is True:
        return False

    if task == "classification":
        return True

    decoded_nms = bool(
        physical_stage == "decoded_nms"
        or physical_family in {"decoded_nms", "bn6_detections"}
    )
    if task == "detection" and decoded_nms:
        return explicitly_not_required

    # An explicit all-false requirement declaration is authoritative for every
    # non-raw endpoint, including older rows without a task/family label.
    return explicitly_not_required


def _completed_execution_host_postprocess(
    row: Mapping[str, Any],
    external_summary: Mapping[str, Any],
    nested_summary: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Strictly resolve the measured v2 completed-detection tail.

    This path deliberately precedes the legacy frozen-host-tail aliases.  An
    explicitly declared v2 contract is either verified in full or rejected;
    incomplete/tampered v2 evidence never falls back to permissive old fields.
    """
    sources = (row, external_summary, nested_summary)
    attestation = next((
        source.get("completed_task_endpoint_attestation")
        for source in sources
        if isinstance(
            source.get("completed_task_endpoint_attestation"), Mapping
        )
    ), {})
    mode = str(
        _first(
            (*sources, attestation),
            "completed_task_completion_mode",
        )
        or ""
    ).strip()
    fast = mode == "native_three_stage_fast_oracle_outside_timing"
    if mode != "detection_completion_execution_v1" and not fast:
        return None
    projection_conflicts = [
        str(value)
        for source in sources
        for value in (
            source.get("completed_v2_projection_conflicts") or []
            if isinstance(
                source.get("completed_v2_projection_conflicts"), list
            )
            else ["malformed_completed_v2_projection_conflicts"]
            if "completed_v2_projection_conflicts" in source
            and source.get("completed_v2_projection_conflicts")
            not in (None, [])
            else []
        )
    ]
    native_command = next((
        source.get("native_command_contract")
        for source in sources
        if isinstance(source.get("native_command_contract"), Mapping)
    ), {})
    runtime_options = (
        native_command.get("runtime_options")
        if isinstance(native_command.get("runtime_options"), Mapping)
        else {}
    )
    execution = next((
        source.get("completion_execution_contract")
        for source in sources
        if isinstance(source.get("completion_execution_contract"), Mapping)
    ), runtime_options.get("completion_execution_contract"))
    try:
        if projection_conflicts:
            raise ValueError("completed_v2_projection_conflict")
        from ..native_detection_postprocess import (
            verify_detection_completion_execution_attestation,
            verify_detection_completion_execution_contract,
        )

        contract = verify_detection_completion_execution_contract(
            execution
        )
        if fast:
            from ..native_three_stage import verify_fast_completion_attestation
            verified = verify_fast_completion_attestation(attestation, execution_contract=contract)
        else:
            verified = verify_detection_completion_execution_attestation(
                attestation,
                execution_contract=contract,
                expected_observation_relation="same_hotloop_sentinel",
            )
        completed = int(verified.get("completed_work_units") or 0)
        source_endpoint = dict(contract["source_endpoint"])
        completed_endpoint = dict(
            contract["completed_endpoint_contract"]
        )
        comparison_endpoint = dict(
            contract["comparison_endpoint_contract"]
        )
        processor = dict(contract["processor_contract"])

        def explicit_values(*names: str) -> list[Any]:
            return [
                source.get(name)
                for source in sources
                for name in names
                if name in source and source.get(name) not in (None, "")
            ]

        def all_equal(expected: Any, *names: str) -> bool:
            values = explicit_values(*names)
            return bool(values) and all(value == expected for value in values)

        def all_mappings_equal(
            expected: Mapping[str, Any], *names: str,
        ) -> bool:
            values = explicit_values(*names)
            return bool(values) and all(
                isinstance(value, Mapping)
                and dict(value) == dict(expected)
                for value in values
            )

        source_hash = source_endpoint["endpoint_contract_hash"]
        source_id = source_endpoint["output_endpoint_id"]
        completed_hash = completed_endpoint["endpoint_contract_hash"]
        completed_id = completed_endpoint["output_endpoint_id"]
        comparison_hash = comparison_endpoint["endpoint_contract_hash"]
        comparison_id = comparison_endpoint["output_endpoint_id"]
        count_values = explicit_values(
            "completed_work_units", "completed_frames",
        )
        valid = bool(
            completed > 0
            and verified.get("exact_result_claim_bound") is True
            and (fast or verified.get("same_hotloop_sentinel") is True)
            and (fast or verified.get("postprocess_completion_verified") is True)
            and int(verified.get("postprocess_completed_frames") or 0)
            == completed
            and all(
                not isinstance(value, bool) and int(value) == completed
                for value in count_values
            )
            and bool(count_values)
            and all_equal(
                completed,
                "postprocess_completed_frames",
            )
            and all_equal(
                True,
                "postprocess_completion_verified",
            )
            and all_equal(
                True,
                "postprocess_included",
            )
            and all_equal(
                True,
                "completed_task_endpoint_attested",
            )
            and all_equal(
                "passed",
                "completed_task_endpoint_attestation_status",
            )
            and all_equal(
                "postflight_oracle_sentinel" if fast else "same_hotloop_sentinel",
                "completion_observation_relation",
            )
            and all_equal(
                True,
                "completion_exact_result_claim_bound",
            )
            and all_mappings_equal(
                verified,
                "completion_execution_attestation",
                "completed_task_endpoint_attestation",
            )
            and all_mappings_equal(
                contract,
                "completion_execution_contract",
            )
            and all_equal(
                contract["contract_sha256"],
                "completion_execution_contract_sha256",
            )
            and all_equal(
                verified["artifact_sha256"],
                "completion_artifact_sha256",
            )
            and all_equal(
                verified["schema_sha256"],
                "completion_schema_sha256",
            )
            and all_equal(
                verified["content_sha256"],
                "completion_content_sha256",
            )
            and all_equal(
                verified["invocation_sha256"],
                "completion_invocation_sha256",
            )
            and all_equal(
                verified["relation_sha256"],
                "completion_relation_sha256",
            )
            and all_equal(
                source_hash,
                "endpoint_contract_hash",
                "accelerator_endpoint_contract_hash",
            )
            and all(
                value == source_id
                for value in explicit_values(
                    "output_endpoint_id",
                    "physical_output_endpoint_id",
                )
            )
            and all_equal(
                completed_hash,
                "completed_task_endpoint_contract_hash",
            )
            and all_equal(
                completed_id,
                "completed_task_output_endpoint_id",
            )
            and all_equal(
                comparison_hash,
                "completed_task_comparison_endpoint_contract_hash",
            )
            and all_equal(
                comparison_id,
                "completed_task_comparison_output_endpoint_id",
            )
            and all_mappings_equal(
                completed_endpoint,
                "completed_task_endpoint_contract",
            )
            and all_mappings_equal(
                comparison_endpoint,
                "completed_task_comparison_endpoint_contract",
            )
        )
        decoder_id = str(
            processor.get("decoder_id")
            or processor.get("normalizer_id")
            or processor.get("materializer_id")
            or ""
        )
        frozen_hash = str(
            processor.get("contract_sha256") or ""
        ).strip().lower()
        if not valid:
            raise ValueError(
                "completion_execution_projection_mismatch"
            )
    except Exception:
        return {
            "available": False,
            "status": "failed_invalid_completed_detection_execution_v1",
            "source": mode,
            "canonical_evidence_present": True,
            "legacy_alias_conflict": False,
            "frozen_contract_sha256": "",
            "completed_endpoint_contract_sha256": "",
            "decoder_id": "",
        }
    legacy_values = [
        source.get(name)
        for source in sources
        for name in (
            "host_tail_available", "host_postprocessing_available",
        )
        if name in source and source.get(name) not in (None, "")
    ]
    legacy_conflict = any(value is not True for value in legacy_values)
    return {
        "available": not legacy_conflict,
        "status": (
            "host_postprocess_evidence_conflict"
            if legacy_conflict else "passed"
        ),
        "source": mode,
        "canonical_evidence_present": True,
        "legacy_alias_conflict": legacy_conflict,
        "frozen_contract_sha256": frozen_hash,
        "completed_endpoint_contract_sha256": completed_hash,
        "decoder_id": decoder_id,
    }


def _generic_execution_host_postprocess(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read the generic runner's measured frozen tail without Native aliases."""
    evidence = row.get("generic_completion_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("task") != "detection":
        return None
    contract = evidence.get("postprocess_contract") or {}
    # Direct BN6 normalization has its own endpoint contract. This branch
    # consumes the existing frozen decode/NMS contract used by raw-head Full.
    if isinstance(contract, Mapping) and contract.get("normalizer_id"):
        return None
    result = {
        "available": False,
        "status": "failed_invalid_generic_host_postprocess_evidence",
        "source": "measured_generic_task_completion",
        "canonical_evidence_present": True,
        "legacy_alias_conflict": False,
        "frozen_contract_sha256": "",
        "completed_endpoint_contract_sha256": "",
        "decoder_id": "",
    }
    try:
        from ..runners.task_completion import validate_completion
        from ..native_detection_postprocess import FrozenPostprocessError, verify_frozen_postprocess_contract

        backend = str(row.get("backend") or "").lower()
        producer = {
            "hailo8": "generic_hailo_full", "hailo10": "generic_hailo_full",
            "hailo10h": "generic_hailo_full", "deepx_m1": "generic_deepx_full",
            "tensorrt": "generic_ort_full", "ort_tensorrt": "generic_ort_full",
            "cuda_ort": "generic_ort_full", "cpu_ort": "generic_ort_full",
        }.get(backend)
        if (
            not producer or row.get("variant") != "full"
            or row.get("task") != "detection"
        ):
            raise ValueError("generic_host_tail_role_mismatch")
        validate_completion(evidence, task="detection", producer=producer)
        verified = verify_frozen_postprocess_contract(contract)
        if verified["model_id"] != row.get("model_id"):
            raise ValueError("generic_host_tail_model_mismatch")
        if (
            row.get("postprocess_included") is not True
            or row.get("postprocess_completion_verified") is not True
            or row.get("postprocess_completed_frames") != len(evidence["frames"])
        ):
            raise ValueError("generic_host_tail_completion_mismatch")
        for name in ("host_tail_available", "host_postprocessing_available"):
            if row.get(name) not in (None, "") and row[name] is not True:
                result["legacy_alias_conflict"] = True
                raise ValueError("generic_host_tail_alias_conflict")
        result.update(
            available=True, status="passed",
            frozen_contract_sha256=verified["contract_sha256"],
            decoder_id=verified["decoder_id"],
        )
    except (ValueError, TypeError, KeyError, FrozenPostprocessError):
        pass
    return result


def resolve_host_postprocess_evidence(
    row: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one fail-closed availability decision for host decode/NMS."""
    nested_summary = (
        row.get("deployment_contract_summary")
        if isinstance(row.get("deployment_contract_summary"), Mapping)
        else {}
    )
    external_summary = summary if isinstance(summary, Mapping) else {}
    completed_execution = _completed_execution_host_postprocess(
        row, external_summary, nested_summary,
    )
    if completed_execution is not None:
        return completed_execution
    generic_execution = _generic_execution_host_postprocess(row)
    if generic_execution is not None:
        return generic_execution
    attestation = (
        row.get("completed_task_endpoint_attestation")
        if isinstance(row.get("completed_task_endpoint_attestation"), Mapping)
        else {}
    )
    contract = (
        row.get("frozen_host_postprocess_contract")
        if isinstance(row.get("frozen_host_postprocess_contract"), Mapping)
        else {}
    )
    completed_contract = (
        row.get("completed_task_endpoint_contract")
        if isinstance(row.get("completed_task_endpoint_contract"), Mapping)
        else attestation.get("completed_endpoint_contract")
        if isinstance(attestation.get("completed_endpoint_contract"), Mapping)
        else {}
    )
    containers = (
        row, external_summary, nested_summary, attestation,
        contract, completed_contract,
    )

    canonical_names = (
        "host_postprocess_frozen",
        "postprocess_included",
        "postprocess_completion_verified",
        "frozen_host_postprocess_contract_sha256",
        "frozen_postprocess_contract_sha256",
        "completed_task_endpoint_attested",
        "completed_task_endpoint_contract_hash",
        "completed_task_output_endpoint_id",
        "completed_task_stage",
        "completed_task_contract_family",
    )
    canonical_present = any(
        name in container and container.get(name) not in (None, "")
        for container in containers
        for name in canonical_names
    )
    frozen = _bool_or_none(_first(containers, "host_postprocess_frozen"))
    included = _bool_or_none(_first(containers, "postprocess_included"))
    completed = _bool_or_none(
        _first(containers, "postprocess_completion_verified")
    )
    completed_attested = _bool_or_none(
        _first(containers, "completed_task_endpoint_attested", "attested")
    )
    stage = str(
        _first(
            (row, external_summary, nested_summary),
            "completed_task_stage",
        )
        or attestation.get("stage")
        or attestation.get("endpoint")
        or ""
    ).strip().lower()
    family = str(
        _first(
            (row, external_summary, nested_summary),
            "completed_task_contract_family",
        )
        or completed_contract.get("output_contract_family")
        or completed_contract.get("contract_family")
        or attestation.get("stage")
        or ""
    ).strip().lower()
    frozen_hash = _sha256(
        _first(
            containers,
            "frozen_host_postprocess_contract_sha256",
            "frozen_postprocess_contract_sha256",
            "contract_sha256",
        )
    )
    endpoint_hash = _sha256(
        _first(
            (row, external_summary, nested_summary),
            "completed_task_endpoint_contract_hash",
            "logical_endpoint_contract_hash",
        )
        or attestation.get("endpoint_contract_hash")
        or completed_contract.get("endpoint_contract_hash")
    )
    strict_attestation_valid = False
    decoder_id = ""
    if attestation and contract:
        try:
            from ..native_detection_postprocess import (
                verify_frozen_postprocess_contract,
            )

            verified_frozen = verify_frozen_postprocess_contract(contract)
            decoder_id = str(verified_frozen.get("decoder_id") or "")
            binding = attestation.get("completed_endpoint_contract")
            if not isinstance(binding, Mapping):
                raise ValueError("completed_endpoint_contract_missing")
            binding_identity = dict(binding)
            binding_complete = binding_identity.pop(
                "endpoint_contract_complete", None,
            )
            binding_hash = _sha256(
                binding_identity.pop("endpoint_contract_hash", ""),
            )
            binding_id = str(
                binding_identity.pop("output_endpoint_id", "") or "",
            ).strip()
            calculated_binding_hash = hashlib.sha256(json.dumps(
                binding_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")).hexdigest()
            completed_frames = int(attestation.get("completed_frames") or 0)
            postprocess_frames = int(
                attestation.get("postprocess_completed_frames") or 0
            )
            result = attestation.get("frozen_postprocess_result")
            top_result = row.get("frozen_host_postprocess_result")
            top_completed_contract = row.get(
                "completed_task_endpoint_contract"
            )
            top_endpoint_hash = _sha256(
                row.get("completed_task_endpoint_contract_hash")
            )
            top_endpoint_id = str(
                row.get("completed_task_output_endpoint_id") or ""
            ).strip()
            strict_attestation_valid = bool(
                binding_complete is True
                and binding_hash == calculated_binding_hash
                and binding_id == (
                    f"detection:decoded_nms:{binding_hash}"
                )
                and _sha256(attestation.get("endpoint_contract_hash"))
                == binding_hash
                and attestation.get("attested") is True
                and str(attestation.get("status") or "").strip().lower()
                == "passed"
                and str(attestation.get("task") or "").strip().lower()
                == "detection"
                and str(attestation.get("stage") or "").strip().lower()
                == "decoded_nms"
                and str(attestation.get("endpoint") or "").strip().lower()
                == "decoded_nms"
                and completed_frames > 0
                and postprocess_frames == completed_frames
                and attestation.get("postprocess_completion_verified")
                is True
                and _sha256(
                    attestation.get(
                        "frozen_postprocess_contract_sha256"
                    )
                ) == verified_frozen["contract_sha256"]
                and isinstance(result, Mapping)
                and str(result.get("task") or "").strip().lower()
                == "detection"
                and str(result.get("contract_family") or "").strip().lower()
                == "decoded_nms"
                and _sha256(result.get("postprocess_contract_sha256"))
                == verified_frozen["contract_sha256"]
                and (
                    not isinstance(top_result, Mapping)
                    or dict(top_result) == dict(result)
                )
                and (
                    not isinstance(top_completed_contract, Mapping)
                    or dict(top_completed_contract) == dict(binding)
                )
                and (
                    not top_endpoint_hash
                    or top_endpoint_hash == binding_hash
                )
                and (
                    not top_endpoint_id
                    or top_endpoint_id == binding_id
                )
            )
        except Exception:
            strict_attestation_valid = False
            decoder_id = ""

    canonical_available = bool(
        frozen is True
        and included is True
        and completed is True
        and completed_attested is True
        and stage == "decoded_nms"
        and family == "decoded_nms"
        and frozen_hash
        and endpoint_hash
        and strict_attestation_valid
    )

    legacy_values = [
        parsed
        for container in (row, external_summary, nested_summary)
        for name in ("host_tail_available", "host_postprocessing_available")
        if name in container
        for parsed in [_bool_or_none(container.get(name))]
        if parsed is not None
    ]
    legacy_true = any(legacy_values)
    legacy_false = bool(legacy_values) and not legacy_true

    not_required = _host_postprocess_not_required(
        row,
        external_summary,
        nested_summary,
        frozen=frozen,
    )
    if not_required and legacy_true:
        return {
            "available": False,
            "status": "host_postprocess_not_required_legacy_conflict",
            "source": "explicit_host_postprocess_not_required",
            "canonical_evidence_present": canonical_present,
            "legacy_alias_conflict": True,
            "frozen_contract_sha256": frozen_hash,
            "completed_endpoint_contract_sha256": endpoint_hash,
            "decoder_id": "",
        }
    if not_required:
        return {
            "available": None,
            "status": "not_required",
            "source": "explicit_host_postprocess_not_required",
            "canonical_evidence_present": canonical_present,
            "legacy_alias_conflict": False,
            "frozen_contract_sha256": "",
            "completed_endpoint_contract_sha256": "",
            "decoder_id": "",
        }
    if canonical_available and legacy_false:
        return {
            "available": False,
            "status": "host_postprocess_evidence_conflict",
            "source": "completed_task_host_postprocess_attestation_v1",
            "canonical_evidence_present": True,
            "legacy_alias_conflict": True,
            "frozen_contract_sha256": frozen_hash,
            "completed_endpoint_contract_sha256": endpoint_hash,
            "decoder_id": decoder_id,
        }
    if canonical_available:
        return {
            "available": True,
            "status": "passed",
            "source": "completed_task_host_postprocess_attestation_v1",
            "canonical_evidence_present": True,
            "legacy_alias_conflict": False,
            "frozen_contract_sha256": frozen_hash,
            "completed_endpoint_contract_sha256": endpoint_hash,
            "decoder_id": decoder_id,
        }
    if canonical_present:
        return {
            "available": False,
            "status": "failed_incomplete_canonical_host_postprocess_evidence",
            "source": "completed_task_host_postprocess_attestation_v1",
            "canonical_evidence_present": True,
            "legacy_alias_conflict": legacy_true,
            "frozen_contract_sha256": frozen_hash,
            "completed_endpoint_contract_sha256": endpoint_hash,
            "decoder_id": decoder_id,
        }
    if legacy_true:
        return {
            "available": True,
            "status": "passed_legacy_alias",
            "source": "legacy_host_tail_available",
            "canonical_evidence_present": False,
            "legacy_alias_conflict": False,
            "frozen_contract_sha256": "",
            "completed_endpoint_contract_sha256": "",
            "decoder_id": "",
        }
    if legacy_false:
        return {
            "available": False,
            "status": "failed_legacy_alias",
            "source": "legacy_host_tail_available",
            "canonical_evidence_present": False,
            "legacy_alias_conflict": False,
            "frozen_contract_sha256": "",
            "completed_endpoint_contract_sha256": "",
            "decoder_id": "",
        }
    return {
        "available": None,
        "status": "unavailable",
        "source": "none",
        "canonical_evidence_present": False,
        "legacy_alias_conflict": False,
        "frozen_contract_sha256": "",
        "completed_endpoint_contract_sha256": "",
        "decoder_id": "",
    }


def apply_host_postprocess_aliases(
    row: MutableMapping[str, Any],
    summary: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Materialise canonical and compatibility field names on one row."""
    resolved = resolve_host_postprocess_evidence(row, summary)
    available = resolved.get("available")
    if resolved.get("status") == "not_required":
        row["host_postprocess_required"] = False
        row["host_tail_required"] = False
        # Keep the established Boolean aliases lossless for older CSV/report
        # consumers.  ``not_required`` is the authoritative N/A distinction;
        # False here means no host tail exists, not that a required tail failed.
        row["host_postprocessing_available"] = False
        row["host_tail_available"] = False
    elif available is not None:
        row["host_postprocessing_available"] = bool(available)
        row["host_tail_available"] = bool(available)
    if available is True:
        row["host_postprocess_required"] = True
        row["host_tail_required"] = True
        row["decoder_contract_pass"] = True
        row["nms_ok"] = True
        row["decoder_id"] = str(resolved.get("decoder_id") or "")
    row["host_postprocessing_evidence_status"] = resolved["status"]
    row["host_postprocessing_evidence_source"] = resolved["source"]
    row["host_postprocessing_legacy_alias_conflict"] = bool(
        resolved["legacy_alias_conflict"]
    )
    return row


__all__ = [
    "apply_host_postprocess_aliases",
    "resolve_host_postprocess_evidence",
]
