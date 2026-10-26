"""Prove a bounded staging relation without replacing either manifest identity.

This reader covers the observed Generic Hailo8 source-only staging changes.
The original manifest must also remain the actual plan's boundary-metadata
artifact.  All other normal plan, Quality, receipt and start checks still apply.
"""
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import PurePosixPath
import re


_ROLES = (
    "part1_runtime", "source_part1_onnx", "source_part2_onnx",
    "part1_build_receipt", "engine", "build_part2_onnx", "engine_build_receipt",
)
_FULL_FIELDS = frozenset({
    "full", "full_build", "full_build_receipt", "full_end_node_names",
    "full_endpoint_mode", "full_output_contract", "full_prepared_baseline",
    "full_receipt_attested", "full_required", "full_true_full_model",
})
_BUILD_DIAGNOSTICS = frozenset({
    "build_evidence", "cache_key", "context_count", "partition_iterations",
    "partition_time_s", "allocation_time_s", "compilation_time_s",
    "expensive_build", "expensive_build_reason", "negative_evidence_hit",
})
_MISSING = object()


def _require(value, message):
    if not value:
        raise ValueError("Generic split-manifest staging: " + message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _document(raw, source):
    _require(isinstance(raw, bytes) and 0 < len(raw) <= 2 * 1024**2,
             "manifest bytes missing or oversized")
    _require(isinstance(source, Mapping), "manifest descriptor missing")
    path = PurePosixPath(str(source.get("path") or ""))
    _require(path.is_absolute() and ".." not in path.parts and path.name == "split_manifest.json",
             "ordinary absolute split-manifest path required")
    _require(_sha(source.get("sha256"))
             and hashlib.sha256(raw).hexdigest() == source["sha256"]
             and type(source.get("size_bytes")) is int and len(raw) == source["size_bytes"],
             "manifest bytes differ from descriptor")
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    value = json.loads(raw, object_pairs_hook=_object, parse_constant=invalid_constant)
    _require(isinstance(value, dict) and value.get("schema") == "onnx-splitpoint/split-manifest"
             and type(value.get("schema_version")) is int and value["schema_version"] == 1,
             "unsupported manifest schema")
    return value, {key: source[key] for key in ("path", "sha256", "size_bytes")}


def _manifest_graph_case(original, staged, *, model, case):
    """Bind display padding to unchanged actual integer-boundary graph names."""
    _require(isinstance(case, str) and re.fullmatch(r"b[0-9]+", case),
             "invalid display cut identity")
    boundary = int(case[1:])
    graph_cases = {case, "b" + str(boundary)}
    matches = [token for token in graph_cases
               if all(document.get("part1") == f"{model}_part1_{token}.onnx"
                      and document.get("part2") == f"{model}_part2_{token}.onnx"
                      and type(document.get("boundary")) is int
                      and type(document.get("boundary_index")) is int
                      and document["boundary"] == document["boundary_index"] == boundary
                      for document in (original, staged))]
    _require(len(matches) == 1, "manifest model/cut identity differs")
    return matches[0]


def _boundary_metadata_locator(boundary_metadata, original_source, *, case):
    """Preserve an explicit original-to-local mirror relation, never relabel it.

    The normal plan validator hashes every artifact's actual local_path with
    read_bound before this call, registering it for the final source rehash.
    This pure relation requires that same original byte identity and the
    existing explicit staging alias; it does not read or manufacture a file.
    """
    _require(boundary_metadata.get("sha256") == original_source["sha256"]
             and type(boundary_metadata.get("size_bytes")) is int
             and boundary_metadata["size_bytes"] == original_source["size_bytes"],
             "original bytes are not the actual boundary-metadata artifact")
    local_value = boundary_metadata.get("local_path")
    if local_value == original_source["path"]:
        return {"original_source": original_source, "actual_boundary_metadata_artifact": deepcopy(boundary_metadata),
                "local_mirror_used": False}
    local = PurePosixPath(str(local_value or ""))
    remote = PurePosixPath(str(boundary_metadata.get("path") or ""))
    expected_suffix = (case, "boundary_metadata", "split_manifest.json")
    _require(set(boundary_metadata) == {"local_path", "path", "sha256", "size_bytes", "source_path_alias", "stage_local"}
             and boundary_metadata["stage_local"] is True
             and boundary_metadata["source_path_alias"] == original_source["path"]
             and local.is_absolute() and ".." not in local.parts and str(local) == local_value
             and remote.is_absolute() and ".." not in remote.parts
             and local.parts[-3:] == remote.parts[-3:] == expected_suffix,
             "unproved original boundary-metadata local mirror alias")
    return {"original_source": original_source, "actual_boundary_metadata_artifact": deepcopy(boundary_metadata),
            "local_mirror_used": True,
            "normal_local_file_verifier": "generic_only_plan.validate_plan/read_bound with final validation_scope source rehash",
            "same_original_sha256": original_source["sha256"], "same_original_size_bytes": original_source["size_bytes"],
            "original_descriptor_or_hash_rewritten": False}


def _historical_build_provenance(build, original, *, arch, model, case,
                                 source_sha256, full_model_sha256, prepared):
    """Validate retained compile evidence, without equating compiler/source graphs.

    The normal caller has already checked the actual measured artifacts and
    receipts. This pure relation additionally binds the historical observation
    being omitted to that same model, cut and compiler recipe.
    """
    from .build_evidence import (
        BuildEvidenceError, boundary_endpoint_contract_sha256,
        canonical_build_key_from_hailo_v3_payload, canonical_sha256,
    )

    evidence = build["build_evidence"]
    _require(isinstance(evidence, dict) and set(evidence) == {
        "cache_key_v3", "cache_payload_v3", "compiler_dispatch_allowed", "context",
        "evidence_origin", "key", "key_sha256", "negative_evidence_hit", "reason",
        "record", "reusable", "state", "status",
    }, "unreviewed historical build evidence")
    _require(evidence["status"] == "MISS" and evidence["reason"] == "exact_key_not_found"
             and evidence["compiler_dispatch_allowed"] is True
             and evidence["negative_evidence_hit"] is False and evidence["reusable"] is False
             and all(evidence[key] is None for key in ("state", "record", "evidence_origin")),
             "historical build was failed, unknown or negative evidence")
    context, payload = evidence["context"], evidence["cache_payload_v3"]
    _require(isinstance(payload, dict) and set(payload) == {
        "activation_part1_sha256", "calibration_batch_size", "calibration_count",
        "calibration_identity", "calibration_memory_cap_bytes", "calibration_storage",
        "disable_rt_metadata_extraction", "end_nodes", "extra_model_script",
        "hailo_sdk_version", "hw_arch", "integrity", "model_sha256", "net_input_shapes",
        "net_name", "optimization_level", "prepared_calibration_identity_sha256",
        "preprocessing_contract", "preprocessing_contract_sha256",
        "requested_calibration_count", "schema", "start_nodes",
    }, "unreviewed historical compiler cache payload")
    _require(isinstance(context, dict) and set(context) == {
        "boundary", "full_source_onnx_path", "full_source_onnx_sha256",
        "model_id", "split_manifest", "stage",
    } and context["boundary"] == int(case[1:]) and context["model_id"] == model
             and context["stage"] == "part1"
             and context["full_source_onnx_sha256"] == full_model_sha256
             and context["full_source_onnx_path"] == original["full_model_source"],
             "historical build context differs from selected model/cut")
    snapshot = context["split_manifest"]
    _require(isinstance(snapshot, dict)
             and {"boundary", "boundary_index", "part1", "part2", "cut", "cut_tensors",
                  "part1_cut_names", "part2_cut_names", "io", "pipeline", "orig_inputs",
                  "orig_outputs", "full_model", "full_model_source"} <= set(snapshot)
             and all(value == original.get(key, _MISSING)
                     for key, value in snapshot.items() if key != "hailo")
             and isinstance(snapshot.get("hailo"), dict)
             and snapshot["hailo"].get("hefs") == {}
             and snapshot["hailo"].get("config") == original["hailo"]["config"]
             and all(value == original["hailo"].get(key, _MISSING)
                     for key, value in snapshot["hailo"].items() if key != "hefs"),
             "historical build snapshot differs from original scientific manifest")
    config = original["hailo"]["config"]
    # Some successful historical H8 records omitted these redundant diagnostics.
    # Their canonical evidence and the exact measured receipt remain required;
    # absence is not a newly inferred calibration-location observation.
    _require(("calib_source" in build and build["calib_source"] == config["calib_dir"]
              or "calib_source" not in build and arch == "hailo8")
             and ("cache_key" in build and build["cache_key"] == evidence["cache_key_v3"]
                  or "cache_key" not in build and arch == "hailo8"),
             "historical compiler diagnostics differ or lack measured H8 receipt scope")
    _require(payload["hw_arch"] == ("hailo8" if arch == "hailo8" else "hailo10h")
             and payload["net_name"] == PurePosixPath(original["part1"]).stem
             and payload["net_input_shapes"] == [1, 3, *prepared["shape"][:2]]
             and payload["optimization_level"] == config["opt_level"]
             and payload["requested_calibration_count"] == config["calib_count"]
             and payload["calibration_batch_size"] == config["calib_batch_size"],
             "historical compiler/calibration recipe differs")
    try:
        endpoint = boundary_endpoint_contract_sha256(
            stage="part1", cache_payload=payload, split_manifest=original)
        key = canonical_build_key_from_hailo_v3_payload(
            payload, builder_source_onnx_sha256=source_sha256,
            full_source_onnx_sha256=full_model_sha256,
            boundary_endpoint_contract_sha256=endpoint,
            expected_cache_key=evidence["cache_key_v3"])
        _require(evidence["key"] == key and evidence["key_sha256"] == canonical_sha256(key),
                 "historical compiler evidence identity differs")
    except BuildEvidenceError as error:
        raise ValueError("Generic split-manifest staging: invalid historical compiler evidence") from error
    _require(build.get("negative_evidence_hit", False) is False
             and all(build.get(key) is False for key in
                     ("mapping_failed", "timed_out", "watchdog_expired")),
             "historical build contains failure evidence")
    flags = tuple(build.get(key) for key in
                  ("single_context_failed", "single_context_used", "multi_context_used"))
    count = build.get("context_count")
    _require(all(type(value) is bool for value in flags)
             and (
        (build["context_mode"] == "single_context_failed_to_multi"
         and flags == (True, False, True) and type(count) is int and count > 1)
        or (build["context_mode"] == "multi_context_used"
            and flags == (False, False, True) and type(count) is int and count > 1)
        or (build["context_mode"] == "single_context_used"
            and (flags in ((False, True, False), (False, False, False))
                 and type(count) is int and count == 1
                 or flags == (False, True, False) and "context_count" not in build))),
        "unknown or inconsistent historical compiler context")
    for name in ("partition_iterations", "partition_time_s", "allocation_time_s", "compilation_time_s"):
        if name in build:
            value = build[name]
            _require(type(value) in (int, float) and math.isfinite(value) and value >= 0
                     and (name != "partition_iterations" or type(value) is int),
                     "invalid historical compiler diagnostic")
    reasons = []
    if build["elapsed_s"] >= 1800:
        reasons.append("elapsed_s>=1800")
    if build["context_mode"] in {"multi_context_used", "single_context_failed_to_multi"}:
        reasons.append(build["context_mode"])
    _require(("expensive_build" not in build and "expensive_build_reason" not in build)
             or (build.get("expensive_build") is True and bool(reasons)
                 and build.get("expensive_build_reason") == ";".join(reasons)),
             "inconsistent historical build-cost diagnostic")
    return {
        "backend_metadata": arch, "executed_artifact_identity_proved": arch == "hailo8",
        "historical_observation": deepcopy(build),
        "compiler_onnx_sha256": key["compiler_onnx_sha256"],
        "source_onnx_sha256": source_sha256,
        "compiler_source_byte_equality_claimed": False,
    }


def _omitted_h10_reuse_provenance(build, original, *, case):
    """Retain a positive unused reuse summary, without proving H10 execution.

    This summary carries no compiler-byte or receipt identity. Only its exact
    known shape and model/cut locators are admitted; the caller separately
    proves unchanged selected H8 artifacts and the full scientific manifest.
    """
    _require(set(build) == {
        "build_receipt_path", "cache_hit", "calib_source", "compiler_onnx_path",
        "context_mode", "elapsed_s", "mapping_failed", "multi_context_used", "ok",
        "single_context_failed", "single_context_used", "skipped", "source_onnx_path",
        "timed_out", "watchdog_expired",
    } and build["ok"] is True and build["cache_hit"] is True
             and build["skipped"] is True
             and build["calib_source"] == "validated_existing_artifact"
             and build["context_mode"] == "skipped"
             and all(build[key] is False for key in (
                 "single_context_failed", "single_context_used", "multi_context_used",
                 "mapping_failed", "timed_out", "watchdog_expired"))
             and type(build["elapsed_s"]) in (int, float)
             and math.isfinite(build["elapsed_s"]) and build["elapsed_s"] >= 0,
             "unknown or nonpositive omitted historical H10 reuse")
    source_value = build["source_onnx_path"]
    source = PurePosixPath(str(source_value or ""))
    _require(source.is_absolute() and ".." not in source.parts
             and str(source) == source_value
             and source.parent.name == case and source.name == original["part1"]
             and build["compiler_onnx_path"] == source_value
             and build["build_receipt_path"] == str(
                 source.parent / "hailo/hailo10/part1/hailo_hef_build_receipt.json"),
             "unbound omitted historical H10 reuse locators")
    return {
        "backend_metadata": "hailo10", "historical_observation": deepcopy(build),
        "historical_reuse_observation": True,
        "executed_artifact_identity_proved": False,
        "compiler_source_byte_equality_claimed": False,
    }


def _receipt_publication_flags(receipt, *, positive_original=False):
    return ((receipt.get("publish_artifacts") is True and receipt.get("diagnostic_only") is False)
            or (positive_original is True and "publish_artifacts" not in receipt and "diagnostic_only" not in receipt))


def _bind_historical_receipt(raw, build, bindings, *, legacy_positive_original=False):
    """Tie old compile provenance to the exact receipt actually measured."""
    receipt_binding = bindings["part1_build_receipt"]
    _require(isinstance(raw, bytes) and 0 < len(raw) <= 2 * 1024**2,
             "actual historical-build receipt bytes required")
    _require(len(raw) == receipt_binding["size_bytes"]
             and hashlib.sha256(raw).hexdigest() == receipt_binding["sha256"],
             "actual historical-build receipt bytes differ from artifact binding")

    def invalid_constant(value):
        raise ValueError("Generic split-manifest staging: nonfinite receipt constant: " + value)

    try:
        receipt = json.loads(raw, object_pairs_hook=_object, parse_constant=invalid_constant)
    except json.JSONDecodeError as error:
        raise ValueError("Generic split-manifest staging: malformed historical-build receipt") from error
    evidence = build["build_evidence"]
    payload, key = evidence["cache_payload_v3"], evidence["key"]
    _require(isinstance(receipt, dict)
             and receipt.get("schema") == "onnx-splitpoint/hailo-hef-build-receipt/v2"
             and _receipt_publication_flags(receipt, positive_original=legacy_positive_original)
             and receipt.get("cache_key") == evidence["cache_key_v3"]
             and ("cache_key" not in build or build["cache_key"] == evidence["cache_key_v3"])
             and receipt.get("cache_payload") == payload
             and receipt.get("source_onnx_sha256") == key["builder_source_onnx_sha256"]
             and receipt.get("compiler_onnx_sha256") == key["compiler_onnx_sha256"]
             and receipt.get("compiler_onnx_filename") == PurePosixPath(build["compiler_onnx_path"]).name
             and receipt.get("hef_sha256") == bindings["part1_runtime"]["sha256"]
             and type(receipt.get("hef_size_bytes")) is int
             and receipt["hef_size_bytes"] == bindings["part1_runtime"]["size_bytes"]
             and receipt.get("hw_arch") == payload["hw_arch"] == "hailo8"
             and receipt.get("net_name") == payload["net_name"]
             and receipt.get("hailo_sdk_version") == payload["hailo_sdk_version"]
             and all(receipt.get(name, _MISSING) == payload[name] for name in (
                 "calibration_identity", "prepared_calibration_identity_sha256",
                 "calibration_count", "requested_calibration_count", "calibration_storage",
                 "calibration_memory_cap_bytes", "preprocessing_contract", "preprocessing_contract_sha256")),
             "historical compiler provenance differs from actual measured receipt")
    # Structural equality alone would equate JSON true with 1. Recheck the
    # existing canonical cache identity against the actual receipt payload.
    from .build_evidence import canonical_sha256
    _require(canonical_sha256(receipt["cache_payload"]) == receipt["cache_key"],
             "actual receipt cache payload identity differs")
    return {**receipt_binding, "cache_key": receipt["cache_key"],
            "source_onnx_sha256": receipt["source_onnx_sha256"],
            "compiler_onnx_sha256": receipt["compiler_onnx_sha256"],
            "hef_sha256": receipt["hef_sha256"], "hef_size_bytes": receipt["hef_size_bytes"],
            "net_name": receipt["net_name"]}


def _h8_receipt_input_authority(raw, bindings, original, staged, prepared, contract,
                              *, model, case, full_model_sha256):
    """Bind native H8 input dimensions to the own measured HEF receipt."""
    from .build_evidence import canonical_sha256
    from .preprocessing_contract import resolve_image_preprocessing_contract

    bound = bindings["part1_build_receipt"]
    _require(isinstance(raw, bytes) and 0 < len(raw) <= 2 * 1024**2,
             "measured H8 input receipt required")
    _require(len(raw) == bound["size_bytes"] and hashlib.sha256(raw).hexdigest() == bound["sha256"],
             "measured H8 input receipt bytes differ")
    def invalid_constant(value):
        raise ValueError("Generic split-manifest staging: nonfinite H8 receipt constant: " + value)
    receipt = json.loads(raw, object_pairs_hook=_object, parse_constant=invalid_constant)
    payload = receipt.get("cache_payload") or {}
    legacy_positive = None
    if "publish_artifacts" not in receipt and "diagnostic_only" not in receipt:
        build = ((original.get("hailo") or {}).get("hefs") or {}).get("hailo8", {}).get("part1_build") or {}
        _require(("publish_artifacts" not in build or build["publish_artifacts"] is True)
                 and ("diagnostic_only" not in build or build["diagnostic_only"] is False),
                 "legacy receipt original build is unpublished or diagnostic")
        legacy_positive = _positive_original_build(build, original, arch="hailo8", model=model,
            case=case, bindings=bindings, full_model_sha256=full_model_sha256, prepared=prepared)
    shape = [1, 3, *prepared["shape"][:2]]
    _require(receipt.get("schema") == "onnx-splitpoint/hailo-hef-build-receipt/v2"
             and receipt.get("hw_arch") == payload.get("hw_arch") == "hailo8"
             and _receipt_publication_flags(receipt, positive_original=legacy_positive is not None)
             and receipt.get("source_onnx_sha256") == bindings["source_part1_onnx"]["sha256"]
             and _sha(receipt.get("compiler_onnx_sha256"))
             and payload.get("model_sha256") == receipt["compiler_onnx_sha256"]
             and receipt.get("hef_sha256") == bindings["part1_runtime"]["sha256"]
             and type(receipt.get("hef_size_bytes")) is int
             and receipt["hef_size_bytes"] == bindings["part1_runtime"]["size_bytes"]
             and receipt.get("net_name") == payload.get("net_name") == PurePosixPath(original["part1"]).stem
             and payload.get("schema") == "onnx-splitpoint/hailo-hef-cache-key-v3"
             and _sha(receipt.get("cache_key")) and canonical_sha256(payload) == receipt["cache_key"]
             and payload.get("net_input_shapes") == shape
             and all(type(dim) is int for dim in payload["net_input_shapes"]),
             "measured H8 input receipt identity or shape differs")
    name = prepared.get("input_name")
    _require(isinstance(name, str) and bool(name)
             and all(document.get("orig_inputs") == [name]
                     and document.get("part1_external_inputs") == [name]
                     and (document.get("io") or {}).get("part1_external_inputs") == [name]
                     for document in (original, staged))
             and prepared.get("format") == "raw" and prepared.get("c_contiguous") is True
             and contract.get("task") == "detection"
             and contract.get("normalization") == "divide_255_float32"
             and contract.get("source_domain") == "uint8_0_255",
             "measured H8 original input name or encoding differs")
    preprocessing, preprocessing_sha = resolve_image_preprocessing_contract(
        task="detection", target_hw=prepared["shape"][:2], declared=receipt.get("preprocessing_contract"))
    _require(receipt.get("preprocessing_contract") == payload.get("preprocessing_contract")
             == contract.get("image_preprocessing_contract") == preprocessing
             and receipt.get("preprocessing_contract_sha256")
             == payload.get("preprocessing_contract_sha256") == preprocessing_sha
             and preprocessing.get("image_scale") == "norm",
             "measured H8 preprocessing identity differs")
    return {"authority": "bound_own_hailo8_build_receipt", "receipt": deepcopy(bound),
            "source_part1_onnx": deepcopy(bindings["source_part1_onnx"]),
            "hef": deepcopy(bindings["part1_runtime"]), "cache_key": receipt["cache_key"],
            "net_name": receipt["net_name"], "net_input_shapes": shape,
            "input_name": name, "preprocessing_contract_sha256": preprocessing_sha,
            "legacy_publication_flags_absent": legacy_positive is not None,
            "positive_original_build_for_legacy_flags": legacy_positive}


def _positive_original_build(build, original, *, arch, model, case, bindings,
                             full_model_sha256, prepared):
    """Retain an actual original compile or strict successful reuse record."""
    _require(build.get("ok") is True, "original Hailo build is not positive")
    if build.get("cache_hit") is False and build.get("skipped") is False:
        return _historical_build_provenance(build, original, arch=arch, model=model, case=case,
            source_sha256=bindings["source_part1_onnx"]["sha256"],
            full_model_sha256=full_model_sha256, prepared=prepared)
    fields = {"build_receipt_path", "cache_hit", "calib_source", "compiler_onnx_path",
        "context_mode", "elapsed_s", "mapping_failed", "multi_context_used", "ok",
        "single_context_failed", "single_context_used", "skipped", "source_onnx_path",
        "timed_out", "watchdog_expired"}
    source = PurePosixPath(str(build.get("source_onnx_path") or ""))
    _require(set(build) in (fields, fields | {"cache_key"})
             and build.get("cache_hit") is True and build.get("skipped") is True
             and build.get("context_mode") == "skipped" and build.get("calib_source") == "validated_existing_artifact"
             and all(build.get(key) is False for key in ("single_context_failed", "single_context_used",
                 "multi_context_used", "mapping_failed", "timed_out", "watchdog_expired"))
             and type(build.get("elapsed_s")) in (int, float)
             and math.isfinite(build["elapsed_s"]) and build["elapsed_s"] >= 0
             and source.is_absolute() and ".." not in source.parts
             and source.parent.name == case and source.name == original["part1"]
             and build.get("compiler_onnx_path") == str(source)
             and build.get("build_receipt_path") == str(source.parent / f"hailo/{arch}/part1/hailo_hef_build_receipt.json")
             and ("cache_key" not in build or _sha(build["cache_key"])),
             "unproved original positive Hailo reuse record")
    return {"backend_metadata": arch, "historical_observation": deepcopy(build),
            "historical_reuse_observation": True, "executed_artifact_identity_proved": False,
            "compiler_source_byte_equality_claimed": False}


def _omitted_deepx_provenance(original, *, source_sha256, prepared, case):
    """Retain a known positive unused backend record, without executing it."""
    dx = original["deepx"]["part1_dxnn"]
    _require(set(original["deepx"]) == {"part1_dxnn"} and set(dx) == {
        "ok", "status", "build_status", "dxnn_path", "config_path", "output_contract",
        "output_contract_sha256", "source_onnx_sha256", "build_onnx_sha256", "build_onnx",
        "build_onnx_adapter_receipt", "dxnn_sha256", "dxnn_size_bytes", "cache_key",
        "cache_schema_version", "cache_contract", "calibration_task", "calibration_manifest",
        "calibration_manifest_identity", "classification_preprocessing", "preprocessing_contract",
        "output_names", "cache_lookup",
    }, "unknown omitted DeepX metadata")
    contract, lookup = dx["cache_contract"], dx["cache_lookup"]
    _require(dx["ok"] is True and dx["status"] == "ok"
             and dx["build_status"] in {"ready_reused", "ready_built"}
             and dx["source_onnx_sha256"] == dx["build_onnx_sha256"] == source_sha256
             and dx["build_onnx"] == original["part1"]
             and dx["build_onnx_adapter_receipt"] == ""
             and dx["dxnn_path"] == original["deepx_part1_dxnn"] == "deepx/deepx_m1/part1/model.dxnn"
             and dx["output_contract"] == original["deepx_part1_output_contract"] == "deepx/deepx_m1/part1/output_contract.json"
             and dx["config_path"] == "deepx/deepx_m1/part1/config_deepx.json"
             and _sha(dx["dxnn_sha256"]) and _sha(dx["output_contract_sha256"])
             and type(dx["dxnn_size_bytes"]) is int and dx["dxnn_size_bytes"] > 0
             and type(dx["cache_schema_version"]) is int and dx["cache_schema_version"] == 2
             and dx["output_names"] == original["part1_cut_names"],
             "unproved positive omitted DeepX artifact metadata")
    _require(set(contract) == {"schema", "schema_version", "task", "calibration_manifest_identity",
        "calibration_count", "calibration_method", "calibration_dir", "preprocessing_contract",
        "compiler_identity", "input_name", "input_shape", "case_id"}
             and contract["schema"] == "onnx-splitpoint/deepx-part1-cache-contract"
             and type(contract["schema_version"]) is int and contract["schema_version"] == 2
             and contract["task"] == dx["calibration_task"] == "detection"
             and contract["case_id"] == case and contract["input_name"] == prepared["input_name"]
             and contract["input_shape"] == [1, 3, *prepared["shape"][:2]]
             and all(type(dim) is int for dim in contract["input_shape"])
             and type(contract["calibration_count"]) is int
             and contract["calibration_count"] == original["hailo"]["config"]["calib_count"]
             and contract["calibration_dir"] == original["hailo"]["config"]["calib_dir"]
             and contract["calibration_method"] == "ema"
             and contract["preprocessing_contract"] == "dxcom_image_loader_v2:letterbox114-rgb-hwc"
             and isinstance(contract["compiler_identity"], str) and bool(contract["compiler_identity"])
             and contract["calibration_manifest_identity"] == dx["calibration_manifest_identity"]
             and re.fullmatch(r"sha256:[0-9a-f]{64}", dx["calibration_manifest_identity"])
             and dx["classification_preprocessing"] == "not_applicable"
             and type(dx["preprocessing_contract"]["schema_version"]) is int
             and dx["preprocessing_contract"] == {"schema": "onnx-splitpoint/deepx-detection-preprocessing",
                 "schema_version": 1, "mode": "letterbox114_div255"},
             "unproved omitted DeepX model/input/calibration contract")
    # A successful historical build retains the MISS that preceded compilation.
    # Preserve that observation; omitting it never claims a current cache hit.
    artifact = PurePosixPath(str(lookup.get("artifact") or ""))
    _require(set(lookup) == {"schema", "schema_version", "outcome", "role", "model_id",
                           "case_id", "identity", "reason", "artifact"}
             and lookup["schema"] == "onnx-splitpoint/deepx-cache-lookup"
             and type(lookup["schema_version"]) is int and lookup["schema_version"] == 1
             and lookup["role"] == "part1" and lookup["case_id"] == case
             and lookup["model_id"] == "benchmark_set"
             and lookup["identity"] == dx["cache_key"]
             and re.fullmatch(r"deepx_m1_part1_" + case + r"_v2_[0-9a-f]{32}", dx["cache_key"])
             and ((dx["build_status"] == "ready_reused"
                   and lookup["outcome"] == "HIT" and lookup["reason"] == "compatible"
                   and artifact.is_absolute() and ".." not in artifact.parts
                   and artifact.parts[-2:] == (dx["cache_key"], "model.dxnn"))
                  or (dx["build_status"] == "ready_built"
                      and lookup["outcome"] == "MISS" and lookup["reason"] == "model_missing"
                      and lookup["artifact"] == "")),
             "unknown or nonpositive omitted DeepX cache observation")
    return {"backend_metadata": "deepx", "historical_observation": deepcopy(original["deepx"]),
            "current_compile_claimed": False, "current_reuse_claimed": False,
            "current_quality_or_hardware_pass_claimed": False}


def _validate_generic_split_manifest_staging(
    original_bytes, staged_bytes, *, original_source, staged_source, row,
    original_artifacts, full_model_sha256, producer_model, boundary_inputs,
    part1_build_receipt_bytes=None,
):
    """Return provenance for actual byte-bound staging, or reject any drift.

    Inputs are supplied by the normal reader after its existing artifact and
    Quality validation.  This function is pure: it neither opens artifacts nor
    changes a descriptor, manifest, plan, admission or scientific result.
    """
    original, old_source = _document(original_bytes, original_source)
    staged, new_source = _document(staged_bytes, staged_source)
    identity = row.get("identity") or {}
    _require(row.get("admission_mode") == "generic_only" and row.get("task") == "detection"
             and identity.get("comparison_backend") == "hailo8"
             and identity.get("direction") == "hailo8_to_trt"
             and identity.get("setup_id") == "orin_nx_hailo8_01"
             and identity.get("precision") == "float32_layout_fp16",
             "unreviewed execution scope")
    case, model = identity.get("case_id"), identity.get("model_id")
    _require(isinstance(case, str) and re.fullmatch(r"b[0-9]+", case)
             and isinstance(model, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", model),
             "model/cut identity missing")
    old_root, new_root = PurePosixPath(old_source["path"]).parent, PurePosixPath(new_source["path"]).parent
    _require(old_root.name == new_root.name == case, "manifest belongs to another cut")
    artifacts = (row.get("generic_artifact_binding") or {}).get("artifacts") or {}
    boundary_metadata = artifacts.get("boundary_metadata") or {}
    locator_provenance = _boundary_metadata_locator(boundary_metadata, old_source, case=case)
    bindings = {}
    for role in _ROLES:
        before, measured = original_artifacts.get(role) or {}, artifacts.get(role) or {}
        _require(_sha(before.get("sha256")) and measured.get("sha256") == before["sha256"]
                 and type(before.get("size_bytes")) is int and before["size_bytes"] > 0
                 and measured.get("size_bytes") == before["size_bytes"],
                 "original compiled/source artifact differs: " + role)
        bindings[role] = {"sha256": before["sha256"], "size_bytes": before["size_bytes"]}
    _require(_sha(full_model_sha256) and producer_model.get("source_onnx_sha256") == full_model_sha256
             and producer_model.get("part1_onnx_sha256") == bindings["source_part1_onnx"]["sha256"]
             and producer_model.get("part2_onnx_sha256") == bindings["source_part2_onnx"]["sha256"],
             "selected full/split model identity differs")
    graph_case = _manifest_graph_case(original, staged, model=model, case=case)
    for document in (original, staged):
        _require(PurePosixPath(str(document.get("full_model") or "")).name == model + ".onnx",
                 "manifest model/cut identity differs")
    for role, key in (("source_part1_onnx", "part1"), ("source_part2_onnx", "part2")):
        _require(artifacts[role].get("local_path") == str(new_root / staged[key]),
                 "measured split model is not the staged source: " + role)
    boundary = (row.get("generic_artifact_binding") or {}).get("boundary_contract") or {}
    expected_inputs = {item["name"]: {key: item[key] for key in ("shape", "dtype")}
                       for item in boundary_inputs}
    _require(len(expected_inputs) == len(boundary_inputs) > 0
             and boundary.get("part2_inputs") == expected_inputs
             and row.get("boundary_observation") == boundary
             and isinstance(boundary.get("passthrough_inputs"), dict),
             "original typed boundary inputs differ")
    p1, p2 = staged.get("part1_cut_names"), staged.get("part2_cut_names")
    passthrough = boundary["passthrough_inputs"]
    _require(isinstance(p1, list) and isinstance(p2, list) and len(set(p1)) == len(p1)
             and len(set(p2)) == len(p2) == len(p1)
             and set(p1) == set(boundary.get("part1_outputs") or {})
             and not set(p2) & set(passthrough)
             and set(p2) | set(passthrough) == set(expected_inputs)
             and boundary.get("part2_to_part1") == dict(zip(p2, p1))
             and all(boundary["part1_outputs"][name] == expected_inputs[target]
                     for target, name in zip(p2, p1)), "complete cut tensor mapping differs")
    canonical_original_inputs = []
    if passthrough:
        external = (staged.get("io") or {}).get("part2_external_inputs")
        quality = row.get("quality_reference") or {}
        artifact_binding = row.get("generic_artifact_binding") or {}
        input_binding = row.get("input_binding") or {}
        canonical = row.get("canonical_full_input_bindings")
        _require(isinstance(external, list) and len(set(external)) == len(external)
                 and set(external) == set(expected_inputs)
                 and all(target == source and source in staged.get("orig_inputs", [])
                         for target, source in passthrough.items())
                 and isinstance(canonical, list) and len(canonical) == 2
                 and quality.get("request") == artifact_binding.get("producer_source")
                 and isinstance(quality.get("request"), dict),
                 "canonical original-input source/manifest partition missing")
        for item in canonical:
            image = item.get("source_image") or {}
            path = PurePosixPath(str(image.get("local_path") or image.get("path") or ""))
            snapshots = item.get("inputs") or {}
            _require(item.get("schema") == "onnx-splitpoint/generic-canonical-full-input/v1"
                     and item.get("producer_source") == quality["request"]
                     and item.get("preprocessing_contract") == (
                         input_binding.get("prepared_input_contract") or {}).get("image_preprocessing_contract")
                     and path.is_absolute() and ".." not in path.parts and _sha(image.get("sha256"))
                     and _sha(item.get("prepared_input_sha256"))
                     and set(snapshots) == set(passthrough.values())
                     and all(isinstance(snapshots[source], dict)
                             and _sha(snapshots[source].get("tensor_sha256"))
                             and all(snapshots[source].get(key) == expected_inputs[target][key]
                                     for key in ("shape", "dtype"))
                             for target, source in passthrough.items()),
                     "canonical original-input bytes/numeric contract differ")
        _require(canonical[0].get("image_id") != canonical[1].get("image_id")
                 and canonical[0].get("prepared_input_sha256") == (
                     input_binding.get("prepared_input") or {}).get("sha256")
                 and canonical[0]["source_image"]["sha256"] == input_binding.get("source_image_sha256"),
                 "canonical original-input baseline/mixed image differs")
        canonical_original_inputs = deepcopy(canonical)
    feed = row.get("input_binding") or {}
    prepared, contract = feed.get("prepared_input") or {}, feed.get("prepared_input_contract") or {}
    # A fresh normal H8-only generation attaches the existing measured receipt
    # and emits no records for unused compilers. This is a distinct, bounded
    # relation from restaging the old multi-backend catalogue wholesale.
    dx_fields = {"deepx", "deepx_part1_dxnn", "deepx_part1_output_contract"}
    compact_attachment = (not dx_fields.intersection(staged)
        and set(staged.get("hailo", {}).get("hefs", {})) == {"hailo8"}
        and set(staged.get("hailo", {}).get("case_variant_availability", {})) == {"hailo8"})
    imported_hint = (not dx_fields.intersection(staged)
        and "part1_original_reuse" in staged.get("hailo", {}).get("hefs", {}).get("hailo8", {}))
    compact_attachment = compact_attachment and not imported_hint
    input_manifest = original if compact_attachment else staged
    _require(feed == (row.get("runtime_config") or {}).get("input_binding")
             and _sha(prepared.get("sha256"))
             and (row.get("functional_reference") or {}).get("input_sha256") == prepared["sha256"]
             and _sha(feed.get("source_image_sha256"))
             and contract.get("source_image_sha256") == feed["source_image_sha256"]
             and prepared.get("input_name") in staged.get("orig_inputs", [])
             and prepared.get("dtype") == contract.get("dtype") == "uint8"
             and prepared.get("layout") == contract.get("layout") == "HWC"
             and isinstance(prepared.get("shape"), list) and len(prepared["shape"]) == 3
             and prepared["shape"] == contract.get("shape")
             and all(type(dim) is int and dim > 0 for dim in prepared["shape"])
             and prepared["shape"][-1] == 3 and feed.get("runtime_input_format") == "float32"
             and prepared.get("size_bytes") == math.prod(prepared["shape"])
             and contract.get("source_image") == feed.get("source_image"),
             "actual typed timing input binding differs")
    h8_input_authority = None
    if imported_hint:
        h8_input_authority = _h8_receipt_input_authority(part1_build_receipt_bytes,
            bindings, original, staged, prepared, contract,
            model=model, case=case, full_model_sha256=full_model_sha256)
    else:
        _require(input_manifest["deepx"]["part1_dxnn"]["cache_contract"]["input_shape"]
                 == [1, 3, *prepared["shape"][:2]], "actual typed timing input binding differs")

    # Replace only individually validated metadata differences in a temporary
    # comparison copy. Both original documents and file hashes remain intact.
    expected = deepcopy(original)
    changes = []
    def permit(parts, reason):
        left, right = expected, staged
        for part in parts[:-1]:
            left, right = left[part], right[part]
        key = parts[-1]
        before, after = left.get(key, _MISSING), right.get(key, _MISSING)
        if before == after:
            return
        change = {"pointer": "/" + "/".join(parts), "reason": reason,
                  "original_present": before is not _MISSING, "staged_present": after is not _MISSING}
        if before is not _MISSING and not isinstance(before, (dict, list)):
            change["original_value"] = before
        if after is not _MISSING and not isinstance(after, (dict, list)):
            change["staged_value"] = after
        changes.append(change)
        if after is _MISSING:
            del left[key]
        else:
            left[key] = deepcopy(after)
    datetime.fromisoformat(original["created_at"])
    datetime.fromisoformat(staged["created_at"])
    permit(("created_at",), "new normal staging timestamp")
    _require(all(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", d["tool"]["core"])
                 for d in (original, staged)), "invalid tool metadata")
    permit(("tool", "core"), "controller version metadata; graph/source bytes separately bound")
    omitted_unexecuted_backend_metadata = []
    old_dx = original.get("deepx", {}).get("part1_dxnn") if imported_hint else original["deepx"]["part1_dxnn"]
    if imported_hint:
        _require(dx_fields.intersection(original) in (set(), dx_fields), "incomplete original DeepX authority")
        if dx_fields.intersection(original):
            omitted_unexecuted_backend_metadata.append(_omitted_deepx_provenance(
                original, source_sha256=bindings["source_part1_onnx"]["sha256"], prepared=prepared, case=case))
            for key in sorted(dx_fields):
                permit((key,), "positive unused DeepX metadata retained in original provenance")
        backend_fields = {"backend_terminal_states", "case_variant_availability", "config", "hefs"}
        full_fields = {"full_end_node_names", "full_endpoint_mode", "full_output_contract"}
        _require(set(original["hailo"]) in (backend_fields, backend_fields | full_fields)
                 and set(staged["hailo"]) in (backend_fields,
                     backend_fields | {"builds_deferred_until_cache_preflight"})
                 and ("builds_deferred_until_cache_preflight" not in staged["hailo"]
                      or staged["hailo"]["builds_deferred_until_cache_preflight"] is True)
                 and set(staged["hailo"]["hefs"]) <= {"hailo8", "hailo10"}
                 and set(staged["hailo"]["case_variant_availability"]) ==
                     ({"hailo8", "hailo10h"} if "hailo10" in staged["hailo"]["hefs"] else {"hailo8"}),
                 "unreviewed original-H8 hint backend scope")
        permit(("hailo", "builds_deferred_until_cache_preflight"), "normal deferred import metadata")
        _require(original["hailo"]["config"]["build"].get("part1") is True
                 and original["hailo"]["config"]["build"].get("part2") is False
                 and type(original["hailo"]["config"]["build"].get("full")) is bool,
                 "unreviewed original-H8 hint build scope")
        permit(("hailo", "config", "build", "full"), "unexecuted Full scope not requested by import")
        for key in sorted(full_fields.intersection(original["hailo"])):
            permit(("hailo", key), "unexecuted Full metadata remains in original authority")
    if compact_attachment:
        omitted_unexecuted_backend_metadata.append(_omitted_deepx_provenance(
            original, source_sha256=bindings["source_part1_onnx"]["sha256"],
            prepared=prepared, case=case))
        for key in sorted(dx_fields):
            permit((key,), "known positive unused DeepX metadata retained in provenance, omitted by H8-only generation")
        current_build = staged["hailo"]["hefs"]["hailo8"]["part1_build"]
        _require(set(current_build) == {"build_receipt_path", "cache_hit", "calib_source",
            "compiler_onnx_path", "context_mode", "elapsed_s", "mapping_failed", "multi_context_used",
            "ok", "single_context_failed", "single_context_used", "skipped", "source_onnx_path",
            "timed_out", "watchdog_expired"}, "unknown compact actual-artifact attachment fields")
        backend_fields = {"backend_terminal_states", "case_variant_availability", "config", "hefs"}
        full_fields = {"full_end_node_names", "full_endpoint_mode", "full_output_contract"}
        _require(set(original["hailo"]) in (backend_fields, backend_fields | full_fields)
                 and set(staged["hailo"]) == backend_fields,
                 "unknown compact backend omission scope")
        original_build = original["hailo"]["config"]["build"]
        _require(set(original_build) == {"full", "part1", "part2"}
                 and type(original_build["full"]) is bool
                 and original_build["part1"] is True and original_build["part2"] is False,
                 "compact generation must only contract known unexecuted Full scope")
        if original_build["full"]:
            permit(("hailo", "config", "build", "full"), "normal H8-only generation does not request Full")
        for key in sorted(full_fields.intersection(original["hailo"])):
            permit(("hailo", key), "unexecuted Full metadata retained in original manifest authority")
    elif not imported_hint:
        new_dx = staged["deepx"]["part1_dxnn"]
    if not compact_attachment and not imported_hint and (old_dx["build_status"] != new_dx["build_status"] or old_dx["cache_lookup"] != new_dx["cache_lookup"]):
        lookup = new_dx["cache_lookup"]
        _require(old_dx["build_status"] == "ready_built" and new_dx["build_status"] == "ready_reused"
                 and old_dx["cache_lookup"]["outcome"] == "MISS"
                 and old_dx["cache_lookup"]["reason"] == "model_missing"
                 and old_dx["cache_lookup"]["artifact"] == ""
                 and lookup["outcome"] == "HIT" and lookup["reason"] == "compatible"
                 and PurePosixPath(lookup["artifact"]).is_absolute()
                 and ".." not in PurePosixPath(lookup["artifact"]).parts
                 and PurePosixPath(lookup["artifact"]).parts[-2:] == (new_dx["cache_key"], "model.dxnn"),
                 "unreviewed DeepX reuse metadata")
        permit(("deepx", "part1_dxnn", "build_status"), "same unused-backend artifact reused")
        for key in ("artifact", "outcome", "reason"):
            permit(("deepx", "part1_dxnn", "cache_lookup", key), "same unused-backend artifact lookup metadata")
    historical_build_reuse = []
    retained_historical_build_metadata = []
    from .build_evidence import canonical_json_bytes
    for arch in ("hailo8", "hailo10"):
        if imported_hint and arch not in original["hailo"]["hefs"] and arch not in staged["hailo"]["hefs"]:
            continue
        old_build = original["hailo"]["hefs"][arch]["part1_build"]
        if imported_hint:
            old_hef, new_hef = original["hailo"]["hefs"][arch], staged["hailo"]["hefs"][arch]
            _require(set(old_hef) <= _FULL_FIELDS | {"part1", "part1_build"}
                     and old_hef.get("part1") == new_hef.get("part1") == f"hailo/{arch}/part1/compiled.hef",
                     "original-H8 hint artifact locator differs")
            proof = _positive_original_build(old_build, original, arch=arch, model=model, case=case,
                bindings=bindings, full_model_sha256=full_model_sha256, prepared=prepared)
            if arch == "hailo8":
                _require(set(new_hef) == {"part1", "part1_original_reuse"}
                         and new_hef["part1_original_reuse"] == {
                             "receipt": "hailo/hailo8/part1/hailo_hef_build_receipt.json",
                             "provenance": "hailo/hailo8/part1/generic_quality_original_hailo_reuse.json"}
                         and old_build.get("cache_key", h8_input_authority["cache_key"]) == h8_input_authority["cache_key"],
                         "original-H8 receipt hint differs")
                if proof.get("historical_reuse_observation") is not True:
                    proof["actual_receipt_binding"] = _bind_historical_receipt(part1_build_receipt_bytes, old_build, bindings,
                        legacy_positive_original=h8_input_authority["legacy_publication_flags_absent"])
                    historical_build_reuse.append(proof)
                permit(("hailo", "hefs", arch, "part1_build"), "original build retained in provenance; no current build record invented")
                permit(("hailo", "hefs", arch, "part1_original_reuse"), "own measured HEF/receipt attached through normal import hint")
            else:
                new_build = new_hef.get("part1_build") or {}
                fields = {"cache_hit", "cache_key", "context_mode", "elapsed_s", "mapping_failed",
                    "multi_context_used", "ok", "single_context_failed", "single_context_used",
                    "skipped", "timed_out", "watchdog_expired"}
                if set(new_build) != fields:
                    _require(set(new_hef) == {"part1", "part1_build"}
                             and new_build.get("cache_hit") is True
                             and new_build.get("ok") is True and new_build.get("skipped") is True,
                             "unproved positive unused H10 import reuse record")
                    staged_reuse = _positive_original_build(new_build, staged, arch=arch,
                        model=model, case=case, bindings=bindings,
                        full_model_sha256=full_model_sha256, prepared=prepared)
                    _require(staged_reuse.get("historical_reuse_observation") is True
                             and ("cache_key" not in new_build
                                  or ("cache_key" not in old_build
                                      or new_build["cache_key"] == old_build["cache_key"])),
                             "unused H10 reuse record cache identity conflicts")
                    retained_historical_build_metadata.append({**proof,
                        "build_record_unchanged": False,
                        "staged_unexecuted_reuse_record": deepcopy(new_build),
                        "staged_reuse_record_provenance": staged_reuse,
                        "staged_cache_key_status": "unverified_unused_reuse_metadata",
                        "staged_cache_key_verified": False,
                        "current_compile_claimed": False, "current_reuse_claimed": False})
                    permit(("hailo", "hefs", arch, "part1_build"),
                           "positive unused H10 reuse metadata; original history retained without a current cache claim")
                    for key in sorted(set(old_hef) & _FULL_FIELDS):
                        permit(("hailo", "hefs", arch, key), "unexecuted H10 Full metadata remains in original authority")
                    continue
                # An old positive reuse observation can predate cache-key
                # recording. Its new unused-backend summary is not authority
                # for a cache hit: retain that extra key explicitly unverified.
                unverified_unused_key = ("cache_key" not in old_build
                    and proof.get("historical_reuse_observation") is True)
                _require(set(new_hef) == {"part1", "part1_build"} and set(new_build) == fields
                         and new_build["cache_hit"] is new_build["ok"] is new_build["skipped"] is True
                         and new_build["context_mode"] == "skipped"
                         and _sha(new_build["cache_key"])
                         and (unverified_unused_key or new_build["cache_key"] == old_build.get("cache_key"))
                         and all(new_build[key] is False for key in ("mapping_failed", "multi_context_used",
                             "single_context_failed", "single_context_used", "timed_out", "watchdog_expired"))
                         and type(new_build["elapsed_s"]) in (int, float)
                         and math.isfinite(new_build["elapsed_s"]) and new_build["elapsed_s"] >= 0,
                         "unproved positive unused H10 import summary")
                retained_historical_build_metadata.append({**proof, "build_record_unchanged": False,
                    "staged_unexecuted_summary": deepcopy(new_build), "current_compile_claimed": False,
                    "current_reuse_claimed": False,
                    "staged_cache_key_status": ("unverified_unused_metadata" if unverified_unused_key
                        else "matches_original_recorded_cache_key"),
                    "staged_cache_key_verified": not unverified_unused_key})
                permit(("hailo", "hefs", arch, "part1_build"), "positive unused H10 summary; original build retained in provenance")
                for key in sorted(set(old_hef) & _FULL_FIELDS):
                    permit(("hailo", "hefs", arch, key), "unexecuted H10 Full metadata remains in original authority")
            continue
        if compact_attachment and arch == "hailo10":
            old_hef = original["hailo"]["hefs"][arch]
            _require(set(old_hef) <= _FULL_FIELDS | {"part1", "part1_build"}
                     and set(old_build) <= set(current_build) | _BUILD_DIAGNOSTICS
                     and old_hef["part1"] == "hailo/hailo10/part1/compiled.hef"
                     and old_build.get("ok") is True
                     and type(old_build.get("elapsed_s")) in (int, float)
                     and math.isfinite(old_build["elapsed_s"]) and old_build["elapsed_s"] >= 0,
                     "unknown or nonpositive omitted historical H10 build")
            if old_build.get("cache_hit") is True and old_build.get("skipped") is True:
                proof = _omitted_h10_reuse_provenance(old_build, original, case=case)
            else:
                _require(old_build.get("cache_hit") is False and old_build.get("skipped") is False,
                         "unknown or nonpositive omitted historical H10 build")
                proof = _historical_build_provenance(old_build, original, arch=arch, model=model, case=case,
                    source_sha256=bindings["source_part1_onnx"]["sha256"],
                    full_model_sha256=full_model_sha256, prepared=prepared)
            omitted_unexecuted_backend_metadata.append({"backend_metadata": arch,
                "historical_observation": deepcopy(old_hef), "validated_historical_build": proof,
                "current_compile_claimed": False, "current_reuse_claimed": False,
                "current_quality_or_hardware_pass_claimed": False})
            permit(("hailo", "hefs", arch), "proven positive unused historical H10 metadata omitted by H8-only generation")
            continue
        new_build = staged["hailo"]["hefs"][arch]["part1_build"]
        reuse = lambda b: b.get("ok") is True and b.get("cache_hit") is True and b.get("skipped") is True
        historical = (old_build.get("ok") is True and old_build.get("cache_hit") is False
                      and old_build.get("skipped") is False)
        # This relation is explicitly scoped above to measured H8. A normal
        # source-only staging may retain the other backend's successful build
        # record verbatim; that is historical provenance, not a new reuse claim.
        from .build_evidence import canonical_json_bytes
        retained_historical = (arch == "hailo10" and historical
                               and canonical_json_bytes(old_build) == canonical_json_bytes(new_build))
        _require(retained_historical or reuse(new_build) and (reuse(old_build) or historical),
                 "part1 must remain actual artifact reuse or a proven successful historical build")
        _require(all(type(b.get("elapsed_s")) in (int, float) and math.isfinite(b["elapsed_s"])
                     and b["elapsed_s"] >= 0 for b in (old_build, new_build)), "invalid reuse elapsed time")
        if historical:
            proof = _historical_build_provenance(
                old_build, original, arch=arch, model=model, case=case,
                source_sha256=bindings["source_part1_onnx"]["sha256"],
                full_model_sha256=full_model_sha256, prepared=prepared)
            if retained_historical:
                retained_historical_build_metadata.append({
                    **proof, "build_record_unchanged": True,
                    "current_compile_claimed": False, "current_reuse_claimed": False,
                })
                continue
            if arch == "hailo8":
                proof["actual_receipt_binding"] = _bind_historical_receipt(
                    part1_build_receipt_bytes, old_build, bindings)
            historical_build_reuse.append(proof)
            _require(new_build.get("context_mode") == "skipped"
                     and new_build.get("calib_source") == "validated_existing_artifact"
                     and all(new_build.get(key) is False for key in (
                         "single_context_failed", "single_context_used", "multi_context_used",
                         "mapping_failed", "timed_out", "watchdog_expired")),
                     "current reuse contains compiler activity or failure")
            for key in ("cache_hit", "skipped", "calib_source", "context_mode",
                        "single_context_failed", "single_context_used", "multi_context_used"):
                permit(("hailo", "hefs", arch, "part1_build", key),
                       "successful historical compile to current artifact-reuse observation")
            for key in sorted(_BUILD_DIAGNOSTICS & old_build.keys()):
                _require(key not in new_build or new_build[key] == old_build[key],
                         "historical diagnostic may only be retained or omitted")
                permit(("hailo", "hefs", arch, "part1_build", key),
                       "historical compiler diagnostic retained in provenance, omitted by reuse")
        for key in ("source_onnx_path", "compiler_onnx_path", "build_receipt_path"):
            if old_build[key] != new_build[key]:
                relative = (f"hailo/{arch}/part1/hailo_hef_build_receipt.json"
                            if key == "build_receipt_path" else staged["part1"])
                old_relative = relative
                if historical and key == "compiler_onnx_path":
                    old_relative = (f"hailo/{arch}/part1/"
                                    f"{PurePosixPath(original['part1']).stem}_hailo_fixed.onnx")
                    if old_build[key] == old_build["source_onnx_path"]:
                        old_relative = staged["part1"]
                if historical:
                    old_case_root = PurePosixPath(old_build["source_onnx_path"]).parent
                    _require(old_build[key] == str(old_case_root / old_relative),
                             "historical compiler/source/receipt staging roots differ")
                old_path = PurePosixPath(old_build[key])
                _require(old_path.is_absolute() and ".." not in old_path.parts
                         and old_build[key].endswith("/" + case + "/" + old_relative)
                         and new_build[key] == str(new_root / relative), "unbound part1 staging path")
                permit(("hailo", "hefs", arch, "part1_build", key),
                       "part1 source/receipt staging; historical compiler graph identity retained separately")
        permit(("hailo", "hefs", arch, "part1_build", "elapsed_s"), "new artifact-reuse observation duration")
    # Source-only Generic parts do not execute the optional Full variants.
    # Allow only removal, never replacement, promotion or changes to part1.
    _require(staged["hailo"]["config"]["build"] == {"full": False, "part1": True, "part2": False},
             "unreviewed Hailo build/variant scope")
    removed_arches = set()
    demoted_unexecuted_full_claims = []
    for arch in ("hailo8", "hailo10h"):
        old_hef = original["hailo"]["hefs"].get(arch, {})
        new_hef = staged["hailo"]["hefs"].get(arch, {})
        removed = (set(old_hef) - set(new_hef)) - (
            {"part1_build"} if imported_hint and arch == "hailo8" else set())
        _require(removed <= _FULL_FIELDS, "non-Full backend field removed")
        changed_full = any(canonical_json_bytes(old_hef[key]) != canonical_json_bytes(new_hef[key])
                           for key in old_hef.keys() & new_hef.keys() & _FULL_FIELDS)
        if changed_full:
            from .hailo_full_contract_promotion import _demote_hailo_suite_full_claims
            _require(len(retained_historical_build_metadata) == 1
                     and any(p["backend_metadata"] == "hailo8" and "actual_receipt_binding" in p
                             for p in historical_build_reuse)
                     and old_hef.get("full_receipt_attested") is True
                     and old_hef.get("full_build", {}).get("ok") is True
                     and old_hef.get("full_output_contract", {}).get("artifact_binding_status") == "verified"
                     and old_hef["full_output_contract"].get("contract_status") == "recorded",
                     "unreviewed metadata or scientific manifest difference: unproved unexecuted Full claim demotion")
            demoted = _demote_hailo_suite_full_claims(
                {"hailo": {"hefs": {arch: old_hef}}})["hailo"]["hefs"][arch]
            # The archived receipt may remain as exact historical metadata;
            # both current attestation flags and the Full contract are demoted.
            if "full_build_receipt" in new_hef:
                _require("full_build_receipt" in old_hef
                         and canonical_json_bytes(new_hef["full_build_receipt"])
                             == canonical_json_bytes(old_hef["full_build_receipt"]),
                         "historical unexecuted Full receipt changed")
                demoted["full_build_receipt"] = deepcopy(old_hef["full_build_receipt"])
            _require(canonical_json_bytes({k: v for k, v in new_hef.items() if k in _FULL_FIELDS})
                     == canonical_json_bytes({k: v for k, v in demoted.items() if k in _FULL_FIELDS}),
                     "unreviewed unexecuted Full contract change")
            for key in sorted((old_hef.keys() | new_hef.keys()) & _FULL_FIELDS):
                permit(("hailo", "hefs", arch, key), "exact normal unexecuted Full receipt-claim demotion")
            demoted_unexecuted_full_claims.append({"backend_metadata": arch,
                "normal_demotion": "_demote_hailo_suite_full_claims",
                "current_full_execution_claimed": False})
            removed = set()  # Claim demotion is not removal of the historical variant.
        if removed:
            removed_arches.add(arch)
            if arch not in staged["hailo"]["hefs"]:
                _require(arch == "hailo10h" and set(old_hef) <= _FULL_FIELDS, "executed backend removed")
                permit(("hailo", "hefs", arch), "unused Full-only alias omitted from source-only suite")
            else:
                for key in sorted(removed):
                    permit(("hailo", "hefs", arch, key), "unexecuted Full variant metadata omitted")
            before = original["hailo"]["case_variant_availability"][arch]
            if compact_attachment and arch == "hailo10h":
                _require(set(before) == {"composed", "full", "full_error", "full_failed", "part1",
                    "part1_error", "part1_failed", "part2", "part2_error", "part2_failed"}
                         and before["full"] is before["part1"] is True
                         and all(before[key] is False for key in ("composed", "part2", "full_failed", "part1_failed", "part2_failed"))
                         and all(before[key] == "" for key in ("full_error", "part1_error", "part2_error")),
                         "unknown or failed omitted H10 availability")
                permit(("hailo", "case_variant_availability", arch), "positive unused historical H10 availability omitted")
                continue
            after = staged["hailo"]["case_variant_availability"][arch]
            _require(before["full"] is True and after["full"] is False,
                     "Full availability may only contract")
            permit(("hailo", "case_variant_availability", arch, "full"), "unexecuted Full variant is unavailable in source-only scope")
    old_states, new_states = original["hailo"]["backend_terminal_states"], staged["hailo"]["backend_terminal_states"]
    expected_states = [s for s in old_states if not (s.get("variant") == "full"
                       and s.get("hw_arch") in removed_arches)]
    omitted_unexecuted_states = []
    if new_states != expected_states:
        # The source-only generator may project its current terminal summary
        # to H8 part1 while retaining all historical availability/build data.
        # This is allowed only after proving measured reuse and the unchanged
        # positive other-backend build above; no failed/unknown state is hidden.
        def ready_state(value):
            return (isinstance(value, dict) and set(value) == {
                "available", "hw_arch", "reason", "status", "terminal", "variant"}
                and value["available"] is True and value["terminal"] is True
                and value["reason"] == "artifact_available" and value["status"] == "ready")
        measured_states = [s for s in expected_states
                           if s.get("hw_arch") == "hailo8" and s.get("variant") == "part1"]
        omitted_unexecuted_states = [s for s in expected_states if s not in measured_states]
        # An original selected H8 reuse has no cold-build proof to rebind.
        # Its already checked artifact/path relation may project the same
        # positive states only beside the exact omitted H10 reuse proof.
        measured_reuse_to_reuse = (compact_attachment
            and any(p["backend_metadata"] == "hailo10"
                    and p["validated_historical_build"].get("historical_reuse_observation") is True
                    for p in omitted_unexecuted_backend_metadata)
            and all(build.get("ok") is True and build.get("cache_hit") is True
                    and build.get("skipped") is True and build.get("context_mode") == "skipped"
                    and build.get("calib_source") == "validated_existing_artifact"
                    and all(build.get(key) is False for key in (
                        "single_context_failed", "single_context_used", "multi_context_used",
                        "mapping_failed", "timed_out", "watchdog_expired"))
                    for build in (original["hailo"]["hefs"]["hailo8"]["part1_build"],
                                  staged["hailo"]["hefs"]["hailo8"]["part1_build"])))
        _require((len(retained_historical_build_metadata) == 1 or compact_attachment
                  and any(p["backend_metadata"] == "hailo10" for p in omitted_unexecuted_backend_metadata))
                 and (any(p["backend_metadata"] == "hailo8" and "actual_receipt_binding" in p
                          for p in historical_build_reuse) or measured_reuse_to_reuse
                      or h8_input_authority is not None)
                 and len(measured_states) == len(new_states) == 1
                 and all(ready_state(s) for s in expected_states + new_states)
                 and len({(s["hw_arch"], s["variant"]) for s in expected_states}) == len(expected_states)
                 and new_states == measured_states
                 and all((s["variant"] == "full" and s["hw_arch"] in {"hailo8", "hailo10h"})
                         or (s["variant"] == "part1" and s["hw_arch"] == "hailo10h")
                         for s in omitted_unexecuted_states),
                 "backend terminal states changed outside proven unexecuted positive scope")
    permit(("hailo", "backend_terminal_states"), "only proven unexecuted variant terminal rows omitted")
    _require(canonical_json_bytes(expected) == canonical_json_bytes(staged),
             "unreviewed metadata or scientific manifest difference")
    return {
        "schema": "onnx-splitpoint/generic-split-manifest-staging-provenance/v1",
        "original_manifest": old_source, "staged_manifest": new_source,
        "original_boundary_metadata_artifact": deepcopy(boundary_metadata),
        "original_boundary_metadata_locator_provenance": locator_provenance,
        "normal_integer_boundary_graph_identity": {"display_case_id": case, "integer_boundary": int(case[1:]),
            "actual_graph_case_token": graph_case, "part1": original["part1"], "part2": original["part2"],
            "original_and_staged_graph_names_unchanged": True},
        "measured_identity": deepcopy(identity), "full_model_sha256": full_model_sha256,
        "unchanged_artifact_bindings": bindings, "boundary_contract": deepcopy(boundary),
        "canonical_original_input_bindings": canonical_original_inputs,
        "prepared_input": {k: prepared[k] for k in ("sha256", "shape", "dtype", "layout", "input_name")},
        "source_image_sha256": feed["source_image_sha256"], "allowed_differences": changes,
        "native_h8_input_authority": h8_input_authority,
        "historical_build_reuse": historical_build_reuse,
        "retained_historical_build_metadata": retained_historical_build_metadata,
        "omitted_unexecuted_backend_metadata": omitted_unexecuted_backend_metadata,
        "omitted_unexecuted_backend_terminal_states": deepcopy(omitted_unexecuted_states),
        "demoted_unexecuted_full_claims": demoted_unexecuted_full_claims,
        "scientific_manifest_fields_unchanged": True,
    }


def validate_generic_split_manifest_staging(
    original_bytes, staged_bytes, *, original_source, staged_source, row,
    original_artifacts, full_model_sha256, producer_model, boundary_inputs,
    part1_build_receipt_bytes=None,
):
    """Validate staging; historical measured H8 builds require exact receipt bytes.

    Reuse-to-reuse callers remain compatible without this optional input. The
    caller reads the already bound receipt; this helper performs no file I/O.
    """
    try:
        return _validate_generic_split_manifest_staging(
            original_bytes, staged_bytes, original_source=original_source,
            staged_source=staged_source, row=row, original_artifacts=original_artifacts,
            full_model_sha256=full_model_sha256, producer_model=producer_model,
            boundary_inputs=boundary_inputs,
            part1_build_receipt_bytes=part1_build_receipt_bytes,
        )
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Generic split-manifest staging: incomplete or malformed binding") from error
