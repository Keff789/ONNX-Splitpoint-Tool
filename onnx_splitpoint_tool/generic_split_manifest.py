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


def _validate_generic_split_manifest_staging(
    original_bytes, staged_bytes, *, original_source, staged_source, row,
    original_artifacts, full_model_sha256, producer_model, boundary_inputs,
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
    _require(boundary_metadata.get("sha256") == old_source["sha256"]
             and boundary_metadata.get("size_bytes") == old_source["size_bytes"]
             and boundary_metadata.get("local_path") == old_source["path"],
             "original bytes are not the actual boundary-metadata artifact")
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
    for document in (original, staged):
        _require(document.get("boundary") == document.get("boundary_index") == int(case[1:])
                 and document.get("part1") == f"{model}_part1_{case}.onnx"
                 and document.get("part2") == f"{model}_part2_{case}.onnx"
                 and PurePosixPath(str(document.get("full_model") or "")).name == model + ".onnx",
                 "manifest model/cut identity differs")
    for role, key in (("source_part1_onnx", "part1"), ("source_part2_onnx", "part2")):
        _require(artifacts[role].get("local_path") == str(new_root / staged[key]),
                 "measured split model is not the staged source: " + role)
    boundary = (row.get("generic_artifact_binding") or {}).get("boundary_contract") or {}
    expected_inputs = {item["name"]: {key: item[key] for key in ("shape", "dtype")}
                       for item in boundary_inputs}
    _require(len(expected_inputs) == len(boundary_inputs) > 0
             and boundary.get("part2_inputs") == expected_inputs
             and row.get("boundary_observation") == boundary and not boundary.get("passthrough_inputs"),
             "original typed boundary inputs differ")
    p1, p2 = staged.get("part1_cut_names"), staged.get("part2_cut_names")
    _require(isinstance(p1, list) and isinstance(p2, list) and len(set(p1)) == len(p1)
             and len(set(p2)) == len(p2) == len(p1) == len(expected_inputs)
             and set(p1) == set(boundary.get("part1_outputs") or {})
             and set(p2) == set(expected_inputs)
             and boundary.get("part2_to_part1") == dict(zip(p2, p1))
             and all(boundary["part1_outputs"][name] == expected_inputs[target]
                     for target, name in zip(p2, p1)), "complete cut tensor mapping differs")
    feed = row.get("input_binding") or {}
    prepared, contract = feed.get("prepared_input") or {}, feed.get("prepared_input_contract") or {}
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
             and contract.get("source_image") == feed.get("source_image")
             and staged["deepx"]["part1_dxnn"]["cache_contract"]["input_shape"]
                 == [1, 3, *prepared["shape"][:2]],
             "actual typed timing input binding differs")

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
    old_dx, new_dx = original["deepx"]["part1_dxnn"], staged["deepx"]["part1_dxnn"]
    if old_dx["build_status"] != new_dx["build_status"] or old_dx["cache_lookup"] != new_dx["cache_lookup"]:
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
    for arch in ("hailo8", "hailo10"):
        old_build = original["hailo"]["hefs"][arch]["part1_build"]
        new_build = staged["hailo"]["hefs"][arch]["part1_build"]
        _require(all(b.get("ok") is True and b.get("cache_hit") is True and b.get("skipped") is True
                     for b in (old_build, new_build)), "part1 must remain actual artifact reuse")
        for key in ("source_onnx_path", "compiler_onnx_path", "build_receipt_path"):
            if old_build[key] != new_build[key]:
                relative = (f"hailo/{arch}/part1/hailo_hef_build_receipt.json"
                            if key == "build_receipt_path" else staged["part1"])
                _require(old_build[key].endswith("/" + case + "/" + relative)
                         and new_build[key] == str(new_root / relative), "unbound part1 staging path")
                permit(("hailo", "hefs", arch, "part1_build", key), "same part1 source/receipt under new normal suite root")
        _require(all(type(b.get("elapsed_s")) in (int, float) and math.isfinite(b["elapsed_s"])
                     and b["elapsed_s"] >= 0 for b in (old_build, new_build)), "invalid reuse elapsed time")
        permit(("hailo", "hefs", arch, "part1_build", "elapsed_s"), "new artifact-reuse observation duration")
    # Source-only Generic parts do not execute the optional Full variants.
    # Allow only removal, never replacement, promotion or changes to part1.
    _require(staged["hailo"]["config"]["build"] == {"full": False, "part1": True, "part2": False},
             "unreviewed Hailo build/variant scope")
    removed_arches = set()
    for arch in ("hailo8", "hailo10h"):
        old_hef = original["hailo"]["hefs"].get(arch, {})
        new_hef = staged["hailo"]["hefs"].get(arch, {})
        removed = set(old_hef) - set(new_hef)
        _require(removed <= _FULL_FIELDS, "non-Full backend field removed")
        if removed:
            removed_arches.add(arch)
            if arch not in staged["hailo"]["hefs"]:
                _require(arch == "hailo10h" and set(old_hef) <= _FULL_FIELDS, "executed backend removed")
                permit(("hailo", "hefs", arch), "unused Full-only alias omitted from source-only suite")
            else:
                for key in sorted(removed):
                    permit(("hailo", "hefs", arch, key), "unexecuted Full variant metadata omitted")
            before = original["hailo"]["case_variant_availability"][arch]
            after = staged["hailo"]["case_variant_availability"][arch]
            _require(before["full"] is True and after["full"] is False,
                     "Full availability may only contract")
            permit(("hailo", "case_variant_availability", arch, "full"), "unexecuted Full variant is unavailable in source-only scope")
    old_states, new_states = original["hailo"]["backend_terminal_states"], staged["hailo"]["backend_terminal_states"]
    _require(new_states == [s for s in old_states if not (s.get("variant") == "full"
             and s.get("hw_arch") in removed_arches)], "backend terminal states changed outside removed Full scope")
    permit(("hailo", "backend_terminal_states"), "only removed Full variant terminal rows omitted")
    _require(expected == staged, "unreviewed metadata or scientific manifest difference")
    return {
        "schema": "onnx-splitpoint/generic-split-manifest-staging-provenance/v1",
        "original_manifest": old_source, "staged_manifest": new_source,
        "original_boundary_metadata_artifact": deepcopy(boundary_metadata),
        "measured_identity": deepcopy(identity), "full_model_sha256": full_model_sha256,
        "unchanged_artifact_bindings": bindings, "boundary_contract": deepcopy(boundary),
        "prepared_input": {k: prepared[k] for k in ("sha256", "shape", "dtype", "layout", "input_name")},
        "source_image_sha256": feed["source_image_sha256"], "allowed_differences": changes,
        "scientific_manifest_fields_unchanged": True,
    }


def validate_generic_split_manifest_staging(
    original_bytes, staged_bytes, *, original_source, staged_source, row,
    original_artifacts, full_model_sha256, producer_model, boundary_inputs,
):
    """Validate the byte-bound relation; malformed or unreviewed inputs deny."""
    try:
        return _validate_generic_split_manifest_staging(
            original_bytes, staged_bytes, original_source=original_source,
            staged_source=staged_source, row=row, original_artifacts=original_artifacts,
            full_model_sha256=full_model_sha256, producer_model=producer_model,
            boundary_inputs=boundary_inputs,
        )
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Generic split-manifest staging: incomplete or malformed binding") from error
