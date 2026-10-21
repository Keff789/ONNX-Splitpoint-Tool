"""Explicit full-model/P2 source roles for Generic composed Quality only.

The canonical completion validator keeps its existing alias guards. A composed
producer's terminal ONNX is a Part2 model, so this opt-in projection preserves
the complete raw authority and binds that distinct role before exposing only
the full-model semantic identity to the canonical completion builder.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

SCHEMA = "onnx-splitpoint/generic-composed-quality-endpoint-role-projection"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _file(path):
    source = Path(path)
    if not source.is_file():
        raise ValueError("Generic Quality endpoint role source file missing")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": source.stat().st_size}


def _identity(value):
    if (not isinstance(value, Mapping) or not isinstance(value.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] <= 0):
        raise ValueError("Generic Quality endpoint role artifact identity invalid")
    return {key: value[key] for key in ("sha256", "size_bytes")}


def _roles(authority, full, part2):
    if (not isinstance(authority, Mapping)
            or authority.get("full_model_sha256") != full["sha256"]
            or authority.get("terminal_model_sha256") != part2["sha256"]):
        raise ValueError("Generic Quality full/terminal source roles differ from actual ONNX bytes")
    if authority.get("model_sha256") not in (None, "", full["sha256"]):
        raise ValueError("Generic Quality semantic model identity conflicts")
    declared = (authority.get("output_endpoint_attestation") or {}).get("declared_contract") or {}
    if any(declared.get(key) not in (None, "", full["sha256"])
           for key in ("model_sha256", "full_model_sha256", "terminal_model_sha256")):
        raise ValueError("Generic Quality nested semantic source role conflicts")
    projected = deepcopy(dict(authority))
    projected.pop("terminal_model_sha256")
    return projected


def project_generic_quality_endpoint(authority, *, full_model, part2_source):
    """Create metadata from independently hashed existing full and P2 ONNX."""
    identities = []
    for value in (full_model, part2_source):
        identity = _identity(value)
        if _file(value["path"]) != identity:
            raise ValueError("Generic Quality endpoint source bytes changed")
        identities.append(identity)
    full, part2 = identities
    projected = _roles(authority, full, part2)
    proof = {"schema": SCHEMA, "schema_version": 1,
        "authority_sha256": _hash(authority), "full_model": full, "part2_source": part2,
        "semantic_source_endpoint_sha256": _hash(projected),
        "field_roles": {"full_model_sha256": "full_model_semantic_identity",
            "terminal_model_sha256": "independently_bound_part2_source_onnx"},
        "projected_fields_removed": ["terminal_model_sha256"]}
    proof["projection_sha256"] = _hash(proof)
    return {"source_endpoint_authority": deepcopy(dict(authority)),
        "source_endpoint_projection": proof, "source_endpoint_contract": projected}


def verify_generic_quality_endpoint_projection(config, *, check_files=False, full_model_path=None):
    """Verify the exact opt-in projection without relaxing paired validation.

    Preparatory loading verifies the independently bound P2 file when requested.
    At actual producer creation, ``full_model_path`` additionally binds the full
    ONNX that the ordinary composed runner really uses. Controller paths from
    preparation are never assumed to exist on the accelerator host.
    """
    source = config.get("source_endpoint_contract") or {}
    authority, proof = config.get("source_endpoint_authority"), config.get("source_endpoint_projection")
    if authority is None and proof is None:
        if any(source.get(key) for key in ("full_model_sha256", "terminal_model_sha256", "model_sha256")):
            raise ValueError("Generic Quality explicit full/terminal source-role authority missing")
        return None  # Existing purely synthetic metadata fixtures have no ONNX identity.
    if (not isinstance(authority, Mapping) or not isinstance(proof, Mapping)
            or proof.get("schema") != SCHEMA or type(proof.get("schema_version")) is not int
            or proof["schema_version"] != 1):
        raise ValueError("Generic Quality endpoint role projection schema invalid")
    unsealed = dict(proof)
    seal = unsealed.pop("projection_sha256", None)
    full, part2 = _identity(proof.get("full_model")), _identity(proof.get("part2_source"))
    expected_part2 = ((config.get("expected_runtime_artifacts") or {}).get("source_part2_onnx") or {})
    projected = _roles(authority, full, part2)
    if (_hash(unsealed) != seal or _identity(expected_part2) != part2
            or proof.get("authority_sha256") != _hash(authority)
            or source != projected or proof.get("semantic_source_endpoint_sha256") != _hash(source)
            or proof.get("field_roles") != {"full_model_sha256": "full_model_semantic_identity",
                "terminal_model_sha256": "independently_bound_part2_source_onnx"}
            or proof.get("projected_fields_removed") != ["terminal_model_sha256"]):
        raise ValueError("Generic Quality endpoint source role projection differs")
    if check_files and _file(expected_part2["path"]) != part2:
        raise ValueError("Generic Quality actual Part2 source differs from role projection")
    if full_model_path is not None and _file(full_model_path) != full:
        raise ValueError("Generic Quality actual full model differs from role projection")
    return deepcopy(dict(proof))
