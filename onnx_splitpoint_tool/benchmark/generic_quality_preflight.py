"""Project explicit Generic source-only artifacts before normal suite staging."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import re


def _contained(root, token):
    relative = Path(str(token or ""))
    target = root / relative
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or not target.resolve().is_relative_to(root)):
        raise ValueError("generic_quality_preflight_path_not_contained")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("generic_quality_preflight_symlink_forbidden")
    return target


def project_generic_quality_requirement(*, suite_dir, model_id, setup_id,
                                        requirement, runtime_contract, hash_file):
    """Return an exact existing-engine recipe, or None for the unchanged route.

    The source sidecar can precede its normal suite materialization. Only local
    source ONNX and runtime-source bytes are checked here; persistent engine,
    receipt and build-ONNX bytes are checked by the normal remote verifier.
    The child still performs the complete normal Quality admission after upload.
    """
    suite = Path(suite_dir).resolve()
    plan = json.loads((suite / "benchmark_plan.json").read_text(encoding="utf-8"))
    ids = set(requirement.get("run_ids") or [])
    runs = [row for row in plan.get("runs", []) if isinstance(row, Mapping)
            and str(row.get("id") or row.get("run_id") or row.get("name") or "") in ids]
    selected = [row for row in runs if "generic_quality_completion_configs" in row]
    if not selected:
        return None
    if len(selected) != 1 or len(runs) != 1:
        raise ValueError("generic_quality_preflight_mixed_run_recipes")
    run = selected[0]
    paths = run["generic_quality_completion_configs"]
    hashes = run.get("generic_quality_completion_config_sha256")
    cases = run.get("case_ids")
    sources = run.get("generic_quality_completion_sources", {})
    if (run.get("quality_source_only") is not True
            or run.get("variants") != ["composed"]
            or run.get("native_trt_build") is not False
            or any(key in run for key in ("native_split_quality_binding",
                                         "native_split_quality_binding_sha256"))
            or not isinstance(cases, list) or not cases
            or len(cases) != len(set(cases))
            or not isinstance(paths, Mapping) or set(paths) != set(cases)
            or not isinstance(hashes, Mapping) or set(hashes) != set(cases)
            or not isinstance(sources, Mapping) or (sources and set(sources) != set(cases))):
        raise ValueError("generic_quality_preflight_explicit_run_invalid")
    case = requirement["case_id"]
    if case not in paths or paths[case] != f"quality_runtime/{case}.json":
        raise ValueError("generic_quality_preflight_case_scope_invalid")
    expected = hashes[case]
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("generic_quality_preflight_config_digest_invalid")
    target = _contained(suite, paths[case])
    source = sources.get(case)
    if source is not None:
        if (not isinstance(source, Mapping) or source.get("sha256") != expected
                or not Path(str(source.get("path") or "")).is_absolute()):
            raise ValueError("generic_quality_preflight_config_source_invalid")
        origin = Path(source["path"])
    else:
        origin = target
    if origin.is_symlink() or not origin.is_file():
        raise ValueError("generic_quality_preflight_config_missing")
    payload = origin.read_bytes()
    if (hashlib.sha256(payload).hexdigest() != expected
            or source is not None and "size_bytes" in source and len(payload) != source["size_bytes"]):
        raise ValueError("generic_quality_preflight_config_changed")
    if target.exists() and (not target.is_file() or hash_file(target) != expected):
        raise ValueError("generic_quality_preflight_staged_config_changed")
    from ..runners.generic_quality_completion import load_quality_completion_config
    config = load_quality_completion_config(origin, check_files=False)
    if hash_file(origin) != expected:
        raise ValueError("generic_quality_preflight_config_changed")
    raw = json.loads(payload)
    identity = config["identity"]
    backend = {"hailo8_to_trt": "hailo8", "hailo10h_to_trt": "hailo10h",
               "deepx_to_trt": "deepx_m1"}.get(requirement.get("expected_backend"))
    if (raw.get("path_root") != "suite_root" or not backend
            or any(identity.get(key) != value for key, value in {
                "model_id": model_id, "setup_id": setup_id, "case_id": case,
                "comparison_backend": backend}.items())
            or run.get("hardware_setup_id", setup_id) != setup_id):
        raise ValueError("generic_quality_preflight_identity_mismatch")
    if (runtime_contract.get("native_trt_precision") != "fp16"
            or runtime_contract.get("trt_runtime") != "native"):
        raise ValueError("generic_quality_preflight_runtime_recipe_mismatch")
    artifacts = raw["expected_runtime_artifacts"]
    for role in ("source_part1_onnx", "source_part2_onnx"):
        descriptor = artifacts[role]
        local = _contained(suite, descriptor["path"])
        if (not local.is_file() or local.stat().st_size != descriptor["size_bytes"]
                or hash_file(local) != descriptor["sha256"]):
            raise ValueError("generic_quality_preflight_source_changed:" + role)
        if role == "source_part2_onnx" and (
                local.resolve() != Path(requirement["source_path"]).resolve()
                or descriptor["sha256"] != requirement["source_sha256"]
                or descriptor["size_bytes"] != requirement["source_size_bytes"]):
            raise ValueError("generic_quality_preflight_part2_source_mismatch")
    for descriptor in raw["runtime_source_files"]:
        local = _contained(suite, descriptor["path"])
        if (not local.is_file() or hash_file(local) != descriptor["sha256"]
                or "size_bytes" in descriptor and local.stat().st_size != descriptor["size_bytes"]):
            raise ValueError("generic_quality_preflight_runtime_source_changed")
    persistent = {role: dict(artifacts[role]) for role in
                  ("engine", "engine_build_receipt", "build_part2_onnx")}
    if any(not Path(row["path"]).is_absolute()
           or type(row.get("size_bytes")) is not int or row["size_bytes"] < 1
           for row in persistent.values()):
        raise ValueError("generic_quality_preflight_persistent_path_invalid")
    build = persistent["build_part2_onnx"]
    result = {"recipe_source": "generic_quality_completion_config",
              "engine_precision": "fp16", "runtime_precision_identity": identity["precision"],
              "generic_quality_config_sha256": expected,
              "generic_quality_artifacts": persistent}
    if "existing_part2_bridge" not in raw:
        if identity["precision"] != "float32_layout_fp16":
            raise ValueError("generic_quality_preflight_identity_mismatch")
        if (build["sha256"] != requirement["source_sha256"]
                or build["size_bytes"] != requirement["source_size_bytes"]):
            raise ValueError("generic_quality_preflight_direct_build_source_mismatch")
        return result
    bridge = raw["existing_part2_bridge"]
    original = raw.get("original_run_cfg")
    if (not isinstance(bridge, Mapping) or set(bridge) != {
            "schema", "schema_version", "binding_sha256", "policy_sha256", "precision", "binding_source"}
            or bridge["schema"] != "onnx-splitpoint/existing-part2-bridge-reference"
            or type(bridge["schema_version"]) is not int or bridge["schema_version"] != 1
            or bridge["precision"] not in {"uint8_dequant_fp16", "float32_layout_fp16"}
            or bridge["precision"] != identity["precision"]
            or any(not isinstance(bridge[key], str)
                   or re.fullmatch(r"[0-9a-f]{64}", bridge[key]) is None
                   for key in ("binding_sha256", "policy_sha256"))
            or not isinstance(original, Mapping)
            or original.get("native_trt_precision") != bridge["precision"]
            or original.get("native_trt_workspace_mb") != runtime_contract.get("native_trt_workspace_mb")
            or original.get("boundary_mode") != (
                "raw_uint8_hailo" if bridge["precision"] == "uint8_dequant_fp16"
                else "canonical_float32")):
        raise ValueError("generic_quality_preflight_existing_bridge_invalid")
    container = bridge["binding_source"]
    if (not isinstance(container, Mapping)
            or container.get("path") != f"quality_runtime/assets/{case}/existing_part2_bridge/composed_request.json"
            or container.get("pointer") != "/native_split_quality_binding"
            or not isinstance(container.get("local_path"), str)
            or not Path(container["local_path"]).is_absolute()
            or not isinstance(container.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", container["sha256"]) is None
            or type(container.get("size_bytes")) is not int or container["size_bytes"] < 1):
        raise ValueError("generic_quality_preflight_bridge_source_invalid")
    container_path = Path(container["local_path"])
    if (container_path.is_symlink() or not container_path.is_file()
            or container_path.stat().st_size != container["size_bytes"]
            or hash_file(container_path) != container["sha256"]):
        raise ValueError("generic_quality_preflight_bridge_source_changed")
    source_request = json.loads(container_path.read_text(encoding="utf-8"))
    compiled_binding = source_request.get("native_split_quality_binding")
    if (not isinstance(compiled_binding, Mapping)
            or compiled_binding.get("binding_sha256") != bridge["binding_sha256"]
            or any(source_request.get(key) != identity[key] for key in ("model_id", "case_id", "setup_id"))
            or source_request.get("runtime_precision_identity") != identity["precision"]):
        raise ValueError("generic_quality_preflight_original_generic_bridge_mismatch")
    overlap = ("part1_runtime", "boundary_metadata", "source_part2_onnx",
               "build_part2_onnx", "engine", "engine_build_receipt")
    identities = {}
    for role in overlap:
        row = artifacts[role]
        if type(row.get("size_bytes")) is not int or row["size_bytes"] < 1:
            raise ValueError("generic_quality_preflight_existing_bridge_artifact_invalid")
        identities[role] = {key: row[key] for key in ("sha256", "size_bytes")}
    result.update(
        engine_precision=bridge["precision"],
        generic_quality_bridge_binding_sha256=bridge["binding_sha256"],
        generic_quality_bridge_policy_sha256=bridge["policy_sha256"],
        generic_quality_bridge_artifact_identities=identities,
        generic_quality_bridge_binding=dict(compiled_binding),
        generic_quality_bridge_container={key: container[key] for key in ("path", "sha256", "size_bytes", "pointer")},
        part1_artifact_sha256=identities["part1_runtime"]["sha256"],
        part1_artifact_size_bytes=identities["part1_runtime"]["size_bytes"],
    )
    return result
