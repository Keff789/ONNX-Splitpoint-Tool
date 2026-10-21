"""Materialize source-bound Generic Quality inputs in a normal generated suite."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(root, relative):
    token = PurePosixPath(str(relative))
    if token.is_absolute() or ".." in token.parts or not token.parts:
        raise ValueError("generic_quality_sidecar_path_not_contained")
    candidate = root.joinpath(*token.parts)
    current = root
    for part in token.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("generic_quality_sidecar_symlink_forbidden")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("generic_quality_sidecar_path_not_contained")
    return candidate


def _check_sources(config, root):
    sources = config.get("runtime_source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("generic_quality_sidecar_source_closure_missing")
    seen = set()
    for source in sources:
        if not isinstance(source, Mapping) or not _digest(source.get("sha256")):
            raise ValueError("generic_quality_sidecar_source_binding_invalid")
        path = _contained(root, source.get("path", ""))
        if path in seen:
            raise ValueError("generic_quality_sidecar_source_closure_duplicated")
        seen.add(path)
        if not path.is_file() or _sha(path) != source["sha256"]:
            raise ValueError("generic_quality_sidecar_runtime_source_changed")
        if "size_bytes" in source and path.stat().st_size != source["size_bytes"]:
            raise ValueError("generic_quality_sidecar_runtime_source_size_changed")


def _publish_new(target, payload):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".quality-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _materialize_assets(config, root, case_id):
    """Copy the exact portable artifact descriptors requested by this case."""
    pending = {}
    prefix = f"quality_runtime/assets/{case_id}/"
    artifacts = config.get("expected_runtime_artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("generic_quality_sidecar_assets_invalid")
    for role, descriptor in artifacts.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError("generic_quality_sidecar_asset_descriptor_invalid")
        relative = descriptor.get("path", "")
        if not isinstance(relative, str) or not relative.startswith("quality_runtime/assets/"):
            continue
        if (not relative.startswith(prefix) or Path(relative).suffix not in {".json", ".hef", ".dxnn"}
                or not _digest(descriptor.get("sha256"))
                or type(descriptor.get("size_bytes")) is not int
                or descriptor["size_bytes"] < 1):
            raise ValueError("generic_quality_sidecar_asset_binding_invalid")
        target = _contained(root, relative)
        local = descriptor.get("local_path")
        if local:
            if not isinstance(local, str) or not Path(local).is_absolute():
                raise ValueError("generic_quality_sidecar_asset_source_invalid")
            origin = Path(local)
            if origin.is_symlink() or not origin.is_file():
                raise ValueError("generic_quality_sidecar_asset_source_invalid")
        else:
            origin = target
        if (not origin.is_file() or origin.stat().st_size != descriptor["size_bytes"]
                or _sha(origin) != descriptor["sha256"]):
            raise ValueError("generic_quality_sidecar_asset_source_changed")
        if target.exists() and (not target.is_file() or _sha(target) != descriptor["sha256"]):
            raise ValueError("generic_quality_sidecar_asset_existing_bytes_differ")
        binding = (descriptor["sha256"], descriptor["size_bytes"])
        if target in pending and pending[target][0] != binding:
            raise ValueError("generic_quality_sidecar_asset_conflicting_bindings")
        pending[target] = (binding, origin, role)
    records = []
    for target, (binding, origin, role) in pending.items():
        expected, size = binding
        if target.exists():
            disposition = "reused_exact_bytes"
        else:
            payload = origin.read_bytes()
            if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("generic_quality_sidecar_asset_source_changed")
            try:
                _publish_new(target, payload)
            except FileExistsError:
                if not target.is_file() or _sha(target) != expected:
                    raise ValueError("generic_quality_sidecar_asset_concurrent_bytes_differ")
            disposition = "copied_exact_bytes"
        records.append({"role": role, "path": target.relative_to(root).as_posix(),
                        "sha256": expected, "size_bytes": size, "status": disposition})
    return records


def materialize_generic_quality_sidecars(*, suite_dir, model_id, benchmark_plan):
    """Copy only the exact requested JSON bytes; preserve existing unequal files.

    Endpoint/artifact admission remains in the normal Generic Quality loader.
    This controller boundary checks the requested sidecar and its shipped source
    closure before the normal suite bundler can transfer either one.
    """
    root = Path(suite_dir).resolve()
    records = []
    for run in benchmark_plan.get("runs", benchmark_plan.get("planned_runs", [])):
        if not isinstance(run, dict) or "generic_quality_completion_configs" not in run:
            continue
        paths = run["generic_quality_completion_configs"]
        hashes = run.get("generic_quality_completion_config_sha256")
        sources = run.get("generic_quality_completion_sources", {})
        cases = run.get("case_ids")
        if (run.get("quality_source_only") is not True or not isinstance(paths, Mapping)
                or not paths or not isinstance(cases, list) or len(cases) != len(set(cases))
                or set(cases) != set(paths) or not isinstance(hashes, Mapping)
                or set(hashes) != set(paths) or not isinstance(sources, Mapping)
                or (sources and set(sources) != set(paths))):
            raise ValueError("generic_quality_sidecar_case_scope_invalid")
        for case_id, relative in paths.items():
            if (not isinstance(case_id, str) or re.fullmatch(r"b[0-9]+", case_id) is None
                    or relative != f"quality_runtime/{case_id}.json"
                    or not _digest(hashes[case_id])):
                raise ValueError("generic_quality_sidecar_binding_invalid")
            target = _contained(root, relative)
            expected = hashes[case_id]
            source = sources.get(case_id)
            if source:
                if (not isinstance(source, Mapping) or source.get("sha256") != expected
                        or not isinstance(source.get("path"), str)
                        or not Path(source["path"]).is_absolute()):
                    raise ValueError("generic_quality_sidecar_source_descriptor_invalid")
                origin = Path(source["path"])
                if origin.is_symlink() or not origin.is_file():
                    raise ValueError("generic_quality_sidecar_source_file_invalid")
                payload = origin.read_bytes()
                if (hashlib.sha256(payload).hexdigest() != expected
                        or ("size_bytes" in source and len(payload) != source["size_bytes"])):
                    raise ValueError("generic_quality_sidecar_source_bytes_changed")
            elif target.is_file():
                payload = target.read_bytes()
            else:
                raise ValueError("generic_quality_sidecar_source_missing")
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("generic_quality_sidecar_bytes_changed")
            config = json.loads(payload)
            if not isinstance(config, Mapping):
                raise ValueError("generic_quality_sidecar_config_invalid")
            identity = config.get("identity", {})
            if (config.get("schema") != "onnx-splitpoint/generic-composed-quality-completion-config"
                    or type(config.get("schema_version")) is not int
                    or config.get("schema_version") != 1
                    or config.get("path_root") != "suite_root"
                    or not isinstance(identity, Mapping)
                    or identity.get("model_id") != model_id
                    or identity.get("case_id") != case_id
                    or (run.get("hardware_setup_id") and
                        identity.get("setup_id") != run["hardware_setup_id"])):
                raise ValueError("generic_quality_sidecar_identity_mismatch")
            _check_sources(config, root)
            assets = _materialize_assets(config, root, case_id)
            if target.exists():
                if not target.is_file() or _sha(target) != expected:
                    raise ValueError("generic_quality_sidecar_existing_bytes_differ")
                disposition = "reused_exact_bytes"
            else:
                try:
                    _publish_new(target, payload)
                except FileExistsError:
                    if not target.is_file() or _sha(target) != expected:
                        raise ValueError("generic_quality_sidecar_concurrent_bytes_differ")
                disposition = "copied_exact_bytes"
            records.append({"run_id": run.get("id"), "case_id": case_id,
                            "path": relative, "sha256": expected,
                            "size_bytes": len(payload), "status": disposition,
                            "assets": assets})
    return records


def bind_existing_generic_cpu_reference(*, suite_dir, run_root, model_id,
                                        benchmark_plan, quality_gate_policy,
                                        require_projected=False):
    """Reuse a hash-bound old CPU artifact through normal immutable publication.

    This explicit Generic source-only route never schedules CPU inference.
    The ordinary active-run reference reader and central request loader still
    verify the published bytes and each new candidate's semantic pairing.
    """
    from ..accuracy_gates import AccuracyGatePolicy
    from ..management_reference import (
        _publish_immutable_reference, _sha256_regular_file,
        _successful_reference_status,
    )
    from ..quality_service import _load_quality_artifact
    from ..preprocessing_contract import canonical_image_preprocessing_contract

    policy = dict(quality_gate_policy or {})
    management = policy.get("management_reference", {})
    explicit_present = isinstance(management, Mapping) and "existing_reference" in management
    declared = management.get("existing_reference") if isinstance(management, Mapping) else None
    runs = benchmark_plan.get("runs", benchmark_plan.get("planned_runs", []))
    selected = [r for r in runs if isinstance(r, Mapping)
                and "generic_quality_completion_configs" in r]
    if not explicit_present and not selected:
        return None
    if not isinstance(declared, Mapping) or not declared or not selected:
        raise ValueError("generic_existing_cpu_reference_explicit_binding_required")
    if str((policy.get("statistics") or {}).get("execution_location")) != "central_management":
        raise ValueError("generic_existing_cpu_reference_location_invalid")
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("generic_existing_cpu_reference_run_invalid")
        if run.get("semantic_reference_only") is True:
            continue
        if (run not in selected or run.get("quality_source_only") is not True
                or not isinstance(run.get("generic_quality_completion_configs"), Mapping)
                or not run["generic_quality_completion_configs"]
                or run.get("variants") != ["composed"]):
            raise ValueError("generic_existing_cpu_reference_requires_source_only_scope")
    descriptor = dict(declared)
    source_path = Path(str(descriptor.get("reference_path") or ""))
    expected_sha = descriptor.get("reference_sha256")
    expected_size = descriptor.get("reference_size_bytes")
    if (not source_path.is_absolute() or not _digest(expected_sha)
            or type(expected_size) is not int or expected_size < 1
            or descriptor.get("reference_role") != "canonical_cpu_ort"
            or descriptor.get("semantic_reference_only") is not True
            or not descriptor.get("source_evaluation_run_id")):
        raise ValueError("generic_existing_cpu_reference_descriptor_invalid")
    source_status = descriptor.get("source_status")
    policy_source = descriptor.get("policy_source")
    if not isinstance(source_status, Mapping) or not isinstance(policy_source, Mapping):
        raise ValueError("generic_existing_cpu_reference_source_binding_missing")
    status, _, _ = _load_quality_artifact(source_status, source_path, verify=True)
    original_policy, _, _ = _load_quality_artifact(policy_source, source_path, verify=True)
    if AccuracyGatePolicy.from_mapping(original_policy).sha256() != AccuracyGatePolicy.from_mapping(policy).sha256():
        raise ValueError("generic_existing_cpu_reference_policy_changed")
    token = lambda value: str(value or "").removeprefix("sha256:")
    source_contract = token(status.get("source_contract_sha256"))
    if (not _digest(source_contract)
            or status.get("schema") != "onnx-splitpoint/management-cpu-reference-job"
            or type(status.get("schema_version")) is not int or status["schema_version"] != 1
            or status.get("status") not in {"completed", "cache_hit"}
            or status.get("model_id") != model_id
            or token(status.get("reference_sha256")) != expected_sha
            or status.get("reference_size_bytes") != expected_size
            or status.get("record_count") != 5000
            or status.get("provider") != "onnxruntime_cpu"
            or status.get("semantic_reference_only") is not True
            or status.get("execution_location") != "central_management"
            or any(status.get(k) is not False for k in
                   ("include_in_latency_fps_energy", "include_in_ranking", "include_in_pareto"))):
        raise ValueError("generic_existing_cpu_reference_source_status_invalid")
    if (source_path.stat().st_size != expected_size
            or token(_sha256_regular_file(source_path, label="explicit existing CPU reference")) != expected_sha):
        raise ValueError("generic_existing_cpu_reference_bytes_changed")
    reference, _, _ = _load_quality_artifact(
        {"path": str(source_path), "sha256": expected_sha, "size_bytes": expected_size},
        source_path, verify=True,
    )
    contract = reference.get("quality_contract")
    if (contract != descriptor.get("quality_contract") or contract != status.get("quality_contract")
            or reference.get("quality_contract_sha256") != descriptor.get("quality_contract_sha256")):
        raise ValueError("generic_existing_cpu_reference_contract_changed")
    manifest_path = Path(run_root) / "models" / model_id / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_hash = token(manifest.get("sha256") or manifest.get("model_sha256"))
    if model_hash != token((contract.get("model") or {}).get("sha256")):
        raise ValueError("generic_existing_cpu_reference_model_changed")
    records = reference.get("records", [])
    ids = sorted(row.get("image_id") for row in records)
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise ValueError("generic_existing_cpu_reference_image_ids_invalid")
    root = Path(suite_dir).resolve()
    dataset_sha = (contract.get("dataset") or {}).get("manifest_sha256")
    for run in selected:
        if (str(run.get("image_scale") or "norm") != "norm"
                or run.get("letterbox", True) is not True
                or run.get("preprocess_mode", "letterbox") != "letterbox"):
            raise ValueError("generic_existing_cpu_reference_preprocessing_changed")
        validation = Path(str(run.get("validation_images") or ""))
        validation = validation if validation.is_absolute() else _contained(root, validation.as_posix())
        validation_manifest = validation / "manifest.json" if validation.is_dir() else validation
        if not validation_manifest.is_file() or _sha(validation_manifest) != dataset_sha:
            raise ValueError("generic_existing_cpu_reference_dataset_changed")
        for case_id, relative in run["generic_quality_completion_configs"].items():
            config_path = _contained(root, relative)
            if _sha(config_path) != (run.get("generic_quality_completion_config_sha256") or {}).get(case_id):
                raise ValueError("generic_existing_cpu_reference_config_changed")
            config = json.loads(config_path.read_text())
            if sorted(config.get("expected_image_ids", [])) != ids:
                raise ValueError("generic_existing_cpu_reference_candidate_image_ids_changed")
            hw = (config.get("completion_contract") or {}).get("processor_contract", {}).get("input_hw", [])
            prep = (contract.get("preprocessing") or {}).get("identity")
            if (config.get("preprocess") != {"mode": "letterbox", "rgb": True, "pad_value": 114}
                    or prep != canonical_image_preprocessing_contract("detection", hw)):
                raise ValueError("generic_existing_cpu_reference_preprocessing_changed")
    target = Path(run_root) / "quality_management" / "references" / model_id / "by_source_contract" / source_contract / "canonical_cpu_reference.json"
    projected = {**descriptor, "reference_path": str(target)}
    if require_projected and benchmark_plan.get("management_cpu_reference") != projected:
        raise ValueError("generic_existing_cpu_reference_plan_not_projected")
    metadata, published_sha, _ = _publish_immutable_reference(source_path, target)
    if token(published_sha) != expected_sha:
        raise ValueError("generic_existing_cpu_reference_publication_changed")
    reuse_status = _successful_reference_status(
        status="cache_hit", model_id=model_id, workers=int(status.get("cpu_threads") or 1),
        source_contract_sha256="sha256:" + source_contract,
        reference_path=target, reference_sha256=published_sha, reference_metadata=metadata,
        extra={"explicit_existing_reference_reuse": True,
               "source_evaluation_run_id": descriptor["source_evaluation_run_id"],
               "source_reference_path": str(source_path), "source_status": dict(source_status)},
    )
    benchmark_plan["management_cpu_reference"] = projected
    return reuse_status
