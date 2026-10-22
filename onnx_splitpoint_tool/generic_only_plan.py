"""Admission of real Generic producers without any Native performance claim.

The paired historical supplement is deliberately a separate mode.  This module
consumes the normal producer's files and Central Quality artifacts; it neither
builds a model nor fabricates a Native command or a quality decision.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Mapping

from .generic_completion_plan import completion_identity


MODE = "generic_only_completed_task"
ARTIFACT_ROLES = (
    "part1_runtime", "engine", "source_part1_onnx", "source_part2_onnx",
    "build_part2_onnx", "engine_build_receipt", "boundary_metadata", "part1_build_receipt",
)
FORBIDDEN_NATIVE = (
    "native_source", "native_uid", "native_command_contract", "historical_pair",
    "native_command_contract_sha256", "native_quality_binding_sha256",
)

_VALIDATION_SCOPE = ContextVar("generic_only_validation_scope", default=None)


def _file_version(path):
    def fields(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    return fields(path.lstat()), fields(path.stat())


class _ValidationScope:
    """Operation-local digests and small proofs, never prediction/report data."""
    def __init__(self):
        self.sources = {}
        self.proofs = {}

    def unchanged(self):
        for path, (version, _) in self.sources.items():
            if _file_version(path) != version:
                raise ValueError("Generic-only source changed during validation: " + str(path))

    def digest(self, path):
        path = Path(os.path.abspath(path))
        version = _file_version(path)
        previous = self.sources.get(path)
        if previous is not None:
            if previous[0] != version:
                raise ValueError("Generic-only source changed during validation: " + str(path))
            return previous[1]
        # A full 420-row operation fits comfortably; never grow with report size.
        if len(self.sources) >= 16384:
            raise ValueError("Generic-only validation source bound exceeded")
        digest = _file_sha256(path)
        if _file_version(path) != version:
            raise ValueError("Generic-only source changed while being read: " + str(path))
        self.sources[path] = (version, digest)
        return digest

    def finish(self):
        self.unchanged()
        for path, (version, digest) in self.sources.items():
            if _file_sha256(path) != digest or _file_version(path) != version:
                raise ValueError("Generic-only source changed during validation: " + str(path))


@contextmanager
def validation_scope():
    """Reuse validation only within one operation, fully rechecking at its end.

    Preparation callers can nest this around their existing normal operations.
    No cache survives the outermost operation and no large JSON is retained here.
    """
    existing = _VALIDATION_SCOPE.get()
    if existing is not None:
        yield existing
        return
    scope = _ValidationScope()
    token = _VALIDATION_SCOPE.set(scope)
    try:
        yield scope
        scope.finish()
    finally:
        _VALIDATION_SCOPE.reset(token)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path):
    scope = _VALIDATION_SCOPE.get()
    return scope.digest(path) if scope is not None else _file_sha256(path)


def read_bound(descriptor):
    """Read an existing immutable source, never create a missing authority."""
    if not isinstance(descriptor, Mapping):
        raise ValueError("Generic-only source descriptor missing")
    path = Path(str(descriptor.get("local_path") or descriptor.get("path") or ""))
    if (not path.is_absolute() or ".." in path.parts or not path.is_file()
            or file_sha256(path) != descriptor.get("sha256")):
        raise ValueError("Generic-only source missing or changed: " + str(path))
    return path


@lru_cache(maxsize=4)
def _json_at_version(path, version, digest):
    # In-memory parsing reuse only: large ordinary Central Quality summaries
    # contain many cases. Any file version change causes a fresh hash/read.
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if _file_version(source) != version:
        raise ValueError("Generic-only JSON source changed while being read")
    return value


def _json(descriptor):
    if not isinstance(descriptor, Mapping):
        raise ValueError("Generic-only JSON source missing")
    path = Path(str(descriptor.get("local_path") or descriptor.get("path") or ""))
    if not path.is_absolute() or ".." in path.parts or not path.is_file():
        raise ValueError("Generic-only JSON source missing or escaped")
    # Always register/check the source, including an existing parsed-JSON hit.
    version = _file_version(path)
    if file_sha256(path) != descriptor.get("sha256") or _file_version(path) != version:
        raise ValueError("Generic-only JSON source changed: " + str(path))
    value = _json_at_version(str(path), version, descriptor.get("sha256"))
    if _file_version(path) != version:
        raise ValueError("Generic-only JSON source changed while being read")
    selector = descriptor.get("selector")
    if selector is not None:
        if not isinstance(selector, Mapping) or not selector:
            raise ValueError("Generic-only source record selector malformed")
        records = value.get("results", value.get("rows", [])) if isinstance(value, Mapping) else []
        matches = [item for item in records if isinstance(item, Mapping)
                   and all(item.get(key) == wanted for key, wanted in selector.items())]
        if len(matches) != 1:
            raise ValueError("Generic-only source record is absent or ambiguous")
        value = matches[0]
    return value


def _completed_quality_binding(*args, **kwargs):
    from .runners.generic_quality_completion import validate_completed_quality_binding
    return validate_completed_quality_binding(*args, **kwargs)


def validate_quality_input_binding(row, proof):
    """Join one observed Quality feed to the exact later timing input bytes."""
    import numpy as np
    prepared = (row.get("input_binding") or {}).get("prepared_input") or {}
    representative = proof.get("representative_input") or {}
    comparison = proof.get("representative_input_comparison") or {}
    quality_prepared = representative.get("prepared_input") or {}
    source_image_sha = (row.get("input_binding") or {}).get("source_image_sha256")
    original_wh = (row.get("completion_contract") or {}).get("processor_contract", {}).get("original_wh")
    if (not representative or not comparison or not source_image_sha or not original_wh
            or representative.get("source_image_sha256") != source_image_sha
            or representative.get("original_wh") != original_wh
            or any(quality_prepared.get(key) != prepared.get(key) for key in ("sha256", "shape", "dtype"))):
        raise ValueError("Generic-only Quality representative differs from the exact timing image/input")
    from .runners.task_completion import load_completion_bound_array
    array = load_completion_bound_array({**prepared, "path": str(read_bound(prepared))}, file_sha256=file_sha256)
    tensor = comparison.get("prepared_tensor") or {}
    if (list(array.shape) != prepared.get("shape") or str(array.dtype) != prepared.get("dtype")
            or tensor.get("shape") != list(array.shape) or tensor.get("dtype") != str(array.dtype)
            or tensor.get("tensor_sha256") != hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()):
        raise ValueError("Generic-only Quality representative tensor differs from actual timing input bytes")


def validate_quality_reference(row):
    """Reuse only a small successful proof inside a fully verified operation."""
    with validation_scope() as scope:
        if row.get("task") == "classification":
            return _validate_quality_reference(row)
        fields = ("task", "identity", "quality_reference", "completion_contract", "input_binding",
                  "generic_artifact_binding", "boundary_observation", "canonical_full_input_bindings")
        key = hashlib.sha256(json.dumps({key: row.get(key) for key in fields},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        scope.unchanged()
        if key in scope.proofs:
            return copy.deepcopy(scope.proofs[key])
        proof = _validate_quality_reference(row)
        scope.unchanged()
        if len(scope.proofs) < 16 and len(json.dumps(proof)) <= 65536:
            scope.proofs[key] = copy.deepcopy(proof)
        return proof


def _validate_quality_reference(row):
    """Use the ordinary portable quality loader; do not rerun its bootstrap."""
    if row.get("task") == "classification":
        from .generic_only_classification_quality import validate_classification_quality_reference
        return validate_classification_quality_reference(row)
    from .quality_service import quality_request_from_manifest
    from .quality_service import _validate_generic_composed_candidate_execution_contract
    reference = row.get("quality_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("Generic-only CPU Quality evidence is missing")
    request_path = read_bound(reference.get("request"))
    result = _json(reference.get("result"))
    cpu_path = read_bound(reference.get("cpu_reference"))
    policy_path = read_bound(reference.get("policy_source"))
    import yaml
    policy_document = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    policy = policy_document.get("quality_gate", policy_document)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    # These files are read by the normal validators below through their own
    # readers. Bind their versions too before allowing an operation-local hit.
    from .runners import generic_quality_completion as completion_validator
    candidate = request.get("candidate") or {}
    relative = Path(str(candidate.get("path") or ""))
    candidate_path = (request_path.parent / relative).resolve()
    if (not relative.parts or relative.is_absolute()
            or not candidate_path.is_relative_to(request_path.parent.resolve())):
        raise ValueError("Generic-only candidate path escapes its request")
    read_bound({**candidate, "path": str(candidate_path), "local_path": str(candidate_path)})
    modules = [completion_validator, completion_validator._product(), completion_validator._three_stage(),
               completion_validator._product()._yolo_module, completion_validator._endpoint]
    if (request.get("generic_completed_quality") or {}).get("existing_part2_bridge"):
        modules.append(completion_validator._bridge_validator())
    for module in modules:
        file_sha256(module.__file__)
    for binding in row.get("canonical_full_input_bindings") or []:
        read_bound(binding.get("source_image"))
    if (result.get("schema") != "onnx-splitpoint/management-paired-quality-result"
            or result.get("status") != "completed" or result.get("technical_status") != "completed"
            or result.get("task") != "detection" or result.get("variant") != "composed"
            or result.get("cpu_reference_semantic_only") is not True
            or result.get("native_split_quality_binding_required") is True
            or result.get("native_split_quality_binding")
            or result.get("artifact_provenance_binding_status") != "verified"
            or result.get("execution_role") != "generic_composed_quality_candidate"):
        raise ValueError("Generic-only Quality must be a completed normal Generic CPU comparison")
    identity = row["identity"]
    if any(result.get(key) != identity[key] for key in ("model_id", "case_id", "setup_id")):
        raise ValueError("Generic-only Quality belongs to another case/setup")
    if (str(result.get("source_request_sha256") or "").removeprefix("sha256:")
            != reference["request"]["sha256"]):
        raise ValueError("Generic-only Quality request/result binding differs")
    cpu = result.get("management_cpu_reference") or {}
    # The normal management reference reader treats both as successful jobs.
    # Cached references retain the same bound artifact and full checks below.
    if (cpu.get("provider") != "onnxruntime_cpu" or cpu.get("status") not in {"completed", "cache_hit"}
            or cpu.get("reference_sha256") != reference["cpu_reference"]["sha256"]
            or cpu.get("semantic_reference_only") is not True):
        raise ValueError("Generic-only Quality lacks its actual canonical CPU reference")
    producer, producer_sha = _validate_generic_composed_candidate_execution_contract(
        result.get("producer_identity"), role="Generic-only source", task="detection")
    if producer_sha != result.get("producer_identity_sha256"):
        raise ValueError("Generic-only Quality producer hash differs")
    from .native_detection_postprocess import verify_detection_completion_execution_contract
    completion = verify_detection_completion_execution_contract(row["completion_contract"])
    if (completion["source_endpoint"]["endpoint_contract_hash"] != result.get("endpoint_contract_hash")
            or completion["source_endpoint"]["output_endpoint_id"] != identity["output_endpoint_id"]):
        raise ValueError("Generic-only completion differs from its Quality physical endpoint")
    artifacts = row["generic_artifact_binding"]["artifacts"]
    physical = {item["role"]: item for item in producer["runtime_artifacts"]}
    for quality_role, role in (("stage1_runtime", "part1_runtime"), ("stage2_runtime", "engine")):
        if physical.get(quality_role, {}).get("sha256") != artifacts[role]["sha256"]:
            raise ValueError("Generic-only Quality compiled artifact differs: " + role)
    if (producer.get("runtime_precision_identity") != identity["precision"]
            or producer["model"]["part1_onnx_sha256"] != artifacts["source_part1_onnx"]["sha256"]
            or producer["model"]["part2_onnx_sha256"] != artifacts["source_part2_onnx"]["sha256"]):
        raise ValueError("Generic-only Quality precision or split ONNX differs")
    for field in ("policy_sha256", "quality_contract_sha256", "endpoint_contract_hash",
                  "candidate_execution_completion_contract_sha256", "producer_identity_sha256"):
        if not result.get(field) or result[field] != request.get(field):
            raise ValueError("Generic-only Quality request/result differs: " + field)
    if (reference.get("policy_sha256") != result["policy_sha256"]
            or reference.get("dataset_manifest_sha256") != result.get("validation_dataset_manifest_sha256")):
        raise ValueError("Generic-only Quality policy/dataset authority differs")
    metric_policy = {**policy.get("detection", {}), "reporting_policy": policy.get("reporting_policy")}
    if (request.get("metric_gate_config") != metric_policy
            or any(request.get("statistics", {}).get(key) != policy.get("statistics", {}).get(key)
                   for key in ("method", "confidence_level", "bootstrap_repetitions", "seed", "decision"))):
        raise ValueError("Generic-only Quality thresholds/statistics differ from the bound producer profile")
    completion_proof = _completed_quality_binding(request, result, completion, artifacts,
        request_path=request_path, reference_descriptor=reference["cpu_reference"])
    validate_quality_input_binding(row, completion_proof)
    loaded = quality_request_from_manifest(request_path, verify_artifacts=True,
        reference_artifact={"path": str(cpu_path), "sha256": reference["cpu_reference"]["sha256"],
                            "size_bytes": cpu_path.stat().st_size})
    expected_images = ((policy_document.get("validation_execution") or {}).get("max_items") or {}).get("detection")
    if (type(expected_images) is not int or expected_images < 1
            or len(loaded.reference_records) != expected_images
            or len(loaded.reference_records) != result.get("evaluated_images")
            or result.get("n") != len(loaded.reference_records)
            or loaded.repetitions != result.get("primary", {}).get("bootstrap_repetitions_requested")
            or result.get("metric_gate_config") != request.get("metric_gate_config")):
        raise ValueError("Generic-only Quality image/statistics/threshold contract differs")
    status = (result.get("accuracy_assessment") or {}).get("accuracy_class")
    if status not in {"reference_close", "accuracy_loss", "inconclusive"}:
        raise ValueError("Generic-only Quality classification is missing or technical failure")
    from .runners.generic_passthrough import validate_quality_bindings
    validate_quality_bindings(row, request.get("generic_completed_quality") or {})
    return {"status": status, "quality_eligible": status == "reference_close",
            "policy_sha256": result["policy_sha256"],
            "dataset_manifest_sha256": reference["dataset_manifest_sha256"],
            "producer_identity_sha256": producer_sha,
            "completed_quality_binding": completion_proof,
            "request_sha256": reference["request"]["sha256"],
            "result_sha256": reference["result"]["sha256"]}


def environment(row):
    value = row.get("generic_runtime_environment") or {}
    source = read_bound(value.get("source"))
    observed = _json(value["source"])
    python = str(value.get("python") or "")
    if (not PurePosixPath(python).is_absolute()
            or len(str(value.get("python_sha256") or "")) != 64
            or not isinstance(value.get("environment"), dict)
            or observed.get("executable_sha256") != value.get("python_sha256")
            or observed.get("executable") != python):
        raise ValueError("Generic-only producer interpreter binding is missing")
    return {"python": python, "environment": copy.deepcopy(value["environment"]),
            "source": str(source), "original_line": str(value.get("original_line") or "")}


def _project_artifacts(row, case_dir):
    """Copy explicitly staged preserved bytes; retain producer identities."""
    binding = copy.deepcopy(row["generic_artifact_binding"])
    staged = []
    for role, descriptor in binding["artifacts"].items():
        if descriptor.get("stage_local") is not True:
            continue
        if role in {"engine", "build_part2_onnx"}:
            raise ValueError("Generic-only TensorRT engine/build source must retain its verified receipt path")
        local = read_bound(descriptor)
        remote = case_dir + "/artifacts/" + role + "/" + local.name
        staged.append({"local": str(local), "remote": remote, "sha256": descriptor["sha256"],
                       "role": "generic_" + role})
        descriptor["path"] = remote
    expected = {role: {key: binding["artifacts"][role][key] for key in ("path", "sha256")}
                for role in ("part1_runtime", "engine")}
    return binding, expected, staged


def validate_plan(plan, scope_csv, primary):
    if plan.get("scope_mode") != MODE or plan.get("admission_mode") != "generic_only":
        raise ValueError("Generic-only plan requires its explicit separated admission mode")
    count = plan.get("authorized_row_count")
    rows = plan.get("rows")
    if (type(count) is not int or not 1 <= count <= 420
            or type(plan.get("row_count")) is not int or plan["row_count"] != count
            or not isinstance(rows, list) or len(rows) != count):
        raise ValueError("Generic-only scope must explicitly authorize one through 420 rows")
    if any(plan.get(key) is not False for key in ("build_allowed", "fallback_allowed", "automatic_retry")):
        raise ValueError("Generic-only throughput preparation forbids builds, fallback and automatic retries")
    if plan.get("cell_execution_control") is not None:
        from .workflow.generic_only_cell_policy import require_bound_control
        require_bound_control(plan["cell_execution_control"], plan["cell_execution_control"])
    if file_sha256(scope_csv) != plan.get("scope_source_sha256"):
        raise ValueError("Generic-only scope CSV changed")
    with Path(scope_csv).open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    identities = [completion_identity(r.get("identity", r)) for r in rows]
    if (len(source) != count or len(set(identities)) != count
            or set(identities) != {completion_identity(r) for r in source}):
        raise ValueError("Generic-only scope identities differ or repeat")
    manifest = _json(plan.get("producer_run_manifest"))
    if read_bound(plan["producer_run_manifest"]).resolve() != Path(primary).resolve() / "run_manifest.json":
        raise ValueError("Generic-only producer run manifest belongs to another run")
    targets = {item["id"] for item in manifest.get("hardware_targets", [])}
    requests = {"hailo8": ("synchronous", 1), "hailo10h": ("hailo_async_slots_v1", 8),
                "deepx": ("deepx_async_slots_v1", 4)}
    for row in rows:
        identity = row["identity"]
        if (row.get("admission_mode") != "generic_only" or row.get("task") not in {"classification", "detection"}
                or any(row.get(key) not in (None, "", {}) for key in FORBIDDEN_NATIVE)
                or row.get("energy_capture") or row.get("acceptance_mode", "scientific") != "scientific"):
            raise ValueError("Generic-only row cannot import Native, energy or functional-only scope")
        if any(str(row.get(key) or "") != str(identity[key]) for key in identity):
            raise ValueError("Generic-only row identity projection differs")
        if identity["setup_id"] not in targets or identity["comparison_backend"] not in requests:
            raise ValueError("Generic-only requires a bound H8/H10/DeepX target")
        if Path(row["original_run"]).resolve() != Path(primary).resolve():
            raise ValueError("Generic-only row is outside its bound producer run")
        selection = _json(row.get("selection_source"))
        selected = selection.get("selected_candidates") or []
        if (selection.get("model_id") != identity["model_id"]
                or sum(item.get("case_id") == identity["case_id"] for item in selected) != 1):
            raise ValueError("Generic-only cut is absent or duplicated in the fixed candidate selection")
        local_case = Path(row["original_local_case_dir"]).resolve()
        split = read_bound(row.get("split_manifest"))
        if split.resolve() != local_case / "split_manifest.json":
            raise ValueError("Generic-only split manifest differs from its actual case")
        artifacts = (row.get("generic_artifact_binding") or {}).get("artifacts") or {}
        binding = row.get("generic_artifact_binding") or {}
        if (binding.get("quality_reference") != row.get("quality_reference")
                or binding.get("producer_source") != (row.get("quality_reference") or {}).get("request")):
            raise ValueError("Generic-only producer source/Quality binding differs")
        boundary = binding.get("boundary_contract") or {}
        if ((len(boundary.get("part1_outputs") or {}) >= 2 or boundary.get("passthrough_inputs"))
                and row.get("boundary_observation") != boundary):
            raise ValueError("Generic-only multi-tensor cuts require the actual bound boundary observation")
        if any(role not in artifacts for role in ARTIFACT_ROLES):
            raise ValueError("Generic-only actual artifact/build receipt set is incomplete")
        for role, artifact in artifacts.items():
            if (not PurePosixPath(str(artifact.get("path") or "")).is_absolute()
                    or len(str(artifact.get("sha256") or "")) != 64):
                raise ValueError("Generic-only artifact descriptor invalid: " + role)
            if artifact.get("local_path"):
                read_bound(artifact)
        expected = {role: {k: artifacts[role][k] for k in ("path", "sha256")}
                    for role in ("part1_runtime", "engine")}
        if row.get("expected_runtime_artifacts") != expected:
            raise ValueError("Generic-only runtime artifacts differ from their producer binding")
        if (row.get("original_run_cfg") or {}).get("native_split_quality_binding"):
            raise ValueError("Generic-only must not masquerade as a Native split binding")
        runtime_environment = environment(row)
        proof = validate_quality_reference(row)
        if row.get("quality_status") != proof["status"]:
            raise ValueError("Generic-only quality decision differs from the actual CPU report")
        relative = PurePosixPath(row["case_output_relative"])
        if str(relative) != "/".join(row[k] for k in ("setup_id", "model_id", "case_id")):
            raise ValueError("Generic-only output path differs from its identity")
        if "runtime_config" in row:
            config = row["runtime_config"]
            if (row.get("remote_python_executable") != runtime_environment["python"]
                    or row.get("remote_environment") != runtime_environment["environment"]):
                raise ValueError("Generic-only dispatch interpreter/environment differs from its actual source")
            from .runners.generic_only import validate_generic_only_config
            validate_generic_only_config(config)
            policy, depth = requests[identity["comparison_backend"]]
            if config.get("stage1_execution") != {"policy": policy, "inflight": depth}:
                raise ValueError("Generic-only fixed asynchronous backend policy differs")
            if config.get("identity") != identity or config.get("execution_id") != plan.get("execution_id"):
                raise ValueError("Generic-only prepared config identity differs")
            if (any(config.get(key) != value for key, value in
                    (("repetitions", 3), ("warmup_count", 100), ("task_count", 1000)))
                    or config.get("measurement_boundary") != "prepared_input_before_p1_to_last_completed_task"
                    or Path(config.get("original_run", "")).resolve() != Path(primary).resolve()):
                raise ValueError("Generic-only work budget, boundary or producer run differs")
            projected, projected_runtime, _ = _project_artifacts(row, config["case_dir"])
            if (config.get("quality_reference") != row["quality_reference"]
                    or config.get("generic_artifact_binding") != projected
                    or config.get("expected_runtime_artifacts") != projected_runtime):
                raise ValueError("Generic-only prepared source/quality binding differs")
            remote = PurePosixPath(row["remote_root"])
            if not remote.is_absolute() or ".." in remote.parts or not plan.get("execution_id"):
                raise ValueError("Generic-only session root or execution identity invalid")
            expected_config, expected_staging = runtime_configuration(row, str(remote), plan["execution_id"],
                stage1_execution={"policy": policy, "inflight": depth})
            # runtime_source_files/execution_environment are added by the normal
            # frozen-source binder. Every operation-selecting input is already
            # determined by this source projection and may not drift afterwards.
            if any(config.get(key) != value for key, value in expected_config.items()):
                raise ValueError("Generic-only prepared operation differs from its authorized source projection")
            if row.get("staging_files") != expected_staging:
                raise ValueError("Generic-only prepared staging differs from its authorized source projection")
            root = Path(plan["output_root"]).resolve()
            if root.is_relative_to(Path(primary).resolve()):
                raise ValueError("Generic-only output must preserve its producer run")
            expected_result = root / "results" / str(relative) / "generic_completion_results.json"
            if Path(row["result_path"]).resolve() != expected_result:
                raise ValueError("Generic-only result escapes its authorized output")
            if not remote.is_absolute() or any(remote not in PurePosixPath(x["remote"]).parents
                    or ".." in PurePosixPath(x["remote"]).parts for x in row["staging_files"]):
                raise ValueError("Generic-only staging path escapes its session")
    return rows


def runtime_configuration(row, remote_root, execution_id, *, stage1_execution=None):
    """Stage existing producer metadata; compiled cache files remain read-only."""
    from .runners.task_completion import normalize_stage1_execution
    case_dir = f"{remote_root}/cases/{row['model_id']}/{row['case_id']}"
    config_dir = f"{remote_root}/configs/{row['model_id']}/{row['case_id']}"
    original = copy.deepcopy(row["original_run_cfg"])
    original["manifest"] = case_dir + "/split_manifest.json"
    original["native_split_quality_binding"] = ""
    staged = [{"local": str(read_bound(row["split_manifest"])), "remote": original["manifest"],
               "sha256": row["split_manifest"]["sha256"], "role": "original_manifest"}]
    binding, expected, artifact_staging = _project_artifacts(row, case_dir)
    staged.extend(artifact_staging)
    manifest = _json(row["split_manifest"])
    for role in ("part1", "part2"):
        relative = manifest.get(role) or (manifest.get("models", {}).get(role) or {}).get("path")
        relative = PurePosixPath(str(relative or ""))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
            raise ValueError("Generic-only source ONNX path escapes its case")
        local = Path(row["original_local_case_dir"]) / str(relative)
        descriptor = row["generic_artifact_binding"]["artifacts"]["source_" + role + "_onnx"]
        if file_sha256(local) != descriptor["sha256"]:
            raise ValueError("Generic-only source ONNX changed: " + role)
        staged.append({"local": str(local), "remote": case_dir + "/" + str(relative),
                       "sha256": descriptor["sha256"], "role": role + "_onnx_metadata",
                       "existing_remote": descriptor["path"] if descriptor.get("reuse_remote") is True else None})
    config = {key: copy.deepcopy(row[key]) for key in (
        "identity", "task", "original_run", "input_binding", "completion_contract",
        "functional_reference", "expected_runtime_artifacts", "generic_artifact_binding", "quality_reference")}
    config.update(admission_mode="generic_only", measurement_mode=MODE, execution_id=execution_id,
        case_dir=case_dir, config_directory=config_dir, original_run_cfg=original,
        generic_artifact_binding=binding, expected_runtime_artifacts=expected,
        repetitions=3, warmup_count=100, task_count=1000,
        build_allowed=False, fallback_allowed=False, automatic_retry=False,
        queue_depth=int(original["throughput_queue_depth"]), timeout_s=3600,
        measurement_boundary="prepared_input_before_p1_to_last_completed_task",
        stage1_execution=normalize_stage1_execution(stage1_execution or row.get("stage1_execution")))
    if row.get("mixed_input_control"):
        config["mixed_input_control"] = copy.deepcopy(row["mixed_input_control"])
    if row.get("boundary_observation"):
        config["boundary_observation"] = copy.deepcopy(row["boundary_observation"])
    if row.get("additional_staging_files"):
        staged.extend(copy.deepcopy(row["additional_staging_files"]))
    from .runners.generic_passthrough import project_bindings
    canonical, canonical_staging = project_bindings(row, case_dir)
    if canonical is not None:
        config["canonical_full_input_bindings"] = canonical
        staged.extend(canonical_staging)
    return config, staged


def configuration_comparison(row, current, current_path):
    """Compare an explicitly bound producer snapshot, retaining unknowns."""
    from .resources.remote_scripts.native_host_telemetry import _identity
    descriptor = row.get("configuration_reference")
    if not descriptor:
        raise ValueError("Generic-only producer configuration reference is missing")
    historical = _json(descriptor)
    left, right = _identity(historical), _identity(current)
    checks, unknown, differences = {}, [], []
    for key in ("cpu_governors", "cpu_min_max", "device_governors", "device_min_max"):
        if not left.get(key) or not right.get(key):
            unknown.append(key)
        else:
            checks[key] = {"current": right[key], "historical": left[key], "matches": left[key] == right[key]}
            if left[key] != right[key]:
                differences.append(key)
    def mode(value):
        return str(((value.get("nvidia_commands") or {}).get("nvpmodel_mode") or {}).get("stdout") or "").strip()
    if mode(historical) and mode(current):
        checks["nvpmodel_mode"] = {"matches": mode(historical) == mode(current),
            "current": mode(current), "historical": mode(historical)}
        if mode(historical) != mode(current):
            differences.append("nvpmodel_mode")
    else:
        unknown.append("nvpmodel_mode")
    return {"status": "conflict" if differences else "observed_no_known_configuration_conflict",
        "current_snapshot": str(current_path), "historical_snapshots": [descriptor["path"]],
        "checks": checks, "differences": differences,
        "unknown_or_not_equalized": unknown + ["historical_thermal_state_not_equalized"],
        "note": "Generic-only producer comparison; no Native reference or device tuning."}
