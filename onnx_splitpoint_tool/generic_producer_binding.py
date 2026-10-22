"""Explicit creation-file binding for new Generic FPS plans.

The original run manifest remains live lifecycle evidence.  No copy or digest
of that mutable document is substituted for the real creation authorities.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import yaml


MODE = "creation_snapshot_v1"
SCHEMA = "onnx-splitpoint/generic-producer-run-binding"
MEMBERS = {
    "creation_snapshot": "profile_start_snapshot.json",
    "source_profile": "profile_source.yaml",
    "profile": "profile.yaml",
    "execution_plan": "effective_execution_plan.json",
}


def _require(value, message):
    if not value:
        raise ValueError("Generic producer creation binding: " + message)


def _root(primary):
    path = Path(primary).absolute()
    _require(path.is_dir() and not any(p.is_symlink() for p in (path, *path.parents)), "original run is missing or aliased")
    return path.resolve()


def _member(primary, name):
    path = primary / name
    _require(not any(p.is_symlink() for p in (path, *path.parents)) and path.is_file(), "missing creation/source member: " + name)
    return path


def _live(primary, name):
    """Read one complete atomic-writer version, without binding progress bytes."""
    path = _member(primary, name)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        value = json.load(stream)
        after = os.fstat(stream.fileno())
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    _require(stat.S_ISREG(before.st_mode) and fields(before) == fields(after) and isinstance(value, dict), "live source changed while read: " + name)
    return value


def _descriptor(path):
    from .generic_only_plan import file_sha256
    return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _creation(primary, binding):
    from .generic_only_plan import read_bound
    from .workflow.start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata
    values = {}
    for key, name in MEMBERS.items():
        descriptor = binding[key]
        path = _member(primary, name)
        _require(isinstance(descriptor, dict) and Path(str(descriptor.get("path") or "")) == path
                 and read_bound(descriptor) == path
                 and type(descriptor.get("size_bytes")) is int and descriptor["size_bytes"] == path.stat().st_size,
                 "creation descriptor differs: " + name)
        values[key] = yaml.safe_load(path.read_text()) if path.suffix == ".yaml" else json.loads(path.read_text())
        _require(isinstance(values[key], dict), "creation object missing: " + name)
    archived = values["creation_snapshot"]
    rebuilt = build_profile_start_snapshot(
        **{key: str(archived.get(key) or "") for key in ("profile_request", "profile_id", "profile_path", "profile_source")},
        source_profile=values["source_profile"], resolved_profile=values["profile"],
        runtime_bindings=archived.get("runtime_bindings") or {},
        schema_version=int(archived.get("schema_version") or 1))
    _require(public_start_snapshot_metadata(rebuilt) == archived, "normal start snapshot differs")
    targets = (archived.get("runtime_bindings") or {}).get("hardware_targets")
    _require(isinstance(targets, list) and targets and all(isinstance(t, dict) and t.get("id") for t in targets)
             and len({t["id"] for t in targets}) == len(targets), "creation hardware targets missing or duplicate")
    profile = values["profile"]
    runs = [r for r in profile.get("run_profiles", []) if r.get("enabled", True)]
    _require((profile.get("energy") or {}).get("enabled") is False
             and (profile.get("native_producers") or {}).get("enabled") is False
             and runs and all(r.get("quality_source_only") is True for r in runs), "not a source-only Quality producer")
    return values, targets, runs


def _projected_guard(primary, expected):
    from .workflow.run_control import build_resume_contract
    manifest = _live(primary, "run_manifest.json")
    control = _live(primary, "jobs/workflow_control.json")
    sessions = manifest.get("execution_sessions")
    session = sessions[-1] if isinstance(sessions, list) and sessions and isinstance(sessions[-1], dict) else {}
    stable_options = build_resume_contract(profile_payload={}, effective_execution_plan={}, options=manifest.get("options") or {})
    contract = expected["resume_contract"]
    _require(manifest.get("schema") == "onnx-splitpoint/evaluation-run-manifest"
             and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
             and manifest.get("run_id") == expected["run_id"] == primary.name
             and manifest.get("run_dir") == str(primary)
             and manifest.get("current_session_id") == session.get("session_id") == expected["session_id"]
             and manifest.get("resume_contract") == contract
             and stable_options["stable_options_sha256"] == contract["stable_options_sha256"]
             and all(session.get(k) == contract[k] for k in ("profile_sha256", "plan_sha256"))
             and (manifest.get("profile_start_snapshot") or {}).get("snapshot_sha256") == expected["snapshot_sha256"]
             and manifest.get("hardware_targets") == expected["hardware_targets"], "original run/session/target contract changed")
    _require(control.get("schema") == "onnx-splitpoint/workflow-control"
             and type(control.get("schema_version")) is int and control["schema_version"] == 1
             and control.get("run_id") == expected["run_id"] and control.get("session_id") == expected["session_id"],
             "original workflow control run/session changed")


def producer_group_identity(binding):
    _require(isinstance(binding, dict) and binding.get("mode") == MODE and binding.get("schema") == SCHEMA,
             "explicit creation_snapshot_v1 mode required")
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_producer_run_binding(primary, binding, rows=()):
    from . import generic_only_plan as admission
    from .workflow.run_control import build_resume_contract
    primary = _root(primary)
    required = {"schema", "schema_version", "mode", "run_id", "run_dir", "session_id", "resume_contract", *MEMBERS}
    _require(isinstance(binding, dict) and set(binding) == required and binding.get("schema") == SCHEMA
             and type(binding.get("schema_version")) is int and binding["schema_version"] == 1 and binding.get("mode") == MODE
             and binding.get("run_id") == primary.name and binding.get("run_dir") == str(primary)
             and re.fullmatch(r"[0-9a-f]{32}", str(binding.get("session_id") or "")), "explicit original run/session binding missing")
    values, targets, runs = _creation(primary, binding)
    manifest = _live(primary, "run_manifest.json")
    contract = build_resume_contract(profile_payload=values["profile"], effective_execution_plan=values["execution_plan"], options=manifest.get("options") or {})
    _require(contract == binding["resume_contract"], "creation resume contract differs")
    expected = {"run_id": binding["run_id"], "session_id": binding["session_id"], "resume_contract": contract,
                "snapshot_sha256": values["creation_snapshot"]["snapshot_sha256"], "hardware_targets": targets}
    _projected_guard(primary, expected)
    models = {r.get("id") for r in (values["profile"].get("model_suite") or {}).get("primary", [])}
    plan = values["execution_plan"]
    _require(models and set(plan.get("models") or []) == models
             and set(plan.get("logical_run_profiles") or []) == {r.get("id") for r in runs}, "creation model/run scope differs")
    target_ids = {t["id"] for t in targets}
    for row in rows:
        identity = row.get("identity") or row
        _require(row.get("producer_run_binding") == binding and Path(row.get("original_run") or "").resolve() == primary,
                 "row creation binding differs")
        _require(identity.get("model_id") in models and identity.get("setup_id") in target_ids
                 and sum(r.get("hardware_setup_id") == identity.get("setup_id") and identity.get("case_id") in (r.get("case_ids") or []) for r in runs) == 1,
                 "case/setup absent or ambiguous in creation scope")
    scope = admission._VALIDATION_SCOPE.get()
    if scope is not None:
        key = producer_group_identity(binding)
        _require(key in scope.producer_guards or len(scope.producer_guards) < 16, "creation guard bound exceeded")
        frozen_expected = deepcopy(expected)
        scope.producer_guards[key] = lambda: _projected_guard(primary, frozen_expected)
    return {"run_id": binding["run_id"], "session_id": binding["session_id"], "hardware_targets": deepcopy(targets)}


def build_producer_run_binding(primary, *, session_id):
    from .generic_only_plan import validation_scope
    from .workflow.run_control import build_resume_contract
    primary = _root(primary)
    with validation_scope():
        binding = {"schema": SCHEMA, "schema_version": 1, "mode": MODE, "run_id": primary.name,
                   "run_dir": str(primary), "session_id": session_id,
                   **{key: _descriptor(_member(primary, name)) for key, name in MEMBERS.items()}}
        values, _, _ = _creation(primary, binding)
        binding["resume_contract"] = build_resume_contract(profile_payload=values["profile"],
            effective_execution_plan=values["execution_plan"], options=_live(primary, "run_manifest.json").get("options") or {})
        validate_producer_run_binding(primary, binding)
        return binding


def load_completed_case_result(primary, result_path, *, producer_run_binding):
    """Load an actual per-case result; full scientific admission remains normal."""
    from .generic_only_plan import _json, read_bound, validation_scope
    primary = _root(primary)
    with validation_scope():
        validate_producer_run_binding(primary, producer_run_binding)
        path = Path(result_path).absolute()
        _require(path.is_relative_to(primary) and path.name.endswith("_central_result.json"), "normal per-case result path required")
        _member(primary, path.relative_to(primary).as_posix())
        source = _descriptor(path)
        result = _json(source)
        _require(isinstance(result, dict) and result.get("schema") == "onnx-splitpoint/management-paired-quality-result"
                 and result.get("status") == result.get("technical_status") == "completed"
                 and result.get("task") == "detection" and result.get("variant") == "composed"
                 and result.get("execution_role") == "generic_composed_quality_candidate"
                 and result.get("eval_run_id") == (result.get("producer_identity") or {}).get("eval_run_id") == primary.name
                 and result.get("collection_eval_run_id", primary.name) == primary.name, "per-case result producer/run is incomplete or different")
        relative = Path(str(result.get("source_request") or ""))
        _require(relative.parts and not relative.is_absolute() and ".." not in relative.parts, "per-case request path escaped")
        request = _member(primary, relative.as_posix())
        _require(path == request.with_name(request.stem.replace("_request", "") + "_central_result.json"), "per-case result is not its normal request sibling")
        request_source = {"path": str(request), "sha256": str(result.get("source_request_sha256") or "").removeprefix("sha256:")}
        read_bound(request_source)
        request_value = _json(request_source)
        _require(request_value.get("eval_run_id") == (request_value.get("producer_identity") or {}).get("eval_run_id") == primary.name,
                 "per-case request belongs to another producer run")
        row = {"original_run": str(primary), "producer_run_binding": producer_run_binding,
               "identity": {k: result.get(k) for k in ("model_id", "case_id", "setup_id")}}
        validate_producer_run_binding(primary, producer_run_binding, [row])
        return deepcopy(result), source
