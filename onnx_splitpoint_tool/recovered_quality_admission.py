"""Explicit first publication of completed, recovered Generic Quality evidence.

This entry point can read an existing completed statistics cache only. It never
constructs a service, computes bootstrap draws, dispatches a runner or changes
the interrupted EvaluationRun's lifecycle. A new publication is dated now.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import yaml

from .quality_cache import PersistentQualityCache, json_fingerprint
from .quality_result_enrichment import enrich_completed_quality_result
from .quality_service import (quality_request_from_manifest, prepare_evaluation,
    _cached_guardrail_contract_matches_request, _rebind_request_scoped_result)
from .workflow.run_control import (EvaluationRunLock, build_resume_contract,
    validate_generic_orphan_recovery_owner)
from .workflow.runner import EvaluationWorkflowRunner as Runner
from .workflow import generic_quality_sidecars
from .workflow.artifact_cache_preflight import resolve_generated_benchmark_suite_dir
from .workflow.scientific_reporting import project_central_quality_status


def _require(condition, reason):
    if not condition:
        raise ValueError("recovered_quality_admission: " + reason)


class _Sources:
    def __init__(self):
        self.rows = {}

    def read(self, descriptor):
        _require(isinstance(descriptor, dict), "source descriptor missing")
        path = Path(str(descriptor.get("path") or descriptor.get("local_path") or ""))
        _require(path.is_absolute() and ".." not in path.parts
                 and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe source path")
        size, digest = descriptor.get("size_bytes"), descriptor.get("sha256")
        _require(type(size) is int and size >= 0 and isinstance(digest, str) and len(digest) == 64,
                 "source size/digest missing")
        data = Runner._read_management_reference_nofollow(path.parent, Path(path.name),
            expected_size=size, require_single_link=False)
        _require(hashlib.sha256(data).hexdigest() == digest, "source bytes changed: " + str(path))
        row = {"path": str(path), "size_bytes": size, "sha256": digest}
        _require(path not in self.rows or self.rows[path] == row, "source has conflicting bindings")
        self.rows[path] = row
        return data

    def pin(self, path):
        path = Path(path)
        _require(path.is_absolute() and ".." not in path.parts
                 and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe source path")
        data = Runner._read_management_reference_nofollow(path.parent, Path(path.name),
            expected_size=path.stat().st_size, require_single_link=False)
        row = {"path": str(path), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.read(row)
        return row

    def bind(self, path):
        row = self.pin(path)
        return json.loads(self.read(row)), row

    def verify(self):
        for row in list(self.rows.values()):
            self.read(row)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_original_setup_plan_binding(logical, plan, setup):
    """Join normal logical group names without confusing them with DUT ids."""
    from .execution_plan import _setup_class
    declared, groups = plan.get("logical_run_profiles"), plan.get("setup_groups")
    reason = "original setup/plan binding mismatch: "
    _require(isinstance(declared, list) and declared
             and all(isinstance(run_id, str) and run_id for run_id in declared)
             and len(set(declared)) == len(declared)
             and isinstance(groups, dict) and groups,
             reason + "logical run/group declarations invalid")
    members = []
    for group, run_ids in groups.items():
        _require(isinstance(group, str) and group and isinstance(run_ids, list) and run_ids
                 and all(isinstance(run_id, str) and run_id for run_id in run_ids),
                 reason + "group membership invalid")
        members.extend((group, run_id) for run_id in run_ids)
    planned = [run_id for _, run_id in members]
    _require(len(planned) == len(set(planned)) and set(planned) == set(declared),
             reason + "group membership missing, foreign or ambiguous")
    logical_id = logical["id"]
    selected = [group for group, run_id in members if run_id == logical_id]
    _require(logical_id in declared and len(selected) == 1
             and selected[0] in {setup, _setup_class(logical_id)},
             reason + "logical profile does not belong to its setup group")


def _cache_result(request, cache_root, statistics, sources, identity):
    key, payload = prepare_evaluation(request)
    if statistics.get("engine") == "optimized_coco_v1":
        from .quality_statistics import numerical_environment, validate_bound_order, validate_canonical_prediction_fields
        validate_bound_order(payload)
        validate_canonical_prediction_fields(payload)
        key = json_fingerprint({"schema": "optimized-quality-pair-v1", "legacy": key,
            "image_ids": payload["image_ids"], "annotations": payload["annotations"],
            "environment": numerical_environment()})
        payload["evaluation_fingerprint"] = key
    _require(Path(cache_root).is_dir(), "completed cache root missing")
    cache = PersistentQualityCache(cache_root)
    _require(cache.path_for(key).is_file(), "completed statistics cache miss")
    _, cache_source = sources.bind(cache.path_for(key))
    result = cache.get(key)
    _require(result is not None and result.get("status") == "completed"
             and _cached_guardrail_contract_matches_request(request.metric_gate_config, result),
             "completed statistics cache miss or contract mismatch")
    _require(all(result.get(field) == payload[field] for field in (
        "evaluation_fingerprint", "reference_predictions_sha256", "candidate_predictions_sha256", "annotations_sha256")),
        "completed cache scientific identity mismatch")
    _require(("eval_run_id" not in result or result["eval_run_id"] == identity["eval_run_id"])
             and ("source_request_sha256" not in result
                  or Runner._quality_sha256_token(result["source_request_sha256"])
                  == Runner._quality_sha256_token(identity["source_request_sha256"])),
             "cached request identity mismatch")
    result = _rebind_request_scoped_result(result, payload)
    result["statistics_observation"] = {**result.get("statistics_observation", {}),
        "cache_state": "complete_result_reused", "draws_recomputed": 0}
    return result, cache_source


def admit_preserved_completed_quality(*, original_run_dir, recovery_receipt,
        request_descriptors, completed_cache_root, canonical_reference_descriptor,
        profile_descriptor, authorization_descriptor, out_dir):
    """Publish a newly authorized, cache-only recovery of an original run.

    Descriptors bind absolute path, SHA-256 and byte size. ``out_dir`` is a new
    member of ``quality_management/recovered_publications`` in the original run
    so the ordinary source reader can retain run-relative request paths. The
    normal summary is created exclusively and explicitly records recovery scope;
    an existing historical summary or result is never replaced.
    """
    run = Path(original_run_dir).absolute()
    out = Path(out_dir).absolute()
    sources = _Sources()
    _require(run.is_dir() and not any(p.is_symlink() for p in (run, *run.parents)), "unsafe original run")
    _require(out.parent == run / "quality_management/recovered_publications"
             and out.name not in {"", ".", ".."} and not out.exists()
             and not any(p.is_symlink() for p in (out, *out.parents)), "publication output must be a new original-run member")
    canonical_summary = run / "quality_management/central_quality_summary.json"
    _require(not canonical_summary.exists(), "original summary already exists; preserve it")
    _require(isinstance(request_descriptors, list) and request_descriptors
             and len({d.get("sha256") for d in request_descriptors}) == len(request_descriptors), "request scope invalid")
    auth = json.loads(sources.read(authorization_descriptor))
    receipt = json.loads(sources.read(recovery_receipt))
    profile = yaml.safe_load(sources.read(profile_descriptor))
    _require(Path(profile_descriptor["path"]) == run / "profile.yaml", "profile is not the original run profile")
    manifest, manifest_source = sources.bind(run / "run_manifest.json")
    plan, _ = sources.bind(run / "effective_execution_plan.json")
    _require(receipt.get("run_id") == manifest.get("run_id") == run.name
             and receipt.get("run_dir") == manifest.get("run_dir") == str(run)
             and receipt.get("session_id") == manifest.get("current_session_id")
             and receipt.get("manifest_after_sha256") == manifest_source["sha256"]
             and manifest.get("schema") == "onnx-splitpoint/evaluation-run-manifest"
             and manifest.get("status") == manifest.get("technical_status") == "cancelled"
             and any(s.get("session_id") == receipt.get("session_id") and s.get("status") == "cancelled"
                     for s in manifest.get("execution_sessions", [])), "original cancelled run/recovery binding mismatch")
    _require(build_resume_contract(profile_payload=profile, effective_execution_plan=plan,
                options=manifest.get("options") or {}) == manifest.get("resume_contract"), "original profile/plan contract changed")
    _require(auth.get("schema") == "onnx-splitpoint/recovered-quality-publication-authorization"
             and auth.get("schema_version") == 1 and auth.get("approved") is True
             and auth.get("action") == "publish_preserved_completed_quality"
             and auth.get("run_id") == run.name and auth.get("session_id") == receipt["session_id"]
             and sorted(auth.get("request_sha256s") or []) == sorted(d["sha256"] for d in request_descriptors)
             and auth.get("profile_sha256") == profile_descriptor["sha256"]
             and auth.get("reference_sha256") == canonical_reference_descriptor["sha256"]
             and auth.get("recovery_receipt_sha256") == recovery_receipt["sha256"]
             and auth.get("authorized_at") and auth.get("provenance"), "explicit publication authorization mismatch")
    lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner={})
    _require(lock.lock_path.is_file(), "original owner authority missing")
    lock.acquire_for_recovery()
    staging = None
    try:
        _require(validate_generic_orphan_recovery_owner(lock) == receipt, "committed recovery owner mismatch")
        import inspect
        sources.pin(__file__)
        sources.pin(inspect.getfile(enrich_completed_quality_result))
        sources.pin(inspect.getfile(Runner))
        results, copies, reference_statuses, cache_sources = [], [], {}, []
        seen = set()
        quality = profile.get("quality_gate") or {}
        statistics = quality.get("statistics") or {}
        _require(statistics.get("execution_location") == "central_management", "original central policy required")
        for descriptor in request_descriptors:
            request_path = Path(descriptor["path"])
            payload = json.loads(sources.read(descriptor))
            producer = payload.get("producer_identity") or {}
            model, case, setup = (producer.get(k) for k in ("model_id", "case_id", "setup_id"))
            _require(all(isinstance(token, str) and token not in {"", ".", ".."}
                         and Path(token).parts == (token,) for token in (model, case, setup))
                     and (model, case, setup) not in seen
                     and payload.get("eval_run_id") == producer.get("eval_run_id") == run.name,
                     "request original producer/run identity mismatch")
            seen.add((model, case, setup))
            runs = [r for r in profile.get("run_profiles", []) if r.get("enabled", True)
                    and r.get("hardware_setup_id") == setup and case in (r.get("case_ids") or [])]
            _require(len(runs) == 1 and runs[0].get("quality_source_only") is True
                     and runs[0].get("variants") == ["composed"]
                     and setup in (profile.get("hardware") or {}).get("selected_setups", [])
                     and model in plan.get("models", []), "request is outside original source-only scope")
            logical = runs[0]
            _validate_original_setup_plan_binding(logical, plan, setup)
            config_source = (logical.get("generic_quality_completion_sources") or {}).get(case)
            config = json.loads(sources.read(config_source))
            _require(config_source["sha256"] == (logical.get("generic_quality_completion_config_sha256") or {}).get(case)
                     == (payload.get("generic_completed_quality") or {}).get("config_file", {}).get("sha256"), "request/config source binding mismatch")
            sources.bind(run / "models" / model / "model_manifest.json")
            _require((producer.get("model") or {}).get("source_onnx_sha256")
                     == Runner._read_canonical_model_sha256_for_quality(run_dir=run, model_id=model), "canonical model source mismatch")
            identity = Runner._quality_request_identity(request_path, model_id=model, manifest=payload)
            _require(identity.get("identity_valid") is True and identity.get("producer_identity_validated") is True
                     and identity.get("execution_role") == "generic_composed_quality_candidate"
                     and identity.get("eval_run_id") == run.name, "normal Generic producer identity rejected")
            if model not in reference_statuses:
                suite = run / "models" / model / "benchmark_set/legacy_suite"
                benchmark, _ = sources.bind(suite / "benchmark_plan.json")
                declared = (quality.get("management_reference") or {}).get("existing_reference") or {}
                for key in ("source_status", "policy_source"):
                    sources.read(declared.get(key))
                sources.read({"path": declared.get("reference_path"), "sha256": declared.get("reference_sha256"),
                              "size_bytes": declared.get("reference_size_bytes")})
                # The ordinary adoption route may reuse, but must not create a
                # missing historical reference during this recovery operation.
                reference_bytes = sources.read(canonical_reference_descriptor)
                _require(Path(canonical_reference_descriptor["path"]).is_relative_to(run), "reference outside original run")
                status = generic_quality_sidecars.bind_existing_generic_cpu_reference(suite_dir=suite, run_root=run, model_id=model,
                    benchmark_plan=benchmark, quality_gate_policy=quality, require_projected=True)
                _require(status is not None, "adopted reference status missing")
                records, status, sealed_bytes = Runner._read_management_reference_status(run_dir=run, model_id=model,
                    raw_status=status, expected_source_contract=status["source_contract_sha256"])
                _require(status.get("reference_path") == canonical_reference_descriptor["path"]
                         and status.get("reference_sha256") == canonical_reference_descriptor["sha256"], "adopted reference identity mismatch")
                _require(sealed_bytes == reference_bytes, "adopted reference bytes changed")
                reference_statuses[model] = (status, sealed_bytes)
            status, reference_bytes = reference_statuses[model]
            loaded = quality_request_from_manifest(request_path, verify_artifacts=True,
                reference_artifact=canonical_reference_descriptor, reference_artifact_bytes=reference_bytes)
            expected = ((profile.get("validation_execution") or {}).get("max_items") or {}).get("detection")
            expected_gate = dict(quality.get("detection") or {})
            if quality.get("reporting_policy") is not None:
                expected_gate["reporting_policy"] = quality["reporting_policy"]
            _require(payload.get("task") == "detection" and payload.get("variant") == "composed"
                     and type(expected) is int and len(loaded.reference_records) == expected
                     and loaded.repetitions == statistics.get("bootstrap_repetitions")
                     and loaded.seed == statistics.get("seed")
                     and loaded.confidence_level == statistics.get("confidence_level")
                     and payload.get("metric_gate_config") == expected_gate,
                     "original frozen dataset/policy/statistics mismatch")
            candidate = Runner._attest_preserved_quality_candidate(request_path, payload)
            sources.read(candidate)
            raw, cache_source = _cache_result(loaded, completed_cache_root, statistics, sources, identity)
            cache_sources.append(cache_source)
            final_request = out / "quality_inputs" / setup / case / ("results_" + identity["source_run_id"]) / "task_quality_inputs" / request_path.name
            result = enrich_completed_quality_result(raw, request_identity=identity, manifest=payload,
                model_id=model, request_path=final_request, run_dir=run, collection_eval_run_id=run.name,
                case_id=case, run_id=identity["source_run_id"], reference_status=status,
                producer_flat_field_names=Runner._trt_quality_flat_field_names())
            from .runners.generic_quality_completion import validate_completed_quality_binding
            from .runners import generic_quality_completion as verifier
            observer = Path(verifier.__file__)
            observer_sha = verifier._sha(observer)
            proof = payload.get("generic_completed_quality") or {}
            same_observer = isinstance(proof, dict) and any(
                isinstance(source, dict) and Path(str(source.get("path") or "")).name == observer.name
                and source.get("sha256") == observer_sha for source in proof.get("runtime_source_files") or [])
            authority = None
            if not same_observer:
                suite = resolve_generated_benchmark_suite_dir(run_dir=run, model_id=model)
                benchmark, _ = sources.bind(suite / "benchmark_plan.json")
                authority = generic_quality_sidecars.admit_generic_quality_producer_sources(
                    profile=profile, benchmark_plan=benchmark, suite_dir=suite,
                    config_path=Path(config_source["path"]), model_id=model)
            validate_completed_quality_binding(payload, result, config["completion_contract"],
                config["expected_runtime_artifacts"], request_path=request_path,
                reference_descriptor=canonical_reference_descriptor,
                producer_source_authority=authority)
            modules = [verifier, verifier._product(), verifier._three_stage(),
                       verifier._product()._yolo_module, verifier._endpoint]
            if (payload.get("generic_completed_quality") or {}).get("existing_part2_bridge"):
                modules.append(verifier._bridge_validator())
            for module in modules:
                sources.pin(module.__file__)
            result["recovery_publication"] = {"scope": "recovered_completed_quality", "authorization": authorization_descriptor,
                "original_recovery_receipt": recovery_receipt, "original_run_status": "cancelled", "statistics_recomputed": False}
            results.append(result)
            copies.extend([(request_path, final_request.relative_to(out), descriptor),
                           (Path(candidate["path"]), final_request.relative_to(out).with_name(Path(candidate["path"]).name), candidate)])
        created = datetime.now(timezone.utc).isoformat()
        summary = {"schema": "onnx-splitpoint/central-quality-summary", "schema_version": 1,
            "created_at": created, "status": "completed", "request_count": len(results), "results": results,
            "publication_scope": "recovered_completed_quality", "eval_run_id": run.name,
            "original_run_status": "cancelled", "original_evaluation_run_completed": False,
            "recovery_receipt": recovery_receipt, "authorization": authorization_descriptor,
            "publication_receipt": str(out / "publication_receipt.json"), "statistics_recomputed": False}
        reporting = project_central_quality_status(summary, dataset_tier=str(quality.get("dataset_tier") or ""))
        for field in ("technical_status", "quality_decision", "scientific_status", "scientific_pass",
                      "full_reference_quality_decision", "full_reference_scientific_pass",
                      "campaign_quality_decision", "campaign_quality_missing_count", "decision_counts"):
            summary[field] = reporting[field]
        summary["aggregate_quality_decision"] = reporting["quality_decision"]
        summary["quality_result_count"] = reporting["result_count"]
        sources.verify()
        _require(validate_generic_orphan_recovery_owner(lock) == receipt, "recovery owner changed before publication")
        out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".quality-admission-", dir=out.parent))
        for original, relative, descriptor in copies:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            sources.read({**descriptor, "path": str(target)})
            del sources.rows[target]
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
        result_descriptors = []
        for index, result in enumerate(results):
            d = _write(staging / f"result_{index:03d}.json", result)
            result_descriptors.append({**d, "path": str(out / Path(d["path"]).name)})
        summary_descriptor = _write(staging / "central_quality_summary.json", summary)
        summary_descriptor["path"] = str(out / "central_quality_summary.json")
        for model, (status, _) in reference_statuses.items():
            _write(staging / (model + "_reference_adoption.json"), status)
        publication = {"schema": "onnx-splitpoint/recovered-quality-publication", "schema_version": 1,
            "state": "committed", "created_at": created, "run_id": run.name, "session_id": receipt["session_id"],
            "run_dir": str(run), "publication_id": out.name, "original_manifest": manifest_source,
            "original_run_status": "cancelled", "authorization": authorization_descriptor,
            "original_recovery_receipt": recovery_receipt, "sources": list(sources.rows.values()),
            "cache_sources": cache_sources, "results": result_descriptors, "summary": summary_descriptor,
            "collected_sources": [{"original_path": str(original), "path": str(out / relative),
                                   "sha256": descriptor["sha256"], "size_bytes": descriptor["size_bytes"]}
                                  for original, relative, descriptor in copies],
            "statistics_recomputed": False, "hardware_dispatched": False}
        receipt_descriptor = _write(staging / "publication_receipt.json", publication)
        receipt_descriptor["path"] = str(out / "publication_receipt.json")
        # Recheck originals after copying; staged members are checked above.
        for path in [p for p in sources.rows if p.is_relative_to(staging)]:
            del sources.rows[path]
        sources.verify()
        for directory in [p for p in staging.rglob("*") if p.is_dir()] + [staging]:
            _sync_directory(directory)
        os.rename(staging, out)
        staging = None
        _sync_directory(out.parent)
        os.link(out / "central_quality_summary.json", canonical_summary)
        _sync_directory(canonical_summary.parent)
        return {"summary": summary_descriptor, "results": result_descriptors,
                "publication_receipt": receipt_descriptor, "canonical_summary": str(canonical_summary)}
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        lock.release(write_released_state=False)
