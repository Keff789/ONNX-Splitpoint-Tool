"""Selective continuation of an existing EvaluationRun through the normal runner.

The plan is a projection of existing required scopes and results, not a new
identity store. No device, compiler or measurement is touched by plan-only.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import fields
import json
import hashlib
from pathlib import Path
import shutil
from typing import Any, Mapping

from .artifacts import read_json, sha256_file, stable_json_dumps, write_json
from .contracts import WorkflowOptions


def run_token(value: Any) -> str:
    return str(value or "").lower().replace("_to_tensorrt", "_to_trt")


def result_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("model_id") or row.get("model") or ""),
        str(row.get("setup_id") or row.get("expected_setup_id") or ""),
        run_token(row.get("source_run_id") or row.get("run_id")),
        str(row.get("case_id") or row.get("case") or "full"),
        str(row.get("variant") or ""),
    )


def options_from_run(run_dir: Path) -> WorkflowOptions:
    manifest = read_json(run_dir / "run_manifest.json", default={}) or {}
    if manifest.get("schema") != "onnx-splitpoint/evaluation-run-manifest":
        raise ValueError("supplement_run_manifest_missing")
    archived = manifest.get("options")
    if not isinstance(archived, Mapping):
        raise ValueError("supplement_archived_options_missing")
    unknown = set(archived) - {f.name for f in fields(WorkflowOptions)}
    if unknown:
        raise ValueError("supplement_unknown_archived_options:" + ",".join(sorted(unknown)))
    payload = copy.deepcopy(dict(archived))
    payload.update(out=str(run_dir.parent), run_id=run_dir.name, resume=True,
                   resume_missing=True, require_fresh_run=False)
    if any(payload.get(k) for k in ("force_stage", "hailo_force_build", "rerun_generated_only",
                                   "resume_missing_full_quality_only", "only_model", "max_models")):
        raise ValueError("supplement_conflicting_archived_controls")
    return WorkflowOptions(**payload)


# These are outputs of the existing cache-preflight writers, not measurement
# inputs. Only the named probe bookkeeping fields may differ. Every other byte
# of the canonical JSON must still reproduce the ORIGINAL indexed digest.
_PROBE_METADATA_NAMES = frozenset({
    "build_backend_artifacts_request.json", "backend_artifact_decisions.json",
    "artifact_reuse_manifest.json", "hailo_artifact_service_plan.json",
    "hailo_artifact_status.json", "hailo_expected_unsupported.json",
    "hailo_build_attempts.json", "hailo_build_reuse_plan.json",
    "hailo_build_queue.json", "hailo_service_dispatch.json",
    "hailo_build_service_dispatch.json", "hailo_build_service_status.json",
    "legacy_suite/hailo_artifacts_manifest.json",
    "deepx/deepx_artifact_status.json", "deepx/deepx_build_queue.json",
    "deepx/deepx_output_contracts.json",
})


def _regular_evidence(root: Path, raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("supplement_unsafe_original_path:" + raw)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("supplement_unsafe_original_path:" + raw)
    if not path.is_file():
        raise ValueError("supplement_original_evidence_missing:" + raw)
    return path


def prepare_source_evidence(run_dir: Path) -> tuple[Path | None, dict]:
    """Use the first attempt's original index; never rebase on a failed probe."""
    manifest = read_json(run_dir / "run_manifest.json", default={}) or {}
    attempts = run_dir / "jobs/supplement_attempts"
    originals = {path.parent.name: path for path in attempts.glob("*/original")}
    for session in manifest.get("execution_sessions", []):
        expected = str(session.get("supplement_source_root") or "")
        if not expected and str(session.get("error_detail") or "").startswith("supplement_"):
            # The failed initial 2.91.1 attempt predates the session pointer.
            expected = "jobs/supplement_attempts/" + str(session.get("session_id") or "") + "/original"
        if expected:
            parts = Path(expected).parts
            if (len(parts) != 4 or parts[:2] != ("jobs", "supplement_attempts")
                    or parts[-1] != "original" or parts[2] in {".", ".."}):
                raise ValueError("supplement_original_attempt_path_invalid")
            _regular_evidence(run_dir, expected + "/artifact_index.json")
            original = run_dir / expected
            saved = read_json(_regular_evidence(run_dir, expected + "/run_manifest.json"), default={}) or {}
            if saved.get("run_id") != manifest.get("run_id"):
                raise ValueError("supplement_original_attempt_run_conflict")
            return original, read_json(original / "artifact_index.json", default={}) or {}
    if not originals:
        return None, read_json(run_dir / "artifact_index.json", default={}) or {}
    sessions = [str(row.get("session_id") or "")
                for row in manifest.get("execution_sessions", [])]
    ordered = [originals[session] for session in sessions if session in originals]
    if not ordered:
        # A crash during admission may precede publication of its session.
        # One such original is unambiguous; several require investigation.
        if len(originals) != 1:
            raise ValueError("supplement_original_attempt_ambiguous")
        ordered = list(originals.values())
    original = ordered[0]
    relative = original.relative_to(run_dir).as_posix()
    index_path = _regular_evidence(run_dir, relative + "/artifact_index.json")
    saved = read_json(_regular_evidence(run_dir, relative + "/run_manifest.json"), default={}) or {}
    if saved.get("run_id") != manifest.get("run_id"):
        raise ValueError("supplement_original_attempt_run_conflict")
    return original, read_json(index_path, default={}) or {}


def probe_metadata_binding(run_dir: Path, original_root: Path | None, raw: str,
                           descriptor: Mapping, profile_payload: Mapping) -> dict | None:
    """Prove a fresh probe against old bytes, including the 2.91.1 backup gap.

    Reconstruction is in memory only and succeeds solely on the existing size
    AND SHA256. It neither updates a receipt nor restores any live file.
    """
    parts = Path(raw).parts
    if (len(parts) < 4 or parts[0] != "models" or parts[2] != "benchmark_set"
            or "/".join(parts[3:]) not in _PROBE_METADATA_NAMES or original_root is None):
        return None
    name, model = "/".join(parts[3:]), parts[1]
    current = read_json(_regular_evidence(run_dir, raw), default=None)
    if not isinstance(current, dict) or not isinstance(current.get("created_at"), str):
        return None
    expected = str(descriptor.get("sha256") or "").removeprefix("sha256:")
    def matches(data):
        return (type(descriptor.get("size_bytes")) is int
                and len(data) == descriptor["size_bytes"]
                and hashlib.sha256(data).hexdigest() == expected)
    archived = original_root / raw
    proof_source = "preserved_original"
    if archived.exists() or archived.is_symlink():
        old_bytes = _regular_evidence(original_root, raw).read_bytes()
        if not matches(old_bytes):
            return None
        old = json.loads(old_bytes)
    else:
        # The first 2.91.1 snapshot omitted these four specific outputs. Their
        # writer timestamps survive in the adjacent original plan and stage.
        if name not in {"legacy_suite/hailo_artifacts_manifest.json",
                        "deepx/deepx_artifact_status.json", "deepx/deepx_build_queue.json",
                        "deepx/deepx_output_contracts.json"}:
            return None
        old = copy.deepcopy(current)
        stamp_path = (f"models/{model}/benchmark_set/hailo_artifact_service_plan.json"
                      if name in {"legacy_suite/hailo_artifacts_manifest.json", "deepx/deepx_artifact_status.json"}
                      else f"models/{model}/stages/build_backend_artifacts/stage_result.json")
        anchor = read_json(_regular_evidence(original_root, stamp_path), default={}) or {}
        old["created_at"] = anchor.get("created_at") if "service_plan" in stamp_path else anchor.get("finished_at")
        if name == "deepx/deepx_artifact_status.json":
            old["mode"] = (profile_payload.get("deepx_build") or {}).get("mode", "reuse_only")
        proof_source = "reconstructed_exact_original_digest"
    candidate = copy.deepcopy(current)
    candidate["created_at"] = old.get("created_at")
    fields = ["created_at"]
    if name == "hailo_artifact_service_plan.json":
        nested = candidate.get("source_build_backend_artifacts_request")
        old_nested = old.get("source_build_backend_artifacts_request")
        if not isinstance(nested, dict) or not isinstance(old_nested, dict):
            return None
        nested["created_at"] = old_nested.get("created_at")
        fields.append("source_build_backend_artifacts_request.created_at")
    if name == "deepx/deepx_artifact_status.json":
        if old.get("mode") != (profile_payload.get("deepx_build") or {}).get("mode", "reuse_only"):
            return None
        if current.get("mode") not in {old.get("mode"), "cache_verify_only"}:
            return None
        candidate["mode"] = old.get("mode")
        fields.append("mode:cache_verify_only")
    if not matches((stable_json_dumps(candidate) + "\n").encode("utf-8")):
        return None
    return {"path": raw, "original_sha256": descriptor.get("sha256"),
            "proof_source": proof_source, "probe_only_fields": fields,
            "original_root": str(original_root)}


def generation_runtime_binding(run_dir: Path, original_root: Path | None, raw: str,
                               descriptor: Mapping, source_index: Mapping,
                               current_index: Mapping) -> dict | None:
    """Verify downstream writer output without rebasing generation evidence.

    Runtime finalization projects the *original* physical scope into the plan
    aliases. Harness refresh installs the canonical suite template. Both are
    mutable dispatch inputs, also listed as outputs of the completed generator.
    Their existing current bindings do not replace the first attempt's source
    bindings: prove the precise writer transformation as well as those bindings.
    """
    parts = Path(raw).parts
    if (original_root is None or len(parts) not in {4, 5}
            or parts[0] != "models" or parts[2] != "benchmark_set"):
        return None
    model, name = parts[1], "/".join(parts[3:])
    if descriptor.get("model_id") != model:
        return None

    def bound_bytes(root, logical, binding):
        data = _regular_evidence(root, logical).read_bytes()
        if (binding.get("path") != logical
                or type(binding.get("size_bytes")) is not int
                or len(data) != binding["size_bytes"]
                or hashlib.sha256(data).hexdigest()
                != str(binding.get("sha256") or "").removeprefix("sha256:")):
            raise ValueError("supplement_generation_binding_changed:" + logical)
        return data

    current_binding = current_index.get(raw, {})
    if current_binding.get("model_id") != model:
        return None
    current_bytes = bound_bytes(run_dir, raw, current_binding)
    proof = {"path": raw, "original_sha256": descriptor.get("sha256"),
             "current_sha256": current_binding.get("sha256"),
             "original_root": str(original_root),
             "original_measurement_source_preserved": True}
    if name == "legacy_suite/benchmark_suite.py":
        if descriptor.get("producer_stage") != "generate_benchmark_set":
            return None
        # Early supplement snapshots omitted this generated script. Its old
        # index binding remains provenance, never a claim about the new bytes.
        archived = original_root / raw
        if archived.exists() or archived.is_symlink():
            bound_bytes(original_root, raw, descriptor)
        from ..gui.controller import load_template_text
        expected = load_template_text("benchmark_suite.py.txt").replace(
            "__BENCH_JSON__", "benchmark_set.json").encode("utf-8")
        if current_bytes != expected:
            return None
        return {**proof, "proof_source": "index_bound_canonical_harness_refresh",
                "original_script_archived": archived.is_file()}

    # These are the exact aliases written by finalize_suite_for_runtime, not
    # filename exemptions. Compare every field after the existing projection.
    if name not in {"benchmark_set.json", "benchmark_plan.json",
                    "legacy_suite/benchmark_set.json", "legacy_suite/benchmark_plan.json"}:
        return None
    if (descriptor.get("producer_stage") not in {"generate_benchmark_set", "run_benchmarks"}
            or current_binding.get("producer_stage") != "run_benchmarks"):
        return None
    old = json.loads(bound_bytes(original_root, raw, descriptor))
    scope_path = "required_run_scope.json"
    scope_binding = source_index.get(scope_path, {})
    old_scope_bytes = bound_bytes(original_root, scope_path, scope_binding)
    bound_bytes(run_dir, scope_path, scope_binding)
    from .required_run_scope import authoritative_run_descriptors, merge_authoritative_run_descriptors
    from .artifacts import sha256_json
    descriptors = authoritative_run_descriptors(json.loads(old_scope_bytes), model_id=model)
    if not descriptors:
        return None

    def project(payload):
        projected = copy.deepcopy(payload)
        rows = merge_authoritative_run_descriptors(payload, descriptors)["runs"]
        for key in ("runs", "planned_runs"):
            if key in projected:
                projected[key] = rows
        invariant = projected.get("management_cpu_reference_invariant")
        if isinstance(invariant, dict):
            invariant["run_plan_sha256"] = sha256_json(rows)
        if isinstance(projected.get("plan"), dict):
            projected["plan"] = project(projected["plan"])
        return projected

    if (stable_json_dumps(project(old)) + "\n").encode("utf-8") != current_bytes:
        return None
    return {**proof, "proof_source": "original_scope_runtime_projection",
            "scope_source": scope_path, "scope_sha256": scope_binding.get("sha256")}


def _cohort(run_dir: Path, models: list[str]) -> tuple[list[dict], dict]:
    rows, suites = [], {}
    for model in models:
        suite_dir = run_dir / "models" / model / "benchmark_set" / "legacy_suite"
        suite = read_json(suite_dir / "benchmark_set.json", default={}) or {}
        if not isinstance(suite.get("cases"), list) or not suite["cases"]:
            raise ValueError("supplement_original_suite_missing:" + model)
        suites[model] = suite
        seen = set()
        for case in suite["cases"]:
            case_id = str(case.get("folder") or case.get("case_dir") or "")
            if not case_id or case_id in seen or Path(case_id).name != case_id:
                raise ValueError("supplement_cohort_ambiguous:" + model)
            seen.add(case_id)
            path = suite_dir / case_id / "split_manifest.json"
            manifest = read_json(path, default={}) or {}
            inputs = (manifest.get("io") or {}).get("part2_external_inputs")
            if not isinstance(inputs, list) or not inputs:
                raise ValueError("supplement_part2_inputs_missing:" + str(path))
            rows.append({"model_id": model, "case_id": case_id,
                         "part2_external_inputs": inputs,
                         "native_supported": len(inputs) == 1,
                         "backend_exclusions": list((manifest.get("hailo") or {}).get("backend_exclusions") or []),
                         "source_manifest": str(path.relative_to(run_dir))})
    return rows, suites


def _stored_compile_failure(suite: Mapping, row: Mapping) -> dict | None:
    if row.get("variant") != "split":
        return None
    backend = {"hailo8_to_trt": "hailo8", "hailo10_to_trt": "hailo10"}.get(run_token(row.get("run_id")))
    if not backend:
        return None
    for case in suite.get("cases", []):
        if str(case.get("folder") or case.get("case_dir")) != row.get("case_id"):
            continue
        evidence = (case.get("hailo_compile") or {}).get(backend, {}).get("part1", {})
        error = str(evidence.get("error") or "")
        deterministic = (evidence.get("mapping_failed") is True
                         or error.startswith("Parsing failed. The errors found in the graph are:")
                         or error.startswith("Compilation failed: No successful assignments:")
                         or (error.startswith("feature splitter layer ")
                             and "successor name is missing, output shape is ambiguous" in error))
        if (evidence.get("ok") is False and deterministic
                and not any(evidence.get(k) for k in ("timed_out", "watchdog_expired", "cancelled"))):
            # Preserve the recorded outcome without reclassifying a local
            # exporter/parser problem as scientific compiler infeasibility.
            return {k: evidence.get(k) for k in ("error", "build_receipt_path", "source_onnx_path",
                    "compiler_onnx_path", "timed_out", "watchdog_expired")}
    return None


def completed_quality(run_dir: Path, *, verify_requests: bool = True) -> tuple[list[dict], dict]:
    summary = read_json(run_dir / "quality_management" / "central_quality_summary.json", default={}) or {}
    kept, requests = [], {}
    # A request publishes its own result before the aggregate quality stage.
    # Recover those exact completed requests after an interruption as well.
    roots = list(run_dir.glob("models/*/benchmark_results/quality_inputs"))
    roots += list(run_dir.glob("models/*/benchmark_results/attempts/*/quality_inputs"))
    def records():
        yield from ((row, None) for row in summary.get("results", []))
        for root in sorted(roots):
            for local in sorted(root.rglob("task_quality_inputs/*_central_result.json")):
                relative = local.relative_to(run_dir)
                if any(p.is_symlink() for p in [local, *local.parents[:len(relative.parts)-1]]):
                    raise ValueError("supplement_quality_result_symlink")
                row = read_json(local, default={}) or {}
                if not isinstance(row, Mapping):
                    raise ValueError("supplement_quality_result_invalid:" + str(relative))
                yield row, local
    for row, local in records():
        if row.get("status") != "completed" or row.get("technical_status") != "completed":
            continue
        raw = str(row.get("source_request") or "")
        relative = Path(raw)
        if not raw or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("supplement_quality_request_path_invalid")
        path = run_dir / relative
        if any(p.is_symlink() for p in [path, *path.parents[:len(relative.parts)-1]]):
            raise ValueError("supplement_quality_request_symlink")
        if local is not None and local != path.with_name(
            path.stem.replace("_request", "") + "_central_result.json"
        ):
            # Local-only completion publishes into the new attempt while
            # retaining the original request and its failed result unchanged.
            source_parts = relative.parts
            local_parts = local.relative_to(run_dir).parts
            try:
                source_tail = source_parts[source_parts.index("quality_inputs") + 1:]
                expected_tail = (*source_tail[:-1], path.stem.replace("_request", "") + "_central_result.json")
                valid_attempt = (
                    local_parts[:4] == ("models", source_parts[1], "benchmark_results", "attempts")
                    and local_parts[5] == "quality_inputs"
                    and local_parts[6:] == expected_tail
                )
            except (ValueError, IndexError):
                valid_attempt = False
            if not valid_attempt:
                raise ValueError("supplement_quality_result_request_mismatch:" + raw)
        expected = str(row.get("source_request_sha256") or "").removeprefix("sha256:")
        if not expected or not path.is_file():
            raise ValueError("supplement_completed_request_missing:" + raw)
        if verify_requests and str(sha256_file(path)).removeprefix("sha256:") != expected:
            raise ValueError("supplement_completed_request_drift:" + raw)
        request = read_json(path, default={}) or {}
        candidate = request.get("candidate") or {}
        candidate_name = str(candidate.get("path") or "")
        candidate_path = path.parent / candidate_name
        if (not candidate_name or Path(candidate_name).name != candidate_name
                or candidate_path.is_symlink() or not candidate_path.is_file()
                or candidate_path.stat().st_size != candidate.get("size_bytes")):
            raise ValueError("supplement_completed_predictions_missing:" + raw)
        # candidate_predictions_sha256 describes normalized evaluator records,
        # not the serialized candidate file. Its file descriptor is already
        # bound by the verified original source request above.
        nested = row.get("request_identity") or {}
        nested_sha = str(nested.get("source_request_sha256") or "").removeprefix("sha256:")
        if nested_sha and nested_sha != expected:
            raise ValueError("supplement_completed_request_identity_conflict:" + raw)
        for field in ("model_id", "case_id", "variant"):
            if request.get(field) and str(row.get(field) or "") != str(request[field]):
                raise ValueError("supplement_completed_request_identity_conflict:" + raw)
            if nested.get(field) and str(row.get(field) or "") != str(nested[field]):
                raise ValueError("supplement_completed_request_identity_conflict:" + raw)
        setup = str(row.get("setup_id") or row.get("source_setup_id") or nested.get("setup_id") or "")
        if any(str(value) != setup for value in (
            row.get("setup_id"), row.get("source_setup_id"), nested.get("setup_id"), request.get("setup_id")
        ) if value):
            raise ValueError("supplement_completed_request_identity_conflict:" + raw)
        # Existing request/result identities remain authoritative; do not hash
        # all archived prediction payloads again or rewrite their provenance.
        key = str(path.resolve())
        if key in requests:
            if local is None:
                raise ValueError("supplement_duplicate_completed_request:" + raw)
            previous = requests[key]
            def identity(result):
                nested = result.get("request_identity") or {}
                return tuple(str(result.get(field) or nested.get(field) or "")
                             for field in ("model_id", "case_id", "variant", "source_run_id", "decision")) + (
                                 str(result.get("setup_id") or result.get("source_setup_id") or nested.get("setup_id") or ""),
                             )
            if (previous["source_request_sha256"] != expected
                    or identity(previous["result"]) != identity(row)):
                raise ValueError("supplement_completed_request_identity_conflict:" + raw)
            continue
        requests[key] = {"source_request_sha256": expected, "result": row}
        kept.append(row)
    return kept, requests


def _completed_normalized_rows(run_dir: Path, model: str, identities: list[dict]) -> list[dict]:
    """Reuse exact normalized observations even before the final reporter ran."""
    from .logical_measurement import select_logical_primary_rows

    payload = read_json(run_dir / "models" / model / "benchmark_results/normalized_results.json", default={}) or {}
    if not payload:
        return []
    if (payload.get("schema") != "onnx-splitpoint/normalized-benchmark-results"
            or payload.get("model_id") != model
            or payload.get("evaluation_run_id") != run_dir.name):
        raise ValueError("supplement_normalized_results_identity_conflict:" + model)
    required = {result_key(row): row for row in identities}
    completed = []
    for row in payload.get("results", []):
        if row.get("required_scope_binding_status") not in {"exact_physical_match", "exact_terminal_failure"}:
            continue
        expected = required.get(result_key(row))
        if (not expected or not row.get("logical_identity_sha256")
                or row["logical_identity_sha256"] != expected.get("logical_identity_sha256")):
            raise ValueError("supplement_normalized_results_scope_conflict:" + model)
        if not row.get("source_row_sha256") or not row.get("logical_measurement_id"):
            raise ValueError("supplement_normalized_results_proof_missing:" + model)
        source = Path(str(row.get("source_path") or ""))
        if not source.is_absolute():
            source = run_dir / source
        try:
            relative = source.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError("supplement_normalized_results_source_external:" + model) from exc
        if (".." in relative.parts or not source.is_file()
                or any(p.is_symlink() for p in [source, *source.parents[:len(relative.parts)-1]])):
            raise ValueError("supplement_normalized_results_source_missing_or_unsafe:" + model)
        completed.append(row)
    primaries, errors = select_logical_primary_rows(completed)
    if errors or len({result_key(row) for row in primaries}) != len(primaries):
        raise ValueError("supplement_normalized_results_ambiguous:" + model)
    return primaries


def _verify_original_binding(path: Path, raw: str, source_index: Mapping | None) -> None:
    if source_index is None:
        return
    descriptor = source_index.get(raw)
    if not isinstance(descriptor, Mapping):
        raise ValueError("supplement_original_binding_missing:" + raw)
    expected = str(descriptor.get("sha256") or "").removeprefix("sha256:")
    if (type(descriptor.get("size_bytes")) is not int
            or path.stat().st_size != descriptor["size_bytes"]
            or not expected or sha256_file(path).removeprefix("sha256:") != expected):
        raise ValueError("supplement_original_evidence_changed:" + raw)


def _preserved_native_measurements(run_dir: Path, original: Path, old_matrix: Mapping,
                                   source_index: Mapping | None = None) -> None:
    """Keep the actual Native dispatcher authority, not just its matrix count."""
    from ..native_job_identity import complete_historical_identity, native_identity_key

    raw = "reports/native_producer_summary.json"
    contexts = list(old_matrix.get("present_expected_rows") or []) + list(old_matrix.get("missing_expected_rows") or [])
    if not (original / raw).exists() and not any(row.get("actual_ok") is True for row in contexts):
        return
    old_path = _regular_evidence(original, raw)
    current_path = _regular_evidence(run_dir, raw)
    unchanged = (old_path.stat().st_size == current_path.stat().st_size
                 and sha256_file(old_path) == sha256_file(current_path))
    # These are existing source/command and measured-value fields. Quality,
    # claim and validation projections may legitimately gain new annotations.
    fields = (
        "precision", "report", "source_reports", "source_root", "source_roots",
        "native_command_contract", "full_command_contract", "native_command_contract_sha256",
        "full_command_contract_sha256", "workload_contract_sha256", "input_image_sha256",
        "output_manifest_sha256", "native_split_semantic_output_manifest_sha256",
        "native_split_semantic_boundary_manifest_sha256", "fps_makespan", "paper_fps",
        "frames", "warmup", "handoff_ms", "p1_ms", "p2_run_ms", "p1_thread_ms", "p2_thread_ms",
        "fps_repetition_samples", "latency_mean_repetition_samples_ms",
        "repetition_count_requested", "repetition_count_attempted", "repetition_count_valid",
        "repetition_records", "completed_frames", "completed_work_units",
    )

    checked_sources = set()

    def project(path: Path, *, check_sources: bool) -> dict:
        payload = read_json(path, default=None)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("rows"), list):
            raise ValueError("supplement_original_native_summary_invalid:" + str(path))
        result = {}
        for raw_row in payload["rows"]:
            if raw_row.get("ok", raw_row.get("result_ok")) is not True:
                continue
            row = complete_historical_identity(raw_row, contexts)
            key = native_identity_key(row)
            if key in result or row.get("identity_conflicts"):
                raise ValueError("supplement_original_native_identity_conflict:" + str(key))
            result[key] = {field: row[field] for field in fields if field in row}
            if check_sources:
                sources = set(row.get("source_reports") or [])
                if row.get("report"):
                    sources.add(row["report"])
                if str(row.get("backend") or "").startswith("native_full_"):
                    # _rows_from_native_full reads this collected source;
                    # its "report" field can name remote engine-cache
                    # metadata which was never a local measurement payload.
                    source_root = Path(str(row.get("source_root") or ""))
                    if not source_root.is_absolute():
                        raise ValueError("supplement_original_native_source_missing:" + str(key))
                    sources = {str(source_root / "analysis_tables/native_full_baseline_eval.json")}
                if not sources:
                    raise ValueError("supplement_original_native_source_missing:" + str(key))
                for source in sources:
                    relative = Path(str(source))
                    if relative.is_absolute():
                        try:
                            relative = relative.relative_to(run_dir)
                        except ValueError as exc:
                            # Legacy full rows retain their remote report
                            # label. The frozen command root and recorded
                            # local source root give the existing transfer
                            # mapping; neither filenames nor vendor guesses
                            # establish this authority.
                            candidates = set()
                            local_root = Path(str(row.get("source_root") or ""))
                            for field in ("native_command_contract", "full_command_contract"):
                                contract = row.get(field) or {}
                                remote_root = Path(str(contract.get("root") or ""))
                                if not remote_root.is_absolute() or not local_root.is_absolute():
                                    continue
                                try:
                                    mapped = local_root / relative.relative_to(remote_root)
                                    candidates.add(mapped.relative_to(run_dir))
                                except ValueError:
                                    continue
                            if len(candidates) != 1:
                                raise ValueError("supplement_original_native_source_external:" + str(source)) from exc
                            relative = candidates.pop()
                    logical = relative.as_posix()
                    if logical not in checked_sources:
                        source_path = _regular_evidence(run_dir, logical)
                        _verify_original_binding(source_path, logical, source_index)
                        checked_sources.add(logical)
        return result

    # Only one large summary is decoded on the usual unchanged path. If new
    # results were published, release the old parse before reading the current
    # summary; retain only the compact measured/source projection between them.
    old_rows = project(old_path, check_sources=True)
    if unchanged:
        return
    current_rows = project(current_path, check_sources=False)
    for key, old_row in old_rows.items():
        current_row = current_rows.get(key)
        if current_row is None or any(current_row.get(field) != value for field, value in old_row.items()):
            raise ValueError("supplement_original_native_measurement_missing_or_changed:" + str(key))


def _preserved_inventory(run_dir: Path, original: Path, performance: list,
                         quality: list, matrix: Mapping,
                         source_index: Mapping | None = None) -> None:
    """An interrupted projection must not turn old completed work into tasks."""
    normalized_sources = list(original.glob("models/*/benchmark_results/normalized_results.json"))
    if source_index is not None:
        unpublished_native = frozenset()
        native_baseline = ("reports/native_expected_matrix.json", "reports/native_producer_summary.json")
        if any(not (original / raw).is_file() for raw in native_baseline):
            from .yolo_augmentation_reentry import unpublished_native_baseline
            unpublished_native = unpublished_native_baseline(run_dir, original, source_index)
        baseline_paths = ["reports/scientific/performance_results.json",
                          "quality_management/central_quality_summary.json",
                          "reports/native_expected_matrix.json", "reports/native_producer_summary.json"]
        baseline_paths += [path.relative_to(original).as_posix() for path in normalized_sources]
        for raw in baseline_paths:
            if raw in unpublished_native:
                continue
            path = _regular_evidence(original, raw)
            _verify_original_binding(path, raw, source_index)
    old_performance = read_json(original / "reports/scientific/performance_results.json", default=[]) or []
    current = {result_key(row): row for row in performance}
    for row in old_performance:
        if current.get(result_key(row)) != row:
            raise ValueError("supplement_original_measurement_missing_or_changed:" + str(result_key(row)))
    old_quality = read_json(original / "quality_management/central_quality_summary.json", default={}) or {}
    by_request = {row.get("source_request"): row for row in quality}
    for row in old_quality.get("results", []):
        if row.get("status") == "completed" and row.get("technical_status") == "completed":
            if by_request.get(row.get("source_request")) != row:
                raise ValueError("supplement_original_quality_missing_or_changed:" + str(row.get("source_request")))
    def native_key(row):
        return tuple(str(row.get(key) or "") for key in
                     ("model", "setup_id", "backend", "execution_mode", "case", "precision"))
    old_matrix = read_json(original / "reports/native_expected_matrix.json", default={}) or {}
    _preserved_native_measurements(run_dir, original, old_matrix, source_index)
    native_current = {native_key(row): row for row in
                      list(matrix.get("present_expected_rows") or []) + list(matrix.get("missing_expected_rows") or [])}
    for row in old_matrix.get("present_expected_rows", []):
        if row.get("actual_ok") is True and native_current.get(native_key(row), {}).get("actual_ok") is not True:
            raise ValueError("supplement_original_native_missing_or_changed:" + str(native_key(row)))
    # Raw Generic result files are required even if an old aggregate survives.
    # Check their presence using the preserved normalized provenance; no tree
    # scan or prediction-payload hashing is added to the admission path.
    checked_sources = set()
    for source in normalized_sources:
        payload = read_json(source, default={}) or {}
        for row in payload.get("results", []):
            raw = str(row.get("source_path") or "")
            if not raw:
                continue
            path = Path(raw)
            if path.is_absolute():
                try:
                    raw = path.relative_to(run_dir).as_posix()
                except ValueError as exc:
                    raise ValueError("supplement_original_measurement_external:" + raw) from exc
            if raw not in checked_sources:
                path = _regular_evidence(run_dir, raw)
                _verify_original_binding(path, raw, source_index)
                checked_sources.add(raw)


def build_supplement_plan(run_dir: Path, *, verify_requests: bool = True,
                          original_root: Path | None = None,
                          source_index: Mapping | None = None) -> dict:
    run_dir = Path(run_dir).resolve()
    plan = read_json(run_dir / "effective_execution_plan.json", default={}) or {}
    models = list(plan.get("models") or [])
    if not models:
        raise ValueError("supplement_original_plan_missing")
    cohorts, suites = _cohort(run_dir, models)
    performance = read_json(run_dir / "reports/scientific/performance_results.json", default=[]) or []
    present = {result_key(row): row for row in performance}
    if len(present) != len(performance):
        raise ValueError("supplement_duplicate_generic_identity")
    generic = []
    cohort_by_key = {(r["model_id"], r["case_id"]): r for r in cohorts}
    for model in models:
        source = f"models/{model}/benchmark_set/required_run_scope.json"
        scope = read_json(run_dir / source, default={}) or {}
        identities = scope.get("identities") or []
        if not identities:
            raise ValueError("supplement_required_scope_missing:" + model)
        if any(result_key(row) not in present for row in identities):
            for row in _completed_normalized_rows(run_dir, model, identities):
                present.setdefault(result_key(row), row)
        for identity in identities:
            row = {**identity, "setup_id": identity.get("expected_setup_id"), "source_scope": source}
            negative = _stored_compile_failure(suites[model], row)
            backend = {"hailo8_to_trt": "hailo8", "hailo10_to_trt": "hailo10"}.get(run_token(row.get("run_id")))
            cohort = cohort_by_key.get((model, row.get("case_id")), {})
            policy = next((item for item in cohort.get("backend_exclusions", [])
                           if row.get("variant") == "split" and backend
                           and item.get("hw_arch") == backend
                           and item.get("folder") == row.get("case_id")
                           and item.get("stage") == "part1" and item.get("status") == "rejected"
                           and item.get("reason") == "hailo_yolo26_static_part1_guard"), None)
            if result_key(row) in present:
                row["action"] = "preserve"
            elif policy:
                row.update(action="policy_excluded", reason=policy["reason"],
                           evidence=policy, source_manifest=cohort["source_manifest"])
            elif negative:
                row.update(action="negative_completed", reason="stored_compile_failure", evidence=negative)
            else:
                row["action"] = "execute"
            generic.append(row)
    required_keys = {result_key(row) for row in generic}
    if set(present) - required_keys:
        raise ValueError("supplement_generic_outside_original_scope")
    quality, requests = completed_quality(run_dir, verify_requests=verify_requests)
    # Failed statistics with retained predictions are local work. They must not
    # become a fresh device request merely because their decision is unfinished.
    summary = read_json(run_dir / "quality_management/central_quality_summary.json", default={}) or {}
    pending_quality = []
    for result in summary.get("results", []):
        raw = str(result.get("source_request") or "")
        if str((run_dir / raw).resolve()) in requests:
            continue
        path = _regular_evidence(run_dir, raw)
        expected = str(result.get("source_request_sha256") or "").removeprefix("sha256:")
        if not expected or (verify_requests and sha256_file(path).removeprefix("sha256:") != expected):
            raise ValueError("supplement_pending_request_drift:" + raw)
        request = read_json(path, default={}) or {}
        candidate = request.get("candidate") or {}
        name = str(candidate.get("path") or "")
        if (not name or Path(name).name != name or (path.parent / name).is_symlink()
                or not (path.parent / name).is_file()
                or (path.parent / name).stat().st_size != candidate.get("size_bytes")):
            raise ValueError("supplement_pending_predictions_missing:" + raw)
        pending_quality.append({**result, "setup_id": result.get("setup_id") or result.get("source_setup_id")
                                or (result.get("request_identity") or {}).get("setup_id") or request.get("setup_id"),
                                "action": "local_statistics", "source_request": raw})
    quality_companions = []
    for model in models:
        for identity in plan.get("setup_local_tensorrt_quality_companion_identities", []):
            row = {**identity, "model_id": model, "case_id": "full", "stage": "quality_companion"}
            found = [q for q in quality if result_key(q) == result_key(row)]
            local = [q for q in pending_quality if result_key(q) == result_key(row)]
            row["action"] = "preserve" if found else "local_statistics" if local else "execute"
            quality_companions.append(row)
    cohort_by_key = {(r["model_id"], r["case_id"]): r for r in cohorts}
    matrix = read_json(run_dir / "reports/native_expected_matrix.json", default={}) or {}
    if original_root is not None:
        _preserved_inventory(run_dir, original_root, performance, quality, matrix, source_index)
    native = []
    for original in list(matrix.get("present_expected_rows") or []) + list(matrix.get("missing_expected_rows") or []):
        row = dict(original)
        cohort = cohort_by_key.get((row.get("model"), row.get("case")))
        if row.get("execution_mode") == "native_split" and not cohort:
            raise ValueError("supplement_native_case_outside_cohort")
        if cohort and not cohort["native_supported"]:
            row.update(action="not_supported", reason="multiple_actual_part2_inputs",
                       source_manifest=cohort["source_manifest"], part2_external_inputs=cohort["part2_external_inputs"])
        elif row.get("actual_ok") is True:
            row["action"] = "preserve_and_validate_locally"
        else:
            row["action"] = "execute"
        native.append(row)
    from .status_reporting import energy_attempt_failure
    return {"schema": "onnx-splitpoint/evaluation-supplement-plan", "schema_version": 1,
            "run_dir": str(run_dir), "run_id": run_dir.name,
            "source_plan": "effective_execution_plan.json", "cohort": cohorts,
            "generic_rows": generic, "quality_companion_rows": quality_companions,
            "pending_quality_rows": pending_quality,
            "native_rows": native, "completed_quality_count": len(quality),
            "completed_quality_requests": [{"path": str(Path(k).relative_to(run_dir)),
                "source_request_sha256": v["source_request_sha256"]} for k,v in requests.items()],
            "counts": {"generic": dict(Counter(r["action"] for r in generic)),
                       "quality_companions": dict(Counter(r["action"] for r in quality_companions)),
                       "native": dict(Counter(r["action"] for r in native)),
                       "cohort": len(cohorts), "completed_quality": len(quality)},
            "measurement_started": False,
            "previous_energy_attempt_failure": energy_attempt_failure(run_dir),
            "energy": {"action": "replan_then_resume_existing_rows_and_replicates",
                       "maximum_supported_rows": sum(r["action"] != "not_supported" for r in native)}}


def preserve_metadata(run_dir: Path, session_id: str) -> Path:
    """Copy only mutable plans/projections, never models, outputs or caches."""
    out = run_dir / "jobs" / "supplement_attempts" / session_id / "original"
    out.mkdir(parents=True, exist_ok=False)
    paths = set(run_dir.glob("*.json")) | set(run_dir.glob("*.md"))
    for pattern in ("stages/**/*.json", "models/*/stages/**/*.json", "models/*/benchmark_results/*.json",
                    "models/*/benchmark_set/*.json", "models/*/benchmark_set/legacy_suite/benchmark*.json",
                    "models/*/benchmark_set/legacy_suite/benchmark_suite.py",
                    "models/*/benchmark_set/legacy_suite/hailo_artifacts_manifest.json",
                    "models/*/benchmark_set/legacy_suite/output_contracts.json",
                    "models/*/benchmark_set/deepx/*.json", "models/*/full_baselines/*.json",
                    "reports/*.json", "reports/*.csv", "reports/*.md", "reports/scientific/*",
                    "reports/native_energy_measurements/*.json", "reports/native_energy_measurements/plan/*.json",
                    "reports/native_validation/*.json", "reports/native_validation/*.md", "quality_management/*.json"):
        paths.update(run_dir.glob(pattern))
    from .yolo_augmentation_reentry import augmentation_reentry_snapshot_paths
    paths.update(augmentation_reentry_snapshot_paths(run_dir))
    for source in sorted(paths):
        if not source.is_file() or source.is_symlink():
            continue
        destination = out / source.relative_to(run_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return out


def merge_preserved_results(previous: list[dict], new: list[dict]) -> list[dict]:
    """Retain original measured rows verbatim, merge newly completed identities."""
    by_key = {result_key(r): r for r in previous}
    for row in new:
        key = result_key(row)
        if key not in by_key:
            by_key[key] = row
    return list(by_key.values())


def write_plan(plan: Mapping, out: Path) -> tuple[Path, Path]:
    json_path = write_json(out / "supplement_plan.json", plan)
    lines = ["# Ergänzungsplan", "", f"Run: `{plan['run_dir']}`", "",
             "Originalkohorte unverändert; bestehende Messungen und positive wie negative Qualityresultate bleiben erhalten.",
             "", "## Bilanz nach Modell und Stufe", "", "| Modell | Stufe | Aktion | Anzahl |", "|---|---|---|---:|"]
    for stage, field in (("Generic", "generic_rows"), ("Full-Quality-Companion", "quality_companion_rows"), ("Native", "native_rows")):
        counts = Counter((r.get("model_id") or r.get("model"), r["action"]) for r in plan[field])
        lines += [f"| {model} | {stage} | {action} | {n} |" for (model, action), n in sorted(counts.items())]
    lines += ["", f"Abgeschlossene zentrale Qualityresultate: {plan['completed_quality_count']} (kein neuer Bootstrapauftrag).",
              "", "## Noch auf Hardware auszuführen", "", "| Modell | Setup | Stufe | Run/Backend | Fall |", "|---|---|---|---|---|"]
    for stage, field in (("Generic", "generic_rows"), ("Full-Quality-Companion", "quality_companion_rows"), ("Native", "native_rows")):
        for r in plan[field]:
            if r["action"] == "execute":
                lines.append(f"| {r.get('model_id') or r.get('model')} | {r.get('setup_id')} | {stage} | {r.get('run_id') or r.get('backend')} | {r.get('case_id') or r.get('case')} |")
    lines += ["", "## Erhaltene und nicht erneut auszuführende Fälle", "",
              "`preserve`: behalten; `preserve_and_validate_locally`: Messung behalten, lokal validieren;",
              "`negative_completed`: gespeicherten negativen Buildbefund behalten; `policy_excluded`: statischer Part1-Ausschluss (Part2-only bleibt separat); `not_supported`: für Native nicht anwendbar.",
              "", "| Modell | Setup | Stufe | Run/Backend | Variante | Aktion | Fälle |", "|---|---|---|---|---|---|---|"]
    for stage, field in (("Generic", "generic_rows"), ("Full-Quality-Companion", "quality_companion_rows"), ("Native", "native_rows")):
        groups = {}
        for r in plan[field]:
            if r["action"] == "execute":
                continue
            key = tuple(str(value or "") for value in (
                r.get("model_id") or r.get("model"), r.get("setup_id"), stage,
                r.get("run_id") or r.get("backend"), r.get("variant") or r.get("execution_mode"), r["action"]))
            groups.setdefault(key, []).append(str(r.get("case_id") or r.get("case") or "full"))
        for key, cases in sorted(groups.items()):
            lines.append("| " + " | ".join((*key, ", ".join(sorted(cases)))) + " |")
    lines += ["", f"Offene lokale Statistikrequests mit gespeicherten Inputs: {len(plan.get('pending_quality_rows', []))}.",
              "", str((plan.get("previous_energy_attempt_failure") or {}).get("energy_not_started_reason") or ""),
              "", "Energie: korrekt neu planen; bestehende Zeilen/Replikate und Budget fortsetzen. Keine Aufnahme durch diesen Plan.",
              "", "Alle erhaltenen und ausgeschlossenen Einzelidentitäten mit Originalquellen stehen in supplement_plan.json."]
    md = out / "supplement_plan.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="Read-only local admission and inventory; no hardware or compiler calls")
    parser.add_argument("--output-dir", type=Path, help="Required for --plan-only; must be outside the original run")
    parser.add_argument("--import-only", action="store_true", help="Project a saved energy parent result without any new capture")
    parser.add_argument("--resume", action="store_true", help="Continue only missing repeats of a saved energy parent")
    parser.add_argument("--recovery-authorization", type=Path, help="Existing explicit normal resource recovery authorization")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    from .energy_parent_flow import is_energy_parent, run_parent
    if is_energy_parent(run_dir):
        return run_parent(run_dir,plan_only=args.plan_only,output_dir=args.output_dir,
            import_only=args.import_only,resume=args.resume,recovery_authorization=args.recovery_authorization)
    if args.import_only or args.resume or args.recovery_authorization:
        parser.error("Energy-parent controls require an existing prepared energy parent")
    from .runner import EvaluationWorkflowRunner
    runner = EvaluationWorkflowRunner(
        options_from_run(run_dir), log=lambda message: print(message, flush=True),
    )
    runner.log("[supplement] validating archived run and preserved identities")
    runner._load_profile()
    # Read-only admission must work even when the archive is mounted read-only.
    # The actual run() performs the normal write/ownership admission itself.
    runner.run_dir, runner.run_id = run_dir, run_dir.name
    runner.manifest_path = run_dir / "run_manifest.json"
    runner.artifact_index_path = run_dir / "artifact_index.json"
    runner._resume_contract = runner._build_current_resume_contract()
    runner._validate_resume_target_paths()
    runner._validate_resume_profile_snapshot()
    runner._validate_resume_contract()
    if args.plan_only:
        if args.output_dir is None or args.output_dir.resolve().is_relative_to(run_dir):
            parser.error("--plan-only requires --output-dir outside the original run")
        original_root, source_index = prepare_source_evidence(run_dir)
        plan = build_supplement_plan(run_dir, original_root=original_root,
            source_index={str(row.get("path") or ""): row
                          for row in source_index.get("artifacts", []) if isinstance(row, Mapping)})
        paths = write_plan(plan, args.output_dir.resolve())
        if plan.get("previous_energy_attempt_failure"):
            print(plan["previous_energy_attempt_failure"]["energy_not_started_reason"], flush=True)
        print(json.dumps({"counts": plan["counts"], "outputs": [str(p) for p in paths]}, indent=2))
        return 0
    if args.output_dir:
        parser.error("--output-dir is only valid with --plan-only")
    result = runner.run()
    print(json.dumps({"status": result.status, "run_dir": result.run_dir}, indent=2))
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
