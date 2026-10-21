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
    diagnostics = _quality_technical_diagnostic_bindings(run_dir)
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
        if _quality_result_is_technical_diagnostic(run_dir, row, request, diagnostics):
            continue  # Historical completed sidecars remain unchanged, outside canonical Quality.
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


# These helpers are inserted into the existing workflow/supplement.py module.
# They verify one observed software defect; they do not run or import results.
_BAD_HAILO_INTERFACE_BODY = "8384c0f2a56a8179530fbbdf60841b7b1739358ce802cc409b48564460f6bf74"
_FIXED_HAILO_INTERFACE_BODY = "87738f67b922a88efe22bf0001e92eb385c0f069fe9e365343478f0e5af32f6a"
_TECHNICAL_REVALIDATION = "hailo_uint8_float_interface_v1"


def _technical_require(ok, reason):
    if not ok:
        raise ValueError("supplement_technical_revalidation_" + reason)


def _technical_sha(value):
    return hashlib.sha256(value).hexdigest()


def _technical_body(value):
    return _technical_sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode())


def _technical_file(run_dir, raw, descriptor=None):
    path = _regular_evidence(run_dir, str(raw))
    data = path.read_bytes()
    binding = {"path": str(raw), "sha256": _technical_sha(data), "size_bytes": len(data)}
    if descriptor is not None:
        _technical_require(type(descriptor.get("size_bytes")) is int
                           and descriptor["size_bytes"] == len(data)
                           and str(descriptor.get("sha256") or "").removeprefix("sha256:") == binding["sha256"],
                           "indexed_bytes_changed:" + str(raw))
    return binding, data


def _technical_index(value):
    result = {}
    for row in value.get("artifacts", []):
        _technical_require(isinstance(row, Mapping) and row.get("path") not in result, "ambiguous_index")
        result[row["path"]] = row
    return result


def _technical_callsite(source):
    import ast
    bodies = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Try):
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        calls = [part for part in ast.walk(body) if isinstance(part, ast.Call)]
        if not any(isinstance(call.func, ast.Name)
                   and call.func.id == "_compare_tensor_maps_by_shape_or_name" for call in calls):
            continue
        encoded = ast.dump(body, include_attributes=False).encode()
        if b"hailo_p1" in encoded:
            bodies.append(_technical_sha(encoded))
    _technical_require(len(bodies) == 1, "comparator_source_ambiguous")
    return bodies[0]


def _technical_adopted_source(run_dir, key, request, index):
    model, setup, backend, case, variant = key
    logical = f"models/{model}/benchmark_set/legacy_suite/{case}/run_split_onnxruntime.py"
    producer = request.get("producer_identity") or {}
    found = []
    for raw in index:
        parts = Path(raw).parts
        if (len(parts) != 4 or parts[:2] != ("jobs", "post_terminal_reference_repairs")
                or parts[-1] != "repair.json"):
            continue
        receipt_binding, encoded = _technical_file(run_dir, raw, index[raw])
        receipt = json.loads(encoded)
        if receipt.get("schema") != "onnx-splitpoint/post-terminal-reference-repair":
            continue
        adoption = receipt.get("case_runner_adoption") or {}
        files = [row for row in adoption.get("files", []) if row.get("path") == logical]
        if not files:
            continue
        _technical_require(len(files) == 1 and receipt.get("run_id") == run_dir.name
                           and receipt.get("schema_version") == 1 and receipt.get("operation_id") == parts[2]
                           and receipt.get("status") == "reference_validated_pre_seal"
                           and receipt.get("own_process_cleanup") == "proven"
                           and receipt.get("main_execution_sessions_rewritten") is False
                           and receipt.get("scientific_budgets_changed") is False, "source_receipt_invalid")
        row = files[0]
        expected_before = str(Path(raw).parent / "before" / logical)
        before = row.get("before_path")
        _technical_require(before == expected_before and before in index, "source_backup_missing")
        old_binding, source = _technical_file(run_dir, before, index[before])
        _technical_require(old_binding["sha256"] == row.get("old_sha256")
                           == str(producer.get("implementation_runner_sha256") or "").removeprefix("sha256:")
                           and old_binding["size_bytes"] == row.get("old_size_bytes")
                           == producer.get("implementation_runner_size_bytes"), "old_producer_source_mismatch")
        prior = adoption.get("prior_index") or {}
        prior_raw = str(Path(raw).parent / "before/artifact_index.json")
        _technical_require(prior.get("path") == prior_raw and prior_raw in index, "old_index_missing")
        prior_binding, prior_bytes = _technical_file(run_dir, prior_raw, index[prior_raw])
        _technical_require(prior_binding["sha256"] == str(prior.get("sha256") or "").removeprefix("sha256:"),
                           "old_index_binding_changed")
        prior_rows = _technical_index(json.loads(prior_bytes))
        _technical_require(prior_rows.get(logical) == row.get("old_index_record"), "old_source_receipt_index_conflict")
        _technical_file(run_dir, before, prior_rows.get(logical) or {})
        current_binding, current = _technical_file(run_dir, logical, index.get(logical) or {})
        _technical_require(current_binding["sha256"] == row.get("new_sha256")
                           and current_binding["size_bytes"] == row.get("new_size_bytes")
                           and current_binding["sha256"] == (adoption.get("rendered") or {}).get("sha256")
                           and _technical_callsite(source) == _BAD_HAILO_INTERFACE_BODY
                           and _technical_callsite(current) == _FIXED_HAILO_INTERFACE_BODY,
                           "unreviewed_comparator_source")
        found.append({"receipt": receipt_binding, "prior_index": prior_binding,
                      "original_source": old_binding, "current_source": current_binding,
                      "original_callsite_sha256": _BAD_HAILO_INTERFACE_BODY,
                      "current_callsite_sha256": _FIXED_HAILO_INTERFACE_BODY})
    _technical_require(len(found) == 1, "source_adoption_missing_or_ambiguous")
    return found[0]


def _technical_report_paths(paths, model, key, prefix):
    case, backend = key[3], key[2]
    result = []
    for raw in paths:
        parts = Path(raw).parts
        if (raw.startswith(prefix) and "case_reports" in parts
                and len(parts) >= 3 and parts[-3] == case
                and run_token(parts[-2].removeprefix("results_")) == backend
                and parts[-1] == "validation_report.json"):
            result.append(raw)
    return result


def _technical_collector_reports(run_dir, reports, key, prefix, index=None):
    """Collapse only the normal collector's two exact copies of one report."""
    if len(reports) <= 1:
        return reports, None
    _technical_require(len(reports) == 2, "collector_report_mirror_ambiguous")
    scoped_prefix = prefix + "remote_diagnostics/" + key[1] + "/case_reports/"
    scoped = [raw for raw in reports if raw.startswith(scoped_prefix)]
    _technical_require(len(scoped) == 1, "collector_report_mirror_setup_unbound")
    chosen = scoped[0]
    tail = chosen[len(scoped_prefix):]
    other = prefix + "remote_diagnostics/case_reports/" + tail
    _technical_require(set(reports) == {chosen, other}, "collector_report_mirror_path_unbound")
    manifest_raw = prefix + "remote_diagnostics/" + key[1] + "/result_copy_manifest.json"
    manifest_binding, data = _technical_file(run_dir, manifest_raw, None if index is None else index.get(manifest_raw) or {})
    manifest = json.loads(data)
    _technical_require(manifest.get("schema") == "onnx-splitpoint/remote-result-copy-manifest"
                       and manifest.get("schema_version") == 2 and manifest.get("status") in {"ok", "partial"}
                       and manifest.get("target_id") == key[1]
                       and Path(str(manifest.get("destination"))) == run_dir / prefix,
                       "collector_report_manifest_unbound")
    expected_source = str(Path(str(manifest.get("remote_local_run_dir"))) / tail)
    rows = [row for row in manifest.get("copied", []) if row.get("kind") == "case_validation_report"
            and row.get("source") == expected_source]
    _technical_require(len(rows) == 2
                       and {row.get("destination") for row in rows} == {str(run_dir / raw) for raw in reports},
                       "collector_report_manifest_mapping_unbound")
    descriptors = [_technical_file(run_dir, raw, None if index is None else index.get(raw) or {})[0] for raw in reports]
    _technical_require(len({(row["sha256"], row["size_bytes"]) for row in descriptors}) == 1
                       and all(row.get("size_bytes") == descriptors[0]["size_bytes"] for row in rows),
                       "collector_report_mirror_bytes_changed")
    mirror = {"manifest": manifest_binding, "reports": sorted(descriptors, key=lambda row: row["path"])}
    if manifest.get("status") == "partial":
        # One setup group may also contain an unrelated missing Full result.
        # Keep that group partial; bind this exact case to its canonical row.
        from .results import _rows_from_json, _source_row_sha256_v2797
        # Collector run IDs are filename hints (the normal writer adds _auto).
        def canonical_run(value):
            return run_token(str(value or "").removesuffix("_auto"))
        missing = manifest.get("canonical_run_ids_without_rows") or []
        _technical_require(isinstance(missing, list) and missing
                           and all(isinstance(value, str) and value and canonical_run(value) != key[2]
                                   for value in missing)
                           and manifest.get("errors") == [], "collector_partial_group_unbound")
        report = json.loads(_technical_file(run_dir, chosen, descriptors[reports.index(chosen)])[1])
        found = []
        for row in manifest.get("canonical_files") or []:
            if (canonical_run(row.get("run_id")) != key[2] or row.get("parseable") is not True
                    or row.get("nonempty") is not True):
                continue
            path = Path(str(row.get("path") or ""))
            if path.suffix.lower() != ".json":
                continue
            _technical_require(path.parent == run_dir / prefix and path.name.startswith("benchmark_results_"),
                               "collector_case_canonical_path_unbound")
            raw = path.relative_to(run_dir).as_posix()
            source, _ = _technical_file(run_dir, raw, None if index is None else index.get(raw) or {})
            copied = [entry for entry in manifest.get("copied", [])
                      if entry.get("kind") == "benchmark_result" and entry.get("canonical") is True
                      and entry.get("destination") == str(path)]
            _technical_require(len(copied) == 1 and copied[0].get("size_bytes") == source["size_bytes"],
                               "collector_case_canonical_copy_unbound")
            for body in _rows_from_json(path):
                if (body.get("setup_id") == key[1] and body.get("case_id") == key[3]
                        and run_token(body.get("run_id")) == key[2]):
                    found.append((source, body))
        _technical_require(len(found) == 1, "collector_case_canonical_row_missing_or_ambiguous")
        source, body = found[0]
        _technical_require(all(name in body and body[name] == value for name, value in report.items()),
                           "collector_case_canonical_report_changed")
        mirror["partial_group_case_source"] = {**source, "source_row_sha256": _source_row_sha256_v2797(body)}
    return [chosen], mirror


def _technical_request(run_dir, report, key, paths, prefix, *, indexed):
    from ..native_split_quality import validate_native_split_quality_binding
    descriptor = ((report.get("task_quality_input_requests_by_variant") or {}).get("composed") or {}).get("request") or {}
    expected = str(descriptor.get("sha256") or "").removeprefix("sha256:")
    candidates = []
    for raw in paths:
        parts = Path(raw).parts
        if (raw.startswith(prefix) and "quality_inputs" in parts and key[1] in parts
                and key[3] in parts and parts[-1:] == ("composed_request.json",)):
            binding, data = _technical_file(run_dir, raw, paths[raw] if indexed else None)
            if binding["sha256"] == expected and binding["size_bytes"] == descriptor.get("size_bytes"):
                candidates.append((binding, json.loads(data)))
    _technical_require(len(candidates) == 1, "request_missing_or_ambiguous")
    binding, request = candidates[0]
    _technical_require(request.get("eval_run_id") == run_dir.name
                       and (request.get("model_id"), request.get("setup_id"), run_token(request.get("source_run_id")),
                            request.get("case_id"), request.get("variant")) == (*key[:4], "composed")
                       and request.get("record_count") == request.get("reference_record_count") == 5000
                       and request.get("execution_role") == "generic_composed_quality_candidate"
                       and request.get("performance_claims_emitted") is False
                       and request.get("native_split_quality_binding_required") is True,
                       "request_identity_or_scope_changed")
    split, status = validate_native_split_quality_binding(request.get("native_split_quality_binding"),
        expected_identity={"model": key[0], "case": key[3], "setup_id": key[1], "backend": key[2],
                           "task": "detection", "precision": "uint8_dequant_fp16"})
    _technical_require(split is not None and split.get("eval_run_id") == run_dir.name
                       and split.get("binding_sha256") == request.get("native_split_quality_binding_sha256"),
                       "split_binding_invalid:" + status)
    candidate = request.get("candidate") or {}
    name = str(candidate.get("path") or "")
    _technical_require(name and Path(name).name == name, "unsafe_candidate_path")
    candidate_raw = str(Path(binding["path"]).parent / name)
    candidate_binding, _ = _technical_file(run_dir, candidate_raw, candidate)
    return binding, request, candidate_binding


def _technical_original_report(run_dir, descriptor, key):
    """Read the original report file or its already bound canonical source row."""
    from .results import _rows_from_json, _source_row_sha256_v2797
    _, data = _technical_file(run_dir, descriptor["path"], descriptor)
    if "source_row_sha256" not in descriptor:
        return json.loads(data)
    rows = [value for value in _rows_from_json(run_dir / descriptor["path"])
            if _source_row_sha256_v2797(value) == descriptor["source_row_sha256"]]
    _technical_require(len(rows) == 1, "original_report_row_missing_or_ambiguous")
    report = rows[0]
    cfg = report.get("run_cfg") or {}
    _technical_require(cfg.get("quality_evidence_eval_id") == run_dir.name
                       and (cfg.get("quality_evidence_model_id"), report.get("setup_id"),
                            run_token(report.get("run_id")), report.get("case_id"), "split") == tuple(key)
                       and cfg.get("quality_evidence_setup_id") == key[1]
                       and run_token(cfg.get("quality_evidence_source_run_id")) == key[2],
                       "original_report_row_identity_changed")
    return report


def _technical_original(run_dir, row, index):
    """Recognize only the evidenced raw-UINT8/ORT-FLOAT callsite defect."""
    from .results import _rows_from_json, _source_row_sha256_v2797
    key = result_key(row)
    if key[2] not in {"hailo8_to_trt", "hailo10_to_trt"} or key[4] != "split" or not all(key):
        return None
    prefix = f"models/{key[0]}/benchmark_results/"
    matches = []
    reports = [raw for raw in _technical_report_paths(index, key[0], key, prefix) if "/attempts/" not in raw]
    descriptors = [index[raw] for raw in reports]
    if not descriptors:
        # The normal aggregate retains the complete per-case report even when
        # the collector's diagnostic size cap omits the separate report file.
        # Reuse only the exact original pre-projection row, never a new import.
        source = Path(str(row.get("source_path") or ""))
        if source.is_absolute():
            source = source.relative_to(run_dir)
        raw = source.as_posix()
        binding, _ = _technical_file(run_dir, raw, index.get(raw) or {})
        descriptors = [{**binding, "source_row_sha256": row.get("source_row_sha256")}]
    for descriptor in descriptors:
        report_binding = {key: descriptor[key] for key in ("path", "sha256", "size_bytes")}
        report_binding["sha256"] = str(report_binding["sha256"]).removeprefix("sha256:")
        if "source_row_sha256" in descriptor:
            report_binding["source_row_sha256"] = descriptor["source_row_sha256"]
        report = _technical_original_report(run_dir, report_binding, key)
        io = ((report.get("deployment_contract") or {}).get("hailo_io_contracts") or {}).get("part1") or {}
        # FLOAT reports, including genuine negative b031, are not stale.
        if (io.get("runtime_output_format") != "uint8" or io.get("quantized_outputs") is not True
                or io.get("outputs_dequantized") is not False):
            continue
        check = ((report.get("interface_checks") or {}).get("checks") or {}).get("stage1_vs_ort") or {}
        if check.get("comparison_domain") is not None:
            continue  # Correct-domain negative results are ordinary preserved evidence.
        cfg = report.get("run_cfg") or {}
        stage1 = key[2].split("_to_", 1)[0]
        _technical_require(report.get("setup_id") == key[1] and report.get("case_id") == key[3]
                           and cfg.get("stage1_provider") in {stage1, stage1 + "h"}
                           and cfg.get("stage2_provider") == "tensorrt"
                           and check.get("comparison_domain") is None and check.get("status") == "ok"
                           and check.get("reference_backend") == "onnxruntime_cpu"
                           and check.get("candidate_backend") in {stage1, stage1 + "h"}
                           and check.get("eps") == cfg.get("interface_eps") == 0.05,
                           "original_guard_identity_changed")
        # The IO header describes the InferModel session. The archived H10
        # canonicalizer dequantizes that session's UINT8 tensors BEFORE this
        # comparison at canonical_float32 (including the auto alias). Keep
        # these ordinary FLOAT observations, including genuine numeric FAILs.
        if key[2] == "hailo10_to_trt":
            mode = str(cfg.get("boundary_mode") or "").strip().lower()
            if mode in {"canonical_float32", "auto"}:
                continue
            _technical_require(mode in {"raw_uint8_hailo", "raw_uint8_to_float32",
                                        "quantized_int8_contract", "uint8_cast_bridge"},
                               "original_boundary_mode_changed")
        request_binding, request, candidate = _technical_request(run_dir, report, key, index, prefix, indexed=True)
        source = Path(str(row.get("source_path") or ""))
        if source.is_absolute():
            source = source.relative_to(run_dir)
        source_binding, _ = _technical_file(run_dir, source.as_posix(), index.get(source.as_posix()) or {})
        bodies = [value for value in _rows_from_json(run_dir / source)
                  if _source_row_sha256_v2797(value) == row.get("source_row_sha256")]
        _technical_require(len(bodies) == 1
                           and ((bodies[0].get("interface_checks") or {}).get("checks") or {}).get("stage1_vs_ort") == check,
                           "original_normalized_body_unbound")
        source_proof = _technical_adopted_source(run_dir, key, request, index)
        matches.append({"schema": _TECHNICAL_REVALIDATION, "key": list(key),
            "original_row_sha256": _technical_body(row), "source_row_sha256": row["source_row_sha256"],
            "original_report": report_binding, "original_row_source": source_binding,
            "original_request": request_binding, "original_candidate": candidate,
            "original_native_binding": request["native_split_quality_binding"],
            "dispatch_proof": {
                "schema": _TECHNICAL_REVALIDATION, "key": list(key), "eval_run_id": run_dir.name,
                "original_binding_sha256": request["native_split_quality_binding"]["binding_sha256"],
                "reference_quality_contract_sha256": request["reference"]["quality_contract_sha256"],
                "original_artifacts": {role: {field: record[field] for field in ("sha256", "size_bytes")}
                    for role, record in request["native_split_quality_binding"]["artifacts"].items()},
                "original_preselection_sha256": request["native_split_quality_binding"]["preselection_sha256"],
                "original_boundary_contract_sha256": request["native_split_quality_binding"]["boundary_contract_sha256"],
                "original_boundary_metadata_value_sha256": request["native_split_quality_binding"]["boundary_metadata_value_sha256"],
            }, **source_proof})
    _technical_require(len(matches) <= 1, "original_report_ambiguous")
    return matches[0] if matches else None


def technical_revalidation_rows(run_dir, generic, source_index, *, reference_models=None):
    """Keep measured identities; add only a source-proven technical obligation."""
    index = source_index or _technical_index(read_json(run_dir / "artifact_index.json", default={}) or {})
    rows = []
    by_model = {}
    for selected in generic:
        if selected.get("action") != "preserve" or result_key(selected)[2] not in {"hailo8_to_trt", "hailo10_to_trt"}:
            continue
        model = selected["model_id"]
        if model not in by_model:
            identities = [row for row in generic if row.get("model_id") == model]
            primaries = _completed_normalized_rows(run_dir, model, identities)
            payload = read_json(run_dir / f"models/{model}/benchmark_results/normalized_results.json", default={}) or {}
            raw_rows = {result_key(row): row for row in payload.get("results", [])}
            _technical_require(len(raw_rows) == len(payload.get("results", [])), "normalized_identity_ambiguous")
            # The logical selector adds observer annotations. Bind the actual
            # archived row bytes, not that temporary annotated representation.
            by_model[model] = {result_key(row): raw_rows[result_key(row)] for row in primaries}
        old = by_model[model].get(result_key(selected))
        if old is None:
            continue
        if old.get("technical_interface_revalidation"):
            validate_technical_revalidation(run_dir, old)
            if reference_models is not None:
                reference_models.add(model)
            continue  # A corrected positive OR negative observation is terminal.
        proof = _technical_original(run_dir, old, index)
        if proof:
            retained = _technical_retained(run_dir, proof)
            rows.append({**selected, "stage": "quality_source", "action": "preserve" if retained else "execute",
                "reason": "source_proven_raw_uint8_float_comparator", "performance_action": "preserve",
                "technical_revalidation": proof, "retained_revalidation": retained})
    return rows


def _technical_recipe(request):
    """Compare semantic recipe bytes, allowing only attested source-identity hashes.

    The caller independently proves the old/new generated sources and the exact
    reviewed callsite. Dataset/model/preprocessing/threshold contents are never
    removed. Only hashes whose inputs contain the changed runner SHA differ.
    """
    contract = copy.deepcopy(request.get("quality_contract") or {})
    for parent, name in ((contract, "quality_contract_sha256"),
                         (contract, "quality_record_endpoint_contract_sha256")):
        parent.pop(name, None)
    for name in ("decoder", "nms", "quality_record_endpoint"):
        part = contract.get(name) or {}
        part.pop("sha256", None)
        identity = part.get("identity") or {}
        identity.pop("implementation_runner_sha256", None)
        if name == "quality_record_endpoint":
            identity.pop("decoder_contract_sha256", None)
            identity.pop("nms_contract_sha256", None)
    fields = ("eval_run_id", "model_id", "setup_id", "source_run_id", "backend", "case_id", "variant", "task",
              "record_count", "reference_record_count", "expected_image_ids", "expected_image_ids_sha256",
              "statistics", "policy_sha256", "metric_gate_config", "runtime_precision_identity",
              "source_model_sha256", "part1_model_sha256", "part2_model_sha256", "endpoint_contract",
              "endpoint_contract_hash", "preprocessing_contract_sha256")
    return {"fields": {key: request.get(key) for key in fields}, "quality_contract": contract}


def _technical_float_check(check, split, old_check):
    import math
    contract = split["boundary_contract"]
    _technical_require(check.get("status") == "ok"
                       and check.get("enabled") is old_check.get("enabled") is True
                       and check.get("required", True) is old_check.get("required", True) is True
                       and check.get("reference_backend") == old_check.get("reference_backend") == "onnxruntime_cpu"
                       and check.get("candidate_backend") == old_check.get("candidate_backend")
                       and check.get("comparison_domain") == "canonical_onnx_float32"
                       and check.get("candidate_quantized_outputs") is True
                       and check.get("comparison_binding_sha256") == split["binding_sha256"]
                       and check.get("eps") == old_check.get("eps") == 0.05
                       and all(type(check.get(name)) is int and check[name] == 1
                               for name in ("num_compared_tensors", "num_ref_tensors", "num_out_tensors"))
                       and check.get("unmatched_ref_names") == check.get("unmatched_out_names") == [],
                       "fresh_comparison_incomplete")
    tensors = check.get("per_tensor") or []
    _technical_require(len(tensors) == 1 and all(type(check.get(k)) is bool for k in
                       ("mapping_pass", "shape_pass", "structural_pass", "numeric_pass", "pass")),
                       "fresh_comparison_flags_invalid")
    value = tensors[0]
    tensor_name = contract["boundary_tensor_name"]
    _technical_require(value.get("ref_name") == value.get("out_name") == tensor_name
                       and value.get("match") == "name" and value.get("shape_match") is True
                       and value.get("ref_shape") == value.get("out_shape")
                       == split["boundary_metadata_payload"]["boundary_tensor"]["canonical_part2_shape"],
                       "fresh_tensor_contract_changed")
    for number in (check.get("max_abs"), check.get("mean_abs_mean"), value.get("max_abs"), value.get("mean_abs")):
        _technical_require(type(number) in (int, float) and math.isfinite(number) and number >= 0,
                           "fresh_comparison_nonfinite")
    numeric = value["max_abs"] <= check["eps"]
    _technical_require(check["mapping_pass"] is True and check["shape_pass"] is True
                       and check["structural_pass"] is True and check["numeric_pass"] is numeric
                       and check["pass"] is numeric and value.get("pass") is numeric
                       and check["max_abs"] == value["max_abs"] and check["mean_abs_mean"] == value["mean_abs"],
                       "fresh_comparison_contradictory")




def _technical_quality_recipe_relocation(run_dir, proof, request, old_request, session_id):
    """Prove only the normal completion workspace's output-contract relocation."""
    current_recipe, original_recipe = _technical_recipe(request), _technical_recipe(old_request)
    if current_recipe == original_recipe:
        return None
    from .execution_binding import _remote_transport_run_id
    model, setup = proof["key"][:2]
    old_id = _remote_transport_run_id(run_root=run_dir, model_id=model, target_id=setup, gates={})
    new_id = _remote_transport_run_id(run_root=run_dir, model_id=model, target_id=setup,
        gates={"selective_completion": True}, workflow_session_id=session_id)
    old_attestation = (old_request.get("endpoint_contract") or {}).get("output_endpoint_attestation") or {}
    fresh_attestation = (request.get("endpoint_contract") or {}).get("output_endpoint_attestation") or {}
    old_declaration = old_attestation.get("declared_contract") or {}
    fresh_declaration = fresh_attestation.get("declared_contract") or {}
    old_source = str(old_attestation.get("declaration_source") or "")
    parts = Path(old_source).parts
    _technical_require(Path(old_source).is_absolute() and ".." not in parts
                       and tuple(parts[-4:]) == (old_id, "1", "suite", "output_contracts.json")
                       and old_declaration.get("declaration_source") == old_source,
                       "fresh_endpoint_original_declaration_unbound")
    expected = str(Path(*parts[:-4], new_id, "1", "suite", "output_contracts.json"))
    _technical_require(fresh_attestation.get("declaration_source") == expected
                       and fresh_declaration.get("declaration_source") == expected,
                       "fresh_endpoint_completion_declaration_unbound")
    raw = f"models/{model}/benchmark_set/legacy_suite/output_contracts.json"
    index = _technical_index(read_json(run_dir / "artifact_index.json", default={}) or {})
    _technical_require(raw in index, "fresh_endpoint_source_not_indexed")
    source, data = _technical_file(run_dir, raw, index[raw])
    record_sha = str(old_declaration.get("source_contract_sha256") or "")
    _technical_require(source["sha256"] == old_declaration.get("source_contracts_sha256")
                       == fresh_declaration.get("source_contracts_sha256")
                       and len(record_sha) == 64 and record_sha == fresh_declaration.get("source_contract_sha256")
                       and sum(_technical_body(row) == record_sha
                               for row in (json.loads(data).get("contracts") or [])) == 1,
                       "fresh_endpoint_contract_bytes_changed")
    # Compare every other recipe field exactly. Never rewrite either request.
    current_recipe = copy.deepcopy(current_recipe)
    attestation = current_recipe["fields"]["endpoint_contract"]["output_endpoint_attestation"]
    attestation["declaration_source"] = old_source
    attestation["declared_contract"]["declaration_source"] = old_source
    _technical_require(current_recipe == original_recipe, "fresh_frozen_quality_recipe_changed")
    return {"scope": "normal_completion_output_contracts_relocation", "source_contracts": source,
            "source_contract_sha256": record_sha, "original_declaration_source": old_source,
            "completion_declaration_source": expected, "completion_transport_run_id": new_id}


def _technical_negative_quality_input_context(fresh, original, request, old_request, check):
    """Admit only a negative frozen-QC observation, never validate old timing."""
    from ..preprocessing_contract import resolve_image_preprocessing_contract
    _technical_require(check.get("status") == "ok"
                       and check.get("numeric_pass") is False and check.get("pass") is False
                       and check.get("comparison_domain") == "canonical_onnx_float32"
                       and all(check.get(name) is True for name in ("mapping_pass", "shape_pass", "structural_pass")),
                       "fresh_changed_input_requires_numeric_failure")
    old = copy.deepcopy(original.get("timing_input_binding") or {})
    current = copy.deepcopy(fresh.get("timing_input_binding") or {})
    preprocessing = (old_request.get("quality_contract") or {}).get("preprocessing") or {}
    identity = preprocessing.get("identity") or {}
    _technical_require(preprocessing == ((request.get("quality_contract") or {}).get("preprocessing") or {})
                       and identity.get("task") == "detection", "fresh_quality_input_recipe_changed")
    expected, expected_sha = resolve_image_preprocessing_contract(
        task=identity.get("task"), target_hw=identity.get("target_hw"), declared=identity)
    _technical_require(current.get("preprocessing_contract") == expected
                       and current.get("preprocessing_contract_sha256") == preprocessing.get("sha256") == expected_sha
                       and current.get("canonical_preprocessing_executed") is True,
                       "fresh_quality_input_not_canonical")
    ids = old_request.get("expected_image_ids") or []
    first = str(ids[0]) if ids else ""
    def digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    _technical_require(first and ids == request.get("expected_image_ids")
                       and Path(str(old.get("input_image") or "")).name
                       == Path(str(current.get("input_image") or "")).name == first
                       and old.get("image_sequence") == current.get("image_sequence") == [first]
                       and digest(old.get("input_image_sha256"))
                       and old["input_image_sha256"] == current.get("input_image_sha256"),
                       "fresh_quality_first_frame_changed")
    _technical_require(digest(current.get("prepared_rgb_sha256"))
                       and current["prepared_rgb_sha256"] == current.get("canonical_rgb_sha256")
                       and digest(current.get("generic_prepared_input_sha256")),
                       "fresh_quality_prepared_input_unbound")
    old_backend = old.get("backend_prepared_inputs") or {}
    current_backend = current.get("backend_prepared_inputs") or {}
    def encodings(value):
        return {name: {k: v for k, v in record.items() if k != "sha256"}
                for name, record in value.items()}
    _technical_require(old_backend and encodings(old_backend) == encodings(current_backend)
                       and all(digest(record.get("sha256")) for record in current_backend.values()),
                       "fresh_quality_backend_encoding_changed")
    # Only the normal canonical preprocessing outputs may differ. Other input
    # bookkeeping remains exactly bound to the original observation.
    changes = {"input_image", "preprocessing_contract", "preprocessing_contract_sha256",
               "canonical_preprocessing_executed", "geometry", "prepared_rgb_sha256", "canonical_rgb_sha256",
               "generic_prepared_input_sha256", "backend_prepared_inputs"}
    _technical_require({k: v for k, v in old.items() if k not in changes}
                       == {k: v for k, v in current.items() if k not in changes}, "fresh_input_recipe_changed")
    return {"scope": "frozen_quality_first_frame_negative_only",
            "same_input_as_original_timing": False, "original_timing_guard_replaced": False,
            "validates_original_timing_observation": False,
            "original_timing_input_binding": old, "quality_input_binding": current,
            "original_quality_preprocessing": copy.deepcopy(preprocessing),
            "first_expected_image_id": first, "input_image_sha256": old["input_image_sha256"]}


def _technical_fresh(run_dir, proof, report_binding, session_id, cpu_reference):
    key = tuple(proof["key"])
    prefix = f"models/{key[0]}/benchmark_results/attempts/{session_id}/"
    _technical_require(len(session_id) == 32 and all(c in "0123456789abcdef" for c in session_id)
                       and report_binding["path"].startswith(prefix)
                       and _technical_report_paths([report_binding["path"]], key[0], key, prefix),
                       "fresh_attempt_identity_invalid")
    bound, data = _technical_file(run_dir, report_binding["path"], report_binding)
    fresh = json.loads(data)
    original = _technical_original_report(run_dir, proof["original_report"], proof["key"])
    cfg = fresh.get("run_cfg") or {}
    _technical_require(fresh.get("quality_source_only") is True and fresh.get("performance_measurement_executed") is False
                       and fresh.get("measured_variants") == [] and fresh.get("primary_variant") == "composed"
                       and fresh.get("setup_id") == key[1] and fresh.get("case_id") == key[3]
                       and cfg.get("quality_source_only") is True and cfg.get("variants") == "composed"
                       and cfg.get("quality_evidence_eval_id") == run_dir.name
                       and cfg.get("quality_evidence_model_id") == key[0]
                       and cfg.get("quality_evidence_setup_id") == key[1]
                       and run_token(cfg.get("quality_evidence_source_run_id")) == key[2]
                       and cfg.get("stage1_provider") == (original.get("run_cfg") or {}).get("stage1_provider")
                       and cfg.get("stage2_provider") == "tensorrt"
                       and cfg.get("native_trt_precision") == "uint8_dequant_fp16"
                       and cfg.get("interface_eps") == 0.05
                       and all(type(cfg.get(k)) is int and cfg[k] == 0 for k in
                               ("warmup", "runs", "phase_runs", "throughput_frames", "throughput_warmup_frames"))
                       and cfg.get("native_trt_build") is False and cfg.get("native_trt_no_fallback") is True
                       and cfg.get("energy_measurement_only") is False and cfg.get("existing_management_cpu_reference_json"),
                       "fresh_not_exact_quality_source_only")
    def input_recipe(value):
        value = copy.deepcopy(value)
        if value.get("input_image"):
            value["input_image"] = Path(value["input_image"]).name
        return value
    same_timing_input = (input_recipe(fresh.get("timing_input_binding") or {})
                         == input_recipe(original.get("timing_input_binding") or {}))
    # These are the normal canonical quality-input imports for this one attempt.
    roots = run_dir / prefix / "quality_inputs" / key[1]
    paths = {p.relative_to(run_dir).as_posix(): {} for p in roots.rglob("composed_request.json")}
    req_binding, request, candidate = _technical_request(run_dir, fresh, key, paths, prefix, indexed=False)
    old_request = json.loads(_technical_file(run_dir, proof["original_request"]["path"], proof["original_request"])[1])
    old_split = old_request["native_split_quality_binding"]
    split = request["native_split_quality_binding"]
    replay = split.get("cache_verify_replay") or {}
    _technical_require(replay.get("artifact_policy") == "cache_verify_only"
                       and replay.get("compiler_dispatched") is False
                       and replay.get("source_binding_sha256") == old_split["binding_sha256"],
                       "fresh_exact_cache_replay_required")
    def original_binding(value):
        return {key: item for key, item in value.items() if key not in {
            "binding_sha256", "cache_verify_replay", "producer_binding_sha256", "source_request_sha256",
            "central_result_sha256", "central_quality_selection", "central_quality_selection_sha256"}}
    _technical_require(original_binding(split) == original_binding(old_split), "fresh_original_artifact_set_changed")
    recipe_relocation = _technical_quality_recipe_relocation(run_dir, proof, request, old_request, session_id)
    source = proof["current_source"]
    current_source, current_bytes = _technical_file(run_dir, source["path"], source)
    producer = request.get("producer_identity") or {}
    _technical_require(current_source["sha256"] == producer.get("implementation_runner_sha256")
                       and current_source["size_bytes"] == producer.get("implementation_runner_size_bytes")
                       and _technical_callsite(current_bytes) == _FIXED_HAILO_INTERFACE_BODY,
                       "fresh_source_unbound")
    split = request["native_split_quality_binding"]
    io = ((fresh.get("deployment_contract") or {}).get("hailo_io_contracts") or {}).get("part1") or {}
    _technical_require(io.get("runtime_output_format") == "uint8" and io.get("quantized_outputs") is True
                       and io.get("outputs_dequantized") is False
                       and io.get("artifact") == split["artifacts"]["part1_runtime"]["path"],
                       "fresh_runtime_domain_unbound")
    check = ((fresh.get("interface_checks") or {}).get("checks") or {}).get("stage1_vs_ort") or {}
    old_check = original["interface_checks"]["checks"]["stage1_vs_ort"]
    _technical_float_check(check, split, old_check)
    input_context = None if same_timing_input else _technical_negative_quality_input_context(
        fresh, original, request, old_request, check)
    from ..quality_service import quality_request_from_manifest
    reference_binding, reference_bytes = _technical_file(run_dir, cpu_reference["path"], cpu_reference)
    loaded = quality_request_from_manifest(run_dir / req_binding["path"],
        reference_artifact={**reference_binding, "path": str(run_dir / reference_binding["path"])},
        reference_artifact_bytes=reference_bytes)
    _technical_require(len(loaded.reference_records) == len(loaded.candidate_records) == 5000,
                       "fresh_cpu_reference_or_candidate_cardinality_changed")
    mirror = None
    scoped_prefix = prefix + "remote_diagnostics/" + key[1] + "/case_reports/"
    if bound["path"].startswith(scoped_prefix):
        other = prefix + "remote_diagnostics/case_reports/" + bound["path"][len(scoped_prefix):]
        _paths, mirror = _technical_collector_reports(run_dir, [bound["path"], other], key, prefix)
    return {"report": bound, "request": req_binding, "candidate": candidate,
            "session_id": session_id, "stage1_vs_ort": check, "cpu_reference": reference_binding,
            **({"comparison_input_context": input_context} if input_context is not None else {}),
            **({"endpoint_declaration_relocation": recipe_relocation} if recipe_relocation is not None else {}),
            **({"collector_mirror": mirror} if mirror is not None else {})}


def _technical_retained(run_dir, proof):
    """Reuse a closed imported technical observation instead of more inference."""
    payload = read_json(run_dir / "artifact_index.json", default={}) or {}
    index = _technical_index(payload)
    key = tuple(proof["key"])
    prefix = f"models/{key[0]}/benchmark_results/attempts/"
    reports = _technical_report_paths(index, key[0], key, prefix)
    if not reports:
        return None
    _technical_require((payload.get("terminal_closure") or {}).get("status") == "pass",
                       "retained_attempt_not_closed")
    status_raw = f"quality_management/references/{key[0]}/management_cpu_reference_status.json"
    _, data = _technical_file(run_dir, status_raw, index.get(status_raw) or {})
    status = json.loads(data)
    _technical_require(status.get("status") in {"completed", "cache_hit"}
                       and status.get("reference_immutable") is True and status.get("model_id") == key[0],
                       "retained_cpu_reference_not_completed")
    reference_raw = Path(status["reference_path"]).relative_to(run_dir).as_posix()
    reference, _ = _technical_file(run_dir, reference_raw, index.get(reference_raw) or {})
    _technical_require(reference["sha256"] == str(status.get("reference_sha256") or "").removeprefix("sha256:")
                       and reference["size_bytes"] == status.get("reference_size_bytes"), "retained_cpu_reference_changed")
    accepted = []
    for session in sorted({Path(raw).parts[4] for raw in reports}):
        session_prefix = prefix + session + "/"
        selected, _mirror = _technical_collector_reports(run_dir,
            [raw for raw in reports if raw.startswith(session_prefix)], key, session_prefix, index)
        for raw in selected:
            accepted.append(_technical_fresh(run_dir, proof, index[raw], session, reference))
    _technical_require(len(accepted) == 1, "retained_attempt_ambiguous")
    return accepted[0]


def technical_diagnostic_request_bindings(run_dir, model_id, selections=None):
    """Keep proven technical replay requests outside canonical Quality choice.

    Old N5000 predictions remain the ordinary scientific inputs. The new
    quality_source request exists only because the normal live QC path is
    required to recover an unsaved cut-tensor check. It is retained in attempts,
    with the exact report/source/request provenance, and is never a second
    competing Central -> Native producer for that same measured identity.
    """
    bindings = {}
    def retain(fresh):
        descriptor = {**fresh["request"], "native_split_quality_binding_sha256":
                      fresh["stage1_vs_ort"]["comparison_binding_sha256"]}
        _technical_file(run_dir, descriptor["path"], descriptor)
        previous = bindings.setdefault(descriptor["path"], descriptor)
        _technical_require(previous == descriptor, "diagnostic_request_ambiguous")
    normalized = read_json(run_dir / f"models/{model_id}/benchmark_results/normalized_results.json", default={}) or {}
    for row in normalized.get("results", []):
        if row.get("technical_interface_revalidation"):
            _technical_require(row.get("model_id") == model_id, "diagnostic_model_changed")
            validate_technical_revalidation(run_dir, row)
            retain(row["technical_interface_revalidation"]["fresh"])
    if selections is None:
        # A failed import may leave closed, collected observations without a
        # normalized projection. Reuse the ordinary scope/retention selector;
        # never infer diagnostic status from case names or paths alone.
        scope = read_json(run_dir / f"models/{model_id}/benchmark_set/required_run_scope.json", default={}) or {}
        identities = scope.get("identities") or []
        primary = {result_key(row) for row in _completed_normalized_rows(run_dir, model_id, identities)}
        generic = [{**row, "setup_id": row.get("expected_setup_id"), "action": "preserve"}
                   for row in identities
                   if result_key({**row, "setup_id": row.get("expected_setup_id")}) in primary]
        _original, source_index = prepare_source_evidence(run_dir)
        selections = technical_revalidation_rows(run_dir, generic, _technical_index(source_index))
    for selection in selections or []:
        retained = selection.get("retained_revalidation")
        if selection.get("model_id") != model_id or not retained:
            continue
        proof = selection["technical_revalidation"]
        _technical_require(tuple(proof["key"])[0] == model_id, "retained_diagnostic_model_changed")
        fresh = _technical_fresh(run_dir, proof, retained["report"], retained["session_id"], retained["cpu_reference"])
        _technical_require(fresh == retained, "retained_diagnostic_changed")
        retain(fresh)
    return bindings


def _quality_technical_diagnostic_bindings(run_dir):
    """Use only fully revalidated normal technical observations for selection."""
    bindings = {}
    for source in sorted(run_dir.glob("models/*/benchmark_results/normalized_results.json")):
        model_id = source.parent.parent.name
        for raw, descriptor in technical_diagnostic_request_bindings(run_dir, model_id).items():
            previous = bindings.setdefault(raw, descriptor)
            _technical_require(previous == descriptor, "diagnostic_request_ambiguous")
    return bindings


def _quality_result_is_technical_diagnostic(run_dir, result, request, bindings):
    """Keep exact technical results historical; reject unbound replay mirrors."""
    raw = str(result.get("source_request") or "")
    expected = str(result.get("source_request_sha256") or "").removeprefix("sha256:")
    descriptor = bindings.get(raw)
    if descriptor is not None:
        _technical_require(expected == descriptor["sha256"], "diagnostic_result_request_changed")
        _technical_file(run_dir, raw, descriptor)
        identity = result.get("request_identity") or {}
        for field in ("model_id", "setup_id", "case_id", "variant"):
            value = result.get(field) or identity.get(field)
            _technical_require(not value or value == request.get(field), "diagnostic_result_identity_changed")
        nested_sha = str(identity.get("source_request_sha256") or "").removeprefix("sha256:")
        _technical_require(not nested_sha or nested_sha == expected, "diagnostic_result_request_changed")
        return True
    hashes = {value["sha256"] for value in bindings.values()}
    replay_bindings = {value["native_split_quality_binding_sha256"] for value in bindings.values()}
    split = request.get("native_split_quality_binding") or {}
    _technical_require(expected not in hashes
                       and str(request.get("native_split_quality_binding_sha256") or "") not in replay_bindings
                       and str(split.get("binding_sha256") or "") not in replay_bindings,
                       "diagnostic_result_unbound_request_mirror")
    return False


_TECHNICAL_FIELDS = frozenset({
    "interface_check_pass", "interface_check_status", "interface_gate_blocking",
    "interface_stage1_status", "interface_stage1_pass", "interface_stage1_max_abs", "interface_stage1_mean_abs_mean",
    "interface_stage1_num_compared_tensors", "interface_stage1_structural_pass", "interface_stage1_mapping_pass",
    "interface_stage1_shape_pass", "interface_stage1_numeric_pass", "strict_boundary_numeric_pass",
    "strict_boundary_numeric_status", "numerical_similarity_pass", "numerical_similarity_status",
    "numerical_similarity_reason", "numerical_similarity_scope", "numerical_similarity_metric",
    "numerical_similarity_value", "numerical_similarity_threshold", "numerical_similarity_policy_id",
})


def _technical_projection(original, check, old_report, *, input_context=None):
    """Project identical-input evidence; retain old guards for QC-only negatives."""
    out = copy.deepcopy(original)
    if input_context is not None:
        _technical_require(input_context.get("scope") == "frozen_quality_first_frame_negative_only"
                           and input_context.get("same_input_as_original_timing") is False
                           and input_context.get("original_timing_guard_replaced") is False
                           and input_context.get("validates_original_timing_observation") is False
                           and check.get("numeric_pass") is False and check.get("pass") is False
                           and original.get("interface_gate_blocking") is True
                           and original.get("interface_check_pass") is False,
                           "quality_negative_must_preserve_original_blocking_guard")
        return out  # New QC context remains separately bound in fresh provenance.
    checks = copy.deepcopy(old_report["interface_checks"]["checks"])
    checks["stage1_vs_ort"] = copy.deepcopy(check)
    required = [value for value in checks.values() if isinstance(value, Mapping) and value.get("required", True) is not False]
    overall = all(value.get("pass") is True for value in required)
    others = [value for name, value in checks.items() if name != "stage1_vs_ort"
              and isinstance(value, Mapping) and value.get("required", True) is not False]
    others_negative = any(value.get("pass") is False for value in others)
    others_unresolved = any(type(value.get("pass")) is not bool for value in others)
    numeric = (False if not check["numeric_pass"] or others_negative or original.get("raw_head_drift_pass") is False
               else None if others_unresolved else True)
    out.update(interface_check_pass=overall, interface_check_status="passed" if overall else "failed",
        interface_gate_blocking=not overall, interface_stage1_status=check["status"], interface_stage1_pass=check["pass"],
        interface_stage1_max_abs=check["max_abs"], interface_stage1_mean_abs_mean=check["mean_abs_mean"],
        interface_stage1_num_compared_tensors=check["num_compared_tensors"],
        interface_stage1_structural_pass=check["structural_pass"], interface_stage1_mapping_pass=check["mapping_pass"],
        interface_stage1_shape_pass=check["shape_pass"], interface_stage1_numeric_pass=check["numeric_pass"],
        strict_boundary_numeric_pass=numeric, strict_boundary_numeric_status=("preserved_other_guard_unresolved" if numeric is None
            else "preserved_other_numeric_failure" if others_negative else "corrected_stage1_comparison"),
        numerical_similarity_pass=numeric, numerical_similarity_status="passed" if numeric is True else "inconclusive" if numeric is None else "failed",
        numerical_similarity_reason="preserved_other_guard_unresolved" if numeric is None else "source_bound_corrected_stage1_comparison", numerical_similarity_scope="split_boundary",
        numerical_similarity_metric="max_abs", numerical_similarity_value=check["max_abs"],
        numerical_similarity_threshold=check["eps"], numerical_similarity_policy_id=_TECHNICAL_REVALIDATION)
    return out


def merge_technical_revalidations(run_dir, previous, source_records, selections, session_id, original_root, reference_status):
    """Use the normal imported attempt report without replacing a measured row."""
    selected = {tuple(row["technical_revalidation"]["key"]): row["technical_revalidation"]
                for row in selections if (row.get("action") == "execute" or row.get("retained_revalidation"))
                and row.get("technical_revalidation")}
    retained = {tuple(row["technical_revalidation"]["key"]): row["retained_revalidation"]
                for row in selections if row.get("retained_revalidation")}
    if not selected:
        return previous
    _technical_require(reference_status.get("schema") == "onnx-splitpoint/management-cpu-reference-job"
                       and reference_status.get("status") in {"completed", "cache_hit"}
                       and reference_status.get("reference_immutable") is True
                       and reference_status.get("semantic_reference_only") is True
                       and reference_status.get("provider") == "onnxruntime_cpu", "existing_cpu_reference_required")
    reference_path = Path(reference_status["reference_path"]).relative_to(run_dir).as_posix()
    reference = {"path": reference_path, "sha256": reference_status["reference_sha256"],
                 "size_bytes": reference_status["reference_size_bytes"]}
    records = []
    for record in source_records:
        raw = Path(str(record.get("path") or ""))
        if raw.is_absolute():
            raw = raw.relative_to(run_dir)
        records.append(raw.as_posix())
    output = []
    for original in previous:
        proof = selected.get(result_key(original))
        if proof is None:
            output.append(original)
            continue
        _technical_require(_technical_body(original) == proof["original_row_sha256"], "original_row_body_changed_before_merge")
        key = tuple(proof["key"])
        prefix = f"models/{key[0]}/benchmark_results/attempts/{session_id}/"
        _technical_require(reference_status.get("model_id") == key[0], "cpu_reference_model_changed")
        if key in retained:
            previous_observation = retained[key]
            fresh = _technical_fresh(run_dir, proof, previous_observation["report"], previous_observation["session_id"], reference)
            _technical_require(fresh == previous_observation, "retained_observation_changed_before_merge")
        else:
            reports = _technical_report_paths(records, key[0], key, prefix)
            reports, _mirror = _technical_collector_reports(run_dir, reports, key, prefix)
            _technical_require(len(reports) == 1, "fresh_report_missing_or_ambiguous")
            binding, _ = _technical_file(run_dir, reports[0])
            fresh = _technical_fresh(run_dir, proof, binding, session_id, reference)
        baseline_raw = (original_root / f"models/{key[0]}/benchmark_results/normalized_results.json").relative_to(run_dir).as_posix()
        baseline, data = _technical_file(run_dir, baseline_raw)
        matches = [row for row in json.loads(data).get("results", []) if result_key(row) == key]
        _technical_require(len(matches) == 1 and matches[0] == original, "preserved_normalized_baseline_changed")
        old_report = _technical_original_report(run_dir, proof["original_report"], proof["key"])
        merged = _technical_projection(original, fresh["stage1_vs_ort"], old_report,
            input_context=fresh.get("comparison_input_context"))
        merged["technical_interface_revalidation"] = {"schema": _TECHNICAL_REVALIDATION,
            "proof": proof, "original_normalized": baseline, "fresh": fresh,
            "old_observation_retained": True, "timing_quality_native_energy_replaced": False}
        output.append(merged)
    _technical_require(set(selected).issubset({result_key(row) for row in output}), "selected_measured_identity_missing")
    return output


def _technical_central_projection(run_dir, row, metadata, entry):
    """Replay only a normal result for the preserved original composed request."""
    from .runner import EvaluationWorkflowRunner
    from .central_quality_join import apply_central_quality_result_to_row
    from .artifacts import sha256_json
    descriptor = entry["result"]
    _, data = _technical_file(run_dir, descriptor["path"], descriptor)
    normal_index = read_json(run_dir / "artifact_index.json", default={}) or {}
    indexed = _technical_index(normal_index)
    if descriptor["path"] in indexed:
        _technical_file(run_dir, descriptor["path"], indexed[descriptor["path"]])
    else:
        _technical_require((normal_index.get("terminal_closure") or {}).get("status") != "pass",
                           "central_projection_terminal_index_missing")
    result = json.loads(data)
    proof = metadata["proof"]
    original_request = proof["original_request"]
    key = tuple(proof["key"])
    identity, errors = EvaluationWorkflowRunner._central_quality_result_identity(result)
    arguments = entry["writer_arguments"]
    expected_arguments = {"variant": "composed", "match_role": "primary", "result_cancelled": False,
                          "smoke_quality": False, "match_mode": arguments.get("match_mode")}
    _technical_require(arguments == expected_arguments and arguments["match_mode"] in {
        "exact_request_sha_logical_identity", "legacy_strict_identity_fallback"}, "central_projection_scope_changed")
    _technical_require(not errors and result.get("source_request") == original_request["path"]
                       and str(result.get("source_request_sha256") or "").removeprefix("sha256:") == original_request["sha256"]
                       and str(identity.get("source_request_sha256") or "").removeprefix("sha256:") == original_request["sha256"]
                       and result.get("model_id") == key[0] and result.get("case_id") == key[3]
                       and result.get("variant") == "composed"
                       and result.get("status") in {"completed", "failed"}
                       and result.get("technical_status") in {"completed", "failed"}, "central_projection_original_request_unbound")
    expected_identity = EvaluationWorkflowRunner._quality_request_identity_for_row_variant(row, "composed")
    _technical_require(EvaluationWorkflowRunner._central_quality_contract_identity_matches(expected_identity, identity)
                       and EvaluationWorkflowRunner._central_quality_producer_matches_row(row, result,
                           require_setup=True, request_hash_bound=True), "central_projection_normal_contract_changed")
    gate = row.get("task_quality_gate") or {}
    if (gate.get("execution_location") == "central_management"
            and gate.get("technical_status") == "completed" and gate.get("management_result_sha256")):
        _technical_require(gate["management_result_sha256"] == sha256_json(result),
                           "central_projection_completed_quality_changed")
    apply_central_quality_result_to_row(row, result, identity, **arguments)
    return row


def record_technical_central_projection(run_dir, row, before, result, result_identity, *, session_id,
                                       variant, match_mode, match_role, result_cancelled, smoke_quality):
    """Bind the exact existing normal Central writer; preserve original products."""
    metadata = before["technical_interface_revalidation"]
    raw_request = metadata["proof"]["original_request"]["path"]
    request = Path(raw_request)
    parts = request.parts
    _technical_require("quality_inputs" in parts and len(session_id) == 32
                       and all(c in "0123456789abcdef" for c in session_id), "central_projection_session_invalid")
    suffix = Path(*parts[parts.index("quality_inputs") + 1:])
    result_name = request.stem.replace("_request", "") + "_central_result.json"
    candidates = [str(Path(f"models/{before['model_id']}/benchmark_results/attempts/{session_id}/quality_inputs")
                      / suffix.with_name(result_name)), request.with_name(result_name).as_posix()]
    candidates.extend(entry["result"]["path"] for entry in metadata.get("central_projections", []))
    matches = []
    for raw in dict.fromkeys(candidates):
        if not (run_dir / raw).exists():
            continue
        binding, data = _technical_file(run_dir, raw)
        if json.loads(data) == result:
            matches.append(binding)
    if not matches:
        # Local statistics can have completed in an earlier normal attempt,
        # while the original failed sidecar remains preserved beside its
        # request. Reuse only exact, indexed completed products for that request.
        indexed = _technical_index(read_json(run_dir / "artifact_index.json", default={}) or {})
        expected_tail = suffix.with_name(result_name).parts
        prefix = ("models", str(before["model_id"]), "benchmark_results", "attempts")
        for raw, descriptor in sorted(indexed.items()):
            historical_parts = Path(raw).parts
            if (len(historical_parts) != 6 + len(expected_tail)
                    or historical_parts[:4] != prefix
                    or historical_parts[5] != "quality_inputs"
                    or historical_parts[6:] != expected_tail
                    or historical_parts[4] == session_id
                    or len(historical_parts[4]) != 32
                    or any(c not in "0123456789abcdef" for c in historical_parts[4])
                    or raw in candidates):
                continue
            binding, data = _technical_file(run_dir, raw, descriptor)
            previous = json.loads(data)
            _technical_require(isinstance(previous, Mapping), "central_projection_result_invalid")
            if (previous.get("status") == previous.get("technical_status") == "completed"
                    and previous.get("source_request") == raw_request
                    and str(previous.get("source_request_sha256") or "").removeprefix("sha256:")
                    == metadata["proof"]["original_request"]["sha256"]
                    and previous == result):
                matches.append(binding)
    _technical_require(bool(matches), "central_projection_result_not_bound")
    entry = {"result": matches[0], "writer_arguments": {"variant": variant, "match_mode": match_mode,
        "match_role": match_role, "result_cancelled": result_cancelled, "smoke_quality": smoke_quality}}
    expected = _technical_central_projection(run_dir, copy.deepcopy(before), metadata, entry)
    _technical_require(expected == row, "central_projection_unexpected_row_change")
    projections = copy.deepcopy(metadata.get("central_projections", []))
    # Record actual normal writer applications, including repeated joins whose
    # derived accuracy fields settle over more than one existing application.
    # An exact no-op adds no new provenance or result generation.
    if row != before:
        projections.append(entry)
    row["technical_interface_revalidation"]["central_projections"] = projections



def validate_technical_revalidation(run_dir, current, old=None):
    """Recheck provenance and exact projection before relaxing one equality."""
    metadata = current.get("technical_interface_revalidation") or {}
    _technical_require(metadata.get("schema") == _TECHNICAL_REVALIDATION
                       and metadata.get("old_observation_retained") is True
                       and metadata.get("timing_quality_native_energy_replaced") is False, "projection_metadata_invalid")
    proof = metadata["proof"]
    baseline = metadata["original_normalized"]
    _, data = _technical_file(run_dir, baseline["path"], baseline)
    rows = [row for row in json.loads(data).get("results", []) if list(result_key(row)) == proof["key"]]
    _technical_require(len(rows) == 1 and _technical_body(rows[0]) == proof["original_row_sha256"], "projection_baseline_unbound")
    index = _technical_index(read_json(run_dir / "artifact_index.json", default={}) or {})
    _technical_require(_technical_original(run_dir, rows[0], index) == proof, "projection_original_proof_changed")
    fresh = _technical_fresh(run_dir, proof, metadata["fresh"]["report"], metadata["fresh"]["session_id"],
                             metadata["fresh"]["cpu_reference"])
    _technical_require(fresh == metadata["fresh"], "projection_fresh_proof_changed")
    original_report = _technical_original_report(run_dir, proof["original_report"], proof["key"])
    expected = _technical_projection(rows[0], fresh["stage1_vs_ort"], original_report,
        input_context=fresh.get("comparison_input_context"))
    expected["technical_interface_revalidation"] = copy.deepcopy(metadata)
    for entry in metadata.get("central_projections", []):
        expected = _technical_central_projection(run_dir, expected, metadata, entry)
    _technical_require(current == expected, "preserved_measurement_or_quality_changed")
    _technical_require(old is None or old == rows[0], "original_measurement_baseline_changed")
    return rows[0]


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
        observed = current.get(result_key(row))
        if observed != row:
            if not isinstance(observed, Mapping) or not observed.get("technical_interface_revalidation"):
                raise ValueError("supplement_original_measurement_missing_or_changed:" + str(result_key(row)))
            validate_technical_revalidation(run_dir, observed, old=row)
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
    diagnostics = _quality_technical_diagnostic_bindings(run_dir)
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
        if _quality_result_is_technical_diagnostic(run_dir, result, request, diagnostics):
            continue
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
    technical_reference_models = set()
    technical = technical_revalidation_rows(run_dir, generic, source_index, reference_models=technical_reference_models)
    from .status_reporting import energy_attempt_failure
    return {"schema": "onnx-splitpoint/evaluation-supplement-plan", "schema_version": 1,
            "run_dir": str(run_dir), "run_id": run_dir.name,
            "source_plan": "effective_execution_plan.json", "cohort": cohorts,
            "generic_rows": generic, "quality_companion_rows": quality_companions,
            "pending_quality_rows": pending_quality,
            "technical_revalidation_rows": technical,
            "technical_reference_models": sorted(technical_reference_models),
            "native_rows": native, "completed_quality_count": len(quality),
            "completed_quality_requests": [{"path": str(Path(k).relative_to(run_dir)),
                "source_request_sha256": v["source_request_sha256"]} for k,v in requests.items()],
            "counts": {"generic": dict(Counter(r["action"] for r in generic)),
                       "quality_companions": dict(Counter(r["action"] for r in quality_companions)),
                       "technical_revalidation": dict(Counter(r["action"] for r in technical)),
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
    from .runner import ALL_STAGES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="Read-only local admission and inventory; no hardware or compiler calls")
    parser.add_argument("--output-dir", type=Path, help="Required for --plan-only; must be outside the original run")
    parser.add_argument("--stop-after", choices=ALL_STAGES, default=None,
                        help="Stop the normal EvaluationRun continuation after this stage. Invocation control only; --plan-only still inventories the full required scope.")
    parser.add_argument("--import-only", action="store_true", help="Project a saved energy parent result without any new capture")
    parser.add_argument("--resume", action="store_true", help="Continue only missing repeats of a saved energy parent")
    parser.add_argument("--recovery-authorization", type=Path, help="Existing explicit normal resource recovery authorization")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    from .energy_parent_flow import is_energy_parent, run_parent
    if is_energy_parent(run_dir):
        if args.stop_after is not None:
            parser.error("--stop-after requires an EvaluationRun supplement, not an energy parent")
        return run_parent(run_dir,plan_only=args.plan_only,output_dir=args.output_dir,
            import_only=args.import_only,resume=args.resume,recovery_authorization=args.recovery_authorization)
    if args.import_only or args.resume or args.recovery_authorization:
        parser.error("Energy-parent controls require an existing prepared energy parent")
    from .runner import EvaluationWorkflowRunner
    options = options_from_run(run_dir)
    if args.stop_after is not None:
        options.stop_after = args.stop_after
    runner = EvaluationWorkflowRunner(
        options, log=lambda message: print(message, flush=True),
    )
    if args.stop_after is not None:
        runner.log(f"[supplement] stop_after={args.stop_after}; invocation control only; the plan retains the full required scope")
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
