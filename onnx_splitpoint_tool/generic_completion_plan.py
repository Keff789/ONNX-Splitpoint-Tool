"""Read-only selection of the already paired Generic completed-task supplement.

The input CSV is the selection authority.  All runtime options and artifact
identities come from the actual Generic report and its paired Native command;
this module neither discovers new candidates nor dispatches work.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .native_command_contract import verify_native_command_contract
from .native_performance_identity import canonical_native_backend
from .native_split_quality import validate_native_split_quality_binding
from .workflow.artifacts import sha256_file


IDENTITY_FIELDS = (
    "model_id", "case_id", "direction", "precision", "setup_id",
    "comparison_backend", "output_endpoint_id",
)
MODEL_COUNTS = {
    "mobilenet_v3_large": 17, "resnet50": 19, "regnet_x_1_6gf": 20,
    "yolo11l": 2, "yolo26m": 3, "yolo26s": 2, "yolov7_paper": 1,
}
SETUPS = {
    "orin_nx_hailo8_01": "hailo8", "orin_nx_hailo10_01": "hailo10h",
    "orin_nx_deepx_m1_01": "deepx",
}


def completion_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Keep the existing seven F04 dimensions; canonicalize backend aliases."""
    values = {field: str(row.get(field) or "").strip() for field in IDENTITY_FIELDS}
    if not all(values.values()):
        raise ValueError("generic_completion_identity_incomplete")
    backend = canonical_native_backend(values["direction"])
    comparison = canonical_native_backend(values["comparison_backend"])
    if backend not in SETUPS.values() or backend != comparison:
        raise ValueError("generic_completion_backend_identity_conflict")
    if not values["case_id"].startswith("b") or not values["case_id"][1:].isdigit():
        raise ValueError("generic_completion_requires_selected_split")
    values["direction"] = {"deepx": "deepx_to_trt", "hailo8": "hailo8_to_trt", "hailo10h": "hailo10h_to_trt"}[backend]
    values["comparison_backend"] = comparison
    return tuple(values[field] for field in IDENTITY_FIELDS)


def validate_completion_scope(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [completion_identity(row) for row in rows]
    if len(keys) != 192 or len(set(keys)) != 192:
        raise ValueError("generic_completion_scope_requires_192_unique_identities")
    expected = Counter({(model, setup): count for model, count in MODEL_COUNTS.items() for setup in SETUPS})
    if Counter((row["model_id"], row["setup_id"]) for row in rows) != expected:
        raise ValueError("generic_completion_scope_model_setup_counts_drift")
    for key in keys:
        if key[5] != SETUPS[key[4]]:
            raise ValueError("generic_completion_scope_setup_backend_conflict")


def select_start_cases(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Cover the three detection contracts, never select by measured speed."""
    detection_models = {
        "orin_nx_hailo8_01": "yolo11l", "orin_nx_hailo10_01": "yolo26m",
        "orin_nx_deepx_m1_01": "yolov7_paper",
    }
    selected = []
    for setup in sorted(SETUPS):
        for task in ("classification", "detection"):
            candidates = [r for r in rows if r["setup_id"] == setup and r["task"] == task]
            if task == "detection":
                candidates = [r for r in candidates if r["model_id"] == detection_models[setup]]
            if not candidates:
                raise ValueError("generic_completion_start_task_missing")
            selected.append(min(candidates, key=lambda r: (r["model_id"], r["case_id"], r["case_key"]))["case_key"])
    return selected


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_summary_context(native: Mapping[str, Any], validation: Mapping[str, Any], run: Path) -> None:
    authority = native.get("native_split_quality_authority") or {}
    previous = validation.get("native_split_quality_authority") or {}
    fields = ("run_id", "stage_run_id", "profile_start_snapshot_sha256", "profile_selection_fingerprint")
    if authority.get("valid") is not True or authority.get("run_id") != run.name:
        raise ValueError("generic_completion_corrected_native_authority_invalid")
    if previous.get("valid") is not True or validation.get("diagnostic_only") is not False or any(
            not authority.get(key) or previous.get(key) != authority[key] for key in fields):
        raise ValueError("generic_completion_corrected_validation_context_conflict")


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _inside(path: str | Path, root: Path) -> Path:
    result = Path(path)
    if not result.is_absolute():
        result = root / result
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("generic_completion_original_path_escapes_run")
    return result


def _native_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("model") or row.get("model_id") or ""),
        str(row.get("case") or row.get("case_id") or ""),
        str(row.get("setup_id") or ""),
        str(row.get("precision") or ""),
        canonical_native_backend(row.get("backend") or row.get("direction")),
        canonical_native_backend(row.get("comparison_backend")),
    )


def _one(rows: Sequence[Mapping[str, Any]], key: tuple[str, ...], role: str) -> dict[str, Any]:
    matches = [dict(row) for row in rows if _native_key(row) == key]
    if len(matches) != 1:
        raise ValueError(f"generic_completion_{role}_identity_not_unique:{key}")
    return matches[0]


def _native_local_path(remote: str, command: Mapping[str, Any], native: Mapping[str, Any], run: Path) -> Path:
    remote_path = Path(remote)
    try:
        relative = remote_path.relative_to(str(command["benchmark_set"]))
    except ValueError as exc:
        raise ValueError("generic_completion_native_artifact_outside_bound_benchmark_set") from exc
    local_set = Path(str(native["source_root"])) / str(native["model"]) / "benchmark_set"
    return _inside(local_set / relative, run)


def _bound_file(path: Path, expected: Mapping[str, Any], role: str) -> None:
    if not path.is_file() or not expected.get("sha256"):
        raise ValueError(f"generic_completion_{role}_bound_file_missing:{path}")
    if str(sha256_file(path) or "").removeprefix("sha256:") != str(expected["sha256"]).removeprefix("sha256:"):
        raise ValueError(f"generic_completion_{role}_file_hash_mismatch:{path}")
    if expected.get("size_bytes") is not None and path.stat().st_size != expected["size_bytes"]:
        raise ValueError(f"generic_completion_{role}_file_size_mismatch:{path}")


def _input_binding(command: Mapping[str, Any], native: Mapping[str, Any], run: Path) -> dict[str, Any]:
    prepared = dict(command["prepared_input_contract"])
    artifacts = command["artifacts"]
    entries = prepared.get("entries") or [{
        "name": prepared.get("name") or ("images" if native["task"] == "detection" else "input"),
        "artifact_name": "prepared_input", "shape": prepared.get("shape"), "dtype": prepared.get("dtype"),
    }]
    bound = []
    for entry in entries:
        artifact = dict(artifacts[entry["artifact_name"]])
        local = _native_local_path(artifact["path"], command, native, run)
        _bound_file(local, artifact, "prepared_input")
        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[-1] != 3 or entry.get("dtype") != "uint8":
            raise ValueError("generic_completion_prepared_runtime_input_contract_unsupported")
        bound.append({**entry, **artifact, "local_path": str(local), "remote_path": artifact["path"]})
    if len(bound) != 1 or prepared.get("source_image_sha256") != command.get("input_image_sha256"):
        raise ValueError("generic_completion_prepared_input_source_conflict")
    return {
        "prepared_input_contract": prepared, "entries": bound,
        "source_image": command["input_image"], "source_image_sha256": command["input_image_sha256"],
        "sequence": "single_exact_prepared_feed_repeated", "batch_size": 1,
        "storage_at_timer_start": "host_memory_prepared_runtime_tensor",
        "preprocessing_timed": False, "runtime_adapter_must_preserve_exact_prepared_bytes": True,
        "source": native["report"],
    }


def _generic_artifacts(generic: Mapping[str, Any], command: Mapping[str, Any], local_case: Path, run: Path) -> dict[str, Any]:
    """Cross-check actual Generic sessions, never substitute a Native runner."""
    trt = ((generic.get("native_tensorrt") or {}).get("sessions") or {}).get("part2:tensorrt") or {}
    artifacts = command["artifacts"]
    for field, role, hash_field in (("engine", "engine", "engine_sha256"), ("source_model", "build_part2_onnx", "source_model_sha256")):
        if trt.get(field) != artifacts[role]["path"] or trt.get(hash_field) != artifacts[role]["sha256"]:
            raise ValueError(f"generic_completion_generic_native_{role}_conflict")
    if trt.get("precision") != command["precision"] or trt.get("fallback_disabled") is not True:
        raise ValueError("generic_completion_generic_precision_or_fallback_conflict")
    backend = canonical_native_backend(command["backend"])
    if backend == "deepx":
        relative = generic.get("deepx_part1_dxnn")
        p1 = artifacts["dxnn"]
        if (generic.get("deepx") or {}).get("part1_dxnn", {}).get("dxnn_sha256") != p1["sha256"]:
            raise ValueError("generic_completion_generic_native_dxnn_conflict")
    else:
        arch = "hailo10" if backend == "hailo10h" else "hailo8"
        relative = ((generic.get("hailo") or {}).get("hefs") or {}).get(arch, {}).get("part1")
        p1 = artifacts["hef"]
    if not relative:
        raise ValueError("generic_completion_generic_part1_path_missing")
    local_p1 = _inside(local_case / relative, run)
    _bound_file(local_p1, p1, "generic_part1")
    result = {}
    for name in ("engine", "source_part2_onnx", "build_part2_onnx", "native_trt_meta", "boundary_metadata", "engine_build_receipt"):
        result[name] = {**artifacts[name], "remote_path": artifacts[name]["path"], "local_path": None}
    result["part1_runtime"] = {**p1, "remote_path": p1["path"], "local_path": str(local_p1), "generic_relative_path": str(relative)}
    # The ordinary Generic source ONNX is an available exact restaging source.
    source_p2 = _inside(local_case / str(generic["part2"]), run)
    _bound_file(source_p2, artifacts["source_part2_onnx"], "source_part2_onnx")
    result["source_part2_onnx"]["local_path"] = str(source_p2)
    return result


def _quality_request_binding(generic: Mapping[str, Any], native: Mapping[str, Any], source: Path, run: Path) -> tuple[Path, dict[str, Any]]:
    request = (generic.get("task_quality_input_requests_by_variant") or {}).get("composed") or {}
    descriptor = request.get("request") or {}
    expected = {
        "eval_run_id": run.name, "model_id": native["model"], "case_id": native["case"],
        "setup_id": native["setup_id"], "runtime_precision_identity": native["precision"],
    }
    if any(request.get(k) != v for k, v in expected.items()) or canonical_native_backend(request.get("backend")) != canonical_native_backend(native["backend"]):
        raise ValueError("generic_completion_original_quality_request_identity_conflict")
    if descriptor.get("sha256") != native.get("source_request_sha256"):
        raise ValueError("generic_completion_original_quality_request_hash_conflict")
    local = _inside(source.parent / "quality_inputs" / native["setup_id"] / "results" / native["case"] /
        str(generic["run_cfg"]["out_dir"]) / "task_quality_inputs" / "composed_request.json", run)
    _bound_file(local, descriptor, "original_quality_request")
    if _read(local) != {key: value for key, value in request.items() if key != "request"}:
        raise ValueError("generic_completion_original_quality_request_projection_conflict")
    binding, status = validate_native_split_quality_binding(request.get("native_split_quality_binding"), expected_identity={
        "model_id": native["model"], "case_id": native["case"], "setup_id": native["setup_id"],
        "backend": native["backend"], "precision": native["precision"]})
    if binding is None or binding["binding_sha256"] != request.get("native_split_quality_binding_sha256"):
        raise ValueError("generic_completion_original_quality_binding_invalid:" + status)
    selected = native["native_command_contract"]["native_split_quality_binding"]
    if binding["binding_sha256"] != selected.get("producer_binding_sha256", selected.get("binding_sha256")):
        raise ValueError("generic_completion_original_quality_producer_binding_conflict")
    return local, binding


def _boundaries(native: Mapping[str, Any], original: Mapping[str, Any], command: Mapping[str, Any], input_binding: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    options = command["runtime_options"]
    for key, expected in (("frames", 1000), ("warmup", 100), ("repetitions", 3)):
        if type(options.get(key)) is not int or options[key] != expected:
            raise ValueError(f"generic_completion_native_{key}_contract_conflict")
    counts = native.get("completed_task_work_unit_counts")
    times = native.get("completed_task_measurement_times_s")
    if counts != [1000, 1000, 1000] or not isinstance(times, list) or len(times) != 3 or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("generic_completion_native_completed_count_time_invalid")
    if original.get("measurement_boundary") != "workers_ready_to_last_completed_task_frame" or original.get("warmup_contract") != "fully_drained_before_worker_start":
        raise ValueError("generic_completion_native_start_end_or_warmup_boundary_conflict")
    completion = native.get("completion_execution_contract") or {}
    mode = completion.get("completion_mode") or "classification_top1_top5"
    common = {
        "task": native["task"], "completion_mode": mode,
        "input_sequence": input_binding["sequence"], "batch_size": 1,
        "timer_start": "workers_ready_before_first_prepared_input_admission",
        "timer_end": "last_required_task_completion_after_synchronized_host_output_and_task_postprocess",
        "preprocessing_included": False, "host_device_transfers_included": True,
        "handoff_included": True, "output_materialization_included": True,
        "pipeline_fill_included": True, "pipeline_drain_included": True,
        "warmup_count": 100, "warmup_fully_completed_and_drained_before_timer": True,
        "measured_task_count": 1000, "repetitions": 3,
        "completion_output_endpoint_id": native.get("completed_task_comparison_output_endpoint_id") or native["output_endpoint_id"],
    }
    native_boundary = {**common, "source": native["report"], "native_command_contract_sha256": command["contract_sha256"],
        "execution_model": command["runner"], "queue_depth": options.get("queue_depth"), "inflight": options.get("inflight"),
        "output_materialization_policy": original.get("trt_output_materialization_policy"),
        "observed_completed_counts": counts, "observed_measurement_times_s": times}
    generic_boundary = {**common, "source": "Generic existing P1/P2 adapters and streaming pipeline with completion hook",
        "execution_model": "generic_existing_stage1_stage2_streaming", "queue_depth_source": "original_generic_run_cfg.throughput_queue_depth",
        "native_queue_or_inflight_not_copied": True}
    comparison = {"status": "bound_plan_requires_runtime_confirmation", "checks": {
        "input_sequence_matches": True, "batch_matches": True, "prepared_input_start_matches": True,
        "completion_endpoint_matches": True, "fill_and_drain_matches": True,
        "transfer_and_materialization_scope_matches": True},
        "sources": {"native_source": native["report"], "generic_source": generic_boundary["source"], "prepared_input": input_binding["entries"]},
        "runtime_confirmation_required": True}
    return generic_boundary, native_boundary, comparison


def build_completion_plan(scope_csv: str | Path, native_summary: str | Path, validation_summary: str | Path,
                          original_run: str | Path, *, pair_csv: str | Path | None = None) -> dict[str, Any]:
    """Bind exactly the selected 192 rows without any remote call or write."""
    scope_path = Path(scope_csv); run = Path(original_run)
    scope = _csv(scope_path); validate_completion_scope(scope)
    pairs = _csv(Path(pair_csv) if pair_csv else scope_path.with_name("F04_cross_runner_pairs_192.csv"))
    pair_index = {completion_identity(row): row for row in pairs}
    if len(pair_index) != 192 or set(pair_index) != {completion_identity(row) for row in scope}:
        raise ValueError("generic_completion_scope_pair_identity_drift")
    result = bind_completion_rows(scope, pairs, native_summary, validation_summary, run,
        scope_source=str(scope_path), start_selection=select_start_cases)
    result.update(selection_policy="exact_existing_F04_192_no_quality_filter",
        start_selection_policy="lexical_classification_and_predetermined_detection_contract_coverage_per_setup")
    return result


def bind_completion_rows(scope, pairs, native_summary, validation_summary, run, *,
                         scope_source, start_selection):
    """Bind an already authorized exact cohort through the same normal contracts.

    Scope authorization belongs to the caller. The historical public entry
    keeps its exact 192 guards; this reader does not dispatch or select cases.
    """
    run = Path(run)
    keys = [completion_identity(row) for row in scope]
    pair_index = {completion_identity(row): row for row in pairs}
    if not keys or len(keys) != len(set(keys)) or set(pair_index) != set(keys):
        raise ValueError("generic_completion_scope_pair_identity_drift")
    native_data = _read(Path(native_summary)); validation_data = _read(Path(validation_summary))
    _validate_summary_context(native_data, validation_data, run)
    native_fields = {
        "model", "case", "backend", "setup_id", "comparison_backend", "precision", "status", "task", "report", "source_root",
        "native_command_contract", "native_command_contract_sha256", "source_request_sha256", "native_split_quality_authority_valid",
        "native_split_quality_binding_sha256", "output_endpoint_id", "completed_task_comparison_output_endpoint_id",
        "completed_task_work_unit_counts", "completed_task_measurement_times_s", "completion_execution_contract",
        "input_image_sha256", "model_sha256", "native_split_semantic_output_manifest", "quality_gate_status", "completed_task_result_artifact_path",
    }
    native_rows = [{key: value for key, value in row.items() if key in native_fields} for row in native_data.pop("rows")]
    validation_fields = {"model", "case", "backend", "setup_id", "precision", "comparison_backend", "semantic_ok",
        "native_command_contract_sha256", "input_image_sha256", "model_sha256", "source_request_sha256"}
    validation_rows = [{key: value for key, value in row.items() if key in validation_fields}
                       for row in (validation_data.pop("rows", None) or validation_data.pop("results", None) or [])]
    generic_files: dict[str, Any] = {}
    rows = []
    for selected in scope:
        ident = completion_identity(selected)
        match_key = (ident[0], ident[1], ident[4], ident[3], ident[5], ident[5])
        native = _one(native_rows, match_key, "native")
        validation = _one(validation_rows, match_key, "validation")
        if any(not native.get(key) or validation.get(key) != native[key] for key in
                ("native_command_contract_sha256", "input_image_sha256", "model_sha256", "source_request_sha256")):
            raise ValueError("generic_completion_validation_original_binding_conflict")
        if native.get("status") != "ok" or native.get("native_split_quality_authority_valid") is not True:
            raise ValueError("generic_completion_native_measurement_not_valid")
        if native.get("output_endpoint_id") != selected["output_endpoint_id"]:
            raise ValueError("generic_completion_selected_endpoint_drift")
        command, status = verify_native_command_contract(native.get("native_command_contract"), expected_identity={
            "model": ident[0], "case": ident[1], "setup_id": ident[4], "comparison_backend": native["comparison_backend"], "precision": ident[3]})
        if command is None or command.get("contract_sha256") != native.get("native_command_contract_sha256"):
            raise ValueError("generic_completion_native_command_invalid:" + status)
        if command.get("eval_run_id") != run.name or command.get("source_request_sha256") != native.get("source_request_sha256"):
            raise ValueError("generic_completion_native_original_run_or_request_conflict")
        source = _inside(selected["original_source"], run)
        if str(source) not in generic_files:
            generic_fields = {"case_id", "setup_id", "run_id", "run_cfg", "plan", "generic_runtime_source_binding",
                "native_tensorrt", "deepx_part1_dxnn", "deepx", "hailo", "part2", "task_quality_input_requests_by_variant"}
            generic_files[str(source)] = [{key: value for key, value in row.items() if key in generic_fields} for row in _read(source)]
        matches = [r for r in generic_files[str(source)] if r.get("case_id") == selected["case_id"]]
        if len(matches) != 1:
            raise ValueError("generic_completion_generic_original_not_unique")
        generic = matches[0]; cfg = generic["run_cfg"]
        if generic.get("setup_id") != ident[4] or cfg.get("quality_evidence_eval_id") != run.name or cfg.get("model_id") != ident[0] or canonical_native_backend(generic.get("run_id")) != ident[5]:
            raise ValueError("generic_completion_generic_original_identity_conflict")
        original_request, generic_quality_binding = _quality_request_binding(generic, native, source, run)
        local_case = _inside(Path(generic["plan"]["legacy_suite_dir"]) / ident[1], run)
        remote_runner = Path(generic["generic_runtime_source_binding"]["generated_runner"])
        original_report = _inside(native["report"], run)
        original = _read(original_report)
        if original.get("native_command_contract") != native["native_command_contract"]:
            raise ValueError("generic_completion_original_native_command_conflict")
        input_binding = _input_binding(command, native, run)
        artifacts = _generic_artifacts(generic, command, local_case, run)
        generic_boundary, native_boundary, comparison = _boundaries(native, original, command, input_binding)
        task = native["task"]
        completion = native.get("completion_execution_contract") or {"task": "classification", "operation": "top1_top5", "label_offset": 0}
        model_roles = [r for r in generic["plan"]["model_suite"]["primary"] if r["id"] == ident[0]]
        if len(model_roles) != 1:
            raise ValueError("generic_completion_original_model_role_missing")
        reference = {"source_report": str(original_report), "output_manifest": native.get("native_split_semantic_output_manifest"),
            "input_image_sha256": native["input_image_sha256"], "model_sha256": native["model_sha256"],
            "classification_topk": original.get("classification_topk"), "validation_source": str(validation_summary),
            "validation_identity": {key: validation.get(key) for key in ("model", "case", "backend", "setup_id", "precision", "comparison_backend")},
            "historical_semantic_ok": validation.get("semantic_ok"), "quality_gate_status": native.get("quality_gate_status")}
        output_manifest_path = _inside(reference["output_manifest"], run)
        _bound_file(output_manifest_path, command["artifacts"]["semantic_output_manifest"], "native_output_manifest")
        output_manifest = _read(output_manifest_path)
        if (output_manifest.get("input_image") != input_binding["source_image"] or
                output_manifest.get("input_image_sha256") not in (None, input_binding["source_image_sha256"])):
            raise ValueError("generic_completion_reference_input_binding_conflict")
        reference["output_manifest_sha256"] = command["artifacts"]["semantic_output_manifest"]["sha256"]
        reference["outputs"] = []
        for entry in output_manifest.get("outputs") or []:
            local_output = _inside(output_manifest_path.parent / str(entry["file"]), output_manifest_path.parent)
            _bound_file(local_output, entry, "native_reference_output")
            reference["outputs"].append({**entry, "path": str(local_output), "local_path": str(local_output), "format": "raw"})
        if not reference["outputs"]:
            raise ValueError("generic_completion_native_reference_outputs_missing")
        if task == "detection":
            if not completion.get("contract_sha256") or completion != command["runtime_options"].get("completion_execution_contract"):
                raise ValueError("generic_completion_detection_execution_contract_conflict")
            artifact_path = _native_local_path(native["completed_task_result_artifact_path"], command, native, run)
            expected = {"sha256": original.get("completed_task_result_artifact_file_sha256")}
            _bound_file(artifact_path, expected, "completed_result")
            reference.update(completed_result_path=str(artifact_path), completed_result_file_sha256=expected["sha256"])
        case_key = "|".join(ident)
        rows.append({**dict(zip(IDENTITY_FIELDS, ident)), "case_key": case_key,
            "identity": dict(zip(IDENTITY_FIELDS, ident)), "task": task,
            "case_output_relative": f"{ident[4]}/{ident[0]}/{ident[1]}",
            "original_run": str(run), "generic_raw_source": str(source), "native_source": str(original_report),
            "historical_pair": pair_index[ident], "original_generic_identity": {k: selected[k] for k in IDENTITY_FIELDS},
            "original_run_cfg": cfg, "original_local_case_dir": str(local_case), "local_case_dir": str(local_case),
            "original_manifest": str(local_case / "split_manifest.json"), "model_sha256": native["model_sha256"],
            "original_remote_case_dir": str(remote_runner.parent), "original_remote_runner": str(remote_runner),
            "remote_python_executable": command["python_executable"],
            "native_command_contract_sha256": command["contract_sha256"],
            "native_command_contract": {k: command[k] for k in ("backend", "model", "case", "setup_id", "comparison_backend", "precision", "benchmark_set", "python_executable", "runner", "artifacts", "prepared_input_contract", "boundary_contract", "runtime_options", "eval_run_id", "source_request_sha256", "contract_sha256")},
            "native_quality_binding_source": cfg.get("native_split_quality_binding"),
            "native_quality_binding_sha256": native.get("native_split_quality_binding_sha256"),
            "generic_original_quality_request": str(original_request),
            "generic_native_split_quality_binding": generic_quality_binding,
            "expected_runtime_artifacts": artifacts, "input_binding": input_binding,
            "completion_contract": completion, "functional_reference": reference,
            "measurement_boundary": generic_boundary, "native_measurement_boundary": native_boundary,
            "measurement_boundary_comparison": comparison,
            "runtime_environment_comparison": {"status": "not_observed", "reasons": ["current_device_snapshot_required"],
                "historical_sources": [str(path) for phase in ("pre", "post")
                    if (path := Path(native["source_root"]) / "host_telemetry" / f"{ident[5]}_performance_{phase}.json").is_file()]},
            "evaluation_role": model_roles[0].get("evaluation_role"), "evaluation_role_source": str(source) + "#plan.model_suite.primary",
            "warmup_count": 100, "task_count": 1000, "repetitions": 3,
            "build_allowed": False, "fallback_allowed": False,
        })
    starts = start_selection(rows)
    for row in rows:
        row["start_case"] = row["case_key"] in starts
    boundaries = {}
    for row in rows:
        mode = row["completion_contract"].get("completion_mode") or "classification_top1_top5"
        boundary_class = "|".join((row["comparison_backend"], row["task"], mode))
        row["boundary_class"] = boundary_class
        item = boundaries.setdefault(boundary_class, {"boundary_class": boundary_class, "case_count": 0,
            "representative_case_key": row["case_key"], "generic": row["measurement_boundary"],
            "native": row["native_measurement_boundary"], "input_binding": row["input_binding"],
            "completion_contract": row["completion_contract"]})
        item["case_count"] += 1
    return {"schema": "onnx-splitpoint/generic-completed-task-plan", "schema_version": 1,
        "original_run": str(run), "scope_source": str(scope_source), "native_summary_source": str(native_summary),
        "validation_summary_source": str(validation_summary), "row_count": len(rows),
        "selection_policy": "caller_authorized_exact_scope_no_quality_filter", "start_selection_policy": "caller_bound_startcases",
        "start_case_keys": starts, "rows": rows, "measurement_boundaries": list(boundaries.values()),
        "planned_repetitions": 3 * len(rows), "planned_measured_completions": 3000 * len(rows)}
