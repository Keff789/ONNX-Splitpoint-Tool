"""Generic-only completion evidence, explicitly without Native pair tables."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np

from .generic_completion_reporting import (
    _csv, _read_result, case_result_path, validate_completion_case,
)
from .generic_only_plan import MODE, validate_quality_reference
from .workflow.checkpoints import atomic_write_json, atomic_write_text


def validate_completion_attestations(observation, row, *, expected_repetitions=3):
    """Verify actual frozen task completion, including the postflight sentinel."""
    from .native_detection_postprocess import (
        DetectionCompletionRuntime,
        _canonical_detection_records,
        verify_detection_completion_execution_contract,
        verify_detection_completion_execution_attestation,
    )
    from .native_three_stage import verify_fast_completion_attestation
    config = row.get("runtime_config", row)
    contract = verify_detection_completion_execution_contract(config["completion_contract"])
    from .runners.task_completion import normalize_stage1_execution
    execution = normalize_stage1_execution(config["stage1_execution"])
    source_map = {item["path"]: item["sha256"] for item in config.get("runtime_source_files", [])}
    root = Path(row["result_path"]).parent.resolve()
    proofs = []
    process_identities = set()
    runtime_instances = set()
    for repetition in observation.get("repetitions", []):
        boot, pid, ticks = (repetition.get(key) for key in ("boot_id", "pid", "process_start_ticks"))
        instance = repetition.get("runtime_instance_id")
        if (not isinstance(boot, str) or not boot or type(pid) is not int or pid <= 0
                or type(ticks) is not int or ticks <= 0
                or not isinstance(instance, str) or not instance
                or repetition.get("execution_id") != config.get("execution_id")
                or not config.get("execution_id")):
            raise ValueError("Generic-only fresh process/execution identity missing or foreign")
        identity = (boot, pid, ticks)
        if identity in process_identities or instance in runtime_instances:
            raise ValueError("Generic-only repetitions share a process or runtime instance")
        process_identities.add(identity)
        runtime_instances.add(instance)
        if (repetition.get("stage1_execution") != execution
                or repetition.get("stage1_peak_uncollected_jobs") != execution["inflight"]):
            raise ValueError("Generic-only actual stage1 asynchronous host peak differs")
        source = repetition.get("completion_source_check") or {}
        before = source.get("before") or {}
        if (source.get("unchanged") is not True or not before or source.get("after") != before
                or any(source_map.get(path) != digest for path, digest in before.items())
                or not any(path.endswith("/native_detection_postprocess.py") for path in before)):
            raise ValueError("Generic-only completion implementation source binding differs")
        raw = repetition.get("completion_execution_attestation") or {}
        count = repetition["warmup_completed_count"] + repetition["completed_task_count"]
        if (count != 1100 or raw.get("completed_work_units") != count
                or raw.get("completion_count") != count
                or repetition.get("completion_attestation_scope") != "last_completed_task_postflight_including_warmup_count"):
            raise ValueError("Generic-only completion attestation task count differs")
        outputs = {name: np.load(root / descriptor["path"], allow_pickle=False)
                   for name, descriptor in repetition["raw_output_samples"].items()}
        if repetition.get("completion_runtime_mode") == "fast_oracle_outside_timing":
            # The shared normal parser has already verified containment and
            # array hashes; replay the canonical oracle over those real bytes.
            attestation = verify_fast_completion_attestation(raw, execution_contract=contract, outputs=outputs)
        elif repetition.get("completion_runtime_mode") in {"strict", "bound_dense"}:
            attestation = verify_detection_completion_execution_attestation(raw,
                execution_contract=contract, expected_observation_relation="same_hotloop_sentinel")
        else:
            raise ValueError("Generic-only completion runtime mode unsupported")
        last = attestation.get("last_result") or {}
        observed = repetition.get("last_completed_result") or {}
        counter_key = "completed_count" if repetition["completion_runtime_mode"] == "fast_oracle_outside_timing" else "invocation_index"
        def semantic(value):
            return {"task": value.get("task"), "contract_family": value.get("contract_family"),
                "coordinate_space": value.get("coordinate_space"), "detection_count": value.get("detection_count"),
                "detections": _canonical_detection_records(value.get("detections"),
                    max_detections=int(contract["processor_contract"]["max_detections"]))}
        replay = DetectionCompletionRuntime(contract).process(outputs)
        if (semantic(observed) != semantic(last) or semantic(observed) != semantic(replay)
                or observed.get(counter_key) != count):
            raise ValueError("Generic-only completion attestation sentinel differs from the actual last task")
        proofs.append({"repetition_index": repetition["repetition_index"],
            "attestation_sha256": attestation["attestation_sha256"], "completed_work_units": count,
            "process_identity": {"boot_id": boot, "pid": pid, "process_start_ticks": ticks},
            "execution_id": repetition["execution_id"]})
    if len(proofs) != expected_repetitions:
        raise ValueError("Generic-only completion attestation count differs")
    return proofs


def generic_only_admission(observation, row):
    config = row.get("runtime_config", row)
    reasons = []
    if config.get("admission_mode") != "generic_only" or config.get("measurement_mode") != MODE:
        reasons.append("explicit_generic_only_mode_missing")
    if observation.get("status") != "completed" or observation.get("valid_repetition_count") != 3:
        reasons.append("three_valid_independent_completed_repetitions_missing")
    if (row.get("runtime_environment_comparison") or {}).get("status") != "observed_no_known_configuration_conflict":
        reasons.append("observed_runtime_configuration_conflict_or_missing")
    if row.get("quality_status") not in {"reference_close", "accuracy_loss", "inconclusive"}:
        reasons.append("completed_cpu_quality_evidence_missing")
    proof = []
    if not reasons:
        try:
            proof = validate_completion_attestations(observation, row)
        except (ValueError, TypeError, KeyError, OSError) as error:
            reasons.append(str(error))
    return {"admission_mode": "generic_only", "case_key": row["case_key"],
        "eligible_for_generic_only_execution": not reasons, "reasons": reasons,
        "completion_attestations": proof,
        "quality_status": row.get("quality_status"),
        "quality_eligible": row.get("quality_status") == "reference_close",
        "native_performance_claimed": False}


def write_completion_reports(plan, results_dir, out_dir):
    if plan.get("scope_mode") != MODE or plan.get("admission_mode") != "generic_only":
        raise ValueError("Generic-only writer requires its separated plan mode")
    cases = plan.get("rows")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Generic-only planned cases missing")
    paths = [case_result_path(row, results_dir).resolve() for row in cases]
    if len(set(paths)) != len(paths):
        raise ValueError("Generic-only identities share result files")
    out = Path(out_dir).resolve()
    observations, admissions = [], []
    for row, path in zip(cases, paths):
        config = row.get("runtime_config", row)
        if config.get("admission_mode") != "generic_only" or any(row.get(key) for key in (
                "native_source", "native_command_contract", "native_uid", "historical_pair")):
            raise ValueError("Generic-only report cannot import a paired/Native row")
        quality = validate_quality_reference(row)
        if (quality["status"] != row.get("quality_status")
                or config.get("quality_reference") != row.get("quality_reference")):
            raise ValueError("Generic-only report Quality differs from its actual CPU evidence/config")
        if out.is_relative_to(Path(row["original_run"]).resolve()):
            raise ValueError("Generic-only reports must preserve the producer run")
        try:
            observation = validate_completion_case(_read_result(path), row, result_root=path.parent)
        except (ValueError, OSError) as error:
            observation = validate_completion_case(None, row)
            observation.update(status="blocked", errors=[str(error)])
        observation.update(result_path=str(path), admission_mode="generic_only",
            quality_status=row.get("quality_status"), quality_reference=row.get("quality_reference"),
            source_selection=row.get("selection_source"),
            quality_eligible=row.get("quality_status") == "reference_close")
        if row.get("execution_error"):
            observation["errors"].append(str(row["execution_error"]))
        if observation["status"] == "unstarted" and row.get("execution_status") in {"failed", "blocked", "cancelled"}:
            observation["status"] = row["execution_status"]
        decision = generic_only_admission(observation, row)
        if observation["status"] == "completed" and not decision["eligible_for_generic_only_execution"]:
            observation.update(status="blocked", parser_measurement_status="completed")
            observation["errors"].extend(decision["reasons"])
        observations.append(observation)
        admissions.append(decision)
    counts = Counter(row["status"] for row in observations)
    summary = {"schema": "onnx-splitpoint/generic-only-completion-summary/v1",
        "admission_mode": "generic_only", "measurement_mode": MODE,
        "planned_cases": len(cases), "planned_repetitions": len(cases) * 3,
        "status_counts": {key: counts.get(key, 0) for key in (
            "completed", "partial", "failed", "blocked", "cancelled", "unstarted")},
        "valid_repetitions": sum(row["valid_repetition_count"] for row in observations),
        "completed_task_count": sum(row["completed_task_count"] for row in observations),
        "functional_admissions": sum(row["eligible_for_generic_only_execution"] for row in admissions),
        "quality_status_counts": dict(Counter(row.get("quality_status") for row in cases)),
        "all_cases_measured": counts.get("completed", 0) == len(cases),
        "native_performance_claimed": False, "energy_measurement_performed": False,
        "analysis_scope": "fixed_generic_only_candidates"}
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "generic_only_observations.json", {"rows": observations})
    atomic_write_json(out / "generic_only_admissions.json", {"rows": admissions})
    atomic_write_json(out / "generic_only_summary.json", summary)
    _csv(out / "generic_only_observations.csv", observations)
    _csv(out / "generic_only_raw_repetitions.csv", [
        {"case_key": row["case_key"], "result_path": row["result_path"], **repetition}
        for row in observations for repetition in row["raw_attempts"]])
    return summary


def write_status_report(plan, summary, output, *, status, error=""):
    text = ["Generic-only completed-task measurement", "Status: " + status,
        "Execution: " + plan["execution_id"],
        f"Cases: {summary['status_counts']}; repetitions: {summary['valid_repetitions']}/{summary['planned_repetitions']}",
        f"Measured completed tasks: {summary['completed_task_count']}",
        "Reference: serial execution of the same compiled parts, outside throughput timing.",
        "Task Quality: the separately bound normal CPU-ONNX dataset comparison; quality FAILs remain results.",
        "No Native performance/selection claim or energy measurement is produced.",
        "Reports: reports/generic_only_observations.json and generic_only_summary.json."]
    if error:
        text.append("Stopped without automatic retry: " + error)
    atomic_write_text(Path(output) / "STATUS.txt", "\n".join(text) + "\n")
