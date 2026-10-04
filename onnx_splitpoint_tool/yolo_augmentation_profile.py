"""Materialize the small augmentation through the normal profile resolver.

Only a new explicit profile and its ordinary start snapshot are written. This
helper does not construct a workflow, generate a suite, compile or infer.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import fields
from pathlib import Path

from .benchmark.evaluation_profiles import (
    load_evaluation_profile, resolve_evaluation_profile, save_evaluation_profile_yaml,
)
from .workflow.selection_request_counts import requested_cases_for_model
from .workflow.contracts import WorkflowOptions
from .workflow.start_snapshot import resolve_runtime_profile_start_snapshot
from .yolo_augmentation_plan import COHORT_ID


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_original_contract(profile):
    """Refuse a drifted source contract instead of silently repairing its recipe."""
    expected = {
        "hailo_build": {"preset": "balanced", "optimization_level": 1, "calib_count": 500, "calib_batch_size": 8, "force_build": False},
        "deepx_build": {"calibration_method": "ema", "opt_level": 0, "calib_count": 500, "force_build": False},
        "native_producers": {"frames": 1000, "warmup": 100, "repetitions": 3, "queue_depth": 3, "inflight": 8},
    }
    for section, values in expected.items():
        for key, value in values.items():
            actual = profile.get(section, {}).get(key)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"original augmentation contract mismatch: {section}.{key}")
    statistics = profile["quality_gate"]["statistics"]
    for key, value in {"bootstrap_repetitions": 1000, "seed": 20260710, "workers": 4, "max_active_requests": 1}.items():
        if type(statistics.get(key)) is not int or statistics[key] != value:
            raise ValueError(f"original augmentation contract mismatch: quality_gate.statistics.{key}")
    if profile["validation_execution"]["max_items"]["detection"] != 5000:
        raise ValueError("original augmentation contract mismatch: detection N5000")
    energy = profile["native_producers"]["energy"]
    if (energy.get("duration_s") != 60 or energy.get("physical_scope") != "FS"
            or energy.get("window_label") != "command"
            or energy.get("task_budget") != {"enabled": True, "max_retries": 2, "max_transport_failures": 20}):
        raise ValueError("original augmentation contract mismatch: Native energy FS60/retries2/source20")


def build_augmentation_profile(plan: dict, output_root: str | Path) -> dict:
    """Keep the original resolved recipes; narrow only the additive work scope."""
    run = Path(plan["original_run"]).resolve()
    completion = Path(plan["base_completion_root"]).resolve()
    out = Path(output_root).resolve()
    if any(out == source or source in out.parents for source in (run, completion)):
        raise ValueError("augmentation profile output must be outside preserved bases")
    if plan.get("cohort_id") != COHORT_ID:
        raise ValueError("augmentation profile cohort mismatch")
    original_options = _read(run / "run_manifest.json")["options"]
    original_snapshot = original_options["profile_start_snapshot"]
    original = original_snapshot["resolved_profile"]
    _check_original_contract(original)
    profile = copy.deepcopy(original)
    # Already resolved values are authoritative. Re-applying an old preset
    # would re-enable its old Full builds and drift the explicitly narrowed work.
    profile.pop("execution_preset", None)
    namespace = re.sub(r"[^a-zA-Z0-9_.-]", "_", out.parent.name if out.name == "plan" else out.name)
    profile["name"] = namespace
    profile["purpose"] = "Post-planned capability/graph augmentation; original THESIS20 and completed192 remain unchanged; no hold-out claim."
    profile["implementation_note"] = "Explicit resolved original profile with a new cohort; no current global run-mode defaults applied."
    selected = {model: value["primary_cases"] for model, value in plan["models"].items() if value["primary_cases"]}
    if sum(map(len, selected.values())) != 4 or set(selected) != {"yolo11l", "yolo26s", "yolov7_paper"}:
        raise ValueError("augmentation profile requires the bounded four primary boundaries")
    suite = []
    for entry in original["model_suite"]["primary"]:
        model = entry["id"]
        if model not in selected:
            continue
        row = copy.deepcopy(entry)
        row.update(onnx=plan["models"][model]["source_graph"],
                   model_sha256=plan["models"][model]["model_sha256"],
                   evaluation_role="development", generalization_scope="development",
                   candidate_universe={"mode": "declared_shortlist", "seed": 20260710})
        if "resolved_path" in row:
            row["resolved_path"] = row["onnx"]
        suite.append(row)
    if len(suite) != 3:
        raise ValueError("augmentation models missing from original profile")
    profile["model_suite"]["primary"] = suite
    profile["model_suite"]["reserve"] = []
    profile["run_profiles"] = [copy.deepcopy(r) for r in original["run_profiles"] if r.get("type") == "mixed_backend"
        and r.get("stage1") in {"hailo8", "hailo10", "hailo10h", "deepx_m1"} and r.get("stage2") == "tensorrt"]
    if len(profile["run_profiles"]) != 3:
        raise ValueError("augmentation requires three existing mixed run profiles")
    for run_profile in profile["run_profiles"]:
        run_profile["variants"] = ["composed"]
    policy = profile["selection_policy"]
    policy.update(forced_cases=selected, selection_strategy="predicted_topk", require_single_part2_input=True,
                  score_independent_audit_enabled=False, max_accepted_cases_per_model=2, preferred_shortlist=2,
                  min_gap=1, full_model_hailo_preflight_policy="skip")
    policy["backend_backfill"]["enabled"] = False
    profile["campaign"].update(id=namespace, mode="development", claim_scope="evaluated_matrix",
        frozen_before_final_campaign=False, require_campaign_freeze=False, prediction_freeze_enabled=False)
    profile["hailo_build"].update(build_full=False, build_part1=True, build_part2=False,
        deferred_full_baseline_required=False, full_baseline_cold_build_policy="cache_or_defer", force_build=False)
    profile["deepx_build"]["force_build"] = False
    profile["build_scheduler"]["prefetch_deepx_full"] = False
    profile.setdefault("workflow_execution", {})["native_release_mode"] = "per_case"
    native = profile["native_producers"]
    native["full_baselines"] = {"enabled": False, "backends": [], "backends_by_producer": {},
                               "auto_enabled": False, "source": "augmentation_split_only"}
    native.update(enabled=True, models=list(selected), case_policy="case_map_only", case_map=copy.deepcopy(selected),
                  frames=1000, warmup=100, repetitions=3,
                  remote_root=f"/home/nx/native_fifo_evalsets/{namespace}",
                  remote_tool_dir=f"/home/nx/splitpoint_runs/{namespace}/tool")
    native["energy"].update(enabled=True, mode="measure", include_full_baselines=False, include_split_rows=True)
    native["energy"]["window_method_validation_probe"]["enabled"] = False
    profile["energy"].update(enabled=False, generic_enabled=False, requested_native_energy=True)
    profile["workflow"].update(include_reserve=False, skip_runtime_benchmarks=False)
    profile["remote_cache"]["suite_cache_root"] = f"~/.cache/onnx_splitpoint/suites/{namespace}"
    # No shared artifact-cache root, compilation recipe, dataset, precision,
    # quality margin, statistics setting, device setting or energy calibration
    # is modified. These exact dictionaries are audited after normal resolution.
    required = {
        "quality_gate": original["quality_gate"], "validation_execution": original["validation_execution"],
        "official_coco_evaluation": original["official_coco_evaluation"], "hardware": original["hardware"],
    }
    out.mkdir(parents=True, exist_ok=True)
    path = save_evaluation_profile_yaml(out / "augmentation-profile.yaml", profile)
    loaded = load_evaluation_profile(str(path))
    resolved = loaded.raw_profile
    for field, expected in required.items():
        if resolved[field] != expected:
            raise ValueError(f"augmentation resolved profile changed preserved contract: {field}")
    if resolved != profile:
        raise ValueError("augmentation explicit profile drifted during resolution")
    model_checks = {}
    for entry in suite:
        result = resolve_evaluation_profile(str(path), model_path=entry["onnx"])
        if not result.matched or result.overrides.get("acc_cpu") or result.overrides.get("hailo_custom_full"):
            raise ValueError("augmentation profile enabled a CPU/Full benchmark")
        model_checks[entry["id"]] = {"matched": result.matched, "run_profile_ids": result.overrides["run_profile_ids"],
            "acc_cpu": result.overrides["acc_cpu"], "hailo_custom_full": result.overrides["hailo_custom_full"],
            "requested_cases": requested_cases_for_model(policy, entry["id"], default=2)}
    snapshot_path = out / "augmentation-profile-start-snapshot.json"
    augmentation_run = Path(plan["augmentation_run"]).resolve()
    if any(augmentation_run == source or source in augmentation_run.parents for source in (run, completion)):
        raise ValueError("augmentation run must be outside preserved bases")
    allowed_options = {field.name for field in fields(WorkflowOptions)}
    options = {key: copy.deepcopy(value) for key, value in original_options.items() if key in allowed_options}
    options.update(profile=str(path), out=str(augmentation_run.parent), run_id=augmentation_run.name,
        resume=False, dry_run=False, only_model=None, max_models=None, stop_after=None,
        execution_mode="generate_and_run", artifact_policy="normal", include_reserve=False,
        result_sources=[], benchmark_sets_root="", benchmark_results_root="", force_stage=[],
        rerun_generated_only=False, resume_missing=False, resume_missing_full_quality_only=False,
        skip_analysis=False, skip_benchmarks=False, no_remote=False,
        hailo_build_full=False, hailo_build_part1=True, hailo_build_part2=False, hailo_force_build=False,
        native_producer_enabled=True, native_producer_model_case_map="", native_producer_case_policy="case_map_only",
        native_producer_remote_root=native["remote_root"], native_producer_remote_tool_dir=native["remote_tool_dir"],
        native_producer_energy_enabled=True, native_producer_energy_mode="measure",
        window_method_validation_probe_enabled=False,
        remote_base_dir=f"~/splitpoint_runs/{namespace}", remote_working_dir="",
        energy_enabled=False, profile_start_snapshot={},
        # The augmentation controller owns the exclusive, explicit new run ID.
        # The generic fresh-run guard intentionally forbids all explicit IDs.
        required_run_mode="", require_fresh_run=False,
        force_build_confirmed_backends=(), force_build_confirmation_source="")
    runtime_profile, snapshot = resolve_runtime_profile_start_snapshot(
        profile_request=str(path), source_profile=loaded.source_profile, resolved_profile=resolved,
        profile_id=loaded.profile_id, profile_path=str(path), profile_source=loaded.source,
        options=WorkflowOptions(**options))
    _check_original_contract(runtime_profile)
    for field in ("quality_gate", "validation_execution", "official_coco_evaluation", "hailo_build", "deepx_build", "run_profiles"):
        if runtime_profile[field] != resolved[field]:
            raise ValueError(f"augmentation runtime materialization changed contract: {field}")
    if runtime_profile["native_producers"]["full_baselines"]["enabled"]:
        raise ValueError("augmentation runtime materialization re-enabled Full baselines")
    options["profile_start_snapshot"] = copy.deepcopy(snapshot)
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    audit = {"profile_path": str(path), "snapshot_path": str(snapshot_path), "profile": runtime_profile,
        "snapshot": snapshot, "options": options,
        "source_run_manifest": str(run / "run_manifest.json"), "source_profile_snapshot_sha256": original_snapshot.get("snapshot_sha256"),
        "model_checks": model_checks, "force_rebuild": False, "compiler_or_inference_executed": False,
        "base_reference_reuse_required": True,
        "composed_only_required": True,
        "variant_scope_note": "Explicit mixed composed-only variants are projected by the ordinary generation writer before required scope is sealed.",
        "energy_retry_contract": {"max_retries": 2, "invalid_repeat_max_retries": 2, "max_transport_failures": 20,
            "source": "native_producers.energy.task_budget; bounded_energy_task campaign_retries inheritance", "shared_campaign_budget_required": True},
        "reference_note": "Mixed cpu_full labels are reference semantics. The orchestrator must restore strictly bound existing CPU outputs and prohibit new CPU reference inference.",
        "scope_note": "Only primary cases. Standby reserves require a separately recorded technical shortfall and normal bounded activation."}
    (out / "augmentation-profile-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
