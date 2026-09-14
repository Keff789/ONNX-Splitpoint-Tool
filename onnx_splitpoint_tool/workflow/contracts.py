from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

WORKFLOW_SCHEMA_VERSION = 1
STAGE_STATUSES = {
    "ok", "partial", "skipped", "failed", "planned", "warn", "cancelled",
}
STAGE_STATES = {
    "not_started", "running", "completed", "failed", "cancelled",
}
ERROR_CLASSES = {
    "parse_failed",
    "compile_failed",
    "runtime_failed",
    "validation_failed",
    "timeout",
    "unsupported_op",
    "resource_infeasible",
    "missing_artifact",
    "stale_artifact",
    "profile_mismatch",
    "not_implemented",
    "unknown_error",
}


@dataclass
class StageResult:
    stage: str
    model_id: Optional[str]
    status: str
    started_at: str
    finished_at: str
    job_id: str = ""
    parent_job_id: str = ""
    job_type: str = ""
    execution_mode: str = "workflow_contract"
    state: str = "completed"
    complete: bool = True
    input_hash: str = ""
    output_hash: str = ""
    artifacts: List[str] = field(default_factory=list)
    skip_reason: str = ""
    error_class: str = ""
    error_detail: str = ""
    notes: List[str] = field(default_factory=list)
    child_job_ids: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = WORKFLOW_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        if out.get("status") not in STAGE_STATUSES:
            out["status"] = "warn"
            out["error_class"] = out.get("error_class") or "unknown_error"
        if out.get("error_class") and out.get("error_class") not in ERROR_CLASSES:
            out["error_class"] = "unknown_error"
        state = str(out.get("state") or "").strip().lower()
        if state not in STAGE_STATES:
            state = "failed"
            out["error_class"] = out.get("error_class") or "unknown_error"
        out["state"] = state
        # ``complete`` is an execution-lifecycle marker, not a positive-result
        # claim.  A terminal failed stage has a complete decision; cancelled or
        # still-running work never does.
        if state in {"not_started", "running", "cancelled"}:
            out["complete"] = False
        else:
            out["complete"] = bool(out.get("complete"))
        return out


@dataclass
class WorkflowOptions:
    profile: str
    out: str
    models_root: str = ""
    resume: bool = False
    dry_run: bool = False
    stop_after: Optional[str] = None
    only_model: Optional[str] = None
    include_reserve: bool = False
    skip_benchmarks: bool = False
    no_remote: bool = False
    run_id: Optional[str] = None
    max_models: Optional[int] = None
    force_stage: List[str] = field(default_factory=list)
    no_model_hash: bool = False
    skip_analysis: bool = False
    execution_mode: str = "contracts_only"  # contracts_only | generate_benchmarksets | generate_and_run | legacy_profile_campaign
    artifact_policy: str = "normal"  # normal | cache_verify_only
    benchmark_sets_root: str = ""
    benchmark_results_root: str = ""
    result_sources: List[str] = field(default_factory=list)
    benchmark_provider: str = ""
    benchmark_warmup: int = 1
    benchmark_runs: int = 3
    benchmark_timeout_s: int = 0
    benchmark_preset: str = "auto"
    benchmark_image: str = "default"
    benchmark_extra_args: List[str] = field(default_factory=list)

    # v49e-v49i formal Hailo/remote execution binding. These fields are safe to
    # serialize into run_manifest.json; no SSH secrets are stored here.
    benchmark_execution_backend: str = "auto"  # auto | local | remote
    remote_host_json: str = ""
    remote_host_id: str = ""
    remote_host: str = ""
    remote_user: str = ""
    remote_port: int = 22
    remote_base_dir: str = "~/splitpoint_runs"
    remote_ssh_extra_args: str = ""
    remote_hosts_file: str = ""
    remote_working_dir: str = ""
    remote_provider: str = "auto"
    remote_venv: str = ""
    remote_transfer_mode: str = "bundle"
    remote_reuse_bundle: bool = True
    remote_no_reuse_bundle: bool = False
    remote_resume: bool = True
    remote_no_resume: bool = False
    remote_repeats: int = 1
    remote_warmup: int = 10
    remote_iters: int = 50
    remote_timeout_s: int = 0
    remote_add_args: str = ""
    remote_throughput_frames: int = 24
    remote_throughput_warmup_frames: int = 6
    remote_throughput_queue_depth: int = 2
    remote_validation_images: str = ""
    remote_validation_max_images: int = 0
    # v60r: Generated Evaluation Workflow suites carry an authoritative
    # run-mode budget.  Legacy GUI/CLI defaults (historically 50 images) must
    # not override that budget once a Smoke/Standard/Final profile has been
    # materialised.
    remote_validation_budget_authoritative: bool = False
    remote_validation_reference_mode: str = "auto"
    remote_mini_coco_ap50: bool = False
    remote_benchmark_task: str = "auto"
    remote_mini_classification_eval: bool = False

    # v58 evaluation-workflow u.RECS energy integration.  The actual u.RECS
    # address is resolved from the selected hardware setup; these fields only
    # decide whether Evaluation Workflow remote dispatches should be measured and
    # how many u.RECS repeats to run.  energy_repeat_override=0 means: use the
    # benchmark/remote repeat count.
    energy_enabled: bool = False
    energy_scope: str = "row_variant"  # row_variant | dispatch
    energy_repeat_override: int = 0
    energy_phases: List[str] = field(default_factory=list)
    # v58b/v58p: keep evaluation energy practical by selecting only relevant targets
    # by default. Supported: all, canonical_only, deepx_only, best_valid_only, best_plus_predicted, manual.
    energy_target_policy: str = "canonical_only"
    energy_skip_backends: List[str] = field(default_factory=list)
    energy_include_run_ids: List[str] = field(default_factory=list)
    energy_exclude_run_ids: List[str] = field(default_factory=list)
    energy_heartbeat_s: int = 60
    energy_include_raw_parquet_in_debug_pack: bool = False
    energy_strict: bool = False
    # Optional safety caps for long evaluation-energy windows. 0 disables a cap.
    energy_max_targets_per_run_id: int = -1
    energy_max_work_units_per_window: int = 0
    energy_max_window_duration_s: int = 0
    energy_timeout_s_per_window: int = 0
    energy_sizing_probe_max_work_units: int = 256

    # Hailo build/reuse queue options. v49i keeps the default non-destructive:
    # existing prepared/full/raw-head HEFs are reused and missing HEFs are
    # written as explicit queue entries. Future patches can flip this to local
    # build mode in environments with DFC/Hailo tooling.
    hailo_build_mode: str = "reuse_only"  # auto | reuse_only | reuse_and_build_missing | request | local | venv | wsl
    hailo_hw_arch: str = "hailo8"
    hailo_build_targets: List[str] = field(default_factory=list)
    hailo_build_backend: str = "auto"
    hailo_build_timeout_s: int = 3600
    hailo_build_full: bool = True
    hailo_build_part1: bool = True
    hailo_build_part2: bool = True
    hailo_preset: str = "quick"
    hailo_optimization_level: int = 0
    hailo_calib_dir: str = ""
    hailo_calib_count: int = 16
    hailo_calib_batch_size: int = 8
    hailo_force_build: bool = False
    hailo_keep_artifacts: bool = False

    # v49i validation + hardware smoke binding. Defaults are intentionally safe:
    # summarize validation metrics emitted by the benchmark suites, write gaps,
    # and do not launch heavy validation/DFC/hardware probes implicitly.
    validation_mode: str = "summary_only"  # summary_only | strict | disabled
    validation_max_cases: int = 0
    validation_require_explicit: bool = False
    validation_require_task_metrics: bool = False
    validation_max_abs_error_threshold: float = 1e-3
    validation_mean_abs_error_threshold: float = 1e-4
    validation_cosine_threshold: float = 0.999
    validation_min_top1: float = 0.0
    validation_min_top5: float = 0.0
    validation_min_ap50: float = 0.0
    hardware_smoke_mode: str = "summary_only"  # summary_only | strict | disabled
    skip_hardware_smoke: bool = False
    hardware_smoke_timeout_s: int = 5

    # v52 multi-hardware registry selection.  These values are also persisted
    # into the run manifest so a dissertation run can prove which physical
    # Orin NX / accelerator setup was selected for each backend.
    hardware_setups_file: str = ""
    hardware_setup_ids: List[str] = field(default_factory=list)
    hardware_group_ids: List[str] = field(default_factory=list)

    # GUI/CLI fast path for re-running already generated suites with new runner/report code.
    # Generated and build stages are reused even when the tool-version resume hash changed.
    rerun_generated_only: bool = False

    # Invocation-only repair for an interrupted Full-only central-quality run.
    # The runner derives the missing identities from the sealed execution plan
    # and the existing central summary; callers cannot choose endpoints.
    resume_missing_full_quality_only: bool = False

    # v59bq Native Producer EvalRunner integration.  Native producers are a
    # strict additional execution mode: they only run supported IO contracts and
    # report unsupported/missing cases instead of silently falling back.
    native_producer_enabled: bool = False
    native_producer_backends: List[str] = field(default_factory=list)
    native_producer_model_case_map: str = ""
    native_producer_case_policy: str = "all_accepted"
    native_producer_remote_root: str = "/home/nx/native_fifo_evalsets"
    native_producer_remote_tool_dir: str = "/home/nx/ONNX-Splitpoint-Tool"
    native_producer_hailo8_ssh: str = ""
    native_producer_hailo10_ssh: str = ""
    native_producer_deepx_ssh: str = ""
    native_producer_hailo8_env: str = ""
    native_producer_hailo10_env: str = "export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate"
    native_producer_deepx_env: str = "source ~/venvs/deepx-runtime/bin/activate"
    native_producer_precision: str = "uint8_cast_fp16"
    native_producer_frames: int = 1000
    native_producer_warmup: int = 100
    native_producer_queue_depth: int = 3
    native_producer_inflight: int = 8
    native_producer_hailo_format: str = "uint8"
    native_producer_dump_outputs: bool = False
    native_producer_no_copy: bool = False
    native_producer_no_build_missing_engines: bool = False
    # Native producer u.RECS energy is separate from the generic EvalRun energy checkbox.
    native_producer_energy_enabled: bool = False
    native_producer_energy_mode: str = "plan"
    native_producer_energy_frames: int = 5000
    native_producer_energy_warmup: int = 200
    native_producer_energy_timeout_s: int = 900
    # Separate marker-v2 vs. historical-window screening probe. ``None`` means
    # the resolved profile/Tool Config value remains authoritative.
    window_method_validation_probe_enabled: Optional[bool] = None
    window_method_validation_probe_repeats: int = 0
    window_method_validation_probe_include_raw_parquet: Optional[bool] = None
    window_method_validation_probe_strict: Optional[bool] = None

    # Optional entry guards used by thin automation around the normal runner.
    # They validate the resolved profile/start intent before a run directory or
    # any remote work is created; they do not define another execution mode.
    required_run_mode: str = ""
    require_fresh_run: bool = False

    # Retain this position so the long-standing positional constructor contract
    # WorkflowOptions(profile, out, models_root, ...) remains intact.
    # GUI starts freeze the exact source/resolved profile pair that was shown
    # in the start summary.  The worker consumes this in-memory snapshot instead
    # of resolving YAML and central registries a second time.
    profile_start_snapshot: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Active invocation only: never load these from archived options/profile.
    force_build_confirmed_backends: tuple[str, ...] = ()
    force_build_confirmation_source: str = ""


    def to_dict(self) -> Dict[str, Any]:
        # Keep this round-trippable: WorkflowOptions(**options.to_dict()) is used
        # by integration code and must retain the validated runtime snapshot.
        return asdict(self)


@dataclass
class WorkflowRunResult:
    ok: bool
    status: str
    run_id: str
    run_dir: str
    manifest_path: str
    artifact_index_path: str
    report_paths: List[str] = field(default_factory=list)
    stage_results: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    completed: bool = True
    evidence_status: Dict[str, Any] = field(default_factory=dict)
    # ``status``/``ok`` remain the process/workflow execution axis for return
    # code compatibility.  Scientific quality is an independent axis: a run
    # can execute successfully and still fail or remain inconclusive.
    technical_status: str = "unavailable"
    quality_decision: str = "not_evaluated"
    scientific_status: str = "not_evaluated"
    completion: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
