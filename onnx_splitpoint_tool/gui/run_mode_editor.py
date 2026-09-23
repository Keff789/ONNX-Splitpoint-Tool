from __future__ import annotations

"""GUI editor for the central Smoke / Standard / Final run modes."""

import copy
import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Mapping

from ..config_values import parse_config_bool
from ..run_modes import (
    RunModesConflictError,
    default_run_modes_config,
    default_run_modes_path,
    load_run_modes_config,
    mode_summary,
    normalize_mode_id,
    run_mode_display_rows,
    run_modes_revision,
    save_run_modes_config,
    validate_run_modes_config,
)
from .widgets.tooltip import attach_tooltip


FieldSpec = tuple[str, str, str, Any, tuple[Any, ...] | None, str]


def _get_path(data: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _remove_path(data: dict[str, Any], path: str) -> None:
    """Keep an optional setting absent; an empty string is an explicit opt-out."""
    parts = path.split(".")
    cur: Any = data
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def apply_optional_manifest(mode: dict[str, Any], path: str, text: str, explicit: bool) -> None:
    """A typed/browsed nonempty path is a selection, never silently discarded."""
    value = str(text).strip()
    if value or explicit:
        _set_path(mode, path, value)
    else:
        _remove_path(mode, path)


def validate_hailo8_overlay_selection(
    mode: Mapping[str, Any], *, selected_python: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect the same selected overlay as a cold build, without starting one.

    CPU selection deliberately does not inspect the filesystem or the DFC venv.
    GPU validation only reads the existing package/component metadata. It does
    not claim GPU compute readiness and never changes the GUI process environment.
    """
    from ..hailo_compiler_context import resolve_compute_selection
    from ..hailo_dependency_plan import validated_overlay_components

    hailo = _get_path(mode, "build.hailo", {})
    selection = resolve_compute_selection(
        "hailo8", compute_by_family=hailo.get("compute_by_family"), env=env,
    )
    manifest = selection.get("dependency_manifest")
    result = {"selection": selection, "manifest": manifest, "gpu_compute_test": "not_run"}
    if selection["device"] == "cpu":
        return {**result, "status": "not_used_for_cpu"}
    if not manifest:
        return {**result, "status": "no_overlay_selected"}
    if selected_python is None:
        from ..hailo_backend import _resolve_managed_venv_python
        _, selected_python, _ = _resolve_managed_venv_python(
            hw_arch="hailo8", venv_activate=str(hailo.get("venv_activate") or "auto"),
        )
    components = validated_overlay_components(
        family="hailo8", selected_python=selected_python, manifest_path=manifest,
    )
    return {**result, "status": "metadata_valid", "selected_python": str(selected_python),
            "components": components, "manifest": str(Path(manifest).expanduser().absolute())}


def _open_path(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif os.uname().sysname == "Darwin":
            os.spawnlp(os.P_NOWAIT, "open", "open", str(path))
        else:
            os.spawnlp(os.P_NOWAIT, "xdg-open", "xdg-open", str(path))
    except Exception:
        pass


_SECTIONS: list[tuple[str, list[FieldSpec]]] = [
    (
        "Data & build",
        [
            ("Label", "label", "str", "", None, "Human-readable name shown in the Evaluation Profile editor."),
            ("Description", "description", "str", "", None, "Short explanation of effort and intended evidence level."),
            ("Recommended for", "recommended_for", "str", "", None, "Typical use of this mode."),
            ("Use final dataset registry", "data.use_final_dataset_registry", "bool", True, None, "Bind ImageNet/COCO manifests from Tool Config. Smoke can use the small screening datasets instead."),
            ("Classification calibration items", "data.calibration_items.classification", "int", 8, None, "Maximum train-derived images used for classifier quantisation/calibration."),
            ("Detection calibration items", "data.calibration_items.detection", "int", 8, None, "Maximum train-derived images used for detector quantisation/calibration."),
            ("Classification validation items", "data.validation_items.classification", "int", 16, None, "0 means the complete validation manifest (ImageNet: 50,000)."),
            ("Detection validation items", "data.validation_items.detection", "int", 12, None, "0 means the complete validation manifest (COCO val2017: 5,000)."),
            ("Model preparation", "build.model_preparation", "choice", "current", ("current", "screen_yolo_full_hailo"), "Model preparation strategy."),
            ("Hailo build mode", "build.hailo.mode", "choice", "reuse_and_build_missing", ("reuse_only", "reuse_and_build_missing", "auto", "request", "local", "venv", "wsl"), "Reuse/build policy for HEFs."),
            ("Hailo preset", "build.hailo.preset", "str", "smoke", None, "Descriptive build preset stored in provenance."),
            ("Hailo optimisation level", "build.hailo.optimization_level", "int", 0, None, "DFC model-optimisation effort."),
            ("Hailo calibration items", "build.hailo.calibration_items", "int", 8, None, "Compatibility/default count; task-specific counts above remain authoritative."),
            ("Hailo calibration batch", "build.hailo.calibration_batch_size", "int", 8, None, "Calibration batch size."),
            ("Hailo calibration storage", "build.hailo.calibration_storage", "choice", "memmap", ("memory", "memmap", "auto"), "Memory is fastest for Smoke; memmap allows the requested Standard/Final sample count without a large RAM allocation."),
            ("Hailo calibration memory cap (MiB)", "build.hailo.calibration_memory_cap_mb", "int", 256, None, "Only applies to in-memory calibration."),
            ("Reuse Hailo build artefacts", "build.hailo.cache_enabled", "bool", True, None, "Content-addressed HEF cache across evaluation runs."),
            ("Hailo artefact cache root", "build.hailo.cache_root", "str", "~/.cache/onnx_splitpoint/hailo_hef", None, "Persistent local compiler cache."),
            ("Hailo cache integrity", "build.hailo.cache_integrity", "choice", "relaxed", ("relaxed", "strict"), "Relaxed uses stable metadata identities; strict content verification is available for an explicit campaign."),
            ("Hailo8 compute for new builds", "build.hailo.compute_by_family.hailo8.device", "choice", "cpu", ("cpu", "gpu"), "CPU bleibt Standard. GPU erfordert die eigene Hailo8-Modellabnahme; passende HEFs bleiben wiederverwendbar."),
            ("Hailo8 dependency manifest", "build.hailo.compute_by_family.hailo8.dependency_manifest", "optional_manifest", "", None, "Optionaler vorhandener Zusatzbestand. Nicht ausgewählt: bisherige Umgebungswahl bleibt möglich. Ausgewählt und leer: bewusst kein Overlay. CPU verwendet den gespeicherten Pfad nicht."),
            ("Hailo10H compute for new builds", "build.hailo.compute_by_family.hailo10h.device", "choice", "cpu", ("cpu", "gpu"), "Familienlokale Auswahl. GPU erst nach dem isolierten Modellbuild bewusst aktivieren; kein stiller CPU-Fallback."),
            ("Hailo timeout (s)", "build.hailo.timeout_s", "int", 900, None, "Hard build timeout per artefact."),
            ("Missing Full-HEF policy", "build.hailo.full_baseline_cold_build_policy", "choice", "build_missing", ("cache_or_defer", "build_missing"), "Smoke uses cache_or_defer: exact cache/library hits are reused and a heavy missing Full baseline is recorded immediately. Standard/Final build missing Full baselines."),
            ("Explicit cold-build timeout (s)", "build.hailo.cold_build_timeout_s", "int", 3600, None, "Timeout for an explicit cold-build hardware smoke; ordinary Smoke uses the cache/defer policy."),
            ("Require deferred Full baseline outside Smoke", "build.hailo.deferred_full_baseline_required", "bool", True, None, "Standard/Final treat a deferred or missing selected Full baseline as an incomplete matrix; Smoke may continue diagnostically."),
            ("Preserve Hailo build-attempt provenance", "build.hailo.preserve_attempt_provenance", "bool", True, None, "Keep attempted/cache-miss/timeout state, last compiler stage and elapsed time in queue and report artifacts."),
            ("Build full HEF", "build.hailo.build_full", "bool", True, None, "Build full Hailo baselines when selected."),
            ("Build part 1 HEF", "build.hailo.build_part1", "bool", True, None, "Build accelerator producer subgraphs."),
            ("Build part 2 HEF", "build.hailo.build_part2", "bool", True, None, "Build accelerator consumer subgraphs when selected."),
            ("Legacy Hailo Force (must be OFF)", "build.hailo.force_build", "bool", False, None, "Für normale Starts und Resume AUS lassen. Fehlende Artefakte werden weiterhin gebaut; alte aktive Werte hier ausschalten."),
            ("Keep Hailo intermediate artefacts", "build.hailo.keep_artifacts", "bool", False, None, "Keep HAR and compiler diagnostics."),
            ("DeepX build mode", "build.deepx.mode", "choice", "reuse_and_build_missing", ("reuse_only", "reuse_and_build_missing", "auto"), "DeepX artefact policy."),
            ("DeepX optimisation level", "build.deepx.optimization_level", "int", 0, None, "DX-COM optimisation effort."),
            ("DeepX calibration items", "build.deepx.calibration_items", "int", 8, None, "Task-aware calibration sample count."),
            ("DeepX calibration method", "build.deepx.calibration_method", "str", "ema", None, "DX-COM calibration method."),
            ("DeepX classification preprocessing default", "build.deepx.classification_preprocessing", "choice", "imagenet_mean_std", ("imagenet_mean_std", "current_scale_only"), "Modusdefault für DeepX-Klassifikation. Eine explizite Auswahl im Evaluationsprofil hat Vorrang; current_scale_only bleibt eine Legacy-/Diagnoseauswahl."),
            ("Legacy DeepX Force (must be OFF)", "build.deepx.force_build", "bool", False, None, "Für normale Starts und Resume AUS lassen; vorhandene passende DXNNs wiederverwenden."),
            ("Unified artifact library", "build.artifact_store.enabled", "bool", True, None, "Register and reuse exact HEF/DXNN build contracts across EvaluationRuns."),
            ("Artifact library root", "build.artifact_store.root", "str", "~/.onnx_splitpoint_tool/artifact_store", None, "SQLite index plus content-addressed objects."),
            ("Verify artifacts on reuse", "build.artifact_store.verify_on_reuse", "choice", "metadata", ("metadata", "strict"), "Metadata is the packaged default; strict re-hashing is available for an explicit campaign."),
            ("Pin artifacts for campaign", "build.artifact_store.pin_for_campaign", "bool", False, None, "Prevent Final-campaign artifacts from automatic pruning."),
            ("Register Hailo HEFs", "build.artifact_store.register_hailo", "bool", True, None, "Index generated/reused HEFs in the unified artifact library."),
            ("Register DeepX DXNNs", "build.artifact_store.register_deepx", "bool", True, None, "Index generated/reused DXNNs in the unified artifact library."),
            ("Register TensorRT engines", "build.artifact_store.register_tensorrt", "bool", False, None, "Optionally index TensorRT engines; disabled by default because they are hardware/runtime specific."),
            ("Resource-aware build scheduler", "build.scheduler.enabled", "bool", True, None, "Overlap independent Hailo-8, Hailo-10H and DeepX compiler tasks."),
            ("Build scheduler workers", "build.scheduler.max_workers", "int", 3, None, "Maximum concurrent local compiler tasks."),
            ("Build CPU token budget (0=auto)", "build.scheduler.cpu_tokens", "int", 0, None, "Total CPU tokens; vendor compilers can use internal workers."),
            ("Build RAM budget MiB (0=unlimited)", "build.scheduler.ram_mb", "int", 0, None, "Optional aggregate scheduler RAM budget."),
            ("Hailo-8 concurrent builds", "build.scheduler.family_limits.hailo8", "int", 1, None, "Per-toolchain concurrency guard."),
            ("Hailo-10 concurrent builds", "build.scheduler.family_limits.hailo10", "int", 1, None, "Per-toolchain concurrency guard."),
            ("DeepX concurrent builds", "build.scheduler.family_limits.deepx", "int", 1, None, "Per-toolchain concurrency guard."),
            ("Hailo-8 scheduler CPU tokens", "build.scheduler.weights.hailo8.cpu_tokens", "int", 4, None, "Relative CPU budget consumed by one Hailo-8 build."),
            ("Hailo-8 scheduler RAM MiB", "build.scheduler.weights.hailo8.ram_mb", "int", 6144, None, "Estimated RAM reservation for one Hailo-8 build."),
            ("Hailo-10 scheduler CPU tokens", "build.scheduler.weights.hailo10.cpu_tokens", "int", 4, None, "Relative CPU budget consumed by one Hailo-10 build."),
            ("Hailo-10 scheduler RAM MiB", "build.scheduler.weights.hailo10.ram_mb", "int", 6144, None, "Estimated RAM reservation for one Hailo-10 build."),
            ("DeepX scheduler CPU tokens", "build.scheduler.weights.deepx.cpu_tokens", "int", 2, None, "Relative CPU budget consumed by one DeepX build."),
            ("DeepX scheduler RAM MiB", "build.scheduler.weights.deepx.ram_mb", "int", 4096, None, "Estimated RAM reservation for one DeepX build."),
            ("Prefetch DeepX full while Hailo builds", "build.scheduler.prefetch_deepx_full", "bool", True, None, "Build/cache the independent full DXNN concurrently with BenchmarkSet/Hailo generation."),
            ("Experimental next-model pipeline", "build.scheduler.pipeline_next_model", "bool", False, None, "Reserved for explicit build/remote overlap; disabled by default."),
            ("Hardware smoke policy", "build.hardware_smoke", "choice", "summary_only", ("disabled", "summary_only", "strict"), "Remote/compiler readiness checks."),
        ],
    ),
    (
        "Task quality & reporting",
        [
            ("Quality profile ID", "quality.profile_id", "str", "task_quality_smoke", None, "Identifier embedded in every runtime quality report."),
            ("Dataset tier", "quality.dataset_tier", "choice", "screening", ("screening", "final"), "Determines claim eligibility."),
            ("Confidence level", "quality.confidence_level", "float", 0.95, None, "One-sided non-inferiority confidence level."),
            ("Bootstrap repetitions", "quality.bootstrap_repetitions", "int", 50, None, "Smoke/Standard can be small; Final normally uses 5,000."),
            ("Bootstrap seed", "quality.bootstrap_seed", "int", 20260710, None, "Deterministic resampling seed."),
            ("Quality execution", "quality.execution_location", "choice", "central_management", ("local", "central_management"), "Smoke, Standard and Final run the semantic CPU reference and paired uncertainty on the management node."),
            ("Statistics processes", "quality.workers", "int", 4, None, "Management statistics workers; all packaged modes default to four."),
            ("CPU reference threads", "quality.reference_intra_op_threads", "optional_int", None, None, "Blank retains historical inheritance; explicitly set to decouple reference inference from statistics processes."),
            ("Active statistics requests", "quality.statistics_max_active_requests", "int", 1, None, "One or two active requests share one bounded worker pool."),
            ("Statistics engine", "quality.statistics_engine", "choice", "legacy", ("legacy", "optimized_coco_v1"), "Exact compact COCO statistics; no change to image/draw budgets."),
            ("Draws per block", "quality.statistics_block_repetitions", "int", 256, None, "Task size only, not total repetitions."),
            ("Save completed statistics blocks", "quality.statistics_checkpoint_blocks", "bool", False, None, "Resume validated complete blocks in this run."),
            ("Prepared data MiB / worker", "quality.statistics_prepared_cache_limit_mib", "int", 512, None, "Bounded prepared-data cache."),
            ("Classification margin (pp)", "quality.classification_margin_pp", "float", 1.0, None, "Absolute Top-1 percentage-point budget."),
            ("Top-5 guardrail (pp)", "quality.classification_top5_margin_pp", "float", 1.0, None, "Absolute Top-5 percentage-point budget."),
            ("Detection margin (AP points)", "quality.detection_margin_ap", "float", 1.0, None, "Absolute COCO AP@[.50:.95] budget."),
            ("AP50 guardrail", "quality.detection_ap50_margin", "float", 1.0, None, "Absolute AP50 budget."),
            ("AP75 guardrail", "quality.detection_ap75_margin", "float", 1.0, None, "Absolute AP75 budget."),
            ("Canonical quality reference", "quality.canonical_reference", "str", "canonical_full_onnx", None, "Full-model reference used by the task-quality gate."),
            ("Quality evaluation cadence", "quality.cadence", "choice", "once_per_artifact", ("once_per_artifact", "once_per_run", "each_repeat"), "When task-quality evaluation is executed."),
            ("Cache task-quality evaluation", "quality.cache_task_quality", "bool", True, None, "Evaluate once per artefact instead of once per timing repeat."),
            ("Official COCOeval", "quality.official_coco_enabled", "bool", False, None, "Generate the official pycocotools artefact."),
            ("Official COCOeval required", "quality.official_coco_required", "bool", False, None, "Block final detector claims when the official artefact is missing."),
            ("Archive COCO precision/recall tensors", "quality.archive_coco_eval_tensors", "bool", False, None, "Keep detailed COCOeval arrays."),
            ("Canonical scientific report", "reporting.canonical_scientific_report", "bool", True, None, "Generate the single canonical scientific report tree."),
            ("Decision summary", "reporting.include_decision_summary", "bool", True, None, "Include build/runtime/quality eligibility decisions."),
            ("Backend-drift report", "reporting.include_backend_drift_block", "bool", True, None, "Include full-backend drift diagnostics."),
            ("Task-specific quality block", "reporting.include_task_specific_quality_block", "bool", True, None, "Include classification/detection quality details."),
            ("Thesis LaTeX tables", "reporting.generate_thesis_tex", "bool", False, None, "Generate canonical thesis-ready TeX tables."),
            ("Thesis PDF/PNG figures", "reporting.generate_thesis_figures", "bool", False, None, "Generate thesis-ready plots."),
            ("Clean legacy reports", "reporting.cleanup_legacy_reports", "bool", True, None, "Remove superseded report files."),
            ("Campaign-readiness report", "reporting.include_campaign_readiness", "bool", True, None, "Write development/final readiness diagnostics."),
        ],
    ),
    (
        "Runtime & native",
        [
            ("Default native runner", "defaults.native_enabled", "bool", False, None, "Initial Native Runner switch in a new Evaluation Profile."),
            ("Default Native system-energy measurement", "defaults.energy_enabled", "bool", False, None, "Initial Native system-energy switch. Generic Runner energy is intentionally disabled."),
            ("Workflow execution mode", "runtime.execution_mode", "choice", "generate_and_run", ("generate_and_run", "generate_only", "run_existing"), "Generate/build and/or execute benchmark suites."),
            ("Skip runtime benchmarks", "runtime.skip_runtime_benchmarks", "bool", False, None, "Build/report only; normally false."),
            ("Benchmark provider", "runtime.benchmark.provider", "str", "auto", None, "Generic benchmark provider."),
            ("Benchmark warmup", "runtime.benchmark.warmup", "int", 1, None, "Warmup dispatches per timing row."),
            ("Benchmark runs", "runtime.benchmark.runs", "int", 1, None, "Independent timing repetitions."),
            ("Benchmark timeout (s)", "runtime.benchmark.timeout_s", "int", 600, None, "0 means automatic/no explicit cap."),
            ("Parallel remote setups", "runtime.parallel.remote_setups", "bool", True, None, "Run different physical setups concurrently."),
            ("Maximum parallel setups", "runtime.parallel.max_setups", "int", 3, None, "Fleet-level parallelism."),
            ("Maximum parallel uploads", "runtime.parallel.max_uploads", "int", 1, None, "Upload concurrency."),
            ("Power-calculation workers", "runtime.parallel.powercalc_workers", "int", 1, None, "Parallel post-processing workers for energy traces."),
            ("Upload suite once per model/setup", "runtime.remote_cache.upload_once_per_model_setup", "bool", True, None, "Caches the packaged suite on each remote host and reuses it for all run IDs."),
            ("Reuse TensorRT engines", "runtime.remote_cache.reuse_tensorrt_engines", "bool", True, None, "Persistent model-namespaced TensorRT engine cache on the target host."),
            ("Remote cache root", "runtime.remote_cache.stable_cache_root", "str", "~/.cache/onnx_splitpoint", None, "Parent of remote suite and engine caches."),
            ("Remote suite cache root", "runtime.remote_cache.suite_cache_root", "str", "~/.cache/onnx_splitpoint/suites", None, "Persistent cache for uploaded/extracted suite bundles."),
            ("Remote engine cache root", "runtime.remote_cache.engine_cache_root", "str", "~/.cache/onnx_splitpoint/tensorrt", None, "Persistent TensorRT engine cache."),
            ("Remote suite copy strategy", "runtime.remote_cache.copy_strategy", "choice", "auto", ("auto", "reflink", "extract"), "Auto prefers copy-on-write/reflink and falls back to extraction."),
            ("Native backends", "runtime.native.backends", "list", ["hailo8", "hailo10h", "deepx"], None, "Comma-separated Native Producer backends."),
            ("Native case policy", "runtime.native.case_policy", "choice", "all_accepted", ("all_accepted", "case_map_only", "preferred_then_backfill"), "Which accepted cases enter the native stage."),
            ("Native precision", "runtime.native.precision", "str", "uint8_cast_fp16", None, "Default boundary bridge/precision."),
            ("Native frames", "runtime.native.frames", "int", 100, None, "Measured steady-state frames."),
            ("Native warmup", "runtime.native.warmup", "int", 10, None, "Warmup frames."),
            ("Native performance repetitions", "runtime.native.repetitions", "int", 1, None, "Independent Native timing repetitions (minimum 1). These are separate from Native energy repeats."),
            ("Native queue depth", "runtime.native.queue_depth", "int", 2, None, "FIFO queue depth."),
            ("Native inflight", "runtime.native.inflight", "int", 4, None, "Backend in-flight jobs."),
            ("Native full baselines", "runtime.native.full_baselines", "bool", False, None, "Generate fair native full-model denominators."),
            ("Native contract validation", "runtime.native.validation", "bool", True, None, "Dump and validate boundary/output contracts."),
            ("Native contract selection", "runtime.native.contract_selection", "choice", "metadata_first", ("metadata_first", "probe_first", "explicit_only"), "Prefer explicit/metadata-bound I/O contracts before heuristic probe candidates."),
            ("Preserve Native row failures", "runtime.native.preserve_row_failures", "bool", True, None, "Propagate row-level failure reason, return code, timeout and log tails into stage summaries and Debug Packs."),
            ("Require detection contract-family match", "runtime.native.require_contract_family_match", "bool", True, None, "Reject raw-head probes for decoded/NMS contracts and vice versa instead of selecting by score alone."),
            ("Dump native outputs", "runtime.native.dump_outputs", "bool", True, None, "Retain native output/reference dumps."),
            ("Build missing native engines", "runtime.native.build_missing_engines", "bool", True, None, "Build missing TensorRT/native engines automatically."),
            ("Copy benchmarksets to native stage", "runtime.native.copy_benchmarksets", "bool", True, None, "Stage the selected benchmarksets for native execution."),
            ("Native supported contracts only", "runtime.native.strict_supported_only", "bool", True, None, "Skip unsupported native I/O contracts rather than guessing."),
        ],
    ),
    (
        "Native energy & reproducibility",
        [
            ("Reproducibility level", "reproducibility.level", "choice", "relaxed", ("relaxed", "strict"), "Relaxed records provenance and caches checks; strict re-reads and verifies content."),
            ("Verify model content", "reproducibility.verify_model_content", "bool", False, None, "Strict model-content verification."),
            ("Verify dataset content", "reproducibility.verify_dataset_content", "bool", False, None, "Strict validation/calibration manifest verification."),
            ("Remote script verification", "reproducibility.verify_remote_scripts", "choice", "capabilities", ("capabilities", "strict"), "Capabilities is the packaged default; strict exact-content verification is an explicit campaign option."),
            ("Cache unchanged files", "reproducibility.cache_unchanged_files", "bool", True, None, "Avoid repeatedly hashing unchanged large files."),
            ("Dataset verification sample", "reproducibility.dataset_sample_size", "int", 24, None, "Packaged modes verify a deterministic sample; 0 requests a full manifest verification for an explicit strict campaign."),
            ("Campaign mode", "campaign.mode", "choice", "development", ("development", "screening", "final"), "Final mode enables claim-level gates."),
            ("Campaign claim scope", "campaign.claim_scope", "choice", "evaluated_matrix", ("evaluated_matrix", "ranking_generalization"), "evaluated_matrix limits claims to measured workloads/hardware; ranking_generalization opts into the strict prospective hold-out contract."),
            ("Campaign enforcement", "campaign.enforcement", "choice", "warn", ("warn", "strict"), "Whether missing final requirements block the run."),
            ("Frozen before final campaign", "campaign.frozen_before_final_campaign", "bool", False, None, "Final profile declaration."),
            ("Auto-bind dataset registry", "campaign.auto_bind_dataset_registry", "bool", False, None, "Fill missing task manifests from the central final-data registry."),
            ("Require prediction-freeze approval", "campaign.require_prediction_freeze_approval", "bool", False, None, "Final hold-out approval prerequisite."),
            ("Require cryptographic prediction signature", "campaign.require_cryptographic_prediction_signature", "bool", False, None, "Require a detached signature for the approved hold-out freeze."),
            ("Native energy scope", "energy.scope", "choice", "row_variant", ("row_variant", "run_id", "model"), "How measurement targets are grouped."),
            ("Physical energy scope", "energy.physical_scope", "choice", "FS", ("FS",), "u.RECS measures full-system input power."),
            ("Energy window", "energy.window_label", "choice", "command", ("command",), "Bind energy to the workload command window."),
            ("Native energy repeats", "energy.repeats", "int", 1, None, "Independent command windows when Energy is enabled in the Evaluation Profile."),
            ("Native energy phases", "energy.phases", "list", ["streaming"], None, "latency, streaming"),
            ("Native energy target policy", "energy.target_policy", "choice", "best_valid_only", ("all", "canonical_only", "best_valid_only", "best_plus_predicted", "manual"), "Which valid rows are measured."),
            ("Skip Native energy backends", "energy.skip_backends", "list", ["ort_cpu"], None, "Backends excluded from energy measurement."),
            ("Energy heartbeat (s)", "energy.heartbeat_s", "int", 60, None, "Progress heartbeat during u.RECS windows."),
            ("Energy max targets/run ID", "energy.max_targets_per_run_id", "int", 1, None, "0 means unlimited."),
            ("Max work units/window", "energy.max_work_units_per_window", "int", 0, None, "0 means automatic/unlimited."),
            ("Max window duration (s)", "energy.max_window_duration_s", "int", 0, None, "0 means automatic."),
            ("Energy timeout/window (s)", "energy.timeout_s_per_window", "int", 0, None, "0 means automatic."),
            ("Sizing probe max work units", "energy.sizing_probe_max_work_units", "int", 256, None, "Initial work-unit cap used to size the command window."),
            ("Energy strict", "energy.strict", "bool", False, None, "Require complete accepted traces."),
            ("Archive raw energy parquet", "energy.include_raw_parquet_in_debug_pack", "bool", False, None, "Useful for final audit, expensive for packs."),
            ("Native energy mode", "energy.native_mode", "choice", "plan", ("plan", "measure"), "Native energy execution when master Energy and Native are enabled."),
            ("Native energy duration (s)", "energy.native_duration_s", "int", 0, None, "0 uses the workload/measurement contract default."),
        ],
    ),
    (
        "Hold-out & ranking",
        [
            ("Ranking validation", "ranking.enabled", "bool", False, None, "Compare the pre-registered methods, including the selector actually used by the workflow."),
            ("Ranking methods", "ranking.methods", "list", ["cut_bytes_only", "weighted_score", "cycle_time_no_handover", "cycle_time_with_handover", "onnx_real_boundary_hardware_aware"], None, "Methods included in the comparison."),
            ("Ranking bootstrap repetitions", "ranking.bootstrap_repetitions", "int", 100, None, "Uncertainty of ranking metrics."),
            ("Ranking bootstrap seed", "ranking.bootstrap_seed", "int", 20260710, None, "Deterministic ranking-metric bootstrap seed."),
            ("Ranking k values", "ranking.k_values", "list_int", [1, 3, 5], None, "Shortlist sizes."),
            ("Elite q values", "ranking.elite_q_values", "list_int", [1, 3], None, "Measured elite-set sizes."),
            ("Primary k", "ranking.primary_k", "int", 5, None, "Primary shortlist size for the thesis."),
            ("Minimum candidates for correlation", "ranking.minimum_candidates_for_correlation", "int", 3, None, "Minimum n for Spearman/Kendall."),
            ("Near-optimal relative epsilon", "ranking.near_optimal_relative_epsilon", "float", 0.01, None, "Relative tolerance used for the near-optimal set."),
            ("Candidate universe", "holdout.candidate_universe", "choice", "deterministic_audit", ("declared_shortlist", "deterministic_audit", "all_feasible"), "Hold-out measurement universe."),
            ("Audit size", "holdout.audit_size", "int", 20, None, "Score-independent deterministic audit size; independent of the deployment shortlist."),
            ("Minimum valid audit candidates", "holdout.minimum_valid_candidates", "int", 10, None, "Minimum valid measured candidates required for audit-relative ranking metrics."),
            ("Candidate-universe seed", "holdout.seed", "int", 20260710, None, "Deterministic audit-universe seed."),
            ("Enable prediction freeze", "holdout.prediction_freeze_enabled", "bool", False, None, "Freeze predictions before measurement independently of whether ranking statistics are enabled."),
            ("Require complete universe", "holdout.require_complete_candidate_universe", "bool", False, None, "Required for exact Top-k recall."),
            ("Require frozen predictions", "holdout.require_frozen_predictions", "bool", False, None, "Prospective hold-out freeze."),
            ("Require unseen attestation", "holdout.require_unseen_attestation", "bool", False, None, "Named declaration for final hold-out models."),
            ("Require fitted stage-time model", "campaign.require_fitted_stage_time", "bool", False, None, "Final model-validation prerequisite."),
            ("Require native handover model", "campaign.require_native_handover_model", "bool", False, None, "Final Native FIFO cost-model prerequisite."),
            ("Require campaign freeze", "campaign.require_campaign_freeze", "bool", False, None, "Archive/freeze prerequisite."),
            ("Require protocol freeze", "campaign.require_protocol_freeze", "bool", False, None, "Versioned prospective protocol prerequisite for ranking-generalization claims."),
        ],
    ),
]


class RunModeEditDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, *, mode_id: str, config: dict[str, Any], on_saved: Callable[[dict[str, Any]], bool | None]) -> None:
        # Validate before creating a window or BooleanVar: Tk must never turn
        # malformed persisted values into checked Force boxes.
        validated = validate_run_modes_config(config)
        super().__init__(master)
        self.mode_id = normalize_mode_id(mode_id)
        self.config = copy.deepcopy(validated)
        self.mode = copy.deepcopy(dict(self.config["modes"][self.mode_id]))
        self.on_saved = on_saved
        self.vars: dict[str, tk.Variable] = {}
        self._manifest_explicit_vars: dict[str, tk.BooleanVar] = {}
        self._manifest_status_vars: dict[str, tk.StringVar] = {}
        self.title(f"Run mode bearbeiten — {self.mode_id}")
        self.geometry("1050x780")
        self.minsize(900, 650)
        self.transient(master.winfo_toplevel())
        self.grab_set()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        nb = ttk.Notebook(self)
        nb.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        for title, specs in _SECTIONS:
            tab = ttk.Frame(nb)
            nb.add(tab, text=title)
            self._build_section(tab, specs)

        foot = ttk.Frame(self)
        foot.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        foot.columnconfigure(0, weight=1)
        self.status = tk.StringVar(self, value="Build- und Prüfmodi werden zentral gespeichert; DeepX Classification wird separat im Evaluationsprofil gewählt.")
        ttk.Label(foot, textvariable=self.status).grid(row=0, column=0, sticky="w")
        ttk.Button(foot, text="Reset mode defaults", command=self._reset).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(foot, text="Save", command=self._save).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(foot, text="Cancel", command=self.destroy).grid(row=0, column=3, padx=(8, 0))

    def _build_section(self, tab: ttk.Frame, specs: list[FieldSpec]) -> None:
        container = ttk.Frame(tab)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        inner.columnconfigure(1, weight=1)
        for row, (label, path, kind, default, choices, tip) in enumerate(specs):
            ttk.Label(inner, text=label + ":").grid(row=row, column=0, sticky="w", padx=(12, 8), pady=5)
            value = _get_path(self.mode, path, default)
            if kind == "bool":
                var: tk.Variable = tk.BooleanVar(
                    self,
                    value=parse_config_bool(value, field=f"modes.{self.mode_id}.{path}"),
                )
                widget = ttk.Checkbutton(inner, variable=var)
                if path in {"build.hailo.force_build", "build.deepx.force_build"}:
                    # A preserved legacy true may be deliberately cleared,
                    # but the editor cannot enable productive Force again.
                    if not value:
                        widget.state(["disabled"])
                    else:
                        widget.configure(command=lambda w=widget, v=var: w.state(["disabled"]) if not v.get() else None)
            elif kind == "optional_int":
                var = tk.StringVar(self, value="" if value is None else str(value))
                widget = ttk.Entry(inner, textvariable=var)
            elif kind == "int":
                var = tk.StringVar(self, value=str(int(value or 0)))
                widget = ttk.Entry(inner, textvariable=var)
            elif kind == "float":
                var = tk.StringVar(self, value=str(float(value or 0.0)))
                widget = ttk.Entry(inner, textvariable=var)
            elif kind == "choice":
                var = tk.StringVar(self, value=str(value or default))
                widget = ttk.Combobox(inner, textvariable=var, values=list(choices or ()), state="readonly")
            elif kind == "list_int":
                var = tk.StringVar(self, value=", ".join(str(int(x)) for x in (value or [])))
                widget = ttk.Entry(inner, textvariable=var)
            elif kind == "list":
                var = tk.StringVar(self, value=", ".join(str(x) for x in (value or [])))
                widget = ttk.Entry(inner, textvariable=var)
            elif kind == "optional_manifest":
                var = tk.StringVar(self, value=str(value or ""))
                explicit = tk.BooleanVar(self, value=_get_path(self.mode, path, None) is not None)
                self._manifest_explicit_vars[path] = explicit
                widget = ttk.Frame(inner)
                widget.columnconfigure(0, weight=1)
                ttk.Checkbutton(widget, text="Explizite Auswahl (leer = bewusst kein Overlay)",
                                variable=explicit,
                                command=lambda p=path: self._toggle_manifest(p)).grid(row=0, column=0, columnspan=2, sticky="w")
                # Typing, pasting and browsing have identical save semantics.
                # Clearing the checkbox explicitly returns to inheritance.
                var.trace_add("write", lambda *_a, v=var, e=explicit: e.set(True) if str(v.get()).strip() else None)
                ttk.Entry(widget, textvariable=var).grid(row=1, column=0, sticky="ew")
                ttk.Button(widget, text="Datei…", command=lambda p=path: self._browse_manifest(p)).grid(row=1, column=1)
                ttk.Button(widget, text="Lesend prüfen", command=lambda p=path: self._validate_manifest(p)).grid(row=2, column=0, sticky="w")
                status = tk.StringVar(self, value="Noch nicht geprüft; vorhandene Artefakte bleiben wiederverwendbar.")
                self._manifest_status_vars[path] = status
                ttk.Label(widget, textvariable=status, wraplength=420, justify="left").grid(row=3, column=0, columnspan=2, sticky="w")
            else:
                var = tk.StringVar(self, value=str(value or ""))
                widget = ttk.Entry(inner, textvariable=var)
            self.vars[path] = var
            widget.grid(row=row, column=1, sticky="ew", padx=(0, 12), pady=5)
            if tip:
                attach_tooltip(widget, tip, delay_ms=300, wraplength=480)
                ttk.Label(inner, text=tip, foreground="#666", wraplength=520, justify="left").grid(row=row, column=2, sticky="w", padx=(0, 12), pady=5)

    def _coerce(self, spec: FieldSpec) -> Any:
        _label, path, kind, default, _choices, _tip = spec
        var = self.vars[path]
        value = var.get()
        if kind == "bool":
            return parse_config_bool(value, field=f"modes.{self.mode_id}.{path}")
        if kind == "optional_int":
            return int(str(value).strip()) if str(value).strip() else None
        if kind == "int":
            parsed = int(str(value).strip() or 0)
            if path == "runtime.native.repetitions" and parsed < 1:
                raise ValueError("Native performance repetitions must be at least 1")
            return parsed
        if kind == "float":
            return float(str(value).strip() or 0.0)
        if kind == "list_int":
            return [int(x.strip()) for x in str(value).replace(";", ",").split(",") if x.strip()]
        if kind == "list":
            return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]
        return str(value).strip()

    def _save(self) -> None:
        try:
            for _title, specs in _SECTIONS:
                for spec in specs:
                    if spec[2] == "optional_manifest":
                        apply_optional_manifest(self.mode, spec[1], self._coerce(spec),
                                                self._manifest_explicit_vars[spec[1]].get())
                    else:
                        _set_path(self.mode, spec[1], self._coerce(spec))
            self.config["modes"][self.mode_id] = self.mode
            self.config = validate_run_modes_config(self.config)
            if self.on_saved(self.config) is False:
                self.status.set("Nicht gespeichert. Nach Neu laden diesen Dialog schließen und den Editor frisch öffnen.")
                return
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Run modes", str(exc), parent=self)

    def _browse_manifest(self, path: str) -> None:
        filename = filedialog.askopenfilename(
            parent=self, title="Vorhandenes Hailo8 overlay_manifest.json auswählen",
            filetypes=[("JSON manifest", "*.json"), ("Alle Dateien", "*")],
        )
        if filename:
            self.vars[path].set(filename)
            self._manifest_explicit_vars[path].set(True)
            self._manifest_status_vars[path].set("Ausgewählt; noch nicht geprüft.")

    def _toggle_manifest(self, path: str) -> None:
        if not self._manifest_explicit_vars[path].get():
            self.vars[path].set("")
            self._manifest_status_vars[path].set("Nicht gesetzt; vorhandene Umgebungswahl bleibt möglich.")

    def _validate_manifest(self, path: str) -> None:
        mode = copy.deepcopy(self.mode)
        _set_path(mode, "build.hailo.compute_by_family.hailo8.device",
                  str(self.vars["build.hailo.compute_by_family.hailo8.device"].get()))
        apply_optional_manifest(mode, path, self.vars[path].get(), self._manifest_explicit_vars[path].get())
        try:
            result = validate_hailo8_overlay_selection(mode)
            source = result["selection"].get("dependency_manifest_source", "unset")
            if result["status"] == "not_used_for_cpu":
                text = f"Für CPU nicht verwendet. Quelle: {source}"
            elif result["status"] == "no_overlay_selected":
                text = f"Kein Overlay ausgewählt. Quelle: {source}. Eine passende Venv kann ausreichen; hier nicht GPU-geprüft."
            else:
                text = (f"Bestand gültig (lesende Prüfung). Quelle: {source}\n"
                        f"{result['manifest']}\nVenv-Python: {result['selected_python']}\n"
                        "GPU-/XLA-Rechnung nicht ausgeführt.")
        except Exception as exc:
            text = f"Prüfung fehlgeschlagen: {type(exc).__name__}: {exc}"
        self._manifest_status_vars[path].set(text)

    def _reset(self) -> None:
        defaults = default_run_modes_config()["modes"][self.mode_id]
        if not messagebox.askyesno("Run modes", f"Reset {self.mode_id} to packaged defaults?", parent=self):
            return
        self.config["modes"][self.mode_id] = copy.deepcopy(defaults)
        try:
            if self.on_saved(self.config) is not False:
                self.destroy()
        except Exception as exc:
            messagebox.showerror("Run modes", str(exc), parent=self)


class RunModesPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, *, app: Any | None = None) -> None:
        super().__init__(master)
        self.app = app
        self.path = default_run_modes_path()
        try:
            self.config = load_run_modes_config(self.path)
        except Exception as exc:
            messagebox.showerror(
                "Run modes: ungültiges Feld",
                f"Konfiguration konnte nicht geöffnet werden:\n{exc}\n\n{self.path}",
                parent=self,
            )
            raise
        self._baseline = copy.deepcopy(self.config)
        self._loaded_revision = run_modes_revision(self.config)
        self.mode_id = tk.StringVar(self, value=str(self.config.get("default_mode") or "standard"))
        self.summary = tk.StringVar(self, value="")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        top = ttk.LabelFrame(self, text="Evaluation run modes")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Default run mode:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
        values = [row["id"] for row in run_mode_display_rows(self.config)]
        combo = ttk.Combobox(top, textvariable=self.mode_id, values=values, state="readonly", width=16)
        combo.grid(row=0, column=1, sticky="w", pady=8)
        combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh())
        ttk.Button(top, text="Edit selected…", command=self._edit).grid(row=0, column=2, padx=(8, 0), pady=8)
        ttk.Button(top, text="Reset selected", command=self._reset).grid(row=0, column=3, padx=(8, 0), pady=8)
        ttk.Button(top, text="Open YAML", command=lambda: _open_path(self.path)).grid(row=0, column=4, padx=(8, 8), pady=8)
        ttk.Label(
            top,
            text=(
                "Evaluation Profiles now select only Smoke, Standard or Final. The detailed build, validation, "
                "hold-out, reporting, energy and reproducibility settings are configured here."
            ),
            foreground="#555",
            wraplength=1050,
            justify="left",
        ).grid(row=1, column=0, columnspan=5, sticky="ew", padx=8, pady=(0, 8))

        cards = ttk.Frame(self)
        cards.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for col in range(3):
            cards.columnconfigure(col, weight=1)
        for col, row in enumerate(run_mode_display_rows(self.config)):
            card = ttk.LabelFrame(cards, text=row["label"])
            card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 4, 0 if col == 2 else 4))
            ttk.Label(card, text=row["description"], wraplength=330, justify="left").pack(anchor="w", padx=8, pady=(8, 4))
            ttk.Label(card, text=row["recommended_for"], foreground="#666", wraplength=330, justify="left").pack(anchor="w", padx=8, pady=(0, 8))

        detail = ttk.LabelFrame(self, text="Effective selected-mode summary")
        detail.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        detail.columnconfigure(0, weight=1)
        detail.rowconfigure(0, weight=1)
        text = tk.Text(detail, height=15, wrap="word")
        text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.text = text
        buttons = ttk.Frame(detail)
        buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(buttons, text="Validate & save", command=self._save).pack(side="left")
        ttk.Button(buttons, text="Reload", command=self._reload).pack(side="left", padx=(8, 0))
        ttk.Label(buttons, text=str(self.path), foreground="#666").pack(side="right")
        self._refresh()

    def _refresh(self) -> None:
        mid = normalize_mode_id(self.mode_id.get())
        self.mode_id.set(mid)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", mode_summary(mid, self.config) + "\n\n" + json.dumps(self.config["modes"][mid], indent=2, ensure_ascii=False))
        self.text.configure(state="disabled")

    def _persist(
        self,
        config: dict[str, Any],
        *,
        expected_revision: str | None = None,
        baseline: Mapping[str, Any] | None = None,
    ) -> bool:
        pending = copy.deepcopy(config)
        pending["default_mode"] = normalize_mode_id(self.mode_id.get())
        try:
            self.path = save_run_modes_config(
                self.path,
                pending,
                expected_revision=(self._loaded_revision if expected_revision is None else expected_revision),
                baseline=(self._baseline if baseline is None else baseline),
            )
        except RunModesConflictError as exc:
            fields = "\n".join(str(field) for field in exc.changed_fields) or "Registryrevision"
            if messagebox.askyesno(
                "Run modes: Konfiguration geändert",
                "Nicht gespeichert: Seit dem Öffnen wurden diese Felder geändert:\n"
                + fields
                + "\n\nDie neuere Datei bleibt erhalten. Jetzt neu laden? "
                "Lokale Änderungen werden verworfen; öffne den Editor danach erneut.",
                parent=self,
                default=messagebox.NO,
            ):
                self._reload()
            return False
        self.config = load_run_modes_config(self.path)
        self._baseline = copy.deepcopy(self.config)
        self._loaded_revision = run_modes_revision(self.config)
        self._refresh()
        if self.app is not None:
            try:
                self.app._persist_settings()
            except Exception:
                pass
        return True

    def _edit(self) -> None:
        # Each dialog keeps its own loaded revision even if its parent reloads.
        revision = self._loaded_revision
        baseline = copy.deepcopy(self._baseline)
        try:
            RunModeEditDialog(
                self,
                mode_id=self.mode_id.get(),
                config=self.config,
                on_saved=lambda config: self._persist(
                    config, expected_revision=revision, baseline=baseline
                ),
            )
        except Exception as exc:
            messagebox.showerror("Run modes", str(exc), parent=self)

    def _reset(self) -> None:
        mid = normalize_mode_id(self.mode_id.get())
        if not messagebox.askyesno("Run modes", f"Reset {mid} to packaged defaults?", parent=self):
            return
        pending = copy.deepcopy(self.config)
        pending["modes"][mid] = copy.deepcopy(default_run_modes_config()["modes"][mid])
        try:
            self._persist(pending)
        except Exception as exc:
            messagebox.showerror("Run modes", str(exc), parent=self)

    def _save(self) -> None:
        try:
            if self._persist(self.config):
                messagebox.showinfo("Run modes", f"Saved and validated:\n{self.path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Run modes", str(exc), parent=self)

    def _reload(self) -> None:
        try:
            self.config = load_run_modes_config(self.path)
            self._baseline = copy.deepcopy(self.config)
            self._loaded_revision = run_modes_revision(self.config)
            self.mode_id.set(str(self.config.get("default_mode") or "standard"))
            self._refresh()
        except Exception as exc:
            messagebox.showerror("Run modes", str(exc), parent=self)


def build_run_modes_panel(parent: tk.Misc, *, app: Any | None = None) -> RunModesPanel:
    panel = RunModesPanel(parent, app=app)
    panel.pack(fill="both", expand=True)
    return panel
