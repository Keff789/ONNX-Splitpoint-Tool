from __future__ import annotations

from onnx_splitpoint_tool.native_full_quality import (
    enabled_run_profiles,
    normalise_evaluation_profile,
    resolve_native_full_plan,
    resolve_native_split_plan,
)
"""Central run-mode configuration and evaluation-profile materialisation.

The GUI exposes only a small number of high-level choices for an evaluation
run (run mode, models, candidate strategy, hardware run profiles, native and
energy switches).  All detailed build, validation, reporting, hold-out and
reproducibility parameters live in a central, editable run-mode registry.

A resolved snapshot is written into every evaluation profile/run so changing
Tool Config later never makes an already archived run ambiguous.
"""

import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml

from .cache_verify_policy import apply_cache_verify_only_policy
from .hailo_timeout_policy import (
    canonical_hailo_disable_tokens,
    parse_hailo_timeout_seconds,
)

RUN_MODE_SCHEMA = "onnx-splitpoint/run-modes"
RUN_MODE_SCHEMA_VERSION = 13
DEFAULT_MODE_ID = "standard"
_MODE_ORDER = ("smoke", "standard", "final")
EVALUATED_MATRIX_CLAIM_SCOPE = "evaluated_matrix"
RANKING_GENERALIZATION_CLAIM_SCOPE = "ranking_generalization"
CAMPAIGN_CLAIM_SCOPES = {
    EVALUATED_MATRIX_CLAIM_SCOPE,
    RANKING_GENERALIZATION_CLAIM_SCOPE,
}

# Native Full baselines are selected by the same logical Full profiles that the
# user sees in the simplified Evaluation Profile editor.  The mapping is kept
# here (rather than in the GUI) so CLI-loaded and hand-edited YAML profiles have
# identical semantics.
_NATIVE_FULL_PROFILE_BACKENDS = {
    "ort_tensorrt": "tensorrt",
    "tensorrt": "tensorrt",
    "tensorrt_full": "tensorrt",
    "hailo8": "hailo8",
    "hailo8_full": "hailo8",
    "hailo10": "hailo10h",
    "hailo10h": "hailo10h",
    "hailo10_full": "hailo10h",
    "hailo10h_full": "hailo10h",
    "deepx_m1": "deepx",
    "deepx_m1_full": "deepx",
    "deepx": "deepx",
    "deepx_full": "deepx",
}
_NATIVE_FULL_BACKEND_ORDER = ("tensorrt", "hailo8", "hailo10h", "deepx")


def native_full_backends_from_run_profiles(run_profiles: Any) -> list[str]:
    """Return the selected Native Full backends in deterministic order.

    A Full checkbox in the simplified GUI is authoritative.  Mixed-backend
    profiles such as ``hailo8_to_trt`` do not implicitly request either Full
    baseline; the corresponding Full checkbox must be selected as well.
    """
    selected: set[str] = set()
    for raw in enabled_run_profiles(run_profiles):
        rid = str(raw.get("id") or raw.get("run_id") or "").strip().lower().replace("-", "_")
        backend = _NATIVE_FULL_PROFILE_BACKENDS.get(rid)
        if not backend:
            full = str(raw.get("full") or "").strip().lower().replace("-", "_")
            if full in {"tensorrt", "trt", "ort_tensorrt"}:
                backend = "tensorrt"
            elif full in {"hailo8", "hailo_8"}:
                backend = "hailo8"
            elif full in {"hailo10", "hailo10h", "hailo_10", "hailo_10h"}:
                backend = "hailo10h"
            elif full in {"deepx", "deepx_m1", "dx_m1"}:
                backend = "deepx"
        if backend:
            selected.add(backend)
    return [name for name in _NATIVE_FULL_BACKEND_ORDER if name in selected]


def native_energy_requested(profile: Mapping[str, Any]) -> bool:
    """Read the persisted Native-energy master switch across schema revisions."""
    native = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    native_energy = native.get("energy") if isinstance(native.get("energy"), Mapping) else {}
    if "enabled" in native_energy:
        return bool(native_energy.get("enabled"))
    energy = profile.get("energy") if isinstance(profile.get("energy"), Mapping) else {}
    if "requested_native_energy" in energy:
        return bool(energy.get("requested_native_energy"))
    preset = profile.get("execution_preset") if isinstance(profile.get("execution_preset"), Mapping) else {}
    overrides = preset.get("overrides") if isinstance(preset.get("overrides"), Mapping) else {}
    if "energy_enabled" in overrides:
        return bool(overrides.get("energy_enabled"))
    # Legacy pre-native-only profiles used the Generic energy flag as the master.
    return bool(energy.get("enabled"))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def default_run_modes_path() -> Path:
    override = str(os.environ.get("ONNX_SPLITPOINT_RUN_MODES_FILE", "") or "").strip()
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override))).resolve()
    return Path.home() / ".onnx_splitpoint_tool" / "run_modes.yaml"


def _mode(
    *,
    label: str,
    description: str,
    recommended_for: str,
    default_native: bool,
    default_energy: bool,
    integrity: str,
    campaign_mode: str,
    campaign_enforcement: str,
    campaign_frozen: bool,
    use_final_registry: bool,
    calibration_items: int,
    validation_cls: int,
    validation_det: int,
    bootstrap: int,
    ranking_bootstrap: int,
    hailo_opt: int,
    hailo_preset: str,
    hailo_timeout: int,
    benchmark_warmup: int,
    benchmark_runs: int,
    benchmark_timeout: int,
    native_frames: int,
    native_warmup: int,
    native_full: bool,
    official_coco: bool,
    official_coco_required: bool,
    ranking_enabled: bool,
    require_holdout_freeze: bool,
    thesis_outputs: bool,
    energy_repeats: int,
    energy_policy: str,
    claim_scope: str = EVALUATED_MATRIX_CLAIM_SCOPE,
) -> Dict[str, Any]:
    final = campaign_mode == "final"
    ranking_generalization = claim_scope == RANKING_GENERALIZATION_CLAIM_SCOPE
    return {
        "label": label,
        "description": description,
        "recommended_for": recommended_for,
        "defaults": {
            "native_enabled": bool(default_native),
            "energy_enabled": bool(default_energy),
        },
        "reproducibility": {
            "level": integrity,  # relaxed | strict
            "verify_model_content": bool(final),
            "verify_dataset_content": bool(final),
            "verify_remote_scripts": "strict" if final else "capabilities",
            "cache_unchanged_files": not final,
            "dataset_sample_size": 0 if final else (4 if label.lower().startswith("smoke") else 24),
        },
        "campaign": {
            "claim_scope": claim_scope,
            "mode": campaign_mode,
            "enforcement": campaign_enforcement,
            "frozen_before_final_campaign": bool(campaign_frozen),
            "auto_bind_dataset_registry": bool(use_final_registry),
            "require_protocol_freeze": bool(ranking_generalization),
            "require_fitted_stage_time": bool(ranking_generalization),
            "require_native_handover_model": bool(ranking_generalization),
            "require_campaign_freeze": False,
            "require_prediction_freeze_approval": bool(ranking_generalization and require_holdout_freeze),
            "require_cryptographic_prediction_signature": False,
        },
        "data": {
            "use_final_dataset_registry": bool(use_final_registry),
            "calibration_items": {
                "classification": int(calibration_items),
                "detection": int(calibration_items),
            },
            # 0 means full manifest.  Packaged Final defaults to 5,000 items per task;
            # users can set 0 centrally for the complete ImageNet/COCO manifests.
            "validation_items": {
                "classification": int(validation_cls),
                "detection": int(validation_det),
            },
        },
        "build": {
            "model_preparation": "current",
            "hailo": {
                "mode": "reuse_and_build_missing",
                "preset": hailo_preset,
                "optimization_level": int(hailo_opt),
                "calibration_items": int(calibration_items),
                "calibration_batch_size": 8 if calibration_items >= 8 else max(1, calibration_items),
                "calibration_storage": "memory" if label.lower().startswith("smoke") else "memmap",
                "calibration_memory_cap_mb": 256,
                "cache_enabled": True,
                "cache_root": "~/.cache/onnx_splitpoint/hailo_hef",
                "cache_integrity": "strict" if final else "relaxed",
                "timeout_s": int(hailo_timeout),
                # v60u: Smoke must not spend its whole budget compiling a heavy
                # uncached Full baseline.  It reuses an exact HEF/Artifact-Library
                # hit or records a deferred cold-build request.  Standard and
                # Final still build every missing required baseline.
                "full_baseline_cold_build_policy": (
                    "cache_or_defer" if label.lower().startswith("smoke") else "build_missing"
                ),
                "deferred_full_baseline_required": not label.lower().startswith("smoke"),
                "cold_build_timeout_s": max(int(hailo_timeout), 3600),
                "preserve_attempt_provenance": True,
                "build_full": True,
                "build_part1": True,
                "build_part2": True,
                # Strict cache identity makes reuse safe in Final; an explicit
                # central override can still force a clean rebuild.
                "force_build": False,
                "keep_artifacts": bool(final),
            },
            "deepx": {
                "mode": "reuse_and_build_missing",
                "optimization_level": 0 if not final else 1,
                "calibration_items": int(calibration_items),
                "calibration_method": "ema",
                "force_build": False,
            },
            "artifact_store": {
                "enabled": True,
                "root": "~/.onnx_splitpoint_tool/artifact_store",
                "verify_on_reuse": "strict" if final else "metadata",
                "pin_for_campaign": bool(final),
                "register_hailo": True,
                "register_deepx": True,
                "register_tensorrt": False,
            },
            "scheduler": {
                "enabled": True,
                "max_workers": 2 if final else 3,
                "cpu_tokens": 0,
                "ram_mb": 0,
                "family_limits": {"hailo8": 1, "hailo10": 1, "deepx": 1},
                "weights": {
                    "hailo8": {"cpu_tokens": 4, "ram_mb": 6144},
                    "hailo10": {"cpu_tokens": 4, "ram_mb": 6144},
                    "deepx": {"cpu_tokens": 2, "ram_mb": 4096},
                },
                "prefetch_deepx_full": True,
                "pipeline_next_model": False,
            },
            "hardware_smoke": "strict" if final else "summary_only",
        },
        "runtime": {
            "execution_mode": "generate_and_run",
            "skip_runtime_benchmarks": False,
            "benchmark": {
                "provider": "auto",
                "warmup": int(benchmark_warmup),
                "runs": int(benchmark_runs),
                "timeout_s": int(benchmark_timeout),
            },
            "parallel": {
                "remote_setups": True,
                "max_setups": 3,
                "max_uploads": 1,
                "powercalc_workers": 1,
            },
            "remote_cache": {
                "upload_once_per_model_setup": True,
                "reuse_tensorrt_engines": True,
                "stable_cache_root": "~/.cache/onnx_splitpoint",
                "suite_cache_root": "~/.cache/onnx_splitpoint/suites",
                "engine_cache_root": "~/.cache/onnx_splitpoint/tensorrt",
                "copy_strategy": "auto",
            },
            "native": {
                "backends": ["hailo8", "hailo10h", "deepx"],
                "case_policy": "all_accepted",
                "precision": "uint8_cast_fp16",
                "frames": int(native_frames),
                "warmup": int(native_warmup),
                # Independent end-to-end measurements.  Aggregate performance
                # is the median with a 95 % interval; no repeat is selected as
                # a best-of result.
                "repetitions": 1 if label.lower().startswith("smoke") else (5 if final else 3),
                "queue_depth": 2 if label.lower().startswith("smoke") else 3,
                "inflight": 4 if label.lower().startswith("smoke") else 8,
                "full_baselines": bool(native_full),
                "validation": True,
                "dump_outputs": True,
                "build_missing_engines": True,
                "copy_benchmarksets": True,
                "strict_supported_only": True,
                "contract_selection": "metadata_first",
                "preserve_row_failures": True,
                "require_contract_family_match": True,
            },
        },
        "quality": {
            "profile_id": f"task_quality_{campaign_mode}_{bootstrap}",
            "dataset_tier": "final" if final else "screening",
            "canonical_reference": "canonical_full_onnx",
            "classification_margin_pp": 1.0,
            "classification_top5_margin_pp": 1.0,
            "detection_margin_ap": 1.0,
            "detection_ap50_margin": 1.0,
            "detection_ap75_margin": 1.0,
            "confidence_level": 0.95,
            "bootstrap_repetitions": int(bootstrap),
            "bootstrap_seed": 20260710,
            # Every run mode uses exactly the same semantic-reference contract:
            # one ORT-CPU reference is produced on the management node and the
            # paired uncertainty calculation runs in the central service.  The
            # Smoke budget remains small through its 25 resamples and tiny
            # validation subset, not by moving this work back to an accelerator
            # host.  Four workers configure both ORT inference threads and the
            # deterministic bootstrap service; the reference is never a
            # latency/FPS/energy result.
            "execution_location": "central_management",
            "workers": 4,
            "cache_task_quality": True,
            "cadence": "once_per_artifact",
            "official_coco_enabled": bool(official_coco),
            "official_coco_required": bool(official_coco_required),
            "archive_coco_eval_tensors": bool(final),
        },
        "holdout": {
            "candidate_universe": "deterministic_audit",
            "audit_size": 12 if label.lower().startswith("smoke") else 20,
            "minimum_valid_candidates": 10,
            "seed": 20260710,
            "prediction_freeze_enabled": bool(ranking_generalization and require_holdout_freeze),
            "require_complete_candidate_universe": bool(ranking_generalization and final),
            "require_frozen_predictions": bool(ranking_generalization and require_holdout_freeze),
            "require_unseen_attestation": bool(ranking_generalization and final),
        },
        "ranking": {
            "enabled": bool(ranking_enabled),
            "methods": [
                "cut_bytes_only",
                "weighted_score",
                "cycle_time_no_handover",
                "cycle_time_with_handover",
                "onnx_real_boundary_hardware_aware",
            ],
            "k_values": [1, 3, 5],
            "elite_q_values": [1, 3],
            "primary_k": 5,
            "minimum_candidates_for_correlation": 3,
            "near_optimal_relative_epsilon": 0.01,
            "bootstrap_repetitions": int(ranking_bootstrap),
            "bootstrap_seed": 20260710,
        },
        "reporting": {
            "include_decision_summary": True,
            "include_backend_drift_block": True,
            "include_task_specific_quality_block": True,
            "canonical_scientific_report": True,
            "cleanup_legacy_reports": True,
            "generate_thesis_tex": bool(thesis_outputs),
            "generate_thesis_figures": bool(thesis_outputs),
            "include_campaign_readiness": True,
        },
        "energy": {
            "scope": "row_variant",
            "physical_scope": "FS",
            "window_label": "command",
            "repeats": int(energy_repeats),
            "phases": ["streaming"],
            "target_policy": energy_policy,
            "skip_backends": ["ort_cpu"],
            "heartbeat_s": 60,
            "max_targets_per_run_id": 0 if final else 1,
            "max_work_units_per_window": 0,
            "max_window_duration_s": 0,
            "timeout_s_per_window": 0,
            "sizing_probe_max_work_units": 256,
            "include_raw_parquet_in_debug_pack": bool(final),
            "strict": bool(final),
            "native_mode": "measure" if final else "plan",
            "native_duration_s": 0,
        },
    }


def _legacy_v11_final_mode() -> Dict[str, Any]:
    """Return the strict Final default shipped through release 2.75.28.

    The value is kept only for a value-sensitive registry migration.  It must
    not be used for new profiles: since schema v12, the user-facing Final mode
    is deliberately Standard plus a stronger task-quality budget.
    """

    return _mode(
        label="Final",
        description="Maximum-effort frozen evaluated-matrix campaign with 5,000 validation items per task, strict provenance and thesis reporting.",
        recommended_for="Thesis tables for the declared workload/hardware matrix, Native/Full baselines and calibrated system-energy results.",
        default_native=True,
        default_energy=True,
        integrity="strict",
        campaign_mode="final",
        campaign_enforcement="strict",
        campaign_frozen=True,
        use_final_registry=True,
        calibration_items=500,
        validation_cls=5000,
        validation_det=5000,
        bootstrap=5000,
        ranking_bootstrap=5000,
        hailo_opt=2,
        hailo_preset="final",
        hailo_timeout=10800,
        benchmark_warmup=10,
        benchmark_runs=10,
        benchmark_timeout=0,
        native_frames=5000,
        native_warmup=500,
        native_full=True,
        official_coco=True,
        official_coco_required=True,
        ranking_enabled=False,
        require_holdout_freeze=False,
        thesis_outputs=True,
        energy_repeats=5,
        energy_policy="all",
        claim_scope=EVALUATED_MATRIX_CLAIM_SCOPE,
    )


def _final_quality_mode(standard_mode: Mapping[str, Any]) -> Dict[str, Any]:
    """Build Final Quality as Standard with only a stronger quality budget.

    This intentionally retains the complete Standard execution path: relaxed
    sampled integrity checks, balanced compiler settings, timing and Native
    budgets, cache policy, reporting and energy behaviour.  The only semantic
    effort increase is the task-quality evaluation itself.
    """

    mode = copy.deepcopy(dict(standard_mode))
    mode.update({
        "label": "Final Quality (Standard+)",
        "description": (
            "Standard execution with stronger final-quality evaluation: "
            "5,000 validation items per task and 5,000 bootstrap repetitions."
        ),
        "recommended_for": (
            "Final thesis-quality model and split-point results without a "
            "separate freeze, sealing or full-dataset hashing workflow."
        ),
    })
    mode["data"]["validation_items"] = {
        "classification": 5000,
        "detection": 5000,
    }
    mode["quality"].update({
        "profile_id": "task_quality_final_5000",
        "dataset_tier": "final",
        "bootstrap_repetitions": 5000,
    })
    return mode


def default_run_modes_config() -> Dict[str, Any]:
    standard_mode = _mode(
        label="Standard (balanced)",
        description="Balanced development run with useful accuracy screening and moderate build effort.",
        recommended_for="Normal development results, candidate comparison and pre-final hardware campaigns.",
        default_native=True,
        default_energy=False,
        integrity="relaxed",
        campaign_mode="development",
        campaign_enforcement="warn",
        campaign_frozen=False,
        use_final_registry=True,
        calibration_items=500,
        validation_cls=500,
        validation_det=500,
        bootstrap=500,
        ranking_bootstrap=500,
        hailo_opt=1,
        hailo_preset="balanced",
        hailo_timeout=3600,
        benchmark_warmup=3,
        benchmark_runs=5,
        benchmark_timeout=0,
        native_frames=1000,
        native_warmup=100,
        native_full=True,
        official_coco=True,
        official_coco_required=False,
        ranking_enabled=True,
        require_holdout_freeze=False,
        thesis_outputs=True,
        energy_repeats=3,
        energy_policy="best_valid_only",
    )
    return {
        "schema": RUN_MODE_SCHEMA,
        "schema_version": RUN_MODE_SCHEMA_VERSION,
        "default_mode": DEFAULT_MODE_ID,
        "modes": {
            "smoke": _mode(
                label="Smoke",
                description="Fastest end-to-end check. Minimal calibration, a tiny validation sample and one timing run.",
                recommended_for="GUI/remote wiring, compiler/runtime regression and contract smoke tests.",
                default_native=False,
                default_energy=False,
                integrity="relaxed",
                campaign_mode="development",
                campaign_enforcement="warn",
                campaign_frozen=False,
                use_final_registry=True,
                calibration_items=8,
                validation_cls=16,
                validation_det=12,
                bootstrap=25,
                ranking_bootstrap=50,
                hailo_opt=0,
                hailo_preset="smoke",
                hailo_timeout=900,
                benchmark_warmup=1,
                benchmark_runs=1,
                benchmark_timeout=3600,
                native_frames=100,
                native_warmup=10,
                native_full=False,
                official_coco=False,
                official_coco_required=False,
                ranking_enabled=False,
                require_holdout_freeze=False,
                thesis_outputs=False,
                energy_repeats=1,
                energy_policy="best_valid_only",
            ),
            "standard": standard_mode,
            "final": _final_quality_mode(standard_mode),
        },
    }


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        out = {str(k): copy.deepcopy(v) for k, v in base.items()}
        for key, value in override.items():
            if key in out:
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = copy.deepcopy(value)
        return out
    return copy.deepcopy(override)


def _replace_matching_defaults(
    current: MutableMapping[str, Any],
    old_default: Mapping[str, Any],
    new_default: Mapping[str, Any],
) -> None:
    """Migrate unchanged default leaves while preserving user customisations."""

    for key, new_value in new_default.items():
        if key not in current or key not in old_default:
            continue
        old_value = old_default[key]
        current_value = current[key]
        if (
            isinstance(current_value, MutableMapping)
            and isinstance(old_value, Mapping)
            and isinstance(new_value, Mapping)
        ):
            _replace_matching_defaults(current_value, old_value, new_value)
        elif current_value == old_value:
            current[key] = copy.deepcopy(new_value)


def _migrate_run_modes_config_v60p(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Upgrade untouched pre-v60q Smoke defaults without overriding user choices.

    The central ``~/.onnx_splitpoint_tool/run_modes.yaml`` intentionally survives
    tool upgrades.  v60n introduced schema version 2 before the smaller v60p
    Smoke budgets were finalised.  Consequently existing schema-v2 files could
    retain the old 32/25 validation budgets, 50/100 bootstrap settings and the
    600 s remote timeout.  Schema version 4 performs a value-sensitive migration:
    only values that still exactly match those historical defaults are changed.
    Explicit user customisations are preserved.
    """
    data = copy.deepcopy(dict(payload or {}))
    try:
        version = int(data.get("schema_version") or 1)
    except Exception:
        version = 1
    if version >= RUN_MODE_SCHEMA_VERSION:
        return data
    modes = data.get("modes") if isinstance(data.get("modes"), Mapping) else {}
    smoke = modes.get("smoke") if isinstance(modes.get("smoke"), MutableMapping) else None
    if smoke is not None:
        campaign = smoke.get("campaign") if isinstance(smoke.get("campaign"), MutableMapping) else {}
        data_cfg = smoke.get("data") if isinstance(smoke.get("data"), MutableMapping) else {}
        runtime = smoke.get("runtime") if isinstance(smoke.get("runtime"), MutableMapping) else {}
        bench = runtime.get("benchmark") if isinstance(runtime.get("benchmark"), MutableMapping) else {}
        quality = smoke.get("quality") if isinstance(smoke.get("quality"), MutableMapping) else {}
        ranking = smoke.get("ranking") if isinstance(smoke.get("ranking"), MutableMapping) else {}
        val = data_cfg.get("validation_items") if isinstance(data_cfg.get("validation_items"), MutableMapping) else {}
        if campaign.get("auto_bind_dataset_registry") is False:
            campaign["auto_bind_dataset_registry"] = True
        if data_cfg.get("use_final_dataset_registry") is False:
            data_cfg["use_final_dataset_registry"] = True
        if int(val.get("classification") or 0) == 32 and int(val.get("detection") or 0) == 25:
            val["classification"], val["detection"] = 16, 12
        if int(quality.get("bootstrap_repetitions") or 0) == 50:
            quality["bootstrap_repetitions"] = 25
            if str(quality.get("profile_id") or "") == "task_quality_development_50":
                quality["profile_id"] = "task_quality_development_25"
        if int(ranking.get("bootstrap_repetitions") or 0) == 100:
            ranking["bootstrap_repetitions"] = 50
        if int(bench.get("timeout_s") or 0) == 600:
            bench["timeout_s"] = 1800
        # Schema v9 raises only the untouched v8 Smoke remote-execution
        # timeout.  A real three-model hardware Smoke can legitimately spend
        # more than 30 minutes building the setup-local TensorRT quality
        # engines on the DeepX host.  Keep explicit user values intact.
        if version < 9 and int(bench.get("timeout_s") or 0) == 1800:
            bench["timeout_s"] = 3600
        # Schema v8 activates the management-node semantic reference and the
        # central paired-quality service for Smoke as well.  Migrate only the
        # legacy Smoke execution defaults (local/one worker).  Explicit custom
        # worker counts are retained, while an absent worker value receives the
        # new four-worker default.  This migration is persisted by
        # ``load_run_modes_config`` so an existing Tool Config installation is
        # updated once instead of silently retaining the v2.63 behaviour.
        if version < 8:
            execution_location = str(quality.get("execution_location") or "").strip().lower().replace("-", "_")
            legacy_local = execution_location in {"", "local"}
            try:
                workers = int(quality.get("workers") or 0)
            except (TypeError, ValueError):
                workers = 0
            if legacy_local and workers in {0, 1}:
                quality["execution_location"] = "central_management"
                quality["workers"] = 4
        build_cfg = smoke.get("build") if isinstance(smoke.get("build"), MutableMapping) else {}
        hailo_cfg = build_cfg.get("hailo") if isinstance(build_cfg.get("hailo"), MutableMapping) else {}
        if not str(hailo_cfg.get("full_baseline_cold_build_policy") or "").strip():
            hailo_cfg["full_baseline_cold_build_policy"] = "cache_or_defer"
        # Missing legacy values receive the historic finite Smoke budget.
        # An explicit zero/off token is authoritative and must not be migrated
        # back to a timeout now that unlimited Full builds are a public
        # profile contract.
        if "cold_build_timeout_s" not in hailo_cfg:
            hailo_cfg["cold_build_timeout_s"] = max(int(hailo_cfg.get("timeout_s") or 900), 3600)
        if "preserve_attempt_provenance" not in hailo_cfg:
            hailo_cfg["preserve_attempt_provenance"] = True
    # Schema v11 separates measured-matrix claims from prospective ranking
    # generalisation.  A byte-for-value packaged v10 Final mode is migrated to
    # the new evaluated-matrix default.  Any customised/hold-out-oriented mode
    # remains conservative and receives the strict scope instead.
    final_mode = modes.get("final") if isinstance(modes.get("final"), MutableMapping) else None
    if version < 11 and final_mode is not None:
        campaign = final_mode.get("campaign") if isinstance(final_mode.get("campaign"), MutableMapping) else {}
        if not str(campaign.get("claim_scope") or "").strip():
            historical_final = _legacy_v11_final_mode()
            historical_final["description"] = "Maximum-effort frozen campaign with 5,000 validation items per task by default, strict provenance and thesis reporting."
            historical_final["recommended_for"] = "Thesis tables, hold-out claims, native/full baselines and calibrated system-energy results."
            historical_campaign = historical_final["campaign"]
            historical_campaign.pop("claim_scope", None)
            historical_campaign.pop("require_protocol_freeze", None)
            historical_campaign.update({
                "require_fitted_stage_time": True,
                "require_native_handover_model": True,
                "require_campaign_freeze": True,
                "require_prediction_freeze_approval": True,
            })
            historical_final["ranking"]["enabled"] = True
            historical_final["holdout"].update({
                "prediction_freeze_enabled": True,
                "require_complete_candidate_universe": True,
                "require_frozen_predictions": True,
                "require_unseen_attestation": True,
            })
            historical_final["energy"]["phases"] = ["latency", "streaming"]
            untouched_packaged_final = dict(final_mode) == historical_final
            if untouched_packaged_final:
                new_final = default_run_modes_config()["modes"]["final"]
                final_mode.clear()
                final_mode.update(copy.deepcopy(new_final))
            else:
                ranking = final_mode.get("ranking") if isinstance(final_mode.get("ranking"), Mapping) else {}
                holdout = final_mode.get("holdout") if isinstance(final_mode.get("holdout"), Mapping) else {}
                strict_intent = bool(
                    ranking.get("enabled")
                    or campaign.get("require_fitted_stage_time")
                    or campaign.get("require_native_handover_model")
                    or campaign.get("require_prediction_freeze_approval")
                    or holdout.get("prediction_freeze_enabled")
                    or holdout.get("require_frozen_predictions")
                    or holdout.get("require_unseen_attestation")
                )
                campaign["claim_scope"] = (
                    RANKING_GENERALIZATION_CLAIM_SCOPE
                    if strict_intent
                    else EVALUATED_MATRIX_CLAIM_SCOPE
                )
                campaign.setdefault("require_protocol_freeze", strict_intent)
    # Schema v12 replaces the user-facing strict/frozen Final default with
    # Final Quality (Standard+).  Migrate every leaf that still has the exact
    # v11 packaged value and preserve genuinely customised leaves.  This is
    # important because ~/.onnx_splitpoint_tool/run_modes.yaml intentionally
    # survives source upgrades.
    if version < 12 and final_mode is not None:
        _replace_matching_defaults(
            final_mode,
            _legacy_v11_final_mode(),
            default_run_modes_config()["modes"]["final"],
        )
    # Schema v13 makes the physical Full-System/command-window contract
    # explicit in every run mode.  Migrate only missing values and the exact
    # historical MB default; malformed or genuinely custom values remain
    # visible to validation instead of being silently relabelled.
    if version < 13:
        for raw_mode in modes.values():
            if not isinstance(raw_mode, MutableMapping):
                continue
            energy = raw_mode.get("energy")
            if not isinstance(energy, MutableMapping):
                energy = {}
                raw_mode["energy"] = energy
            if (
                "physical_scope" not in energy
                or energy.get("physical_scope") == "MB"
            ):
                energy["physical_scope"] = "FS"
            if "window_label" not in energy:
                energy["window_label"] = "command"
    data["schema_version"] = RUN_MODE_SCHEMA_VERSION
    return data


def validate_run_modes_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Run-mode configuration must be a mapping")
    migrated = _migrate_run_modes_config_v60p(payload)
    data = _deep_merge(default_run_modes_config(), migrated)
    if str(data.get("schema") or RUN_MODE_SCHEMA) != RUN_MODE_SCHEMA:
        raise ValueError(f"Unsupported run-mode schema: {data.get('schema')!r}")
    modes = data.get("modes")
    if not isinstance(modes, Mapping):
        raise ValueError("run_modes.modes must be a mapping")
    for mode_id in _MODE_ORDER:
        mode = modes.get(mode_id)
        if not isinstance(mode, Mapping):
            raise ValueError(f"Missing run mode: {mode_id}")
        q = mode.get("quality") if isinstance(mode.get("quality"), Mapping) else {}
        b = mode.get("build") if isinstance(mode.get("build"), Mapping) else {}
        h = b.get("hailo") if isinstance(b.get("hailo"), Mapping) else {}
        rt = mode.get("runtime") if isinstance(mode.get("runtime"), Mapping) else {}
        bm = rt.get("benchmark") if isinstance(rt.get("benchmark"), Mapping) else {}
        data_cfg = mode.get("data") if isinstance(mode.get("data"), Mapping) else {}
        campaign = mode.get("campaign") if isinstance(mode.get("campaign"), Mapping) else {}
        claim_scope = str(campaign.get("claim_scope") or "").strip()
        if claim_scope not in CAMPAIGN_CLAIM_SCOPES:
            raise ValueError(f"Invalid {mode_id}.campaign.claim_scope: {claim_scope!r}")
        for key, value, lower in (
            (f"{mode_id}.quality.bootstrap_repetitions", q.get("bootstrap_repetitions"), 0),
            (f"{mode_id}.quality.workers", q.get("workers"), 1),
            (f"{mode_id}.build.hailo.calibration_items", h.get("calibration_items"), 1),
            (f"{mode_id}.runtime.benchmark.runs", bm.get("runs"), 1),
        ):
            try:
                if int(value) < lower:
                    raise ValueError
            except Exception as exc:
                raise ValueError(f"Invalid {key}: {value!r}") from exc
        quality_location = str(q.get("execution_location") or "").strip()
        if quality_location not in {"local", "central_management"}:
            raise ValueError(
                f"Invalid {mode_id}.quality.execution_location: {quality_location!r}"
            )
        if int(q.get("workers") or 0) > 64:
            raise ValueError(f"Invalid {mode_id}.quality.workers: {q.get('workers')!r}")
        full_cache_policy = str(h.get("full_baseline_cold_build_policy") or "build_missing").strip().lower()
        if full_cache_policy not in {"cache_or_defer", "build_missing"}:
            raise ValueError(f"Invalid {mode_id}.build.hailo.full_baseline_cold_build_policy: {full_cache_policy!r}")
        val_items = data_cfg.get("validation_items") if isinstance(data_cfg.get("validation_items"), Mapping) else {}
        for task in ("classification", "detection"):
            try:
                if int(val_items.get(task, 0)) < 0:
                    raise ValueError
            except Exception as exc:
                raise ValueError(f"Invalid {mode_id}.data.validation_items.{task}") from exc
    default_mode = str(data.get("default_mode") or DEFAULT_MODE_ID)
    if default_mode not in modes:
        data["default_mode"] = DEFAULT_MODE_ID
    data["schema"] = RUN_MODE_SCHEMA
    data["schema_version"] = RUN_MODE_SCHEMA_VERSION
    return data


def ensure_run_modes_file(path: str | Path | None = None) -> Path:
    dst = Path(path).expanduser() if path else default_run_modes_path()
    dst = dst.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        save_run_modes_config(dst, default_run_modes_config())
    return dst


def load_run_modes_config(path: str | Path | None = None) -> Dict[str, Any]:
    src = ensure_run_modes_file(path)
    try:
        raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"Could not read run-mode configuration {src}: {exc}") from exc
    raw_mapping = raw if isinstance(raw, Mapping) else {}
    data = validate_run_modes_config(raw_mapping)
    try:
        raw_version = int(raw_mapping.get("schema_version") or 1)
    except Exception:
        raw_version = 1
    if raw_version < RUN_MODE_SCHEMA_VERSION:
        # Persist value-sensitive migrations once so the GUI and later CLI
        # invocations see the same Smoke budget without reapplying compatibility
        # logic on every load.  save_run_modes_config validates again but does
        # not call load_run_modes_config, so this is not recursive.
        save_run_modes_config(src, data)
    return data


def save_run_modes_config(path: str | Path | None, payload: Mapping[str, Any]) -> Path:
    dst = Path(path).expanduser().resolve() if path else default_run_modes_path().resolve()
    data = validate_run_modes_config(payload)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, dst)
    return dst


def reset_run_mode(mode_id: str, path: str | Path | None = None) -> Path:
    mid = normalize_mode_id(mode_id)
    data = load_run_modes_config(path)
    data["modes"][mid] = copy.deepcopy(default_run_modes_config()["modes"][mid])
    return save_run_modes_config(path, data)


def normalize_mode_id(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "quick": "smoke",
        "test": "smoke",
        "smoke_test": "smoke",
        "balanced": "standard",
        "development": "standard",
        "dev": "standard",
        "medium": "standard",
        "normal": "standard",
        "production": "final",
        "thesis": "final",
        "claim": "final",
    }
    return aliases.get(raw, raw if raw in _MODE_ORDER else DEFAULT_MODE_ID)


def get_run_mode(mode_id: str, config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    cfg = validate_run_modes_config(config or load_run_modes_config())
    mid = normalize_mode_id(mode_id or cfg.get("default_mode"))
    return copy.deepcopy(dict(cfg["modes"][mid]))


def resolve_run_mode(mode_id: str, config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Backward-compatible public alias for :func:`get_run_mode`.

    v60s tests and third-party integrations used this name while the internal
    implementation was renamed.  Keep the alias stable for source and wheel
    consumers.
    """
    return get_run_mode(mode_id, config)


def run_mode_display_rows(config: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    cfg = validate_run_modes_config(config or load_run_modes_config())
    rows: list[dict[str, str]] = []
    for mode_id in _MODE_ORDER:
        mode = dict(cfg["modes"][mode_id])
        rows.append({
            "id": mode_id,
            "label": str(mode.get("label") or mode_id.title()),
            "description": str(mode.get("description") or ""),
            "recommended_for": str(mode.get("recommended_for") or ""),
        })
    return rows


def mode_summary(mode_id: str, config: Mapping[str, Any] | None = None) -> str:
    mode = get_run_mode(mode_id, config)
    data = mode.get("data") if isinstance(mode.get("data"), Mapping) else {}
    calib = data.get("calibration_items") if isinstance(data.get("calibration_items"), Mapping) else {}
    val = data.get("validation_items") if isinstance(data.get("validation_items"), Mapping) else {}
    quality = mode.get("quality") if isinstance(mode.get("quality"), Mapping) else {}
    hailo = ((mode.get("build") or {}).get("hailo") or {}) if isinstance(mode.get("build"), Mapping) else {}
    runtime = mode.get("runtime") if isinstance(mode.get("runtime"), Mapping) else {}
    bench = runtime.get("benchmark") if isinstance(runtime.get("benchmark"), Mapping) else {}
    native = runtime.get("native") if isinstance(runtime.get("native"), Mapping) else {}
    repro = mode.get("reproducibility") if isinstance(mode.get("reproducibility"), Mapping) else {}
    cls_val = int(val.get("classification") or 0)
    det_val = int(val.get("detection") or 0)
    val_text = f"CLS {cls_val or 'full'} / DET {det_val or 'full'}"
    return (
        f"{mode.get('label', mode_id)} — {mode.get('description', '')}\n"
        f"Calibration: CLS {calib.get('classification', '?')} / DET {calib.get('detection', '?')} · "
        f"Validation: {val_text} · Bootstrap: {quality.get('bootstrap_repetitions', '?')}\n"
        f"Hailo: preset={hailo.get('preset', '?')}, opt={hailo.get('optimization_level', '?')} · "
        f"Timing: warmup={bench.get('warmup', '?')}, runs={bench.get('runs', '?')} · "
        f"Native: {native.get('frames', '?')} frames · Reproducibility: {repro.get('level', '?')}"
    )


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def infer_run_mode(profile: Mapping[str, Any]) -> str:
    preset = profile.get("execution_preset") if isinstance(profile, Mapping) else None
    if isinstance(preset, Mapping) and str(preset.get("id") or "").strip():
        return normalize_mode_id(preset.get("id"))
    campaign = profile.get("campaign") if isinstance(profile.get("campaign"), Mapping) else {}
    if str(campaign.get("mode") or "").lower() == "final" or bool(campaign.get("frozen_before_final_campaign")):
        return "final"
    quality = profile.get("quality_gate") if isinstance(profile.get("quality_gate"), Mapping) else {}
    stats = quality.get("statistics") if isinstance(quality.get("statistics"), Mapping) else {}
    hailo = profile.get("hailo_build") if isinstance(profile.get("hailo_build"), Mapping) else {}
    try:
        boot = int(stats.get("bootstrap_repetitions") or 0)
    except Exception:
        boot = 0
    try:
        calib = int(hailo.get("calib_count") or 0)
    except Exception:
        calib = 0
    bench = profile.get("benchmark_execution") if isinstance(profile.get("benchmark_execution"), Mapping) else {}
    try:
        runs = int(bench.get("runs") or 0)
    except Exception:
        runs = 0
    if boot <= 100 and calib <= 32 and runs <= 1:
        return "smoke"
    return "standard"


def _task_calibration_items(mode: Mapping[str, Any]) -> Dict[str, int]:
    data = mode.get("data") if isinstance(mode.get("data"), Mapping) else {}
    raw = data.get("calibration_items") if isinstance(data.get("calibration_items"), Mapping) else {}
    return {
        "classification": max(1, int(raw.get("classification") or 1)),
        "detection": max(1, int(raw.get("detection") or 1)),
    }


def _materialized_blocks(mode_id: str, mode: Mapping[str, Any], *, profile_name: str) -> Dict[str, Any]:
    campaign_cfg = dict(mode.get("campaign") or {})
    data_cfg = dict(mode.get("data") or {})
    build_cfg = dict(mode.get("build") or {})
    runtime_cfg = dict(mode.get("runtime") or {})
    quality_cfg = dict(mode.get("quality") or {})
    holdout_cfg = dict(mode.get("holdout") or {})
    ranking_cfg = dict(mode.get("ranking") or {})
    reporting_cfg = dict(mode.get("reporting") or {})
    energy_cfg = dict(mode.get("energy") or {})
    repro_cfg = dict(mode.get("reproducibility") or {})
    hailo_cfg = dict(build_cfg.get("hailo") or {})
    deepx_cfg = dict(build_cfg.get("deepx") or {})
    artifact_store_cfg = dict(build_cfg.get("artifact_store") or {})
    scheduler_cfg = dict(build_cfg.get("scheduler") or {})
    benchmark_cfg = dict(runtime_cfg.get("benchmark") or {})
    parallel_cfg = dict(runtime_cfg.get("parallel") or {})
    native_cfg = dict(runtime_cfg.get("native") or {})
    val_items = dict(data_cfg.get("validation_items") or {})
    calib_items = _task_calibration_items(mode)
    final = str(campaign_cfg.get("mode") or "").lower() == "final"
    claim_scope = str(campaign_cfg.get("claim_scope") or EVALUATED_MATRIX_CLAIM_SCOPE)
    ranking_generalization = claim_scope == RANKING_GENERALIZATION_CLAIM_SCOPE
    integrity_level = str(repro_cfg.get("level") or ("strict" if final else "relaxed")).lower()
    integrity_mode = "strict" if integrity_level == "strict" else ("fast" if mode_id == "smoke" else "relaxed")
    hailo_timeout_s = parse_hailo_timeout_seconds(
        hailo_cfg.get("timeout_s"),
        default=0,
        label="build.hailo.timeout_s",
    )
    hailo_cold_timeout_s = parse_hailo_timeout_seconds(
        (
            hailo_cfg.get("cold_build_timeout_s")
            if "cold_build_timeout_s" in hailo_cfg
            else hailo_timeout_s
        ),
        default=hailo_timeout_s,
        label="build.hailo.cold_build_timeout_s",
    )

    deepx_build_block: Dict[str, Any] = {
        "mode": str(deepx_cfg.get("mode") or "reuse_and_build_missing"),
        "target": "deepx_m1",
        "calib_dir": "",
        "calib_count": max(calib_items.values()),
        "calibration_method": str(deepx_cfg.get("calibration_method") or "ema"),
        "opt_level": max(0, int(deepx_cfg.get("optimization_level") or 0)),
        "force_build": bool(deepx_cfg.get("force_build")),
    }
    # DeepX classification A/B is deliberately opt-in.  Absence preserves the
    # historical Full-DXNN cache path for existing Standard/YOLO profiles;
    # frozen diagnostic profiles project the explicit arm byte-for-byte.
    if "classification_preprocessing" in deepx_cfg:
        classification_preprocessing = str(
            deepx_cfg.get("classification_preprocessing") or ""
        ).strip().lower()
        if classification_preprocessing not in {
            "current_scale_only", "imagenet_mean_std",
        }:
            raise ValueError(
                "build.deepx.classification_preprocessing must be "
                "current_scale_only or imagenet_mean_std"
            )
        deepx_build_block["classification_preprocessing"] = (
            classification_preprocessing
        )
    if "cache_dir" in deepx_cfg:
        deepx_build_block["cache_dir"] = str(
            deepx_cfg.get("cache_dir") or ""
        ).strip()

    blocks: Dict[str, Any] = {
        "integrity_policy": {
            "mode": integrity_mode,
            "effective_mode": integrity_mode,
            "strict_required_for_final": True,
            "cache_unchanged_files": bool(repro_cfg.get("cache_unchanged_files", not final)),
            "dataset_sample_size": max(0, int(repro_cfg.get("dataset_sample_size") or 0)),
        },
        "campaign": {
            "id": profile_name,
            "claim_scope": claim_scope,
            "mode": str(campaign_cfg.get("mode") or "development"),
            "enforcement": str(campaign_cfg.get("enforcement") or "warn"),
            "frozen_before_final_campaign": bool(campaign_cfg.get("frozen_before_final_campaign")),
            "dataset_registry": str(default_dataset_registry_path()),
            "auto_bind_dataset_registry": bool(campaign_cfg.get("auto_bind_dataset_registry")),
            "dataset_manifests": {
                "classification": {"calibration": "", "validation": ""},
                "detection": {"calibration": "", "validation": ""},
            },
            "pipeline_contract_manifest": "",
            "holdout_registry": "campaign_inputs/holdout_registry.json" if ranking_generalization else "",
            "ranking_model_bundle": "",
            "energy_calibration_manifest": "",
            "campaign_freeze_artifact": "",
            "require_protocol_freeze": bool(campaign_cfg.get("require_protocol_freeze")),
            "require_fitted_stage_time": bool(campaign_cfg.get("require_fitted_stage_time")),
            "require_native_handover_model": bool(campaign_cfg.get("require_native_handover_model")),
            "require_campaign_freeze": bool(campaign_cfg.get("require_campaign_freeze")),
            "require_prediction_freeze_approval": bool(campaign_cfg.get("require_prediction_freeze_approval")),
            "require_cryptographic_prediction_signature": bool(campaign_cfg.get("require_cryptographic_prediction_signature")),
            "prediction_freeze_public_key": "",
            "prediction_freeze_enabled": bool(holdout_cfg.get("prediction_freeze_enabled", mode_id in {"standard", "final"})),
        },
        "validation": {
            "split_fidelity_reference_mode": "auto",
            "classification_metrics": ["top1", "top5", "logit_cosine_similarity"],
            "detection_metrics": ["coco_ap_50_95", "ap50", "ap75", "semantic_proxy_match_ratio"],
            "backend_drift_reference": "cpu_full",
            "report_blocked_and_infeasible": True,
            "mode": "strict" if final else "summary_only",
            "require_explicit": bool(final),
            "require_task_metrics": bool(final),
        },
        "validation_execution": {
            "mode": "final" if final else "screening",
            "cadence": str(quality_cfg.get("cadence") or "once_per_artifact"),
            "cache_task_quality": bool(quality_cfg.get("cache_task_quality", True)),
            "repeat_task_quality_for_timing_repeats": False,
            "max_items": {
                "classification": max(0, int(val_items.get("classification") or 0)),
                "detection": max(0, int(val_items.get("detection") or 0)),
            },
        },
        "quality_gate": {
            "schema": "onnx-splitpoint/task-quality-policy",
            "schema_version": 3,
            "name": str(quality_cfg.get("profile_id") or f"task_quality_{mode_id}"),
            "profile_id": str(quality_cfg.get("profile_id") or f"task_quality_{mode_id}"),
            "frozen_before_final_campaign": bool(final),
            "dataset_tier": str(quality_cfg.get("dataset_tier") or ("final" if final else "screening")),
            "canonical_reference": str(quality_cfg.get("canonical_reference") or "canonical_full_onnx"),
            "enforcement": {
                "technical_quality_error": (
                    "partial_continue_diagnostic" if mode_id == "smoke" else "hard_fail"
                ),
                "metric_threshold_miss": (
                    "warning_only" if mode_id == "smoke" else "enforced_quality_decision"
                ),
                "native_energy_after_technical_error": (
                    "collect_raw_quality_unqualified"
                ),
                "claim_eligibility": (
                    "always_false" if mode_id == "smoke" else "complete_valid_evidence_only"
                ),
            },
            "diagnostic_only": bool(mode_id == "smoke"),
            "claim_eligible": False if mode_id == "smoke" else None,
            "classification": {
                "primary_metric": "top1_accuracy",
                "non_inferiority_margin": max(0.0, float(quality_cfg.get("classification_margin_pp") or 1.0)) / 100.0,
                "guardrails": {
                    "top5_accuracy_margin": max(0.0, float(quality_cfg.get("classification_top5_margin_pp") or 1.0)) / 100.0,
                },
            },
            "detection": {
                "primary_metric": "coco_ap_50_95",
                "non_inferiority_margin": max(0.0, float(quality_cfg.get("detection_margin_ap") or 1.0)) / 100.0,
                "guardrails": {
                    "ap50_margin": max(0.0, float(quality_cfg.get("detection_ap50_margin") or 1.0)) / 100.0,
                    "ap75_margin": max(0.0, float(quality_cfg.get("detection_ap75_margin") or 1.0)) / 100.0,
                },
            },
            "statistics": {
                "method": "paired_bootstrap",
                "confidence_level": min(0.999999, max(0.5, float(quality_cfg.get("confidence_level") or 0.95))),
                "bootstrap_repetitions": max(0, int(quality_cfg.get("bootstrap_repetitions") or 0)),
                "seed": int(quality_cfg.get("bootstrap_seed") or 20260710),
                "decision": "lower_one_sided_bound",
                "execution_location": str(
                    quality_cfg.get("execution_location") or "central_management"
                ),
                "workers": max(
                    1,
                    int(quality_cfg.get("workers") or 4),
                ),
            },
            "native_contract": {
                "detection_self_reference_min_match_ratio": 0.80,
                "detection_self_reference": {
                    "policy_id": "class_aware_iou50_postnms_v2",
                    "min_reference_match_ratio": 0.80,
                    "min_mean_matched_iou": 0.85,
                    "iou_threshold": 0.50,
                    "confidence_threshold": 0.25,
                    "denominator": "reference_detections",
                    "class_aware": True,
                },
                "numerical_similarity_required_for_claim": True,
                "self_reference_counts_as_task_accuracy": False,
            },
            "screening_eligible_for_ranking": False,
            "legacy_point_estimate_eligible_for_ranking": False,
        },
        "ranking_validation": {
            "enabled": bool(ranking_cfg.get("enabled")),
            "holdout_unit": "model_direction_runner",
            "candidate_universe": str(holdout_cfg.get("candidate_universe") or "deterministic_audit"),
            "audit_size": max(0, int(holdout_cfg.get("audit_size") or 0)),
            "minimum_valid_audit_candidates": max(1, int(holdout_cfg.get("minimum_valid_candidates") or 10)),
            "audit_seed": int(holdout_cfg.get("seed") or 20260710),
            "require_frozen_predictions": bool(holdout_cfg.get("require_frozen_predictions")),
            "require_complete_candidate_universe": bool(holdout_cfg.get("require_complete_candidate_universe")),
            "k_values": list(ranking_cfg.get("k_values") or [1, 3, 5]),
            "elite_q_values": list(ranking_cfg.get("elite_q_values") or [1, 3]),
            "primary_k": int(ranking_cfg.get("primary_k") or 5),
            "minimum_candidates_for_correlation": int(ranking_cfg.get("minimum_candidates_for_correlation") or 3),
            "near_optimal_relative_epsilon": float(ranking_cfg.get("near_optimal_relative_epsilon") or 0.01),
            "bootstrap_repetitions": max(0, int(ranking_cfg.get("bootstrap_repetitions") or 0)),
            "bootstrap_seed": int(ranking_cfg.get("bootstrap_seed") or 20260710),
            "targets": ["pipeline_cycle_ms"],
            "methods": list(ranking_cfg.get("methods") or []),
            "weighted_score": {"w_comm": 1.0, "w_imb": 3.0, "w_tensors": 0.2, "log_comm": True},
            "cycle_time_no_handover": {"backend_throughput_gops": {}, "stage_time_models": {}},
            "cycle_time_with_handover": {"backend_throughput_gops": {}, "stage_time_models": {}, "handover_models": {"generic": {}, "native_fifo": {}}},
        },
        "official_coco_evaluation": {
            "enabled": bool(quality_cfg.get("official_coco_enabled")),
            "required_for_final": bool(quality_cfg.get("official_coco_required")),
            "annotations": "",
            "remote_annotations": "",
            "iou_type": "bbox",
            "archive_predictions": True,
            "archive_eval_tensors": bool(quality_cfg.get("archive_coco_eval_tensors")),
            "max_detections": [1, 10, 100],
            "require_pycocotools": bool(quality_cfg.get("official_coco_required")),
        },
        "reporting": {
            "include_decision_summary": bool(reporting_cfg.get("include_decision_summary", True)),
            "include_backend_drift_block": bool(reporting_cfg.get("include_backend_drift_block", True)),
            "include_task_specific_quality_block": bool(reporting_cfg.get("include_task_specific_quality_block", True)),
            "aggregate_by_model": True,
            "aggregate_by_run_profile": True,
            "canonical_scientific_report": bool(reporting_cfg.get("canonical_scientific_report", True)),
            "cleanup_legacy_reports": bool(reporting_cfg.get("cleanup_legacy_reports", True)),
            "generate_thesis_tex": bool(reporting_cfg.get("generate_thesis_tex")),
            "generate_thesis_figures": bool(reporting_cfg.get("generate_thesis_figures")),
            "include_campaign_readiness": bool(reporting_cfg.get("include_campaign_readiness", True)),
        },
        "workflow": {
            "execution_mode": str(runtime_cfg.get("execution_mode") or "generate_and_run"),
            "skip_runtime_benchmarks": bool(runtime_cfg.get("skip_runtime_benchmarks")),
            # Fast/relaxed modes cache unchanged-file hashes; they do not omit
            # model identity.  Only an explicit legacy CLI override may opt
            # out, otherwise candidate freezes and Scientific Reports would
            # not be bound to the model bytes used by the run.
            "no_model_hash": False,
            "include_reserve": False,
            "parallel_remote_setups": bool(parallel_cfg.get("remote_setups", True)),
            "max_parallel_setups": max(1, int(parallel_cfg.get("max_setups") or 3)),
            "max_parallel_uploads": max(0, int(parallel_cfg.get("max_uploads") or 1)),
            "powercalc_workers": max(0, int(parallel_cfg.get("powercalc_workers") or 1)),
            # v2.79.20: every materialised profile records the diagnostic
            # cache-preflight default.  Strict warm-cache acceptance remains
            # an explicit profile choice and is never silently enabled by a
            # Smoke/Standard/Final mode change.
            "artifact_cache_preflight": {
                "enabled": True,
                "default_expectation": "unspecified",
                "block_on_unexpected_cold_builds": False,
            },
        },
        "benchmark_execution": {
            "provider": str(benchmark_cfg.get("provider") or "auto"),
            "warmup": max(0, int(benchmark_cfg.get("warmup") or 0)),
            "runs": max(1, int(benchmark_cfg.get("runs") or 1)),
            "timeout_s": max(0, int(benchmark_cfg.get("timeout_s") or 0)),
            "backend": "auto",
        },
        "model_preparation": {
            "mode": str(build_cfg.get("model_preparation") or "current"),
            "note": f"Resolved from run mode {mode_id}.",
        },
        "hailo_build": {
            "mode": str(hailo_cfg.get("mode") or "reuse_and_build_missing"),
            "hw_arch": "hailo8",
            "targets": ["hailo8"],
            "timeout_s": hailo_timeout_s,
            "full_baseline_cold_build_policy": str(hailo_cfg.get("full_baseline_cold_build_policy") or "build_missing"),
            # Presence, rather than truthiness, is authoritative: explicit
            # zero is the long-run unlimited contract and must not fall back
            # to the normal finite Part-1 timeout.
            "cold_build_timeout_s": hailo_cold_timeout_s,
            "deferred_full_baseline_required": bool(hailo_cfg.get("deferred_full_baseline_required", mode_id != "smoke")),
            "preserve_attempt_provenance": bool(hailo_cfg.get("preserve_attempt_provenance", True)),
            "immutable_attempt_receipts": bool(hailo_cfg.get("immutable_attempt_receipts", True)),
            "terminal_attempt_selection": str(
                hailo_cfg.get("terminal_attempt_selection")
                or "last_attempt_even_on_failure"
            ),
            "hard_timeout_disable_tokens": canonical_hailo_disable_tokens(),
            "build_full": bool(hailo_cfg.get("build_full", True)),
            "build_part1": bool(hailo_cfg.get("build_part1", True)),
            "build_part2": bool(hailo_cfg.get("build_part2", True)),
            "preset": str(hailo_cfg.get("preset") or mode_id),
            "optimization_level": max(0, int(hailo_cfg.get("optimization_level") or 0)),
            "calib_dir": "",
            "calib_count": max(calib_items.values()),
            "calib_batch_size": max(1, int(hailo_cfg.get("calibration_batch_size") or 1)),
            "calibration_storage": str(hailo_cfg.get("calibration_storage") or ("memory" if mode_id == "smoke" else "memmap")),
            "calibration_memory_cap_mb": max(32, int(hailo_cfg.get("calibration_memory_cap_mb") or 256)),
            "cache_enabled": bool(hailo_cfg.get("cache_enabled", True)),
            "cache_root": str(hailo_cfg.get("cache_root") or "~/.cache/onnx_splitpoint/hailo_hef"),
            "cache_integrity": str(hailo_cfg.get("cache_integrity") or ("strict" if final else "relaxed")),
            "force_build": bool(hailo_cfg.get("force_build")),
            "keep_artifacts": bool(hailo_cfg.get("keep_artifacts")),
            "source": "run_mode_registry",
        },
        "deepx_build": deepx_build_block,
        "artifact_store": {
            "enabled": bool(artifact_store_cfg.get("enabled", True)),
            "root": str(artifact_store_cfg.get("root") or "~/.onnx_splitpoint_tool/artifact_store"),
            "verify_on_reuse": str(artifact_store_cfg.get("verify_on_reuse") or ("strict" if final else "metadata")),
            "pin_for_campaign": bool(artifact_store_cfg.get("pin_for_campaign", final)),
            "register_hailo": bool(artifact_store_cfg.get("register_hailo", True)),
            "register_deepx": bool(artifact_store_cfg.get("register_deepx", True)),
            "register_tensorrt": bool(artifact_store_cfg.get("register_tensorrt", False)),
        },
        "build_scheduler": {
            "enabled": bool(scheduler_cfg.get("enabled", True)),
            "max_workers": max(1, int(scheduler_cfg.get("max_workers") or (2 if final else 3))),
            "cpu_tokens": max(0, int(scheduler_cfg.get("cpu_tokens") or 0)),
            "ram_mb": max(0, int(scheduler_cfg.get("ram_mb") or 0)),
            "ram_reserve_mb": max(
                0, int(scheduler_cfg.get("ram_reserve_mb") or 2048)
            ),
            "family_limits": dict(scheduler_cfg.get("family_limits") or {"hailo8": 1, "hailo10": 1, "deepx": 1}),
            "weights": dict(scheduler_cfg.get("weights") or {}),
            "prefetch_deepx_full": bool(scheduler_cfg.get("prefetch_deepx_full", True)),
            "pipeline_next_model": bool(scheduler_cfg.get("pipeline_next_model", False)),
        },
        "hardware_smoke": {
            "mode": str(build_cfg.get("hardware_smoke") or "summary_only"),
            "require_remote_for_hailo": bool(final),
            "note": f"Resolved from run mode {mode_id}.",
        },
        "remote_cache": {
            "upload_once_per_model_setup": bool((runtime_cfg.get("remote_cache") or {}).get("upload_once_per_model_setup", True)),
            "reuse_tensorrt_engines": bool((runtime_cfg.get("remote_cache") or {}).get("reuse_tensorrt_engines", True)),
            "stable_cache_root": str((runtime_cfg.get("remote_cache") or {}).get("stable_cache_root") or "~/.cache/onnx_splitpoint"),
            "suite_cache_root": str((runtime_cfg.get("remote_cache") or {}).get("suite_cache_root") or "~/.cache/onnx_splitpoint/suites"),
            "engine_cache_root": str((runtime_cfg.get("remote_cache") or {}).get("engine_cache_root") or "~/.cache/onnx_splitpoint/tensorrt"),
            "copy_strategy": str((runtime_cfg.get("remote_cache") or {}).get("copy_strategy") or "auto"),
        },
        "native_producers": {
            "enabled": False,  # high-level override is applied afterwards
            "backends": list(native_cfg.get("backends") or ["hailo8", "hailo10h", "deepx"]),
            "case_policy": str(native_cfg.get("case_policy") or "all_accepted"),
            "precision": str(native_cfg.get("precision") or "uint8_cast_fp16"),
            "frames": max(1, int(native_cfg.get("frames") or 100)),
            "warmup": max(0, int(native_cfg.get("warmup") or 10)),
            "repetitions": max(1, int(native_cfg.get("repetitions") or (1 if mode_id == "smoke" else (5 if final else 3)))),
            "queue_depth": max(1, int(native_cfg.get("queue_depth") or 2)),
            "inflight": max(1, int(native_cfg.get("inflight") or 4)),
            "hailo_format": "uint8",
            "remote_root": "/home/nx/native_fifo_evalsets",
            "remote_tool_dir": "/home/nx/ONNX-Splitpoint-Tool",
            "build_missing_engines": bool(native_cfg.get("build_missing_engines", True)),
            "copy_benchmarksets": bool(native_cfg.get("copy_benchmarksets", True)),
            "strict_supported_only": bool(native_cfg.get("strict_supported_only", True)),
            "full_baselines": {"enabled": bool(native_cfg.get("full_baselines"))},
            "validation": {"enabled": bool(native_cfg.get("validation", True)), "mode": "dump_and_visual", "topk": 5},
            "dump_outputs": bool(native_cfg.get("dump_outputs", True)),
            # No remote host/IP block: hardware_setups.yaml is authoritative.
            "energy": {
                "enabled": False,
                "mode": str(energy_cfg.get("native_mode") or "plan"),
                "duration_s": max(0, int(energy_cfg.get("native_duration_s") or 0)),
                "physical_scope": str(
                    energy_cfg.get("physical_scope") or "FS"
                ),
                "window_label": str(
                    energy_cfg.get("window_label") or "command"
                ),
            },
        },
        "energy": {
            # v60o: Generic Runner energy is deliberately disabled for EvalRuns.
            # The high-level energy switch controls semantically gated Native energy only.
            "enabled": False,
            "generic_enabled": False,
            "measurement_path": "native_only",
            "final_all_split_energy": False,
            "final_energy_skip_cpu_ort": True if final else False,
            "scope": str(energy_cfg.get("scope") or "row_variant"),
            "physical_scope": str(
                energy_cfg.get("physical_scope") or "FS"
            ),
            "window_label": str(
                energy_cfg.get("window_label") or "command"
            ),
            "repeat_override": max(0, int(energy_cfg.get("repeats") or 0)),
            "phases": list(energy_cfg.get("phases") or ["streaming"]),
            "target_policy": str(energy_cfg.get("target_policy") or "best_valid_only"),
            "skip_backends": list(energy_cfg.get("skip_backends") or ["ort_cpu"]),
            "include_run_ids": [],
            "exclude_run_ids": [],
            "heartbeat_s": max(10, int(energy_cfg.get("heartbeat_s") or 60)),
            "max_targets_per_run_id": max(0, int(energy_cfg.get("max_targets_per_run_id") or 0)),
            "max_work_units_per_window": max(0, int(energy_cfg.get("max_work_units_per_window") or 0)),
            "max_window_duration_s": max(0, int(energy_cfg.get("max_window_duration_s") or 0)),
            "timeout_s_per_window": max(0, int(energy_cfg.get("timeout_s_per_window") or 0)),
            "sizing_probe_max_work_units": max(1, int(energy_cfg.get("sizing_probe_max_work_units") or 256)),
            "include_raw_parquet_in_debug_pack": bool(energy_cfg.get("include_raw_parquet_in_debug_pack")),
            "strict": bool(energy_cfg.get("strict")),
        },
    }
    return blocks


def default_dataset_registry_path() -> Path:
    override = str(os.environ.get("ONNX_SPLITPOINT_DATASET_REGISTRY", "") or "").strip()
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override))).resolve()
    return Path.home() / ".onnx_splitpoint_tool" / "final_datasets" / "dataset_registry.json"


def apply_run_mode(
    profile: Mapping[str, Any],
    *,
    mode_id: str | None = None,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    follow_tool_config: bool | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return a fully materialised profile and a concise resolution audit."""
    original = copy.deepcopy(dict(profile or {}))
    preset = original.get("execution_preset") if isinstance(original.get("execution_preset"), Mapping) else {}
    mid = normalize_mode_id(mode_id or preset.get("id") or infer_run_mode(original))
    configured_path = config_path or preset.get("config_path")
    cfg_path = Path(configured_path).expanduser().resolve() if configured_path else default_run_modes_path().resolve()
    cfg = validate_run_modes_config(config or load_run_modes_config(cfg_path))
    mode = copy.deepcopy(dict(cfg["modes"][mid]))
    follow = bool(preset.get("follow_tool_config", True)) if follow_tool_config is None else bool(follow_tool_config)
    # If a profile explicitly requests a frozen snapshot, use that exact mode
    # definition rather than the current Tool Config registry.
    existing_snapshot = preset.get("snapshot") if isinstance(preset.get("snapshot"), Mapping) else None
    if not follow and existing_snapshot:
        mode = copy.deepcopy(dict(existing_snapshot))

    name = str(original.get("name") or "custom_splitpoint_eval_v1").strip() or "custom_splitpoint_eval_v1"
    blocks = _materialized_blocks(mid, mode, profile_name=name)

    # Preserve the high-level parts that remain editable in the simplified GUI.
    preserved = {
        key: copy.deepcopy(original[key])
        for key in (
            "name",
            "purpose",
            "implementation_note",
            "models_root_hint",
            "selection_policy",
            "model_suite",
            "run_profiles",
            "measurement_campaign",
            "execution_guard",
            # Artifact-cache acceptance is orthogonal to run effort.  A
            # profile-level strict/expected-cold declaration must survive
            # Smoke/Standard/Final materialisation unchanged.
            "artifact_cache_preflight",
            # A Full-only Quality canary is an explicit execution contract,
            # not a run-mode convenience setting.  Dropping it here silently
            # expands the projected plan back to Generic/Composed rows when a
            # normal saved Standard profile is loaded a second time.
            "quality_canary",
        )
        if key in original
    }
    resolved: Dict[str, Any] = _deep_merge(blocks, preserved)

    # The resolver also accepts the policy below ``workflow`` for compact
    # hand-written profiles.  Preserve that exact nested policy without
    # preserving unrelated workflow execution knobs owned by the run mode.
    old_workflow = (
        original.get("workflow")
        if isinstance(original.get("workflow"), Mapping)
        else {}
    )
    if isinstance(old_workflow.get("artifact_cache_preflight"), Mapping):
        resolved.setdefault("workflow", {})["artifact_cache_preflight"] = (
            copy.deepcopy(dict(old_workflow["artifact_cache_preflight"]))
        )

    # The run-mode registry owns Hailo build effort (preset, calibration,
    # timeout, cache policy, and optimisation), but it does not describe the
    # physical compiler target selected by an evaluation profile.  Preserve
    # only that explicit physical axis.  In particular, ``targets: []`` is an
    # authoritative no-Hailo selection for a DeepX-only view and must not be
    # confused with a missing key that receives the legacy Hailo-8 default.
    old_hailo_build = (
        original.get("hailo_build")
        if isinstance(original.get("hailo_build"), Mapping)
        else {}
    )
    if "hw_arch" in old_hailo_build:
        resolved.setdefault("hailo_build", {})["hw_arch"] = copy.deepcopy(
            old_hailo_build["hw_arch"]
        )
    if "targets" in old_hailo_build:
        resolved.setdefault("hailo_build", {})["targets"] = copy.deepcopy(
            old_hailo_build["targets"]
        )
    elif "hw_arch" in old_hailo_build:
        # Match the legacy profile-options default for a genuinely absent
        # targets key, but derive it from the preserved architecture rather
        # than the run-mode's generic Hailo-8 materialisation.
        resolved.setdefault("hailo_build", {})["targets"] = [
            copy.deepcopy(old_hailo_build["hw_arch"])
        ]

    # DeepX classification preprocessing is an explicit profile-owned
    # scientific axis.  The central run mode supplies the default for profiles
    # that do not name it, but it must not erase an explicit editor/YAML value.
    # Preserve only this field rather than the complete legacy deepx_build
    # block so Tool Config remains authoritative for compiler effort/settings.
    old_deepx_build = (
        original.get("deepx_build")
        if isinstance(original.get("deepx_build"), Mapping)
        else {}
    )
    if "classification_preprocessing" in old_deepx_build:
        classification_preprocessing = str(
            old_deepx_build.get("classification_preprocessing") or ""
        ).strip().lower()
        if classification_preprocessing not in {
            "current_scale_only", "imagenet_mean_std",
        }:
            raise ValueError(
                "deepx_build.classification_preprocessing must be "
                "current_scale_only or imagenet_mean_std"
            )
        resolved.setdefault("deepx_build", {})[
            "classification_preprocessing"
        ] = classification_preprocessing

    # Native execution is a profile-level contract.  A run-mode change may
    # change build/data/quality effort, but it must never replace an already
    # configured Native runner or Native-energy contract with the mode's
    # convenience defaults.  New/legacy profiles without either block still
    # receive the materialised defaults above; once present, the blocks are
    # authoritative and survive every subsequent mode projection byte-for-
    # value (apart from the canonical Full-binding projection below).
    old_native = (
        copy.deepcopy(dict(original.get("native_producers") or {}))
        if isinstance(original.get("native_producers"), Mapping)
        else None
    )
    old_energy = (
        copy.deepcopy(dict(original.get("energy") or {}))
        if isinstance(original.get("energy"), Mapping)
        else None
    )
    if old_native is not None:
        resolved["native_producers"] = old_native
    if old_energy is not None:
        resolved["energy"] = old_energy

    # Keep task-specific manifests/annotations that have already been resolved or
    # explicitly entered.  Registry binding fills missing values later.
    old_campaign = original.get("campaign") if isinstance(original.get("campaign"), Mapping) else {}
    old_manifests = old_campaign.get("dataset_manifests") if isinstance(old_campaign.get("dataset_manifests"), Mapping) else None
    if old_manifests:
        resolved["campaign"]["dataset_manifests"] = copy.deepcopy(old_manifests)
    for key in (
        "claim_scope",
        "dataset_registry",
        "pipeline_contract_manifest",
        "holdout_registry",
        "ranking_model_bundle",
        "energy_calibration_manifest",
        "campaign_freeze_artifact",
        "prediction_freeze_public_key",
    ):
        value = old_campaign.get(key)
        if value not in (None, ""):
            resolved["campaign"][key] = copy.deepcopy(value)
    old_claim_scope = str(old_campaign.get("claim_scope") or "").strip()
    if old_claim_scope == RANKING_GENERALIZATION_CLAIM_SCOPE:
        for key in (
            "require_protocol_freeze",
            "protocol_freeze",
            "require_fitted_stage_time",
            "require_native_handover_model",
            "require_campaign_freeze",
            "require_prediction_freeze_approval",
            "require_cryptographic_prediction_signature",
            "prediction_freeze_enabled",
        ):
            if key in old_campaign:
                resolved["campaign"][key] = copy.deepcopy(old_campaign[key])
        if isinstance(original.get("ranking_validation"), Mapping):
            resolved["ranking_validation"] = copy.deepcopy(dict(original["ranking_validation"]))
    old_coco = original.get("official_coco_evaluation") if isinstance(original.get("official_coco_evaluation"), Mapping) else {}
    for key in ("annotations", "remote_annotations"):
        if old_coco.get(key):
            resolved["official_coco_evaluation"][key] = old_coco[key]

    overrides = dict(preset.get("overrides") or {}) if isinstance(preset.get("overrides"), Mapping) else {}
    # Explicit profile blocks outrank the historical execution-preset mirror.
    # This prevents a stale snapshot/default from flipping either switch when
    # the user changes only Smoke/Standard/Final.
    if old_native is not None and "enabled" in old_native:
        overrides["native_enabled"] = bool(old_native.get("enabled"))
    elif "native_enabled" not in overrides:
        overrides["native_enabled"] = bool((mode.get("defaults") or {}).get("native_enabled"))
    has_explicit_native_energy = bool(
        old_native is not None
        and isinstance(old_native.get("energy"), Mapping)
        and "enabled" in old_native.get("energy", {})
    )
    has_explicit_native_energy_mode = bool(
        old_native is not None
        and isinstance(old_native.get("energy"), Mapping)
        and str(old_native.get("energy", {}).get("mode") or "").strip()
    )
    has_explicit_energy_request = bool(
        old_energy is not None and "requested_native_energy" in old_energy
    )
    if has_explicit_native_energy or has_explicit_energy_request:
        overrides["energy_enabled"] = native_energy_requested(original)
    elif "energy_enabled" not in overrides:
        overrides["energy_enabled"] = bool((mode.get("defaults") or {}).get("energy_enabled"))

    native_enabled = bool(overrides.get("native_enabled"))

    # v61a: preserve the producer-specific Native Full matrix all the way
    # through materialisation. TensorRT Full is repeated on each active setup.
    _native_plan_profile = dict(resolved)
    _native_plan_profile.setdefault("native_producers", {})["enabled"] = native_enabled
    native_full_plan = resolve_native_full_plan(_native_plan_profile)
    native_split_plan = resolve_native_split_plan(_native_plan_profile)
    selected_native_full_backends = list(native_full_plan.selected_full_backends)
    existing_full = (
        dict(resolved["native_producers"].get("full_baselines") or {})
        if isinstance(resolved["native_producers"].get("full_baselines"), Mapping)
        else {}
    )
    existing_full.update({
        "enabled": bool(native_full_plan.enabled),
        "backends": sorted({b for rows in native_full_plan.backends_by_producer.values() for b in rows}),
        "backends_by_producer": {k: list(v) for k, v in native_full_plan.backends_by_producer.items()},
        "active_producers": list(native_full_plan.active_producers),
        "source": "selected_full_run_profiles",
        "auto_enabled": True,
    })
    resolved["native_producers"]["full_baselines"] = existing_full
    # ``backends`` remains the configured adapter/capability inventory.  The
    # selected split matrix is a separate, authoritative projection of the
    # visible logical run profiles; an empty list is meaningful for Full-only.
    resolved["native_producers"]["split_backends"] = list(
        native_split_plan.selected_split_backends
    )
    resolved["native_producers"]["split_selection_source"] = (
        native_split_plan.source
    )

    # The simplified Energy switch continues to mean Native system energy.
    # A separately and explicitly requested Generic path must, however, survive
    # materialisation instead of being silently rewritten to ``native_only``.
    native_energy_enabled = bool(overrides.get("energy_enabled"))
    old_energy_path = str(
        (old_energy or {}).get("measurement_path") or ""
    ).strip().lower()
    explicit_generic_energy = bool(
        old_energy is not None
        and (
            str(old_energy.get("generic_enabled") or "")
            .strip().lower() in {"1", "true", "yes", "on"}
            or old_energy_path in {"generic", "native_and_generic"}
        )
    )
    generic_energy_enabled = bool(explicit_generic_energy)
    if generic_energy_enabled and native_enabled and native_energy_enabled:
        energy_measurement_path = "native_and_generic"
    elif generic_energy_enabled:
        energy_measurement_path = "generic"
    elif native_enabled and native_energy_enabled:
        energy_measurement_path = "native_only"
    else:
        energy_measurement_path = "disabled"
    resolved["native_producers"]["enabled"] = native_enabled
    resolved["energy"]["enabled"] = generic_energy_enabled
    resolved["energy"]["generic_enabled"] = generic_energy_enabled
    resolved["energy"]["measurement_path"] = energy_measurement_path
    resolved["energy"]["requested_native_energy"] = bool(native_energy_enabled)
    if generic_energy_enabled:
        resolved["energy"]["requested"] = True
    native_energy_cfg = resolved["native_producers"].get("energy")
    if not isinstance(native_energy_cfg, Mapping):
        native_energy_cfg = {}
    native_energy_cfg = dict(native_energy_cfg)
    native_energy_cfg["enabled"] = bool(native_enabled and native_energy_enabled)
    # Preserve an explicitly configured plan/measure mode.  The simplified GUI
    # already writes its intended value; only profiles without one need the
    # compatibility default.
    if not has_explicit_native_energy_mode:
        native_energy_cfg["mode"] = "measure" if (native_enabled and native_energy_enabled) else "plan"
    resolved["native_producers"]["energy"] = native_energy_cfg
    if not native_enabled:
        resolved["native_producers"]["energy"]["enabled"] = False

    # Targets are implied by the selected run profiles.  Legacy remote host
    # blocks are not persisted here, but the validated ``hardware`` block is a
    # profile-owned binding: in particular a frozen setups registry must not be
    # silently replaced with the mutable per-user default during run-mode
    # materialisation.
    resolved.pop("remote_execution", None)
    resolved.pop("hardware_setups", None)
    resolved.pop("hardware_groups", None)
    resolved.pop("hardware_targets", None)
    resolved.pop("build_environments", None)
    old_hardware = (
        copy.deepcopy(dict(original.get("hardware") or {}))
        if isinstance(original.get("hardware"), Mapping)
        else {}
    )
    old_hardware.setdefault("selected_setups", [])
    old_hardware.setdefault("selected_groups", [])
    resolved["hardware"] = old_hardware

    # Final evaluated-matrix profiles must be self-describing even when they
    # originated in the simplified editor.  Fill only absent metadata; explicit
    # scientific roles or candidate contracts are never overwritten.
    resolved_claim_scope = str((resolved.get("campaign") or {}).get("claim_scope") or "")
    resolved_final = str((resolved.get("campaign") or {}).get("mode") or "").lower() == "final"
    if resolved_final and resolved_claim_scope == EVALUATED_MATRIX_CLAIM_SCOPE:
        suite = resolved.get("model_suite") if isinstance(resolved.get("model_suite"), MutableMapping) else {}
        for tier in ("primary", "reserve"):
            entries = suite.get(tier) if isinstance(suite.get(tier), list) else []
            for entry in entries:
                if not isinstance(entry, MutableMapping):
                    continue
                model_id = str(entry.get("id") or "model").strip().lower().replace("-", "_")
                family = str(entry.get("family") or "").strip()
                if not family:
                    if model_id.startswith("yolo26"):
                        family = "yolo26"
                    elif model_id.startswith("yolov7") or model_id.startswith("yolo7"):
                        family = "yolo7"
                    elif model_id.startswith("resnet"):
                        family = "resnet"
                    else:
                        family = model_id.split("_", 1)[0] or "model"
                    entry["family"] = family
                entry.setdefault("family_id", family)
                entry.setdefault("evaluation_role", "development")
                entry.setdefault("generalization_scope", "development")
                entry.setdefault("validation_tier", "final")
                entry.setdefault(
                    "candidate_universe",
                    {"mode": "declared_shortlist", "seed": 20260710},
                )

    # Per-task calibration counts remain available to the build binding while
    # the legacy single count uses their maximum for compatibility.
    resolved["execution_preset"] = {
        "id": mid,
        "label": str(mode.get("label") or mid.title()),
        "follow_tool_config": bool(follow),
        "config_path": str(cfg_path),
        "config_sha256": _json_hash(cfg),
        "snapshot_sha256": _json_hash(mode),
        "resolved_at": now_iso(),
        "overrides": {
            "native_enabled": native_enabled,
            "energy_enabled": native_energy_enabled,
        },
        "snapshot": mode,
        "effective": {
            "calibration_items": _task_calibration_items(mode),
            "validation_items": copy.deepcopy((mode.get("data") or {}).get("validation_items") or {}),
            "bootstrap_repetitions": int(((mode.get("quality") or {}).get("bootstrap_repetitions") or 0)),
            "integrity_level": str(((mode.get("reproducibility") or {}).get("level") or "relaxed")),
            "campaign_claim_scope": resolved_claim_scope,
            "generic_energy_enabled": generic_energy_enabled,
            "energy_measurement_path": energy_measurement_path,
            "native_full_baselines_enabled": bool(native_enabled and selected_native_full_backends),
            "native_full_backends": list(selected_native_full_backends),
            "native_full_backends_by_producer": {k: list(v) for k, v in native_full_plan.backends_by_producer.items()},
            "native_split_backends": list(native_split_plan.selected_split_backends),
            "native_split_enabled": bool(native_split_plan.enabled),
            "native_performance_repetitions": max(1, int(resolved["native_producers"].get("repetitions") or 1)),
            "upload_once_per_model_setup": bool(((mode.get("runtime") or {}).get("remote_cache") or {}).get("upload_once_per_model_setup", True)),
            "reuse_tensorrt_engines": bool(((mode.get("runtime") or {}).get("remote_cache") or {}).get("reuse_tensorrt_engines", True)),
            "reuse_hailo_artifacts": bool(((mode.get("build") or {}).get("hailo") or {}).get("cache_enabled", True)),
            "dataset_verification": "full" if str(((mode.get("reproducibility") or {}).get("level") or "relaxed")) == "strict" else ("manifest_only" if mid == "smoke" else "sampled"),
        },
    }
    resolved["implementation_note"] = (
        "Generated by the simplified Evaluation Profile editor. Detailed build, validation, reporting, "
        f"hold-out and reproducibility settings are resolved from Tool Config run mode '{mid}'."
    )

    # The cache-verification guard is orthogonal to Smoke/Standard/Final.  It
    # must be projected only after the normal Tool Config resolution so the
    # artifact identity (for example Standard opt/calibration axes) remains
    # unchanged while every build/force switch is frozen fail-closed.
    resolved = apply_cache_verify_only_policy(resolved)

    audit = {
        "mode_id": mid,
        "label": str(mode.get("label") or mid.title()),
        "follow_tool_config": follow,
        "config_path": str(cfg_path),
        "config_sha256": resolved["execution_preset"]["config_sha256"],
        "snapshot_sha256": resolved["execution_preset"]["snapshot_sha256"],
        "native_enabled": native_enabled,
        "energy_enabled": native_energy_enabled,
        "native_full_baselines_enabled": bool(native_enabled and selected_native_full_backends),
        "native_full_backends": list(selected_native_full_backends),
        "native_full_backends_by_producer": {k: list(v) for k, v in native_full_plan.backends_by_producer.items()},
        "final": str((resolved.get("campaign") or {}).get("mode") or "") == "final",
        "campaign_claim_scope": resolved_claim_scope,
        "artifact_policy": str(
            ((resolved.get("execution_guard") or {}) if isinstance(resolved.get("execution_guard"), Mapping) else {}).get("mode")
            or "normal"
        ),
    }
    return resolved, audit


def run_mode_profile_brief(profile: Mapping[str, Any]) -> str:
    preset = profile.get("execution_preset") if isinstance(profile.get("execution_preset"), Mapping) else {}
    mid = normalize_mode_id(preset.get("id") or infer_run_mode(profile))
    label = str(preset.get("label") or mid.title())
    overrides = preset.get("overrides") if isinstance(preset.get("overrides"), Mapping) else {}
    native_cfg = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    full_cfg = native_cfg.get("full_baselines") if isinstance(native_cfg.get("full_baselines"), Mapping) else {}
    full_backends = ",".join(str(x) for x in list(full_cfg.get("backends") or [])) or "none"
    energy_cfg = (
        profile.get("energy")
        if isinstance(profile.get("energy"), Mapping) else {}
    )
    generic_energy = bool(
        str(energy_cfg.get("generic_enabled") or "")
        .strip().lower() in {"1", "true", "yes", "on"}
        or str(energy_cfg.get("measurement_path") or "")
        .strip().lower() in {"generic", "native_and_generic"}
    )
    return f"{label} · native={'on' if overrides.get('native_enabled') else 'off'} · native full={full_backends if full_cfg.get('enabled') else 'off'} · native energy={'on' if overrides.get('energy_enabled') else 'off'} · generic energy={'on' if generic_energy else 'off'}"


# v60z resolved-profile normalisation wrappers
from onnx_splitpoint_tool.native_full_quality import normalise_evaluation_profile, resolve_native_full_plan as _v60z_normalise_profile
_v60z_original_resolve_run_mode = resolve_run_mode
def resolve_run_mode(*args, **kwargs):
    result = _v60z_original_resolve_run_mode(*args, **kwargs)
    if isinstance(result, dict):
        _v60z_normalise_profile(result)
    return result
resolve_run_mode._v60z_wrapped = True


def official_coco_policy_for_mode(mode):
    name=str(mode or "standard").strip().lower()
    if name in {"standard", "final"}: return {"enabled": True, "required_for_final": False, "archive_predictions": True, "archive_eval_tensors": False}
    return {"enabled": False, "required_for_final": False, "archive_predictions": False, "archive_eval_tensors": False}
