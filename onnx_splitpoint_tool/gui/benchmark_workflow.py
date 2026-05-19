"""Benchmark benchmark-set workflow controller extracted from gui_app.py.

This module keeps the heavy benchmark-suite generation orchestration out of the
legacy Tk root class so gui_app.py can stay focused on widget state and thin
entrypoints.
"""

from __future__ import annotations

import importlib
import logging
import os
import queue
import threading
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from .. import __version__ as TOOL_VERSION
from ..benchmark_case_utils import build_benchmark_case_rejection
from ..benchmark.generation_state import find_latest_resumable_set, read_json as read_generation_json
from ..benchmark.hailo_scoring import rerank_candidates_for_hailo
from ..benchmark.evaluation_profiles import load_export_metadata_for_model, resolve_evaluation_profile
from ..benchmark.model_preparation import find_latest_preparation_full_hailo_baseline, load_preparation_full_hailo_baseline, load_preparation_full_hailo_end_nodes, normalize_model_preparation_mode, preparation_result_is_selected_model
from ..benchmark.resume_integrity import reconcile_generation_state
from ..benchmark.schema import stamp_benchmark_set_payload, write_json_atomic as write_benchmark_json_atomic
from ..benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationOrchestrationConfig,
    BenchmarkGenerationOrchestrationService,
    BenchmarkGenerationService,
    normalize_full_hef_policy,
    normalize_hailo_full_model_preflight_policy,
)
from .controller import write_benchmark_suite_script
from .widgets.benchmark_completion_dialog import show_benchmark_completion_dialog
from ..hailo.backend_mode import normalize_hailo_backend
from ..resources_utils import copy_resource_tree
from ..workdir import ensure_workdir

__version__ = TOOL_VERSION
logger = logging.getLogger(__name__)


@dataclass
class ResolvedHailoBenchmarkHelpers:
    hailo_build_hef_fn: Optional[Any] = None
    hailo_parse_check_fn: Optional[Any] = None
    hailo_build_unavailable: Optional[str] = None
    hailo_part2_precheck_fn: Optional[Any] = None
    hailo_part2_precheck_error_fn: Optional[Any] = None
    hailo_part2_parser_precheck_fn: Optional[Any] = None
    hailo_part2_parser_precheck_error_fn: Optional[Any] = None
    hailo_part2_import_error: Optional[str] = None


def resolve_tool_core_version(*, importer=None) -> str:
    importer = importer or importlib.import_module
    try:
        mod = importer("onnx_splitpoint_tool.api")
        return str(getattr(mod, "__version__", "?"))
    except Exception:
        logger.debug("Failed to resolve tool core version from onnx_splitpoint_tool.api", exc_info=True)
        return "?"


def resolve_hailo_benchmark_helpers(*, need_build: bool, need_part2: bool, importer=None) -> ResolvedHailoBenchmarkHelpers:
    importer = importer or importlib.import_module
    resolved = ResolvedHailoBenchmarkHelpers()
    if not need_build and not need_part2:
        return resolved

    try:
        hailo_mod = importer("onnx_splitpoint_tool.hailo_backend")
    except Exception as exc:
        if need_build:
            resolved.hailo_build_unavailable = f"Hailo HEF build unavailable: {exc}"
        if need_part2:
            resolved.hailo_part2_import_error = f"{type(exc).__name__}: {exc}"
        return resolved

    if need_build:
        try:
            resolved.hailo_build_hef_fn = getattr(hailo_mod, "hailo_build_hef_auto")
        except Exception as exc:
            resolved.hailo_build_unavailable = f"Hailo HEF build unavailable: {exc}"
        try:
            resolved.hailo_parse_check_fn = getattr(hailo_mod, "hailo_parse_check_auto")
        except Exception:
            logger.debug("Failed to resolve hailo_parse_check_auto for benchmark workflow", exc_info=True)

    if need_part2:
        try:
            resolved.hailo_part2_precheck_error_fn = getattr(hailo_mod, "format_hailo_part2_activation_precheck_error")
            resolved.hailo_part2_parser_precheck_error_fn = getattr(hailo_mod, "format_hailo_part2_parser_blocker_error")
            resolved.hailo_part2_precheck_fn = getattr(hailo_mod, "hailo_part2_activation_precheck_from_manifest")
            resolved.hailo_part2_parser_precheck_fn = getattr(hailo_mod, "hailo_part2_parser_blocker_precheck_from_model")
        except Exception as exc:
            resolved.hailo_part2_import_error = f"{type(exc).__name__}: {exc}"

    return resolved




def _prepared_full_hailo_endpoint_override(model_path: str | Path) -> tuple[List[str], str]:
    """Read prepared full-Hailo endpoint override from model sidecar metadata."""
    try:
        info = load_preparation_full_hailo_end_nodes(model_path)
        nodes = [str(x).strip() for x in list(info.get('end_node_names') or []) if str(x).strip()]
        strategy = str(info.get('strategy') or '').strip()
        return nodes, strategy
    except Exception:
        logger.debug('Failed to resolve prepared full-Hailo endpoint override for %s', model_path, exc_info=True)
        return [], ''


def _prepared_full_hailo_baseline(model_path: str | Path) -> Dict[str, Any]:
    """Read a prepared full-Hailo HEF artifact from sidecar or latest screening summary."""
    try:
        sidecar = dict(load_preparation_full_hailo_baseline(model_path) or {})
        if bool(sidecar.get('ok')):
            return sidecar
        latest = dict(find_latest_preparation_full_hailo_baseline(model_path) or {})
        if bool(latest.get('ok')):
            return latest
        return sidecar if sidecar else latest
    except Exception:
        logger.debug('Failed to resolve prepared full-Hailo baseline for %s', model_path, exc_info=True)
        return {'selected': False, 'ok': False, 'reason': 'metadata_error'}

def _safe_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


class BenchmarkWorkflowController:
    def __init__(self, app: Any):
        self.app = app

    def generate_benchmark_set(
        self,
        *,
        resume_dir: Optional[str] = None,
        offer_latest_resume: bool = True,
        output_parent_override: Optional[str] = None,
        completion_callback=None,
        show_result_dialogs: bool = True,
    ) -> Optional[str]:
        """Generate or resume a benchmark suite folder for the current model + top-k picks.

        The suite contains one subfolder per split candidate with:
          - part1 / part2 ONNX models
          - split_manifest.json
          - run_split_onnxruntime.py runner script

        At the top level it also contains:
          - benchmark_set.json (list of cases + predicted metrics)
          - benchmark_suite.py (runs all cases and collects results/plots)
        """
        app = self.app
        model_path = app.gui_state.current_model_path or app.model_path
        if app.analysis is None or model_path is None:
            messagebox.showinfo("Nothing to benchmark", "Load a model and run an analysis first.")
            return
        if bool(getattr(app, "_benchmark_generation_active", False)):
            messagebox.showinfo(
                "Benchmark set already running",
                "A benchmark-set generation is already running in the background.\n\n"
                "You can start one remote benchmark in parallel, but only one benchmark-set generation at a time.",
            )
            return
        candidate_pool: List[int] = list(app._benchmark_candidate_pool())
        if not candidate_pool:
            messagebox.showinfo(
                "No candidates",
                "No split candidates available. Try increasing Top-K and re-run Analyse.",
            )
            return

        initial_out = app.default_output_dir or os.path.dirname(model_path)
        try:
            if app.default_output_dir:
                initial_out = str(ensure_workdir(Path(app.default_output_dir)).benchmark_sets)
        except Exception:
            pass

        base = os.path.splitext(os.path.basename(model_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        resume_generation = False
        resume_state_hint: Optional[Dict[str, Any]] = None
        resume_report = None

        if resume_dir is None:
            if output_parent_override is not None:
                out_parent = str(Path(output_parent_override))
                if not out_parent:
                    return None
            else:
                out_parent = filedialog.askdirectory(title="Select parent folder for benchmark set", initialdir=initial_out)
                if not out_parent:
                    return None
            out_dir = os.path.join(out_parent, f"{base}_benchmark_{ts}")
        else:
            out_dir = str(Path(resume_dir))
            out_parent = str(Path(out_dir).parent)
            resume_generation = True
            try:
                resume_state_hint = read_generation_json(Path(out_dir) / "generation_state.json", default={}) or {}
            except Exception:
                resume_state_hint = {}

        # Pull analysis objects once (used for strict-boundary filtering and TeX/plot export).
        a = app.analysis
        strict_boundary = bool(app.var_strict_boundary.get())
        generation_service = getattr(app, "_benchmark_generation_service", BenchmarkGenerationService())
        evaluation_profile_request = str(getattr(app, 'var_bench_evaluation_profile', tk.StringVar(value='')).get() or '').strip()
        evaluation_profile_resolution = None
        if evaluation_profile_request:
            try:
                evaluation_profile_resolution = resolve_evaluation_profile(evaluation_profile_request, model_path=model_path)
            except Exception:
                logger.exception('Failed to resolve benchmark evaluation profile %s', evaluation_profile_request)
                evaluation_profile_resolution = None

        try:
            prep_mode = normalize_model_preparation_mode(getattr(app, 'var_bench_model_preparation_mode', tk.StringVar(value='Use current ONNX')).get())
            export_meta = load_export_metadata_for_model(model_path)
            task_type = str(export_meta.get('task_type') or '').strip().lower()
            source = str(export_meta.get('source') or '').strip().lower()
            if prep_mode == 'screen_yolo_full_hailo' and source == 'ultralytics' and task_type == 'detection' and not preparation_result_is_selected_model(model_path):
                proceed = messagebox.askyesno(
                    'Model preparation recommended',
                    'This YOLO ONNX has not been screened for a full-Hailo-capable export variant yet.\n\n'
                    'For the manual workflow, use “Prepare current model…” first and then rerun Analyse on the selected prepared ONNX.\n\n'
                    'Do you want to continue anyway with the current ONNX?'
                )
                if not proceed:
                    return None
        except Exception:
            logger.debug('Failed to evaluate manual model-preparation guardrail', exc_info=True)

        # Read pruning params from the current GUI state (same source as Analyse).
        _params_for_split = app._read_params()
        prune_skip_block = bool(getattr(_params_for_split, "prune_skip_block", False))
        skip_min_span = int(getattr(_params_for_split, "skip_min_span", 0) or 0)
        if skip_min_span < 0:
            raise ValueError("Min skip span must be an integer ≥ 0.")
        skip_allow_last_n = int(getattr(_params_for_split, "skip_allow_last_n", 0) or 0)
        if skip_allow_last_n < 0:
            raise ValueError("Allow last N inside must be an integer ≥ 0.")

        # Convenience locals used during strict-boundary validation.
        model = a.get("model") if isinstance(a, dict) else None
        nodes = a.get("nodes") if isinstance(a, dict) else None
        order = a.get("order") if isinstance(a, dict) else None
        if model is None or nodes is None or order is None:
            messagebox.showerror(
                "Benchmark set failed",
                "Internal error: analysis data missing (model/nodes/order). Please re-run Analyse.",
            )
            return

        # How many candidates to export / inspect?
        #
        # ``ranked_candidates`` is the preferred shortlist (typically the currently
        # displayed Analyse picks). ``candidate_search_pool`` extends that shortlist
        # with broader ranked candidates so benchmark generation can backfill when
        # some splits are rejected later (for example due to Hailo Part2 activation
        # calibration depending on the original ``images`` input).

        ranked_candidates: List[int] = list(candidate_pool)

        def _strict_filter_boundaries(raw_bounds: List[int]) -> List[int]:
            try:
                return generation_service.strict_filter_boundaries(a, raw_bounds)
            except Exception:
                logger.debug("Strict boundary filtering via BenchmarkGenerationService failed", exc_info=True)
                return []

        # If the user enabled "Strict boundary" AFTER running Analyse (or the
        # analysis was done without strict-boundary metadata), re-check strictness
        # defensively for both the preferred shortlist and the broader backfill pool.
        if strict_boundary:
            ranked_candidates = _strict_filter_boundaries(ranked_candidates)
            if not ranked_candidates:
                messagebox.showinfo(
                    "No strict candidates",
                    "Strict boundary is enabled, but none of the available candidates satisfy the strict-boundary condition.\n\n"
                    "Tip: disable Strict boundary or re-run Analyse with Strict boundary unchecked.",
                )
                return

        candidate_search_pool: List[int] = list(app._benchmark_candidate_search_pool(ranked_candidates))
        if strict_boundary:
            candidate_search_pool = _strict_filter_boundaries(candidate_search_pool)

        if not candidate_search_pool:
            messagebox.showinfo(
                "No candidates",
                "No benchmark-export candidates are available after filtering.\n\n"
                "Tip: increase Top-K, disable Strict boundary, or re-run Analyse.",
            )
            return

        profile_overrides = dict(evaluation_profile_resolution.overrides or {}) if (evaluation_profile_resolution is not None and evaluation_profile_resolution.matched) else {}
        try:
            profile_shortlist = int(profile_overrides.get('preferred_shortlist') or 0)
        except Exception:
            profile_shortlist = 0
        if profile_shortlist > 0:
            ranked_candidates = list(ranked_candidates[:max(1, profile_shortlist)])
        pool_limit = profile_overrides.get('candidate_search_pool')
        try:
            pool_limit = int(pool_limit) if pool_limit is not None else 0
        except Exception:
            pool_limit = 0
        if int(pool_limit or 0) > 0:
            candidate_search_pool = list(candidate_search_pool[:max(1, int(pool_limit))])
        try:
            profile_requested_cases = int(profile_overrides.get('requested_cases') or 0)
        except Exception:
            profile_requested_cases = 0

        # How many cases to generate?
        # Prefer the Benchmark tab entry (var_bench_topk). Fall back to a dialog only
        # if it's missing/invalid. ``k`` is the target number of accepted cases;
        # backfilling can continue deeper into the ranked search pool if needed.
        default_k = min(profile_requested_cases or 20, len(candidate_search_pool))
        k = None
        try:
            k = _safe_int((getattr(app, "var_bench_topk", tk.StringVar(value=str(default_k))).get() or "").strip())
        except Exception:
            k = None
        if k is None:
            k = simpledialog.askinteger(
                "Benchmark set",
                f"How many splits to generate for the benchmark set? (target accepted cases, max {len(candidate_search_pool)}, preferred shortlist {len(ranked_candidates)})",
                initialvalue=default_k,
                minvalue=1,
                maxvalue=len(candidate_search_pool),
            )
        if k is None:
            return
        k = int(k)
        if profile_requested_cases > 0:
            k = min(max(1, int(profile_requested_cases)), len(candidate_search_pool))
        if k < 1 or k > len(candidate_search_pool):
            messagebox.showerror(
                "Benchmark set",
                f"Requested cases must be between 1 and {len(candidate_search_pool)} (search pool size).",
            )
            return

        # Offer to resume the newest incomplete benchmark set for this model.
        if not resume_generation and bool(offer_latest_resume):
            try:
                resumable_dir = find_latest_resumable_set(Path(out_parent), base)
            except Exception:
                resumable_dir = None
            if resumable_dir is not None:
                try:
                    resume_state_candidate = read_generation_json(resumable_dir / "generation_state.json", default={}) or {}
                except Exception:
                    resume_state_candidate = {}
                asked = messagebox.askyesno(
                    "Resume benchmark set?",
                    "Found an incomplete benchmark set for this model:\n\n"
                    f"{resumable_dir}\n\n"
                    "Resume it and reuse already generated artefacts?",
                )
                if asked:
                    out_dir = str(resumable_dir)
                    resume_generation = True
                    if isinstance(resume_state_candidate, dict):
                        resume_state_hint = dict(resume_state_candidate)

        if resume_generation:
            try:
                resume_report = reconcile_generation_state(
                    Path(out_dir),
                    resume_state_hint if isinstance(resume_state_hint, dict) else {},
                )
                resume_state_hint = dict(resume_report.repaired_state)
            except Exception as e:
                proceed = messagebox.askyesno(
                    "Resume benchmark set",
                    "The benchmark-set consistency check failed:\n\n"
                    f"{type(e).__name__}: {e}\n\n"
                    "Continue with the raw state anyway?",
                )
                if not proceed:
                    return
            else:
                if resume_report.changed or resume_report.warnings:
                    proceed = messagebox.askyesno(
                        "Resume consistency check",
                        "The selected benchmark set was checked against the files on disk.\n\n"
                        f"{resume_report.summary() or 'No changes required.'}\n\n"
                        "Continue with the repaired state?",
                    )
                    if not proceed:
                        return

            if isinstance(resume_state_hint, dict):
                try:
                    ranked_candidates = [int(x) for x in (resume_state_hint.get("ranked_candidates") or ranked_candidates)]
                except Exception:
                    pass
                try:
                    candidate_search_pool = [int(x) for x in (resume_state_hint.get("candidate_search_pool") or candidate_search_pool)]
                except Exception:
                    pass
                try:
                    k = int(resume_state_hint.get("requested_cases") or k)
                except Exception:
                    pass

        k = max(1, min(int(k), len(candidate_search_pool)))
        os.makedirs(out_dir, exist_ok=True)

        # Export analysis artefacts (plots + TeX table) into the benchmark folder for paper usage.
        # Keep this anchored to the preferred shortlist so the paper-facing exports
        # stay aligned with the user-visible Analyse ranking, while the actual
        # benchmark generation may backfill from deeper-ranked candidates.
        try:
            app._export_benchmark_paper_assets(Path(out_dir), a, ranked_candidates[:k])
        except Exception as e:
            print(f"[warn] Failed to export paper assets into benchmark folder: {type(e).__name__}: {e}")

        # For a benchmark set we ALWAYS generate a runner skeleton (otherwise the suite isn't runnable).
        do_runner = True

        # Runner skeleton target (auto/cpu/cuda/tensorrt). Read it once here so the
        # background worker does not access Tk variables.
        runner_target = "auto"
        try:
            runner_target = str(app.var_runner_target.get() or "auto").strip().lower()
        except Exception:
            runner_target = "auto"
        if runner_target not in {"auto", "cpu", "cuda", "tensorrt"}:
            runner_target = "auto"

        # ---------------- Accelerators to benchmark (suite plan) ----------------
        # Read once here so the worker thread does not touch Tk variables.
        acc_cpu = bool(getattr(app, "var_bench_acc_cpu", tk.BooleanVar(value=True)).get())
        acc_cuda = bool(getattr(app, "var_bench_acc_cuda", tk.BooleanVar(value=False)).get())
        acc_trt = bool(getattr(app, "var_bench_acc_tensorrt", tk.BooleanVar(value=False)).get())
        acc_h8 = bool(getattr(app, "var_bench_acc_hailo8", tk.BooleanVar(value=False)).get())
        acc_h10 = bool(getattr(app, "var_bench_acc_hailo10", tk.BooleanVar(value=False)).get())

        if not any([acc_cpu, acc_cuda, acc_trt, acc_h8, acc_h10]):
            # Defensive default (otherwise the suite is pointless).
            acc_cpu = True

        # Resolve Hailo hw_arch values from Split&Export settings (single source of truth).
        hailo8_hw = (getattr(app, "var_hailo_hef_hailo8_hw_arch", tk.StringVar(value="hailo8")).get() or "hailo8").strip()
        hailo10_hw = (getattr(app, "var_hailo_hef_hailo10_hw_arch", tk.StringVar(value="hailo10h")).get() or "hailo10h").strip()

        # Per-run image scaling (passed through to the runner harness).
        plan_image_scale = (getattr(app, "var_bench_image_scale", tk.StringVar(value="auto")).get() or "auto").strip().lower()
        plan_validation_images = (getattr(app, "var_bench_validation_images", tk.StringVar(value="")).get() or "").strip()
        try:
            plan_validation_max_images = int((getattr(app, "var_bench_validation_max_images", tk.StringVar(value="0")).get() or "0").strip())
        except Exception:
            plan_validation_max_images = 0
        plan_validation_reference_mode = (getattr(app, "var_bench_validation_reference_mode", tk.StringVar(value="auto")).get() or "auto").strip().lower()
        plan_benchmark_task = (getattr(app, "var_bench_task", tk.StringVar(value="auto")).get() or "auto").strip().lower()
        plan_mini_coco_ap50 = bool(getattr(app, "var_bench_mini_coco_ap50", tk.BooleanVar(value=False)).get())
        plan_mini_classification_eval = bool(getattr(app, "var_bench_mini_classification_eval", tk.BooleanVar(value=False)).get())

        # Normalize the full-model HEF build order once on the GUI thread so the
        # worker and services only see backend tokens (start/end/skip).
        full_hef_policy = normalize_full_hef_policy(
            getattr(
                app,
                "var_hailo_full_hef_order",
                tk.StringVar(value="Build at end (recommended)"),
            ).get()
        )
        full_model_preflight_policy = normalize_hailo_full_model_preflight_policy(
            getattr(
                app,
                "var_hailo_full_model_preflight",
                tk.StringVar(value="Enabled (plan-aware)"),
            ).get()
        )
        plan_hailo_preset = str(getattr(app, "var_hailo_bench_preset", tk.StringVar(value="End-to-end compare")).get() or "")
        plan_hailo_custom_full = bool(getattr(app, "var_hailo_bench_custom_full", tk.BooleanVar(value=True)).get())
        plan_hailo_custom_composed = bool(getattr(app, "var_hailo_bench_custom_composed", tk.BooleanVar(value=True)).get())
        plan_hailo_custom_part1 = bool(getattr(app, "var_hailo_bench_custom_part1", tk.BooleanVar(value=False)).get())
        plan_hailo_custom_part2 = bool(getattr(app, "var_hailo_bench_custom_part2", tk.BooleanVar(value=False)).get())
        plan_matrix_trt_to_hailo = bool(getattr(app, "var_matrix_trt_to_hailo", tk.BooleanVar(value=False)).get())
        plan_matrix_hailo_to_trt = bool(getattr(app, "var_matrix_hailo_to_trt", tk.BooleanVar(value=False)).get())

        if profile_overrides:
            acc_cpu = bool(profile_overrides.get('acc_cpu', acc_cpu))
            acc_cuda = bool(profile_overrides.get('acc_cuda', acc_cuda))
            acc_trt = bool(profile_overrides.get('acc_trt', acc_trt))
            acc_h8 = bool(profile_overrides.get('acc_h8', acc_h8))
            acc_h10 = bool(profile_overrides.get('acc_h10', acc_h10))
            plan_image_scale = str(profile_overrides.get('image_scale', plan_image_scale) or plan_image_scale)
            plan_validation_images = str(profile_overrides.get('validation_images', plan_validation_images) or '')
            try:
                plan_validation_max_images = int(profile_overrides.get('validation_max_images', plan_validation_max_images) or 0)
            except Exception:
                pass
            plan_validation_reference_mode = str(profile_overrides.get('validation_reference_mode', plan_validation_reference_mode) or plan_validation_reference_mode)
            plan_benchmark_task = str(profile_overrides.get('benchmark_task', plan_benchmark_task) or plan_benchmark_task)
            plan_mini_coco_ap50 = bool(profile_overrides.get('mini_coco_ap50', plan_mini_coco_ap50))
            plan_mini_classification_eval = bool(profile_overrides.get('mini_classification_eval', plan_mini_classification_eval))
            plan_hailo_preset = str(profile_overrides.get('hailo_preset', plan_hailo_preset) or plan_hailo_preset)
            plan_hailo_custom_full = bool(profile_overrides.get('hailo_custom_full', plan_hailo_custom_full))
            plan_hailo_custom_composed = bool(profile_overrides.get('hailo_custom_composed', plan_hailo_custom_composed))
            plan_hailo_custom_part1 = bool(profile_overrides.get('hailo_custom_part1', plan_hailo_custom_part1))
            plan_hailo_custom_part2 = bool(profile_overrides.get('hailo_custom_part2', plan_hailo_custom_part2))
            plan_matrix_trt_to_hailo = bool(profile_overrides.get('matrix_trt_to_hailo', plan_matrix_trt_to_hailo))
            plan_matrix_hailo_to_trt = bool(profile_overrides.get('matrix_hailo_to_trt', plan_matrix_hailo_to_trt))
            if 'full_model_preflight_policy' in profile_overrides:
                full_model_preflight_policy = normalize_hailo_full_model_preflight_policy(profile_overrides.get('full_model_preflight_policy'))

        run_plan = generation_service.build_run_plan(
            acc_cpu=bool(acc_cpu),
            acc_cuda=bool(acc_cuda),
            acc_trt=bool(acc_trt),
            acc_h8=bool(acc_h8),
            acc_h10=bool(acc_h10),
            hailo8_hw=str(hailo8_hw or ""),
            hailo10_hw=str(hailo10_hw or ""),
            image_scale=str(plan_image_scale or "auto"),
            validation_images=str(plan_validation_images or ""),
            validation_max_images=int(max(0, int(plan_validation_max_images or 0))),
            validation_reference_mode=str(plan_validation_reference_mode or "auto"),
            mini_coco_ap50=bool(plan_mini_coco_ap50),
            benchmark_task=str(plan_benchmark_task or "auto"),
            mini_classification_eval=bool(plan_mini_classification_eval),
            hailo_preset=str(plan_hailo_preset or ""),
            hailo_custom_full=bool(plan_hailo_custom_full),
            hailo_custom_composed=bool(plan_hailo_custom_composed),
            hailo_custom_part1=bool(plan_hailo_custom_part1),
            hailo_custom_part2=bool(plan_hailo_custom_part2),
            matrix_trt_to_hailo=bool(plan_matrix_trt_to_hailo),
            matrix_hailo_to_trt=bool(plan_matrix_hailo_to_trt),
            full_hef_policy=str(full_hef_policy or "end"),
        )
        bench_plan_runs: List[Dict[str, Any]] = list(run_plan.bench_plan_runs)
        hef_targets: List[str] = list(run_plan.hef_targets)
        hailo_selected = bool(run_plan.hailo_selected)
        hef_full = bool(run_plan.hef_full)
        hef_part1 = bool(run_plan.hef_part1)
        hef_part2 = bool(run_plan.hef_part2)
        hailo_variants = list(run_plan.hailo_variants)

        hailo_compile_rank_meta: Dict[int, Dict[str, Any]] = {}
        hailo_outlook_rows = []
        hailo_outlook_summary = None
        try:
            analysis_candidate_rows_snapshot = [
                dict(row)
                for row in (getattr(app, "_candidate_rows", None) or getattr(app, "candidates", None) or [])
                if isinstance(row, dict)
            ]
        except Exception:
            analysis_candidate_rows_snapshot = []
        if isinstance(a, dict) and candidate_search_pool:
            analysis_for_plan = dict(a)
            if analysis_candidate_rows_snapshot:
                analysis_for_plan["_candidate_rows"] = list(analysis_candidate_rows_snapshot)
            try:
                _plan = generation_service.prepare_generation_plan(
                    analysis_for_plan,
                    ranked_candidates,
                    candidate_search_pool,
                    k,
                    strict_boundary=False,
                    hailo_selected=bool(hailo_selected),
                    outlook_top_n=12,
                )
                ranked_candidates = list(_plan.ranked_candidates)
                candidate_search_pool = list(_plan.candidate_search_pool)
                k = int(_plan.requested_cases)
                hailo_compile_rank_meta = dict(_plan.hailo_compile_rank_meta)
                hailo_outlook_rows = list(_plan.hailo_outlook_rows)
                hailo_outlook_summary = _plan.hailo_outlook_summary
            except Exception as _hailo_rank_exc:
                logger.debug('BenchmarkGenerationService.prepare_generation_plan failed: %s', _hailo_rank_exc)
                if hailo_selected and isinstance(a, dict):
                    try:
                        _reranked_pool, hailo_compile_rank_meta = rerank_candidates_for_hailo(analysis_for_plan, candidate_search_pool)
                        if _reranked_pool:
                            _order = {int(b): idx for idx, b in enumerate(_reranked_pool)}
                            candidate_search_pool = list(_reranked_pool)
                            ranked_candidates = sorted([int(b) for b in ranked_candidates], key=lambda b: (_order.get(int(b), 10**9), int(b)))
                    except Exception as _fallback_exc:
                        logger.debug('Hailo compile-aware fallback reranking failed: %s', _fallback_exc)

        hef_opt_level = _safe_int((getattr(app, "var_hailo_hef_opt_level", tk.StringVar(value="1")).get() or "").strip()) or 1
        hef_calib_count = _safe_int((getattr(app, "var_hailo_hef_calib_count", tk.StringVar(value="64")).get() or "").strip()) or 64
        hef_calib_bs = _safe_int((getattr(app, "var_hailo_hef_calib_batch_size", tk.StringVar(value="8")).get() or "").strip()) or 8
        hef_calib_dir = (getattr(app, "var_hailo_hef_calib_dir", tk.StringVar(value="")).get() or "").strip() or None
        hef_force = bool(getattr(app, "var_hailo_hef_force", tk.BooleanVar(value=False)).get())
        hef_keep = bool(getattr(app, "var_hailo_hef_keep_artifacts", tk.BooleanVar(value=False)).get())

        # Backend selection reuses the Hailo feasibility-check backend controls.
        hef_backend = normalize_hailo_backend(getattr(app, "var_hailo_backend", tk.StringVar(value="auto")).get())
        hef_wsl_distro = (getattr(app, "var_hailo_wsl_distro", tk.StringVar(value="")).get() or "").strip() or None
        hef_wsl_venv = (getattr(app, "var_hailo_wsl_venv", tk.StringVar(value="auto")).get() or "auto").strip() or "auto"
        hef_fixup = bool(getattr(app, "var_hailo_fixup", tk.BooleanVar(value=True)).get())

        do_ctx_full = bool(getattr(app, 'var_split_ctx_full', tk.BooleanVar(value=True)).get())
        do_ctx_cutflow = bool(getattr(app, 'var_split_ctx_cutflow', tk.BooleanVar(value=False)).get())
        ctx_hops = _safe_int(getattr(app, 'var_split_ctx_hops', tk.StringVar(value='2')).get()) or 2

        eps_txt = (app.var_split_eps.get() or "").strip()
        eps_default = 1e-4
        if eps_txt:
            try:
                eps_default = float(eps_txt)
            except Exception:
                eps_default = 1e-4

        # Read batch override once here (avoid reading Tk variables from worker thread).
        params = None
        try:
            params = app._read_params()
            batch_override = params.batch_override
        except Exception:
            batch_override = None

        # Determine a nice padding width for folder names. Prefer the full
        # search pool because benchmark generation may backfill beyond the currently
        # displayed shortlist.
        _pad_source = list(candidate_search_pool or ranked_candidates or app.current_picks or [0])
        pad = max(3, len(str(max(_pad_source)))) if _pad_source else 3

        # Dedicated per-run generation log (mirrors the live dialog output and HEF
        # sub-logs). This makes post-mortems possible even when the GUI log is long
        # or the host reboots mid-run.
        bench_log_path = os.path.join(out_dir, "benchmark_generation.log")

        # Snapshot GUI state before the background worker starts. Tk falls back to
        # ``app.tk`` lookups for missing attributes, so worker-thread access to a
        # missing ``app.candidates`` attribute turned into
        # ``AttributeError: '_tkinter.tkapp' object has no attribute 'candidates'``.
        # Keep the worker on plain Python data only.
        benchmark_gap = int(_safe_int(app.var_min_gap.get()) or 0)
        if profile_overrides:
            try:
                benchmark_gap = int(profile_overrides.get('min_gap', benchmark_gap) or 0)
            except Exception:
                pass
        llm_style_enabled = bool(app.var_llm_enable.get())
        value_bytes_map_snapshot = a.get("value_bytes") if isinstance(a, dict) else None
        analysis_topk_snapshot = int(getattr(params, 'topk', len(app.current_picks)))
        system_spec_payload = asdict(app._build_system_spec(params)) if params else None
        model_path_for_worker = str(model_path)
        full_model_src_for_worker = os.path.abspath(model_path_for_worker)
        hef_timeout_s = int(app._hailo_hef_timeout_seconds())
        hailo_publish_gui_diagnostics_cb = app._hailo_publish_gui_diagnostics
        analysis_predicted_metrics_for_boundary_fn = app._analysis_predicted_metrics_for_boundary
        hailo_parse_entry_for_boundary_fn = app._hailo_parse_entry_for_boundary
        hailo_parse_scalar_fields_fn = app._hailo_parse_scalar_fields
        hailo_part2_enable_suggested_endnode_fallback = (
            bool(getattr(app, 'var_bench_hailo_part2_suggested_fallback').get())
            if hasattr(app, 'var_bench_hailo_part2_suggested_fallback')
            else True
        )
        benchmark_objective_raw = (
            str(getattr(app, 'var_bench_objective', tk.StringVar(value='Use analysis objective')).get() or 'Use analysis objective').strip()
            if hasattr(app, 'var_bench_objective') else 'Use analysis objective'
        )
        if benchmark_objective_raw.lower().startswith('use analysis'):
            benchmark_objective = str(getattr(app, 'var_analysis_objective', tk.StringVar(value='Balanced')).get() or 'Balanced').strip()
        else:
            benchmark_objective = benchmark_objective_raw
        if not benchmark_objective:
            benchmark_objective = 'Balanced'
        orchestration_service = getattr(
            app,
            '_benchmark_generation_orchestration_service',
            BenchmarkGenerationOrchestrationService(generation_service, getattr(app, '_benchmark_generation_execution_service', None)),
        )

        # --- progress dialog + background worker ---
        job_id = f"generate-{base}-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        cancel_event = threading.Event()
        app._jobs_register(
            job_id=job_id,
            kind="generate",
            type_label="Benchmark set",
            title=f"Generating benchmark set — {base}",
            name=str(Path(out_dir).name or base),
            output_dir=str(out_dir),
            log_path=str(bench_log_path),
            initial_status=f"Generating benchmark set with target of up to {k} accepted cases…",
            initial_lines=[
                f"Generating benchmark set with target of up to {k} accepted cases…",
                f"Output dir: {out_dir}",
                "Runs in the background. Close this window any time and reopen it from the Jobs tab.",
            ],
            progress_maximum=max(1, k),
            cancel_callback=lambda: cancel_event.set(),
            can_cancel=True,
            geometry="900x420",
        )

        q: "queue.Queue[tuple]" = queue.Queue()

        def _write_benchmark_suite_script(dst_dir: str, bench_json_name: str = "benchmark_set.json") -> str:
            """Write benchmark suite runner script from a template resource."""
            return write_benchmark_suite_script(dst_dir, bench_json_name=bench_json_name)

        def worker() -> None:
            nonlocal ranked_candidates, candidate_search_pool
            # Legacy regression note: discarded cases are still tracked for benchmark generation,
            # but now live inside BenchmarkGenerationRuntime / BenchmarkGenerationExecutionService.
            # discarded_cases = []
            runtime = None
            try:
                runtime = generation_service.start_generation_runtime(
                    out_dir=out_dir,
                    bench_log_path=bench_log_path,
                    requested_cases=int(k),
                    ranked_candidates=ranked_candidates,
                    candidate_search_pool=candidate_search_pool,
                    hef_full_policy=full_hef_policy,
                    model_name=base,
                    model_source=model_path_for_worker,
                    resume_generation=bool(resume_generation),
                    resume_state_hint=resume_state_hint,
                )
                cases = runtime.cases
                errors = runtime.errors
                discarded_cases = runtime.discarded_cases
                suite_hailo_hefs = runtime.suite_hailo_hefs
                completed_boundaries = runtime.completed_boundaries
                accepted_boundaries = runtime.accepted_boundaries
                discarded_boundaries = runtime.discarded_boundaries
                made = int(len(cases))

                def log(line: str, *, level: int = logging.INFO) -> None:
                    runtime.log(line, queue_put=q.put, level=level)

                log("=== Generating benchmark set ===")
                log(f"out_dir: {out_dir}")
                log(f"model_path: {model_path_for_worker}")
                log(f"generation_log: {bench_log_path}")
                log(f"requested_cases (target accepted): {k}")
                log(f"preferred shortlist: {len(ranked_candidates)}")
                log(f"candidate search pool: {len(candidate_search_pool)}")
                if resume_generation and runtime.completed_boundaries:
                    log(f"[resume] continuing existing benchmark set ({len(cases)} accepted, {len(discarded_cases)} discarded)")
                if resume_generation and resume_report is not None:
                    summary = (resume_report.summary() or "No resume consistency changes required.").splitlines()
                    for line in summary:
                        log(f"[resume] {line}" if line else "[resume]")

                full_model_src = full_model_src_for_worker
                full_model_dst = generation_service.copy_portable_full_model(runtime, full_model_src, log_cb=log)
                log(f"full model (suite copy): {full_model_dst}")

                # Resolve Hailo helpers once; the top-level orchestration now lives in
                # BenchmarkGenerationOrchestrationService instead of gui_app.py.
                hailo_build_hef_fn = None
                hailo_parse_check_fn = None
                hailo_build_unavailable: Optional[str] = None
                hailo_part2_precheck_fn = None
                hailo_part2_precheck_error_fn = None
                hailo_part2_parser_precheck_fn = None
                hailo_part2_parser_precheck_error_fn = None

                def _persist_generation_state(status: str = 'running', current_boundary: Optional[int] = None) -> None:
                    runtime.persist(status=status, current_boundary=current_boundary)

                resolved_hailo_helpers = resolve_hailo_benchmark_helpers(
                    need_build=bool(hef_targets and (hef_full or hef_part1 or hef_part2)),
                    need_part2=bool(hef_targets and hef_part2),
                )
                hailo_build_hef_fn = resolved_hailo_helpers.hailo_build_hef_fn
                hailo_parse_check_fn = resolved_hailo_helpers.hailo_parse_check_fn
                hailo_build_unavailable = resolved_hailo_helpers.hailo_build_unavailable
                hailo_part2_precheck_fn = resolved_hailo_helpers.hailo_part2_precheck_fn
                hailo_part2_precheck_error_fn = resolved_hailo_helpers.hailo_part2_precheck_error_fn
                hailo_part2_parser_precheck_fn = resolved_hailo_helpers.hailo_part2_parser_precheck_fn
                hailo_part2_parser_precheck_error_fn = resolved_hailo_helpers.hailo_part2_parser_precheck_error_fn

                if hailo_build_unavailable:
                    errors.append(hailo_build_unavailable)
                    log(hailo_build_unavailable)
                if resolved_hailo_helpers.hailo_part2_import_error:
                    log(f"hailo part2 precheck unavailable: {resolved_hailo_helpers.hailo_part2_import_error}")

                preferred_shortlist_original = list(ranked_candidates)
                benign_discard_reasons = {
                    "hailo_part2_prefilter",
                    "hailo_part2_precheck",
                    "hailo_part2_auto_filtered",
                    "hailo_part2_parser_prefilter",
                    "hailo_part2_parser_auto_filtered",
                    "hailo_part2_concat_sanity_prefilter",
                    "hailo_part2_concat_sanity_auto_filtered",
                    "hailo_failure_cluster_skip",
                }

                execution_cfg = BenchmarkGenerationExecutionConfig(
                    runtime=runtime,
                    target_cases=int(k),
                    gap=int(benchmark_gap),
                    ranked_candidates=list(ranked_candidates),
                    candidate_search_pool=list(candidate_search_pool),
                    out_dir=Path(out_dir),
                    base=base,
                    pad=int(pad),
                    strict_boundary=bool(strict_boundary),
                    model=model,
                    nodes=nodes,
                    order=order,
                    analysis_payload=a,
                    analysis_candidates=list(analysis_candidate_rows_snapshot),
                    bench_plan_runs=list(bench_plan_runs),
                    runner_target=runner_target,
                    do_ctx_full=bool(do_ctx_full),
                    do_ctx_cutflow=bool(do_ctx_cutflow),
                    ctx_hops=int(ctx_hops),
                    llm_style=bool(llm_style_enabled),
                    value_bytes_map=value_bytes_map_snapshot,
                    full_model_src=str(full_model_src),
                    full_model_dst=str(full_model_dst),
                    tool_gui_version=__version__,
                    tool_core_version=resolve_tool_core_version(),
                    evaluation_profile_meta=(evaluation_profile_resolution.to_metadata() if evaluation_profile_resolution is not None else None),
                    hailo_compile_rank_meta=dict(hailo_compile_rank_meta or {}),
                    hef_targets=list(hef_targets),
                    hef_part1=bool(hef_part1),
                    hef_part2=bool(hef_part2),
                    hef_backend=str(hef_backend),
                    hef_fixup=bool(hef_fixup),
                    hef_opt_level=int(hef_opt_level),
                    hef_calib_dir=hef_calib_dir,
                    hef_calib_count=int(hef_calib_count),
                    hef_calib_bs=int(hef_calib_bs),
                    hef_force=bool(hef_force),
                    hef_keep=bool(hef_keep),
                    hef_wsl_distro=hef_wsl_distro,
                    hef_wsl_venv=str(hef_wsl_venv),
                    hef_timeout_s=int(hef_timeout_s),
                    hailo_build_hef_fn=hailo_build_hef_fn,
                    hailo_build_unavailable=hailo_build_unavailable,
                    hailo_part2_precheck_fn=hailo_part2_precheck_fn,
                    hailo_part2_precheck_error_fn=hailo_part2_precheck_error_fn,
                    hailo_part2_parser_precheck_fn=hailo_part2_parser_precheck_fn,
                    hailo_part2_parser_precheck_error_fn=hailo_part2_parser_precheck_error_fn,
                    hailo_part2_enable_suggested_endnode_fallback=bool(hailo_part2_enable_suggested_endnode_fallback),
                    should_cancel=lambda: bool(cancel_event.is_set()),
                )
                execution_callbacks = BenchmarkGenerationExecutionCallbacks(
                    log=log,
                    queue_put=q.put,
                    persist_state=lambda **kwargs: _persist_generation_state(**kwargs),
                    publish_hailo_diagnostics=lambda label, result, log_cb: hailo_publish_gui_diagnostics_cb(label, result, log_cb=log_cb),
                    predicted_metrics_for_boundary=lambda analysis_payload, boundary: analysis_predicted_metrics_for_boundary_fn(analysis_payload, boundary),
                    hailo_parse_entry_for_boundary=lambda analysis_payload, boundary: hailo_parse_entry_for_boundary_fn(analysis_payload, boundary),
                    hailo_parse_scalar_fields=lambda entry: hailo_parse_scalar_fields_fn(entry),
                )

                analysis_params_payload = {
                    'objective': (str(getattr(app, 'var_analysis_objective', tk.StringVar(value='Balanced')).get() or 'Balanced') if app is not None else 'Balanced'),
                    'ranking': str(getattr(params, 'ranking', 'score')),
                    'topk': int(analysis_topk_snapshot),
                    'min_gap': int(getattr(params, 'min_gap', 0)),
                    'exclude_trivial': bool(getattr(params, 'exclude_trivial', False)),
                    'only_single_tensor': bool(getattr(params, 'only_single_tensor', False)),
                    'strict_boundary': bool(getattr(params, 'strict_boundary', False)),
                    'prune_skip_block': bool(getattr(params, 'prune_skip_block', False)),
                    'skip_min_span': int(getattr(params, 'skip_min_span', 0)),
                    'skip_allow_last_n': int(getattr(params, 'skip_allow_last_n', 0)),
                    'link_model': str(getattr(params, 'link_model', 'ideal')),
                    'bandwidth_value': getattr(params, 'bw_value', None),
                    'bandwidth_unit': str(getattr(params, 'bw_unit', 'MB/s')),
                    'gops_left': getattr(params, 'gops_left', None),
                    'gops_right': getattr(params, 'gops_right', None),
                    'link_overhead_ms': getattr(params, 'overhead_ms', 0.0),
                    'link_energy_pj_per_byte': getattr(params, 'link_energy_pj_per_byte', None),
                    'link_mtu_payload_bytes': getattr(params, 'link_mtu_payload_bytes', None),
                    'link_per_packet_overhead_ms': getattr(params, 'link_per_packet_overhead_ms', None),
                    'link_per_packet_overhead_bytes': getattr(params, 'link_per_packet_overhead_bytes', None),
                    'energy_pj_per_flop_left': getattr(params, 'energy_pj_per_flop_left', None),
                    'energy_pj_per_flop_right': getattr(params, 'energy_pj_per_flop_right', None),
                    'link_max_latency_ms': getattr(params, 'link_max_latency_ms', None),
                    'link_max_energy_mJ': getattr(params, 'link_max_energy_mJ', None),
                    'link_max_bytes': getattr(params, 'link_max_bytes', None),
                    'max_peak_act_left': getattr(params, 'max_peak_act_left', None),
                    'max_peak_act_left_unit': str(getattr(params, 'max_peak_act_left_unit', 'MiB')),
                    'max_peak_act_right': getattr(params, 'max_peak_act_right', None),
                    'max_peak_act_right_unit': str(getattr(params, 'max_peak_act_right_unit', 'MiB')),
                    'batch_override': batch_override,
                    'eps_default': float(eps_default),
                }
                resume_lines = []
                if resume_generation and resume_report is not None and (resume_report.changed or resume_report.warnings):
                    resume_lines = [line for line in (resume_report.summary() or '').splitlines() if line.strip()]

                prepared_full_hailo_baseline = _prepared_full_hailo_baseline(model_path)
                hailo_full_end_node_names, hailo_full_endpoint_mode = _prepared_full_hailo_endpoint_override(model_path)
                if not hailo_full_end_node_names and bool(prepared_full_hailo_baseline.get('ok')):
                    hailo_full_end_node_names = [str(x).strip() for x in list(prepared_full_hailo_baseline.get('end_node_names') or []) if str(x).strip()]
                    hailo_full_endpoint_mode = str(prepared_full_hailo_baseline.get('endpoint_mode') or hailo_full_endpoint_mode or '')
                if hailo_full_end_node_names:
                    log(
                        "suite: using prepared full-Hailo endpoint override "
                        f"mode={hailo_full_endpoint_mode or 'custom'} end_nodes={hailo_full_end_node_names}"
                    )
                if bool(prepared_full_hailo_baseline.get('selected')):
                    if bool(prepared_full_hailo_baseline.get('ok')):
                        log(
                            "suite: prepared full-Hailo HEF baseline available "
                            f"mode={prepared_full_hailo_baseline.get('endpoint_mode') or 'full'} "
                            f"hef={prepared_full_hailo_baseline.get('hef_path')}"
                        )
                    else:
                        log(
                            "suite: prepared full-Hailo metadata found but no usable HEF baseline is available "
                            f"({prepared_full_hailo_baseline.get('reason') or 'unknown'})"
                        )
                execution_cfg = replace(
                    execution_cfg,
                    hailo_full_end_node_names=list(hailo_full_end_node_names or []),
                    hailo_full_endpoint_mode=str(hailo_full_endpoint_mode or ''),
                    hailo_full_output_contract=(
                        dict(prepared_full_hailo_baseline.get('output_contract') or {})
                        if isinstance(prepared_full_hailo_baseline.get('output_contract'), dict)
                        else None
                    ),
                )

                orchestration_cfg = BenchmarkGenerationOrchestrationConfig(
                    runtime=runtime,
                    execution_cfg=execution_cfg,
                    execution_callbacks=execution_callbacks,
                    target_cases=int(k),
                    preferred_shortlist_original=preferred_shortlist_original,
                    ranked_candidates=list(ranked_candidates),
                    candidate_search_pool=list(candidate_search_pool),
                    out_dir=Path(out_dir),
                    base=base,
                    pad=int(pad),
                    full_model_src=full_model_src,
                    full_model_dst=str(full_model_dst),
                    analysis_payload=a,
                    analysis_params_payload=analysis_params_payload,
                    system_spec_payload=system_spec_payload,
                    bench_log_path=str(bench_log_path),
                    bench_plan_runs=bench_plan_runs,
                    hef_targets=list(hef_targets),
                    hef_full=bool(hef_full),
                    hef_part1=bool(hef_part1),
                    hef_part2=bool(hef_part2),
                    hef_backend=str(hef_backend),
                    hef_fixup=bool(hef_fixup),
                    hef_opt_level=int(hef_opt_level),
                    hef_calib_dir=hef_calib_dir,
                    hef_calib_count=int(hef_calib_count),
                    hef_calib_bs=int(hef_calib_bs),
                    hef_force=bool(hef_force),
                    hef_keep=bool(hef_keep),
                    hef_wsl_distro=hef_wsl_distro,
                    hef_wsl_venv=str(hef_wsl_venv),
                    hef_timeout_s=int(hef_timeout_s),
                    full_hef_policy=str(full_hef_policy),
                    full_model_preflight_policy=str(full_model_preflight_policy or 'enabled'),
                    hailo_full_end_node_names=list(hailo_full_end_node_names or []),
                    hailo_full_endpoint_mode=str(hailo_full_endpoint_mode or ''),
                    hailo_full_output_contract=(
                        dict(prepared_full_hailo_baseline.get('output_contract') or {})
                        if isinstance(prepared_full_hailo_baseline.get('output_contract'), dict)
                        else None
                    ),
                    prepared_full_hailo_baseline=dict(prepared_full_hailo_baseline or {}),
                    hailo_build_hef_fn=hailo_build_hef_fn,
                    hailo_parse_check_fn=hailo_parse_check_fn,
                    hailo_build_unavailable=hailo_build_unavailable,
                    hailo_part2_precheck_fn=hailo_part2_precheck_fn,
                    hailo_part2_precheck_error_fn=hailo_part2_precheck_error_fn,
                    hailo_part2_parser_precheck_fn=hailo_part2_parser_precheck_fn,
                    hailo_part2_parser_precheck_error_fn=hailo_part2_parser_precheck_error_fn,
                    resume_generation=bool(resume_generation),
                    resume_report_summary_lines=resume_lines,
                    hailo_selected=bool(hailo_selected),
                    hailo_outlook_summary=hailo_outlook_summary,
                    benign_discard_reasons=sorted(benign_discard_reasons),
                    write_harness_script=_write_benchmark_suite_script,
                    copy_schema_tree=lambda: copy_resource_tree("resources", "schemas", dest=Path(out_dir) / "schemas"),
                    tool_gui_version=__version__,
                    tool_core_version=resolve_tool_core_version(),
                    evaluation_profile_meta=(evaluation_profile_resolution.to_metadata() if evaluation_profile_resolution is not None else None),
                    benchmark_objective=str(benchmark_objective),
                    should_cancel=lambda: bool(cancel_event.is_set()),
                )
                orchestration_result = orchestration_service.run(orchestration_cfg)
                ranked_candidates = list(orchestration_result.ranked_candidates)
                candidate_search_pool = list(orchestration_result.candidate_search_pool)
                final_kind = str(orchestration_result.final_status or 'warn').strip().lower()
                final_msg = str(orchestration_result.final_msg or '')
                final_summary = dict(orchestration_result.summary_data or {})
                is_clean_success = final_kind == 'ok'
                raw_text = str(final_summary.get('raw_text') or '').strip()
                log(raw_text or final_msg, level=logging.INFO if is_clean_success else logging.WARNING)
                if is_clean_success:
                    q.put(("ok", final_msg, final_summary))
                elif final_kind == 'cancelled':
                    q.put(("cancelled", final_msg, final_summary))
                else:
                    q.put(("warn", final_msg, final_summary))
            except Exception as e:
                logging.exception("Benchmark set generation failed")
                try:
                    if runtime is not None:
                        _persist_generation_state(status="failed", current_boundary=None)
                except Exception:
                    pass
                try:
                    if runtime is not None and runtime.bench_log_fp is not None:
                        traceback.print_exc(file=runtime.bench_log_fp)
                        runtime.bench_log_fp.flush()
                except Exception:
                    pass
                q.put(("err", f"{type(e).__name__}: {e}\n\nGeneration log: {bench_log_path}"))
            finally:
                try:
                    if runtime is not None:
                        runtime.close()
                except Exception:
                    pass

        app._set_background_job_active("generate", True)
        gen_thread = threading.Thread(target=worker, daemon=True)
        app._benchmark_generation_thread = gen_thread
        try:
            gen_thread.start()
        except Exception as exc:
            app._set_background_job_active("generate", False)
            app._jobs_finish(job_id, status="error", message="Failed to start benchmark-set generation thread", output_dir=str(out_dir), log_path=str(bench_log_path))
            if callable(completion_callback):
                try:
                    completion_callback('err', str(out_dir), str(bench_log_path), f'{type(exc).__name__}: {exc}', {})
                except Exception:
                    logger.debug('Benchmark generation completion callback failed during startup', exc_info=True)
            raise

        def poll() -> None:
            final_status: Optional[str] = None
            final_msg: str = ""
            final_summary: Dict[str, Any] = {}

            # Drain the queue so log output stays responsive even when a lot of
            # lines arrive quickly (e.g. Hailo DFC compilation).
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break

                if not item:
                    continue

                status = str(item[0])

                if status == 'prog':
                    try:
                        made = int(item[1])
                        what = str(item[2]) if len(item) > 2 else ''
                        app._jobs_set_progress(
                            job_id,
                            value=float(made),
                            label=(f"{made}/{k}: {what}" if what else f"{made}/{k}"),
                            display=f"{made}/{k}",
                            progress_maximum=max(1, k),
                        )
                    except Exception:
                        pass
                    continue

                if status in ('log', 'hef'):
                    try:
                        if status == 'log':
                            line = str(item[1]) if len(item) > 1 else ''
                        else:
                            # ('hef', stream, line)
                            line = str(item[2]) if len(item) > 2 else ''
                        app._jobs_append_log(job_id, line)
                        if status == 'hef' and line:
                            current_value = 0.0
                            try:
                                current_value = float((getattr(app, '_background_jobs', {}) or {}).get(job_id).progress_value)
                            except Exception:
                                current_value = 0.0
                            app._jobs_set_progress(job_id, value=current_value, label=line[:220])
                    except Exception:
                        pass
                    continue

                if status in ('msg', 'note'):
                    try:
                        what = str(item[1]) if len(item) > 1 else ''
                        current_value = 0.0
                        try:
                            current_value = float((getattr(app, '_background_jobs', {}) or {}).get(job_id).progress_value)
                        except Exception:
                            current_value = 0.0
                        app._jobs_set_progress(job_id, value=current_value, label=what)
                    except Exception:
                        pass
                    continue

                if status in ('ok', 'warn', 'err', 'cancelled'):
                    final_status = status
                    final_msg = str(item[1]) if len(item) > 1 else ''
                    final_summary = dict(item[2] or {}) if len(item) > 2 and isinstance(item[2], dict) else {}
                    break

            try:
                app.update_idletasks()
            except Exception:
                pass

            if not final_status:
                app.after(80, poll)
                return

            app._set_background_job_active("generate", False)
            status_map = {
                'ok': 'success',
                'warn': 'warning',
                'cancelled': 'cancelled',
                'err': 'error',
            }
            app._jobs_finish(
                job_id,
                status=status_map.get(final_status, 'error'),
                message=final_msg,
                output_dir=str(out_dir),
                log_path=str(bench_log_path),
            )

            if final_status in {'ok', 'warn'}:
                try:
                    bench_json_path = Path(out_dir) / 'benchmark_set.json'
                    if bench_json_path.exists() and hasattr(app, 'var_remote_benchmark_set'):
                        app.var_remote_benchmark_set.set(str(bench_json_path))
                        try:
                            if hasattr(app, '_persist_settings'):
                                app._persist_settings()
                        except Exception:
                            pass
                        try:
                            refresh_preview = getattr(app, '_benchmark_refresh_accuracy_ui', None)
                            if callable(refresh_preview):
                                refresh_preview()
                        except Exception:
                            logger.debug('Failed to refresh validate-tab accuracy preview after benchmark generation', exc_info=True)
                except Exception:
                    logger.debug('Failed to auto-select generated benchmark_set.json after generation', exc_info=True)

            if callable(completion_callback):
                try:
                    completion_callback(final_status, str(out_dir), str(bench_log_path), final_msg, final_summary)
                except Exception:
                    logger.debug('Benchmark generation completion callback failed', exc_info=True)

            if show_result_dialogs:
                if final_status in {'ok', 'warn', 'cancelled'} and final_summary:
                    dialog_title = {
                        'ok': 'Benchmark set created',
                        'warn': 'Benchmark set created (with warnings)',
                        'cancelled': 'Benchmark set cancelled',
                    }.get(final_status, 'Benchmark set')
                    show_benchmark_completion_dialog(
                        app,
                        title=dialog_title,
                        summary_data=final_summary,
                        fallback_text=final_msg,
                    )
                elif final_status == 'ok':
                    messagebox.showinfo("Benchmark set created", final_msg)
                elif final_status == 'warn':
                    messagebox.showwarning("Benchmark set created (with warnings)", final_msg)
                elif final_status == 'cancelled':
                    messagebox.showwarning("Benchmark set cancelled", final_msg)
                else:
                    messagebox.showerror("Benchmark set failed", final_msg)

        poll()
        return job_id
