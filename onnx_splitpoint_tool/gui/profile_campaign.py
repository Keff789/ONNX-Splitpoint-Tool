from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..benchmark.evaluation_profiles import (
    EvaluationProfileResolution,
    load_evaluation_profile,
    profile_model_entries,
    resolve_evaluation_profile,
    resolve_evaluation_profile_source,
    load_export_metadata_for_model,
)
from ..benchmark.model_preparation import (
    PreparationRuntimeOptions,
    normalize_model_preparation_mode,
    prepare_model_for_benchmark,
)
from ..benchmark.services import BenchmarkAnalysisService, RemoteBenchmarkArgs
from ..hailo.backend_mode import normalize_hailo_backend
from ..gui.state import AnalysisResult
from ..workdir import ensure_workdir
from ..workflow.artifacts import EvaluationRunContext, artifact_relpath, create_evaluation_run_context

logger = logging.getLogger(__name__)


def _norm_token(value: Any) -> str:
    s = str(value or '').strip().lower()
    s = Path(s).stem.lower() if any(ch in s for ch in '/\\.') else s
    out = []
    last_us = False
    for ch in s:
        ok = ('a' <= ch <= 'z') or ('0' <= ch <= '9')
        if ok:
            out.append(ch)
            last_us = False
        elif not last_us:
            out.append('_')
            last_us = True
    return ''.join(out).strip('_')


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def snapshot_preparation_runtime(app: Any) -> PreparationRuntimeOptions:
    return PreparationRuntimeOptions(
        hailo_backend=normalize_hailo_backend(getattr(getattr(app, 'var_hailo_backend', None), 'get', lambda: 'auto')()),
        hw_arch='hailo8',
        hef_fixup=True,
        hef_opt_level=_safe_int(getattr(getattr(app, 'var_hailo_hef_opt_level', None), 'get', lambda: '1')(), 1),
        calib_dir=(str(getattr(getattr(app, 'var_hailo_hef_calib_dir', None), 'get', lambda: '')() or '').strip() or None),
        calib_count=_safe_int(getattr(getattr(app, 'var_hailo_hef_calib_count', None), 'get', lambda: '64')(), 64),
        calib_batch_size=_safe_int(getattr(getattr(app, 'var_hailo_hef_calib_batch_size', None), 'get', lambda: '8')(), 8),
        hef_force=True,
        hef_keep_artifacts=True,
        wsl_distro=(str(getattr(getattr(app, 'var_hailo_wsl_distro', None), 'get', lambda: '')() or '').strip() or None),
        wsl_venv=(str(getattr(getattr(app, 'var_hailo_wsl_venv', None), 'get', lambda: 'auto')() or 'auto').strip() or 'auto'),
        hef_timeout_s=3600,
        verify_mode='ort',
        device='cpu',
    )


@dataclass
class CampaignModelMatch:
    model_id: str
    task: str
    family: str
    tier: str
    model_path: Path
    export_metadata: Dict[str, Any]
    entry: Dict[str, Any]


@dataclass
class ProfileCampaignOptions:
    profile_request: str
    models_root: Path
    benchmark_parent_dir: Path
    include_reserve: bool = False
    auto_remote_run: bool = False
    auto_export_analysis: bool = False
    remote_host: Any = None
    remote_defaults: Optional[Dict[str, Any]] = None
    model_preparation_mode: str = 'current'
    preparation_output_root: Optional[Path] = None
    preparation_runtime: Optional[PreparationRuntimeOptions] = None
    evaluation_run_root: Optional[Path] = None
    workflow_run_id: Optional[str] = None
    resume_workflow: bool = False


class ProfileCampaignError(RuntimeError):
    pass


def _call_on_tk(app: Any, fn, *, timeout_s: float = 300.0):
    q: 'queue.Queue[Tuple[str, Any]]' = queue.Queue(maxsize=1)

    def _runner() -> None:
        try:
            q.put(('ok', fn()))
        except Exception as exc:
            q.put(('err', exc))

    app.root.after(0, _runner)
    status, payload = q.get(timeout=timeout_s)
    if status == 'ok':
        return payload
    raise payload


def _apply_model_loaded(app: Any, model_path: Path) -> None:
    path = str(model_path)
    try:
        clear = getattr(app, '_clear_results', None)
        if callable(clear):
            clear()
    except Exception:
        logger.debug('Failed to clear previous analysis state before profile campaign model load', exc_info=True)
    app.model_path = path
    app.gui_state.current_model_path = path
    app.gui_state.model_type = 'onnx'
    try:
        if hasattr(app, 'lbl_model'):
            app.lbl_model.configure(text=os.path.basename(path))
    except Exception:
        pass
    try:
        app.events.emit_model_loaded({'path': path, 'model_type': 'onnx'})
    except Exception:
        pass


def _apply_analysis_result(app: Any, analysis: Dict[str, Any], picks: List[int], params: Any) -> None:
    app.analysis = analysis
    app.current_picks = list(picks)
    app.analysis_result = AnalysisResult(
        candidates=list(app.current_picks),
        plot_data={'analysis': app.analysis, 'picks': list(app.current_picks), 'params': params},
    )
    app._last_params = params
    try:
        app.events.emit_analysis_done(app.analysis_result)
    except Exception:
        pass
    try:
        save_cache = getattr(app, '_save_hailo_cache', None)
        if callable(save_cache):
            save_cache()
    except Exception:
        logger.debug('Failed to save Hailo cache after automated profile-campaign analysis', exc_info=True)


def discover_profile_models(
    profile_raw: Mapping[str, Any],
    models_root: Path,
    *,
    include_reserve: bool = False,
    profile_dir: Optional[Path] = None,
) -> tuple[List[CampaignModelMatch], List[str]]:
    """Resolve profile model entries for the GUI workflow.

    v49b keeps the model-root scan used by the older Benchmark/Profile Campaign
    UI, but also accepts explicit ONNX paths in Evaluation Profiles.  That keeps
    the final workflow comfortable: a thesis profile may fully describe models
    without forcing the user to reselect a root folder manually.
    """
    root = Path(models_root).expanduser().resolve() if str(models_root or '').strip() else Path.cwd().resolve()
    base = Path(profile_dir).expanduser().resolve() if profile_dir is not None and str(profile_dir).strip() else None
    entries = profile_model_entries(profile_raw, include_reserve=include_reserve)

    matched: List[CampaignModelMatch] = []
    missing: List[str] = []
    pending_scan: List[Mapping[str, Any]] = []
    taken: set[str] = set()

    def _explicit_path(entry: Mapping[str, Any]) -> Optional[Path]:
        raw = entry.get('resolved_path') or entry.get('onnx') or entry.get('model_path') or entry.get('path')
        raw_s = str(raw or '').strip()
        if not raw_s:
            return None
        p = Path(raw_s).expanduser()
        candidates = [p] if p.is_absolute() else []
        if not p.is_absolute():
            if base is not None:
                candidates.append(base / p)
            candidates.append(root / p)
            candidates.append(Path.cwd() / p)
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        # Return the first intended location for clear missing-model diagnostics.
        return (candidates[0] if candidates else p).resolve()

    for entry in entries:
        mid = str(entry.get('id') or '').strip()
        if not mid:
            continue
        explicit = _explicit_path(entry)
        if explicit is None:
            pending_scan.append(entry)
            continue
        if explicit.is_file():
            meta = load_export_metadata_for_model(explicit)
            tier = str(entry.get('_tier') or 'primary')
            matched.append(CampaignModelMatch(
                model_id=mid,
                task=str(entry.get('task') or '').strip().lower(),
                family=str(entry.get('family') or '').strip(),
                tier=str(tier),
                model_path=explicit,
                export_metadata=dict(meta or {}),
                entry=dict(entry),
            ))
            taken.add(str(explicit))
        else:
            missing.append(f'{mid} ({explicit})')

    if not pending_scan:
        return matched, missing
    if not root.is_dir():
        raise ProfileCampaignError(f'Model root not found: {root}')

    candidates: Dict[str, List[Tuple[Path, Dict[str, Any]]]] = {}
    for onnx_path in sorted(root.rglob('*.onnx')):
        if str(onnx_path.resolve()) in taken:
            continue
        meta = load_export_metadata_for_model(onnx_path)
        tokens = {
            _norm_token(onnx_path.stem),
            _norm_token(meta.get('model_name')),
            _norm_token(meta.get('model_ref')),
        }
        for tok in list(tokens):
            if not tok:
                continue
            candidates.setdefault(tok, []).append((onnx_path, meta))

    for entry in pending_scan:
        mid = str(entry.get('id') or '').strip()
        if not mid:
            continue
        tok = _norm_token(mid)
        hits = list(candidates.get(tok) or [])
        if not hits:
            missing.append(mid)
            continue
        hits.sort(key=lambda item: (0 if _norm_token(item[0].stem) == tok else 1, len(str(item[0]))))
        chosen_path, meta = hits[0]
        tier = str(entry.get('_tier') or 'primary')
        matched.append(CampaignModelMatch(
            model_id=mid,
            task=str(entry.get('task') or '').strip().lower(),
            family=str(entry.get('family') or '').strip(),
            tier=str(tier),
            model_path=chosen_path,
            export_metadata=dict(meta or {}),
            entry=dict(entry),
        ))
        taken.add(str(chosen_path.resolve()))
    return matched, missing


def snapshot_remote_defaults(app: Any) -> Dict[str, Any]:
    def _build() -> Dict[str, Any]:
        host = app._remote_get_selected_host()
        return {
            'host': host,
            'provider': app.var_remote_provider.get() if hasattr(app, 'var_remote_provider') else 'auto',
            'warmup': app._parse_remote_int(app.var_remote_warmup.get() if hasattr(app, 'var_remote_warmup') else '10', default=10, label='Remote warmup', minimum=0),
            'iters': app._parse_remote_int(app.var_remote_iters.get() if hasattr(app, 'var_remote_iters') else '100', default=100, label='Remote runs', minimum=1),
            'repeats': app._parse_remote_int(app.var_remote_repeats.get() if hasattr(app, 'var_remote_repeats') else '1', default=1, label='Remote repeats', minimum=1),
            'throughput_frames': app._parse_remote_int(app.var_remote_throughput_frames.get() if hasattr(app, 'var_remote_throughput_frames') else '24', default=24, label='Streaming frames', minimum=0),
            'throughput_warmup_frames': app._parse_remote_int(app.var_remote_throughput_warmup_frames.get() if hasattr(app, 'var_remote_throughput_warmup_frames') else '6', default=6, label='Streaming warmup frames', minimum=0),
            'throughput_queue_depth': app._parse_remote_int(app.var_remote_throughput_queue_depth.get() if hasattr(app, 'var_remote_throughput_queue_depth') else '2', default=2, label='Streaming queue depth', minimum=1),
            'timeout_s': app._parse_remote_outer_timeout(),
            'add_args': app.var_remote_add_args.get() if hasattr(app, 'var_remote_add_args') else '',
            'remote_venv': app.var_remote_venv.get() if hasattr(app, 'var_remote_venv') else '',
            'transfer_mode': app.var_remote_transfer_mode.get() if hasattr(app, 'var_remote_transfer_mode') else 'bundle',
            'reuse_bundle': bool(app.var_remote_reuse_bundle.get()) if hasattr(app, 'var_remote_reuse_bundle') else True,
        }
    return _call_on_tk(app, _build)


def _analysis_progress_logger(log, model_id: str):
    def _progress(msg: str) -> None:
        msg = str(msg or '').strip()
        if msg:
            log(f'[{model_id}][analyse] {msg}')
    return _progress


def analyze_model_for_campaign(app: Any, model_match: CampaignModelMatch, *, log, cancel_event: threading.Event) -> Dict[str, Any]:
    _call_on_tk(app, lambda: _apply_model_loaded(app, model_match.model_path))
    params = _call_on_tk(app, lambda: app._read_params())
    if cancel_event.is_set():
        raise ProfileCampaignError('Cancelled before analysis started.')
    analysis = app._analyse_model(str(model_match.model_path), params, progress_cb=_analysis_progress_logger(log, model_match.model_id))
    if cancel_event.is_set():
        raise ProfileCampaignError('Cancelled during analysis.')
    log(f'[{model_match.model_id}][analyse] Selecting candidates…')
    picks = app._select_picks(analysis, params, progress_cb=_analysis_progress_logger(log, model_match.model_id))
    try:
        app._compute_nordstern(analysis, picks, params)
    except Exception:
        logger.debug('Failed to compute Nordstern diagnostics in profile campaign', exc_info=True)
    _call_on_tk(app, lambda: _apply_analysis_result(app, analysis, list(picks), params))
    return {
        'analysis': analysis,
        'picks': list(picks),
        'params': params,
    }


def generate_benchmark_set_for_campaign(app: Any, model_match: CampaignModelMatch, profile_request: str, benchmark_parent_dir: Path, *, log, cancel_event: threading.Event) -> Dict[str, Any]:
    done = threading.Event()
    result: Dict[str, Any] = {}

    def _callback(status: str, out_dir: str, bench_log_path: str, message: str, summary: Dict[str, Any]) -> None:
        result.update({
            'status': str(status),
            'out_dir': str(out_dir),
            'log_path': str(bench_log_path),
            'message': str(message),
            'summary': dict(summary or {}),
        })
        done.set()

    def _start() -> Optional[str]:
        try:
            if hasattr(app, 'var_bench_evaluation_profile'):
                app.var_bench_evaluation_profile.set(str(profile_request))
        except Exception:
            pass
        try:
            resolved = resolve_evaluation_profile(str(profile_request), model_path=model_match.model_path, export_metadata=model_match.export_metadata)
            if resolved is not None and resolved.matched and 'requested_cases' in resolved.overrides and hasattr(app, 'var_bench_topk'):
                app.var_bench_topk.set(str(int(resolved.overrides.get('requested_cases') or 1)))
        except Exception:
            logger.debug('Failed to refresh UI profile state before benchmark generation', exc_info=True)
        return app._generate_benchmark_set(
            offer_latest_resume=False,
            output_parent_override=str(benchmark_parent_dir),
            completion_callback=_callback,
            show_result_dialogs=False,
        )

    child_job_id = _call_on_tk(app, _start)
    log(f'[{model_match.model_id}] benchmark generation queued' if child_job_id else f'[{model_match.model_id}] benchmark generation started')
    while not done.wait(0.25):
        if cancel_event.is_set():
            if child_job_id:
                try:
                    _call_on_tk(app, lambda: app._jobs_request_cancel(child_job_id))
                except Exception:
                    logger.debug('Failed to cancel benchmark generation child job %s', child_job_id, exc_info=True)
            raise ProfileCampaignError('Cancelled during benchmark-set generation.')
    result['job_id'] = child_job_id
    return result


def build_remote_args_from_defaults(defaults: Mapping[str, Any], resolved: EvaluationProfileResolution) -> RemoteBenchmarkArgs:
    ov = dict(resolved.overrides or {})
    return RemoteBenchmarkArgs(
        provider=str(defaults.get('provider') or 'auto'),
        warmup=int(defaults.get('warmup') or 10),
        iters=int(defaults.get('iters') or 100),
        repeats=int(defaults.get('repeats') or 1),
        timeout_s=defaults.get('timeout_s'),
        throughput_frames=int(defaults.get('throughput_frames') or 24),
        throughput_warmup_frames=int(defaults.get('throughput_warmup_frames') or 6),
        throughput_queue_depth=max(1, int(defaults.get('throughput_queue_depth') or 2)),
        validation_images=str(ov.get('validation_images') or ''),
        validation_max_images=int(ov.get('validation_max_images') or 50),
        validation_reference_mode=str(ov.get('validation_reference_mode') or 'auto'),
        mini_coco_ap50=bool(ov.get('mini_coco_ap50')),
        benchmark_task=str(ov.get('benchmark_task') or 'auto'),
        mini_classification_eval=bool(ov.get('mini_classification_eval')),
        add_args=str(defaults.get('add_args') or ''),
        remote_venv=str(defaults.get('remote_venv') or ''),
        transfer_mode=str(defaults.get('transfer_mode') or 'bundle'),
        reuse_bundle=bool(defaults.get('reuse_bundle', True)),
    )


def run_remote_for_campaign(app: Any, bench_json_path: Path, suite_name: str, defaults: Mapping[str, Any], resolved: EvaluationProfileResolution, *, log, cancel_event: threading.Event) -> Dict[str, Any]:
    host = defaults.get('host')
    if host is None:
        raise ProfileCampaignError('Auto remote run requested, but no remote host is selected.')
    args = build_remote_args_from_defaults(defaults, resolved)
    service = getattr(app, '_remote_service', None)
    if service is None:
        raise ProfileCampaignError('Remote benchmark service unavailable.')
    local_working_dir = Path(getattr(app, 'default_output_dir', '.')).expanduser()
    log(f'[{suite_name}][remote] starting on {host.user_host}')
    out = service.run(
        host=host,
        benchmark_set_json=bench_json_path,
        local_working_dir=local_working_dir,
        run_id=time.strftime('%Y%m%d_%H%M%S'),
        args=args,
        log=lambda s: log(f'[{suite_name}][remote] {s}'),
        progress=lambda p, msg: log(f'[{suite_name}][remote] {int(round(float(p) * 100.0))}% {msg}'),
        cancel_event=cancel_event,
    )
    return dict(out or {})


def auto_export_benchmark_analysis(local_run_dir: Path, *, log) -> Optional[Path]:
    local_run_dir = Path(local_run_dir)
    source = local_run_dir / 'results_bundle_lean.tar.gz'
    if not source.exists():
        source = local_run_dir / 'results'
    if not source.exists():
        log(f'[analysis] skipped: no results source found under {local_run_dir}')
        return None
    output_dir = local_run_dir / 'analysis_auto'
    cache_base = local_run_dir / '.benchmark_analysis_cache'
    service = BenchmarkAnalysisService(cache_base=cache_base)
    loaded = service.load_single(source)
    service.export_single(loaded, output_dir)
    log(f'[analysis] exported benchmark analysis to {output_dir}')
    return output_dir


def run_profile_campaign(app: Any, options: ProfileCampaignOptions, *, job_id: str, log, progress, cancel_event: threading.Event) -> Dict[str, Any]:
    src = resolve_evaluation_profile_source(options.profile_request)
    if src is None:
        raise ProfileCampaignError(f'Profile not found: {options.profile_request}')
    loaded = load_evaluation_profile(src, validate=True)
    if loaded is None:
        raise ProfileCampaignError(f'Profile not found: {options.profile_request}')
    raw_profile = dict(loaded.raw_profile or {})
    matched_models, missing = discover_profile_models(raw_profile, options.models_root, include_reserve=options.include_reserve, profile_dir=Path(loaded.profile_path).parent)
    if not matched_models:
        raise ProfileCampaignError('No matching ONNX models for the selected profile were found in the configured model root.')
    workflow_root = Path(options.evaluation_run_root or (Path(options.benchmark_parent_dir).expanduser().resolve().parent / 'EvaluationRuns')).expanduser().resolve()
    run_ctx = create_evaluation_run_context(
        output_root=workflow_root,
        profile_id=loaded.profile_id,
        profile_path=loaded.profile_path,
        profile_payload=raw_profile,
        run_id=options.workflow_run_id,
        resume=bool(options.resume_workflow),
    )
    run_ctx.set_status('running', message='GUI Evaluation Workflow started')
    log(f'Workflow run directory: {run_ctx.run_dir}')
    run_ctx.stage('resolve_profile', inputs={'profile_request': options.profile_request}).finish(
        'success',
        outputs={
            'profile_id': loaded.profile_id,
            'profile_path': loaded.profile_path,
            'workflow_run_dir': str(run_ctx.run_dir),
        },
    )
    if missing:
        log('Missing models in profile campaign: ' + ', '.join(missing))
        run_ctx.stage('resolve_model', metadata={'scope': 'missing_models'}).finish(
            'failed',
            message='Some profile models could not be resolved from the selected model root.',
            outputs={'missing_models': list(missing), 'models_root': str(options.models_root)},
        )
    total = len(matched_models)
    prep_mode = normalize_model_preparation_mode(options.model_preparation_mode)
    prep_root = Path(options.preparation_output_root or (run_ctx.run_dir / '_prepared_models')).expanduser().resolve()
    prep_runtime = options.preparation_runtime or PreparationRuntimeOptions()
    progress(0.0, f'Queue prepared ({total} model(s))')
    results: List[Dict[str, Any]] = []

    for idx, model_match in enumerate(matched_models, start=1):
        if cancel_event.is_set():
            raise ProfileCampaignError('Campaign cancelled.')
        progress((idx - 1) / max(1, total), f'{idx-1}/{total} finished')
        log(f'=== [{idx}/{total}] {model_match.model_id} ({model_match.task}) ===')
        active_match = model_match
        workflow_model_dir = run_ctx.register_model(
            model_match.model_id,
            model_path=model_match.model_path,
            task=model_match.task,
            family=model_match.family,
            tier=model_match.tier,
            export_metadata=model_match.export_metadata,
            entry=model_match.entry,
            include_sha256=True,
        )
        run_ctx.stage('resolve_model', model_id=model_match.model_id, inputs={'models_root': str(options.models_root)}).finish(
            'success',
            outputs={
                'model_path': str(model_match.model_path),
                'workflow_model_dir': artifact_relpath(run_ctx.run_dir, workflow_model_dir),
            },
        )
        run_ctx.stage('check_validation_assets', model_id=model_match.model_id).finish(
            'success',
            outputs={
                'validation_preset': str(model_match.entry.get('development_subset') or model_match.entry.get('semantic_dataset') or ''),
                'task': model_match.task,
            },
        )
        model_result: Dict[str, Any] = {
            'model_id': model_match.model_id,
            'task': model_match.task,
            'preparation_mode': prep_mode,
            'workflow_model_dir': str(workflow_model_dir),
        }
        if prep_mode != 'current':
            prep_stage = run_ctx.stage('prepare_model', model_id=model_match.model_id, inputs={'mode': prep_mode, 'model_path': str(model_match.model_path)})
            progress((idx - 1) / max(1, total), f'{idx-1}/{total} finished · preparing {model_match.model_id}')
            log(f'[{model_match.model_id}][prepare] mode={prep_mode}')
            prep_out = prepare_model_for_benchmark(
                model_match.model_path,
                export_metadata=model_match.export_metadata,
                requested_mode=prep_mode,
                entry=model_match.entry,
                output_root=prep_root,
                runtime=prep_runtime,
                log=lambda s: log(f'[{model_match.model_id}] {s}'),
                cancel_event=cancel_event,
            )
            model_result.update({
                'preparation_success': bool(prep_out.success),
                'preparation_message': str(prep_out.message or ''),
                'preparation_screening_dir': str(prep_out.screening_dir or ''),
                'prepared_model_path': str(prep_out.selected_model_path or ''),
                'preparation_variants_tested': int(len(prep_out.variants or [])),
                'preparation_tier2_success': bool(getattr(prep_out, 'tier2_success', False)),
                'preparation_tier2_variant_id': str(getattr(prep_out, 'tier2_variant_id', None) or ''),
            })
            if not prep_out.success:
                log(f'[{model_match.model_id}][prepare] failed: {prep_out.message}')
                prep_stage.finish(
                    'failed',
                    message=str(prep_out.message or ''),
                    outputs={
                        'screening_dir': str(prep_out.screening_dir or ''),
                        'variants_tested': int(len(prep_out.variants or [])),
                    },
                )
                results.append(model_result)
                progress(idx / max(1, total), f'{idx}/{total} finished')
                continue
            active_match = CampaignModelMatch(
                model_id=model_match.model_id,
                task=model_match.task,
                family=model_match.family,
                tier=model_match.tier,
                model_path=Path(prep_out.selected_model_path).expanduser().resolve(),
                export_metadata=dict(prep_out.selected_export_metadata or {}),
                entry=dict(model_match.entry),
            )
            log(f'[{model_match.model_id}][prepare] selected {active_match.model_path.name}')
            prep_stage.finish(
                'success',
                message=str(prep_out.message or ''),
                outputs={
                    'selected_model_path': str(active_match.model_path),
                    'screening_dir': str(prep_out.screening_dir or ''),
                    'tier2_success': bool(getattr(prep_out, 'tier2_success', False)),
                    'tier2_variant_id': str(getattr(prep_out, 'tier2_variant_id', None) or ''),
                },
            )
        else:
            run_ctx.stage('prepare_model', model_id=model_match.model_id, inputs={'mode': prep_mode}).finish(
                'skipped',
                message='Model preparation mode is current; original ONNX is used.',
                outputs={'model_path': str(active_match.model_path)},
            )

        progress((idx - 1) / max(1, total), f'{idx-1}/{total} finished · analysing {model_match.model_id}')
        analysis_stage = run_ctx.stage('analyze_model', model_id=model_match.model_id, inputs={'model_path': str(active_match.model_path)})
        analysis_out = analyze_model_for_campaign(app, active_match, log=log, cancel_event=cancel_event)
        analysis_paths = run_ctx.write_analysis_artifacts(
            model_match.model_id,
            analysis=dict(analysis_out.get('analysis') or {}),
            picks=list(analysis_out.get('picks') or []),
            params=analysis_out.get('params'),
            prediction_metadata={'profile_request': options.profile_request, 'active_model_path': str(active_match.model_path)},
        )
        analysis_stage.finish('success', outputs=analysis_paths)
        run_ctx.stage('select_split_candidates', model_id=model_match.model_id, inputs={'analysis_artifacts': analysis_paths}).finish(
            'success',
            outputs={
                'prediction_json': analysis_paths.get('prediction_json', ''),
                'candidate_ranking_json': analysis_paths.get('candidate_ranking_json', ''),
                'candidate_count': int(len(analysis_out.get('picks') or [])),
            },
        )
        log(f'[{model_match.model_id}] analysis done')
        progress((idx - 1) / max(1, total), f'{idx-1}/{total} finished · generating {model_match.model_id}')
        run_ctx.stage('prepare_full_baselines', model_id=model_match.model_id).finish(
            'planned',
            message='Full-baseline HEFs/outputs are prepared inside benchmark-set generation in v49; the manifest records the suite output once generated.',
        )
        model_suite_parent = workflow_model_dir / 'benchmark_set'
        gen_stage = run_ctx.stage(
            'generate_benchmark_set',
            model_id=model_match.model_id,
            inputs={
                'prediction_json': analysis_paths.get('prediction_json', ''),
                'suite_parent': str(model_suite_parent),
            },
        )
        gen_out = generate_benchmark_set_for_campaign(app, active_match, options.profile_request, model_suite_parent, log=log, cancel_event=cancel_event)
        suite_dir = Path(gen_out.get('out_dir') or '')
        bench_json_path = suite_dir / 'benchmark_set.json'
        final_plan_paths = run_ctx.write_final_candidate_plan(model_match.model_id, suite_dir=suite_dir, generation_result=gen_out)
        gen_stage.finish(
            'success' if gen_out.get('status') in {'ok', 'warn'} else 'failed',
            message=str(gen_out.get('message') or ''),
            outputs={
                'suite_dir': str(suite_dir),
                'benchmark_set_json': str(bench_json_path),
                **final_plan_paths,
            },
        )
        run_ctx.stage('build_backend_artifacts', model_id=model_match.model_id).finish(
            'success' if gen_out.get('status') in {'ok', 'warn'} else 'failed',
            message='Backend build artifacts are produced by the benchmark-set generator; see suite manifest/logs.',
            outputs={'suite_dir': str(suite_dir)},
        )
        model_result.update({
            'suite_dir': str(suite_dir),
            'generation_status': str(gen_out.get('status') or ''),
            'generation_message': str(gen_out.get('message') or ''),
        })
        if gen_out.get('status') not in {'ok', 'warn'}:
            log(f'[{model_match.model_id}] generation failed: {gen_out.get("message") or gen_out.get("status")}')
            results.append(model_result)
            continue
        log(f'[{model_match.model_id}] benchmark set ready: {suite_dir}')
        if options.auto_remote_run:
            progress((idx - 1) / max(1, total), f'{idx-1}/{total} finished · remote {model_match.model_id}')
            resolved = resolve_evaluation_profile(options.profile_request, model_path=active_match.model_path, export_metadata=active_match.export_metadata)
            if resolved is None:
                raise ProfileCampaignError(f'Failed to resolve profile overrides for {model_match.model_id}')
            remote_out = run_remote_for_campaign(app, bench_json_path, model_match.model_id, options.remote_defaults or {}, resolved, log=log, cancel_event=cancel_event)
            model_result['remote_status'] = str(remote_out.get('status') or '')
            model_result['remote_local_run_dir'] = str(remote_out.get('local_run_dir') or '')
            model_result['remote_error'] = str(remote_out.get('error') or '')
            if options.auto_export_analysis and str(remote_out.get('local_run_dir') or '').strip():
                try:
                    analysis_dir = auto_export_benchmark_analysis(Path(str(remote_out.get('local_run_dir'))), log=log)
                    if analysis_dir is not None:
                        model_result['analysis_dir'] = str(analysis_dir)
                except Exception as exc:
                    log(f'[{model_match.model_id}][analysis] failed: {type(exc).__name__}: {exc}')
            run_ctx.stage('run_benchmarks', model_id=model_match.model_id).finish(
                'success' if str(model_result.get('remote_status') or '').lower() in {'ok', 'partial', 'success'} else 'failed',
                outputs={
                    'remote_status': str(model_result.get('remote_status') or ''),
                    'remote_local_run_dir': str(model_result.get('remote_local_run_dir') or ''),
                },
            )
            run_ctx.stage('validate_outputs', model_id=model_match.model_id).finish(
                'success' if str(model_result.get('analysis_dir') or '').strip() else 'planned',
                message='Validation/report export is available after remote benchmark results are collected.',
                outputs={'analysis_dir': str(model_result.get('analysis_dir') or '')},
            )
        else:
            run_ctx.stage('run_benchmarks', model_id=model_match.model_id).finish(
                'planned',
                message='Auto remote benchmark execution was disabled for this workflow run.',
            )
            run_ctx.stage('validate_outputs', model_id=model_match.model_id).finish(
                'planned',
                message='Validation waits for benchmark execution/results.',
            )
        results.append(model_result)
        progress(idx / max(1, total), f'{idx}/{total} finished')
    progress(1.0, 'Campaign finished')
    run_ctx.stage('aggregate_results').finish('success', outputs={'models_processed': len(results), 'missing_models': list(missing)})
    reports = run_ctx.write_summary_reports(model_results=results, missing_models=missing)
    run_ctx.stage('generate_report').finish('success', outputs=reports)
    run_ctx.set_status('success', message=f'Processed {len(results)} model(s).')
    return {
        'profile': str(raw_profile.get('name') or src.stem),
        'models_processed': len(results),
        'results': results,
        'missing_models': missing,
        'workflow_run_dir': str(run_ctx.run_dir),
        'run_manifest': str(run_ctx.manifest_path),
        'reports': reports,
    }
