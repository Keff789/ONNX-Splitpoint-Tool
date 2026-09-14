#!/usr/bin/env python3
"""Run native FIFO smokes over a staged EvaluationRun benchmark-set tree.

Expected tree (created by copy_eval_benchmarksets_for_native_fifo.py):
  <root>/<model>/benchmark_set/{benchmark_set.json,bXXX/...}

This script is intentionally a small orchestration layer around
native_fifo_smoke_matrix.py so it works on the Hailo8 NX without the full GUI.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cache_verify_only() -> bool:
    return str(
        os.environ.get('ONNX_SPLITPOINT_ARTIFACT_POLICY') or ''
    ).strip().lower() == 'cache_verify_only'


def _parse_models(s: str | None, root: Path) -> list[str]:
    if s:
        return [x.strip() for x in s.split(',') if x.strip()]
    models = []
    for p in sorted(root.iterdir() if root.exists() else []):
        if (p / 'benchmark_set').exists():
            models.append(p.name)
    return models


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(
            path.read_text(encoding='utf-8'),
            object_pairs_hook=_no_duplicate_json_keys,
        )
    except Exception:
        return {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _no_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate_json_key:{key}')
        value[key] = item
    return value


def _case_map(s: str | None) -> dict[str, list[str]]:
    if not s:
        return {}
    data = json.loads(s, object_pairs_hook=_no_duplicate_json_keys)
    if not isinstance(data, dict):
        raise ValueError('case_map_must_be_object')
    out: dict[str, list[str]] = {}
    for model, cases in data.items():
        if isinstance(cases, str):
            cases = [cases]
        if not isinstance(cases, list):
            raise ValueError(f'case_map_cases_must_be_list:{model}')
        model_id = str(model).strip()
        selected = [str(c).strip() for c in cases if str(c).strip()]
        if not model_id or not selected:
            raise ValueError(f'case_map_model_or_cases_empty:{model}')
        if len(selected) != len(set(selected)):
            raise ValueError(f'case_map_duplicate_cases:{model_id}')
        out[model_id] = selected
    return out


def _run(cmd: list[str], timeout: float | None = None) -> dict[str, Any]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {
            'cmd': cmd,
            'rc': p.returncode,
            'elapsed_s': time.time() - t0,
            'stdout_tail': p.stdout[-8000:],
            'stderr_tail': p.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            'cmd': cmd,
            'rc': 124,
            'elapsed_s': time.time() - t0,
            'stdout_tail': (e.stdout or '')[-8000:] if isinstance(e.stdout, str) else '',
            'stderr_tail': (e.stderr or '')[-8000:] if isinstance(e.stderr, str) else '',
            'timeout': True,
        }


def _find_bs(root: Path, model: str) -> Path | None:
    candidates = [
        root / model / 'benchmark_set',
        root / model / 'benchmark_set' / 'legacy_suite',
        root / model,
    ]
    for p in candidates:
        if (p / 'benchmark_set.json').exists() and (p / 'benchmark_suite.py').exists():
            return p.resolve()
    # fallback: find first matching complete benchmark set below model dir
    base = root / model
    if base.exists():
        for p in sorted(base.rglob('benchmark_set.json')):
            parent = p.parent
            if (parent / 'benchmark_suite.py').exists():
                return parent.resolve()
    return None


def _benchmark_task(benchmark_set: Path) -> str:
    payload = _load_json(benchmark_set / 'benchmark_set.json')
    return str(
        payload.get('benchmark_task')
        or payload.get('task')
        or payload.get('model_task')
        or ''
    ).strip().lower()


def _extract_rows(
    model: str,
    bs: Path,
    matrix: dict[str, Any],
    *,
    setup_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in matrix.get('cases') or []:
        cid = r.get('case_id')
        result_text = str(r.get('native_fifo_result') or '').strip()
        result_path = Path(result_text).expanduser() if result_text else None
        res = _load_json(result_path) if result_path is not None and result_path.is_file() else {}
        quality_first = r.get('native_split_quality_required') is True
        cache_replay = bool(quality_first and _cache_verify_only())
        child_completed_successfully = bool(
            str(r.get('status') or '') == 'ok'
            and r.get('result_ok') is True
            and int(r.get('returncode') or 0) == 0
            and not r.get('timed_out')
            and r.get('child_result_fresh') is True
            and res.get('ok') is True
        )
        result_sha256 = (
            _sha256_file(result_path)
            if result_path is not None and result_path.is_file() else ''
        )
        result_size_bytes = (
            int(result_path.stat().st_size)
            if result_path is not None and result_path.is_file() else 0
        )
        strict_mirrors: dict[str, Any] = {}
        mirror_errors: list[str] = []
        mirror_diagnostics: list[str] = []
        mirror_fields = [
            'eval_run_id', 'source_run_id',
            'native_split_quality_eval_run_id',
            'native_split_quality_source_run_id',
        ]
        mirror_fields.extend(
            [
                'native_split_quality_binding_sha256',
                'native_split_quality_cache_verify_source_binding_sha256',
                'native_split_quality_cache_verify_replay_sha256',
            ]
            if cache_replay else [
                'source_request_sha256',
                'native_split_quality_source_request_sha256',
                'native_split_quality_central_result_sha256',
                'native_split_quality_selection_sha256',
            ]
        )
        for field in mirror_fields:
            left = res.get(field)
            right = r.get(field)
            if quality_first and child_completed_successfully and (
                left in (None, '') or right in (None, '')
                or str(left).strip().lower() != str(right).strip().lower()
            ):
                target = (
                    mirror_diagnostics
                    if cache_replay and field in {
                        'native_split_quality_binding_sha256',
                        'native_split_quality_cache_verify_source_binding_sha256',
                        'native_split_quality_cache_verify_replay_sha256',
                    }
                    else mirror_errors
                )
                target.append(field)
            strict_mirrors[field] = left if left not in (None, '') else right
        for field in (
            'native_split_quality_binding',
            'native_split_quality_consumer_attestation',
        ):
            left = res.get(field)
            right = r.get(field)
            if quality_first and child_completed_successfully and (
                not isinstance(left, dict) or not isinstance(right, dict)
                or left != right
            ):
                (
                    mirror_diagnostics if cache_replay else mirror_errors
                ).append(field)
            strict_mirrors[field] = left if isinstance(left, dict) else right
        if quality_first and child_completed_successfully and (
            not result_sha256
            or str(r.get('native_fifo_result_sha256') or '').strip().lower()
            != result_sha256
            or int(r.get('native_fifo_result_size_bytes') or 0)
            != result_size_bytes
        ):
            (
                mirror_diagnostics if cache_replay else mirror_errors
            ).append('native_fifo_result_sha256')
        existing_diagnostics = r.get('cache_verify_diagnostics')
        if not isinstance(existing_diagnostics, list):
            existing_diagnostics = (
                [str(existing_diagnostics)] if existing_diagnostics else []
            )
        row = {
            'model': model,
            'benchmark_set': str(bs),
            'case_id': cid,
            'status': r.get('status'),
            'result_ok': bool(r.get('result_ok') and res.get('ok'))
            if quality_first else bool(r.get('result_ok') or res.get('ok')),
            'fps_makespan': r.get('fps_makespan') if r.get('fps_makespan') is not None else res.get('fps_makespan'),
            'fps_median': res.get('fps_median', res.get('fps_makespan_median')),
            'fps_ci95_low': res.get('fps_ci95_low', res.get('fps_makespan_ci95_low')),
            'fps_ci95_high': res.get('fps_ci95_high', res.get('fps_makespan_ci95_high')),
            'repetition_count_requested': res.get('repetition_count_requested', res.get('repetitions_requested')),
            'repetition_count_attempted': res.get('repetition_count_attempted', res.get('repetitions_completed')),
            'repetition_count_valid': res.get('repetition_count_valid', res.get('repetitions_completed')),
            'repetition_status': res.get('repetition_status'),
            'repetition_aggregation': res.get('repetition_aggregation'),
            'repetition_records': res.get('repetition_records', res.get('repetition_evidence', [])),
            'repetition_evidence': res.get('repetition_evidence', res.get('repetition_records', [])),
            'paper_fps': r.get('paper_fps') if r.get('paper_fps') is not None else res.get('paper_equivalent_fps'),
            'handoff_ms': r.get('handoff_ms') if r.get('handoff_ms') is not None else res.get('handoff_ms'),
            'preprocess_ms': res.get('preprocess_ms'),
            'p1_ms': res.get('p1_ms'),
            'p2_run_ms': res.get('p2_run_ms'),
            'p1_thread_ms': res.get('p1_thread_ms'),
            'p2_thread_ms': res.get('p2_thread_ms'),
            'single_latency_model_ms': res.get('single_latency_model_ms'),
            'trt_input_bytes': res.get('trt_input_bytes'),
            'trt_input_dtype': res.get('trt_input_dtype'),
            'output_validation_ok': r.get('output_validation_ok'),
            'native_fifo_result': str(result_path) if result_path is not None else '',
            'native_fifo_result_sha256': result_sha256,
            'native_fifo_result_size_bytes': result_size_bytes,
            'child_result_fresh': r.get('child_result_fresh') is True,
            'native_fifo_boundary_manifest': r.get('native_fifo_boundary_manifest') or res.get('native_fifo_boundary_manifest') or '',
            'native_fifo_output_manifest': r.get('native_fifo_output_manifest') or res.get('native_fifo_output_manifest') or '',
            'input_image': res.get('input_image') or r.get('image') or '',
            'input_image_source': res.get('input_image_source') or ('exact_file' if (res.get('input_image') or r.get('image')) else ''),
            'input_image_sha256': res.get('input_image_sha256') or '',
            'native_command_contract': res.get('native_command_contract') or {},
            'native_command_contract_sha256': (
                (res.get('native_command_contract') or {}).get('contract_sha256')
                if isinstance(res.get('native_command_contract'), dict) else ''
            ),
            'workload_contract_sha256': res.get('workload_contract_sha256') or r.get('workload_contract_sha256') or '',
            'setup_id': res.get('setup_id') or r.get('setup_id') or str(setup_id or ''),
            'comparison_backend': 'hailo8',
            'eval_run_id': strict_mirrors.get('eval_run_id') or '',
            'source_run_id': strict_mirrors.get('source_run_id') or '',
            'native_split_quality_binding': strict_mirrors.get('native_split_quality_binding') or {},
            'native_split_quality_binding_sha256': strict_mirrors.get('native_split_quality_binding_sha256') or '',
            'native_split_quality_eval_run_id': strict_mirrors.get('native_split_quality_eval_run_id') or '',
            'native_split_quality_source_run_id': strict_mirrors.get('native_split_quality_source_run_id') or '',
            'source_request_sha256': strict_mirrors.get('source_request_sha256') or '',
            'native_split_quality_source_request_sha256': strict_mirrors.get('native_split_quality_source_request_sha256') or '',
            'native_split_quality_central_result_sha256': strict_mirrors.get('native_split_quality_central_result_sha256') or '',
            'native_split_quality_selection_sha256': strict_mirrors.get('native_split_quality_selection_sha256') or '',
            'native_split_quality_cache_verify_source_binding_sha256': strict_mirrors.get('native_split_quality_cache_verify_source_binding_sha256') or '',
            'native_split_quality_cache_verify_replay_sha256': strict_mirrors.get('native_split_quality_cache_verify_replay_sha256') or '',
            'native_split_quality_consumer_attestation': strict_mirrors.get('native_split_quality_consumer_attestation') or {},
            'native_split_quality_consumer_status': res.get('native_split_quality_consumer_status') or r.get('native_split_quality_consumer_status') or '',
            'native_split_quality_required': r.get('native_split_quality_required') is True,
            'performance_claims_emitted': r.get('performance_claims_emitted') is True,
            'execution_role': str(r.get('execution_role') or ''),
            'boundary_dump_warning': r.get('boundary_dump_warning') or '',
            'reason': r.get('reason') or r.get('failure_reason') or r.get('recommended_action') or r.get('boundary_dump_warning') or '',
            'failure_reason': r.get('failure_reason') or res.get('failure_reason') or '',
            'status_detail': r.get('status_detail') or res.get('status_detail') or '',
            'error': r.get('error') or res.get('error') or '',
            'returncode': r.get('returncode'),
            'timed_out': bool(r.get('timed_out')),
            'stdout_tail': r.get('stdout_tail') or '',
            'stderr_tail': r.get('stderr_tail') or '',
            'unsupported_reason': r.get('unsupported_reason') or '',
            'recommended_action': r.get('recommended_action') or '',
            'steps': r.get('steps') or [],
            'capability_before': r.get('capability_before') or {},
            'capability_after': r.get('capability_after') or {},
            'cache_verify_diagnostics': sorted(set(
                existing_diagnostics
                + [
                    'cache_verify_diagnostic_wrapper_mirror_mismatch:'
                    + field
                    for field in mirror_diagnostics
                ]
            )),
        }
        if child_completed_successfully and mirror_errors:
            reason = (
                'native_split_quality_wrapper_mirror_mismatch:'
                + ','.join(mirror_errors)
            )
            row.update({
                'result_ok': False,
                'status': 'failed',
                'failure_reason': reason,
                'status_detail': reason,
                'error': reason,
                'performance_claims_emitted': False,
            })
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description='Run native FIFO over staged EvalRun benchmark sets')
    ap.add_argument('--root', required=True, help='Root containing <model>/benchmark_set directories')
    ap.add_argument('--models', default='', help='Comma-separated model ids. Default: all subdirs with benchmark_set')
    ap.add_argument('--case-map', default='', help='JSON map model -> list of case ids')
    ap.add_argument('--hw-arch', default='hailo8')
    ap.add_argument('--precision', default='uint8_cast_fp16')
    ap.add_argument('--build-missing-engines', action='store_true')
    ap.add_argument('--engine-build-python', default='auto', help='Python used only for native TensorRT Part2 engine construction.')
    ap.add_argument('--mixed-runtime-python', default=os.environ.get('ONNX_SPLITPOINT_HAILO8_TRT_PYTHON', '/usr/bin/python3'), help='Existing system Python used only by Hailo-8 detection completion.')
    ap.add_argument('--force-rebuild-engines', action='store_true', help='Forward engine rebuild request to native_fifo_smoke_matrix.py.')
    ap.add_argument('--dequant-scale', type=float, default=0.0, help='Forward explicit uint8_dequant_fp16 scale.')
    ap.add_argument('--dequant-zero-point', type=float, default=0.0, help='Forward explicit uint8_dequant_fp16 zero point.')
    ap.add_argument('--boundary-layout', default='as_input', help='Forward raw Hailo boundary memory layout, e.g. memory_nhwc_to_nchw.')
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--repetitions', type=int, default=1, help='Independent performance repetitions; aggregate is median plus 95%% CI, never best-of.')
    ap.add_argument('--queue-depth', type=int, default=3)
    ap.add_argument('--hailo-format', default='uint8')
    ap.add_argument('--preprocess-mode', default='auto', choices=['auto', 'resize', 'letterbox'], help='Forward task-specific preprocessing; auto means classification=resize and detection=letterbox.')
    ap.add_argument('--dump-outputs', action='store_true')
    ap.add_argument('--dump-boundary', action='store_true', help='Forward raw boundary dump flag to Hailo8 native FIFO runner for diagnostics.')
    ap.add_argument('--letterbox-pad-value', type=int, default=0, help='Forward native Hailo input letterbox pad value. Use 114 to match YOLO generic harness semantics.')
    ap.add_argument('--image-map', default='', help='JSON map model->case->image or case->image used for exact native validation dumps.')
    ap.add_argument('--validate-dumps', action='store_true', default=True)
    ap.add_argument('--no-validate-dumps', dest='validate_dumps', action='store_false')
    ap.add_argument('--skip-existing-results', action='store_true')
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--analysis-tag', default='', help='Optional stable tag used to keep per-contract analysis summaries instead of overwriting them.')
    ap.add_argument('--setup-id', default=os.environ.get('ONNX_SPLITPOINT_SETUP_ID', ''))
    ap.add_argument('--native-split-quality-binding-set', default=os.environ.get('ONNX_SPLITPOINT_NATIVE_SPLIT_BINDING_SET', ''))
    args = ap.parse_args()
    if args.force_rebuild_engines:
        ap.error('productive_force_build_disabled: --force-rebuild-engines is disabled; reuse compatible engines and build missing engines normally')
    if int(args.repetitions) < 1:
        ap.error('--repetitions must be >= 1')

    root = Path(args.root).expanduser().resolve()
    models = _parse_models(args.models, root)
    case_map_supplied = bool(str(args.case_map or '').strip())
    try:
        cmap = _case_map(args.case_map)
    except Exception as exc:
        print(json.dumps({
            'ok': False,
            'status': 'selection_invalid',
            'failure_reason': 'case_map_invalid',
            'error': f'{type(exc).__name__}: {exc}',
            'rows': 0,
        }, indent=2))
        return 3
    out_dir = root / 'analysis_tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    model_runs: list[dict[str, Any]] = []

    case_map_active = case_map_supplied
    quality_first_required = bool(
        str(args.native_split_quality_binding_set or '').strip()
    )

    selection_errors: list[str] = []
    if not models:
        selection_errors.append('requested_or_discovered_models_empty')
    if case_map_active:
        unknown_models = sorted(set(cmap).difference(models))
        missing_models = sorted(set(models).difference(cmap))
        if unknown_models:
            selection_errors.append(
                'case_map_unknown_models:' + ','.join(unknown_models)
            )
        if missing_models:
            selection_errors.append(
                'case_map_missing_requested_models:' + ','.join(missing_models)
            )
    for model in models:
        bs = _find_bs(root, model)
        if bs is None:
            selection_errors.append(f'benchmark_set_not_found:{model}')
            continue
        if case_map_active:
            available = {
                path.name for path in bs.iterdir()
                if path.is_dir() and path.name.startswith('b')
            }
            missing_cases = [case for case in cmap.get(model, []) if case not in available]
            if missing_cases:
                selection_errors.append(
                    f'case_map_cases_not_found:{model}:' + ','.join(missing_cases)
                )
    if selection_errors:
        print(json.dumps({
            'ok': False,
            'status': 'selection_invalid',
            'failure_reason': 'requested_benchmark_set_or_case_unavailable',
            'selection_errors': selection_errors,
            'models': models,
            'case_map': cmap,
            'rows': 0,
        }, indent=2))
        return 3

    for model in models:
        bs = _find_bs(root, model)
        mrec: dict[str, Any] = {'model': model, 'benchmark_set': str(bs or ''), 'ok': False, 'steps': []}
        if not bs:
            mrec['error'] = 'benchmark_set_not_found'
            model_runs.append(mrec)
            continue
        task = _benchmark_task(bs)
        if task not in {'classification', 'detection'}:
            mrec['error'] = 'benchmark_task_missing_or_invalid'
            mrec['failure_reason'] = 'benchmark_task_missing_or_invalid'
            model_runs.append(mrec)
            continue
        mrec['task'] = task
        cmd = [sys.executable, str(ROOT / 'scripts' / 'native_fifo_smoke_matrix.py'),
               '--benchmark-set', str(bs),
               '--model-id', str(model),
               '--setup-id', str(args.setup_id),
               '--hw-arch', args.hw_arch,
               '--precision', args.precision,
               '--frames', str(args.frames),
               '--warmup', str(args.warmup),
               '--repetitions', str(args.repetitions),
               '--queue-depth', str(args.queue_depth),
               '--hailo-format', args.hailo_format,
               '--task', task,
               '--mixed-runtime-python', str(args.mixed_runtime_python),
               '--preprocess-mode', str(args.preprocess_mode),
               '--letterbox-pad-value', str(int(args.letterbox_pad_value)),
               '--timeout', str(args.timeout)]
        if quality_first_required:
            cmd += [
                '--native-split-quality-binding-set',
                str(args.native_split_quality_binding_set),
            ]
        if args.build_missing_engines and not quality_first_required:
            cmd.append('--build-missing-engines')
        if args.engine_build_python:
            cmd += ['--engine-build-python', str(args.engine_build_python)]
        if args.force_rebuild_engines and not quality_first_required:
            cmd.append('--force-rebuild-engines')
        if str(args.precision) == 'uint8_dequant_fp16':
            if float(args.dequant_scale or 0.0) > 0.0:
                cmd += ['--dequant-scale', str(float(args.dequant_scale))]
            cmd += ['--dequant-zero-point', str(float(args.dequant_zero_point or 0.0))]
        if str(args.precision) in {'uint8_dequant_fp16', 'float32_layout_fp16'}:
            if str(args.boundary_layout or 'as_input') != 'as_input':
                cmd += ['--boundary-layout', str(args.boundary_layout)]
        if args.dump_outputs:
            cmd.append('--dump-outputs')
        if args.dump_boundary:
            cmd.append('--dump-boundary')
        if args.image_map:
            cmd += ['--image-map', args.image_map]
        if args.validate_dumps:
            cmd.append('--validate-dumps')
        else:
            cmd.append('--no-validate-dumps')
        if args.skip_existing_results:
            cmd.append('--skip-existing-results')
        selected_cases = cmap.get(model, []) if case_map_active else []
        for cid in selected_cases:
            cmd += ['--case', str(cid)]
        print(f"[native-fifo-eval] {model}: {' '.join(cmd)}", flush=True)
        step = _run(cmd, timeout=max(args.timeout * max(1, len(selected_cases or [1])), args.timeout + 60))
        mrec['steps'].append({'name': 'native_fifo_smoke_matrix', **step})
        matrix_path = bs / 'analysis_tables' / 'native_fifo_smoke_matrix.json'
        matrix = _load_json(matrix_path)
        mrec['orchestration_ok'] = step.get('rc') == 0 and bool(matrix.get('ok', True))
        mrec['ok'] = bool(mrec['orchestration_ok'])
        mrec['matrix'] = str(matrix_path)
        rows = _extract_rows(
            model,
            bs,
            matrix,
            setup_id=str(args.setup_id or ''),
        )
        mrec['row_count'] = len(rows)
        mrec['ok_count'] = sum(1 for r in rows if r.get('result_ok'))
        mrec['failed_count'] = sum(1 for r in rows if not r.get('result_ok'))
        mrec['evidence_status'] = 'complete' if rows and mrec['ok_count'] == len(rows) else ('partial' if mrec['ok_count'] else 'unavailable')
        mrec['native_split_quality_required'] = quality_first_required
        mrec['performance_claims_emitted'] = bool(
            quality_first_required and not _cache_verify_only()
        )
        mrec['execution_role'] = (
            'cache_verify_diagnostic_replay'
            if quality_first_required and _cache_verify_only()
            else 'quality_first_native_split'
            if quality_first_required else 'legacy_manual_diagnostic'
        )
        mrec['failure_reasons'] = sorted({str(r.get('failure_reason') or r.get('reason') or '') for r in rows if not r.get('result_ok') and str(r.get('failure_reason') or r.get('reason') or '')})
        all_rows.extend(rows)
        model_runs.append(mrec)

    ok_count = sum(1 for r in all_rows if r.get('result_ok'))
    failed_count = sum(1 for r in all_rows if not r.get('result_ok'))
    orchestration_ok = all(bool(m.get('orchestration_ok')) for m in model_runs) if model_runs else False
    overall_ok = bool(
        orchestration_ok and all_rows and failed_count == 0
    )
    evidence_status = 'complete' if all_rows and ok_count == len(all_rows) else ('partial' if ok_count else 'unavailable')
    summary = {
        'ok': overall_ok,
        'orchestration_status': 'ok' if overall_ok else 'failed',
        'evidence_status': evidence_status,
        'root': str(root),
        'hw_arch': args.hw_arch,
        'precision': args.precision,
        'performance_repetitions': int(args.repetitions),
        'analysis_tag': str(args.analysis_tag or ''),
        'models': model_runs,
        'rows': all_rows,
        'row_count': len(all_rows),
        'ok_count': ok_count,
        'failed_count': failed_count,
        'failure_reasons': sorted({str(r.get('failure_reason') or r.get('reason') or '') for r in all_rows if not r.get('result_ok') and str(r.get('failure_reason') or r.get('reason') or '')}),
        'native_split_quality_required':quality_first_required,
        'performance_claims_emitted':bool(
            quality_first_required and not _cache_verify_only()
        ),
        'execution_role':(
            'cache_verify_diagnostic_replay'
            if quality_first_required and _cache_verify_only()
            else 'quality_first_native_split'
            if quality_first_required else 'legacy_manual_diagnostic'
        ),
    }
    tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(args.analysis_tag or '').strip()).strip('._-')
    stem = 'native_fifo_eval_runner' + (f'__{tag}' if tag else '')
    json_path = out_dir / f'{stem}.json'
    csv_path = out_dir / f'{stem}.csv'
    md_path = out_dir / f'{stem}.md'
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    fields = ['model','case_id','setup_id','comparison_backend','status','result_ok','fps_makespan','fps_median','fps_ci95_low','fps_ci95_high','repetition_count_requested','repetition_count_attempted','repetition_count_valid','repetition_status','repetition_aggregation','paper_fps','handoff_ms','preprocess_ms','p1_ms','p2_run_ms','p1_thread_ms','p2_thread_ms','single_latency_model_ms','trt_input_bytes','trt_input_dtype','output_validation_ok','native_fifo_result','native_fifo_boundary_manifest','input_image','input_image_source','input_image_sha256','native_command_contract_sha256','boundary_dump_warning','reason','failure_reason','status_detail','error','returncode','timed_out','unsupported_reason','recommended_action','stdout_tail','stderr_tail','benchmark_set']
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, '') for k in fields})
    lines = ['# Native FIFO eval runner summary', '', f'Root: `{root}`', f'Precision: `{args.precision}`', '', '| model | case | status | ok | fps | handoff ms | reason |', '|---|---|---|---:|---:|---:|---|']
    for r in all_rows:
        lines.append(f"| {r.get('model')} | {r.get('case_id')} | {r.get('status')} | {r.get('result_ok')} | {r.get('fps_makespan') or ''} | {r.get('handoff_ms') or ''} | {r.get('reason') or ''} |")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'ok': overall_ok, 'orchestration_status': summary['orchestration_status'], 'evidence_status': evidence_status, 'rows': len(all_rows), 'ok_count': summary['ok_count'], 'failed_count': failed_count, 'json': str(json_path), 'csv': str(csv_path), 'md': str(md_path), 'analysis_tag': tag}, indent=2))
    return 0 if overall_ok else 3


if __name__ == '__main__':
    raise SystemExit(main())
