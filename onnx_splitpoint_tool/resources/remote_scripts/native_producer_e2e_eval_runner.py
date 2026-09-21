#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess, sys, csv, hashlib, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_job_identity import (
    planned_native_identity, attach_identity_without_conflicts, failed_native_result,
)
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    canonical_native_split_backend,
    native_split_quality_selection_duplicates,
    validate_central_native_split_quality_selection,
    validate_native_split_quality_binding,
)

def _parse_case_map(s):
    if not s: return {}
    data=json.loads(s, object_pairs_hook=_no_duplicate_json_keys)
    if not isinstance(data,dict):
        raise ValueError('case_map_must_be_object')
    out={}
    for raw_model,raw_cases in data.items():
        model=str(raw_model).strip()
        cases=[raw_cases] if isinstance(raw_cases,str) else raw_cases
        if not isinstance(cases,list):
            raise ValueError(f'case_map_cases_must_be_list:{model}')
        selected=[str(case).strip() for case in cases if str(case).strip()]
        if not model or not selected:
            raise ValueError(f'case_map_model_or_cases_empty:{model}')
        if len(selected) != len(set(selected)):
            raise ValueError(f'case_map_duplicate_cases:{model}')
        out[model]=selected
    return out

def _parse_image_map(s):
    if not s: return {}
    try: return json.loads(s)
    except Exception: return {}

def _resolve_image(bs: Path, value: str) -> str:
    if not value: return ''
    p=Path(value).expanduser()
    if p.is_file(): return str(p.resolve())
    # Accept basenames/relative paths and find them under the staged BenchmarkSet.
    name=Path(value).name
    for cand in bs.rglob(name):
        if cand.is_file(): return str(cand.resolve())
    q=bs/value
    if q.is_file(): return str(q.resolve())
    return value

def _benchmark_task(bs: Path) -> str:
    for p in (bs / 'benchmark_set.json', bs.parent / 'benchmark_set.json'):
        data = _load_json(p)
        for key in ('task', 'benchmark_task', 'model_task'):
            value = str(data.get(key) or '').strip().lower()
            if value in {'classification', 'detection'}:
                return value
        model = data.get('model') if isinstance(data.get('model'), dict) else {}
        value = str(model.get('task') or '').strip().lower()
        if value in {'classification', 'detection'}:
            return value
    return ''

def _default_validation_image(bs: Path) -> tuple[str, str]:
    """Select one deterministic image from the materialised run-mode subset."""
    task = _benchmark_task(bs)
    roots = []
    if task:
        roots.append(bs / 'resources' / 'validation' / task)
    roots.append(bs / 'resources' / 'validation')
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key in seen or not root.is_dir():
            continue
        seen.add(key)
        images = sorted(
            p for p in root.rglob('*')
            if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        )
        if images:
            return str(images[0].resolve()), 'materialised_validation_subset:first_sorted'
    return '', ''

def _sha256_file(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return ''
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def _result_file_state(path: Path):
    """Observe an existing result without deleting evidence from an earlier run."""
    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except FileNotFoundError:
        return None

def _no_duplicate_json_keys(pairs):
    value={}
    for key,item in pairs:
        if key in value: raise ValueError(f'duplicate_json_key:{key}')
        value[key]=item
    return value

def _load_json(p: Path):
    try:
        return json.loads(
            p.read_text(encoding='utf-8'),
            object_pairs_hook=_no_duplicate_json_keys,
        )
    except Exception: return {}

def _valid_quality_first_binding_set(value, *, setup_id: str) -> bool:
    if not isinstance(value,dict): return False
    bindings=value.get('bindings_by_model_case_backend')
    if (
        value.get('schema') != 'onnx-splitpoint/native-split-quality-binding-set'
        or int(value.get('schema_version') or 0) != 2
        or str(value.get('setup_id') or '') != str(setup_id or '')
        or not str(value.get('eval_run_id') or '')
        or not isinstance(bindings,dict)
        or len(str(value.get('central_quality_summary_sha256') or '').strip()) != 64
    ): return False
    declared=str(value.get('binding_set_sha256') or '').strip().lower()
    unhashed=dict(value); unhashed.pop('binding_set_sha256',None)
    if len(declared) != 64 or canonical_json_sha256(unhashed) != declared:
        return False
    for binding in bindings.values():
        verified,_=validate_native_split_quality_binding(
            binding, verification_mode='portable',
        )
        if verified is None: return False
        receipt,_=validate_central_native_split_quality_selection(
            verified, required=True,
        )
        if receipt is None: return False
        if str(verified.get('eval_run_id') or '') != str(value.get('eval_run_id') or ''):
            return False
    return True


_COMPLETION_PROJECTION_EXACT_FIELDS = {
    'task',
    'stage',
    'contract_family',
    'endpoint_contract_complete',
    'endpoint_contract_hash',
    'output_endpoint_id',
    'completed_frames',
    'completed_work_units',
    'postprocess_included',
    'postprocess_completed_frames',
    'postprocess_completion_verified',
    'completed_task_endpoint_contract',
    'completed_task_endpoint_contract_hash',
    'comparison_endpoint_contract',
    'comparison_endpoint_contract_hash',
    'measurement_boundary',
    'last_completion_source',
    'postprocess_ms',
}


def _completion_summary_projection(
    result: dict,
) -> dict:
    """Project completion evidence without flattening or re-hashing it."""
    if not isinstance(result, dict):
        return {}
    return {
        key: value
        for key, value in result.items()
        if (
            key in _COMPLETION_PROJECTION_EXACT_FIELDS
            or key.startswith('completion_')
            or key.startswith('completed_task_')
        )
    }


def _same_hotloop_completion_error(
    result: dict,
    *,
    require_contract: bool,
) -> str:
    """Return a stable failure reason for incomplete detection completion."""
    if not isinstance(result, dict):
        return 'detection_same_hotloop_completion_result_missing'
    attestation = result.get('completion_execution_attestation')
    completed_endpoint = result.get('completed_task_endpoint_contract')
    comparison_endpoint = result.get('comparison_endpoint_contract')
    try:
        completed = int(result.get('completed_work_units'))
        completed_frames = int(result.get('completed_frames'))
        postprocess_completed = int(
            result.get('postprocess_completed_frames')
        )
    except (TypeError, ValueError, OverflowError):
        return 'detection_same_hotloop_completion_count_invalid'
    try:
        attested_completed = int(
            attestation.get('completed_work_units')
        ) if isinstance(attestation, dict) else 0
        attested_count = int(
            attestation.get('completion_count')
        ) if isinstance(attestation, dict) else 0
    except (TypeError, ValueError, OverflowError):
        return 'detection_same_hotloop_completion_count_invalid'
    if (
        completed <= 0
        or completed_frames != completed
        or postprocess_completed != completed
        or result.get('postprocess_included') is not True
        or result.get('postprocess_completion_verified') is not True
        or result.get('completion_observation_relation')
        != 'same_hotloop_sentinel'
        or result.get('completion_exact_result_claim_bound') is not True
        or result.get('measurement_boundary')
        != 'workers_ready_to_last_completed_task_frame'
        or result.get('last_completion_source')
        != 'same_hotloop_completed_task_sentinel'
        or not isinstance(attestation, dict)
        or attestation.get('attested') is not True
        or attestation.get('status') != 'passed'
        or attestation.get('observation_relation')
        != 'same_hotloop_sentinel'
        or attestation.get('same_hotloop_sentinel') is not True
        or attestation.get('exact_result_claim_bound') is not True
        or attested_completed != completed
        or attested_count != completed
        or not isinstance(completed_endpoint, dict)
        or not str(
            completed_endpoint.get('endpoint_contract_hash') or ''
        )
        or not isinstance(comparison_endpoint, dict)
        or not str(
            comparison_endpoint.get('endpoint_contract_hash') or ''
        )
        or attestation.get('completed_endpoint_contract')
        != completed_endpoint
        or attestation.get('comparison_endpoint_contract')
        != comparison_endpoint
    ):
        return 'detection_same_hotloop_completion_attestation_invalid'
    execution_hash = str(
        result.get('completion_execution_contract_sha256') or ''
    )
    if (
        len(execution_hash) != 64
        or str(attestation.get('execution_contract_sha256') or '')
        != execution_hash
    ):
        return 'detection_same_hotloop_completion_contract_hash_mismatch'
    if require_contract:
        contract = result.get('completion_execution_contract')
        if (
            not isinstance(contract, dict)
            or str(contract.get('contract_sha256') or '')
            != execution_hash
            or contract.get('completed_endpoint_contract')
            != completed_endpoint
            or contract.get('comparison_endpoint_contract')
            != comparison_endpoint
        ):
            return 'detection_same_hotloop_completion_contract_missing'
    for field in (
        'artifact_sha256', 'schema_sha256', 'content_sha256',
        'invocation_sha256', 'relation_sha256',
    ):
        projected = str(result.get(f'completion_{field}') or '')
        if len(projected) != 64 or projected != str(
            attestation.get(field) or ''
        ):
            return (
                'detection_same_hotloop_completion_hash_projection_'
                f'mismatch:{field}'
            )
    artifact = attestation.get('artifact')
    artifact_sha256 = str(
        attestation.get('artifact_sha256') or ''
    ).strip().lower()
    persisted_artifact = result.get(
        'completed_task_result_artifact'
    )
    persisted_path_raw = str(
        result.get('completed_task_result_artifact_path') or ''
    ).strip()
    persisted_path = Path(persisted_path_raw).expanduser()
    if (
        result.get('completed_task_result_artifact_saved') is not True
        or not isinstance(artifact, dict)
        or not isinstance(persisted_artifact, dict)
        or persisted_artifact != artifact
        or canonical_json_sha256(artifact) != artifact_sha256
        or str(
            result.get('completed_task_result_artifact_sha256') or ''
        ).strip().lower() != artifact_sha256
        or str(
            result.get(
                'completed_task_result_artifact_file_sha256'
            ) or ''
        ).strip().lower() != artifact_sha256
        or not persisted_path_raw
        or not persisted_path.is_absolute()
        or not persisted_path.is_file()
        or persisted_path.is_symlink()
        or _sha256_file(str(persisted_path)) != artifact_sha256
        or _load_json(persisted_path) != artifact
    ):
        return 'detection_same_hotloop_completion_artifact_not_persisted'
    return ''


def _detection_completion_error(result: dict) -> str:
    error = _same_hotloop_completion_error(
        result, require_contract=True,
    )
    if error:
        return error
    repetitions = result.get(
        'repetition_records', result.get('repetition_evidence')
    )
    if not isinstance(repetitions, list) or not repetitions:
        return 'detection_same_hotloop_completion_repetitions_missing'
    for index, record in enumerate(repetitions):
        error = _same_hotloop_completion_error(
            record, require_contract=False,
        )
        if error:
            return f'{error}:repetition_{index + 1}'
    return ''


def _case_dirs(bs: Path):
    try:
        return sorted([p.name for p in bs.iterdir() if p.is_dir() and p.name.startswith('b')])
    except Exception:
        return []

def _resolve_benchmark_set(root: Path, model: str) -> Path | None:
    base = root / model / 'benchmark_set'
    candidates = [
        base / 'legacy_suite',
        base,
    ]
    # Prefer a directory that actually contains b*/ case folders.  Some EvalRun
    # copies store the executable BenchmarkSet under benchmark_set/legacy_suite,
    # while benchmark_set itself is only a wrapper.
    for c in candidates:
        if c.exists() and _case_dirs(c):
            return c
    for c in candidates:
        if (c / 'benchmark_set.json').is_file():
            return c
    return base if base.exists() else None

def main():
    ap=argparse.ArgumentParser(description='Run native producer E2E runners/scaffolds over staged BenchmarkSets')
    ap.add_argument('--root', required=True); ap.add_argument('--backend', choices=['hailo10h','deepx'], required=True)
    ap.add_argument('--models', default=''); ap.add_argument('--case-map', default=''); ap.add_argument('--precision', default='uint8_cast_fp16')
    ap.add_argument('--frames', type=int, default=1000); ap.add_argument('--warmup', type=int, default=100); ap.add_argument('--repetitions', type=int, default=1, help='Independent performance repetitions; aggregate is median plus 95%% CI, never best-of.'); ap.add_argument('--duration-s', type=float, default=0.0, help='Run each native producer workload for a fixed duration (energy mode).'); ap.add_argument('--queue-depth', type=int, default=3); ap.add_argument('--inflight', type=int, default=8)
    ap.add_argument('--build-missing-engine', action='store_true')
    ap.add_argument('--force-rebuild-engine', action='store_true')
    ap.add_argument('--dump-outputs', action='store_true')
    ap.add_argument('--dump-boundary', action='store_true')
    ap.add_argument('--engine-build-python', default='auto', help='Forwarded to native producer scripts for TRT engine construction.')
    ap.add_argument('--boundary-layout', default='as_input')
    ap.add_argument('--dequant-scale', type=float, default=0.0)
    ap.add_argument('--dequant-zero-point', type=float, default=0.0)
    ap.add_argument('--hailo-format', default='', help='For Hailo10H: uint8 keeps quantized outputs, float32 requests FLOAT32 HailoRT outputs.')
    ap.add_argument('--preprocess-mode', default='auto', choices=['auto','resize','letterbox'])
    ap.add_argument('--letterbox-pad-value', type=int, default=114)
    ap.add_argument('--image-map', default='', help='JSON map model->case->image or model->image used for exact native validation dumps.')
    ap.add_argument('--inspect-runtime', action='store_true', help='For DeepX: run runtime API introspection and include it in status report')
    ap.add_argument('--analysis-tag', default='', help='Optional stable tag used to keep per-contract analysis summaries instead of overwriting them.')
    ap.add_argument('--timeout', type=float, default=3600.0, help='Per-row native producer timeout in seconds.')
    ap.add_argument('--setup-id', default=os.environ.get('ONNX_SPLITPOINT_SETUP_ID',''))
    ap.add_argument('--deepx-classification-profile-json', default='', help='Selected workflow numeric preprocessing admission; no model rebuild is performed here.')
    ap.add_argument('--native-job-prerequisites', default='', help='Dispatcher-owned planned jobs and their upstream blocking evidence.')
    ap.add_argument('--native-split-quality-binding-set', default=os.environ.get('ONNX_SPLITPOINT_NATIVE_SPLIT_BINDING_SET',''))
    ns=ap.parse_args()
    if ns.force_rebuild_engine:
        ap.error('productive_force_build_disabled: --force-rebuild-engine is disabled; --build-missing-engine remains available')
    if int(ns.repetitions) < 1:
        ap.error('--repetitions must be >= 1')
    if float(ns.duration_s or 0.0) > 0.0 and int(ns.repetitions) != 1:
        ap.error('--duration-s energy workloads require --repetitions=1')
    runtime_precision_default=str(ns.precision or '')
    if ns.backend == 'hailo10h' and runtime_precision_default in {'', 'uint8_cast_fp16'}:
        # This Hailo-10 deployment exposes native UINT8 output.  Its fallback must use
        # the exact QuantInfo dequant bridge even without a row Quality binding;
        # the shared historical cast precision remains untouched for Hailo-8.
        runtime_precision_default='uint8_dequant_fp16'
    root=Path(ns.root).expanduser().resolve(); models=[x.strip() for x in ns.models.split(',') if x.strip()] if ns.models else [p.name for p in root.iterdir() if p.is_dir()]
    quality_first_required=bool(str(ns.native_split_quality_binding_set or '').strip())
    binding_set_path=Path(str(ns.native_split_quality_binding_set or '')).expanduser().resolve()
    binding_set=_load_json(binding_set_path) if quality_first_required else {}
    binding_set_invalid = quality_first_required and not _valid_quality_first_binding_set(
        binding_set, setup_id=str(ns.setup_id or ''),
    )
    if quality_first_required and (ns.build_missing_engine or ns.force_rebuild_engine):
        ap.error('Quality-FIRST native split forbids coordinator engine build/rebuild flags')
    case_map_supplied=bool(str(ns.case_map or '').strip())
    try:
        cmap=_parse_case_map(ns.case_map)
    except Exception as exc:
        print(json.dumps({
            'ok':False,'status':'selection_invalid',
            'failure_reason':'case_map_invalid',
            'error':f'{type(exc).__name__}: {exc}','rows':0,
        },indent=2))
        return 3
    imap=_parse_image_map(ns.image_map); rows=[]
    prerequisites=_load_json(Path(ns.native_job_prerequisites)) if ns.native_job_prerequisites else {}
    prerequisite_rows=prerequisites.get('rows', []) if isinstance(prerequisites,dict) else []
    classification_admissions={}
    case_map_active = case_map_supplied
    selection_errors=[]
    if not models:
        selection_errors.append('requested_or_discovered_models_empty')
    if case_map_active:
        unknown=sorted(set(cmap).difference(models))
        missing=sorted(set(models).difference(cmap))
        if unknown:
            selection_errors.append('case_map_unknown_models:'+','.join(unknown))
        if missing:
            selection_errors.append('case_map_missing_requested_models:'+','.join(missing))
    for model in models:
        benchmark_set=_resolve_benchmark_set(root,model)
        benchmark_index_ok=bool(
            benchmark_set and (
                (benchmark_set/'benchmark_set.json').is_file()
                or (benchmark_set.parent/'benchmark_set.json').is_file()
            )
        )
        if not benchmark_index_ok:
            selection_errors.append(f'benchmark_set_not_found:{model}')
            continue
        if case_map_active:
            available=set(_case_dirs(benchmark_set))
            missing_cases=[case for case in cmap.get(model,[]) if case not in available]
            if missing_cases:
                selection_errors.append(
                    f'case_map_cases_not_found:{model}:'+','.join(missing_cases)
                )
    if selection_errors and case_map_active and not any(cmap.get(model) for model in models):
        # No selected case identity exists to serialize. Do not invent one;
        # the workflow dispatcher retains its own fully planned failed jobs.
        print(json.dumps({'ok':False,'status':'selection_invalid',
                          'failure_reason':'requested_benchmark_set_or_case_unavailable',
                          'selection_errors':selection_errors,'models':models,'case_map':cmap,'rows':0},indent=2))
        return 3
    for m in models:
        bs=_resolve_benchmark_set(root, m)
        if not bs or not bs.exists():
            for missing_case in cmap.get(m, ['']):
                rows.append(failed_native_result(planned_native_identity({'backend':ns.backend,'model':m,'case':missing_case,'setup_id':ns.setup_id,'precision':runtime_precision_default}), failure_stage='selection', failure_reason='missing_benchmark_set'))
            continue
        cases=cmap.get(m, []) if case_map_active else _case_dirs(bs)
        if not cases:
            rows.append({'backend':ns.backend,'model':m,'case':'','ok':False,'status':'no_cases_found','benchmark_set':str(bs)}); continue
        for c in cases:
            source_backend='hailo10h_to_trt' if ns.backend == 'hailo10h' else 'deepx_to_trt'
            binding_key=f'{m}|{c}|{source_backend}'
            binding=(binding_set.get('bindings_by_model_case_backend') or {}).get(binding_key) if quality_first_required else {}
            row_quality_first=bool(
                quality_first_required and isinstance(binding,dict)
            )
            row_quality_error=(
                '' if row_quality_first or not quality_first_required else
                f'native_split_quality_binding_missing:{binding_key}'
            )
            selection=binding.get('preselection') if isinstance(binding,dict) and isinstance(binding.get('preselection'),dict) else {}
            effective_precision=str(selection.get('precision') or '') if row_quality_first else runtime_precision_default
            if row_quality_first and not effective_precision:
                row_quality_first=False
                row_quality_error=f'native_split_quality_precision_missing:{binding_key}'
                effective_precision=runtime_precision_default
            identity=planned_native_identity({'backend':source_backend,'model':m,'case':c,'setup_id':ns.setup_id,'precision':effective_precision})
            if ns.backend == 'deepx' and ns.deepx_classification_profile_json:
                from onnx_splitpoint_tool.deepx.config import classification_profile_admission, declared_deepx_task
                try:
                    profile=json.loads(ns.deepx_classification_profile_json)
                    if not isinstance(profile,dict): raise ValueError('profile must be an object')
                    output_contract=_load_json(bs/c/'deepx/deepx_m1/part1/output_contract.json')
                    task=declared_deepx_task(_load_json(bs/'benchmark_set.json'), output_contract, plan_task=_benchmark_task(bs))
                    admission=classification_profile_admission(profile, {'task':task})
                    classification_admissions[(m,c)]=admission
                    reason=admission['reason'] if not admission['allowed'] else ''
                    if task == 'classification' and admission['allowed']:
                        actual_mode=str(output_contract.get('classification_preprocessing') or '')
                        if actual_mode != admission['classification_preprocessing']:
                            reason='deepx_classification_preprocessing_artifact_mismatch:expected='+admission['classification_preprocessing']+',observed='+(actual_mode or 'missing')
                    if reason:
                        rows.append(failed_native_result(identity,failure_stage='classification_preprocessing_admission',failure_reason=reason))
                        continue
                except (ValueError,TypeError) as exc:
                    rows.append(failed_native_result(identity,failure_stage='classification_preprocessing_admission',failure_reason='deepx_classification_profile_invalid:'+str(exc)))
                    continue
            upstream=[row for row in prerequisite_rows if isinstance(row,dict) and
                      planned_native_identity(row.get('planned_native_identity') or row) == identity]
            blocked=next((row for row in upstream if row.get('prerequisite_status') == 'blocked'), None)
            if blocked is not None:
                rows.append(failed_native_result(identity, failure_stage=str(blocked.get('failure_stage') or 'preparation'),
                    failure_reason=str(blocked.get('primary_failure_reason') or blocked.get('failure_reason') or 'native_prerequisite_blocked'),
                    upstream_evidence_path=blocked.get('upstream_evidence_path','')))
                continue
            if not (bs/c).is_dir():
                rows.append(failed_native_result(identity, failure_stage='selection', failure_reason='case_map_case_not_found'))
                continue
            if binding_set_invalid or row_quality_error:
                rows.append(failed_native_result(identity, failure_stage='evaluate_quality',
                    failure_reason='native_split_quality_binding_set_invalid' if binding_set_invalid else row_quality_error))
                continue
            binding_path=None
            if row_quality_first:
                binding_dir=root/'quality_first'/'consumed_bindings'/str(ns.backend)/m
                binding_dir.mkdir(parents=True,exist_ok=True)
                binding_path=binding_dir/f'{c}.json'
                binding_path.write_text(json.dumps(binding,indent=2,sort_keys=True)+'\n',encoding='utf-8')
            if ns.backend == 'deepx' and not row_quality_first:
                part1_dxnn = bs / c / 'deepx' / 'deepx_m1' / 'part1' / 'model.dxnn'
                if not part1_dxnn.is_file():
                    reason = f'missing_deepx_part1_artifact:{part1_dxnn}'
                    rows.append(failed_native_result(identity, failure_stage='native_preparation', failure_reason=reason, **{
                        'backend': ns.backend, 'model': m, 'case': c, 'precision': runtime_precision_default,
                        'ok': False, 'status': 'missing_deepx_part1_artifact',
                        'status_detail': reason, 'error': reason,
                        'returncode': None, 'rc': None, 'timed_out': False,
                        'stdout_tail': '', 'stderr_tail': '', 'fps_makespan': None,
                        'paper_fps': None, 'handoff_ms': None, 'p1_ms': None, 'p2_run_ms': None,
                        'producer_ready': False, 'consumer_ready': False, 'report': '',
                        'steps': [{'name': 'deepx_part1_preflight', 'rc': 2, 'artifact': str(part1_dxnn), 'status': 'missing'}],
                    }))
                    continue
                if not ns.build_missing_engine:
                    part2_engine = (
                        bs / 'native_trt' / c / 'part2' / effective_precision
                        / f'part2_{effective_precision}.engine'
                    )
                    try:
                        part2_available = part2_engine.is_file() and part2_engine.stat().st_size > 0
                    except OSError:
                        part2_available = False
                    if not part2_available:
                        reason = (
                            'native_trt_part2_variant_missing:'
                            f'model={m},case={c},precision={effective_precision},path={part2_engine}'
                        )
                        rows.append(failed_native_result(identity, failure_stage='native_preparation', failure_reason=reason, **{
                            'backend': ns.backend, 'model': m, 'case': c, 'precision': effective_precision,
                            'ok': False, 'status': 'native_trt_part2_variant_missing',
                            'status_detail': reason, 'error': reason,
                            'returncode': None, 'rc': None, 'timed_out': False,
                            'stdout_tail': '', 'stderr_tail': '', 'fps_makespan': None,
                            'paper_fps': None, 'handoff_ms': None, 'p1_ms': None, 'p2_run_ms': None,
                            'producer_ready': False, 'consumer_ready': False, 'report': '',
                            'steps': [{'name': 'deepx_part2_variant_preflight', 'rc': 2,
                                       'artifact': str(part2_engine), 'status': 'missing'}],
                        }))
                        continue
            image_val = ''
            image_source = ''
            try:
                mv = imap.get(m) if isinstance(imap, dict) else None
                if isinstance(mv, dict): image_val = str(mv.get(c) or mv.get(str(c)) or '')
                elif isinstance(mv, str): image_val = mv
                if image_val:
                    image_val = _resolve_image(bs, image_val)
                    image_source = 'generic_validation_image_map'
            except Exception:
                image_val = ''
                image_source = ''
            if not image_val or not Path(image_val).is_file():
                image_val, image_source = _default_validation_image(bs)
            image_sha256 = _sha256_file(image_val)
            task = _benchmark_task(bs)
            if task not in {'classification', 'detection'}:
                reason = 'benchmark_task_missing_or_invalid'
                rows.append(failed_native_result(identity, failure_stage='native_preparation', failure_reason=reason, **{
                    'backend': ns.backend, 'model': m, 'case': c, 'precision': runtime_precision_default,
                    'ok': False, 'status': reason,                     'status_detail': reason, 'error': reason, 'returncode': None,
                    'rc': None, 'timed_out': False, 'report': '', 'steps': [],
                }))
                continue
            if ns.backend=='hailo10h':
                cmd=[sys.executable, str(ROOT/'scripts'/'native_hailo10_trt_e2e_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', c, '--precision', effective_precision, '--frames', str(ns.frames), '--warmup', str(ns.warmup), '--repetitions', str(ns.repetitions), '--queue-depth', str(ns.queue_depth), '--inflight', str(ns.inflight), '--preprocess-mode', str(ns.preprocess_mode), '--letterbox-pad-value', str(int(ns.letterbox_pad_value)), '--setup-id', str(ns.setup_id), '--model-id', m]
                cmd.append('--quantized-inputs')
                if row_quality_first:
                    cmd += ['--eval-run-id', str(binding_set.get('eval_run_id') or ''), '--source-run-id', canonical_native_split_backend(binding.get('source_run_id') or source_backend, ns.setup_id), '--native-split-quality-binding', str(binding_path)]
                if task:
                    cmd += ['--task', task]
                if ns.duration_s and float(ns.duration_s) > 0:
                    cmd += ['--duration-s', str(float(ns.duration_s))]
                if getattr(ns, 'dump_outputs', False):
                    cmd.append('--dump-outputs')
                if getattr(ns, 'dump_boundary', False):
                    cmd.append('--dump-boundary')
                if ns.boundary_layout and ns.boundary_layout != 'as_input':
                    cmd += ['--boundary-layout', ns.boundary_layout]
                if effective_precision == 'uint8_dequant_fp16':
                    if float(ns.dequant_scale or 0.0) > 0.0:
                        cmd += ['--dequant-scale', str(float(ns.dequant_scale))]
                    cmd += ['--dequant-zero-point', str(float(ns.dequant_zero_point or 0.0))]
                # FLOAT32 layout precision should use HailoRT float32 outputs; uint8 bridge precisions keep quantized outputs.
                hfmt = str(selection.get('hailo_format') or ns.hailo_format or ('float32' if effective_precision == 'float32_layout_fp16' else 'uint8')).strip().lower()
                if hfmt in {'float32','fp32','float'}:
                    cmd.append('--no-quantized-outputs')
                else:
                    cmd.append('--quantized-outputs')
                if ns.engine_build_python:
                    cmd += ['--engine-build-python', ns.engine_build_python]
                if image_val:
                    cmd += ['--image', image_val]
            else:
                cmd=[sys.executable, str(ROOT/'scripts'/'native_deepx_trt_e2e_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', c, '--precision', effective_precision, '--frames', str(ns.frames), '--warmup', str(ns.warmup), '--repetitions', str(ns.repetitions), '--queue-depth', str(ns.queue_depth), '--task', task, '--preprocess-mode', str(ns.preprocess_mode), '--letterbox-pad-value', str(int(ns.letterbox_pad_value)), '--setup-id', str(ns.setup_id), '--model-id', m]
                if row_quality_first:
                    cmd += ['--eval-run-id', str(binding_set.get('eval_run_id') or ''), '--source-run-id', canonical_native_split_backend(binding.get('source_run_id') or source_backend, ns.setup_id), '--native-split-quality-binding', str(binding_path)]
                if ns.duration_s and float(ns.duration_s) > 0:
                    cmd += ['--duration-s', str(float(ns.duration_s))]
                if getattr(ns, 'dump_outputs', False):
                    cmd.append('--dump-outputs')
                if getattr(ns, 'dump_boundary', False):
                    cmd.append('--dump-boundary')
                if ns.boundary_layout:
                    cmd += ['--boundary-layout', ns.boundary_layout]
                if ns.engine_build_python:
                    cmd += ['--engine-build-python', ns.engine_build_python]
                if image_val:
                    cmd += ['--image', image_val]
                # --inspect-runtime is kept for CLI compatibility; the DeepX E2E runner is now an executable adapter.
            if not quality_first_required:
                if ns.build_missing_engine:
                    cmd.append('--build-missing-engine')
                if ns.force_rebuild_engine and ns.backend == 'hailo10h':
                    cmd.append('--force-rebuild-engine')
            print('[producer-e2e-eval]', ns.backend, m, c, ' '.join(cmd), flush=True)
            sub='hailo10h_to_trt' if ns.backend=='hailo10h' else 'deepx_to_trt'
            fname='hailo10_native_fifo_e2e_results.json' if ns.backend=='hailo10h' else 'deepx_native_fifo_e2e_results.json'
            rp=bs/'native_pipeline'/c/sub/effective_precision/fname
            previous_result_state = _result_file_state(rp)
            timed_out = False
            try:
                proc=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=float(ns.timeout))
                returncode = int(proc.returncode)
                stdout = str(proc.stdout or '')
                stderr = str(proc.stderr or '')
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = 124
                stdout = str(getattr(exc, 'stdout', '') or '')
                stderr = str(getattr(exc, 'stderr', '') or '')
            current_result_state = _result_file_state(rp)
            result_fresh = bool(
                current_result_state is not None
                and current_result_state != previous_result_state
            )
            child_json_error = ''
            try:
                j=json.loads(rp.read_text(encoding='utf-8'), object_pairs_hook=_no_duplicate_json_keys) if result_fresh else {}
                if not isinstance(j, dict):
                    raise ValueError('child_result_must_be_object')
            except (OSError, ValueError, TypeError) as exc:
                j = {}
                child_json_error = 'native_result_invalid_json:' + type(exc).__name__
            child_identity=attach_identity_without_conflicts(identity,j)
            identity_error=str(child_identity.get('failure_reason') or '') if child_identity.get('identity_conflicts') else ''
            runtime_success = bool(
                j.get('ok') and returncode == 0 and not timed_out
                and not j.get('error')
            )
            quality_identity_error=''
            if row_quality_first and runtime_success:
                selected=native_split_quality_selection_duplicates(binding)
                required_result={
                    'eval_run_id':str(binding_set.get('eval_run_id') or ''),
                    'native_split_quality_eval_run_id':str(binding_set.get('eval_run_id') or ''),
                    'source_run_id':canonical_native_split_backend(binding.get('source_run_id'), ns.setup_id),
                    'native_split_quality_source_run_id':canonical_native_split_backend(binding.get('source_run_id'), ns.setup_id),
                    'native_split_quality_binding_sha256':str(binding.get('binding_sha256') or ''),
                    **selected,
                }
                drift=[
                    field for field,expected in required_result.items()
                    if (
                        canonical_native_split_backend(j.get(field), ns.setup_id)
                        != canonical_native_split_backend(expected, ns.setup_id)
                        if field in {'source_run_id', 'native_split_quality_source_run_id'}
                        else str(j.get(field) or '').strip().lower()
                        != str(expected or '').strip().lower()
                    )
                ]
                if drift:
                    quality_identity_error=(
                        'native_split_quality_child_result_identity_mismatch:'
                        + ','.join(drift)
                    )
            completion_error = (
                _detection_completion_error(j)
                if runtime_success and task == 'detection' else ''
            )
            completion_projection = _completion_summary_projection(j)
            result_ok = (
                runtime_success
                and not identity_error
                and not quality_identity_error
                and not completion_error
            )
            technical_quality_error = str(
                row_quality_error or quality_identity_error
                or completion_error or ''
            )
            row_quality_qualified = bool(
                row_quality_first and runtime_success and not technical_quality_error
            )
            if result_ok:
                failure = ''
                status = 'ok'
                status_detail = 'ok'
                error_text = ''
            else:
                failure = str(
                    identity_error or ('native_producer_timeout' if timed_out else '') or child_json_error or
                    j.get('error') or j.get('failure_reason') or
                    ('native_producer_nonzero_exit' if returncode != 0 else '') or
                    ('native_result_missing' if current_result_state is None else '') or
                    ('native_result_stale' if not result_fresh else '') or
                    quality_identity_error or completion_error or
                    j.get('status_detail') or j.get('next_action') or
                    'native_result_not_ok'
                )
                status = str(j.get('status') or 'failed')
                if status.lower() in {'ok', 'success', 'passed'}:
                    status = 'failed'
                status_detail = failure
                error_text = str(j.get('error') or stderr[-2000:] or stdout[-2000:] or failure)
                if not runtime_success:
                    technical_quality_error = str(row_quality_error or failure)
            rows.append({
                'backend':ns.backend,'model':m,'case':c,'precision':effective_precision,
                'ok':result_ok,'status':status,
                'runtime_success':runtime_success,
                'failure_reason':failure,'status_detail':status_detail,
                'error':error_text,
                'returncode':returncode,'rc':returncode,'timed_out':timed_out,
                'result_fresh':result_fresh,'native_child_launch_count':1,
                'stdout_tail':stdout[-4000:],'stderr_tail':stderr[-4000:],
                'input_image':image_val,'input_image_source':image_source,'input_image_sha256':image_sha256,
                'fps_makespan':j.get('fps_makespan'), 'paper_fps':j.get('paper_equivalent_fps'),
                'fps_median':j.get('fps_median', j.get('fps_makespan_median')),
                'fps_ci95_low':j.get('fps_ci95_low', j.get('fps_makespan_ci95_low')),
                'fps_ci95_high':j.get('fps_ci95_high', j.get('fps_makespan_ci95_high')),
                'repetition_count_requested':j.get('repetition_count_requested', j.get('repetitions_requested')),
                'repetition_count_attempted':j.get('repetition_count_attempted', j.get('repetitions_completed')),
                'repetition_count_valid':j.get('repetition_count_valid', j.get('repetitions_completed')),
                'repetition_status':j.get('repetition_status'),
                'repetition_aggregation':j.get('repetition_aggregation'),
                'request_latency': j.get('request_latency'),
                'repetition_records':j.get('repetition_records', j.get('repetition_evidence', [])),
                'repetition_evidence':j.get('repetition_evidence', j.get('repetition_records', [])),
                'handoff_ms':j.get('handoff_ms'), 'p1_ms':j.get('p1_ms'), 'p2_run_ms':j.get('p2_run_ms'),
                'producer_ready':j.get('producer_ready'), 'consumer_ready':j.get('consumer_ready'),
                'report':str(rp),
                'setup_id':identity['setup_id'],
                'comparison_backend':identity['comparison_backend'],
                'planned_native_identity':identity,
                'primary_failure_reason':failure,
                'failure_stage':'child_result_identity' if identity_error else 'native_runtime' if failure else '',
                'identity_conflicts':child_identity.get('identity_conflicts',[]),
                'eval_run_id':j.get('eval_run_id'),
                'source_run_id':j.get('source_run_id'),
                'native_split_quality_required':quality_first_required,
                'native_split_quality_available':row_quality_first,
                'technical_quality_error':technical_quality_error,
                'energy_quality_qualified':row_quality_qualified,
                'energy_quality_status':(
                    'quality_qualified' if row_quality_qualified
                    else 'raw_energy_quality_not_qualified'
                ),
                'native_energy_after_technical_error':(
                    'not_applicable_quality_qualified' if row_quality_qualified
                    else 'collect_raw_quality_unqualified'
                ),
                'performance_claims_emitted':bool(
                    row_quality_qualified and result_ok
                ),
                'execution_role':(
                    'quality_first_native_split' if row_quality_qualified
                    else 'runtime_observation_quality_unqualified'
                    if quality_first_required or runtime_success
                    else 'legacy_manual_diagnostic'
                ),
                'native_command_contract':j.get('native_command_contract'),
                'native_command_contract_sha256':j.get('native_command_contract_sha256'),
                'workload_contract_sha256':j.get('workload_contract_sha256'),
                'native_split_quality_binding':j.get('native_split_quality_binding'),
                'native_split_quality_binding_sha256':j.get('native_split_quality_binding_sha256'),
                'native_split_quality_eval_run_id':j.get('native_split_quality_eval_run_id'),
                'native_split_quality_source_run_id':j.get('native_split_quality_source_run_id'),
                'source_request_sha256':j.get('source_request_sha256'),
                'native_split_quality_source_request_sha256':j.get('native_split_quality_source_request_sha256'),
                'native_split_quality_central_result_sha256':j.get('native_split_quality_central_result_sha256'),
                'native_split_quality_selection_sha256':j.get('native_split_quality_selection_sha256'),
                'native_split_quality_consumer_attestation':j.get('native_split_quality_consumer_attestation'),
                'native_split_quality_consumer_status':j.get('native_split_quality_consumer_status'),
                'native_fifo_output_manifest':j.get('native_fifo_output_manifest',j.get('native_output_manifest')),
                'native_fifo_boundary_manifest':j.get('native_fifo_boundary_manifest'),
                'steps':[{'name':'native_producer_e2e','cmd':cmd,'rc':returncode,'timed_out':timed_out,'stdout_tail':stdout[-4000:],'stderr_tail':stderr[-4000:]}],
                **completion_projection,
            })
    for row in rows:
        admission=classification_admissions.get((row.get('model'),row.get('case')))
        if admission is not None:
            row['deepx_classification_admission']=admission
            if not admission.get('claim_eligible',True):
                row.update(diagnostic_only=True,counts_as_benchmark=False,claim_eligible=False,
                           performance_claims_emitted=False,performance_claim_eligible=False,
                           energy_claim_eligible=False,scientific_claim_eligible=False,
                           scientific_claim_exclusion_reason=admission['scientific_claim_exclusion_reason'])
    outdir=root/'analysis_tables'; outdir.mkdir(exist_ok=True)
    tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(ns.analysis_tag or '').strip()).strip('._-')
    stem = f'native_{ns.backend}_producer_e2e_eval' + (f'__{tag}' if tag else '')
    json_path = outdir/f'{stem}.json'; csv_path = outdir/f'{stem}.csv'; md_path = outdir/f'{stem}.md'
    ok_count=sum(1 for r in rows if r.get('ok'))
    failed_count=len(rows)-ok_count
    orchestration_ok=all(
        r.get('ok') is True
        and int(r.get('returncode') if r.get('returncode') is not None else 1)==0
        and not bool(r.get('timed_out'))
        for r in rows
    ) if rows else False
    evidence_status='complete' if rows and ok_count==len(rows) else ('partial' if ok_count else 'unavailable')
    summary={
        'schema':'onnx-splitpoint/native-producer-e2e-eval-summary','schema_version':2,
        'root':str(root),'backend':ns.backend,'analysis_tag':tag,'performance_repetitions':int(ns.repetitions),'rows':rows,
        'row_count':len(rows),'ok_count':ok_count,'failed_count':failed_count,
        'ok':bool(orchestration_ok),
        'orchestration_status':'ok' if orchestration_ok else 'failed',
        'evidence_status':evidence_status,
        'failure_reasons':sorted({str(r.get('failure_reason') or '') for r in rows if not r.get('ok') and str(r.get('failure_reason') or '')}),
    }
    json_path.write_text(json.dumps(summary,indent=2), encoding='utf-8')
    fields=['backend','model','case','precision','ok','status','failure_reason','status_detail','error','returncode','timed_out','stdout_tail','stderr_tail','input_image','input_image_source','input_image_sha256','fps_makespan','fps_median','fps_ci95_low','fps_ci95_high','repetition_count_requested','repetition_count_attempted','repetition_count_valid','repetition_status','repetition_aggregation','paper_fps','handoff_ms','p1_ms','p2_run_ms','completion_tail_ms','completed_work_units','postprocess_included','postprocess_completed_frames','postprocess_completion_verified','completion_observation_relation','completion_exact_result_claim_bound','completion_execution_contract_sha256','completed_task_endpoint_contract_hash','comparison_endpoint_contract_hash','completed_task_result_artifact','completed_task_result_artifact_path','completed_task_result_artifact_saved','completed_task_result_artifact_sha256','completed_task_result_artifact_file_sha256','completion_artifact_sha256','completion_schema_sha256','completion_content_sha256','completion_invocation_sha256','completion_relation_sha256','producer_ready','consumer_ready','report','rc','steps_json']
    csv_rows=[]
    for row in rows:
        rec={key: row.get(key) for key in fields if key != 'steps_json'}
        rec['steps_json']=json.dumps(row.get('steps') or [], ensure_ascii=False, sort_keys=True)
        csv_rows.append(rec)
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(csv_rows)
    md=['# Native producer E2E eval','',f'Backend: `{ns.backend}`',f'Analysis tag: `{tag}`','', '| model | case | ok | FPS | handoff ms | status |','|---|---|---:|---:|---:|---|']
    for r in rows: md.append(f"| {r.get('model')} | {r.get('case')} | {r.get('ok')} | {r.get('fps_makespan') or ''} | {r.get('handoff_ms') or ''} | {r.get('status')} |")
    md_path.write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'ok':orchestration_ok,'orchestration_status':summary['orchestration_status'],'evidence_status':evidence_status,'rows':len(rows),'ok_count':ok_count,'failed_count':failed_count,'json':str(json_path),'md':str(md_path),'analysis_tag':tag}, indent=2))
    return 0 if orchestration_ok else 3
if __name__=='__main__': raise SystemExit(main())
