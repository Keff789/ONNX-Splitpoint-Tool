#!/usr/bin/env python3
"""Two fresh normal cache lookups from the actual H8 MobileNet b056 receipt.

The existing deferred build request is the source of recipe arguments; the
separate saved profile supplies compute selection. No diagnostic request ZIP,
SDK, GPU probe, model build, or publication of a private HEF is needed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
MODEL, CASE = 'mobilenet_v3_large', 'b056'
ENV_KEYS = {'ONNX_SPLITPOINT_HAILO_CACHE_ENABLED', 'ONNX_SPLITPOINT_HAILO_CACHE_ROOT',
    'ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY', 'ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE',
    'ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB', 'ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST'}


def _read(path):
    if path.is_symlink() or not path.is_file(): raise ValueError('missing_or_unsafe_input:' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def request_arguments(source, profile, output):
    from onnx_splitpoint_tool.hailo_backend import _load_valid_hailo_receipt
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    source = Path(source).expanduser().resolve(strict=True)
    directory = source / 'models' / MODEL / 'benchmark_set/legacy_suite' / CASE / 'hailo/hailo8/part1'
    deferred = directory / 'deferred_hailo_build.json'
    request = _read(deferred)
    if request.get('schema') != 'onnx-splitpoint/deferred-hailo-build/v1':
        raise ValueError('deferred_request_schema_invalid')
    model = Path(str(request.get('source_onnx') or '')).expanduser()
    if not model.is_file() or model.is_symlink() or not model.resolve().is_relative_to(source):
        raise ValueError('bound_part1_source_missing_or_outside_run')
    source_sha = _sha(model)
    if source_sha != str(request.get('source_onnx_sha256') or '').removeprefix('sha256:'):
        raise ValueError('bound_part1_source_hash_mismatch')
    hef = directory / 'compiled.hef'
    receipt = _load_valid_hailo_receipt(hef)
    if not receipt or receipt.get('hw_arch') != 'hailo8' or receipt.get('diagnostic_only') is True:
        raise ValueError('valid_productive_hailo8_receipt_missing')
    if receipt.get('source_onnx_sha256') != source_sha:
        raise ValueError('receipt_part1_source_binding_mismatch')
    kwargs = dict(request.get('kwargs') or {})
    context = kwargs.get('build_evidence_context') or {}
    if kwargs.get('hw_arch') != 'hailo8' or context.get('model_id') != MODEL or context.get('boundary') != 56:
        raise ValueError('deferred_request_model_family_boundary_mismatch')
    if kwargs.get('force') is not False or kwargs.get('calib_count') != 500 or kwargs.get('opt_level') != 1 or kwargs.get('calib_batch_size') != 8:
        raise ValueError('deferred_recipe_outside_fixed_B500_opt1_contract')
    saved, _ = load_runtime_profile_snapshot(str(profile))
    compute = saved.get('hailo_build', {}).get('compute_by_family') or {}
    if compute.get('hailo8', {}).get('device') != 'gpu':
        raise ValueError('saved_profile_hailo8_gpu_preference_required')
    kwargs.update(onnx_path=str(model), outdir=str(output), cache_only=True, force=False,
                  publish_artifacts=True, keep_artifacts=False, compute_device=None,
                  compute_by_family=compute)
    environment = {k: str(v) for k, v in (request.get('environment') or {}).items() if k in ENV_KEYS}
    bound = {'deferred_request': str(deferred), 'deferred_request_sha256': _sha(deferred),
             'source_model': str(model), 'source_onnx_sha256': source_sha, 'hef': str(hef),
             'hef_sha256': _sha(hef), 'receipt': receipt, 'profile': str(profile), 'profile_sha256': _sha(Path(profile)),
             'compute_selection': compute.get('hailo8')}
    return kwargs, environment, bound


def worker(source, profile, output):
    from onnx_splitpoint_tool.cache_verify_policy import ARTIFACT_POLICY_ENV, CACHE_VERIFY_ONLY
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    observations = {'sdk_import_attempts': [], 'child_process_attempts': []}
    def forbid(event, args):
        if event == 'import' and str(args[0]).split('.')[0] in {'hailo_sdk_client', 'tensorflow'}:
            observations['sdk_import_attempts'].append(str(args[0])); raise RuntimeError('SDK_import_forbidden_on_reuse')
        if event == 'subprocess.Popen':
            observations['child_process_attempts'].append(str(args[0])); raise RuntimeError('child_dispatch_forbidden_on_reuse')
    sys.addaudithook(forbid)
    result = {'status': 'incomplete', 'controller_pid': os.getpid(), 'model_build': 'not_run',
              'hardware_execution': 'not_run', 'observations': observations}
    try:
        kwargs, environment, bound = request_arguments(source, profile, output / 'materialized')
        os.environ.update(environment); os.environ[ARTIFACT_POLICY_ENV] = CACHE_VERIFY_ONLY
        from onnx_splitpoint_tool.hailo_backend import hailo_build_hef_auto, _load_valid_hailo_receipt
        actual = hailo_build_hef_auto(**kwargs)
        info = actual.calib_info or {}
        receipt = _load_valid_hailo_receipt(Path(actual.hef_path)) if actual.ok and actual.hef_path else None
        if _sha(Path(bound['deferred_request'])) != bound['deferred_request_sha256'] or _sha(Path(profile)) != bound['profile_sha256'] or _sha(Path(bound['source_model'])) != bound['source_onnx_sha256'] or _sha(Path(bound['hef'])) != bound['hef_sha256']:
            raise ValueError('source_changed_during_reuse')
        result.update(cache_hit=info.get('cache_hit') is True, compiler_dispatch_count=info.get('compiler_dispatch_count'),
                      exact_receipt_match=receipt == bound['receipt'], source_inputs_unchanged=True,
                      cache_key=(receipt or {}).get('cache_key'), hef_sha256=(receipt or {}).get('hef_sha256'),
                      compute_selection=bound['compute_selection'], error=actual.error,
                      failure_kind=actual.failure_kind, unsupported_reason=actual.unsupported_reason)
        if (actual.ok and result['cache_hit'] and result['compiler_dispatch_count'] == 0
                and result['exact_receipt_match'] and not any(observations.values())):
            result['status'] = 'cache_hit_pass'
    except Exception as exc:
        result['error'] = type(exc).__name__ + ':' + str(exc)
    _write(output / 'result.json', result)
    return 0 if result['status'] == 'cache_hit_pass' else 2


def collect(source, profile, output, python):
    from hailo_model_build_probe_v27934 import interlock
    from deepx_full_workflow_smoke_v27930 import run_transport
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'incomplete', 'runs': [], 'model_build': 'not_run', 'hardware_execution': 'not_run'}
    try:
        with interlock():
            for number in (1, 2):
                directory = output / f'lookup_{number}'
                with (output / f'lookup_{number}.log').open('w') as log:
                    completed = run_transport([str(python), '-I', '-B', str(Path(__file__).resolve()), '--source-run', str(source),
                        '--profile', str(profile), '--worker', str(directory)], timeout=180, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                row = _read(directory / 'result.json') if (directory / 'result.json').is_file() else {'status': 'worker_result_missing'}
                row['returncode'] = completed.returncode
                report['runs'].append(row)
                if completed.returncode or row['status'] != 'cache_hit_pass': break
            if (len(report['runs']) == 2 and len({r.get('controller_pid') for r in report['runs']}) == 2
                    and all(r.get('status') == 'cache_hit_pass' for r in report['runs'])
                    and report['runs'][0]['hef_sha256'] == report['runs'][1]['hef_sha256']
                    and report['runs'][0]['cache_key'] == report['runs'][1]['cache_key']):
                report['status'] = 'two_fresh_process_hits_pass'
    except Exception as exc:
        report['error'] = type(exc).__name__ + ':' + str(exc)
    _write(output / 'summary.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    args = parser.parse_args(argv)
    return worker(args.source_run, args.profile, args.worker)


if __name__ == '__main__': raise SystemExit(main())
