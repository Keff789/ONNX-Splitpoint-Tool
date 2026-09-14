#!/usr/bin/env python3
"""Two fresh, compiler-free lookups of the existing MobileNet CPU artifact.

The v34 diagnostic request is an input inventory here, never a request to
publish its private GPU artifact. Existing normal graph preparation, cache
keys, receipt validation and read-only ArtifactStore lookup remain authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from onnx_splitpoint_tool.hailo_model_diagnostics_v27934 import (
    check_identity, file_identity, validate_request, write_json,
)
from hailo_model_build_probe_v27934 import interlock
from deepx_full_workflow_smoke_v27930 import run_transport


def reuse_arguments(request, work):
    """Scope the existing verified inventory to the normal Full cache lookup."""
    receipt = validate_request(request)
    recipe = receipt['cache_payload']
    return {
        'onnx_path': request['source_onnx']['path'],
        'outdir': str(Path(work) / 'materialized'),
        'backend': 'venv', 'hw_arch': receipt['hw_arch'],
        'net_name': receipt['net_name'], 'net_input_shapes': recipe['net_input_shapes'],
        'fixup': True, 'add_conv_defaults': True,
        'disable_rt_metadata_extraction': recipe['disable_rt_metadata_extraction'],
        'opt_level': recipe['optimization_level'], 'calib_dir': request['calibration_dir'],
        'calib_count': recipe['requested_calibration_count'],
        'calib_batch_size': recipe['calibration_batch_size'],
        'extra_model_script': recipe['extra_model_script'],
        'start_node_names': recipe['start_nodes'], 'end_node_names': recipe['end_nodes'],
        'task': 'classification', 'preprocessing_contract': receipt['preprocessing_contract'],
        'force': False, 'cache_only': True, 'keep_artifacts': False,
        'publish_artifacts': True, 'compute_device': 'gpu',
        'gpu_selector': request['gpu_selector'],
        'wsl_venv_activate': str(Path(request['venv_python']).parent / 'activate'),
    }


def worker(request_path, work):
    """One new controller; writes only its new local output and attempt data."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    from onnx_splitpoint_tool.cache_verify_policy import ARTIFACT_POLICY_ENV, CACHE_VERIFY_ONLY
    from onnx_splitpoint_tool.release_identity import VERSION, BUILD_ID
    observations = {'sdk_import_attempts': [], 'child_process_attempts': []}

    def no_compiler(event, args):
        if event == 'import' and str(args[0]).split('.')[0] in {'hailo_sdk_client', 'tensorflow'}:
            observations['sdk_import_attempts'].append(str(args[0]))
            raise RuntimeError('reuse_probe_forbids_sdk_import:' + str(args[0]))
        if event == 'subprocess.Popen':
            observations['child_process_attempts'].append(str(args[0]))
            raise RuntimeError('reuse_probe_forbids_child_dispatch')

    # The normal read-only policy is the primary compiler fence. Auditing is
    # additional executable evidence and blocks any accidental SDK/GPU probe.
    sys.addaudithook(no_compiler)
    report = {'status': 'incomplete', 'controller_pid': os.getpid(),
              'version': VERSION, 'build_id': BUILD_ID,
              'model_build': 'NOT_RUN', 'hardware_execution': 'NOT_RUN',
              'quality_acceptance': 'NOT_EVALUATED',
              'productive_force': False, **observations}
    try:
        request_identity = file_identity(request_path)
        request = json.loads(Path(request_path).read_text(encoding='utf-8'))
        arguments = reuse_arguments(request, work)
        expected = request['cpu_build_receipt']
        os.environ[ARTIFACT_POLICY_ENV] = CACHE_VERIFY_ONLY
        os.environ['ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY'] = expected['cache_payload']['integrity']
        os.environ['ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE'] = expected['calibration_storage']
        cap = expected['calibration_memory_cap_bytes']
        if type(cap) is not int or cap <= 0 or cap % (1024 * 1024):
            raise ValueError('reuse_probe_recipe_memory_cap_not_supported')
        os.environ['ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB'] = str(cap // (1024 * 1024))
        if request.get('calibration_manifest'):
            os.environ['ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST'] = request['calibration_manifest']['path']
        else:
            os.environ.pop('ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST', None)
        from onnx_splitpoint_tool.hailo_backend import hailo_build_hef_auto, _load_valid_hailo_receipt
        result = hailo_build_hef_auto(**arguments)
        info = result.calib_info or {}
        actual = _load_valid_hailo_receipt(Path(result.hef_path)) if result.ok and result.hef_path else None
        receipt_matches = bool(actual and actual == expected)
        for name in ('cpu_hef', 'cpu_receipt', 'source_onnx', 'compiler_onnx'):
            check_identity(request[name])
        check_identity(request_identity)
        report.update(
            input_request=request_identity,
            cache_hit=info.get('cache_hit') is True,
            compiler_dispatch_count=info.get('compiler_dispatch_count'),
            exact_cpu_receipt_match=receipt_matches,
            original_inputs_byte_unchanged=True,
            cache_key=(actual or {}).get('cache_key'),
            source_onnx_sha256=(actual or {}).get('source_onnx_sha256'),
            compiler_onnx_sha256=(actual or {}).get('compiler_onnx_sha256'),
            hef_sha256=(actual or {}).get('hef_sha256'),
            cache_source=info.get('cache_source') or info.get('source') or ('local_exact_cache' if info.get('cache_hit') else None),
            failure_kind=result.failure_kind, unsupported_reason=result.unsupported_reason,
            error=result.error,
        )
        if (result.ok is True and info.get('cache_hit') is True
                and info.get('compiler_dispatch_count') == 0 and receipt_matches
                and not observations['sdk_import_attempts'] and not observations['child_process_attempts']):
            report['status'] = 'cache_hit_pass'
        elif result.ok is True:
            report['error'] = 'reuse_probe_hit_or_exact_baseline_receipt_not_proven'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['cancelled'] = isinstance(exc, KeyboardInterrupt)
    finally:
        write_json(work / 'result.json', report)
    return 0 if report['status'] == 'cache_hit_pass' else 2


def collect(request_path, output_parent, timeout):
    parent = Path(output_parent).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='v280_hailo_reuse_', dir=parent))
    report = {'status': 'incomplete', 'model_build': 'NOT_RUN',
              'hardware_execution': 'NOT_RUN', 'gpu_publication': 'NOT_RUN',
              'quality_acceptance': 'NOT_EVALUATED', 'runs': []}
    rc = 2
    try:
        with interlock():
            request_path = Path(request_path).expanduser().resolve(strict=True)
            report['input_request'] = file_identity(request_path)
            for index in (1, 2):
                directory = work / f'lookup_{index}'
                log_path = work / f'lookup_{index}.log'
                with log_path.open('w', encoding='utf-8') as log:
                    completed = run_transport([sys.executable, '-I', '-B', str(Path(__file__).resolve()),
                         '--request', str(request_path), '--worker', str(directory)],
                         timeout=timeout, stdout=log, stderr=subprocess.STDOUT,
                         env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'ORT_DISABLE_TELEMETRY': '1'})
                row = json.loads((directory / 'result.json').read_text(encoding='utf-8')) if (directory / 'result.json').is_file() else {'status': 'result_missing'}
                row['returncode'] = completed.returncode
                report['runs'].append(row)
                if completed.returncode != 0 or row['status'] != 'cache_hit_pass':
                    break
            check_identity(report['input_request'])
            if (len(report['runs']) == 2
                    and len({row.get('controller_pid') for row in report['runs']}) == 2
                    and all(row['returncode'] == 0 and row['status'] == 'cache_hit_pass' for row in report['runs'])
                    and all(row.get('input_request') == report['input_request'] for row in report['runs'])
                    and report['runs'][0]['cache_key'] == report['runs'][1]['cache_key']):
                report['status'] = 'two_fresh_process_cache_hits_pass'
                rc = 0
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['cancelled'] = isinstance(exc, KeyboardInterrupt)
    finally:
        write_json(work / 'summary.json', report)
        archive = work.with_suffix('.zip')
        # Fixed evidence allowlist: never HEF, ONNX, HAR, images, arrays or the
        # temporary output directories, including on failed lookup.
        files = [work / 'summary.json', *work.glob('lookup_*.log'), *work.glob('lookup_*/result.json')]
        with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(files):
                if path.is_file():
                    z.write(path, Path(work.name) / path.relative_to(work))
        print('REUSE_PROBE_STATUS=' + report['status'])
        print('MODEL_BUILD=NOT_RUN')
        print('GPU_PUBLICATION=NOT_RUN')
        print('EVIDENCE_ZIP=' + str(archive))
        print('REUSE_PROBE_RC=' + str(rc))
    return rc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, help='Existing v34 MobileNet prepare request.json')
    parser.add_argument('--output-parent', default=str(Path.home() / 'Downloads'))
    parser.add_argument('--timeout', type=int, default=180, help='Bound per fresh cache-lookup process (30–600 seconds)')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 30 <= args.timeout <= 600:
        parser.error('--timeout must be between 30 and 600')
    return worker(args.request, args.worker) if args.worker else collect(args.request, args.output_parent, args.timeout)


if __name__ == '__main__':
    raise SystemExit(main())
