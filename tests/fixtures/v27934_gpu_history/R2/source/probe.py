#!/usr/bin/env python3
"""Hailo10H diagnostic only: unchanged FIX1 GPU worker with local Triton ptxas.

No installation, no model build, no changes to installed venvs, profiles or
/usr/local/cuda. The temporary toolkit is NOT a production CUDA installation.
"""
from __future__ import annotations
import argparse
import ctypes
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import collect as previous


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('json_object_required:' + str(path))
    return value


def file_info(path: Path) -> dict:
    stat = path.stat()
    return {'path': str(path), 'resolved': str(path.resolve(strict=True)),
            'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def locate(venv: Path) -> dict:
    """Inspect only the requested venv; do not import Triton/torch/tensorflow."""
    python = venv / 'bin/python'
    if not (venv / 'pyvenv.cfg').is_file() or not os.access(python, os.X_OK):
        raise RuntimeError('expected_hailo10_venv_missing:' + str(venv))
    sites = sorted((venv / 'lib').glob('python*/site-packages'))
    roots = [s / 'triton/backends/nvidia' for s in sites
             if (s / 'triton/backends/nvidia/bin/ptxas').is_file()]
    roots = list(dict.fromkeys(r.resolve() for r in roots))
    if len(roots) != 1:
        raise RuntimeError('triton_ptxas_missing_or_ambiguous:' + str([str(r) for r in roots]))
    root = roots[0]
    root.relative_to(venv.resolve())  # Never silently borrow from another venv.
    ptxas = root / 'bin/ptxas'
    if not os.access(ptxas, os.X_OK):
        raise RuntimeError('triton_ptxas_not_executable:' + str(ptxas))
    candidates = [root / 'lib/libdevice.10.bc', root / 'nvvm/libdevice/libdevice.10.bc',
                  root / 'libdevice/libdevice.10.bc']
    devices = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(devices) != 1:
        raise RuntimeError('triton_libdevice_missing_or_ambiguous;checked=' + ','.join(map(str, candidates)))
    libdevice = devices[0]
    libdevice.relative_to(venv.resolve())
    if libdevice.stat().st_size == 0:
        raise RuntimeError('empty_libdevice:' + str(libdevice))
    result = {'python': str(python), 'ptxas': file_info(ptxas), 'libdevice': file_info(libdevice),
              'libdevice_candidates': [str(p) for p in candidates],
              'note': 'Both compiler components are from the installed Triton package; XLA compatibility is under test.'}
    result['packages'] = {}
    for dist in importlib.metadata.distributions(path=[str(s) for s in sites]):
        name = dist.metadata.get('Name', '').lower().replace('_', '-')
        if name in ('hailo-dataflow-compiler', 'tensorflow', 'torch', 'triton'):
            result['packages'][name] = dist.version
    return result


def stage_toolkit(root: Path, ptxas: Path, libdevice: Path, trace: Path) -> None:
    """A logging forwarder never changes the arguments passed to ptxas."""
    (root / 'bin').mkdir(parents=True)
    (root / 'nvvm/libdevice').mkdir(parents=True)
    (root / 'nvvm/libdevice/libdevice.10.bc').symlink_to(libdevice)
    code = r'''import json, os, re, subprocess, sys, time
from pathlib import Path
REAL = __REAL__
TRACE = __TRACE__
args = sys.argv[1:]
row = {'phase': os.environ.get('HAILO_R2_TRACE_PHASE', 'unknown'),
       'pid': os.getpid(), 'real_ptxas': REAL, 'args': args, 'start_ns': time.time_ns()}
row['kind'] = 'version_or_help'
for arg in args:
    if not arg.startswith('-') and Path(arg).is_file():
        with Path(arg).open('rb') as f: head = f.read(131072).decode('utf-8', 'replace')
        target = re.search(r'(?m)^\s*\.target\s+(sm_\d+[a-z]?)', head)
        version = re.search(r'(?m)^\s*\.version\s+([0-9.]+)', head)
        if not target or not version: continue
        row['kind'] = 'assemble'
        row['ptx_target'] = target.group(1)
        row['ptx_version'] = version.group(1)
        break
out = None
for i, arg in enumerate(args):
    if arg in ('-o', '--output-file') and i + 1 < len(args): out = Path(args[i + 1])
    elif arg.startswith(('--output-file=', '-o=')): out = Path(arg.split('=', 1)[1])
def log(event):
    raw = (json.dumps({**row, 'event': event}, ensure_ascii=True) + '\n').encode('utf-8')
    fd = os.open(TRACE, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o600)
    try: os.write(fd, raw)
    finally: os.close(fd)
log('start')
try:
    rc = subprocess.run([REAL, *args], check=False).returncode
    row['returncode'] = rc
    row['output_created_nonempty'] = bool(out and out.is_file() and out.stat().st_size > 0)
    row['end_ns'] = time.time_ns()
    log('finish')
except Exception as exc:
    row.update(returncode=127, error=type(exc).__name__ + ': ' + str(exc))
    log('finish')
    raise
raise SystemExit(rc if rc >= 0 else 128 - rc)
'''
    code = code.replace('__REAL__', repr(str(ptxas))).replace('__TRACE__', repr(str(trace)))
    forward = root / 'ptxas_forwarder.py'
    forward.write_text(code, encoding='utf-8')
    launcher = '#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -I -S -B ' + shlex.quote(str(forward)) + ' "$@"\n'
    (root / 'bin/ptxas').write_text(launcher, encoding='utf-8')
    (root / 'bin/ptxas').chmod(0o700)


def make_env(parent: dict, venv: Path, work: Path, toolkit: Path, gpu: str) -> tuple[dict, dict]:
    if any(c.isspace() for c in str(toolkit)):
        raise RuntimeError('probe_toolkit_path_must_not_contain_whitespace')
    env, detail = previous.child_env(parent, venv, work, gpu)
    old_tokens = shlex.split(env.get('XLA_FLAGS', ''))
    tokens = []
    i = 0
    while i < len(old_tokens):
        token = old_tokens[i]
        if token == '--xla_gpu_cuda_data_dir':
            if i + 1 >= len(old_tokens): raise ValueError('incomplete_inherited_XLA_FLAGS')
            i += 2; continue
        if token.startswith('--xla_gpu_cuda_data_dir='):
            i += 1; continue
        tokens.append(token); i += 1
    tokens.append('--xla_gpu_cuda_data_dir=' + str(toolkit))
    changes = {'XLA_FLAGS': shlex.join(tokens), 'CUDA_HOME': str(toolkit), 'CUDA_PATH': str(toolkit),
               'PATH': str(toolkit / 'bin') + os.pathsep + env['PATH'],
               'HAILO_R2_TRACE_PHASE': 'compute'}
    env.update(changes)
    detail.pop('cuda_toolkit_not_changed', None)
    detail.update({'diagnostic_toolchain_selection': changes,
                   'parent_toolchain_environment': {k: parent.get(k) for k in ('XLA_FLAGS', 'CUDA_HOME', 'CUDA_PATH', 'PATH')},
                   'installed_venv_modified': False, 'system_toolkit_selection_modified': False,
                   'LD_LIBRARY_PATH_unchanged': env.get('LD_LIBRARY_PATH') == parent.get('LD_LIBRARY_PATH')})
    return env, detail


def provenance(trace: Path) -> dict:
    rows = []
    errors = []
    if trace.is_file():
        for i, line in enumerate(trace.read_text(encoding='utf-8').splitlines(), 1):
            try: rows.append(json.loads(line))
            except ValueError: errors.append('invalid_trace_line:' + str(i))
    completed = [r for r in rows if r.get('event') == 'finish' and r.get('phase') == 'compute'
                 and r.get('kind') == 'assemble']
    accepted = [r for r in completed if r.get('returncode') == 0 and r.get('output_created_nonempty')
                and r.get('ptx_target') in ('sm_60', 'sm_61')]
    return {'trace_parse_errors': errors, 'compute_assembly_count': len(completed),
            'successful_pascal_compute_assemblies': len(accepted),
            'observed_ptx_targets': sorted({r.get('ptx_target', '') for r in completed}),
            'observed_ptx_versions': sorted({r.get('ptx_version', '') for r in completed}),
            'verified': bool(accepted) and not errors}


def finish_status(data: dict, process: dict, proof: dict) -> str:
    if not previous.valid_pass(data, process, 'hailo10h'):
        return 'compute_failed'
    gpus = data.get('gpus', [])
    if len(gpus) != 1 or list(gpus[0].get('compute_capability', [])) != [6, 1]:
        return 'compute_pass_unexpected_gpu'
    return 'diagnostic_pass' if proof.get('verified') else 'compute_pass_toolchain_unproven'


def archive(output: Path) -> Path:
    names = ['summary.json', 'assets.json', 'environment.json', 'ptxas_version.log',
             'ptxas_version_process.json', 'sm61_preflight.log', 'sm61_preflight_process.json',
             'ptxas_invocations.jsonl', 'compute/console.log', 'compute/gpu_compute_result.json',
             'compute/process_result.json']
    names += ['source/' + name for name in ('probe.py', 'collect.py', 'worker.py', 'README.md')]
    target = output.with_suffix('.zip')
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in names:
            p = output / name
            if p.is_file() and not p.is_symlink(): z.write(p, name)
    return target


def execute(args: argparse.Namespace) -> int:
    if sys.platform != 'linux': raise RuntimeError('Linux_required')
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('subreaper_unavailable')
    with previous.interlock(args.lock.expanduser()):
        args.output_parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = Path(tempfile.mkdtemp(prefix='hailo10_xla_toolchain_r2_' + stamp + '_', dir=args.output_parent)).resolve()
        work = output / 'compute'; work.mkdir()
        scratch = output / '_temporary'; scratch.mkdir()
        toolkit = scratch / 'cuda'
        trace = output / 'ptxas_invocations.jsonl'
        report = {'schema': 'hailo10-xla-toolchain-probe-r2', 'status': 'running',
                  'diagnostic_only': True, 'model_build': 'NOT_RUN', 'package_installations': False,
                  'tool_implementation_modified': False, 'system_toolkit_selection_modified': False,
                  'model_acceptance': 'NOT_EVALUATED_BY_DIAGNOSTIC', 'scope': 'single Hailo10H DFC venv; synthetic GPU/XLA only'}
        print('SMOKE_STARTED=YES\nEVIDENCE_DIRECTORY=' + str(output), flush=True)
        assets = None
        try:
            (output / 'source').mkdir()
            for name in ('probe.py', 'collect.py', 'worker.py', 'README.md'):
                (output / 'source' / name).write_bytes((ROOT / name).read_bytes())
            assets = locate(args.venv)
            previous.write_json(output / 'assets.json', assets)
            print('PTXAS_CANDIDATE=' + assets['ptxas']['resolved'], flush=True)
            print('LIBDEVICE_CANDIDATE=' + assets['libdevice']['resolved'], flush=True)
            stage_toolkit(toolkit, Path(assets['ptxas']['resolved']), Path(assets['libdevice']['resolved']), trace)
            env, detail = make_env(os.environ, args.venv, work, toolkit, args.gpu)
            previous.write_json(output / 'environment.json', detail)
            pre_env = {**env, 'HAILO_R2_TRACE_PHASE': 'preflight'}
            version = previous.run_command([str(toolkit / 'bin/ptxas'), '--version'], output, pre_env, 20,
                                           log_name='ptxas_version.log')
            previous.write_json(output / 'ptxas_version_process.json', version)
            version_text = (output / 'ptxas_version.log').read_text(encoding='utf-8', errors='replace')
            report['ptxas_version_text'] = version_text
            if version['returncode'] != 0 or not version['cleanup_complete'] or 'release 12.' not in version_text:
                raise RuntimeError('expected_cuda12_ptxas_version_not_confirmed')
            source = scratch / 'sm61.ptx'; cubin = scratch / 'sm61.cubin'
            source.write_text('.version 7.0\n.target sm_61\n.address_size 64\n.visible .entry diagnostic_noop() { ret; }\n')
            check = previous.run_command([str(toolkit / 'bin/ptxas'), '-arch=sm_61', str(source), '-o', str(cubin)],
                                         output, pre_env, 20, log_name='sm61_preflight.log')
            previous.write_json(output / 'sm61_preflight_process.json', check)
            if (check['returncode'] != 0 or check['timed_out'] or not check['cleanup_complete']
                    or check.get('unexpected_live_processes_after_worker_exit')
                    or not cubin.is_file() or cubin.stat().st_size == 0):
                raise RuntimeError('candidate_sm61_preflight_failed')
            print('SM61_PREFLIGHT=PASS\nCOMPUTE_START=hailo10h; unchanged FIX1 worker', flush=True)
            proc = previous.run_command([str(args.venv / 'bin/python'), '-I', '-B', str(ROOT / 'worker.py'),
                                        '--output', str(work), '--expected-venv', str(args.venv), '--family', 'hailo10h'],
                                       work, env, args.timeout)
            previous.write_json(work / 'process_result.json', proc)
            try: data = read_json(work / 'gpu_compute_result.json')
            except (OSError, ValueError): data = {}
            report['compute_process'] = proc
            report['compute_checks'] = data.get('checks', [])
            report['compute_error'] = data.get('error')
            report['ptxas_provenance'] = provenance(trace)
            report['status'] = finish_status(data, proc, report['ptxas_provenance'])
            for row in data.get('checks', []):
                print('CHECK=' + str(row.get('name')) + ':' + str(row.get('status')), flush=True)
                if row.get('error'): print('CHECK_ERROR=' + row['error'], flush=True)
            print('PTXAS_COMPUTE_ASSEMBLIES=' + str(report['ptxas_provenance']['compute_assembly_count']), flush=True)
        except KeyboardInterrupt:
            report.update(status='interrupted', error='KeyboardInterrupt')
        except Exception as exc:
            report.update(status='setup_failed', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
            print('ERROR=' + report['error'], flush=True)
        finally:
            if assets:
                try:
                    report['compiler_files_metadata_unchanged'] = all(
                        file_info(Path(assets[k]['path'])) == assets[k] for k in ('ptxas', 'libdevice'))
                except OSError as exc:
                    report['compiler_files_metadata_unchanged'] = False
                    report['source_recheck_error'] = str(exc)
                if report['compiler_files_metadata_unchanged'] is False:
                    report['status'] = 'source_changed_during_probe'
            # Remove only private directories created by this invocation; never unlink a shared lock or installed file.
            cleanup_errors = []
            for p in [scratch, work / 'tmp', work / 'cuda_cache', work / 'xdg_cache']:
                try:
                    if p.exists(): shutil.rmtree(p)
                except OSError as exc: cleanup_errors.append(str(exc))
            report['temporary_cleanup_complete'] = not cleanup_errors
            report['temporary_cleanup_errors'] = cleanup_errors
            if cleanup_errors and report['status'] == 'diagnostic_pass': report['status'] = 'cleanup_failed'
            previous.write_json(output / 'summary.json', report)
            path = archive(output)
            print('XLA_TOOLCHAIN_STATUS=' + report['status'], flush=True)
            print('MODEL_BUILD=NOT_RUN\nMODEL_ACCEPTANCE=NOT_EVALUATED_BY_DIAGNOSTIC\nDIAGNOSTIC_ZIP=' + str(path), flush=True)
        return 0 if report['status'] == 'diagnostic_pass' else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path.home() / '.onnx_splitpoint_tool/hailo/venv_hailo10')
    parser.add_argument('--output-parent', type=Path, default=Path.home() / 'Downloads')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lock', type=Path, default=Path.home() / '.onnx_splitpoint_tool/locks/workflow_platform_interlock.lock', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.venv = args.venv.expanduser().absolute()
    args.output_parent = args.output_parent.expanduser().absolute()
    if not 10 <= args.timeout <= 600: parser.error('--timeout: 10..600 seconds')
    if not re.fullmatch(r'[0-9]+|GPU-[A-Za-z0-9-]+', args.gpu): parser.error('select one GPU, not -1 or a list')
    try: return execute(args)
    except previous.BusyError:
        print('SMOKE_STARTED=NO\nREASON=workflow_or_platform_operation_active\nDo not delete the lock.', flush=True)
        return 3
    except Exception as exc:
        print('SMOKE_STARTED=NO\nERROR=' + type(exc).__name__ + ': ' + str(exc), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
