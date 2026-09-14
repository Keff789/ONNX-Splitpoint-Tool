#!/usr/bin/env python3
"""DeepX -> TensorRT native FIFO preparation/probe.

DeepX requires a producer implementation based on the DeepX runtime API.  This
script prepares the contract and validates the TensorRT consumer side so we can
identify which cases are ready once the DeepX producer adapter is implemented.
It deliberately does not fake E2E native FIFO numbers.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _case_id(x: str) -> str:
    s=str(x)
    return s if s.startswith('b') else f'b{int(s):03d}'


def _load_json(p: Path) -> Any:
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return None


def _run(cmd: list[str], timeout: float | None=None, env: dict[str, str] | None=None) -> dict[str, Any]:
    t0=time.time(); p=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
    return {'cmd': cmd, 'rc': p.returncode, 'elapsed_s': time.time()-t0, 'stdout_tail': p.stdout[-8000:], 'stderr_tail': p.stderr[-8000:]}


def _python_can_import(py: str, module: str, *, env: dict[str, str] | None = None) -> bool:
    try:
        p = subprocess.run([py, '-c', f'import {module}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20, env=env)
        return p.returncode == 0
    except Exception:
        return False


def _candidate_engine_pythons(requested: str) -> list[tuple[str, dict[str, str] | None, str]]:
    out: list[tuple[str, dict[str, str] | None, str]] = []
    base_env = os.environ.copy()
    def add(py: str | None, env: dict[str, str] | None, label: str):
        if not py:
            return
        p = shutil.which(py) or py
        if p and p not in [x[0] for x in out]:
            out.append((p, env, label))
    if requested and requested != 'auto':
        add(requested, base_env, 'requested')
        return out
    add(sys.executable, base_env, 'current')
    env_user = base_env.copy(); env_user.pop('PYTHONNOUSERSITE', None)
    add(sys.executable, env_user, 'current_user_site_enabled')
    add('/usr/bin/python3', env_user, 'system_python_user_site_enabled')
    add('/usr/bin/python', env_user, 'system_python_user_site_enabled')
    add('python3', env_user, 'python3_user_site_enabled')
    return out


def _build_native_trt_part2(bs: Path, case: str, precision: str, timeout: float, requested_python: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for py, env, label in _candidate_engine_pythons(requested_python):
        can_onnx = _python_can_import(py, 'onnx', env=env)
        step_prefix = {'engine_build_python': py, 'engine_build_python_label': label, 'engine_build_python_can_import_onnx': can_onnx}
        if not can_onnx:
            steps.append({'name':'build_native_trt_part2_skip_python_missing_onnx', **step_prefix, 'rc':127, 'elapsed_s':0.0, 'stdout_tail':'', 'stderr_tail':'python cannot import onnx'})
            continue
        cmd=[py, str(ROOT/'scripts'/'native_trt_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', case, '--variants', 'part2', '--precision', precision, '--run-smoke', '--iterations', '100', '--workspace-mb', '4096', '--workspace-mode', 'auto']
        st=_run(cmd, timeout=timeout, env=env); st.update(step_prefix); st['name']='build_native_trt_part2'; steps.append(st)
        if st.get('rc') == 0:
            break
    return steps


def _find_deepx_artifacts(case_dir: Path) -> list[str]:
    roots = [case_dir/'deepx', case_dir/'deepx_m1', case_dir/'dx_m1']
    exts = {'.dxnn','.bin','.json','.npy','.npz','.so'}
    out=[]
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob('*'):
            if p.is_file() and (p.suffix.lower() in exts or 'deepx' in p.name.lower() or 'dx' in p.name.lower()):
                out.append(str(p))
    return sorted(out)


def _find_contract(bs: Path, case: str) -> dict[str, Any]:
    for p in [bs/'io_contracts'/case/'io_contract.json', bs/case/'io_contract.json']:
        j=_load_json(p)
        if isinstance(j, dict): return j
    summ=_load_json(bs/'io_contracts'/'summary.json')
    if isinstance(summ, dict):
        for c in summ.get('cases', []) or []:
            if str(c.get('case_id')) == case:
                return c
    return {}


def _native_trt_engine(bs: Path, case: str, precision: str) -> Path:
    name = 'part2_uint8_cast_fp16.engine' if precision == 'uint8_cast_fp16' else ('part2_uint8_dequant_fp16.engine' if precision == 'uint8_dequant_fp16' else f'part2_{precision}.engine')
    return bs/'native_trt'/case/'part2'/precision/name


def main() -> int:
    ap=argparse.ArgumentParser(description='Probe DeepX->NativeTRT FIFO readiness for one BenchmarkSet case')
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--precision', default='uint8_cast_fp16', choices=['uint8_cast_fp16','uint8_dequant_fp16','fp16'])
    ap.add_argument('--build-missing-engine', action='store_true')
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--engine-build-python', default='auto', help='Python used to build missing native TRT engines. auto tries current env and system Python with user-site enabled.')
    ns=ap.parse_args()
    bs=Path(ns.benchmark_set).expanduser().resolve(); case=_case_id(ns.case); case_dir=bs/case
    out_dir=bs/'native_pipeline'/case/'deepx_to_trt'/ns.precision; out_dir.mkdir(parents=True, exist_ok=True)
    report_path=out_dir/'deepx_native_fifo_probe.json'
    artifacts=_find_deepx_artifacts(case_dir)
    engine=_native_trt_engine(bs, case, ns.precision)
    report={'ok': False, 'mode': 'deepx_native_fifo_producer_probe', 'native_fifo_e2e_implemented': False,
            'note': 'DeepX producer adapter is not implemented yet. This probe records artifacts and validates the TensorRT consumer side.',
            'benchmark_set': str(bs), 'case': case, 'precision': ns.precision, 'deepx_artifacts': artifacts,
            'part2_engine': str(engine) if engine.exists() else '', 'steps': [], 'contract': _find_contract(bs, case)}
    # Runtime import probe.
    probe_code="""\nmods=[]\nfor m in ('dx_engine','dx_com','dxrt','deepx'):\n    try:\n        __import__(m); mods.append(m)\n    except Exception:\n        pass\nprint(','.join(mods))\nraise SystemExit(0 if mods else 5)\n"""
    p=subprocess.run([sys.executable, '-c', probe_code], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report['deepx_python_probe']={'rc': p.returncode, 'modules': [x for x in p.stdout.strip().split(',') if x], 'stderr_tail': p.stderr[-4000:]}
    if not engine.exists() and ns.build_missing_engine:
        report['steps'].extend(_build_native_trt_part2(bs, case, ns.precision, ns.timeout, ns.engine_build_python))
    report['part2_engine'] = str(engine) if engine.exists() else ''
    report['producer_ready'] = bool(artifacts and report['deepx_python_probe']['modules'])
    report['consumer_ready'] = bool(engine.exists())
    report['ok'] = bool(report['producer_ready'] and report['consumer_ready'])
    if not report['ok']:
        if not artifacts:
            report['next_action'] = 'build or copy DeepX part1 runtime artifacts for this case'
        elif not report['deepx_python_probe']['modules']:
            report['next_action'] = 'run in DeepX runtime environment or add DeepX site-packages to SPLITPOINT_EXTRA_SITES'
        elif not engine.exists():
            report['next_action'] = 'build native TensorRT part2 engine with --build-missing-engine'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'ok': report['ok'], 'producer_ready': report['producer_ready'], 'consumer_ready': report['consumer_ready'], 'report': str(report_path), 'next_action': report.get('next_action','')}, indent=2))
    return 0 if report['ok'] else 4

if __name__ == '__main__':
    raise SystemExit(main())
