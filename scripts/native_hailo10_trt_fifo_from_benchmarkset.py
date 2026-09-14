#!/usr/bin/env python3
"""Hailo-10H -> TensorRT native FIFO preparation/probe.

This script is intentionally a *producer probe* first.  The Hailo-8 native FIFO
runner is a C++ VStreams+TensorRT hotloop.  Hailo-10H must use the HailoRT
InferModel/run_async style API, so the producer implementation is different.

For a BenchmarkSet case this script verifies the three pieces needed for a
future Hailo10H native FIFO producer:
  1. Hailo10H Part1 HEF exists and can run through the tool's InferModel path.
  2. Native TensorRT Part2 engine exists/can be built.
  3. The projected paper-style cycle is plausible: max(Hailo10 producer, TRT P2).

It does not yet claim E2E native FIFO handoff.  Reports are written so the case
can be selected for the next implementation step.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _case_id(x: str) -> str:
    s = str(x)
    if s.startswith('b'):
        return s
    return f"b{int(s):03d}"


def _case_dir(bs: Path, case: str) -> Path:
    p = bs / case
    if not p.is_dir():
        raise FileNotFoundError(f"case directory not found: {p}")
    return p


def _find_part1_onnx(case_dir: Path, case: str) -> Path | None:
    pats = [f"*part1*{case[1:]}*.onnx", "*part1*.onnx"]
    for pat in pats:
        xs = sorted(case_dir.glob(pat))
        if xs:
            return xs[0]
    return None


def _find_hailo10_hef(case_dir: Path) -> Path | None:
    rels = [
        ('hailo', 'hailo10h', 'part1', 'compiled.hef'),
        ('hailo', 'hailo10', 'part1', 'compiled.hef'),
        ('hailo', 'hailo10n', 'part1', 'compiled.hef'),
        ('hailo', 'hailo15h', 'part1', 'compiled.hef'),
    ]
    for rel in rels:
        p = case_dir.joinpath(*rel)
        if p.is_file():
            return p
    xs = sorted(case_dir.glob('hailo/hailo10*/part1/*.hef')) + sorted(case_dir.glob('hailo/hailo15*/part1/*.hef'))
    return xs[0] if xs else None


def _native_trt_engine(bs: Path, case: str, precision: str) -> Path:
    if precision == 'uint8_cast_fp16':
        name = 'part2_uint8_cast_fp16.engine'
    elif precision == 'uint8_dequant_fp16':
        name = 'part2_uint8_dequant_fp16.engine'
    elif precision == 'fp16':
        name = 'part2_fp16.engine'
    else:
        name = f'part2_{precision}.engine'
    return bs / 'native_trt' / case / 'part2' / precision / name


def _run(cmd: list[str], *, timeout: float | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    t0 = time.time()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
    return {
        'cmd': cmd,
        'rc': p.returncode,
        'elapsed_s': time.time() - t0,
        'stdout_tail': p.stdout[-8000:],
        'stderr_tail': p.stderr[-8000:],
    }


def _python_can_import(py: str, module: str, *, env: dict[str, str] | None = None) -> bool:
    try:
        p = subprocess.run([py, '-c', f'import {module}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20, env=env)
        return p.returncode == 0
    except Exception:
        return False


def _candidate_engine_pythons(requested: str) -> list[tuple[str, dict[str, str] | None, str]]:
    """Return candidate Python interpreters for ONNX graph surgery / TensorRT engine build.

    Hailo-10 environments often run with PYTHONNOUSERSITE=1 so the HailoRT/NumPy
    stack stays stable.  The lightweight ONNX package needed to create uint8-bridge
    ONNX files may still live in the system/user Python.  Native TRT engine builds do
    not require HailoRT, so it is safe to fall back to system Python for this step.
    """
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
    # Current venv with user-site enabled may see onnx from ~/.local if it is intentionally installed there.
    env_user = base_env.copy()
    env_user.pop('PYTHONNOUSERSITE', None)
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
            steps.append({'name': 'build_native_trt_part2_skip_python_missing_onnx', **step_prefix, 'rc': 127, 'elapsed_s': 0.0, 'stdout_tail': '', 'stderr_tail': 'python cannot import onnx'})
            continue
        cmd = [py, str(ROOT/'scripts'/'native_trt_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', case, '--variants', 'part2', '--precision', precision, '--run-smoke', '--iterations', '100', '--workspace-mb', '4096', '--workspace-mode', 'auto']
        st = _run(cmd, timeout=timeout, env=env)
        st.update(step_prefix)
        st['name'] = 'build_native_trt_part2'
        steps.append(st)
        if st.get('rc') == 0:
            break
    return steps


def _parse_trtexec_log(p: Path) -> dict[str, float | None]:
    txt = p.read_text(encoding='utf-8', errors='ignore') if p.exists() else ''
    out: dict[str, float | None] = {'latency_mean_ms': None, 'throughput_qps': None, 'gpu_compute_mean_ms': None, 'h2d_mean_ms': None, 'd2h_mean_ms': None}
    # TensorRT log formats differ; keep regex permissive.
    m = re.search(r'Throughput\s*:\s*([0-9.]+)\s*qps', txt, re.I)
    if m: out['throughput_qps'] = float(m.group(1))
    m = re.search(r'Latency\s*:\s*min\s*=\s*[0-9.]+\s*ms\s*,\s*max\s*=\s*[0-9.]+\s*ms\s*,\s*mean\s*=\s*([0-9.]+)\s*ms', txt, re.I)
    if m: out['latency_mean_ms'] = float(m.group(1))
    m = re.search(r'GPU Compute Time\s*:\s*min\s*=\s*[0-9.]+\s*ms\s*,\s*max\s*=\s*[0-9.]+\s*ms\s*,\s*mean\s*=\s*([0-9.]+)\s*ms', txt, re.I)
    if m: out['gpu_compute_mean_ms'] = float(m.group(1))
    m = re.search(r'H2D Latency\s*:\s*min\s*=\s*[0-9.]+\s*ms\s*,\s*max\s*=\s*[0-9.]+\s*ms\s*,\s*mean\s*=\s*([0-9.]+)\s*ms', txt, re.I)
    if m: out['h2d_mean_ms'] = float(m.group(1))
    m = re.search(r'D2H Latency\s*:\s*min\s*=\s*[0-9.]+\s*ms\s*,\s*max\s*=\s*[0-9.]+\s*ms\s*,\s*mean\s*=\s*([0-9.]+)\s*ms', txt, re.I)
    if m: out['d2h_mean_ms'] = float(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare/probe Hailo10H->NativeTRT FIFO eligibility for one BenchmarkSet case')
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--hw-arch', default='hailo10h')
    ap.add_argument('--precision', default='uint8_cast_fp16', choices=['uint8_cast_fp16','uint8_dequant_fp16','fp16'])
    ap.add_argument('--build-missing-engine', action='store_true')
    ap.add_argument('--frames', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--inflight', type=int, default=8)
    ap.add_argument('--runtime-api', default='infer_model', choices=['auto','infer_model','vstreams'])
    ap.add_argument('--quantized-inputs', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--quantized-outputs', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--copy-inputs', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--engine-build-python', default='auto', help='Python used to build missing native TRT engines. auto tries current env and system Python with user-site enabled.')
    ns = ap.parse_args()

    bs = Path(ns.benchmark_set).expanduser().resolve()
    case = _case_id(ns.case)
    case_dir = _case_dir(bs, case)
    out_dir = bs / 'native_pipeline' / case / 'hailo10h_to_trt' / ns.precision
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'hailo10_native_fifo_probe.json'

    hef = _find_hailo10_hef(case_dir)
    part1_onnx = _find_part1_onnx(case_dir, case)
    engine = _native_trt_engine(bs, case, ns.precision)

    report: dict[str, Any] = {
        'ok': False,
        'mode': 'hailo10h_native_fifo_producer_probe',
        'native_fifo_e2e_implemented': False,
        'note': 'Hailo10H producer uses InferModel/run_async; this probe verifies producer and TRT consumer readiness but does not yet run a C++ E2E FIFO handoff.',
        'benchmark_set': str(bs),
        'case': case,
        'hw_arch': str(ns.hw_arch),
        'precision': str(ns.precision),
        'hef': str(hef) if hef else '',
        'part1_onnx': str(part1_onnx) if part1_onnx else '',
        'part2_engine': str(engine) if engine.exists() else '',
        'steps': [],
    }

    if not hef:
        report['error'] = 'missing_hailo10_part1_hef'
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'ok': False, 'error': report['error'], 'report': str(report_path)}, indent=2))
        return 2

    if not engine.exists() and ns.build_missing_engine:
        report['steps'].extend(_build_native_trt_part2(bs, case, ns.precision, ns.timeout, ns.engine_build_python))
    # Re-evaluate after optional build. Older versions left part2_engine blank even
    # when the fallback engine-build Python succeeded; downstream reports then
    # showed consumer=False despite a valid trtexec probe.
    engine_exists_after_build = engine.exists()
    report['part2_engine'] = str(engine) if engine_exists_after_build else ''
    report['consumer_ready'] = bool(engine_exists_after_build)
    if not engine_exists_after_build:
        report['error'] = 'missing_native_trt_part2_engine'
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'ok': False, 'error': report['error'], 'report': str(report_path)}, indent=2))
        return 3

    # Hailo10 producer throughput probe through the already supported Python backend.
    hailo_json = out_dir / 'hailo10_part1_throughput.json'
    cmd = [sys.executable, str(ROOT/'scripts'/'smoke_hailo10_hef_runner.py'), '--hef', str(hef), '--hw-arch', str(ns.hw_arch), '--runtime-api', str(ns.runtime_api), '--throughput-mode', '--frames', str(ns.frames), '--warmup', str(ns.warmup), '--inflight', str(ns.inflight), '--json-out', str(hailo_json)]
    if part1_onnx:
        cmd.extend(['--onnx', str(part1_onnx)])
    cmd.append('--quantized-inputs' if ns.quantized_inputs else '--no-quantized-inputs')
    cmd.append('--quantized-outputs' if ns.quantized_outputs else '--no-quantized-outputs')
    cmd.append('--copy-inputs' if ns.copy_inputs else '--no-copy-inputs')
    step = _run(cmd, timeout=ns.timeout)
    report['steps'].append({'name': 'hailo10_part1_throughput_probe', **step})
    hailo_report = _load_json(hailo_json) if hailo_json.exists() else None
    report['hailo10_part1_probe'] = hailo_report or {}

    # TRT part2 smoke metadata; do not rebuild if a log already exists, but run smoke if missing.
    trt_dir = engine.parent
    run_log = trt_dir / 'run_trtexec.log'
    if not run_log.exists():
        report['steps'].extend(_build_native_trt_part2(bs, case, ns.precision, ns.timeout, ns.engine_build_python))
    report['native_trt_part2_probe'] = _parse_trtexec_log(run_log)

    hthr = (hailo_report or {}).get('throughput') if isinstance(hailo_report, dict) else None
    hailo_completion_interval_ms = None
    if isinstance(hthr, dict):
        value = hthr.get('completion_interval_mean_ms')
        if value is not None:
            hailo_completion_interval_ms = float(value)
    hailo_fps = float(hthr.get('fps')) if isinstance(hthr, dict) and hthr.get('fps') is not None else None
    trt_ms = report['native_trt_part2_probe'].get('latency_mean_ms')
    if hailo_completion_interval_ms is not None and trt_ms is not None:
        cycle = max(float(hailo_completion_interval_ms), float(trt_ms))
        report['projected_cycle_ms_without_handoff'] = cycle
        report['projected_fps_without_handoff'] = 1000.0 / cycle if cycle > 0 else None
        report['projected_cycle_diagnostic_only'] = True
        report['hailo_completion_interval_ms'] = hailo_completion_interval_ms
    report['producer_ready'] = bool(step.get('rc') == 0 and hailo_report)
    report['consumer_ready'] = bool(engine.exists())
    report['ok'] = bool(report['producer_ready'] and report['consumer_ready'])
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'ok': bool(report['ok']), 'report': str(report_path), 'hailo_fps': hailo_fps, 'hailo_completion_interval_ms': hailo_completion_interval_ms, 'trt_latency_ms': trt_ms, 'projected_fps_without_handoff': report.get('projected_fps_without_handoff')}, indent=2))
    return 0 if report['ok'] else 4

if __name__ == '__main__':
    raise SystemExit(main())
