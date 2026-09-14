#!/usr/bin/env python3
"""Locate, refresh, and run a generated benchmark_suite.py from a BenchmarkSet.

Typical usage:

  python scripts/run_benchmark_suite_from_set.py --benchmark-set /path/to/set -- --list-runs

v59af: if the suite harness is missing or stale, try to materialize it in-place
using onnx_splitpoint_tool.benchmark.suite_refresh.refresh_suite_harness.  Also
provides --refresh-only and clearer stale-suite diagnostics for Native-TRT args.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# v59ag: when this script is executed as scripts/run_benchmark_suite_from_set.py,
# Python puts the scripts/ directory on sys.path, not necessarily the tool root.
# Make the local checkout importable so --refresh-only can import
# onnx_splitpoint_tool.benchmark.suite_refresh without requiring pip install -e .
_THIS_FILE = Path(__file__).resolve()
_TOOL_ROOT = _THIS_FILE.parents[1] if _THIS_FILE.parent.name == "scripts" else _THIS_FILE.parents[3]
if str(_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOL_ROOT))


def _find_suite_no_raise(bs: Path) -> Path | None:
    candidates = [
        bs / 'legacy_suite' / 'benchmark_suite.py',
        bs / 'benchmark_suite.py',
    ]
    candidates += sorted(bs.glob('**/benchmark_suite.py'))
    seen: set[str] = set()
    for p in candidates:
        try:
            rp = str(p.resolve()) if p.exists() else str(p)
        except Exception:
            rp = str(p)
        if rp in seen:
            continue
        seen.add(rp)
        if p.is_file():
            return p
    return None


def _guess_benchmark_json(bs: Path, explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
        p2 = bs / explicit
        if p2.is_file():
            return p2.resolve()
    for name in ('benchmark_set.json', 'benchmark_plan.json', 'generation_state.json'):
        p = bs / name
        if p.is_file():
            return p.resolve()
    js = sorted([p for p in bs.glob('*.json') if p.is_file()])
    return js[0].resolve() if js else None


def _refresh_suite(bs: Path, *, benchmark_json: Path | None, force: bool = False) -> dict:
    from onnx_splitpoint_tool.benchmark.suite_refresh import refresh_suite_harness

    log_lines: list[str] = []
    def _log(line: str) -> None:
        log_lines.append(str(line))
        print('[suite-refresh]', line)

    kwargs = {'log': _log}
    if benchmark_json is not None:
        kwargs['benchmark_set_json'] = benchmark_json
    stats = refresh_suite_harness(bs, **kwargs)  # type: ignore[arg-type]
    stats['_log_lines'] = log_lines[-30:]
    return stats


def find_suite(bs: Path, *, refresh: bool = True, benchmark_json: Path | None = None) -> Path:
    suite = _find_suite_no_raise(bs)
    if suite is not None and not refresh:
        return suite
    if refresh:
        try:
            stats = _refresh_suite(bs, benchmark_json=benchmark_json)
            print('[suite-wrapper] refresh changed:', bool(stats.get('changed')), 'case_count:', stats.get('case_count'))
        except Exception as exc:
            print('[suite-wrapper] refresh failed:', f'{type(exc).__name__}: {exc}', file=sys.stderr)
            # Continue to normal discovery so old suites still work.
        suite = _find_suite_no_raise(bs)
        if suite is not None:
            return suite
    # Better diagnostics than a plain FileNotFoundError.
    existing = ', '.join(sorted(p.name for p in bs.iterdir())) if bs.is_dir() else '<not a directory>'
    raise FileNotFoundError(
        f'No benchmark_suite.py found under {bs}. Tried legacy_suite/benchmark_suite.py and recursive search.\n'
        f'BenchmarkSet contents: {existing}\n'
        'This usually means the folder is a raw split-artifact folder without a generated suite harness.\n'
        'Create/refresh the suite on the GUI host, or pass the complete BenchmarkSet directory that contains '
        'benchmark_set.json/benchmark_plan.json and bXXX/split_manifest.json files. For remote-copied folders, '
        'copy the full BenchmarkSet from the GUI host, not only bXXX/ and native_trt/.'
    )



def _suite_supports_native_trt_args(suite: Path) -> bool:
    try:
        txt = suite.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return False
    return ('--trt-runtime-mode' in txt and '--native-trt-precision' in txt and '--boundary-mode' in txt)


def _wants_native_args(args: list[str]) -> bool:
    joined = ' '.join(args or [])
    return any(tok in joined for tok in ('--trt-runtime-mode', '--native-trt-precision', '--boundary-mode', '--handoff-profile'))



def _looks_like_hailo_run(suite_args: list[str]) -> bool:
    joined = ' '.join(str(a) for a in (suite_args or [])).lower()
    return any(tok in joined for tok in ('hailo8', 'hailo10', 'hailo10h', 'hailo10n', 'hailo_to_trt', 'hailo8_to_trt', 'hailo10h_to_trt'))


def _default_hailo_extra_sites() -> str:
    """Return existing common Hailo Python site dirs on Jetson testbeds.

    This mirrors the GUI/remote pattern: run the benchmark with system Python
    for CUDA/TensorRT/ORT and inject Hailo Python bindings via extra site dirs.
    It is only used by the convenience wrapper when SPLITPOINT_EXTRA_SITES is
    not already set.
    """
    cands = []
    for base in (
        Path('/home/nx/hailo_py'),
        Path('/home/nx/venvs/hailo8'),
        Path('/home/nx/venvs/hailo10'),
        Path.home() / 'hailo_py',
        Path.home() / 'venvs' / 'hailo8',
        Path.home() / 'venvs' / 'hailo10',
    ):
        for rel in (
            'lib/python3.10/site-packages',
            'local/lib/python3.10/dist-packages',
            'lib/python3/dist-packages',
            'lib/python3.10/dist-packages',
        ):
            p = base / rel
            if p.is_dir():
                sp = str(p)
                if sp not in cands:
                    cands.append(sp)
    # System dist-packages are helpful for HailoRT/cv2 dependencies on some Jetsons.
    for p in ('/usr/local/lib/python3.10/dist-packages', '/usr/lib/python3/dist-packages', '/usr/lib/python3.10/dist-packages'):
        if Path(p).is_dir() and p not in cands:
            cands.append(p)
    return ':'.join(cands)

def main() -> int:
    ap = argparse.ArgumentParser(description='Run generated benchmark_suite.py from a BenchmarkSet directory.')
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--benchmark-set-json', default=None, help='Optional benchmark_set.json/benchmark_plan.json path if not in the BenchmarkSet root.')
    ap.add_argument('--python', default=os.environ.get('ONNX_SPLITPOINT_SUITE_PYTHON', 'auto'), help="Python used for benchmark_suite.py: auto|current|system|/path/to/python. auto uses /usr/bin/python when SPLITPOINT_EXTRA_SITES is set, otherwise the current interpreter.")
    ap.add_argument('--use-system-python', action='store_true', help='Convenience alias for --python system; useful for Hailo setups where CUDA/TensorRT live in system Python and Hailo bindings are injected via SPLITPOINT_EXTRA_SITES.')
    ap.add_argument('--case', '--case-id', '--only-case', dest='case_filter', default='', help='Restrict generated suite execution to one case id/folder, e.g. b066. Alias forwards --case to refreshed suites or --energy-target-case to older suites.')
    ap.add_argument('--no-refresh', action='store_true', help='Do not try to refresh/materialize benchmark_suite.py before running.')
    ap.add_argument('--refresh-only', action='store_true', help='Refresh/materialize benchmark_suite.py and per-case runners, then exit without running.')
    ap.add_argument('suite_args', nargs=argparse.REMAINDER, help='Arguments after -- are passed to benchmark_suite.py')
    args = ap.parse_args()
    bs = Path(args.benchmark_set).expanduser().resolve()
    if not bs.is_dir():
        raise FileNotFoundError(f'BenchmarkSet directory not found: {bs}')
    bench_json = _guess_benchmark_json(bs, args.benchmark_set_json)
    suite = find_suite(bs, refresh=not args.no_refresh, benchmark_json=bench_json)
    suite_args = args.suite_args
    if suite_args and suite_args[0] == '--':
        suite_args = suite_args[1:]

    if str(getattr(args, 'case_filter', '') or '').strip():
        wanted = str(args.case_filter).strip()
        existing_flags = {'--case', '--case-id', '--only-case', '--energy-target-case'}
        if not any(a in existing_flags for a in suite_args):
            # Refreshed v59ai+ suites understand --case.  Older v59af/v59ag
            # suites can still use --energy-target-case, but refresh should update.
            suite_args = ['--case', wanted] + list(suite_args)

    supports_native = _suite_supports_native_trt_args(suite)
    if args.refresh_only:
        print('[suite-wrapper] benchmark_set:', bs)
        print('[suite-wrapper] benchmark_json:', bench_json or '<auto:none>')
        print('[suite-wrapper] suite:', suite)
        print('[suite-wrapper] supports_native_trt_args:', supports_native)
        return 0 if supports_native or not _wants_native_args(suite_args) else 3

    if _wants_native_args(suite_args) and not supports_native:
        print('[suite-wrapper] ERROR: benchmark_suite.py is stale and does not support Native-TRT/Boundary args.', file=sys.stderr)
        print('[suite-wrapper] The refresh step did not update it. Make sure you are running v59af+ on the NX and that the BenchmarkSet root contains benchmark_set.json plus b*/split_manifest.json.', file=sys.stderr)
        print(f'[suite-wrapper] Try first: python scripts/run_benchmark_suite_from_set.py --benchmark-set {bs} --refresh-only', file=sys.stderr)
        return 64

    def _select_python(spec: str) -> str:
        val = str(spec or 'auto').strip()
        if val in {'current', 'venv'}:
            return sys.executable
        if val in {'system', 'sys'}:
            return '/usr/bin/python' if Path('/usr/bin/python').exists() else '/usr/bin/python3'
        if val in {'auto', ''}:
            # Match the GUI/remote pattern: system Python provides CUDA/ORT/TRT;
            # Hailo bindings are injected via SPLITPOINT_EXTRA_SITES.
            if os.environ.get('SPLITPOINT_EXTRA_SITES'):
                return '/usr/bin/python' if Path('/usr/bin/python').exists() else '/usr/bin/python3'
            return sys.executable
        return val

    suite_py = _select_python('system' if getattr(args, 'use_system_python', False) else args.python)
    cmd = [suite_py, str(suite)] + suite_args
    env = os.environ.copy()
    # v59ao: if a local Hailo run is launched with system Python but the caller
    # forgot SPLITPOINT_EXTRA_SITES, auto-inject the common Hailo venv site dirs.
    # This avoids the confusing "No module named hailort"/"hailo_platform"
    # failure in local smokes while preserving explicit user configuration.
    if (not env.get('SPLITPOINT_EXTRA_SITES')) and _looks_like_hailo_run(suite_args):
        auto_sites = _default_hailo_extra_sites()
        if auto_sites:
            env['SPLITPOINT_EXTRA_SITES'] = auto_sites
            print('[suite-wrapper] auto extra_sites:', auto_sites)
        else:
            print('[suite-wrapper] WARNING: Hailo run requested but SPLITPOINT_EXTRA_SITES is unset and no common Hailo venv site dirs were found.', file=sys.stderr)
    # Make the current tool checkout visible to generated suites and refreshed
    # runner scripts when the BenchmarkSet is executed from another folder.
    pp = [str(_TOOL_ROOT)]
    if env.get('PYTHONPATH'):
        pp.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = ':'.join(pp)
    env.setdefault('PYTHONUNBUFFERED', '1')
    print('[suite-wrapper] benchmark_set:', bs)
    print('[suite-wrapper] benchmark_json:', bench_json or '<auto:none>')
    print('[suite-wrapper] suite:', suite)
    print('[suite-wrapper] python:', suite_py)
    if env.get('SPLITPOINT_EXTRA_SITES'):
        print('[suite-wrapper] extra_sites:', env.get('SPLITPOINT_EXTRA_SITES'))
    print('[suite-wrapper] cmd:', ' '.join(cmd))
    return subprocess.call(cmd, cwd=str(suite.parent), env=env)


if __name__ == '__main__':
    raise SystemExit(main())
