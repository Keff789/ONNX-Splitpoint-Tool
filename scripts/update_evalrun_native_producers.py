#!/usr/bin/env python3
"""Compatibility entry point for updating native-producer results.

The implementation used to duplicate the native coordinator and therefore
silently missed newer measurement-contract features.  It now delegates to the
canonical ``update_evalset_native_producers.py`` flow so repetitions, host
telemetry, report gates, and runner synchronization cannot diverge.
"""
from __future__ import annotations
import argparse, json, shlex, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec='seconds')


def _read_json(p: Path, default: Any = None) -> Any:
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        pass
    return default


def _write_json(p: Path, payload: Any) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return p


def _as_list(x: Any) -> list[str]:
    if not x:
        return []
    if isinstance(x, str):
        return [a.strip() for a in x.replace(';', ',').split(',') if a.strip()]
    if isinstance(x, (list, tuple, set)):
        return [str(a).strip() for a in x if str(a).strip()]
    return [str(x).strip()]


def _norm_backend(b: str) -> str:
    b = str(b or '').strip().lower()
    if b in {'hailo8_to_trt'}: return 'hailo8'
    if b in {'hailo10', 'hailo10h_to_trt'}: return 'hailo10h'
    if b in {'deepx_m1', 'deepx_to_trt'}: return 'deepx'
    return b


def _parse_case_map(s: str) -> dict[str, list[str]]:
    if not s or not s.strip(): return {}
    try:
        data = json.loads(s)
    except Exception:
        return {}
    if not isinstance(data, Mapping): return {}
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        vals = v if isinstance(v, list) else [v]
        out[str(k)] = [str(x) for x in vals if str(x)]
    return out


def _find_models(eval_run: Path, explicit: list[str]) -> list[str]:
    models_dir = eval_run / 'models'
    if explicit:
        return [m for m in explicit if (models_dir / m / 'benchmark_set' / 'benchmark_set.json').is_file()]
    return sorted([p.name for p in models_dir.iterdir() if (p / 'benchmark_set' / 'benchmark_set.json').is_file()]) if models_dir.is_dir() else []


def _case_dirs(eval_run: Path, model: str) -> list[str]:
    bs = eval_run / 'models' / model / 'benchmark_set'
    return sorted([p.name for p in bs.glob('b*') if p.is_dir()])


def _resolve_case_map(eval_run: Path, models: list[str], requested: dict[str, list[str]], policy: str) -> dict[str, list[str]]:
    policy = (policy or 'all_accepted').strip().lower()
    if policy not in {'all_accepted','preferred_then_backfill','case_map_only','first'}:
        policy = 'all_accepted'
    out: dict[str, list[str]] = {}
    for m in models:
        avail = _case_dirs(eval_run, m)
        pref = [c for c in requested.get(m, []) if c in avail]
        if policy == 'case_map_only':
            chosen = pref
        elif policy == 'preferred_then_backfill':
            chosen = list(pref)
            for c in avail:
                if c not in chosen: chosen.append(c)
        elif policy == 'first':
            chosen = pref[:1] if pref else avail[:1]
        else:
            chosen = avail
        out[m] = chosen
    return out


def _remote_for(ns: argparse.Namespace, backend: str) -> dict[str, str]:
    if backend == 'hailo8':
        return {'ssh': ns.hailo8_ssh, 'env': ns.hailo8_env}
    if backend == 'hailo10h':
        return {'ssh': ns.hailo10_ssh, 'env': ns.hailo10_env}
    if backend == 'deepx':
        return {'ssh': ns.deepx_ssh, 'env': ns.deepx_env}
    return {'ssh': '', 'env': ''}


def _q(x: Any) -> str:
    return shlex.quote(str(x))


def _canonical_delegate_command(ns: argparse.Namespace, eval_run: Path) -> list[str]:
    """Translate the legacy CLI into the canonical native coordinator CLI."""
    requested = _parse_case_map(ns.case_map)
    explicit_models = _as_list(ns.models)
    selected_models = _find_models(eval_run, explicit_models)
    if explicit_models and not selected_models:
        raise ValueError("--models did not match any generated BenchmarkSet")

    policy = str(ns.case_policy or "all_accepted")
    effective_map: dict[str, list[str]] = {}
    # The canonical coordinator discovers all model folders.  Preserve the
    # legacy --models filter by turning it into an explicit case-only map.
    # Also materialize the retired ``first`` policy rather than approximating
    # it in a second execution implementation.
    if explicit_models or policy in {"case_map_only", "first"}:
        effective_map = _resolve_case_map(eval_run, selected_models, requested, policy)
        policy = "case_map_only"

    backends: list[str] = []
    for raw in ns.backend or []:
        backend = _norm_backend(raw)
        if backend and backend not in backends:
            backends.append(backend)
    if not backends:
        for backend, ssh in (
            ("hailo8", ns.hailo8_ssh),
            ("hailo10h", ns.hailo10_ssh),
            ("deepx", ns.deepx_ssh),
        ):
            if str(ssh).strip():
                backends.append(backend)
    if not backends:
        raise ValueError("no native backend selected and no configured SSH target found")

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "update_evalset_native_producers.py"),
        "--eval-run-dir", str(eval_run),
        "--run-native-producers",
        "--backends", ",".join(backends),
        "--case-policy", policy,
        "--remote-root", str(ns.remote_root),
        "--remote-tool-dir", str(ns.remote_tool_dir),
        "--precision", str(ns.precision),
        "--frames", str(ns.frames),
        "--warmup", str(ns.warmup),
        "--repetitions", str(ns.repetitions),
        "--queue-depth", str(ns.queue_depth),
        "--inflight", str(ns.inflight),
        "--hailo-format", str(ns.hailo_format),
        "--native-letterbox-pad-value", str(ns.native_letterbox_pad_value),
        "--native-telemetry-label", str(ns.native_telemetry_label),
        "--timeout", str(ns.timeout),
    ]
    if effective_map:
        cmd += ["--case-map", json.dumps(effective_map, separators=(",", ":"))]
    for option, value in (
        ("--hailo8-ssh", ns.hailo8_ssh),
        ("--hailo10-ssh", ns.hailo10_ssh),
        ("--deepx-ssh", ns.deepx_ssh),
        ("--hailo8-env", ns.hailo8_env),
        ("--hailo10-env", ns.hailo10_env),
        ("--deepx-env", ns.deepx_env),
    ):
        if str(value).strip():
            cmd += [option, str(value)]
    if ns.no_copy:
        cmd.append("--no-copy")
    if ns.no_build_missing_engines:
        cmd.append("--no-build-missing-engines")
    if ns.dump_outputs:
        cmd.append("--dump-outputs")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eval-run-dir', required=True)
    ap.add_argument('--backend', action='append', default=[], help='hailo8, hailo10h, deepx. Repeatable. Default: all configured ssh targets.')
    ap.add_argument('--models', default='', help='Comma-separated model ids. Default: all models with benchmark_set.json')
    ap.add_argument('--case-map', default='', help='JSON mapping model -> case list')
    ap.add_argument('--case-policy', default='all_accepted', choices=['all_accepted','preferred_then_backfill','case_map_only','first'])
    ap.add_argument('--remote-root', default='/home/nx/native_fifo_evalsets')
    ap.add_argument('--remote-tool-dir', default='/home/nx/ONNX-Splitpoint-Tool')
    ap.add_argument('--hailo8-ssh', default='')
    ap.add_argument('--hailo10-ssh', default='')
    ap.add_argument('--deepx-ssh', default='')
    ap.add_argument('--hailo8-env', default='')
    ap.add_argument('--hailo10-env', default='export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate')
    ap.add_argument('--deepx-env', default='source ~/venvs/deepx-runtime/bin/activate')
    ap.add_argument('--precision', default='uint8_cast_fp16')
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--repetitions', type=int, default=3, help='Independent performance repetitions. Legacy safe default: Standard contract (3).')
    ap.add_argument('--queue-depth', type=int, default=3)
    ap.add_argument('--inflight', type=int, default=8)
    ap.add_argument('--hailo-format', default='uint8')
    ap.add_argument('--native-letterbox-pad-value', type=int, default=0, help='Native Hailo8 input letterbox pad value. Use 114 to align YOLO native preprocessing with the generic harness.')
    ap.add_argument('--timeout', type=int, default=7200)
    ap.add_argument('--native-telemetry-label', default='legacy_standard', help='Stable label for non-blocking pre/post host telemetry evidence.')
    ap.add_argument('--no-copy', action='store_true')
    ap.add_argument('--no-build-missing-engines', action='store_true')
    ap.add_argument('--dump-outputs', action='store_true')
    ns = ap.parse_args()
    if ns.repetitions < 1:
        ap.error('--repetitions must be >= 1')

    eval_run = Path(ns.eval_run_dir).expanduser().resolve()
    reports = eval_run / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    try:
        cmd = _canonical_delegate_command(ns, eval_run)
    except ValueError as exc:
        ap.error(str(exc))
    pr = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    compatibility = {
        'schema': 'onnx-splitpoint/native-producer-legacy-delegate',
        'schema_version': 1,
        'delegated_to': str(ROOT / 'scripts' / 'update_evalset_native_producers.py'),
        'performance_repetitions': ns.repetitions,
        'host_telemetry': 'canonical_pre_post_non_blocking',
        'cmd': cmd,
        'rc': pr.returncode,
        'stdout_tail': pr.stdout[-4000:],
        'stderr_tail': pr.stderr[-4000:],
    }
    report = _write_json(reports / 'update_evalrun_native_producers_legacy_delegate.json', compatibility)
    print(json.dumps({
        'ok': pr.returncode == 0,
        'delegated': True,
        'canonical_script': str(ROOT / 'scripts' / 'update_evalset_native_producers.py'),
        'compatibility_report': str(report),
    }, indent=2))
    return int(pr.returncode)

if __name__ == '__main__':
    raise SystemExit(main())
