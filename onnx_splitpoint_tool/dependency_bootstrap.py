from __future__ import annotations

"""Shared lightweight dependency bootstrap helpers.

This module intentionally uses only the Python standard library so it can be
invoked very early during startup or from subprocess-oriented flows such as the
YOLO model-preparation exporter.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

LogFn = Callable[[str], None] | None


@dataclass(frozen=True)
class DependencySpec:
    package: str
    module: str


DEPENDENCY_GROUPS: dict[str, tuple[DependencySpec, ...]] = {
    'gui_core': (
        DependencySpec('onnx', 'onnx'),
        DependencySpec('numpy', 'numpy'),
        DependencySpec('matplotlib', 'matplotlib'),
        DependencySpec('pillow', 'PIL'),
        DependencySpec('PyYAML', 'yaml'),
        DependencySpec('jsonschema', 'jsonschema'),
    ),
    'export_screening': (
        DependencySpec('torch', 'torch'),
        DependencySpec('torchvision', 'torchvision'),
        DependencySpec('ultralytics', 'ultralytics'),
        DependencySpec('onnx', 'onnx'),
        DependencySpec('onnxruntime', 'onnxruntime'),
        DependencySpec('onnxscript', 'onnxscript'),
        DependencySpec('onnxslim', 'onnxslim'),
        DependencySpec('numpy', 'numpy'),
    ),
}


def _log(log: LogFn, message: str) -> None:
    if callable(log):
        try:
            log(str(message))
        except Exception:
            pass


def specs_for_groups(groups: Sequence[str]) -> List[DependencySpec]:
    ordered: List[DependencySpec] = []
    seen: set[str] = set()
    for name in groups:
        for spec in DEPENDENCY_GROUPS.get(str(name).strip(), ()):  # type: ignore[arg-type]
            key = f'{spec.package}|{spec.module}'
            if key not in seen:
                seen.add(key)
                ordered.append(spec)
    return ordered


def normalize_python_executable(exe: str | Path) -> str:
    raw = str(exe or '').strip()
    if not raw:
        return ''
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    if os.sep in raw or (os.altsep and os.altsep in raw):
        return str((Path.cwd() / p).absolute())
    found = shutil.which(raw)
    return str(Path(found).absolute()) if found else raw


def same_interpreter(a: str | Path, b: str | Path) -> bool:
    aa = normalize_python_executable(a)
    bb = normalize_python_executable(b)
    if not aa or not bb:
        return False
    try:
        return os.path.samefile(aa, bb)
    except Exception:
        return aa == bb


def ordered_python_candidates(*candidates: str | Path) -> List[str]:
    ordered: List[str] = []
    for cand in candidates:
        norm = normalize_python_executable(cand)
        if not norm:
            continue
        if any(same_interpreter(norm, existing) for existing in ordered):
            continue
        ordered.append(norm)
    return ordered


def current_env_python_candidates(*, cwd: str | Path | None = None) -> List[str]:
    """Return interpreter candidates that preserve the active environment path.

    On some systems a venv may ultimately resolve to the system interpreter via
    symlinks.  Invoking that resolved target directly loses the venv context and
    causes package installation to fall back to the user site.  We therefore
    explicitly prefer the venv launcher path when it is available.
    """

    root = Path(cwd or Path.cwd()).expanduser().absolute()
    candidates: List[str | Path] = []

    env_hint = str(os.environ.get('OSP_PYTHON') or '').strip()
    if env_hint:
        candidates.append(env_hint)

    venv_home = str(os.environ.get('VIRTUAL_ENV') or '').strip()
    if venv_home:
        candidates.append(Path(venv_home) / ('Scripts' if os.name == 'nt' else 'bin') / ('python.exe' if os.name == 'nt' else 'python'))

    local_venv = root / '.venv' / ('Scripts' if os.name == 'nt' else 'bin') / ('python.exe' if os.name == 'nt' else 'python')
    if local_venv.exists():
        candidates.append(local_venv)

    candidates.extend([
        sys.executable,
        shutil.which('python3') or '',
        shutil.which('python') or '',
    ])
    return ordered_python_candidates(*candidates)


def missing_specs_for_python(python_exe: str | Path, specs: Sequence[DependencySpec]) -> List[DependencySpec]:
    exe = normalize_python_executable(python_exe)
    if not exe:
        return list(specs)
    missing: List[DependencySpec] = []
    for spec in specs:
        try:
            proc = subprocess.run([exe, '-c', f'import {spec.module}'], text=True, capture_output=True)
        except Exception:
            return list(specs)
        if proc.returncode != 0:
            missing.append(spec)
    return missing


def ensure_dependency_groups_for_python(
    python_exe: str | Path,
    groups: Sequence[str],
    *,
    log: LogFn = None,
) -> tuple[bool, List[DependencySpec]]:
    exe = normalize_python_executable(python_exe)
    specs = specs_for_groups(groups)
    missing = missing_specs_for_python(exe, specs)
    if not missing:
        return True, []
    packages: List[str] = []
    for spec in missing:
        if spec.package not in packages:
            packages.append(spec.package)
    _log(log, f'[deps] Missing Python packages in {exe}: {", ".join(packages)}')
    _log(log, '[deps] Installing into the active interpreter environment via pip ...')
    try:
        proc = subprocess.run(
            [exe, '-m', 'pip', '--disable-pip-version-check', 'install', *packages],
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        _log(log, f'[deps] pip install failed: {type(exc).__name__}: {exc}')
        return False, missing
    if proc.stdout.strip():
        for line in proc.stdout.splitlines():
            if line.strip():
                _log(log, f'[deps][stdout] {line}')
    if proc.stderr.strip():
        for line in proc.stderr.splitlines():
            if line.strip():
                _log(log, f'[deps][stderr] {line}')
    remaining = missing_specs_for_python(exe, specs)
    return proc.returncode == 0 and not remaining, remaining


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Ensure runtime dependency groups are installed.')
    ap.add_argument('--groups', nargs='+', required=True)
    ap.add_argument('--python', dest='python_exe', default=sys.executable)
    ns = ap.parse_args(argv)
    ok, remaining = ensure_dependency_groups_for_python(ns.python_exe, ns.groups, log=print)
    if ok:
        print('[deps] Dependency check complete.')
        return 0
    if remaining:
        pkgs = ', '.join(sorted({spec.package for spec in remaining}))
        print(f'[deps] Unresolved packages remain: {pkgs}', file=sys.stderr)
    return 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(_main())
