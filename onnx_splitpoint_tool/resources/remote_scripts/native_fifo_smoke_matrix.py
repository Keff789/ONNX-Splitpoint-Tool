#!/usr/bin/env python3
"""Build/test native FIFO fastpath over candidate cases in a BenchmarkSet.

This is a scouting helper: it first writes a capability report, optionally builds
missing uint8 bridge TensorRT engines, then runs short native FIFO smokes for all
cases that become supported.  Failures are recorded per case and do not abort the
whole sweep.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, site, subprocess, sys, sysconfig, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    native_split_quality_selection_duplicates,
    validate_central_native_split_quality_selection,
    validate_native_split_quality_binding,
)


def _cache_verify_only() -> bool:
    return str(
        os.environ.get('ONNX_SPLITPOINT_ARTIFACT_POLICY') or ''
    ).strip().lower() == 'cache_verify_only'


def _cache_verify_replay(binding: dict) -> dict:
    replay = binding.get('cache_verify_replay') if isinstance(binding, dict) else None
    if not isinstance(replay, dict):
        return {}
    source_sha = str(replay.get('source_binding_sha256') or '').strip().lower()
    if (
        replay.get('artifact_policy') != 'cache_verify_only'
        or replay.get('compiler_dispatched') is not False
        or not str(replay.get('local_validation_status') or '').startswith(
            'local_files_rehashed_'
        )
    ):
        return {}
    return replay


def _binding_selection_evidence(binding: dict) -> dict[str, str]:
    replay = _cache_verify_replay(binding) if _cache_verify_only() else {}
    if replay:
        return {
            'native_split_quality_cache_verify_source_binding_sha256': str(
                replay.get('source_binding_sha256') or ''
            ),
            'native_split_quality_cache_verify_replay_sha256': (
                canonical_json_sha256(replay)
            ),
        }
    return native_split_quality_selection_duplicates(binding)


def _no_duplicate_json_keys(pairs):
    value={}
    for key,item in pairs:
        if key in value: raise ValueError(f'duplicate_json_key:{key}')
        value[key]=item
    return value

def _load(p: Path) -> dict:
    try:
        return json.loads(
            p.read_text(encoding='utf-8'),
            object_pairs_hook=_no_duplicate_json_keys,
        )
    except Exception:
        return {}

def _valid_quality_first_binding_set(value: dict, *, setup_id: str) -> bool:
    bindings=value.get('bindings_by_model_case_backend') if isinstance(value,dict) else None
    cache_replay = bool(
        _cache_verify_only()
        and value.get('mode') == 'cache_verify_only'
        and value.get('diagnostic_only') is True
        and value.get('claim_eligible') is False
    ) if isinstance(value,dict) else False
    if (
        not isinstance(value,dict)
        or value.get('schema') != 'onnx-splitpoint/native-split-quality-binding-set'
        or int(value.get('schema_version') or 0) != 2
        or str(value.get('setup_id') or '') != str(setup_id or '')
        or not str(value.get('eval_run_id') or '')
        or not isinstance(bindings,dict)
        or (
            not cache_replay
            and len(str(value.get('central_quality_summary_sha256') or '').strip()) != 64
        )
    ): return False
    if cache_replay:
        attestation=value.get('cache_verify_attestation')
        if not isinstance(attestation,dict): return False
        if (
            attestation.get('status') != 'verified'
            or attestation.get('artifact_policy') != 'cache_verify_only'
            or attestation.get('compiler_dispatch_allowed') is not False
            or attestation.get('compiler_dispatched') is not False
        ): return False
    declared=str(value.get('binding_set_sha256') or '').strip().lower()
    unhashed=dict(value); unhashed.pop('binding_set_sha256',None)
    if (
        not cache_replay
        and (len(declared) != 64 or canonical_json_sha256(unhashed) != declared)
    ):
        return False
    for binding in bindings.values():
        verified,_=validate_native_split_quality_binding(
            binding, verification_mode='portable',
        )
        if verified is None: return False
        if cache_replay:
            if not _cache_verify_replay(verified): return False
        else:
            receipt,_=validate_central_native_split_quality_selection(
                verified, required=True,
            )
            if receipt is None: return False
        if str(verified.get('eval_run_id') or '') != str(value.get('eval_run_id') or ''):
            return False
    return True


def _run(cmd: list[str], timeout: float | None = None, env: dict[str, str] | None = None) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
        return {"cmd": cmd, "rc": p.returncode, "elapsed_s": time.time()-t0, "stdout_tail": p.stdout[-8000:], "stderr_tail": p.stderr[-8000:], "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd, "rc": 124, "elapsed_s": time.time()-t0, "timed_out": True,
            "stdout_tail": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
        }


def _hailo_site_dirs_from_current_interpreter() -> list[str]:
    """Return only existing site dirs that actually provide hailo_platform."""
    candidates: list[str] = []
    candidates.extend(
        str(item)
        for item in str(
            os.environ.get("SPLITPOINT_EXTRA_SITES") or ""
        ).split(os.pathsep)
        if str(item or "").strip()
    )
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    for key in ("purelib", "platlib"):
        value = sysconfig.get_paths().get(key)
        if value:
            candidates.append(value)
    selected: list[str] = []
    for raw in candidates:
        path = Path(str(raw)).expanduser()
        if (
            not path.is_dir()
            or not (
                (path / "hailo_platform").is_dir()
                or any(path.glob("hailo_platform-*.dist-info"))
            )
        ):
            continue
        resolved = str(path.resolve())
        if resolved not in selected:
            selected.append(resolved)
    return selected


def _python_major_minor(python_executable: str) -> tuple[int, int] | None:
    """Read an interpreter ABI without importing any accelerator package."""

    try:
        proc = subprocess.run(
            [
                str(python_executable), "-B", "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except Exception:
        return None
    match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", proc.stdout or "")
    if proc.returncode != 0 or match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _hailo_site_dirs_from_interpreter(
    python_executable: str,
    *,
    expected_abi: tuple[int, int] | None,
) -> list[str]:
    """Ask the Hailo interpreter for the package root it actually imports.

    A venv interpreter must be invoked through its venv path (not its resolved
    symlink), otherwise Python silently falls back to the system prefix.  The
    returned site is admitted only when its major/minor ABI matches the mixed
    TensorRT interpreter.
    """

    code = r"""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.find_spec("hailo_platform")
if spec is None:
    raise SystemExit(3)
roots = []
for raw in list(spec.submodule_search_locations or []):
    path = Path(raw)
    if path.is_dir():
        roots.append(str(path.parent.resolve()))
origin = str(spec.origin or "")
if origin and origin not in {"built-in", "frozen"}:
    path = Path(origin)
    if path.is_file():
        parent = path.parent.parent if path.parent.name == "hailo_platform" else path.parent
        roots.append(str(parent.resolve()))
unique = []
for path in roots:
    if path not in unique:
        unique.append(path)
print(json.dumps({
    "abi": [sys.version_info.major, sys.version_info.minor],
    "sites": unique,
}))
"""
    try:
        proc = subprocess.run(
            [str(python_executable), "-B", "-c", code],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        payload = json.loads((proc.stdout or "").strip())
    except Exception:
        return []
    if proc.returncode != 0 or not isinstance(payload, dict):
        return []
    try:
        observed_abi = tuple(int(value) for value in payload.get("abi") or [])
    except Exception:
        return []
    if len(observed_abi) != 2 or (
        expected_abi is not None and observed_abi != expected_abi
    ):
        return []
    selected: list[str] = []
    for raw in list(payload.get("sites") or []):
        path = Path(str(raw)).expanduser()
        if not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved not in selected:
            selected.append(resolved)
    return selected


def _candidate_hailo_pythons() -> list[str]:
    """Return bounded, explicit Hailo-runtime interpreter candidates."""

    home = Path.home()
    raw_candidates = [
        os.environ.get("HAILO_PY"),
        os.environ.get("ONNX_SPLITPOINT_HAILO_PYTHON"),
        str(home / "hailo_py/bin/python3"),
        str(home / "hailo_py/bin/python"),
        str(home / "hailo_venv/bin/python3"),
        str(home / "hailort_venv/bin/python3"),
        str(home / ".venvs/hailo/bin/python3"),
        str(home / ".venvs/hailort/bin/python3"),
        str(home / "venvs/hailo8/bin/python3"),
        str(home / "venvs/hailo10/bin/python3"),
        sys.executable,
    ]
    selected: list[str] = []
    for raw in raw_candidates:
        value = str(raw or "").strip()
        if not value:
            continue
        executable = Path(value).expanduser()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            continue
        # Preserve the lexical venv path; resolving it can erase activation.
        lexical = str(executable.absolute())
        if lexical not in selected:
            selected.append(lexical)
    return selected


def _hailo_site_dirs_for_mixed_runtime(
    mixed_runtime_python: str,
) -> list[str]:
    """Resolve Hailo sites independently of the launcher interpreter."""

    expected_abi = _python_major_minor(mixed_runtime_python)
    if expected_abi is None:
        return []
    selected = (
        _hailo_site_dirs_from_current_interpreter()
        if tuple(sys.version_info[:2]) == expected_abi
        else []
    )
    for python_executable in _candidate_hailo_pythons():
        for path in _hailo_site_dirs_from_interpreter(
            python_executable, expected_abi=expected_abi,
        ):
            if path not in selected:
                selected.append(path)
    return selected


def _mixed_runtime_python(requested: str) -> str:
    raw = str(
        requested
        or os.environ.get("ONNX_SPLITPOINT_HAILO8_TRT_PYTHON")
        or "/usr/bin/python3"
    ).strip()
    resolved = shutil.which(raw) or raw
    path = Path(resolved).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"hailo8_detection_mixed_runtime_python_unavailable:{raw}"
        )
    return str(path.resolve())


def _mixed_runtime_preflight(
    python_executable: str,
    extra_sites: list[str],
    *,
    timeout: float = 60.0,
) -> dict:
    """Probe the complete import closure without opening accelerator hardware."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SPLITPOINT_EXTRA_SITES"] = os.pathsep.join(extra_sites)
    env["SPLITPOINT_TOOL_ROOT"] = str(ROOT)
    code = r"""
import ctypes
import ctypes.util
import importlib
import json
import os
import site
import sys
from pathlib import Path

extra_sites = []
for raw in str(os.environ.get("SPLITPOINT_EXTRA_SITES") or "").split(os.pathsep):
    path = str(raw or "").strip()
    if not path:
        continue
    if not Path(path).is_dir():
        raise RuntimeError("splitpoint_extra_site_missing:" + path)
    site.addsitedir(path)
    extra_sites.append(path)

modules = {}
for name in ("tensorrt", "hailo_platform", "numpy", "PIL"):
    module = importlib.import_module(name)
    modules[name] = str(getattr(module, "__file__", "") or "")

def _under_any(path_value, roots):
    if not path_value:
        return False
    path = Path(path_value).resolve()
    for raw_root in roots:
        root = Path(raw_root).resolve()
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False

if not _under_any(modules.get("hailo_platform"), extra_sites):
    raise RuntimeError("hailo_platform_origin_not_in_injected_sites")
if _under_any(modules.get("tensorrt"), extra_sites):
    raise RuntimeError("tensorrt_origin_shadowed_by_hailo_sites")

cudart = ctypes.util.find_library("cudart") or "libcudart.so"
ctypes.CDLL(cudart)
tool_root = Path(os.environ["SPLITPOINT_TOOL_ROOT"]).resolve()
sys.path[:0] = [str(tool_root / "scripts"), str(tool_root)]
from native_hailo10_trt_e2e_from_benchmarkset import NativeTRT
from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend
if NativeTRT is None or HailoBackend is None:
    raise RuntimeError("hailo8_detection_source_closure_invalid")
print(json.dumps({
    "status": "ready",
    "runtime_mode": "system_tensorrt_with_process_local_hailo_sites",
    "python_executable": sys.executable,
    "resolved_python_executable": str(Path(sys.executable).resolve()),
    "extra_sites": extra_sites,
    "modules": modules,
    "module_origins_ok": True,
    "cudart": cudart,
    "source_closure_ok": True,
}, sort_keys=True))
"""
    step = _run(
        [python_executable, "-B", "-c", code],
        timeout=timeout,
        env=env,
    )
    payload: dict = {}
    if int(step.get("rc") or 0) == 0:
        lines = [
            line.strip()
            for line in str(step.get("stdout_tail") or "").splitlines()
            if line.strip()
        ]
        try:
            payload = json.loads(lines[-1]) if lines else {}
        except Exception:
            payload = {}
    ok = bool(
        int(step.get("rc") or 0) == 0
        and payload.get("status") == "ready"
        and payload.get("source_closure_ok") is True
        and payload.get("module_origins_ok") is True
        and str(payload.get("resolved_python_executable") or "")
        == str(Path(python_executable).resolve())
        and list(payload.get("extra_sites") or []) == list(extra_sites)
    )
    return {
        "name": "hailo8_detection_mixed_runtime_preflight",
        **step,
        "ok": ok,
        "contract": payload,
        "failure_reason": (
            ""
            if ok
            else "hailo8_detection_mixed_runtime_preflight_failed"
        ),
    }


def _result_file_identity(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return int(stat.st_mtime_ns), int(stat.st_size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



def _python_can_import(py: str, module: str, *, env: dict[str, str] | None = None) -> tuple[bool, str]:
    try:
        proc = subprocess.run([py, '-c', f'import {module}; print(getattr({module}, "__version__", "import_ok"))'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, env=env)
        return proc.returncode == 0, (proc.stdout or proc.stderr or '').strip()[-1000:]
    except Exception as exc:
        return False, f'{type(exc).__name__}: {exc}'


def _candidate_engine_pythons(requested: str) -> list[tuple[str, dict[str, str], str]]:
    base = os.environ.copy()
    user = base.copy(); user.pop('PYTHONNOUSERSITE', None)
    candidates: list[tuple[str, dict[str, str], str]] = []
    seen: set[str] = set()
    def add(value: str | None, env: dict[str, str], label: str) -> None:
        raw = str(value or '').strip()
        if not raw:
            return
        resolved = shutil.which(raw) or raw
        if resolved in seen:
            return
        seen.add(resolved); candidates.append((resolved, env.copy(), label))
    if requested and requested != 'auto':
        add(requested, user, 'requested')
        return candidates
    add(os.environ.get('ONNX_SPLITPOINT_ENGINE_BUILD_PYTHON'), user, 'environment_override')
    tool_root = ROOT
    for cand, label in [
        (tool_root/'.venv/bin/python', 'tool_venv'),
        (Path.home()/'ONNX-Splitpoint-Tool/.venv/bin/python', 'remote_tool_venv'),
        (Path.home()/'.venv/bin/python', 'home_venv'),
        (Path.home()/'venvs/tensorrt/bin/python', 'tensorrt_venv'),
        (Path.home()/'venvs/deepx-runtime/bin/python3', 'deepx_runtime_venv'),
    ]:
        if cand.is_file(): add(str(cand), user, label)
    add('/usr/bin/python3', user, 'system_python')
    add('/usr/bin/python', user, 'system_python')
    add('python3', user, 'path_python3')
    add(sys.executable, user, 'current_user_site_enabled')
    return candidates


def _build_native_trt_part2(bs: Path, cid: str, ns: argparse.Namespace) -> list[dict]:
    base = [str(ROOT/'scripts'/'native_trt_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', cid, '--variants', 'part2', '--precision', ns.precision, '--run-smoke', '--iterations', '50', '--workspace-mb', '4096', '--workspace-mode', 'auto']
    if str(ns.precision) == 'uint8_dequant_fp16':
        if float(ns.dequant_scale or 0.0) > 0.0:
            base += ['--dequant-scale', str(float(ns.dequant_scale))]
        base += ['--dequant-zero-point', str(float(ns.dequant_zero_point or 0.0))]
    if str(ns.precision) in {'uint8_dequant_fp16', 'float32_layout_fp16'} and str(ns.boundary_layout or 'as_input') != 'as_input':
        base += ['--boundary-layout', str(ns.boundary_layout)]
    steps=[]
    for py, env, label in _candidate_engine_pythons(str(ns.engine_build_python or 'auto')):
        can_onnx, probe = _python_can_import(py, 'onnx', env=env)
        prefix={'engine_build_python':py,'engine_build_python_label':label,'engine_build_python_can_import_onnx':can_onnx,'engine_build_python_probe':probe}
        if not can_onnx:
            steps.append({'name':'build_native_trt_part2_skip_python_missing_onnx', **prefix, 'rc':127, 'elapsed_s':0.0, 'stdout_tail':'', 'stderr_tail':'python cannot import onnx', 'timed_out':False})
            continue
        step=_run([py]+base, timeout=ns.timeout, env=env)
        steps.append({'name':'build_native_trt_part2', **prefix, **step})
        if int(step.get('rc') or 0) == 0:
            break
    if not steps:
        steps.append({'name':'build_native_trt_part2_no_python_candidates','rc':127,'elapsed_s':0.0,'stdout_tail':'','stderr_tail':'no Python candidates were available for ONNX bridge generation','timed_out':False})
    return steps

def _failure_from_step(step: dict, *, default: str) -> dict:
    stderr = str(step.get("stderr_tail") or "")
    stdout = str(step.get("stdout_tail") or "")
    if step.get("timed_out") or int(step.get("rc") or 0) == 124:
        reason = "native_runner_timeout"
    elif int(step.get("rc") or 0) != 0:
        reason = "native_runner_nonzero_exit"
    else:
        reason = default
    return {
        "failure_reason": reason,
        "status_detail": reason,
        "returncode": int(step.get("rc") or 0),
        "timed_out": bool(step.get("timed_out")),
        "stdout_tail": stdout,
        "stderr_tail": stderr,
        "error": (stderr.strip() or stdout.strip() or reason)[-4000:],
    }



_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}

def _image_from_manifest(manifest: Path) -> Path | None:
    try:
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        samples = payload.get('samples') if isinstance(payload, dict) else None
        if not isinstance(samples, list):
            return None
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            rel = str(sample.get('image') or sample.get('file') or '').strip()
            if not rel:
                continue
            candidate = (manifest.parent / rel).resolve()
            if candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES:
                return candidate
    except Exception:
        return None
    return None

def _first_validation_image(root: Path) -> Path | None:
    if not root.exists():
        return None
    # Prefer the run-mode subset manifest because it records the exact sample
    # selected for the evaluation contract, including nested ImageNet classes.
    for manifest in sorted(root.glob('**/manifest.json')):
        image = _image_from_manifest(manifest)
        if image is not None:
            return image
    images = sorted(
        p.resolve() for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    return images[0] if images else None

def _default_image(bs: Path) -> Path | None:
    resources = bs / 'resources'
    exact = resources / 'test_image_coco.png'
    if exact.is_file():
        return exact.resolve()
    selected = _first_validation_image(resources / 'validation')
    if selected is not None:
        return selected
    return _first_validation_image(bs)


def _benchmark_task(bs: Path) -> str:
    payload = _load(bs / 'benchmark_set.json')
    return str(
        payload.get('benchmark_task')
        or payload.get('task')
        or payload.get('model_task')
        or ''
    ).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser(description='Native FIFO smoke matrix for BenchmarkSet cases')
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--model-id', default='')
    ap.add_argument('--setup-id', default='')
    ap.add_argument('--native-split-quality-binding-set', default='')
    ap.add_argument('--hw-arch', default='hailo8')
    ap.add_argument('--precision', default='uint8_cast_fp16')
    ap.add_argument('--frames', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--repetitions', type=int, default=1, help='Independent performance repetitions; aggregate is median plus 95%% CI, never best-of.')
    ap.add_argument('--queue-depth', type=int, default=3)
    ap.add_argument('--hailo-format', default='uint8')
    ap.add_argument('--task', default='', choices=['', 'classification', 'detection'], help='Benchmark task. If omitted it is read from benchmark_set.json.')
    ap.add_argument('--preprocess-mode', default='auto', choices=['auto', 'resize', 'letterbox'], help='Image preprocessing; auto means classification=resize and detection=letterbox.')
    ap.add_argument('--image', default='', help='Image file or image directory to pass to every native FIFO case. If omitted, auto-discovery supports detection and classification validation sets.')
    ap.add_argument('--image-map', default='', help='JSON map case->image or model->case->image for exact validation dumps.')
    ap.add_argument('--case', action='append', default=[], help='Restrict to case id(s), repeatable.')
    ap.add_argument('--build-missing-engines', action='store_true')
    ap.add_argument('--engine-build-python', default='auto', help='Python used only for ONNX bridge/TensorRT engine construction. auto probes tool/system interpreters with onnx support.')
    ap.add_argument('--mixed-runtime-python', default=os.environ.get('ONNX_SPLITPOINT_HAILO8_TRT_PYTHON', '/usr/bin/python3'), help='Existing system Python used only for Hailo-8 detection completion, where TensorRT and process-local Hailo sites must coexist.')
    ap.add_argument('--force-rebuild-engines', action='store_true', help='Rebuild native TensorRT Part2 engine even if an engine already exists. Useful after changing dequant/layout bridge parameters.')
    ap.add_argument('--dequant-scale', type=float, default=0.0, help='Forward explicit uint8_dequant_fp16 scale to native_trt_from_benchmarkset.py.')
    ap.add_argument('--dequant-zero-point', type=float, default=0.0, help='Forward explicit uint8_dequant_fp16 zero point to native_trt_from_benchmarkset.py.')
    ap.add_argument('--boundary-layout', default='as_input', help='Forward raw Hailo boundary memory layout to native_trt_from_benchmarkset.py, e.g. memory_nhwc_to_nchw.')
    ap.add_argument('--skip-existing-results', action='store_true')
    ap.add_argument('--dump-outputs', action='store_true')
    ap.add_argument('--dump-boundary', action='store_true', help='Dump raw producer boundary payload for boundary contract diagnostics.')
    ap.add_argument('--letterbox-pad-value', type=int, default=0, help='Forward native Hailo input letterbox pad value. Use 114 to match YOLO generic harness semantics.')
    ap.add_argument('--validate-dumps', action='store_true', default=True, help='Run generic finite/shape validation on native FIFO output dumps when present.')
    ap.add_argument('--no-validate-dumps', dest='validate_dumps', action='store_false')
    ap.add_argument('--timeout', type=float, default=1800)
    ns = ap.parse_args()
    if ns.force_rebuild_engines:
        ap.error('productive_force_build_disabled: --force-rebuild-engines is disabled; --build-missing-engines still permits missing engines')
    if int(ns.repetitions) < 1:
        ap.error('--repetitions must be >= 1')
    bs = Path(ns.benchmark_set).expanduser().resolve()
    quality_first=bool(str(ns.native_split_quality_binding_set or '').strip())
    binding_set={}
    if quality_first:
        binding_set_path = Path(ns.native_split_quality_binding_set).expanduser().resolve()
        binding_set = _load(binding_set_path)
        if (
            not _valid_quality_first_binding_set(
                binding_set, setup_id=str(ns.setup_id or ''),
            )
            or not str(ns.model_id or '')
            or not str(ns.setup_id or '')
        ):
            raise RuntimeError('native_split_quality_binding_set_invalid_or_wrong_setup')
    out_dir = bs / 'analysis_tables'; out_dir.mkdir(parents=True, exist_ok=True)
    image = Path(ns.image).expanduser().resolve() if ns.image else _default_image(bs)
    if image is not None and image.is_dir():
        image = _first_validation_image(image)
    try:
        image_map = json.loads(ns.image_map) if ns.image_map else {}
    except Exception:
        image_map = {}
    def _case_image(cid):
        val = ''
        if isinstance(image_map, dict):
            val = image_map.get(cid) or image_map.get(str(cid)) or ''
            if not val:
                # Support model-level maps: {model:{case:image}}.
                for mv in image_map.values():
                    if isinstance(mv, dict) and (mv.get(cid) or mv.get(str(cid))):
                        val = mv.get(cid) or mv.get(str(cid)); break
        if val:
            p = Path(str(val)).expanduser()
            if p.is_file(): return p.resolve()
            if p.is_dir():
                selected = _first_validation_image(p)
                if selected is not None: return selected
            name = p.name
            for cand in bs.rglob(name):
                if cand.is_file(): return cand.resolve()
                if cand.is_dir():
                    selected = _first_validation_image(cand)
                    if selected is not None: return selected
        return image
    report_cmd = [sys.executable, str(ROOT/'scripts'/'native_fifo_capability_report.py'), '--benchmark-set', str(bs), '--hw-arch', ns.hw_arch, '--precision', ns.precision]
    cap_run = _run(report_cmd, timeout=120)
    cap_path = bs/'analysis_tables'/'native_fifo_capability_report.json'
    if int(cap_run.get('rc') or 0) != 0:
        out = {
            'ok': False, 'orchestration_status': 'failed',
            'evidence_status': 'unavailable', 'benchmark_set': str(bs),
            'hw_arch': ns.hw_arch, 'precision': ns.precision,
            'image': str(image or ''), 'capability_probe': cap_run,
            'failure_reason': 'native_fifo_capability_report_nonzero_exit',
            'cases': [], 'row_count': 0, 'ok_count': 0, 'failed_count': 0,
        }
        out_path = out_dir/'native_fifo_smoke_matrix.json'
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'ok': False, 'failure_reason': out['failure_reason'], 'out': str(out_path)}, indent=2))
        return int(cap_run.get('rc') or 4)
    cap = _load(cap_path)
    if not isinstance(cap.get('cases'), list):
        out = {
            'ok': False, 'orchestration_status': 'failed',
            'evidence_status': 'unavailable', 'benchmark_set': str(bs),
            'hw_arch': ns.hw_arch, 'precision': ns.precision,
            'image': str(image or ''), 'capability_probe': cap_run,
            'failure_reason': 'native_fifo_capability_report_missing_or_invalid_json',
            'capability_report': str(cap_path),
            'cases': [], 'row_count': 0, 'ok_count': 0, 'failed_count': 0,
        }
        out_path = out_dir/'native_fifo_smoke_matrix.json'
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({'ok': False, 'failure_reason': out['failure_reason'], 'out': str(out_path)}, indent=2))
        return 4
    task = str(ns.task or _benchmark_task(bs)).strip().lower()
    if task not in {'classification', 'detection'}:
        raise RuntimeError('benchmark_task_missing_or_invalid')
    mixed_runtime_python = ''
    mixed_runtime_sites: list[str] = []
    mixed_runtime_preflight: dict = {}
    mixed_runtime_env: dict[str, str] | None = None
    if task == 'detection':
        mixed_runtime_python = _mixed_runtime_python(
            str(ns.mixed_runtime_python or '')
        )
        mixed_runtime_sites = _hailo_site_dirs_for_mixed_runtime(
            mixed_runtime_python
        )
        mixed_runtime_preflight = _mixed_runtime_preflight(
            mixed_runtime_python,
            mixed_runtime_sites,
            timeout=min(float(ns.timeout), 60.0),
        )
        mixed_runtime_env = os.environ.copy()
        mixed_runtime_env['PYTHONDONTWRITEBYTECODE'] = '1'
        mixed_runtime_env['SPLITPOINT_EXTRA_SITES'] = os.pathsep.join(
            mixed_runtime_sites
        )
    cases = list(cap.get('cases') or [])
    if ns.case:
        wanted = {c if str(c).startswith('b') else f'b{int(c):03d}' for c in ns.case}
        cases = [r for r in cases if str(r.get('case_id')) in wanted]
    rows = []
    for r in cases:
        cid = str(r.get('case_id'))
        binding_key = f"{ns.model_id}|{cid}|hailo8_to_trt"
        binding = (
            binding_set['bindings_by_model_case_backend'].get(binding_key)
            if quality_first else {}
        )
        row_quality_first = bool(
            quality_first and isinstance(binding, dict)
        )
        row_quality_error = (
            '' if row_quality_first or not quality_first else
            f'native_split_quality_binding_missing:{binding_key}'
        )
        binding_path: Path | None = None
        if row_quality_first:
            selection = binding.get('preselection') if isinstance(binding.get('preselection'), dict) else {}
            if (
                str(selection.get('precision') or '') != str(ns.precision)
                or str(selection.get('setup_id') or '') != str(ns.setup_id)
                or str(selection.get('model_id') or '') != str(ns.model_id)
                or str(selection.get('case_id') or '') != cid
            ):
                row_quality_first = False
                row_quality_error = (
                    f'native_split_quality_binding_scope_mismatch:{binding_key}'
                )
            else:
                binding_path = (
                    bs / 'native_pipeline' / cid / 'hailo_to_trt' / ns.precision
                    / 'native_split_quality_binding.json'
                )
                binding_path.parent.mkdir(parents=True, exist_ok=True)
                binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        row = {
            'case_id': cid,
            'initial_supported': bool(r.get('native_fifo_supported')),
            'image': str(image or ''), 'steps': [],
            'native_split_quality_required': quality_first,
            'native_split_quality_available': row_quality_first,
            'technical_quality_error': row_quality_error,
            'energy_quality_qualified': row_quality_first,
            'energy_quality_status': (
                'quality_qualified' if row_quality_first
                else 'raw_energy_quality_not_qualified'
            ),
            'native_energy_after_technical_error': (
                'not_applicable_quality_qualified' if row_quality_first
                else 'collect_raw_quality_unqualified'
            ),
            'performance_claims_emitted': bool(
                row_quality_first and not _cache_verify_only()
            ),
            'execution_role': (
                'cache_verify_diagnostic_replay'
                if row_quality_first and _cache_verify_only()
                else 'quality_first_native_split' if row_quality_first
                else 'runtime_observation_quality_unqualified'
                if quality_first
                else 'legacy_manual_diagnostic'
            ),
        }
        if task == 'detection':
            row['mixed_runtime_preflight'] = mixed_runtime_preflight
            row['steps'].append(mixed_runtime_preflight)
            if mixed_runtime_preflight.get('ok') is not True:
                reason = str(
                    mixed_runtime_preflight.get('failure_reason')
                    or 'hailo8_detection_mixed_runtime_preflight_failed'
                )
                row.update({
                    'status': 'failed',
                    'reason': reason,
                    'failure_reason': reason,
                    'status_detail': reason,
                    'returncode': int(
                        mixed_runtime_preflight.get('rc') or 0
                    ),
                    'timed_out': bool(
                        mixed_runtime_preflight.get('timed_out')
                    ),
                    'stdout_tail': str(
                        mixed_runtime_preflight.get('stdout_tail') or ''
                    ),
                    'stderr_tail': str(
                        mixed_runtime_preflight.get('stderr_tail') or ''
                    ),
                    'error': str(
                        mixed_runtime_preflight.get('stderr_tail')
                        or mixed_runtime_preflight.get('stdout_tail')
                        or reason
                    )[-4000:],
                })
                rows.append(row)
                continue
        if r.get('native_fifo_ok') and ns.skip_existing_results:
            row.update({'status': 'skipped_existing_result', 'native_fifo_result': r.get('native_fifo_result'), 'fps_makespan': r.get('native_fifo_fps_makespan'), 'handoff_ms': r.get('native_fifo_handoff_ms')})
            rows.append(row); continue
        # Managed Quality-FIRST owns the exact engine.  A direct legacy CLI is
        # still useful diagnostically and may build its historical scouting
        # engine, but its rows are explicitly non-claimable above.
        if not quality_first and (
            ns.force_rebuild_engines
            or (ns.build_missing_engines and not r.get('native_fifo_supported'))
        ):
            row['steps'].extend(_build_native_trt_part2(bs, cid, ns))
        # Refresh capability for this case after possible build.
        cap_refresh = _run(report_cmd, timeout=120)
        row['steps'].append({'name': 'refresh_native_fifo_capability', **cap_refresh})
        if int(cap_refresh.get('rc') or 0) != 0:
            failure = _failure_from_step(
                cap_refresh, default='native_fifo_capability_refresh_failed',
            )
            row.update({'status': 'failed', 'reason': failure['failure_reason'], **failure})
            rows.append(row)
            continue
        cap2 = _load(cap_path)
        if not isinstance(cap2.get('cases'), list):
            row.update({
                'status': 'failed',
                'reason': 'native_fifo_capability_refresh_missing_or_invalid_json',
                'failure_reason': 'native_fifo_capability_refresh_missing_or_invalid_json',
                'status_detail': 'native_fifo_capability_refresh_missing_or_invalid_json',
                'returncode': 0, 'timed_out': False,
            })
            rows.append(row)
            continue
        r2 = next((x for x in cap2.get('cases', []) if str(x.get('case_id')) == cid), r)
        if not r2.get('native_fifo_supported') and not (
            binding_path is not None and binding_path.is_file()
        ):
            reason = str(r2.get('unsupported_reason') or 'native_contract_unsupported')
            row.update({
                'status': 'unsupported', 'reason': reason, 'failure_reason': reason,
                'status_detail': reason, 'unsupported_reason': reason,
                'recommended_action': r2.get('recommended_action'),
                'capability_before': r, 'capability_after': r2,
                'returncode': None, 'timed_out': False,
            })
            rows.append(row); continue
        if image is None:
            reason = 'no_default_image_or_validation_dir_found'
            row.update({'status': 'failed', 'reason': reason, 'failure_reason': reason, 'status_detail': reason, 'recommended_action': 'pass --image explicitly or copy validation resources', 'returncode': None, 'timed_out': False})
            rows.append(row); continue
        # Prepare/reuse the source-bound wrapper outside the 120 s inference
        # budget. This operation never builds model artifacts.
        result = bs/'native_pipeline'/cid/'hailo_to_trt'/ns.precision/'native_fifo_results.json'
        result_before = _result_file_identity(result)
        child_python = (
            mixed_runtime_python if task == 'detection' else sys.executable
        )
        child_env = mixed_runtime_env if task == 'detection' else None
        cmd = [child_python, str(ROOT/'scripts'/'native_hailo_trt_fifo_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', cid, '--hw-arch', ns.hw_arch, '--precision', ns.precision, '--frames', str(ns.frames), '--warmup', str(ns.warmup), '--repetitions', str(ns.repetitions), '--queue-depth', str(ns.queue_depth), '--hailo-format', ns.hailo_format, '--task', task, '--preprocess-mode', str(ns.preprocess_mode), '--letterbox-pad-value', str(int(ns.letterbox_pad_value)), '--image', str(_case_image(cid)), '--setup-id', str(ns.setup_id), '--model-id', str(ns.model_id), '--result-json', str(result)]
        if row_quality_first:
            cmd += ['--eval-run-id', str(binding_set['eval_run_id']), '--source-run-id', str(binding.get('source_run_id') or 'hailo8_to_trt'), '--native-split-quality-binding', str(binding_path)]
        if ns.dump_outputs:
            cmd.append('--dump-outputs')
        if ns.dump_boundary:
            cmd.append('--dump-boundary')
        from onnx_splitpoint_tool.native_progress import run_streaming
        prepare_cmd = [*cmd, '--no-run']
        prepared = run_streaming(prepare_cmd, timeout=300, env=child_env,
                                 label='hailo8-wrapper-prepare')
        row['steps'].append({'name': 'native_wrapper_prepare', 'cmd': prepare_cmd,
                             'rc': prepared.returncode, 'elapsed_s': prepared.elapsed_s,
                             'stdout_tail': prepared.stdout[-8000:]})
        if prepared.returncode:
            row.update(status='failed', reason='native_wrapper_prepare_failed',
                       failure_reason='native_wrapper_prepare_failed',
                       returncode=prepared.returncode)
            rows.append(row)
            continue
        cmd.append('--no-build')
        step = _run(cmd, timeout=ns.timeout, env=child_env)
        row['steps'].append({'name': 'native_fifo_smoke', **step})
        result_after = _result_file_identity(result)
        result_fresh = bool(
            result_after is not None
            and (
                result_before is None
                or result_after != result_before
            )
        )
        child_process_ok = bool(
            int(step.get('rc') or 0) == 0
            and not step.get('timed_out')
        )
        res = _load(result) if child_process_ok and result_fresh else {}
        child_result_ok = bool(child_process_ok and res.get('ok'))
        result_sha256 = (
            _sha256_file(result) if child_result_ok and result.is_file() else ''
        )
        result_size_bytes = (
            int(result.stat().st_size)
            if child_result_ok and result.is_file() else 0
        )
        quality_identity_error=''
        quality_diagnostics=[]
        if row_quality_first and child_result_ok:
            selected=_binding_selection_evidence(binding)
            required_result={
                'eval_run_id':str(binding_set.get('eval_run_id') or ''),
                'native_split_quality_eval_run_id':str(binding_set.get('eval_run_id') or ''),
                'source_run_id':str(binding.get('source_run_id') or ''),
                'native_split_quality_source_run_id':str(binding.get('source_run_id') or ''),
            }
            diagnostic_result={
                'native_split_quality_binding_sha256':str(binding.get('binding_sha256') or ''),
                **selected,
            }
            if not _cache_verify_only():
                required_result.update(diagnostic_result)
                diagnostic_result={}
            drift=[
                field for field,expected in required_result.items()
                if str(res.get(field) or '').strip().lower()
                != str(expected or '').strip().lower()
            ]
            diagnostic_drift=[
                field for field,expected in diagnostic_result.items()
                if str(res.get(field) or '').strip().lower()
                != str(expected or '').strip().lower()
            ]
            if diagnostic_drift:
                quality_diagnostics.append(
                    'cache_verify_diagnostic_identity_mismatch:'
                    + ','.join(diagnostic_drift)
                )
            embedded_binding=res.get('native_split_quality_binding')
            if _cache_verify_only():
                embedded_verified,_=validate_native_split_quality_binding(
                    embedded_binding,
                    expected_identity={
                        'model':str(ns.model_id or ''),
                        'case':cid,
                        'setup_id':str(ns.setup_id or ''),
                        'backend':'hailo8_to_trt',
                        'task':task,
                        'precision':str(ns.precision or ''),
                    },
                    verification_mode='portable',
                )
                if (
                    embedded_verified is None
                    or str(embedded_verified.get('eval_run_id') or '')
                    != str(binding_set.get('eval_run_id') or '')
                ):
                    drift.append('native_split_quality_binding_semantics')
            if drift:
                quality_identity_error=(
                    'native_split_quality_child_result_identity_mismatch:'
                    + ','.join(drift)
                )
        runtime_success = bool(child_result_ok)
        result_ok = bool(runtime_success and not quality_identity_error)
        technical_quality_error = str(
            row_quality_error or quality_identity_error or ''
        )
        row_quality_qualified = bool(
            row_quality_first and not technical_quality_error
        )
        endpoint_results = (
            dict(res.get('endpoint_results') or {})
            if isinstance(res.get('endpoint_results'), dict) else {}
        )
        raw_endpoint = (
            dict(endpoint_results.get('raw_model_outputs') or {})
            if isinstance(endpoint_results.get('raw_model_outputs'), dict)
            else {}
        )
        completed_endpoint = (
            dict(endpoint_results.get('completed_task') or {})
            if isinstance(endpoint_results.get('completed_task'), dict)
            else {}
        )
        primary_endpoint = raw_endpoint or res
        application_endpoint = completed_endpoint or res
        row.update({
            'status': 'ok' if result_ok else 'failed',
            'runtime_success': runtime_success,
            'technical_quality_error': technical_quality_error,
            'energy_quality_qualified': row_quality_qualified,
            'energy_quality_status': (
                'quality_qualified' if row_quality_qualified
                else 'raw_energy_quality_not_qualified'
            ),
            'native_energy_after_technical_error': (
                'not_applicable_quality_qualified'
                if row_quality_qualified
                else 'collect_raw_quality_unqualified'
            ),
            'performance_claims_emitted': bool(
                row_quality_qualified and result_ok
            ),
            'execution_role': (
                'cache_verify_diagnostic_replay'
                if row_quality_qualified and _cache_verify_only()
                else 'quality_first_native_split'
                if row_quality_qualified
                else 'runtime_observation_quality_unqualified'
                if quality_first or runtime_success
                else 'legacy_manual_diagnostic'
            ),
            'native_fifo_result': str(result),
            'native_fifo_result_sha256': result_sha256,
            'native_fifo_result_size_bytes': result_size_bytes,
            'native_fifo_boundary_manifest': res.get('native_fifo_boundary_manifest'),
            'native_fifo_output_manifest': res.get('native_fifo_output_manifest'),
            # Compatibility fps aliases now represent the primary raw
            # performance endpoint.  Completed-Task values remain explicit.
            'fps_makespan': primary_endpoint.get('fps_makespan'),
            'fps_median': primary_endpoint.get('fps_median', primary_endpoint.get('fps_makespan_median', primary_endpoint.get('fps_makespan'))),
            'fps_ci95_low': primary_endpoint.get('fps_ci95_low', primary_endpoint.get('fps_makespan_ci95_low')),
            'fps_ci95_high': primary_endpoint.get('fps_ci95_high', primary_endpoint.get('fps_makespan_ci95_high')),
            'performance_endpoint': 'raw_model_outputs' if raw_endpoint else 'legacy_single_endpoint',
            'primary_performance_endpoint': res.get('primary_performance_endpoint') or ('raw_model_outputs' if raw_endpoint else ''),
            'application_performance_endpoint': res.get('application_performance_endpoint') or ('completed_task' if completed_endpoint else ''),
            'energy_performance_endpoint': res.get('energy_performance_endpoint') or ('completed_task' if completed_endpoint else ''),
            'endpoint_relation': res.get('endpoint_relation') or {},
            'endpoint_relation_verified': res.get('endpoint_relation_verified'),
            'endpoint_results': endpoint_results,
            'raw_model_outputs_fps_makespan': primary_endpoint.get('fps_makespan') if raw_endpoint else None,
            'raw_model_outputs_fps_median': primary_endpoint.get('fps_median', primary_endpoint.get('fps_makespan')) if raw_endpoint else None,
            'raw_model_outputs_paper_equivalent_fps': primary_endpoint.get('paper_equivalent_fps') if raw_endpoint else None,
            'raw_model_outputs_preprocess_ms': primary_endpoint.get('preprocess_ms') if raw_endpoint else None,
            'raw_model_outputs_p1_ms': primary_endpoint.get('p1_ms') if raw_endpoint else None,
            'raw_model_outputs_handoff_ms': primary_endpoint.get('handoff_ms') if raw_endpoint else None,
            'raw_model_outputs_p2_run_ms': primary_endpoint.get('p2_run_ms') if raw_endpoint else None,
            'completed_task_fps_makespan': application_endpoint.get('fps_makespan') if completed_endpoint else None,
            'completed_task_fps_median': application_endpoint.get('fps_median', application_endpoint.get('fps_makespan')) if completed_endpoint else None,
            'completed_task_completion_tail_ms': application_endpoint.get('completion_tail_ms') if completed_endpoint else None,
            'completed_task_p2_run_ms': application_endpoint.get('p2_run_ms') if completed_endpoint else None,
            'repetition_count_requested': res.get('repetition_count_requested', res.get('repetitions_requested')),
            'repetition_count_attempted': res.get('repetition_count_attempted', res.get('repetitions_completed')),
            'repetition_count_valid': res.get('repetition_count_valid', res.get('repetitions_completed')),
            'repetition_status': res.get('repetition_status'),
            'repetition_aggregation': res.get('repetition_aggregation'),
            'request_latency': res.get('request_latency'),
            'repetition_records': res.get('repetition_records', res.get('repetition_evidence', [])),
            'paper_fps': primary_endpoint.get('paper_equivalent_fps'),
            'handoff_ms': primary_endpoint.get('handoff_ms'),
            'result_ok': result_ok,
            'returncode': int(step.get('rc') or 0),
            'timed_out': bool(step.get('timed_out')),
            'stdout_tail': str(step.get('stdout_tail') or ''),
            'stderr_tail': str(step.get('stderr_tail') or ''),
            'capability_before': r,
            'capability_after': r2,
            'setup_id':res.get('setup_id',ns.setup_id),
            'eval_run_id':res.get('eval_run_id'),
            'source_run_id':res.get('source_run_id'),
            'native_command_contract':res.get('native_command_contract'),
            'native_command_contract_sha256':res.get('native_command_contract_sha256'),
            'workload_contract_sha256':res.get('workload_contract_sha256'),
            'native_split_quality_binding':res.get('native_split_quality_binding'),
            'native_split_quality_binding_sha256':res.get('native_split_quality_binding_sha256'),
            'native_split_quality_eval_run_id':res.get('native_split_quality_eval_run_id'),
            'native_split_quality_source_run_id':res.get('native_split_quality_source_run_id'),
            'source_request_sha256':res.get('source_request_sha256'),
            'native_split_quality_source_request_sha256':res.get('native_split_quality_source_request_sha256'),
            'native_split_quality_central_result_sha256':res.get('native_split_quality_central_result_sha256'),
            'native_split_quality_selection_sha256':res.get('native_split_quality_selection_sha256'),
            'native_split_quality_cache_verify_source_binding_sha256':res.get('native_split_quality_cache_verify_source_binding_sha256'),
            'native_split_quality_cache_verify_replay_sha256':res.get('native_split_quality_cache_verify_replay_sha256'),
            'native_split_quality_consumer_attestation':res.get('native_split_quality_consumer_attestation'),
            'native_split_quality_consumer_status':res.get('native_split_quality_consumer_status'),
            'cache_verify_diagnostics':quality_diagnostics,
            'child_result_fresh': result_fresh,
            'mixed_runtime_contract': (
                mixed_runtime_preflight.get('contract')
                if task == 'detection' else {}
            ),
        })
        if not result_ok:
            if not child_process_ok:
                failure = _failure_from_step(
                    step, default='native_runner_nonzero_exit',
                )
            elif not result_fresh:
                reason = 'native_result_missing_or_stale'
                failure = {
                    'failure_reason': reason,
                    'status_detail': reason,
                    'returncode': int(step.get('rc') or 0),
                    'timed_out': bool(step.get('timed_out')),
                    'stdout_tail': str(step.get('stdout_tail') or ''),
                    'stderr_tail': str(step.get('stderr_tail') or ''),
                    'error': reason,
                }
            elif not res.get('ok'):
                reason = str(
                    res.get('failure_reason')
                    or res.get('status_detail')
                    or res.get('error')
                    or 'native_result_not_ok'
                )
                failure = {
                    'failure_reason': reason,
                    'status_detail': str(
                        res.get('status_detail') or reason
                    ),
                    'returncode': int(step.get('rc') or 0),
                    'timed_out': bool(step.get('timed_out')),
                    'stdout_tail': str(step.get('stdout_tail') or ''),
                    'stderr_tail': str(step.get('stderr_tail') or ''),
                    'error': str(res.get('error') or reason)[-4000:],
                }
            else:
                failure = _failure_from_step(
                    step, default='native_result_not_ok',
                )
            if quality_identity_error and child_result_ok:
                failure.update({
                    'failure_reason':quality_identity_error,
                    'status_detail':quality_identity_error,
                    'error':quality_identity_error,
                })
            row.update(failure)
            row['reason'] = row.get('failure_reason')
        if ns.dump_boundary and res.get('ok') and not res.get('native_fifo_boundary_manifest'):
            expected_bman = result.parent / 'native_fifo_boundary' / 'native_fifo_boundary_manifest.json'
            if expected_bman.is_file():
                row['native_fifo_boundary_manifest'] = str(expected_bman)
            else:
                row['boundary_dump_warning'] = f'dump_boundary_requested_but_manifest_missing expected={expected_bman}'
        manifest = res.get('native_fifo_output_manifest')
        if ns.validate_dumps and manifest:
            vcmd = [sys.executable, str(ROOT/'scripts'/'validate_output_dumps.py'), '--candidate', str(manifest)]
            vstep = _run(vcmd, timeout=300)
            row['steps'].append({'name': 'validate_output_dump', **vstep})
            vout = Path(manifest).parent / 'output_dump_validation.json'
            vjson = _load(vout)
            row['output_validation_ok'] = bool(vjson.get('ok'))
            row['output_validation'] = str(vout) if vout.exists() else ''
        rows.append(row)
    # Refresh capability again so a successful smoke immediately changes support counts.
    final_cap_run = _run(report_cmd, timeout=120)
    ok_count = sum(1 for row in rows if row.get('result_ok'))
    failed_count = sum(1 for row in rows if str(row.get('status') or '') in {'failed', 'unsupported'})
    orchestration_ok = int(cap_run.get('rc') or 0) == 0 and int(final_cap_run.get('rc') or 0) == 0
    evidence_status = 'complete' if rows and ok_count == len(rows) else ('partial' if ok_count else 'unavailable')
    out = {
        'ok': bool(orchestration_ok), 'orchestration_status': 'ok' if orchestration_ok else 'failed',
        'evidence_status': evidence_status, 'benchmark_set': str(bs), 'hw_arch': ns.hw_arch,
        'precision': ns.precision, 'task': task, 'preprocess_mode_requested': str(ns.preprocess_mode),
        'preprocess_mode_effective': ('letterbox' if task == 'detection' else 'resize') if str(ns.preprocess_mode) == 'auto' else str(ns.preprocess_mode),
        'image': str(image or ''), 'capability_probe': cap_run,
        'final_capability_probe': final_cap_run, 'cases': rows, 'row_count': len(rows),
        'ok_count': ok_count, 'failed_count': failed_count,
        'native_split_quality_required':quality_first,
        'performance_claims_emitted':bool(
            quality_first and not _cache_verify_only()
        ),
        'execution_role':(
            'cache_verify_diagnostic_replay'
            if quality_first and _cache_verify_only()
            else 'quality_first_native_split'
            if quality_first else 'legacy_manual_diagnostic'
        ),
        'mixed_runtime_preflight': mixed_runtime_preflight,
    }
    out_path = out_dir/'native_fifo_smoke_matrix.json'
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    # Markdown summary
    lines = ['# Native FIFO smoke matrix', '', f'BenchmarkSet: `{bs}`', f'Image: `{image or ""}`', '', '| case | status | fps | handoff ms | dump ok | reason/action |', '|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r.get('case_id')} | {r.get('status')} | {r.get('fps_makespan') or ''} | {r.get('handoff_ms') or ''} | {r.get('output_validation_ok') if 'output_validation_ok' in r else ''} | {r.get('reason') or r.get('recommended_action') or ''} |")
    (out_dir/'native_fifo_smoke_matrix.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'ok': bool(orchestration_ok), 'cases': len(rows), 'ok_count': ok_count, 'evidence_status': evidence_status, 'out': str(out_path), 'md': str(out_dir/'native_fifo_smoke_matrix.md')}, indent=2))
    return 0 if orchestration_ok else 4

if __name__ == '__main__':
    raise SystemExit(main())
