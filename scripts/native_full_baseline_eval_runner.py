#!/usr/bin/env python3
"""Run and diagnose Native full-model baselines for an EvaluationRun.

v60t keeps the Native-full contract deliberately strict but makes failures
observable.  JSON, CSV and backend logs are treated as complementary evidence;
a missing summary field no longer collapses a useful run to ``missing_fps``
without preserving the underlying return code, timeout and log tails.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
# The shipped remote-script mirror also lives below
# ``onnx_splitpoint_tool/resources/remote_scripts``.  When that exact file is
# invoked from a foreign cwd for an import preflight, locate the source root
# before importing the staged runtime closure.  A remotely copied
# ``<tool>/scripts`` file resolves the same loop at ``<tool>``.
for _candidate_root in Path(__file__).resolve().parents:
    if (_candidate_root / "onnx_splitpoint_tool/preprocessing_contract.py").is_file():
        ROOT = _candidate_root
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        break
DEEPX_PREPARED_FEED_CONTRACT_VERSION = "deepx-sealed-runtime-input-v3"
FULL_COMMAND_CONTRACT_SCHEMA = "onnx-splitpoint/native-full-command-contract"
FULL_COMMAND_CONTRACT_VERSION = 1
TRT_ENGINE_BUILD_RECEIPT_SCHEMA = "onnx-splitpoint/tensorrt-engine-build-receipt"
TRT_ENGINE_BUILD_RECEIPT_VERSION = 1
TRT_QUALITY_PRODUCER_SET_SCHEMA = "onnx-splitpoint/tensorrt-quality-producer-set"
TRT_QUALITY_PRODUCER_SET_VERSION = 1
FULL_QUALITY_BINDING_SET_SCHEMA = (
    "onnx-splitpoint/native-full-quality-request-binding-set"
)
FULL_QUALITY_BINDING_SET_VERSIONS = frozenset({1, 2})
ENERGY_PREFLIGHT_ATTESTATION_SCHEMA = "onnx-splitpoint/energy-preflight-attestation"
ENERGY_PREFLIGHT_ATTESTATION_VERSION = 1
_HAILO_RAW_HEAD_PROBE_SCHEMA = (
    "onnx-splitpoint/hailo-source-raw-head-onnx-probe"
)
_HAILO_RAW_HEAD_PROBE_VERSION = 2
_HAILO_RAW_HEAD_PROBE_MARKER = "ONNX_SPLITPOINT_RAW_HEAD_PROBE_V2="
_HAILO_RAW_HEAD_PROBE_CODE = r'''
import hashlib
import json
import sys

SCHEMA = "onnx-splitpoint/hailo-source-raw-head-onnx-probe"
VERSION = 2
MARKER = sys.argv[2]


def emit(payload, returncode):
    print(MARKER + json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    raise SystemExit(returncode)


try:
    import onnx
except Exception as exc:
    emit({
        "schema": SCHEMA,
        "schema_version": VERSION,
        "status": "error",
        "failure_phase": "import_onnx",
        "exception_type": type(exc).__name__,
        "exception_detail": str(exc),
    }, 21)

try:
    with open(sys.argv[1], "rb") as source_handle:
        source_bytes = source_handle.read()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
except Exception as exc:
    emit({
        "schema": SCHEMA,
        "schema_version": VERSION,
        "status": "error",
        "failure_phase": "read_source_onnx",
        "exception_type": type(exc).__name__,
        "exception_detail": str(exc),
    }, 22)

try:
    graph = onnx.load_model_from_string(source_bytes).graph
    outputs = []
    for value_info in list(graph.output):
        tensor_type = value_info.type.tensor_type
        has_shape = bool(tensor_type.HasField("shape"))
        dimensions = []
        if has_shape:
            for dimension in tensor_type.shape.dim:
                if dimension.HasField("dim_value"):
                    dimensions.append({
                        "kind": "value", "value": int(dimension.dim_value),
                    })
                elif dimension.HasField("dim_param"):
                    dimensions.append({
                        "kind": "parameter", "value": str(dimension.dim_param),
                    })
                else:
                    dimensions.append({"kind": "unset"})
        outputs.append({
            "name": str(value_info.name),
            "element_type": int(tensor_type.elem_type),
            "has_shape": has_shape,
            "dimensions": dimensions,
        })
except Exception as exc:
    emit({
        "schema": SCHEMA,
        "schema_version": VERSION,
        "status": "error",
        "failure_phase": "load_onnx",
        "exception_type": type(exc).__name__,
        "exception_detail": str(exc),
        "source_onnx_sha256": source_sha256,
    }, 23)

emit({
    "schema": SCHEMA,
    "schema_version": VERSION,
    "status": "ok",
    "source_onnx_sha256": source_sha256,
    "outputs": outputs,
}, 0)
'''
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from onnx_splitpoint_tool.native_progress import stream_command
except Exception:  # standalone remote checkout may not have the Python package installed
    from queue import Empty, Queue
    from threading import Thread
    from collections import deque

    def stream_command(cmd, *, timeout=None, cwd=None, env=None, label="native-full-child", heartbeat_s=15.0, **_kwargs):
        """Standalone line-streaming fallback used by remotely synced runners."""
        merged_env = dict(os.environ)
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        merged_env.setdefault("PYTHONUNBUFFERED", "1")
        start = time.time()
        q = Queue()
        tail = deque(maxlen=800)
        proc = subprocess.Popen(
            [str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=merged_env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
        )
        def _reader():
            assert proc.stdout is not None
            try:
                for line in iter(proc.stdout.readline, ""):
                    q.put(line)
            finally:
                q.put(None)
        Thread(target=_reader, daemon=True).start()
        eof = False
        last_output = start
        last_heartbeat = start
        timed_out = False
        while True:
            now = time.time()
            try:
                item = q.get(timeout=0.25)
                if item is None:
                    eof = True
                else:
                    clean = item.rstrip("\n")
                    tail.append(clean)
                    last_output = now
                    print(clean, flush=True)
            except Empty:
                pass
            if heartbeat_s > 0 and now - last_heartbeat >= heartbeat_s and proc.poll() is None:
                print(
                    f"[native-progress] HEARTBEAT label={label} elapsed={now-start:.1f}s silence={now-last_output:.1f}s",
                    flush=True,
                )
                last_heartbeat = now
            if timeout is not None and now - start > float(timeout) and proc.poll() is None:
                timed_out = True
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if proc.poll() is not None and eof and q.empty():
                break
        rc = proc.wait()
        if timed_out and rc == 0:
            rc = 124
        output = "\n".join(tail)
        if timed_out:
            output += f"\nTimeoutExpired after {timeout}s"
        return {
            "cmd": [str(x) for x in cmd], "rc": rc, "returncode": rc,
            "elapsed_s": time.time() - start, "stdout": output, "stderr": "",
            "stdout_tail": output[-12000:], "stderr_tail": "", "timed_out": timed_out,
        }

try:
    from onnx_splitpoint_tool.native_detection_postprocess import (
        FrozenPostprocessError,
        build_completed_detection_endpoint_attestation,
        build_normalized_detection_endpoint_attestation,
        verify_frozen_decoded_nms_normalization_contract,
        verify_frozen_postprocess_contract,
        verify_completed_detection_comparison_endpoint_contract,
    )
except Exception:  # --help and diagnostics must work before suite package sync
    class FrozenPostprocessError(RuntimeError):
        pass

    def verify_frozen_postprocess_contract(_value: Any, **_kwargs: Any) -> dict[str, Any]:
        # Missing runtime closure is never accepted as valid provenance.  The
        # updater installs the real module before any Native Full execution.
        raise FrozenPostprocessError("frozen_postprocess_runtime_closure_unavailable")

    def verify_frozen_decoded_nms_normalization_contract(
        _value: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "direct_normalization_runtime_closure_unavailable"
        )

    def build_normalized_detection_endpoint_attestation(
        _contract: Any, _result: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "direct_normalization_runtime_closure_unavailable"
        )

    def build_completed_detection_endpoint_attestation(
        _contract: Any, _result: Any, **_kwargs: Any,
    ) -> dict[str, Any]:
        raise FrozenPostprocessError(
            "completed_endpoint_runtime_closure_unavailable"
        )

from onnx_splitpoint_tool.preprocessing_contract import (
    RUNTIME_NUMERIC_INPUT_SCHEMA,
    RUNTIME_NUMERIC_INPUT_SCHEMA_VERSION,
    canonical_image_preprocessing_contract,
    canonical_runtime_dtype,
    preprocessing_contract_sha256,
    runtime_numeric_input_identity_errors,
    target_hw_from_shape,
)


def _completed_attestation_aliases(
    attestation: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Project only canonical scalar aliases from a nested attestation."""
    payload = dict(attestation or {})
    status = str(payload.get("status") or "").strip().lower()
    return bool(
        payload.get("attested") is True and status == "passed"
    ), status


def _verified_direct_completed_task_workload(
    row: Mapping[str, Any],
    direct_contract: Mapping[str, Any],
    *,
    completed_frames: int,
    postprocess_completed_frames: int,
) -> dict[str, Any]:
    """Seal one measured Direct-BN6 completion into an Energy workload.

    The successful performance row is authoritative only when its physical
    ``decoded_nms`` endpoint, frozen normalizer, result and completed-task
    attestation form one exact closed chain.  Returning an empty mapping keeps
    producer availability fail-closed.
    """
    try:
        verified = verify_frozen_decoded_nms_normalization_contract(
            direct_contract
        )
        result = row.get(
            "frozen_decoded_nms_normalization_result"
        )
        completion = row.get("completed_task_endpoint_attestation")
        source_attestation = row.get("output_endpoint_attestation")
        if (
            not isinstance(result, Mapping)
            or not isinstance(completion, Mapping)
            or not isinstance(source_attestation, Mapping)
        ):
            return {}
        expected_completion = (
            build_normalized_detection_endpoint_attestation(
                verified,
                result,
                completed_frames=completed_frames,
                postprocess_completed_frames=postprocess_completed_frames,
            )
        )
        row_completed_frames = int(
            row.get("completed_frames")
            or row.get("completed_work_units")
            or 0
        )
        row_postprocess_frames = int(
            row.get("postprocess_completed_frames") or 0
        )
    except (FrozenPostprocessError, TypeError, ValueError):
        return {}

    direct_sha = str(
        verified.get("contract_sha256") or ""
    ).strip().lower()
    source_hash = str(
        verified.get("source_endpoint_contract_hash") or ""
    ).strip().lower()
    source_id = str(
        verified.get("source_output_endpoint_id") or ""
    ).strip()
    source_signature = verified.get("source_output_tensor_signature")
    row_source_id = str(row.get("output_endpoint_id") or "").strip()
    if (
        not direct_sha
        or not source_hash
        or source_id != f"detection:decoded_nms:{source_hash}"
        or not isinstance(source_signature, Mapping)
        or str(
            row.get(
                "frozen_decoded_nms_normalization_contract_sha256"
            )
            or ""
        ).strip().lower()
        != direct_sha
        or str(row.get("endpoint_contract_hash") or "").strip().lower()
        != source_hash
        or (row_source_id and row_source_id != source_id)
        or _canonical_json_sha256(source_attestation)
        != str(
            verified.get(
                "source_output_endpoint_attestation_sha256"
            )
            or ""
        ).strip().lower()
        or dict(completion) != dict(expected_completion)
        or row.get("normalization_frozen") is not True
        or row.get("host_postprocess_frozen") is True
        or row.get("postprocess_included") is not True
        or row.get("postprocess_completion_verified") is not True
        or row_completed_frames != int(completed_frames)
        or row_postprocess_frames != int(postprocess_completed_frames)
        or str(row.get("completed_task_stage") or "").strip()
        != "decoded_nms"
        or str(
            row.get("completed_task_contract_family") or ""
        ).strip()
        != "decoded_nms"
        or str(
            row.get("completed_task_completion_mode") or ""
        ).strip()
        != "integrated_accelerator_plus_frozen_normalization"
        or row.get("completed_task_endpoint_attested") is not True
        or str(
            row.get("completed_task_endpoint_attestation_status")
            or ""
        ).strip().lower()
        != "passed"
        or str(
            row.get("completed_task_endpoint_contract_hash") or ""
        ).strip().lower()
        != str(
            expected_completion.get("endpoint_contract_hash") or ""
        ).strip().lower()
        or str(
            row.get("completed_task_output_endpoint_id") or ""
        ).strip()
        != str(
            expected_completion.get("output_endpoint_id") or ""
        ).strip()
        or dict(
            row.get(
                "completed_task_comparison_endpoint_contract"
            )
            or {}
        )
        != dict(
            expected_completion.get(
                "completed_task_comparison_endpoint_contract"
            )
            or {}
        )
        or str(
            row.get(
                "completed_task_comparison_endpoint_contract_hash"
            )
            or ""
        ).strip().lower()
        != str(
            expected_completion.get(
                "completed_task_comparison_endpoint_contract_hash"
            )
            or ""
        ).strip().lower()
        or str(
            row.get(
                "completed_task_comparison_output_endpoint_id"
            )
            or ""
        ).strip()
        != str(
            expected_completion.get(
                "completed_task_comparison_output_endpoint_id"
            )
            or ""
        ).strip()
    ):
        return {}
    return {
        "host_postprocess_frozen": False,
        "normalization_frozen": True,
        "frozen_decoded_nms_normalization_contract": dict(verified),
        "frozen_decoded_nms_normalization_contract_sha256":
            direct_sha,
        "frozen_decoded_nms_normalization_result": dict(result),
        "postprocess_completed_frames":
            int(postprocess_completed_frames),
        "postprocess_completion_verified": True,
        "source_endpoint_contract_hash": source_hash,
        "source_output_endpoint_id": source_id,
        "source_output_tensor_signature": dict(source_signature),
        "source_output_endpoint_attestation_sha256": str(
            verified.get(
                "source_output_endpoint_attestation_sha256"
            )
            or ""
        ).strip().lower(),
        "letterbox_geometry_contract_sha256": str(
            verified.get("letterbox_geometry_contract_sha256") or ""
        ).strip().lower(),
        "completed_task_completion_mode":
            "integrated_accelerator_plus_frozen_normalization",
        "completed_task_endpoint_attestation":
            dict(expected_completion),
    }


def _verified_raw_completed_task_attestation(
    prepared_feed: Mapping[str, Any],
    frozen_contract: Mapping[str, Any],
    *,
    completed_frames: int,
    postprocess_completed_frames: int,
) -> dict[str, Any]:
    """Verify the DeepX host tail's complete physical-to-task chain."""
    try:
        verified = verify_frozen_postprocess_contract(
            frozen_contract
        )
        result = prepared_feed.get("frozen_host_postprocess_result")
        completion = prepared_feed.get(
            "completed_task_endpoint_attestation"
        )
        source_attestation = prepared_feed.get(
            "source_output_endpoint_attestation"
        )
        source_hash = str(
            prepared_feed.get("source_endpoint_contract_hash") or ""
        ).strip().lower()
        if (
            not isinstance(result, Mapping)
            or not isinstance(completion, Mapping)
            or not isinstance(source_attestation, Mapping)
            or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
        ):
            return {}
        expected_completion = (
            build_completed_detection_endpoint_attestation(
                verified,
                result,
                completed_frames=completed_frames,
                postprocess_completed_frames=postprocess_completed_frames,
                source_endpoint_contract_hash=source_hash,
            )
        )
    except (FrozenPostprocessError, TypeError, ValueError):
        return {}

    source_signature = source_attestation.get("tensor_signature")
    source_family = str(verified.get("source_contract_family") or "")
    if (
        source_family not in {"raw_head", "decoded_pre_nms"}
        or prepared_feed.get("runtime_endpoint_contract_complete")
        is not True
        or str(
            prepared_feed.get("runtime_endpoint_contract_family")
            or ""
        ).strip().lower()
        != source_family
        or source_attestation.get("attested") is not True
        or str(
            source_attestation.get("status") or ""
        ).strip().lower()
        != "passed"
        or str(
            source_attestation.get("stage")
            or source_attestation.get("endpoint")
            or ""
        ).strip().lower()
        != source_family
        or str(
            source_attestation.get("endpoint_contract_hash") or ""
        ).strip().lower()
        != source_hash
        or not isinstance(source_signature, Mapping)
        or dict(source_signature)
        != dict(verified.get("raw_output_tensor_signature") or {})
        or dict(completion) != dict(expected_completion)
        or prepared_feed.get("host_postprocess_frozen") is not True
        or prepared_feed.get("normalization_frozen") is True
        or prepared_feed.get("postprocess_included") is not True
        or prepared_feed.get("postprocess_completion_verified")
        is not True
        or int(prepared_feed.get("completed_frames") or 0)
        != int(completed_frames)
        or int(
            prepared_feed.get("postprocess_completed_frames") or 0
        )
        != int(postprocess_completed_frames)
        or str(
            prepared_feed.get("completed_task_completion_mode")
            or completion.get("completed_task_completion_mode")
            or ""
        ).strip()
        != "frozen_host_tail"
    ):
        return {}
    return {
        "source_endpoint_contract_hash": source_hash,
        "source_output_endpoint_attestation": dict(
            source_attestation
        ),
        "completed_task_endpoint_attestation": dict(
            expected_completion
        ),
    }


def _tail(value: Any, limit: int = 12000) -> str:
    text = str(value or "")
    return text[-limit:]


def _run(
    cmd: list[str], *, timeout: int | float | None = None, cwd: Path | None = None,
    env: Mapping[str, str] | None = None, label: str = "native-full-child",
) -> dict[str, Any]:
    return stream_command(
        cmd, timeout=timeout, cwd=cwd, env=env, label=label,
        heartbeat_s=float(os.environ.get("ONNX_SPLITPOINT_NATIVE_HEARTBEAT_S", "15")),
    )


def _probe_python(python: str, modules: tuple[str, ...]) -> tuple[bool, str]:
    code = "; ".join(f"import {name}" for name in modules) + "; print('ok')"
    try:
        cp = subprocess.run([python, "-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return cp.returncode == 0, (cp.stdout.strip() or cp.stderr.strip())[-2000:]
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _select_engine_python(requested: str) -> tuple[str, dict[str, Any]]:
    candidates: list[str] = []
    if requested and requested != "auto":
        candidates.append(requested)
    env_py = os.environ.get("ONNX_SPLITPOINT_ENGINE_BUILD_PYTHON", "").strip()
    if env_py:
        candidates.append(env_py)
    for path in (ROOT / ".venv" / "bin" / "python", ROOT / ".venv-report" / "bin" / "python"):
        if path.is_file():
            candidates.append(str(path))
    for item in (shutil.which("python3"), "/usr/bin/python3", sys.executable):
        if item:
            candidates.append(str(item))
    seen: set[str] = set(); probes: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate = str(Path(candidate).expanduser()) if "/" in candidate else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        ok, detail = _probe_python(candidate, ("onnx", "numpy", "onnxruntime"))
        probes.append({"python": candidate, "ok": ok, "detail": detail})
        if ok:
            return candidate, {"selected": candidate, "probes": probes}
    # Preserve a useful diagnostic.  The caller will fail with a stable reason.
    return "", {"selected": "", "probes": probes, "failure_reason": "no_onnx_capable_engine_build_python"}


def _site_packages_for_python(python: str) -> list[str]:
    if not python:
        return []
    code = (
        "import json,site; "
        "x=[]; "
        "x.extend(site.getsitepackages() if hasattr(site,'getsitepackages') else []); "
        "u=site.getusersitepackages(); x.extend([u] if isinstance(u,str) else list(u)); "
        "print(json.dumps(x))"
    )
    try:
        cp=subprocess.run([python,"-c",code],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
        vals=json.loads(cp.stdout.strip()) if cp.returncode==0 else []
        return [str(v) for v in vals if str(v) and Path(str(v)).is_dir()]
    except Exception:
        return []



def _dedupe_strings(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _runtime_site_packages(backend: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_site_packages_for_python(sys.executable))
    backend_norm = str(backend or "").lower()
    python_candidates: list[Path] = []
    if "hailo" in backend_norm:
        python_candidates.extend([
            Path("/home/nx/hailo_py/bin/python"),
            Path("/home/nx/venvs/hailo8/bin/python"),
            Path("/home/nx/venvs/hailo10/bin/python"),
            Path.home() / "hailo_py" / "bin" / "python",
        ])
    if "deepx" in backend_norm:
        python_candidates.extend([
            Path("/home/nx/deepx_py/bin/python"),
            Path("/home/nx/venvs/deepx/bin/python"),
            Path.home() / "deepx_py" / "bin" / "python",
        ])
    for python_path in python_candidates:
        if python_path.is_file():
            candidates.extend(_site_packages_for_python(str(python_path)))
    # Last-resort discovery for non-standard Python minor versions.
    roots = [Path("/home/nx/hailo_py"), Path("/home/nx/deepx_py"), Path.home() / "hailo_py", Path.home() / "deepx_py"]
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("lib/python*/site-packages", "lib/python*/dist-packages", "local/lib/python*/dist-packages"):
            candidates.extend(str(path) for path in sorted(root.glob(pattern)) if path.is_dir())
    return _dedupe_strings(candidates)


def _suite_python_env(ns: argparse.Namespace, backend: str) -> tuple[str, dict[str, str], list[str]]:
    python = str(getattr(ns, "engine_python_selected", "") or sys.executable)
    sites = _dedupe_strings(
        list(getattr(ns, "engine_python_sites", []) or []) + _runtime_site_packages(backend)
    )
    env: dict[str, str] = {}
    if sites:
        existing = os.environ.get("SPLITPOINT_EXTRA_SITES", "")
        env["SPLITPOINT_EXTRA_SITES"] = os.pathsep.join(sites + ([existing] if existing else []))
    env["PYTHONUNBUFFERED"] = "1"
    return python, env, sites



def _hailo_python_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    roots = [str(ROOT), str(ROOT / "scripts")]
    env["PYTHONPATH"] = os.pathsep.join(roots + ([existing] if existing else []))
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _deepx_hotloop_env() -> dict[str, str]:
    """Expose the synced tool package to an isolated DeepX child process."""
    env = dict(os.environ)
    existing = str(env.get("PYTHONPATH") or "")
    roots = [str(ROOT), str(ROOT / "scripts")]
    env["PYTHONPATH"] = os.pathsep.join(
        roots + ([existing] if existing else [])
    )
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _probe_hailo_python(python: str) -> tuple[bool, str]:
    code = r'''
import importlib
vendor = None
errors = []
for name in ("hailo_platform", "hailort"):
    try:
        importlib.import_module(name)
        vendor = name
        break
    except Exception as exc:
        errors.append(f"{name}:{type(exc).__name__}:{exc}")
if vendor is None:
    raise RuntimeError("; ".join(errors))
from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend
print(vendor)
'''
    try:
        cp = subprocess.run(
            [python, "-c", code], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=45, env=_hailo_python_env(),
        )
        detail = (cp.stdout.strip() or cp.stderr.strip())[-3000:]
        return cp.returncode == 0, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _select_hailo_python(hw_arch: str) -> tuple[str, dict[str, Any]]:
    candidates: list[str] = []
    env_specific = os.environ.get(
        "ONNX_SPLITPOINT_HAILO8_RUNTIME_PYTHON" if str(hw_arch).lower().startswith("hailo8")
        else "ONNX_SPLITPOINT_HAILO10_RUNTIME_PYTHON", "",
    ).strip()
    if env_specific:
        candidates.append(env_specific)
    env_generic = os.environ.get("ONNX_SPLITPOINT_HAILO_RUNTIME_PYTHON", "").strip()
    if env_generic:
        candidates.append(env_generic)
    candidates.append(sys.executable)
    candidates.extend([
        "/home/nx/hailo_py/bin/python",
        "/home/nx/venvs/hailo8/bin/python",
        "/home/nx/venvs/hailo10/bin/python",
        str(Path.home() / "hailo_py" / "bin" / "python"),
    ])
    probes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        if "/" in value and not Path(value).expanduser().is_file():
            probes.append({"python": value, "ok": False, "detail": "not_found"})
            continue
        ok, detail = _probe_hailo_python(value)
        probes.append({"python": value, "ok": ok, "detail": detail})
        if ok:
            return value, {"selected": value, "hw_arch": hw_arch, "probes": probes}
    return "", {
        "selected": "", "hw_arch": hw_arch, "probes": probes,
        "failure_reason": "no_hailo_runtime_python",
    }


def _hailo_aliases(hw_arch: str) -> list[str]:
    value = str(hw_arch or "").strip().lower()
    out = [value]
    if value == "hailo10h":
        out.append("hailo10")
    elif value == "hailo10":
        out.append("hailo10h")
    return list(dict.fromkeys(x for x in out if x))


def _find_hailo_full_hef(benchmark_set: Path, hw_arch: str) -> Path | None:
    candidates: list[Path] = []
    for alias in _hailo_aliases(hw_arch):
        candidates.extend([
            benchmark_set / "hailo" / alias / "full" / "compiled.hef",
            benchmark_set / "hailo" / alias / "full" / "model.hef",
            benchmark_set / "legacy_suite" / "hailo" / alias / "full" / "compiled.hef",
            benchmark_set / "legacy_suite" / "hailo" / alias / "full" / "model.hef",
        ])
        candidates.extend(sorted(benchmark_set.glob(f"**/hailo/{alias}/full/**/*.hef")))
    seen: set[str] = set()
    for candidate in candidates:
        if ".hailo-generations" in candidate.parts:
            continue  # Retained generations are backups, not active artifacts.
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    return None


_HAILO_HEF_BUILD_RECEIPT_SCHEMA = (
    "onnx-splitpoint/hailo-hef-build-receipt/v2"
)
_HAILO_HEF_CACHE_SCHEMAS = {
    "onnx-splitpoint/hailo-hef-cache-key-v2",
    "onnx-splitpoint/hailo-hef-cache-key-v3",
}


def _strict_sha256_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    return token if re.fullmatch(r"[0-9a-f]{64}", token) else ""


def _strict_positive_int(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _strict_hailo_node_names(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    nodes: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            return None
        nodes.append(raw)
    return nodes if len(nodes) == len(set(nodes)) else None


def _hailo_hw_matches(expected: Any, actual: Any) -> bool:
    """Match physical compiler targets; only the 10/10H name is an alias."""

    expected_value = str(expected or "").strip().lower()
    actual_value = str(actual or "").strip().lower()
    if expected_value in {"hailo10", "hailo10h"}:
        return actual_value in {"hailo10", "hailo10h"}
    return bool(expected_value) and actual_value == expected_value


def _hailo_backend_token(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    for token in ("hailo10h", "hailo10", "hailo8r", "hailo8l", "hailo8"):
        if token in text:
            return token
    return text


def _hailo_full_projection_rows(
    *, benchmark_set: Path, model: str, hw_arch: str,
) -> tuple[list[Mapping[str, Any]], str]:
    rows: list[Mapping[str, Any]] = []
    benchmark_path = benchmark_set / "benchmark_set.json"
    if benchmark_path.is_file():
        try:
            benchmark_payload = _load_strict_json(benchmark_path)
        except Exception:
            return [], "hailo_hef_build_receipt_benchmark_metadata_invalid"
        if not isinstance(benchmark_payload, Mapping):
            return [], "hailo_hef_build_receipt_benchmark_metadata_invalid"
        hailo_payload = benchmark_payload.get("hailo") or {}
        hefs = (
            hailo_payload.get("hefs") or {}
            if isinstance(hailo_payload, Mapping) else {}
        )
        if isinstance(hefs, Mapping):
            for alias, raw_meta in hefs.items():
                if not _hailo_hw_matches(
                    hw_arch, _hailo_backend_token(alias)
                ) or not isinstance(raw_meta, Mapping):
                    continue
                rows.append(raw_meta)
                for key in (
                    "full_build", "full_build_receipt", "full_output_contract",
                ):
                    nested = raw_meta.get(key)
                    if isinstance(nested, Mapping):
                        rows.append(nested)

    contracts_path = benchmark_set / "output_contracts.json"
    if contracts_path.is_file():
        try:
            contracts_payload = _load_strict_json(contracts_path)
        except Exception:
            return [], "hailo_hef_build_receipt_output_contracts_invalid"
        if not isinstance(contracts_payload, Mapping):
            return [], "hailo_hef_build_receipt_output_contracts_invalid"
        contracts = contracts_payload.get("contracts") or []
        if not isinstance(contracts, list):
            return [], "hailo_hef_build_receipt_output_contracts_invalid"
        for raw_contract in contracts:
            if not isinstance(raw_contract, Mapping):
                return [], "hailo_hef_build_receipt_output_contracts_invalid"
            backend = _hailo_backend_token(raw_contract.get("backend"))
            if not _hailo_hw_matches(hw_arch, backend):
                continue
            contract_model = str(
                raw_contract.get("model_id") or raw_contract.get("model") or ""
            ).strip()
            if contract_model and contract_model != str(model):
                continue
            variant = str(raw_contract.get("variant") or "full").strip().lower()
            if variant != "full":
                continue
            rows.append(raw_contract)
    return rows, ""


def _projected_hailo_receipt_claims(
    *, benchmark_set: Path, model: str, hw_arch: str,
) -> tuple[dict[str, Any], str]:
    rows, error = _hailo_full_projection_rows(
        benchmark_set=benchmark_set, model=model, hw_arch=hw_arch,
    )
    if error:
        return {}, error
    aliases = {
        "receipt_file_sha256": (
            "hailo_build_receipt_file_sha256", "build_receipt_file_sha256",
        ),
        "receipt_identity_sha256": (
            "hailo_build_receipt_identity_sha256",
            "build_receipt_identity_sha256",
        ),
        "receipt_schema": ("hailo_build_receipt_schema",),
        "cache_key": ("hailo_build_receipt_cache_key",),
        "cache_payload_sha256": (
            "hailo_build_receipt_cache_payload_sha256",
        ),
        "hw_arch": ("hailo_build_receipt_hw_arch",),
        "hailo_sdk_version": ("hailo_build_receipt_sdk_version",),
        "calibration_identity": (
            "hailo_build_receipt_calibration_identity",
        ),
        "prepared_calibration_identity_sha256": (
            "hailo_build_receipt_prepared_calibration_identity_sha256",
        ),
        "calibration_count": (
            "hailo_build_receipt_calibration_count",
        ),
        "requested_calibration_count": (
            "hailo_build_receipt_requested_calibration_count",
        ),
        "calibration_storage": (
            "hailo_build_receipt_calibration_storage",
        ),
        "calibration_memory_cap_bytes": (
            "hailo_build_receipt_calibration_memory_cap_bytes",
        ),
        "source_onnx_sha256": ("source_onnx_sha256",),
        "compiler_onnx_sha256": ("compiler_onnx_sha256",),
    }
    sha_fields = {
        "receipt_file_sha256", "receipt_identity_sha256", "cache_key",
        "cache_payload_sha256", "prepared_calibration_identity_sha256",
        "source_onnx_sha256", "compiler_onnx_sha256",
    }
    int_fields = {
        "calibration_count", "requested_calibration_count",
        "calibration_memory_cap_bytes",
    }
    claims: dict[str, Any] = {}
    for canonical_key, field_names in aliases.items():
        values: list[Any] = []
        for row in rows:
            for field_name in field_names:
                if field_name not in row:
                    continue
                raw_value = row.get(field_name)
                if canonical_key in sha_fields:
                    value = _strict_sha256_token(raw_value)
                elif canonical_key in int_fields:
                    value = _strict_positive_int(raw_value)
                else:
                    value = str(raw_value or "").strip()
                if value in (None, ""):
                    return {}, (
                        "hailo_hef_build_receipt_projected_claim_invalid:"
                        f"{canonical_key}"
                    )
                values.append(value)
        if not values:
            continue
        if len(set(values)) != 1:
            return {}, (
                "hailo_hef_build_receipt_projected_claim_conflict:"
                f"{canonical_key}"
            )
        claims[canonical_key] = values[0]
    return claims, ""


def _declared_hailo_full_end_nodes(
    *, benchmark_set: Path, model: str, hw_arch: str,
) -> tuple[list[str], bool, str, str]:
    """Resolve the externally recorded raw-head boundary without guessing.

    Receipt/cache end nodes are compiler inputs.  For raw detection heads they
    must also agree with the promoted BenchmarkSet/output-contract metadata;
    otherwise a self-consistent but wrong receipt could select a different
    graph boundary at runtime.
    """

    rows, projection_error = _hailo_full_projection_rows(
        benchmark_set=benchmark_set, model=model, hw_arch=hw_arch,
    )
    if projection_error:
        return [], False, "", projection_error
    raw_required = False
    source_raw_flags: list[bool] = []
    declared_origins: list[str] = []

    candidates: list[list[str]] = []
    for row in rows:
        endpoint_text = " ".join(
            str(row.get(key) or "").strip().lower()
            for key in ("endpoint_mode", "mode", "stage", "contract_family")
        )
        if (
            "raw" in endpoint_text
            or row.get("requires_external_postprocess") is True
            or row.get("host_tail_required") is True
        ):
            raw_required = True
        if "source_onnx_multiscale_raw_head" in row:
            if type(row.get("source_onnx_multiscale_raw_head")) is not bool:
                return [], raw_required, "", (
                    "hailo_source_raw_head_declaration_invalid"
                )
            source_raw_flags.append(
                bool(row.get("source_onnx_multiscale_raw_head"))
            )
        origin = str(row.get("raw_endpoint_origin") or "").strip()
        if origin:
            if origin not in {
                "compiler_end_nodes", "source_onnx_graph_outputs",
            }:
                return [], raw_required, "", (
                    "hailo_raw_endpoint_origin_invalid"
                )
            declared_origins.append(origin)
        for key in (
            "hailo_build_receipt_end_nodes",
            "full_end_node_names",
            "end_node_names",
            "compiler_end_nodes",
        ):
            if key not in row:
                continue
            nodes = _strict_hailo_node_names(row.get(key))
            if nodes is None:
                return [], raw_required, "", "hailo_hef_build_receipt_end_nodes_invalid"
            if nodes:
                candidates.append(nodes)

    distinct = {tuple(nodes) for nodes in candidates}
    if len(distinct) > 1:
        return [], raw_required, "", "hailo_hef_build_receipt_end_nodes_conflict"
    if len(set(declared_origins)) > 1:
        return [], raw_required, "", "hailo_raw_endpoint_origin_conflict"
    declared = list(next(iter(distinct))) if distinct else []
    origin = declared_origins[0] if declared_origins else ""
    if declared:
        if origin == "source_onnx_graph_outputs" or any(source_raw_flags):
            return [], raw_required, "", "hailo_raw_endpoint_origin_conflict"
        return declared, raw_required, "compiler_end_nodes", ""
    if raw_required:
        if (
            source_raw_flags
            and all(source_raw_flags)
            and origin in {"", "source_onnx_graph_outputs"}
        ):
            return [], True, "source_onnx_graph_outputs", ""
        return [], True, "", "hailo_hef_build_receipt_raw_end_nodes_missing"
    if origin:
        return [], False, "", "hailo_raw_endpoint_origin_without_raw_contract"
    return declared, raw_required, "not_applicable", ""


def _bounded_diagnostic_text(value: Any, limit: int = 2000) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _set_raw_head_probe_diagnostics(
    diagnostics_out: dict[str, Any] | None,
    **values: Any,
) -> None:
    if diagnostics_out is None:
        return
    diagnostics_out.clear()
    diagnostics_out.update({
        str(key): value for key, value in values.items()
        if value not in (None, "")
    })


def _resolve_attestation_python(python: str | None) -> Path:
    requested = str(python or "").strip()
    if not requested:
        raise FileNotFoundError("ONNX attestation interpreter is empty")
    candidate = Path(requested).expanduser()
    if not candidate.is_absolute() and "/" not in requested:
        located = shutil.which(requested)
        if not located:
            raise FileNotFoundError(
                f"ONNX attestation interpreter not found: {requested}"
            )
        candidate = Path(located)
    # Keep a virtual-environment launcher path intact: resolving its ``python``
    # symlink to the base executable silently discards the venv's site-packages.
    normalized = Path(os.path.abspath(str(candidate)))
    if not normalized.is_file() or not os.access(normalized, os.X_OK):
        raise PermissionError(
            f"ONNX attestation interpreter is not executable: {normalized}"
        )
    return normalized


def _raw_head_probe_payload(
    *, source_onnx: Path, onnx_python: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Extract only primitive ONNX output metadata in the selected Python."""

    diagnostics: dict[str, Any] = {}
    try:
        interpreter = _resolve_attestation_python(onnx_python)
    except Exception as exc:
        diagnostics.update({
            "failure_phase": "resolve_interpreter",
            "exception_type": type(exc).__name__,
            "exception_detail": _bounded_diagnostic_text(exc),
            "requested_python": str(onnx_python or ""),
        })
        return None, diagnostics
    diagnostics["onnx_python"] = str(interpreter)
    try:
        completed = subprocess.run(
            [
                str(interpreter), "-c", _HAILO_RAW_HEAD_PROBE_CODE,
                str(source_onnx), _HAILO_RAW_HEAD_PROBE_MARKER,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
    except Exception as exc:
        diagnostics.update({
            "failure_phase": "execute_probe",
            "exception_type": type(exc).__name__,
            "exception_detail": _bounded_diagnostic_text(exc),
        })
        return None, diagnostics
    diagnostics.update({
        "probe_returncode": int(completed.returncode),
        "probe_stderr_tail": _bounded_diagnostic_text(
            completed.stderr, 4000,
        ),
    })
    records = [
        line[len(_HAILO_RAW_HEAD_PROBE_MARKER):]
        for line in str(completed.stdout or "").splitlines()
        if line.startswith(_HAILO_RAW_HEAD_PROBE_MARKER)
    ]
    if len(records) != 1:
        diagnostics.update({
            "failure_phase": "parse_probe_protocol",
            "exception_type": "RawHeadProbeProtocolError",
            "exception_detail": (
                "expected exactly one marked probe record; "
                f"observed {len(records)}"
            ),
            "probe_stdout_tail": _bounded_diagnostic_text(
                completed.stdout, 4000,
            ),
        })
        return None, diagnostics
    try:
        payload = json.loads(records[0])
    except Exception as exc:
        diagnostics.update({
            "failure_phase": "parse_probe_json",
            "exception_type": type(exc).__name__,
            "exception_detail": _bounded_diagnostic_text(exc),
        })
        return None, diagnostics
    if not isinstance(payload, Mapping):
        diagnostics.update({
            "failure_phase": "validate_probe_protocol",
            "exception_type": "RawHeadProbeProtocolError",
            "exception_detail": "probe payload is not an object",
        })
        return None, diagnostics
    payload = dict(payload)
    if (
        payload.get("schema") != _HAILO_RAW_HEAD_PROBE_SCHEMA
        or payload.get("schema_version") != _HAILO_RAW_HEAD_PROBE_VERSION
        or payload.get("status") not in {"ok", "error"}
    ):
        diagnostics.update({
            "failure_phase": "validate_probe_protocol",
            "exception_type": "RawHeadProbeProtocolError",
            "exception_detail": "probe envelope identity is invalid",
        })
        return None, diagnostics
    if payload.get("status") == "error":
        diagnostics.update({
            "failure_phase": _bounded_diagnostic_text(
                payload.get("failure_phase") or "child_probe",
            ),
            "exception_type": _bounded_diagnostic_text(
                payload.get("exception_type") or "OnnxProbeError",
            ),
            "exception_detail": _bounded_diagnostic_text(
                payload.get("exception_detail") or "ONNX probe failed",
            ),
        })
        return None, diagnostics
    if completed.returncode != 0:
        diagnostics.update({
            "failure_phase": "validate_probe_returncode",
            "exception_type": "RawHeadProbeProcessError",
            "exception_detail": (
                f"successful probe envelope with return code "
                f"{completed.returncode}"
            ),
        })
        return None, diagnostics
    if set(payload) != {
        "schema", "schema_version", "status", "source_onnx_sha256",
        "outputs",
    } or not _strict_sha256_token(payload.get("source_onnx_sha256")):
        diagnostics.update({
            "failure_phase": "validate_probe_protocol",
            "exception_type": "RawHeadProbeProtocolError",
            "exception_detail": "successful probe fields are not canonical",
        })
        return None, diagnostics
    return payload, diagnostics


def _onnx_multiscale_raw_head_attestation(
    source_onnx: Path,
    *,
    source_onnx_sha256: str,
    compiler_onnx_sha256: str,
    onnx_python: str | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Recompute the narrow Source-ONNX raw-head exception.

    The selected engine-build Python performs only ONNX decoding.  This parent
    process validates the primitive probe payload and constructs the canonical
    attestation, so the Hailo vendor interpreter never needs the ONNX package.
    """

    try:
        canonical_source = source_onnx.expanduser().resolve(strict=True)
    except Exception as exc:
        _set_raw_head_probe_diagnostics(
            diagnostics_out,
            failure_phase="resolve_source_onnx",
            exception_type=type(exc).__name__,
            exception_detail=_bounded_diagnostic_text(exc),
        )
        return None, "hailo_source_raw_head_onnx_unreadable"
    source_hash = _strict_sha256_token(source_onnx_sha256)
    compiler_hash = _strict_sha256_token(compiler_onnx_sha256)
    try:
        observed_initial_source_hash = (
            _sha256_file(canonical_source)
            if canonical_source.is_file() else ""
        )
    except Exception as exc:
        _set_raw_head_probe_diagnostics(
            diagnostics_out,
            failure_phase="validate_parent_identity",
            exception_type=type(exc).__name__,
            exception_detail=_bounded_diagnostic_text(exc),
            expected_source_onnx_sha256=source_hash,
        )
        return None, "hailo_source_raw_head_onnx_identity_invalid"
    if (
        not observed_initial_source_hash
        or not source_hash
        or not compiler_hash
        or observed_initial_source_hash != source_hash
    ):
        _set_raw_head_probe_diagnostics(
            diagnostics_out,
            failure_phase="validate_parent_identity",
            exception_type="OnnxIdentityError",
            exception_detail=(
                "Source ONNX path/hash or compiler hash is invalid"
            ),
            expected_source_onnx_sha256=source_hash,
            observed_source_onnx_sha256=observed_initial_source_hash,
        )
        return None, "hailo_source_raw_head_onnx_identity_invalid"
    probe_payload, probe_diagnostics = _raw_head_probe_payload(
        source_onnx=canonical_source,
        onnx_python=(sys.executable if onnx_python is None else onnx_python),
    )
    try:
        observed_source_hash = _sha256_file(canonical_source)
    except Exception as exc:
        identity_diagnostics = dict(probe_diagnostics)
        identity_diagnostics.update({
            "failure_phase": "validate_parent_identity_post_probe",
            "exception_type": type(exc).__name__,
            "exception_detail": _bounded_diagnostic_text(exc),
            "expected_source_onnx_sha256": source_hash,
        })
        _set_raw_head_probe_diagnostics(
            diagnostics_out, **identity_diagnostics,
        )
        return None, "hailo_source_raw_head_onnx_identity_drift"
    if observed_source_hash != source_hash:
        identity_diagnostics = dict(probe_diagnostics)
        identity_diagnostics.update({
            "failure_phase": "validate_parent_identity_post_probe",
            "exception_type": "OnnxIdentityDriftError",
            "exception_detail": (
                "Source ONNX SHA-256 changed during ONNX probe"
            ),
            "expected_source_onnx_sha256": source_hash,
            "observed_source_onnx_sha256": observed_source_hash,
        })
        _set_raw_head_probe_diagnostics(
            diagnostics_out, **identity_diagnostics,
        )
        return None, "hailo_source_raw_head_onnx_identity_drift"
    if probe_payload is None:
        _set_raw_head_probe_diagnostics(
            diagnostics_out, **probe_diagnostics,
        )
        return None, "hailo_source_raw_head_onnx_unreadable"
    child_source_hash = _strict_sha256_token(
        probe_payload.get("source_onnx_sha256")
    )
    if child_source_hash != source_hash:
        identity_diagnostics = dict(probe_diagnostics)
        identity_diagnostics.update({
            "failure_phase": "validate_child_parsed_identity",
            "exception_type": "OnnxIdentityDriftError",
            "exception_detail": (
                "ONNX probe parsed bytes outside the receipt-bound identity"
            ),
            "expected_source_onnx_sha256": source_hash,
            "observed_source_onnx_sha256": child_source_hash,
        })
        _set_raw_head_probe_diagnostics(
            diagnostics_out, **identity_diagnostics,
        )
        return None, "hailo_source_raw_head_onnx_identity_drift"
    raw_outputs = probe_payload.get("outputs")
    if not isinstance(raw_outputs, list):
        _set_raw_head_probe_diagnostics(
            diagnostics_out,
            **probe_diagnostics,
            failure_phase="validate_parent_payload",
            exception_type="RawHeadProbeProtocolError",
            exception_detail="probe outputs are not a list",
        )
        return None, "hailo_source_raw_head_onnx_unreadable"
    outputs: list[dict[str, Any]] = []
    spatial_sizes: list[int] = []
    class_axes: list[int] = []
    output_names: list[str] = []
    for raw_output in raw_outputs:
        if (
            not isinstance(raw_output, Mapping)
            or set(raw_output) != {
                "name", "element_type", "has_shape", "dimensions",
            }
            or type(raw_output.get("name")) is not str
            or not str(raw_output.get("name"))
            or type(raw_output.get("element_type")) is not int
            or int(raw_output.get("element_type") or 0) <= 0
            or type(raw_output.get("has_shape")) is not bool
            or not isinstance(raw_output.get("dimensions"), list)
        ):
            _set_raw_head_probe_diagnostics(
                diagnostics_out,
                **probe_diagnostics,
                failure_phase="validate_parent_payload",
                exception_type="RawHeadProbeProtocolError",
                exception_detail="probe output descriptor is not canonical",
            )
            return None, "hailo_source_raw_head_onnx_unreadable"
        if raw_output.get("has_shape") is not True:
            return None, "hailo_source_raw_head_shape_missing"
        dims: list[int] = []
        for raw_dimension in list(raw_output.get("dimensions") or []):
            if not isinstance(raw_dimension, Mapping):
                _set_raw_head_probe_diagnostics(
                    diagnostics_out,
                    **probe_diagnostics,
                    failure_phase="validate_parent_payload",
                    exception_type="RawHeadProbeProtocolError",
                    exception_detail="probe dimension is not an object",
                )
                return None, "hailo_source_raw_head_onnx_unreadable"
            kind = raw_dimension.get("kind")
            if kind == "value":
                if (
                    set(raw_dimension) != {"kind", "value"}
                    or type(raw_dimension.get("value")) is not int
                    or int(raw_dimension.get("value") or 0) <= 0
                ):
                    return None, "hailo_source_raw_head_shape_dynamic"
                dims.append(int(raw_dimension["value"]))
            elif (
                kind == "parameter"
                and set(raw_dimension) == {"kind", "value"}
                and type(raw_dimension.get("value")) is str
            ) or (
                kind == "unset" and set(raw_dimension) == {"kind"}
            ):
                return None, "hailo_source_raw_head_shape_dynamic"
            else:
                _set_raw_head_probe_diagnostics(
                    diagnostics_out,
                    **probe_diagnostics,
                    failure_phase="validate_parent_payload",
                    exception_type="RawHeadProbeProtocolError",
                    exception_detail="probe dimension descriptor is invalid",
                )
                return None, "hailo_source_raw_head_onnx_unreadable"
        if (
            len(dims) != 5
            or dims[0] != 1
            or dims[1] != 3
            or dims[2] != dims[3]
            or dims[4] < 6
        ):
            return None, "hailo_source_raw_head_signature_not_multiscale"
        spatial_sizes.append(dims[2])
        class_axes.append(dims[4])
        output_names.append(str(raw_output["name"]))
        outputs.append({
            "name": str(raw_output["name"]),
            "element_type": int(raw_output["element_type"]),
            "rank": len(dims),
            "shape": dims,
        })
    if (
        len(outputs) != 3
        or len(set(output_names)) != 3
        or len(set(spatial_sizes)) != 3
        or len(set(class_axes)) != 1
        or sorted(spatial_sizes, reverse=True) != spatial_sizes
    ):
        return None, "hailo_source_raw_head_signature_not_multiscale"
    body: dict[str, Any] = {
        "schema": "onnx-splitpoint/hailo-source-raw-head-attestation",
        "schema_version": 1,
        "raw_endpoint_origin": "source_onnx_graph_outputs",
        "source_onnx_sha256": source_hash,
        "compiler_onnx_sha256": compiler_hash,
        "outputs": outputs,
    }
    body["attestation_sha256"] = _canonical_json_sha256(body)
    _set_raw_head_probe_diagnostics(
        diagnostics_out,
        onnx_python=probe_diagnostics.get("onnx_python"),
        probe_returncode=probe_diagnostics.get("probe_returncode"),
        verification_status="verified",
    )
    return body, "hailo_source_raw_head_attestation_verified"


def _verified_hailo_full_build_receipt(
    *,
    hef: Path,
    benchmark_set: Path,
    model: str,
    hw_arch: str,
    onnx_python: str | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Bind a runnable HEF to its exact compiler/cache/runtime semantics."""

    receipt_path = hef.parent / "hailo_hef_build_receipt.json"
    if not receipt_path.is_file():
        return None, "hailo_hef_build_receipt_missing"
    try:
        raw = _load_strict_json(receipt_path)
    except Exception:
        return None, "hailo_hef_build_receipt_invalid_json"
    if not isinstance(raw, Mapping):
        return None, "hailo_hef_build_receipt_invalid"
    receipt = dict(raw)
    hef_sha = _sha256_file(hef)
    source_sha = _strict_sha256_token(receipt.get("source_onnx_sha256"))
    compiler_sha = _strict_sha256_token(receipt.get("compiler_onnx_sha256"))
    compiler_filename = str(
        receipt.get("compiler_onnx_filename") or ""
    ).strip()
    preprocessing = receipt.get("preprocessing_contract")
    preprocessing = (
        dict(preprocessing) if isinstance(preprocessing, Mapping) else {}
    )
    preprocessing_sha = _strict_sha256_token(
        receipt.get("preprocessing_contract_sha256")
    )
    receipt_arch = str(receipt.get("hw_arch") or "").strip().lower()
    try:
        observed_preprocessing_sha = preprocessing_contract_sha256(
            preprocessing
        )
    except (TypeError, ValueError, OverflowError):
        return None, "hailo_hef_build_receipt_identity_invalid"
    if (
        receipt.get("schema") != _HAILO_HEF_BUILD_RECEIPT_SCHEMA
        or _strict_sha256_token(receipt.get("hef_sha256")) != hef_sha
        or _strict_positive_int(receipt.get("hef_size_bytes"))
        != int(hef.stat().st_size)
        or not source_sha
        or not compiler_sha
        or not compiler_filename
        or Path(compiler_filename).name != compiler_filename
        or not compiler_filename.lower().endswith(".onnx")
        or not preprocessing_sha
        or not preprocessing
        or observed_preprocessing_sha != preprocessing_sha
        or not str(receipt.get("net_name") or "").strip()
        or not _hailo_hw_matches(hw_arch, receipt_arch)
    ):
        return None, "hailo_hef_build_receipt_identity_invalid"
    try:
        canonical_preprocessing = canonical_image_preprocessing_contract(
            preprocessing.get("task"),
            list(preprocessing.get("target_hw") or []),
        )
    except (TypeError, ValueError):
        canonical_preprocessing = {}
    if preprocessing != canonical_preprocessing:
        return None, "hailo_hef_build_receipt_preprocessing_invalid"
    cache_payload = receipt.get("cache_payload")
    if not isinstance(cache_payload, Mapping):
        return None, "hailo_hef_build_receipt_cache_payload_invalid"
    cache_payload = dict(cache_payload)
    cache_key = _strict_sha256_token(receipt.get("cache_key"))
    if (
        cache_payload.get("schema") not in _HAILO_HEF_CACHE_SCHEMAS
        or not cache_key
        or _canonical_json_sha256(cache_payload) != cache_key
    ):
        return None, "hailo_hef_build_receipt_cache_key_invalid"
    if cache_payload.get("schema") == "onnx-splitpoint/hailo-hef-cache-key-v3":
        shapes = cache_payload.get("net_input_shapes")
        shapes_valid = (
            shapes is None
            or (
                isinstance(shapes, list) and bool(shapes)
                and all(type(value) is int for value in shapes)
            )
            or (
                isinstance(shapes, Mapping) and bool(shapes)
                and all(
                    isinstance(name, str) and bool(name)
                    and isinstance(values, list) and bool(values)
                    and all(type(value) is int for value in values)
                    for name, values in shapes.items()
                )
            )
        )
        if (
            str(cache_payload.get("net_name") or "").strip()
            != str(receipt.get("net_name") or "").strip()
            or type(cache_payload.get("disable_rt_metadata_extraction"))
            is not bool
            or not shapes_valid
        ):
            return None, "hailo_hef_build_receipt_cache_translate_identity_invalid"
    cache_preprocessing = cache_payload.get("preprocessing_contract")
    cache_preprocessing = (
        dict(cache_preprocessing)
        if isinstance(cache_preprocessing, Mapping) else {}
    )
    if (
        _strict_sha256_token(cache_payload.get("model_sha256"))
        != compiler_sha
        or _strict_sha256_token(
            cache_payload.get("preprocessing_contract_sha256")
        ) != preprocessing_sha
        or cache_preprocessing != preprocessing
        or str(cache_payload.get("hw_arch") or "").strip().lower()
        != receipt_arch
    ):
        return None, "hailo_hef_build_receipt_cache_identity_mismatch"
    receipt_sdk = str(receipt.get("hailo_sdk_version") or "").strip()
    cache_sdk = str(cache_payload.get("hailo_sdk_version") or "").strip()
    receipt_count = _strict_positive_int(receipt.get("calibration_count"))
    cache_count = _strict_positive_int(cache_payload.get("calibration_count"))
    receipt_requested = _strict_positive_int(
        receipt.get("requested_calibration_count")
    )
    cache_requested = _strict_positive_int(
        cache_payload.get("requested_calibration_count")
    )
    receipt_storage = str(
        receipt.get("calibration_storage") or ""
    ).strip().lower()
    cache_storage = str(
        cache_payload.get("calibration_storage") or ""
    ).strip().lower()
    receipt_cap = _strict_positive_int(
        receipt.get("calibration_memory_cap_bytes")
    )
    cache_cap = _strict_positive_int(
        cache_payload.get("calibration_memory_cap_bytes")
    )
    receipt_calibration = str(
        receipt.get("calibration_identity") or ""
    ).strip()
    cache_calibration = str(
        cache_payload.get("calibration_identity") or ""
    ).strip()
    prepared_identity = _canonical_json_sha256({
        "calibration_identity": cache_calibration,
        "preprocessing_contract_sha256": preprocessing_sha,
    })
    start_nodes = _strict_hailo_node_names(cache_payload.get("start_nodes"))
    end_nodes = _strict_hailo_node_names(cache_payload.get("end_nodes"))
    if (
        not receipt_sdk
        or receipt_sdk != cache_sdk
        or receipt_count is None
        or cache_count is None
        or receipt_count != cache_count
        or receipt_requested is None
        or cache_requested is None
        or receipt_requested != cache_requested
        or receipt_requested < receipt_count
        or receipt_storage not in {"memory", "memmap"}
        or receipt_storage != cache_storage
        or receipt_cap is None
        or receipt_cap != cache_cap
        or not receipt_calibration
        or receipt_calibration != cache_calibration
        or _strict_sha256_token(
            receipt.get("prepared_calibration_identity_sha256")
        ) != prepared_identity
        or _strict_sha256_token(
            cache_payload.get("prepared_calibration_identity_sha256")
        ) != prepared_identity
        or _strict_positive_int(cache_payload.get("calibration_batch_size"))
        is None
        or str(cache_payload.get("integrity") or "")
        not in {"strict", "relaxed"}
        or start_nodes is None
        or end_nodes is None
    ):
        return None, "hailo_hef_build_receipt_calibration_identity_invalid"

    declared_end_nodes, raw_required, raw_endpoint_origin, end_node_error = (
        _declared_hailo_full_end_nodes(
            benchmark_set=benchmark_set,
            model=model,
            hw_arch=hw_arch,
        )
    )
    if end_node_error:
        return None, end_node_error
    if (
        declared_end_nodes != end_nodes
        and (declared_end_nodes or end_nodes or raw_required)
    ):
        return None, "hailo_hef_build_receipt_end_nodes_mismatch"
    projected_claims, projected_claim_error = _projected_hailo_receipt_claims(
        benchmark_set=benchmark_set,
        model=model,
        hw_arch=hw_arch,
    )
    if projected_claim_error:
        return None, projected_claim_error
    observed_claims: dict[str, Any] = {
        "receipt_file_sha256": _sha256_file(receipt_path),
        "receipt_identity_sha256": _workflow_payload_sha256(receipt),
        "receipt_schema": receipt.get("schema"),
        "cache_key": cache_key,
        "cache_payload_sha256": _canonical_json_sha256(cache_payload),
        "hw_arch": receipt_arch,
        "hailo_sdk_version": receipt_sdk,
        "calibration_identity": receipt_calibration,
        "prepared_calibration_identity_sha256": prepared_identity,
        "calibration_count": receipt_count,
        "requested_calibration_count": receipt_requested,
        "calibration_storage": receipt_storage,
        "calibration_memory_cap_bytes": receipt_cap,
        "source_onnx_sha256": source_sha,
        "compiler_onnx_sha256": compiler_sha,
    }
    for claim_name, expected_value in projected_claims.items():
        actual_value = observed_claims.get(claim_name)
        matches = (
            _hailo_hw_matches(expected_value, actual_value)
            if claim_name == "hw_arch"
            else expected_value == actual_value
        )
        if not matches:
            return None, (
                "hailo_hef_build_receipt_projected_claim_mismatch:"
                f"{claim_name}"
            )
    source_candidates = [
        benchmark_set / "models" / f"{model}.onnx",
        benchmark_set / "legacy_suite" / "models" / f"{model}.onnx",
    ]
    source_candidates.extend(
        sorted(benchmark_set.glob(f"**/models/{model}.onnx"))
    )
    matching_sources: list[Path] = []
    seen: set[str] = set()
    for candidate in source_candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if str(resolved) in seen or not resolved.is_file():
            continue
        seen.add(str(resolved))
        if _sha256_file(resolved) == source_sha:
            matching_sources.append(resolved)
    if not matching_sources:
        return None, "hailo_hef_build_receipt_source_onnx_mismatch"
    source_onnx = matching_sources[0]
    compiler_candidates: list[Path] = []
    if source_sha == compiler_sha:
        compiler_candidates.extend(matching_sources)
    compiler_candidates.append(hef.parent / compiler_filename)
    projection_rows, projection_rows_error = _hailo_full_projection_rows(
        benchmark_set=benchmark_set, model=model, hw_arch=hw_arch,
    )
    if projection_rows_error:
        return None, projection_rows_error
    for row in projection_rows:
        for key in (
            "compiler_onnx_path", "matched_compiler_onnx_path",
            "fixed_onnx_path", "compiler_model_path",
        ):
            raw_path = str(row.get(key) or "").strip()
            if not raw_path:
                continue
            declared_path = Path(raw_path).expanduser()
            if declared_path.is_absolute():
                compiler_candidates.append(declared_path)
            else:
                compiler_candidates.extend([
                    benchmark_set / declared_path,
                    hef.parent / declared_path,
                ])
    compiler_candidates.extend(sorted(hef.parent.glob("*.onnx")))
    compiler_candidates.extend(sorted(benchmark_set.rglob("*.onnx")))
    matching_compilers: list[Path] = []
    seen_compilers: set[str] = set()
    for candidate in compiler_candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen_compilers or not resolved.is_file():
            continue
        seen_compilers.add(key)
        if (
            resolved.name == compiler_filename
            and _sha256_file(resolved) == compiler_sha
        ):
            matching_compilers.append(resolved)
    if not matching_compilers:
        return None, "hailo_hef_build_receipt_compiler_onnx_mismatch"
    compiler_onnx = matching_compilers[0]
    source_raw_head_attestation: dict[str, Any] = {}
    if raw_endpoint_origin == "source_onnx_graph_outputs":
        source_attestation_diagnostics: dict[str, Any] = {}
        source_attestation, source_attestation_status = (
            _onnx_multiscale_raw_head_attestation(
                source_onnx,
                source_onnx_sha256=source_sha,
                compiler_onnx_sha256=compiler_sha,
                onnx_python=onnx_python,
                diagnostics_out=source_attestation_diagnostics,
            )
        )
        if source_attestation is None:
            if diagnostics_out is not None:
                diagnostics_out.clear()
                diagnostics_out.update(source_attestation_diagnostics)
            return None, source_attestation_status
        supplied_attestations = [
            dict(row.get("source_onnx_raw_head_attestation") or {})
            for row in projection_rows
            if isinstance(
                row.get("source_onnx_raw_head_attestation"), Mapping,
            )
        ]
        if supplied_attestations and any(
            supplied != source_attestation
            for supplied in supplied_attestations
        ):
            return None, "hailo_source_raw_head_attestation_mismatch"
        source_raw_head_attestation = source_attestation
    return {
        "status": "verified_exact",
        "path": str(receipt_path.resolve()),
        "file_sha256": _sha256_file(receipt_path),
        "receipt_sha256": _workflow_payload_sha256(receipt),
        "receipt": receipt,
        "hef_sha256": hef_sha,
        "source_onnx_path": str(source_onnx),
        "source_onnx_sha256": source_sha,
        "compiler_onnx_sha256": compiler_sha,
        "compiler_onnx_path": str(compiler_onnx),
        "compiler_onnx_filename": compiler_filename,
        "cache_key": cache_key,
        "cache_payload_sha256": _canonical_json_sha256(cache_payload),
        "hw_arch": receipt_arch,
        "hailo_sdk_version": receipt_sdk,
        "calibration_identity": receipt_calibration,
        "prepared_calibration_identity_sha256": prepared_identity,
        "calibration_count": receipt_count,
        "requested_calibration_count": receipt_requested,
        "calibration_storage": receipt_storage,
        "calibration_memory_cap_bytes": receipt_cap,
        "compiler_start_nodes": start_nodes,
        "compiler_end_nodes": end_nodes,
        "raw_end_nodes_required": raw_required,
        "raw_endpoint_origin": raw_endpoint_origin,
        "source_onnx_raw_head_attestation": (
            source_raw_head_attestation
        ),
        "source_onnx_raw_head_attestation_sha256": str(
            source_raw_head_attestation.get("attestation_sha256") or ""
        ),
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
    }, "hailo_hef_build_receipt_verified_exact"


def _full_artifact_was_deferred(benchmark_set: Path, hw_arch: str) -> tuple[bool, str]:
    aliases = set(_hailo_aliases(hw_arch))
    for path in sorted(benchmark_set.rglob("deferred_full_baselines.json")):
        payload = _load_json(path)
        payload_text = json.dumps(payload, sort_keys=True).lower() if payload is not None else ""
        if any(alias in payload_text for alias in aliases):
            return True, f"deferred_full_baselines:{path}"
    for path in sorted(benchmark_set.rglob("hailo_hef_build_result.json")):
        if ".hailo-generations" in path.parts:
            continue
        if "/full/" not in str(path).replace("\\", "/"):
            continue
        payload = _load_json(path)
        if not isinstance(payload, Mapping):
            continue
        arch = str(payload.get("hw_arch") or "").lower()
        if arch and arch not in aliases:
            continue
        kind = str(payload.get("failure_kind") or payload.get("unsupported_reason") or "").lower()
        if "defer" in kind or "cache_only" in kind:
            return True, f"build_result:{kind or path}"
    return False, ""


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("._")
    return text or "unknown"


def _native_full_identity(
    model: str, backend: str, setup_id: str, comparison_backend: str,
) -> dict[str, str]:
    """Return the complete identity of one Native Full evidence row."""
    return {
        "model": str(model or "").strip(),
        "backend": str(backend or "").strip(),
        "setup_id": str(setup_id or "").strip(),
        "comparison_backend": str(comparison_backend or "").strip(),
    }


def _native_full_dump_dir(
    benchmark_set: Path, model: str, backend: str, ns: argparse.Namespace,
) -> Path:
    """Use a collision-free directory for Native Full semantic evidence.

    Model, runtime backend, physical setup and paired comparison backend are
    deliberately encoded even though ``benchmark_set`` is already model-scoped.
    Copied producer trees are frequently rebased from ``native_full_outputs``;
    retaining the full identity below that marker prevents cross-model and
    cross-setup rebinding during local validation.
    """
    identity = _native_full_identity(
        model, backend,
        str(getattr(ns, "setup_id", "") or ""),
        str(getattr(ns, "comparison_backend", "") or ""),
    )
    explicit_root = str(getattr(ns, "out_dir", "") or "").strip()
    evidence_root = (
        _absolute_without_resolving(Path(explicit_root))
        if explicit_root else benchmark_set
    )
    return (
        evidence_root / "native_full_outputs"
        / f"model={_safe_component(identity['model'])}"
        / f"backend={_safe_component(identity['backend'])}"
        / f"setup={_safe_component(identity['setup_id'] or 'unspecified')}"
        / f"comparison={_safe_component(identity['comparison_backend'] or 'unspecified')}"
    )


def _native_full_manifest_identity_status(
    manifest: Path, *, model: str, backend: str, setup_id: str,
    comparison_backend: str,
) -> tuple[bool, str]:
    payload = _load_json(manifest) if manifest.is_file() else None
    if not isinstance(payload, Mapping):
        return False, "native_full_manifest_unreadable"
    expected = _native_full_identity(model, backend, setup_id, comparison_backend)
    for key, value in expected.items():
        if key not in payload:
            return False, f"native_full_manifest_{key}_missing"
        actual = str(payload.get(key) or "").strip()
        if actual != value:
            return False, f"native_full_manifest_{key}_mismatch:{actual!r}!={value!r}"
    if str(payload.get("case") or "").strip() != "full":
        return False, "native_full_manifest_case_mismatch"
    if str(payload.get("execution_mode") or "").strip() != "native_full_baseline":
        return False, "native_full_manifest_execution_mode_mismatch"
    return True, "ok"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _canonical_lexical_absolute_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    path = Path(raw)
    if not raw or not path.is_absolute() or ".." in path.parts:
        return None
    normalized = Path(os.path.normpath(raw))
    return normalized if str(normalized) == raw else None


def _path_contains_symlink(path: Path) -> bool:
    """Check each lexical component before resolve() can hide a link."""
    candidate = _absolute_without_resolving(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _confined_regular_file(
    path: Path, *, allowed_root: Path, expected_path: Path | None = None,
) -> bool:
    candidate = _absolute_without_resolving(path)
    trusted_root = _absolute_without_resolving(allowed_root)
    exact = (
        _absolute_without_resolving(expected_path)
        if expected_path is not None else None
    )
    if (
        _path_contains_symlink(candidate)
        or _path_contains_symlink(trusted_root)
        or (exact is not None and _path_contains_symlink(exact))
        or not candidate.is_file()
        or not trusted_root.is_dir()
    ):
        return False
    try:
        resolved_candidate = candidate.resolve(strict=True)
        resolved_root = trusted_root.resolve(strict=True)
        resolved_candidate.relative_to(resolved_root)
        if exact is not None and resolved_candidate != exact.resolve(strict=True):
            return False
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_benchmark_set_source_onnx_artifacts(
    benchmark_set: Path, model: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Bind the one canonical Full Source-ONNX declared by BenchmarkSet v2.

    Directory scans are intentionally not accepted here.  The model path must
    be the portable, model-specific entry sealed into ``benchmark_set.json``
    and its artifact manifest.  Live bytes are then confined and hashed before
    they can enter the first Full command contract.
    """
    model_id = str(model or "").strip()
    expected_relative = Path("models") / f"{model_id}.onnx"
    expected_relative_text = expected_relative.as_posix()
    manifest_path = benchmark_set / "benchmark_set.json"
    if (
        not model_id
        or model_id != _safe_component(model_id)
        or Path(model_id).name != model_id
        or not _confined_regular_file(
            manifest_path,
            allowed_root=benchmark_set,
            expected_path=manifest_path,
        )
    ):
        return {}, "benchmark_set_source_onnx_manifest_path_invalid"
    try:
        payload = _load_strict_json(manifest_path)
    except Exception:
        return {}, "benchmark_set_source_onnx_manifest_unreadable"
    if not isinstance(payload, Mapping):
        return {}, "benchmark_set_source_onnx_manifest_invalid"
    artifact_manifest = payload.get("artifact_manifest")
    files = (
        artifact_manifest.get("files")
        if isinstance(artifact_manifest, Mapping) else None
    )
    counts = (
        artifact_manifest.get("counts")
        if isinstance(artifact_manifest, Mapping) else None
    )
    if (
        payload.get("schema") != "onnx-splitpoint/benchmark-set"
        or int(payload.get("schema_version") or 0) != 2
        or str(payload.get("model_name") or "") != model_id
        or str(payload.get("model") or "") != expected_relative_text
        or not isinstance(artifact_manifest, Mapping)
        or artifact_manifest.get("schema")
        != "onnx-splitpoint/benchmark-set"
        or int(artifact_manifest.get("schema_version") or 0) != 2
        or not isinstance(files, Mapping)
        or files.get("models") != [expected_relative_text]
        or not isinstance(counts, Mapping)
        or counts.get("models") != 1
    ):
        return {}, "benchmark_set_source_onnx_manifest_identity_invalid"
    source_path = benchmark_set / expected_relative
    if not _confined_regular_file(
        source_path,
        allowed_root=benchmark_set / "models",
        expected_path=source_path,
    ):
        return {}, "benchmark_set_source_onnx_file_invalid"
    try:
        source_size = int(source_path.stat().st_size)
        manifest_size = int(manifest_path.stat().st_size)
        source_sha = _sha256_file(source_path)
        manifest_sha = _sha256_file(manifest_path)
    except OSError:
        return {}, "benchmark_set_source_onnx_file_unreadable"
    if source_size <= 0 or manifest_size <= 0:
        return {}, "benchmark_set_source_onnx_file_empty"
    return {
        "benchmark_set_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "size_bytes": manifest_size,
            "role": "sealed_benchmark_set_manifest",
        },
        "source_onnx": {
            "path": str(source_path.resolve()),
            "sha256": source_sha,
            "size_bytes": source_size,
            "role": "sealed_benchmark_set_source_onnx",
            "relative_path": expected_relative_text,
            "benchmark_set_manifest_sha256": manifest_sha,
        },
    }, "benchmark_set_source_onnx_verified_exact"


def _workflow_payload_sha256(value: Any) -> str:
    """Match workflow.artifacts.sha256_payload for projected receipt claims."""

    encoded = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _completed_result_artifact_persistence_status(
    payload: Mapping[str, Any],
    *,
    sealed_result: Mapping[str, Any],
    allowed_root: Path,
    expected_path: Path,
) -> tuple[bool, str]:
    """Verify the exact same-hotloop Completed-v2 artifact on disk."""

    artifact = payload.get("completed_task_result_artifact")
    artifact_sha = str(
        payload.get("completed_task_result_artifact_sha256") or ""
    ).strip().lower()
    artifact_file_sha = str(
        payload.get("completed_task_result_artifact_file_sha256") or ""
    ).strip().lower()
    artifact_path = Path(
        str(payload.get("completed_task_result_artifact_path") or "")
    )
    sealed_artifact = sealed_result.get("completed_result_artifact")
    sealed_sha = str(
        sealed_result.get("completed_result_artifact_sha256") or ""
    ).strip().lower()
    if (
        payload.get("completed_task_result_artifact_saved") is not True
        or not isinstance(artifact, Mapping)
        or not isinstance(sealed_artifact, Mapping)
        or dict(artifact) != dict(sealed_artifact)
        or re.fullmatch(r"[0-9a-f]{64}", artifact_sha) is None
        or artifact_sha != sealed_sha
        or artifact_file_sha != artifact_sha
        or _canonical_json_sha256(artifact) != artifact_sha
        or not str(artifact_path)
        or not _confined_regular_file(
            artifact_path, allowed_root=allowed_root,
            expected_path=expected_path,
        )
    ):
        return False, "completed_task_result_artifact_invalid"
    try:
        persisted = _load_strict_json(artifact_path)
    except Exception:
        return False, "completed_task_result_artifact_unreadable"
    if (
        not isinstance(persisted, Mapping)
        or dict(persisted) != dict(artifact)
        or _sha256_file(artifact_path) != artifact_sha
    ):
        return False, "completed_task_result_artifact_persistence_mismatch"
    return True, "verified_exact"


def _verified_trt_engine_build_receipt(
    raw: Any, *, source_onnx: Path, engine: Path, trtexec: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Verify persistent evidence from the successful engine build command."""
    if not isinstance(raw, Mapping):
        return None, "engine_build_receipt_missing"
    receipt = dict(raw)
    declared = str(receipt.pop("receipt_sha256", "") or "").strip().lower()
    if (
        re.fullmatch(r"[0-9a-f]{64}", declared) is None
        or _canonical_json_sha256(receipt) != declared
    ):
        return None, "engine_build_receipt_sha256_mismatch"
    receipt["receipt_sha256"] = declared
    if (
        receipt.get("schema") != TRT_ENGINE_BUILD_RECEIPT_SCHEMA
        or receipt.get("schema_version") != TRT_ENGINE_BUILD_RECEIPT_VERSION
        or receipt.get("build_returncode") != 0
        or receipt.get("dry_run") is not False
    ):
        return None, "engine_build_receipt_schema_or_status_invalid"
    if not source_onnx.is_file() or not engine.is_file() or not trtexec.is_file():
        return None, "engine_build_receipt_artifact_missing"
    expected = {
        "source_onnx": str(source_onnx.resolve()),
        "source_onnx_sha256": _sha256_file(source_onnx),
        "engine": str(engine.resolve()),
        "engine_sha256": _sha256_file(engine),
        "trtexec": str(trtexec.resolve()),
        "trtexec_sha256": _sha256_file(trtexec),
    }
    if any(str(receipt.get(key) or "") != value for key, value in expected.items()):
        return None, "engine_build_receipt_artifact_binding_mismatch"
    command = receipt.get("command")
    if not isinstance(command, list) or not command:
        return None, "engine_build_receipt_command_missing"
    argv = [str(value) for value in command]
    try:
        command_executable = str(Path(argv[0]).expanduser().resolve())
    except Exception:
        return None, "engine_build_receipt_trtexec_command_mismatch"
    if command_executable != str(trtexec.resolve()):
        return None, "engine_build_receipt_trtexec_command_mismatch"
    onnx_args = [value for value in argv[1:] if value.startswith("--onnx=")]
    engine_args = [value for value in argv[1:] if value.startswith("--saveEngine=")]
    if (
        onnx_args != [f"--onnx={source_onnx.resolve()}"]
        or engine_args != [f"--saveEngine={engine.resolve()}"]
    ):
        return None, "engine_build_receipt_source_or_engine_command_mismatch"
    return receipt, "engine_build_receipt_verified"


def _quality_first_trt_producer_identity(
    benchmark_set: Path, model: str, ns: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str, Path]:
    """Load and re-verify the exact setup-local Quality-FIRST engine identity."""
    setup_id = str(getattr(ns, "setup_id", "") or "").strip()
    root_text = str(getattr(ns, "root", "") or "").strip()
    root = _canonical_lexical_absolute_path(root_text)
    producer_file_text = str(
        getattr(ns, "trt_quality_producer_json", "") or ""
    ).strip()
    producer_file = _absolute_without_resolving(
        Path(producer_file_text)
    ) if producer_file_text else Path("")
    if not producer_file_text:
        return None, "quality_first_producer_set_missing", producer_file
    expected_producer_file = (
        root / "quality_first" / "tensorrt_quality_producer_set.json"
        if root is not None else Path("")
    )
    if (
        root is None
        or not _confined_regular_file(
            producer_file, allowed_root=root,
            expected_path=expected_producer_file,
        )
    ):
        return (
            None, "quality_first_producer_set_role_path_mismatch",
            producer_file,
        )
    producer_file = producer_file.resolve(strict=True)
    try:
        producer_set = _load_strict_json(producer_file)
    except Exception:
        return None, "quality_first_producer_set_invalid_json", producer_file
    if not isinstance(producer_set, Mapping):
        return None, "quality_first_producer_set_invalid", producer_file
    producer_set = dict(producer_set)
    producers = producer_set.get("producers_by_model")
    eval_run_id = str(producer_set.get("eval_run_id") or "").strip()
    if (
        producer_set.get("schema") != TRT_QUALITY_PRODUCER_SET_SCHEMA
        or int(producer_set.get("schema_version") or 0)
        != TRT_QUALITY_PRODUCER_SET_VERSION
        or not eval_run_id
        or eval_run_id != root.name
        or str(producer_set.get("setup_id") or "").strip() != setup_id
        or not setup_id
        or not isinstance(producers, Mapping)
    ):
        return None, "quality_first_producer_set_identity_invalid", producer_file
    producer = producers.get(str(model))
    if not isinstance(producer, Mapping):
        return None, "quality_first_model_producer_missing", producer_file
    producer = dict(producer)
    declared_producer_sha = str(
        producer.get("producer_identity_sha256") or ""
    ).strip().lower()
    unhashed = dict(producer)
    unhashed.pop("producer_identity_sha256", None)
    if (
        producer.get("schema")
        != "onnx-splitpoint/tensorrt-central-quality-producer-identity"
        or int(producer.get("schema_version") or 0) != 1
        or producer.get("execution_role") != "full_quality_only"
        or producer.get("backend") != "native_tensorrt"
        or producer.get("variant") != "full"
        or producer.get("case_id") != "full"
        or producer.get("source_run_id") != "native_full_tensorrt"
        or str(producer.get("model_id") or "") != str(model)
        or str(producer.get("setup_id") or "") != setup_id
        or str(producer.get("eval_run_id") or "") != eval_run_id
        or producer.get("performance_claims_emitted") is not False
        or re.fullmatch(r"[0-9a-f]{64}", declared_producer_sha) is None
        or _canonical_json_sha256(unhashed) != declared_producer_sha
    ):
        return None, "quality_first_producer_identity_invalid", producer_file

    artifact_names = ("source_onnx", "build_onnx", "engine", "trtexec")
    artifacts: dict[str, dict[str, Any]] = {}
    for name in artifact_names:
        raw_artifact = producer.get(name)
        if not isinstance(raw_artifact, Mapping):
            return None, f"quality_first_{name}_missing", producer_file
        artifact = dict(raw_artifact)
        raw_path = str(artifact.get("path") or "").strip()
        candidate = Path(raw_path).expanduser()
        path = candidate.resolve()
        expected_sha = str(artifact.get("sha256") or "").strip().lower()
        try:
            expected_size = int(artifact.get("size_bytes"))
        except Exception:
            return None, f"quality_first_{name}_size_invalid", producer_file
        if (
            not raw_path or not candidate.is_absolute() or raw_path != str(path)
            or not path.is_file()
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
            or _sha256_file(path) != expected_sha
            or int(path.stat().st_size) != expected_size
        ):
            return None, f"quality_first_{name}_artifact_mismatch", producer_file
        artifacts[name] = {
            **artifact, "path": str(path), "sha256": expected_sha,
            "size_bytes": expected_size,
        }
    source_sha = artifacts["source_onnx"]["sha256"]
    build_sha = artifacts["build_onnx"]["sha256"]
    engine_sha = artifacts["engine"]["sha256"]
    if (
        source_sha != build_sha
        or str(artifacts["build_onnx"].get("source_onnx_sha256") or "") != source_sha
        or str(artifacts["engine"].get("source_onnx_sha256") or "") != source_sha
        or str(artifacts["engine"].get("build_onnx_sha256") or "") != build_sha
    ):
        return None, "quality_first_model_binding_invalid", producer_file
    # Suite clones are disposable.  All model/engine evidence named by the
    # central producer must live outside that clone before performance starts.
    suite_root = Path(benchmark_set).resolve()
    for name in ("source_onnx", "build_onnx", "engine"):
        try:
            Path(artifacts[name]["path"]).relative_to(suite_root)
        except ValueError:
            continue
        return None, "quality_first_artifact_not_persistent", producer_file

    receipt_binding = producer.get("engine_build_receipt")
    if not isinstance(receipt_binding, Mapping):
        return None, "quality_first_receipt_binding_missing", producer_file
    receipt_binding = dict(receipt_binding)
    raw_receipt_path = str(receipt_binding.get("path") or "").strip()
    receipt_candidate = Path(raw_receipt_path).expanduser()
    receipt_path = receipt_candidate.resolve()
    receipt = receipt_binding.get("receipt")
    verified_receipt, receipt_status = _verified_trt_engine_build_receipt(
        receipt,
        source_onnx=Path(artifacts["build_onnx"]["path"]),
        engine=Path(artifacts["engine"]["path"]),
        trtexec=Path(artifacts["trtexec"]["path"]),
    )
    if (
        verified_receipt is None or not raw_receipt_path
        or not receipt_candidate.is_absolute()
        or raw_receipt_path != str(receipt_path)
        or not receipt_path.is_file()
    ):
        return None, f"quality_first_{receipt_status}", producer_file
    try:
        receipt_path.relative_to(suite_root)
    except ValueError:
        pass
    else:
        return None, "quality_first_receipt_not_persistent", producer_file
    canonical_receipt = json.dumps(
        verified_receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    outer_sha = _canonical_json_sha256(verified_receipt)
    file_sha = _sha256_file(receipt_path)
    inner_sha = str(verified_receipt.get("receipt_sha256") or "").strip().lower()
    signed_file_sha = str(
        producer.get("engine_build_receipt_file_sha256") or ""
    ).strip().lower()
    try:
        persisted_receipt = _load_strict_json(receipt_path)
    except Exception:
        persisted_receipt = None
    if (
        str(receipt_binding.get("sha256") or "").strip().lower() != outer_sha
        or int(receipt_binding.get("size_bytes") or 0) != len(canonical_receipt)
        or re.fullmatch(r"[0-9a-f]{64}", signed_file_sha) is None
        or file_sha != signed_file_sha
        or not isinstance(persisted_receipt, Mapping)
        or dict(persisted_receipt) != dict(verified_receipt)
    ):
        return None, "quality_first_receipt_binding_mismatch", producer_file
    quality_contract = producer.get("quality_contract")
    quality_contract = (
        dict(quality_contract) if isinstance(quality_contract, Mapping) else {}
    )
    quality_contract_sha = str(
        quality_contract.pop("quality_contract_sha256", "") or ""
    ).strip().lower()
    if (
        not quality_contract
        or re.fullmatch(r"[0-9a-f]{64}", quality_contract_sha) is None
        or _canonical_json_sha256(quality_contract) != quality_contract_sha
        or str(producer.get("quality_contract_sha256") or "").strip().lower()
        != quality_contract_sha
    ):
        return None, "quality_first_quality_contract_invalid", producer_file
    endpoint = producer.get("endpoint")
    authority = producer.get("endpoint_authority")
    if not isinstance(endpoint, Mapping) or not isinstance(authority, Mapping):
        return None, "quality_first_endpoint_authority_missing", producer_file
    endpoint_identity = endpoint.get("identity")
    authority_identity = authority.get("identity")
    endpoint_sha = str(endpoint.get("sha256") or "").strip().lower()
    authority_sha = str(authority.get("sha256") or "").strip().lower()
    authority_container_sha = str(
        (authority_identity or {}).get("source_contracts_sha256")
        if isinstance(authority_identity, Mapping) else ""
    ).strip().lower()
    authority_row_sha = str(
        (authority_identity or {}).get("recorded_contract_sha256")
        if isinstance(authority_identity, Mapping) else ""
    ).strip().lower()
    if (
        not isinstance(endpoint_identity, Mapping)
        or not isinstance(authority_identity, Mapping)
        or _canonical_json_sha256(dict(endpoint_identity)) != endpoint_sha
        or _canonical_json_sha256(dict(authority_identity)) != authority_sha
        or authority_identity.get("schema")
        != "onnx-splitpoint/tensorrt-endpoint-authority"
        or int(authority_identity.get("schema_version") or 0) != 1
        or authority_identity.get("graph_binding_source")
        != "authoritative_suite_output_contract_plus_exact_onnx_endpoint:v2"
        or re.fullmatch(r"[0-9a-f]{64}", authority_container_sha) is None
        or re.fullmatch(r"[0-9a-f]{64}", authority_row_sha) is None
        or authority_container_sha == authority_row_sha
        or authority_identity.get("endpoint_contract_complete") is not True
        or authority_identity.get("contract_resolution_status") != "attested"
        or str(authority_identity.get("endpoint_contract_hash") or "")
        != endpoint_sha
        or str(authority_identity.get("full_model_sha256") or "") != source_sha
        or str(authority_identity.get("terminal_model_sha256") or "") != source_sha
        or str(authority_identity.get("stage") or "")
        != str(endpoint_identity.get("stage") or "")
        or str(producer.get("endpoint_contract_hash") or "") != endpoint_sha
        or producer.get("endpoint_contract_complete") is not True
    ):
        return None, "quality_first_endpoint_authority_invalid", producer_file
    endpoint_attestation = authority_identity.get("output_endpoint_attestation")
    if (
        not isinstance(endpoint_attestation, Mapping)
        or endpoint_attestation.get("attested") is not True
        or endpoint_attestation.get("status") != "passed"
        or str(endpoint_attestation.get("stage") or "")
        != str(endpoint_identity.get("stage") or "")
        or str(endpoint_attestation.get("endpoint_contract_hash") or "")
        != endpoint_sha
    ):
        return None, "quality_first_endpoint_attestation_invalid", producer_file
    precision = producer.get("precision")
    precision_identity = (
        precision.get("identity") if isinstance(precision, Mapping) else None
    )
    precision_sha = (
        str(precision.get("sha256") or "").strip().lower()
        if isinstance(precision, Mapping) else ""
    )
    runtime_precision = str(
        (precision_identity or {}).get("runtime_precision_identity")
        if isinstance(precision_identity, Mapping) else ""
    ).strip().lower()
    requested_precision = str(getattr(ns, "trt_precision", "") or "").strip().lower()
    if requested_precision.startswith("uint8_"):
        requested_precision = "fp16"
    if (
        not isinstance(precision_identity, Mapping)
        or _canonical_json_sha256(dict(precision_identity)) != precision_sha
        or str(precision_identity.get("source_onnx_sha256") or "")
        != source_sha
        or str(precision_identity.get("build_onnx_sha256") or "")
        != build_sha
        or str(precision_identity.get("engine_sha256") or "")
        != engine_sha
        or runtime_precision != requested_precision
        or str(producer.get("runtime_precision_identity") or "").strip().lower()
        != runtime_precision
    ):
        return None, "quality_first_precision_mismatch", producer_file
    if (
        str(receipt_binding.get("sha256") or "").strip().lower() != outer_sha
        or str(verified_receipt.get("receipt_sha256") or "").strip().lower()
        != inner_sha
    ):
        return None, "quality_first_receipt_identity_invalid", producer_file
    return producer, "quality_first_identity_verified_exact", producer_file


def _trt_quality_identity_cli_args(
    producer: Mapping[str, Any], *, semantic_dump: bool,
) -> list[str]:
    """Render the exact Quality-FIRST artifact identity for a child runner."""
    source = dict(producer["source_onnx"])
    build = dict(producer["build_onnx"])
    engine = dict(producer["engine"])
    trtexec = dict(producer["trtexec"])
    receipt = dict(producer["engine_build_receipt"])
    receipt_payload = dict(receipt["receipt"])
    trtexec_flag = "--explicit-full-trtexec" if semantic_dump else "--trtexec"
    return [
        "--explicit-full-source-onnx", str(source["path"]),
        "--explicit-full-build-onnx", str(build["path"]),
        "--explicit-full-engine", str(engine["path"]),
        trtexec_flag, str(trtexec["path"]),
        "--explicit-full-build-receipt", str(receipt["path"]),
        "--expected-source-onnx-sha256", str(source["sha256"]),
        "--expected-build-onnx-sha256", str(build["sha256"]),
        "--expected-engine-sha256", str(engine["sha256"]),
        "--expected-trtexec-sha256", str(trtexec["sha256"]),
        "--expected-engine-build-receipt-sha256", str(receipt["sha256"]),
        "--expected-engine-build-receipt-file-sha256", str(
            producer["engine_build_receipt_file_sha256"]
        ),
        "--expected-trt-engine-build-receipt-sha256", str(
            receipt_payload["receipt_sha256"]
        ),
        "--expected-source-onnx-size-bytes", str(source["size_bytes"]),
        "--expected-build-onnx-size-bytes", str(build["size_bytes"]),
        "--expected-engine-size-bytes", str(engine["size_bytes"]),
        "--expected-trtexec-size-bytes", str(trtexec["size_bytes"]),
        "--expected-engine-build-receipt-size-bytes", str(
            receipt["size_bytes"]
        ),
    ]


def _load_full_quality_binding_set(
    ns: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    text = str(
        getattr(ns, "quality_request_binding_set", "") or ""
    ).strip()
    if not text:
        return {}, "not_requested"
    root_text = str(getattr(ns, "root", "") or "").strip()
    root = _canonical_lexical_absolute_path(root_text)
    path = _absolute_without_resolving(Path(text))
    expected_path = (
        root / "quality_first"
        / "vendor_full_quality_request_binding_set.json"
        if root is not None else Path("")
    )
    if (
        root is None
        or not _confined_regular_file(
            path, allowed_root=root, expected_path=expected_path,
        )
    ):
        return {}, "quality_request_binding_set_role_path_mismatch"
    path = path.resolve(strict=True)
    try:
        payload = _load_strict_json(path)
    except Exception:
        return {}, "quality_request_binding_set_invalid_json"
    if not isinstance(payload, Mapping):
        return {}, "quality_request_binding_set_invalid"
    payload = dict(payload)
    bindings = payload.get("bindings_by_backend_model")
    required_keys_raw = payload.get("required_binding_keys")
    required_keys = (
        [str(value) for value in required_keys_raw]
        if isinstance(required_keys_raw, list)
        and all(isinstance(value, str) and value for value in required_keys_raw)
        else []
    )
    declared_set_sha = str(
        payload.get("binding_set_sha256") or ""
    ).strip().lower()
    unhashed_set = dict(payload)
    unhashed_set.pop("binding_set_sha256", None)
    schema_version = payload.get("schema_version")
    common_invalid = (
        payload.get("schema") != FULL_QUALITY_BINDING_SET_SCHEMA
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in FULL_QUALITY_BINDING_SET_VERSIONS
        or str(payload.get("eval_run_id") or "").strip() != root.name
        or str(payload.get("setup_id") or "").strip()
        != str(getattr(ns, "setup_id", "") or "").strip()
        or str(payload.get("comparison_backend") or "").strip().lower()
        != str(getattr(ns, "comparison_backend", "") or "").strip().lower()
        or not isinstance(bindings, Mapping)
        or not required_keys
        or len(required_keys) != len(set(required_keys))
        or re.fullmatch(r"[0-9a-f]{64}", declared_set_sha) is None
        or _canonical_json_sha256(unhashed_set) != declared_set_sha
    )
    if common_invalid:
        return {}, "quality_request_binding_set_identity_invalid"
    binding_keys = set(str(key) for key in bindings)
    required_key_set = set(required_keys)
    if schema_version == 1:
        if (
            payload.get("complete") is not True
            or binding_keys != required_key_set
        ):
            return {}, "quality_request_binding_set_identity_invalid"
    else:
        statuses = payload.get("binding_status_by_backend_model")
        binding_errors = payload.get("binding_errors_by_backend_model")
        if not isinstance(statuses, Mapping) or not isinstance(binding_errors, Mapping):
            return {}, "quality_request_binding_set_status_map_invalid"
        normalized_statuses = {
            str(key): str(value or "").strip().lower()
            for key, value in statuses.items()
        }
        if (
            set(normalized_statuses) != required_key_set
            or any(
                value not in {"verified_exact", "unavailable"}
                for value in normalized_statuses.values()
            )
            or binding_keys != {
                key for key, value in normalized_statuses.items()
                if value == "verified_exact"
            }
            or payload.get("complete") is not bool(
                bool(required_keys)
                and all(
                    value == "verified_exact"
                    for value in normalized_statuses.values()
                )
            )
        ):
            return {}, "quality_request_binding_set_status_map_invalid"
        unavailable_keys = {
            key for key, value in normalized_statuses.items()
            if value == "unavailable"
        }
        if (
            not set(str(key) for key in binding_errors).issubset(unavailable_keys)
            or any(
                not isinstance(binding_errors.get(key), list)
                or not binding_errors.get(key)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in binding_errors.get(key)
                )
                for key in unavailable_keys
            )
        ):
            return {}, "quality_request_binding_set_error_map_invalid"
    for raw_key, raw_binding in bindings.items():
        if not isinstance(raw_binding, Mapping):
            return {}, "quality_request_binding_invalid"
        binding = dict(raw_binding)
        declared_binding_sha = str(
            binding.get("binding_sha256") or ""
        ).strip().lower()
        unhashed_binding = dict(binding)
        unhashed_binding.pop("binding_sha256", None)
        expected_key = (
            f"{binding.get('backend')}|{binding.get('model_id')}"
        )
        if (
            str(raw_key) != expected_key
            or binding.get("schema")
            != "onnx-splitpoint/native-full-quality-request-binding"
            or int(binding.get("schema_version") or 0) != 1
            or str(binding.get("eval_run_id") or "")
            != str(payload.get("eval_run_id") or "")
            or str(binding.get("setup_id") or "")
            != str(payload.get("setup_id") or "")
            or str(binding.get("comparison_backend") or "").strip().lower()
            != str(payload.get("comparison_backend") or "").strip().lower()
            or re.fullmatch(r"[0-9a-f]{64}", declared_binding_sha) is None
            or _canonical_json_sha256(unhashed_binding)
            != declared_binding_sha
        ):
            return {}, "quality_request_binding_identity_invalid"
    return payload, (
        "quality_request_binding_set_verified"
        if payload.get("complete") is True
        else "quality_request_binding_set_verified_partial"
    )


def _attach_full_quality_request_binding(
    row: dict[str, Any],
    full_contract: Mapping[str, Any],
    *,
    model: str,
    ns: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    backend = str(row.get("backend") or "").strip().lower()
    vendor_backend = backend in {
        "native_full_hailo8", "native_full_hailo10h",
        "native_full_deepx",
    }

    def _failed(errors: Sequence[str]) -> tuple[dict[str, Any], str]:
        exclusion_reasons = list(row.get("claim_exclusion_reasons") or [])
        exclusion_reasons.extend(str(value) for value in errors if str(value))
        row.update({
            # Runtime success remains an observation.  Missing/invalid Quality
            # evidence vetoes every scientific axis without rewriting the
            # measured performance result as a hardware failure.
            "quality_request_binding_required": bool(vendor_backend),
            "quality_request_binding_status": "unavailable",
            "quality_request_binding_errors": list(dict.fromkeys(errors)),
            "central_quality_evidence_verified": False,
            "precision_quality_binding_verified": False,
            "task_quality_observation_valid": False,
            "claim_eligible": False,
            "performance_claim_eligible": False,
            "energy_claim_eligible": False,
            "scientific_claim_eligible": False,
            "eligible_for_scientific_claim": False,
            "eligible_for_ranking": False,
            "claim_exclusion_reason": (
                "upstream_vendor_full_quality_binding_missing"
            ),
            "claim_exclusion_reasons": list(dict.fromkeys(exclusion_reasons)),
        })
        return row, "quality_request_binding_failed"

    payload = getattr(ns, "quality_request_binding_set_data", {})
    if not isinstance(payload, Mapping) or not payload:
        if vendor_backend:
            load_status = str(
                getattr(ns, "quality_request_binding_set_load_status", "")
                or ""
            ).strip()
            return _failed([
                load_status
                if load_status not in {"", "not_requested"}
                else "quality_request_binding_set_missing"
            ])
        return row, "not_requested"
    key = f"{backend}|{model}"
    bindings = payload.get("bindings_by_backend_model")
    raw_binding = bindings.get(key) if isinstance(bindings, Mapping) else None
    if raw_binding is None:
        if vendor_backend:
            error_map = payload.get("binding_errors_by_backend_model")
            declared_errors = (
                list(error_map.get(key) or [])
                if isinstance(error_map, Mapping) else []
            )
            return _failed(
                declared_errors or [f"quality_request_binding_key_missing:{key}"]
            )
        return row, "not_available_for_identity"
    binding = dict(raw_binding) if isinstance(raw_binding, Mapping) else {}
    claim_fields = (
        "source_request_sha256",
        "model_sha256",
        "validation_dataset_sha256",
        "validation_dataset_image_ids_sha256",
        "validation_dataset_ground_truth_sha256",
        "task_quality_policy_sha256",
        "quality_contract_sha256",
        "preprocessing_contract_sha256",
        "decoder_contract_sha256",
        "nms_contract_sha256",
        "quality_record_endpoint_contract_sha256",
        "central_quality_result_sha256",
    )
    if backend == "native_full_deepx":
        claim_fields += ("prepared_input_evidence_sha256",)
    malformed = [
        field for field in claim_fields
        if binding.get(field) not in (None, "")
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(binding.get(field) or "").strip().lower(),
        ) is None
    ]
    preprocessing = (
        dict(binding.get("preprocessing_contract") or {})
        if isinstance(binding.get("preprocessing_contract"), Mapping)
        else {}
    )
    preprocessing_identity = (
        dict(preprocessing.get("identity") or {})
        if isinstance(preprocessing.get("identity"), Mapping)
        else {}
    )
    preprocessing_sha = str(
        binding.get("preprocessing_contract_sha256") or ""
    ).strip().lower()
    model_binding = (
        dict(full_contract.get("model_binding") or {})
        if isinstance(full_contract.get("model_binding"), Mapping)
        else {}
    )
    compiled_sha = str(
        model_binding.get("compiled_artifact_sha256") or ""
    ).strip().lower()
    runtime_precision = {
        "native_full_hailo8": "hailo_hef_sha256:",
        "native_full_hailo10h": "hailo_hef_sha256:",
        "native_full_deepx": "deepx_dxnn_sha256:",
    }.get(backend, "") + compiled_sha
    expected_precision = str(
        binding.get("runtime_precision_identity") or ""
    ).strip().lower()
    runtime_binding = (
        dict(full_contract.get("runtime_preprocessing_binding") or {})
        if isinstance(
            full_contract.get("runtime_preprocessing_binding"), Mapping,
        ) else {}
    )
    runtime_preprocessing_identity = (
        dict(runtime_binding.get("identity") or {})
        if isinstance(runtime_binding.get("identity"), Mapping) else {}
    )
    runtime_preprocessing_sha = str(
        runtime_binding.get("sha256") or ""
    ).strip().lower()
    runtime_numeric_identity = (
        dict(runtime_binding.get("runtime_numeric_input_identity") or {})
        if isinstance(
            runtime_binding.get("runtime_numeric_input_identity"), Mapping,
        ) else {}
    )
    runtime_numeric_sha = str(
        runtime_binding.get("runtime_numeric_input_sha256") or ""
    ).strip().lower()
    required_claims = [
        "source_request_sha256", "model_sha256",
        "validation_dataset_sha256",
        "validation_dataset_image_ids_sha256",
        "validation_dataset_ground_truth_sha256",
        "task_quality_policy_sha256",
        "quality_contract_sha256", "preprocessing_contract_sha256",
        "quality_record_endpoint_contract_sha256",
        "central_quality_result_sha256",
    ]
    if str(binding.get("task") or "").strip().lower() == "detection":
        required_claims.extend((
            "decoder_contract_sha256", "nms_contract_sha256",
        ))
    validation_errors = list(malformed)
    if any(not str(binding.get(field) or "") for field in required_claims):
        validation_errors.append("required_claim_hash_missing")
    if (
        not preprocessing_identity
        or str(preprocessing.get("sha256") or "").strip().lower()
        != preprocessing_sha
        or _canonical_json_sha256(preprocessing_identity)
        != preprocessing_sha
    ):
        validation_errors.append("preprocessing_contract_invalid")
    try:
        canonical_preprocessing = canonical_image_preprocessing_contract(
            binding.get("task"),
            list(preprocessing_identity.get("target_hw") or []),
        )
    except (TypeError, ValueError):
        canonical_preprocessing = {}
    if preprocessing_identity != canonical_preprocessing:
        validation_errors.append("preprocessing_contract_not_canonical")
    if str(binding.get("backend") or "").strip().lower() != backend:
        validation_errors.append("backend_mismatch")
    if binding.get("variant") != "full":
        validation_errors.append("variant_mismatch")
    if str(binding.get("model_id") or "") != str(model):
        validation_errors.append("model_mismatch")
    if str(binding.get("setup_id") or "") != str(ns.setup_id or ""):
        validation_errors.append("setup_mismatch")
    if str(binding.get("comparison_backend") or "").strip().lower() != str(
        ns.comparison_backend or ""
    ).strip().lower():
        validation_errors.append("comparison_backend_mismatch")
    if str(model_binding.get("source_onnx_sha256") or "").strip().lower() != str(
        binding.get("model_sha256") or ""
    ).strip().lower():
        validation_errors.append("source_model_sha256_mismatch")
    if not compiled_sha or runtime_precision != expected_precision:
        validation_errors.append("runtime_precision_identity_mismatch")
    if backend.startswith("native_full_hailo"):
        receipt_preprocessing = full_contract.get(
            "hailo_hef_preprocessing_contract"
        )
        receipt_preprocessing = (
            dict(receipt_preprocessing)
            if isinstance(receipt_preprocessing, Mapping) else {}
        )
        if (
            str(
                full_contract.get("hailo_hef_build_receipt_status") or ""
            ) != "hailo_hef_build_receipt_verified_exact"
            or receipt_preprocessing != preprocessing_identity
            or str(
                full_contract.get(
                    "hailo_hef_preprocessing_contract_sha256"
                ) or ""
            ).strip().lower() != preprocessing_sha
            or not str(
                full_contract.get(
                    "hailo_hef_build_receipt_file_sha256"
                ) or ""
            ).strip()
            or not str(
                full_contract.get("hailo_hef_build_receipt_sha256") or ""
            ).strip()
        ):
            validation_errors.append("hailo_hef_build_receipt_mismatch")
    if str(row.get("endpoint_contract_hash") or "").strip().lower() != str(
        binding.get("endpoint_contract_hash") or ""
    ).strip().lower():
        # Full raw tensors can have backend-specific endpoint signatures. Only
        # an explicitly bound completed endpoint plus verified measured host
        # tail permits their quality join. Neither raw hash is overwritten.
        try:
            if binding.get("quality_join_endpoint") != "completed_task_decoded_nms":
                raise ValueError("completed_quality_endpoint_not_bound")
            frozen = verify_frozen_postprocess_contract(row.get("frozen_host_postprocess_contract"))
            comparison = verify_completed_detection_comparison_endpoint_contract(
                row.get("completed_task_comparison_endpoint_contract"), frozen_contract=frozen,
            )
            quality_comparison = verify_completed_detection_comparison_endpoint_contract(
                binding.get("completed_task_endpoint_contract"), frozen_contract=frozen,
            )
            attestation = build_completed_detection_endpoint_attestation(
                frozen, row.get("frozen_host_postprocess_result"),
                completed_frames=int(row.get("completed_frames") or row.get("completed_work_units") or 0),
                postprocess_completed_frames=int(row.get("postprocess_completed_frames") or 0),
                source_endpoint_contract_hash=str(row.get("endpoint_contract_hash") or ""),
            )
            if (comparison != quality_comparison
                    or row.get("completed_task_endpoint_attestation") != attestation
                    or row.get("postprocess_completion_verified") is not True
                    or row.get("completed_task_endpoint_attested") is not True
                    or binding.get("completed_task_endpoint_contract_hash") != comparison["endpoint_contract_hash"]
                    or binding.get("completed_task_output_endpoint_id") != comparison["output_endpoint_id"]
                    or row.get("completed_task_comparison_endpoint_contract_hash") != comparison["endpoint_contract_hash"]
                    or row.get("completed_task_comparison_output_endpoint_id") != comparison["output_endpoint_id"]):
                raise ValueError("completed_quality_endpoint_mismatch")
        except (ValueError, TypeError, KeyError, NameError, FrozenPostprocessError) as exc:
            validation_errors.append("endpoint_contract_hash_mismatch")
            validation_errors.append("completed_quality_endpoint_binding_invalid:" + str(exc))
    if (
        runtime_binding.get("status")
        != "runtime_preprocessing_and_numeric_identity_verified_exact"
        or not runtime_preprocessing_identity
        or runtime_preprocessing_identity != preprocessing_identity
        or runtime_preprocessing_sha != preprocessing_sha
        or _canonical_json_sha256(runtime_preprocessing_identity)
        != runtime_preprocessing_sha
    ):
        validation_errors.append("runtime_preprocessing_contract_mismatch")
    if (
        not runtime_numeric_identity
        or runtime_numeric_identity.get("schema")
        != RUNTIME_NUMERIC_INPUT_SCHEMA
        or int(runtime_numeric_identity.get("schema_version") or 0)
        != RUNTIME_NUMERIC_INPUT_SCHEMA_VERSION
        or re.fullmatch(r"[0-9a-f]{64}", runtime_numeric_sha) is None
        or _canonical_json_sha256(runtime_numeric_identity)
        != runtime_numeric_sha
        or str(
            runtime_numeric_identity.get(
                "preprocessing_contract_sha256"
            ) or ""
        ).strip().lower() != preprocessing_sha
        or str(runtime_numeric_identity.get("backend") or "").strip().lower()
        != backend
    ):
        validation_errors.append("runtime_numeric_input_identity_invalid")
    else:
        try:
            validation_errors.extend(
                runtime_numeric_input_identity_errors(
                    runtime_numeric_identity,
                    runtime_preprocessing_identity,
                )
            )
        except (TypeError, ValueError):
            validation_errors.append("runtime_numeric_input_identity_invalid")
    observed_runtime_fields = {
        "runtime_input_shape": list(row.get("runtime_input_shape") or []),
        "runtime_input_dtype": canonical_runtime_dtype(
            row.get("runtime_input_dtype")
        ),
        "runtime_input_layout": str(
            row.get("runtime_input_layout") or ""
        ).strip().upper(),
        "runtime_color_space": str(
            row.get("runtime_color_space") or ""
        ).strip().upper(),
        "runtime_normalization": str(
            row.get("runtime_normalization") or ""
        ).strip().lower(),
    }
    for field, observed in observed_runtime_fields.items():
        expected = runtime_numeric_identity.get(field)
        if field == "runtime_input_shape":
            expected = list(expected or [])
        elif field in {"runtime_input_layout", "runtime_color_space"}:
            expected = str(expected or "").strip().upper()
        else:
            expected = str(expected or "").strip().lower()
        if observed != expected:
            validation_errors.append(f"observed_{field}_mismatch")
    observed_mode = str(
        row.get("runtime_preprocess_mode") or ""
    ).strip().lower()
    if observed_mode.endswith("_rgb_uint8"):
        observed_mode = observed_mode[: -len("_rgb_uint8")]
    if observed_mode != str(
        runtime_preprocessing_identity.get("preprocess_mode") or ""
    ).strip().lower():
        validation_errors.append("observed_runtime_preprocess_mode_mismatch")
    row_runtime_identity = row.get("runtime_preprocessing_identity")
    row_numeric_identity = row.get("runtime_numeric_input_identity")
    if (
        not isinstance(row_runtime_identity, Mapping)
        or dict(row_runtime_identity) != runtime_preprocessing_identity
        or str(row.get("runtime_preprocessing_sha256") or "").strip().lower()
        != runtime_preprocessing_sha
    ):
        validation_errors.append("row_runtime_preprocessing_identity_mismatch")
    if (
        not isinstance(row_numeric_identity, Mapping)
        or dict(row_numeric_identity) != runtime_numeric_identity
        or str(row.get("runtime_numeric_input_sha256") or "").strip().lower()
        != runtime_numeric_sha
    ):
        validation_errors.append("row_runtime_numeric_input_identity_mismatch")
    if backend == "native_full_deepx":
        join_binding = (
            dict(binding.get("prepared_input_join_binding") or {})
            if isinstance(
                binding.get("prepared_input_join_binding"), Mapping,
            ) else {}
        )
        join_sha = str(
            binding.get("prepared_input_join_binding_sha256") or ""
        ).strip().lower()
        prepared_shape = join_binding.get("prepared_input_shape")
        numeric_shape = list(
            runtime_numeric_identity.get("runtime_input_shape") or []
        )
        numeric_dtype = canonical_runtime_dtype(
            runtime_numeric_identity.get("runtime_input_dtype")
        )
        numeric_layout = str(
            runtime_numeric_identity.get("runtime_input_layout") or ""
        ).strip().upper()
        numeric_name = str(
            runtime_numeric_identity.get("runtime_input_name") or ""
        ).strip()
        numeric_bytes = (
            math.prod(numeric_shape)
            * _RUNTIME_INPUT_DTYPE_BYTES.get(numeric_dtype, 0)
            if numeric_shape else 0
        )
        if (
            join_binding.get("schema")
            != "onnx-splitpoint/deepx-performance-quality-input-binding"
            or int(join_binding.get("schema_version") or 0) != 1
            or join_binding.get("binding_verified") is not True
            or re.fullmatch(r"[0-9a-f]{64}", join_sha) is None
            or _canonical_json_sha256(join_binding) != join_sha
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(binding.get("prepared_input_evidence_sha256") or "")
                .strip().lower(),
            ) is None
            or not str(join_binding.get("source_image_id") or "").strip()
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(join_binding.get("source_image_sha256") or "")
                .strip().lower(),
            ) is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(join_binding.get("prepared_input_sha256") or "")
                .strip().lower(),
            ) is None
            or isinstance(join_binding.get("prepared_input_bytes"), bool)
            or not isinstance(join_binding.get("prepared_input_bytes"), int)
            or int(join_binding.get("prepared_input_bytes") or 0) <= 0
            or not str(join_binding.get("prepared_input_name") or "").strip()
            or not isinstance(prepared_shape, list)
            or not prepared_shape
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in (prepared_shape or [])
            )
            or not str(join_binding.get("prepared_input_dtype") or "").strip()
            or not str(join_binding.get("prepared_input_layout") or "").strip()
            or str(join_binding.get("prepared_input_name") or "").strip()
            != numeric_name
            or list(prepared_shape or []) != numeric_shape
            or canonical_runtime_dtype(
                join_binding.get("prepared_input_dtype")
            ) != numeric_dtype
            or str(
                join_binding.get("prepared_input_layout") or ""
            ).strip().upper() != numeric_layout
            or int(join_binding.get("prepared_input_bytes") or 0)
            != numeric_bytes
            or str(
                join_binding.get("runtime_preprocessing_sha256") or ""
            ).strip().lower() != runtime_preprocessing_sha
            or str(
                join_binding.get("runtime_numeric_input_sha256") or ""
            ).strip().lower() != runtime_numeric_sha
        ):
            validation_errors.append("deepx_quality_input_join_binding_invalid")
        observed_prepared = {
            "source_image_id": str(
                row.get("prepared_input_source_image_id") or ""
            ).strip(),
            "source_image_sha256": str(
                row.get("prepared_input_source_image_sha256") or ""
            ).strip().lower(),
            "prepared_input_sha256": str(
                row.get("prepared_input_sha256") or ""
            ).strip().lower(),
            "prepared_input_bytes": row.get("prepared_input_bytes"),
            "prepared_input_name": str(
                row.get("prepared_input_name") or ""
            ).strip(),
            "prepared_input_shape": list(
                row.get("prepared_input_shape") or []
            ),
            "prepared_input_dtype": str(
                row.get("prepared_input_dtype") or ""
            ).strip().lower(),
            "prepared_input_layout": str(
                row.get("prepared_input_layout") or ""
            ).strip().upper(),
        }
        expected_prepared = {
            "source_image_id": str(
                join_binding.get("source_image_id") or ""
            ).strip(),
            "source_image_sha256": str(
                join_binding.get("source_image_sha256") or ""
            ).strip().lower(),
            "prepared_input_sha256": str(
                join_binding.get("prepared_input_sha256") or ""
            ).strip().lower(),
            "prepared_input_bytes": join_binding.get(
                "prepared_input_bytes"
            ),
            "prepared_input_name": str(
                join_binding.get("prepared_input_name") or ""
            ).strip(),
            "prepared_input_shape": list(prepared_shape or []),
            "prepared_input_dtype": str(
                join_binding.get("prepared_input_dtype") or ""
            ).strip().lower(),
            "prepared_input_layout": str(
                join_binding.get("prepared_input_layout") or ""
            ).strip().upper(),
        }
        for field, observed in observed_prepared.items():
            if observed != expected_prepared[field]:
                validation_errors.append(
                    f"quality_performance_{field}_mismatch"
                )
    if validation_errors:
        return _failed(validation_errors)
    for field in claim_fields:
        row[field] = str(binding.get(field) or "")
    row.update({
        "quality_request_binding": binding,
        "quality_request_binding_sha256": str(
            binding.get("binding_sha256") or ""
        ),
        "quality_request_binding_set_sha256": str(
            payload.get("binding_set_sha256") or ""
        ),
        "quality_request_binding_status": "verified_exact",
        "quality_request_binding_errors": [],
        "quality_source_run_id": str(binding.get("source_run_id") or ""),
        "quality_source_case_id": str(binding.get("source_case_id") or ""),
        "preprocessing_contract": preprocessing_identity,
        "runtime_preprocessing_identity": runtime_preprocessing_identity,
        "runtime_preprocessing_sha256": runtime_preprocessing_sha,
        "runtime_numeric_input_identity": runtime_numeric_identity,
        "runtime_numeric_input_sha256": runtime_numeric_sha,
    })
    return row, "quality_request_binding_verified_exact"


def _resolved_executable(invocation_path: str) -> Path | None:
    """Resolve an executable without losing the originally selected spelling."""
    value = str(invocation_path or "").strip()
    if not value:
        return None
    expanded = Path(value).expanduser()
    if "/" in value or expanded.is_absolute():
        candidate = expanded
    else:
        found = shutil.which(value)
        candidate = Path(found) if found else expanded
    try:
        return candidate.resolve(strict=True) if candidate.is_file() else None
    except Exception:
        return None


def _python_interpreter_artifact(invocation_path: str) -> dict[str, Any] | None:
    """Bind the selected Python invocation to its binary and interpreter identity."""
    resolved = _resolved_executable(invocation_path)
    if resolved is None:
        return None
    probe = (
        "import json,platform,sys;"
        "print(json.dumps({'executable':sys.executable,'version':sys.version,"
        "'implementation':platform.python_implementation(),"
        "'cache_tag':getattr(sys.implementation,'cache_tag','')}))"
    )
    try:
        completed = subprocess.run(
            [str(invocation_path), "-c", probe], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        identity = json.loads(completed.stdout.strip()) if completed.returncode == 0 else None
    except Exception:
        identity = None
    if not isinstance(identity, Mapping) or not all(
        str(identity.get(key) or "").strip()
        for key in ("executable", "version", "implementation")
    ):
        return None
    return {
        "path": str(resolved),
        "invocation_path": str(invocation_path),
        "resolved_path": str(resolved),
        "sha256": _sha256_file(resolved),
        "interpreter_identity": dict(identity),
    }


def _attach_python_artifact(
    artifacts: dict[str, dict[str, Any]], name: str, invocation_path: str,
) -> bool:
    artifact = _python_interpreter_artifact(invocation_path)
    if artifact is None:
        return False
    artifacts[name] = artifact
    return True


_NATIVE_FULL_INPUT_SCHEMA = "onnx-splitpoint/native-full-input-dump"
_NATIVE_FULL_INPUT_SCHEMA_VERSIONS = frozenset({1, 2})
_PREVERIFIED_RUNTIME_INPUT_SCHEMA = "onnx-splitpoint/preverified-runtime-input-tensor"
_PREVERIFIED_RUNTIME_INPUT_VERSION = 1
_RUNTIME_INPUT_DTYPE_BYTES = {
    "uint8": 1, "int8": 1,
    "uint16": 2, "int16": 2, "float16": 2,
    "uint32": 4, "int32": 4, "float32": 4,
    "float64": 8,
}


def _validated_runtime_input_manifest(
    manifest_path: Path,
    *,
    allowed_root: Path,
) -> tuple[dict[str, Any] | None, str]:
    """Validate the semantic runtime tensor once, before an energy command is sealed."""
    expected_manifest = allowed_root / "native_full_input_manifest.json"
    expected_tensor = allowed_root / "runtime_input.bin"
    if not _confined_regular_file(
        manifest_path, allowed_root=allowed_root,
        expected_path=expected_manifest,
    ):
        return None, "runtime_input_manifest_missing"
    manifest_path = manifest_path.resolve(strict=True)
    raw = _load_json(manifest_path)
    if not isinstance(raw, Mapping):
        return None, "runtime_input_manifest_json_invalid"
    payload = dict(raw)
    schema = str(payload.get("schema") or "").strip()
    if schema != _NATIVE_FULL_INPUT_SCHEMA:
        return None, "runtime_input_manifest_schema_incompatible"
    version = payload.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in _NATIVE_FULL_INPUT_SCHEMA_VERSIONS
    ):
        return None, "runtime_input_manifest_schema_version_incompatible"
    manifest_backend = str(payload.get("backend") or "").strip().lower()
    if manifest_backend == "native_full_deepx" and version != 2:
        return None, "runtime_input_manifest_schema_version_incompatible"
    name = str(payload.get("runtime_input_name") or "").strip()
    if not name:
        return None, "runtime_input_name_missing"
    shape_raw = payload.get("runtime_input_shape")
    if (
        not isinstance(shape_raw, (list, tuple))
        or not shape_raw
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape_raw)
    ):
        return None, "runtime_input_shape_invalid"
    shape = [int(dim) for dim in shape_raw]
    dtype_name = str(payload.get("runtime_input_dtype") or "").strip().lower()
    if dtype_name not in _RUNTIME_INPUT_DTYPE_BYTES:
        return None, "runtime_input_dtype_unsupported"
    declared_bytes = payload.get("runtime_input_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) or declared_bytes <= 0:
        return None, "runtime_input_byte_count_invalid"
    expected_bytes = math.prod(shape) * int(_RUNTIME_INPUT_DTYPE_BYTES[dtype_name])
    if int(declared_bytes) != expected_bytes:
        return None, "runtime_input_shape_dtype_byte_count_mismatch"
    tensor_text = str(payload.get("runtime_input_file") or "").strip()
    if not tensor_text:
        return None, "runtime_input_file_missing"
    tensor_path = Path(tensor_text).expanduser()
    if not tensor_path.is_absolute():
        tensor_path = manifest_path.parent / tensor_path
    if not _confined_regular_file(
        tensor_path, allowed_root=allowed_root,
        expected_path=expected_tensor,
    ):
        return None, "runtime_input_file_missing"
    tensor_path = tensor_path.resolve(strict=True)
    if int(tensor_path.stat().st_size) != expected_bytes:
        return None, "runtime_input_file_size_mismatch"
    declared_sha = str(payload.get("runtime_input_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None:
        return None, "runtime_input_sha256_invalid"
    if _sha256_file(tensor_path) != declared_sha:
        return None, "runtime_input_sha256_mismatch"
    preprocess = payload.get("preprocess")
    if preprocess is not None and not isinstance(preprocess, Mapping):
        return None, "runtime_input_preprocess_invalid"
    preprocess = dict(preprocess or {})
    runtime_layout = str(
        payload.get("runtime_input_layout")
        or preprocess.get("layout")
        or ""
    ).strip().upper()
    runtime_color = str(
        payload.get("runtime_color_space")
        or ("RGB" if preprocess.get("rgb") is True else "")
        or ""
    ).strip().upper()
    runtime_normalization = str(
        payload.get("runtime_normalization")
        or preprocess.get("normalization")
        or preprocess.get("ort_model_scale")
        or ""
    ).strip().lower()
    semantic_raw = payload.get("runtime_preprocessing_identity")
    numeric_raw = payload.get("runtime_numeric_input_identity")
    semantic_sha = str(
        payload.get("runtime_preprocessing_sha256") or ""
    ).strip().lower()
    numeric_sha = str(
        payload.get("runtime_numeric_input_sha256") or ""
    ).strip().lower()
    preprocessing_binding_status = "runtime_preprocessing_identity_missing"
    semantic_identity: dict[str, Any] = {}
    numeric_identity: dict[str, Any] = {}
    if any((semantic_raw, numeric_raw, semantic_sha, numeric_sha)):
        if not isinstance(semantic_raw, Mapping):
            return None, "runtime_preprocessing_identity_invalid"
        semantic_identity = dict(semantic_raw)
        if (
            re.fullmatch(r"[0-9a-f]{64}", semantic_sha) is None
            or preprocessing_contract_sha256(semantic_identity)
            != semantic_sha
        ):
            return None, "runtime_preprocessing_identity_sha256_mismatch"
        if not isinstance(numeric_raw, Mapping):
            return None, "runtime_numeric_input_identity_invalid"
        numeric_identity = dict(numeric_raw)
        if (
            re.fullmatch(r"[0-9a-f]{64}", numeric_sha) is None
            or _canonical_json_sha256(numeric_identity) != numeric_sha
        ):
            return None, "runtime_numeric_input_identity_sha256_mismatch"
        try:
            numeric_errors = runtime_numeric_input_identity_errors(
                numeric_identity, semantic_identity,
            )
        except (TypeError, ValueError):
            numeric_errors = ["runtime_numeric_input_identity_invalid"]
        if numeric_errors:
            return None, ";".join(numeric_errors)
        manifest_task = str(payload.get("task") or "").strip().lower()
        expected_numeric = {
            "backend": manifest_backend,
            "task": manifest_task,
            "preprocessing_contract_sha256": semantic_sha,
            "runtime_input_name": name,
            "runtime_input_shape": shape,
            "runtime_input_dtype": canonical_runtime_dtype(dtype_name),
            "runtime_input_layout": runtime_layout,
            "runtime_color_space": runtime_color,
            "runtime_normalization": runtime_normalization,
        }
        for field, expected in expected_numeric.items():
            observed = numeric_identity.get(field)
            if field in {"backend", "task", "preprocessing_contract_sha256"}:
                observed = str(observed or "").strip().lower()
            elif field in {"runtime_input_layout", "runtime_color_space"}:
                observed = str(observed or "").strip().upper()
            elif field in {"runtime_input_dtype", "runtime_normalization"}:
                observed = str(observed or "").strip().lower()
            elif field == "runtime_input_shape":
                observed = list(observed or [])
            if observed != expected:
                return None, f"runtime_numeric_input_{field}_mismatch"
        if str(semantic_identity.get("color_space") or "").strip().upper() != "RGB":
            return None, "runtime_preprocessing_color_space_mismatch"
        if str(semantic_identity.get("input_domain") or "").strip().lower() != "uint8_0_255":
            return None, "runtime_preprocessing_input_domain_mismatch"
        try:
            runtime_hw = list(target_hw_from_shape(shape))
        except (TypeError, ValueError):
            return None, "runtime_input_target_hw_invalid"
        if runtime_hw != list(semantic_identity.get("target_hw") or []):
            return None, "runtime_preprocessing_target_hw_mismatch"
        preprocessing_binding_status = (
            "runtime_preprocessing_and_numeric_identity_verified_exact"
        )
    elif manifest_backend == "native_full_deepx":
        return None, "runtime_preprocessing_identity_missing"
    source_image_text = str(payload.get("input_image") or "").strip()
    source_image_path = Path(source_image_text).expanduser()
    source_image_sha = str(
        payload.get("input_image_sha256") or ""
    ).strip().lower()
    if manifest_backend == "native_full_deepx" and (
        not source_image_text
        or _path_contains_symlink(source_image_path)
        or not source_image_path.is_file()
        or re.fullmatch(r"[0-9a-f]{64}", source_image_sha) is None
        or _sha256_file(source_image_path) != source_image_sha
    ):
        return None, "runtime_input_source_image_binding_invalid"
    validated = {
        "runtime_input_file": str(tensor_path),
        "runtime_input_name": name,
        "runtime_input_shape": shape,
        "runtime_input_dtype": dtype_name,
        "runtime_input_bytes": int(declared_bytes),
        "runtime_input_sha256": declared_sha,
        "runtime_input_layout": runtime_layout,
        "preprocess": preprocess,
        "input_image": str(source_image_path) if source_image_text else "",
        "input_image_sha256": source_image_sha,
    }
    if semantic_identity and numeric_identity:
        validated.update({
            "runtime_color_space": runtime_color,
            "runtime_normalization": runtime_normalization,
            "runtime_preprocessing_identity": semantic_identity,
            "runtime_preprocessing_sha256": semantic_sha,
            "runtime_numeric_input_identity": numeric_identity,
            "runtime_numeric_input_sha256": numeric_sha,
            "runtime_preprocessing_binding_status": (
                preprocessing_binding_status
            ),
        })
    return validated, "runtime_input_manifest_and_tensor_verified"


def _full_command_contract(
    *, row: Mapping[str, Any], root: Path, benchmark_set: Path,
    model: str, backend_arg: str, ns: argparse.Namespace,
) -> dict[str, Any]:
    """Seal the successful Full runner configuration for energy replay."""
    input_image = str(row.get("input_image") or "").strip()
    input_sha = str(row.get("input_image_sha256") or "").strip().lower()
    input_case = str(row.get("input_case") or "").strip()
    image_map: dict[str, dict[str, str]] = {}
    if input_image and input_case:
        image_map = {str(model): {input_case: input_image}}
    artifacts: dict[str, dict[str, Any]] = {}
    command_python_bound = _attach_python_artifact(
        artifacts, "command_python_executable", str(sys.executable),
    )
    hef_path = Path(str(row.get("hef_path") or "")).expanduser()
    if hef_path.is_file():
        artifacts["hef"] = {"path": str(hef_path), "sha256": _sha256_file(hef_path)}
    hailo_receipt_path = Path(
        str(row.get("hailo_hef_build_receipt_path") or "")
    ).expanduser()
    hailo_source_onnx_path = Path(
        str(row.get("hailo_hef_source_onnx_path") or "")
    ).expanduser()
    if (
        hailo_receipt_path.is_file()
        and _sha256_file(hailo_receipt_path)
        == str(
            row.get("hailo_hef_build_receipt_file_sha256") or ""
        ).strip().lower()
    ):
        artifacts["hailo_hef_build_receipt"] = {
            "path": str(hailo_receipt_path.resolve()),
            "sha256": _sha256_file(hailo_receipt_path),
            "receipt_sha256": str(
                row.get("hailo_hef_build_receipt_sha256") or ""
            ),
            "receipt": dict(
                row.get("hailo_hef_build_receipt") or {}
            ) if isinstance(
                row.get("hailo_hef_build_receipt"), Mapping,
            ) else {},
        }
    if (
        hailo_source_onnx_path.is_file()
        and _sha256_file(hailo_source_onnx_path)
        == str(row.get("source_onnx_sha256") or "").strip().lower()
    ):
        artifacts["source_onnx"] = {
            "path": str(hailo_source_onnx_path.resolve()),
            "sha256": _sha256_file(hailo_source_onnx_path),
            "size_bytes": int(hailo_source_onnx_path.stat().st_size),
        }
    report_path = Path(str(row.get("report") or "")).expanduser()
    if report_path.is_file():
        artifacts["performance_report"] = {
            "path": str(report_path), "sha256": _sha256_file(report_path),
        }
    report_payload = _load_json(report_path) if report_path.is_file() else {}
    report_payload = dict(report_payload) if isinstance(report_payload, Mapping) else {}

    def _verified_frozen_contract(value: Any) -> dict[str, Any]:
        try:
            return verify_frozen_postprocess_contract(value)
        except (FrozenPostprocessError, TypeError, ValueError):
            return {}

    def _attach_frozen_implementation_artifacts(
        contract: Mapping[str, Any],
    ) -> bool:
        declared = contract.get("implementation_artifacts")
        if not isinstance(declared, Mapping) or not declared:
            return False
        attached: dict[str, dict[str, Any]] = {}
        for name, raw in declared.items():
            if not isinstance(raw, Mapping):
                return False
            relative = str(raw.get("relative_path") or "").strip()
            expected = str(raw.get("sha256") or "").strip().lower()
            path = (ROOT / relative).resolve()
            if (
                not relative or not path.is_file()
                or re.fullmatch(r"[0-9a-f]{64}", expected) is None
                or _sha256_file(path) != expected
            ):
                return False
            artifact_name = f"frozen_postprocess_{name}"
            artifacts[artifact_name] = {
                "path": str(path), "sha256": expected,
                "role": "frozen_detection_host_postprocess_implementation",
            }
            attached[str(name)] = {
                "artifact": artifact_name,
                "relative_path": relative,
                "sha256": expected,
            }
        return len(attached) == len(declared)
    input_manifest_path = Path(str(row.get("input_manifest") or "")).expanduser()
    input_manifest_payload = _load_json(input_manifest_path) if input_manifest_path.is_file() else {}
    input_manifest_payload = dict(input_manifest_payload) if isinstance(input_manifest_payload, Mapping) else {}
    input_artifact_root = _native_full_dump_dir(
        benchmark_set, model,
        str(row.get("backend") or backend_arg), ns,
    )
    runtime_input_spec, runtime_input_validation_status = _validated_runtime_input_manifest(
        input_manifest_path,
        allowed_root=input_artifact_root,
    )
    runtime_input_bound = runtime_input_spec is not None
    runtime_input_path = Path(str((runtime_input_spec or {}).get("runtime_input_file") or ""))
    runtime_input_name = str((runtime_input_spec or {}).get("runtime_input_name") or "")
    declared_runtime_input_sha = str((runtime_input_spec or {}).get("runtime_input_sha256") or "")
    runtime_preprocessing_binding = {
        "status": str(
            (runtime_input_spec or {}).get(
                "runtime_preprocessing_binding_status"
            ) or runtime_input_validation_status
        ),
        "identity": dict(
            (runtime_input_spec or {}).get(
                "runtime_preprocessing_identity"
            ) or {}
        ),
        "sha256": str(
            (runtime_input_spec or {}).get(
                "runtime_preprocessing_sha256"
            ) or ""
        ),
        "runtime_numeric_input_identity": dict(
            (runtime_input_spec or {}).get(
                "runtime_numeric_input_identity"
            ) or {}
        ),
        "runtime_numeric_input_sha256": str(
            (runtime_input_spec or {}).get(
                "runtime_numeric_input_sha256"
            ) or ""
        ),
        "source_manifest_artifact": (
            "input_manifest" if input_manifest_path.is_file() else ""
        ),
        "source_runtime_input_artifact": (
            "runtime_input_tensor" if runtime_input_bound else ""
        ),
    }
    if input_manifest_path.is_file():
        artifacts["input_manifest"] = {
            "path": str(input_manifest_path.resolve()),
            "sha256": _sha256_file(input_manifest_path),
        }
    if runtime_input_bound:
        artifacts["runtime_input_tensor"] = {
            "path": str(runtime_input_path.resolve()),
            "sha256": declared_runtime_input_sha,
            "bytes": int((runtime_input_spec or {}).get("runtime_input_bytes") or 0),
        }
    runner_path = Path(__file__).resolve()
    runtime_options = {
        "frames": int(row.get("frames") or ns.frames),
        "duration_s": float(row.get("duration_s") or ns.duration_s or 0.0),
        "warmup": int(ns.warmup),
        "performance_repetitions": int(getattr(ns, "repetitions", 1) or 1),
        "inflight": int(
            row.get("measurement_concurrency") or ns.inflight
        ),
        "trt_precision": str(ns.trt_precision or ""),
        "workspace_mb": int(ns.workspace_mb),
        "engine_build_python": str(
            getattr(ns, "engine_python_selected", "") or ns.engine_build_python or ""
        ),
        "no_shapes": bool(ns.no_shapes),
        "dump_outputs": bool(ns.dump_outputs),
        "diagnostic_deepx_input_probes": bool(ns.diagnostic_deepx_input_probes),
        "image_map": image_map,
    }
    backend_name = str(row.get("backend") or "")
    energy_workload: dict[str, Any] = {
        "available": False,
        "status": "full_energy_hotloop_unavailable",
        "kind": "",
    }
    trt_build_receipt: dict[str, Any] = {}
    trt_build_receipt_status = "not_applicable"
    if backend_name.startswith("native_full_hailo"):
        hotloop_runner = (ROOT / "scripts" / "smoke_hailo10_hef_runner.py").resolve()
        runtime_python = str(row.get("runtime_python") or "")
        runtime_python_bound = _attach_python_artifact(
            artifacts, "runtime_python", runtime_python,
        )
        if hotloop_runner.is_file():
            artifacts["hotloop_runner"] = {
                "path": str(hotloop_runner), "sha256": _sha256_file(hotloop_runner),
            }
        report_input_names_raw = report_payload.get("input_names")
        report_output_names_raw = report_payload.get("output_names")
        report_input_names = (
            [str(value).strip() for value in report_input_names_raw]
            if isinstance(report_input_names_raw, list)
            and all(isinstance(value, str) and value.strip() for value in report_input_names_raw)
            else []
        )
        report_output_names = (
            [str(value).strip() for value in report_output_names_raw]
            if isinstance(report_output_names_raw, list)
            and all(isinstance(value, str) and value.strip() for value in report_output_names_raw)
            else []
        )
        hailo_frozen_contract = _verified_frozen_contract(
            report_payload.get("frozen_host_postprocess_contract")
        )
        hailo_raw_postprocess_required = bool(
            report_payload.get("host_postprocess_frozen") is True
            or (
                str(report_payload.get("task") or row.get("task") or "").strip().lower()
                == "detection"
                and str(row.get("contract_family") or "").strip().lower() == "raw_head"
            )
        )
        hailo_direct_normalization_required = bool(
            str(
                report_payload.get("task") or row.get("task") or ""
            ).strip().lower() == "detection"
            and str(row.get("contract_family") or "").strip().lower()
            == "decoded_nms"
        )
        try:
            hailo_direct_contract = (
                verify_frozen_decoded_nms_normalization_contract(
                    report_payload.get(
                        "frozen_decoded_nms_normalization_contract"
                    )
                )
                if hailo_direct_normalization_required
                and report_payload.get("normalization_frozen") is True
                else {}
            )
        except (FrozenPostprocessError, TypeError, ValueError):
            hailo_direct_contract = {}
        hailo_postprocess_required = bool(
            hailo_raw_postprocess_required
            or hailo_direct_normalization_required
        )
        hailo_completed_frames = int(report_payload.get("completed_frames") or 0)
        hailo_postprocess_completed = int(
            report_payload.get("postprocess_completed_frames") or 0
        )
        hailo_original_wh = report_payload.get("original_image_wh")
        hailo_original_wh = (
            [int(value) for value in hailo_original_wh]
            if isinstance(hailo_original_wh, list)
            and len(hailo_original_wh) == 2
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value > 0
                for value in hailo_original_wh
            )
            else []
        )
        hailo_completion_contract = (
            hailo_frozen_contract or hailo_direct_contract
        )
        hailo_frozen_implementation_bound = bool(
            hailo_completion_contract
            and _attach_frozen_implementation_artifacts(
                hailo_completion_contract
            )
        )
        hailo_direct_workload_binding = (
            _verified_direct_completed_task_workload(
                row,
                hailo_direct_contract,
                completed_frames=hailo_completed_frames,
                postprocess_completed_frames=(
                    hailo_postprocess_completed
                ),
            )
            if hailo_direct_normalization_required
            and hailo_direct_contract else {}
        )
        hailo_raw_binding_ok = bool(
            hailo_raw_postprocess_required
            and hailo_frozen_contract
            and report_payload.get("host_postprocess_frozen") is True
            and row.get("host_postprocess_frozen") is True
            and row.get("normalization_frozen") is not True
            and str(
                row.get("frozen_host_postprocess_contract_sha256")
                or ""
            ) == str(
                hailo_frozen_contract.get("contract_sha256") or ""
            )
        )
        hailo_postprocess_binding_ok = bool(
            not hailo_postprocess_required
            or (
                hailo_frozen_implementation_bound
                and report_payload.get("postprocess_included") is True
                and row.get("postprocess_included") is True
                and row.get("postprocess_completion_verified") is True
                and str(
                    row.get(
                        "completed_task_result_artifact_verification_status"
                    ) or ""
                ) == "verified_exact"
                and hailo_completed_frames > 0
                and hailo_postprocess_completed == hailo_completed_frames
                and bool(hailo_original_wh)
                and hailo_original_wh == list(
                    hailo_completion_contract.get("original_wh") or []
                )
                and (
                    hailo_raw_binding_ok
                    or bool(
                        hailo_direct_workload_binding
                        and row.get(
                            "direct_source_endpoint_binding_verified"
                        ) is True
                    )
                )
            )
        )
        spec_shape = list((runtime_input_spec or {}).get("runtime_input_shape") or [])
        report_shape = report_payload.get("runtime_input_shape")
        report_shape = list(report_shape) if isinstance(report_shape, list) else []
        spec_dtype = str((runtime_input_spec or {}).get("runtime_input_dtype") or "")
        report_dtype = str(report_payload.get("runtime_input_dtype") or "").strip().lower()
        expected_hailo_dtype = "uint8" if report_payload.get("quantized_inputs") is True else "float32"
        hailo_runtime_identity_bound = bool(
            runtime_input_bound
            and len(report_input_names) == 1
            and report_input_names == [runtime_input_name]
            and bool(report_output_names)
            and len(set(report_output_names)) == len(report_output_names)
            and report_shape == spec_shape
            and report_dtype == spec_dtype == expected_hailo_dtype
        )
        hailo_energy_available = bool(
            hotloop_runner.is_file()
            and runtime_python_bound
            and input_image and re.fullmatch(r"[0-9a-f]{64}", input_sha) is not None
            and isinstance(artifacts.get("hef"), Mapping)
            and isinstance(
                artifacts.get("hailo_hef_build_receipt"), Mapping,
            )
            and isinstance(artifacts.get("source_onnx"), Mapping)
            and hailo_runtime_identity_bound
            and hailo_postprocess_binding_ok
        )
        preverified_runtime_contract = {
            "schema": _PREVERIFIED_RUNTIME_INPUT_SCHEMA,
            "schema_version": _PREVERIFIED_RUNTIME_INPUT_VERSION,
            "runtime_input_name": runtime_input_name,
            "runtime_input_shape": spec_shape,
            "runtime_input_dtype": spec_dtype,
            "runtime_input_bytes": int((runtime_input_spec or {}).get("runtime_input_bytes") or 0),
            "runtime_input_sha256": declared_runtime_input_sha,
            "runtime_input_layout": str((runtime_input_spec or {}).get("runtime_input_layout") or ""),
            "preprocess": dict((runtime_input_spec or {}).get("preprocess") or {}),
            "source_manifest_artifact": "input_manifest",
            "source_runtime_input_artifact": "runtime_input_tensor",
        }
        energy_workload = {
            "available": hailo_energy_available,
            "status": "available" if hailo_energy_available else "full_energy_hotloop_unavailable",
            "runtime_input_binding_status": (
                "name_shape_dtype_bytes_and_file_verified"
                if hailo_runtime_identity_bound else runtime_input_validation_status
            ),
            "kind": "hailo_full_hotloop",
            "runtime_python_artifact": "runtime_python",
            "runner_artifact": "hotloop_runner",
            "hef_artifact": "hef",
            "hailo_hef_build_receipt_artifact": (
                "hailo_hef_build_receipt"
            ),
            "source_model_artifact": "source_onnx",
            "input_image": input_image,
            "input_image_sha256": input_sha,
            "input_manifest_artifact": "input_manifest",
            "runtime_input_artifact": "runtime_input_tensor",
            "runtime_input_name": runtime_input_name,
            "runtime_input_shape": spec_shape,
            "runtime_input_dtype": spec_dtype,
            "runtime_input_bytes": int((runtime_input_spec or {}).get("runtime_input_bytes") or 0),
            "runtime_input_layout": str((runtime_input_spec or {}).get("runtime_input_layout") or ""),
            "runtime_input_mode": "exact_semantic_dump_runtime_tensor",
            "input_mode": "exact_semantic_dump_runtime_tensor",
            "runtime_input_contract": preverified_runtime_contract,
            "canonical_input_slot_names": report_input_names,
            "canonical_output_slot_names": report_output_names,
            "preprocess": dict((runtime_input_spec or {}).get("preprocess") or {}),
            "hw_arch": str(report_payload.get("hw_arch") or ("hailo8" if "hailo8" in backend_name else "hailo10h")),
            "runtime_api": str(report_payload.get("runtime_api") or ("vstreams" if "hailo8" in backend_name else "infer_model")),
            "task": str(report_payload.get("task") or row.get("task") or ""),
            "e2e_scope": (
                "full_task_pipeline" if hailo_postprocess_required
                else "accelerator_output_endpoint"
            ),
            "postprocess_required": hailo_postprocess_required,
            "postprocess_included": bool(
                report_payload.get("postprocess_included") is True
            ),
            "postprocess_completion_verified": bool(
                row.get("postprocess_completion_verified") is True
            ),
            "postprocess_completed_frames": (
                hailo_postprocess_completed
            ),
            "host_postprocess_frozen": bool(
                hailo_raw_postprocess_required
                and hailo_frozen_contract
            ),
            "normalization_frozen": bool(
                hailo_direct_normalization_required
                and hailo_direct_contract
            ),
            "frozen_postprocess_implementation_bound": hailo_frozen_implementation_bound,
            "successful_run_postprocess_completed_frames": hailo_postprocess_completed,
            "successful_run_completed_frames": hailo_completed_frames,
            "original_image_wh": hailo_original_wh,
            "preprocess_mode": str(getattr(ns, "preprocess_mode", "auto") or "auto"),
            "letterbox_pad_value": int(getattr(ns, "letterbox_pad_value", 114)),
            "warmup": 0,
            "successful_run_warmup": int(ns.warmup),
            "inflight": int(row.get("measurement_concurrency") or 1),
            "quantized_inputs": bool(report_payload.get("quantized_inputs")),
            "quantized_outputs": bool(report_payload.get("quantized_outputs")),
            "persistent_activation": bool(report_payload.get("persistent_activation", True)),
            "hotloop": bool(report_payload.get("hotloop", True)),
            "copy_inputs": bool(report_payload.get("copy_inputs", True)),
            "copy_outputs": bool(report_payload.get("copy_outputs", True)),
            **hailo_direct_workload_binding,
        }
        if hailo_raw_postprocess_required:
            energy_workload.update({
                "frozen_postprocess_contract": (
                    hailo_frozen_contract
                ),
                "frozen_postprocess_contract_sha256": str(
                    hailo_frozen_contract.get("contract_sha256")
                    or ""
                ),
                "completed_task_completion_mode": (
                    "frozen_host_tail"
                ),
            })
    elif backend_name == "native_full_tensorrt":
        quality_first_producer = row.get("quality_first_producer_identity")
        quality_first_producer = (
            dict(quality_first_producer)
            if isinstance(quality_first_producer, Mapping) else {}
        )
        quality_first_producer_sha = str(
            row.get("quality_first_producer_identity_sha256") or ""
        ).strip().lower()
        run_smoke = report_payload.get("run_smoke")
        run_smoke = dict(run_smoke) if isinstance(run_smoke, Mapping) else {}
        successful_argv = [str(value) for value in list(run_smoke.get("cmd") or [])]
        trtexec = Path(successful_argv[0]).expanduser() if successful_argv else Path("")
        engine_arg = next((arg for arg in successful_argv if arg.startswith("--loadEngine=")), "")
        engine = Path(engine_arg.split("=", 1)[1]).expanduser() if engine_arg else Path("")
        source_onnx = Path(str(report_payload.get("onnx") or "")).expanduser()
        receipt_path = Path(
            str(report_payload.get("engine_build_receipt_path") or "")
        ).expanduser()
        raw_receipt = report_payload.get("engine_build_receipt")
        verified_receipt, trt_build_receipt_status = (
            _verified_trt_engine_build_receipt(
                raw_receipt, source_onnx=source_onnx,
                engine=engine, trtexec=trtexec,
            )
        )
        if verified_receipt is not None:
            trt_build_receipt = verified_receipt
            if receipt_path.is_file():
                artifacts["engine_build_receipt"] = {
                    "path": str(receipt_path.resolve()),
                    "sha256": _sha256_file(receipt_path),
                    "file_sha256": _sha256_file(receipt_path),
                    "file_size_bytes": int(receipt_path.stat().st_size),
                    "canonical_sha256": _canonical_json_sha256(
                        verified_receipt
                    ),
                    "canonical_size_bytes": len(json.dumps(
                        verified_receipt, sort_keys=True,
                        separators=(",", ":"), ensure_ascii=False,
                    ).encode("utf-8")),
                    "size_bytes": int(
                        (quality_first_producer.get(
                            "engine_build_receipt"
                        ) or {}).get("size_bytes") or 0
                    ),
                    "receipt_sha256": str(
                        verified_receipt.get("receipt_sha256") or ""
                    ),
                }
            else:
                trt_build_receipt = {}
                trt_build_receipt_status = "engine_build_receipt_file_missing"
        if trtexec.is_file():
            artifacts["trtexec"] = {
                "path": str(trtexec.resolve()), "sha256": _sha256_file(trtexec),
                "size_bytes": int(trtexec.stat().st_size),
            }
        source_onnx_sha256 = ""
        if source_onnx.is_file():
            source_onnx_sha256 = _sha256_file(source_onnx)
            artifacts["source_onnx"] = {
                "path": str(source_onnx.resolve()),
                "sha256": source_onnx_sha256,
                "size_bytes": int(source_onnx.stat().st_size),
                "role": "source_model_used_for_engine_build",
            }
            artifacts["build_onnx"] = {
                "path": str(source_onnx.resolve()),
                "sha256": source_onnx_sha256,
                "size_bytes": int(source_onnx.stat().st_size),
                "role": "exact_onnx_bound_by_engine_build_receipt",
            }
        if engine.is_file():
            artifacts["engine"] = {
                "path": str(engine.resolve()), "sha256": _sha256_file(engine),
                "size_bytes": int(engine.stat().st_size),
            }
            if trt_build_receipt:
                artifacts["engine"]["compiled_from_source_onnx_sha256"] = (
                    source_onnx_sha256
                )
        # The helper report is execution evidence; the complete signed central
        # producer remains the authority.  Reconcile both and retain all three
        # receipt hash domains explicitly.
        producer_receipt = (
            quality_first_producer.get("engine_build_receipt")
            if isinstance(
                quality_first_producer.get("engine_build_receipt"), Mapping
            ) else {}
        )
        producer_inner_receipt = (
            producer_receipt.get("receipt")
            if isinstance(producer_receipt.get("receipt"), Mapping) else {}
        )
        producer_artifact_match = bool(
            quality_first_producer
            and re.fullmatch(r"[0-9a-f]{64}", quality_first_producer_sha)
            and quality_first_producer.get("producer_identity_sha256")
            == quality_first_producer_sha
            and str((quality_first_producer.get("build_onnx") or {}).get("path") or "")
            == str(source_onnx.resolve())
            and int((quality_first_producer.get("build_onnx") or {}).get("size_bytes") or 0)
            == int(source_onnx.stat().st_size)
            and str((quality_first_producer.get("engine") or {}).get("path") or "")
            == str(engine.resolve())
            and int((quality_first_producer.get("engine") or {}).get("size_bytes") or 0)
            == int(engine.stat().st_size)
            and str((quality_first_producer.get("trtexec") or {}).get("path") or "")
            == str(trtexec.resolve())
            and int((quality_first_producer.get("trtexec") or {}).get("size_bytes") or 0)
            == int(trtexec.stat().st_size)
            and str(producer_receipt.get("path") or "")
            == str(receipt_path.resolve())
            and str(producer_receipt.get("sha256") or "")
            == _canonical_json_sha256(verified_receipt or {})
            and int(producer_receipt.get("size_bytes") or 0)
            == len(json.dumps(
                verified_receipt or {}, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False,
            ).encode("utf-8"))
            and str(
                quality_first_producer.get(
                    "engine_build_receipt_file_sha256"
                ) or ""
            ) == _sha256_file(receipt_path)
            and dict(producer_inner_receipt) == dict(verified_receipt or {})
        ) if receipt_path.is_file() else False
        invariant_args = [
            arg for arg in successful_argv[1:]
            if not arg.startswith((
                "--iterations=", "--duration=", "--warmUp=", "--exportTimes=",
                "--loadInputs=",
            ))
        ]
        successful_warmup_ms = next((
            int(arg.split("=", 1)[1]) for arg in successful_argv[1:]
            if arg.startswith("--warmUp=")
        ), 0)
        common_trt_available = bool(
            trtexec.is_file() and engine.is_file()
            and bool(source_onnx_sha256)
            and bool(trt_build_receipt)
            and isinstance(artifacts.get("engine_build_receipt"), Mapping)
            and runtime_input_bound
            and bool(successful_argv)
            and bool(run_smoke.get("returncode") == 0)
            and producer_artifact_match
        )
        trt_contract_family = str(
            row.get("contract_family") or ""
        ).strip().lower()
        trt_raw_completion_required = bool(
            str(row.get("task") or "").strip().lower() == "detection"
            and trt_contract_family in {"raw_head", "decoded_pre_nms"}
        )
        trt_direct_completion_required = bool(
            str(row.get("task") or "").strip().lower() == "detection"
            and trt_contract_family == "decoded_nms"
            and row.get("normalization_frozen") is True
        )
        trt_completed_task_required = bool(
            trt_raw_completion_required
            or trt_direct_completion_required
            or str(row.get("task") or "").strip().lower() == "classification"
        )
        if trt_completed_task_required:
            classification = str(row.get("task") or "").strip().lower() == "classification"
            completed_runner = (
                ROOT / "scripts" / "native_trt_full_completed_hotloop.py"
            ).resolve()
            runtime_python_bound = _attach_python_artifact(
                artifacts,
                "runtime_python",
                str(getattr(ns, "engine_python_selected", "") or ""),
            )
            if completed_runner.is_file():
                artifacts["hotloop_runner"] = {
                    "path": str(completed_runner),
                    "sha256": _sha256_file(completed_runner),
                }
            trt_frozen_contract = _verified_frozen_contract(
                row.get("frozen_host_postprocess_contract")
            ) if trt_raw_completion_required else {}
            try:
                trt_direct_contract = (
                    verify_frozen_decoded_nms_normalization_contract(
                        row.get(
                            "frozen_decoded_nms_normalization_contract"
                        )
                    )
                    if trt_direct_completion_required else {}
                )
            except (FrozenPostprocessError, TypeError, ValueError):
                trt_direct_contract = {}
            trt_frozen_implementation_bound = bool(
                (trt_frozen_contract or trt_direct_contract)
                and _attach_frozen_implementation_artifacts(
                    trt_frozen_contract or trt_direct_contract
                )
            )
            completed_frames = int(
                row.get("completed_frames")
                or row.get("completed_work_units")
                or 0
            )
            postprocess_completed = int(
                row.get("postprocess_completed_frames") or 0
            )
            original_wh = list(
                (
                    trt_frozen_contract
                    or trt_direct_contract
                ).get("original_wh") or []
            )
            direct_workload_binding = (
                _verified_direct_completed_task_workload(
                    row,
                    trt_direct_contract,
                    completed_frames=completed_frames,
                    postprocess_completed_frames=postprocess_completed,
                )
                if trt_direct_completion_required
                and trt_direct_contract else {}
            )
            raw_postprocess_binding_ok = bool(
                trt_raw_completion_required
                and trt_frozen_contract
                and row.get("host_postprocess_frozen") is True
                and row.get("normalization_frozen") is not True
                and str(
                    row.get(
                        "frozen_host_postprocess_contract_sha256"
                    ) or ""
                )
                == str(
                    trt_frozen_contract.get("contract_sha256") or ""
                )
            )
            postprocess_binding_ok = bool(
                trt_frozen_implementation_bound
                and row.get("postprocess_included") is True
                and row.get("postprocess_completion_verified") is True
                and completed_frames > 0
                and postprocess_completed == completed_frames
                and len(original_wh) == 2
                and (
                    raw_postprocess_binding_ok
                    or bool(direct_workload_binding)
                )
            )
            if classification:
                postprocess_binding_ok = bool(
                    row.get("task_complete") is True
                    and row.get("completed_task_stage") == "classification_top1_top5"
                    and row.get("postprocess_included") is True
                    and row.get("postprocess_completion_verified") is True
                    and completed_frames > 0
                    and postprocess_completed == completed_frames
                )
            available = bool(
                common_trt_available
                and completed_runner.is_file()
                and runtime_python_bound
                and postprocess_binding_ok
            )
            energy_workload = {
                "available": available,
                "status": (
                    "available"
                    if available else "full_energy_hotloop_unavailable"
                ),
                "kind": "tensorrt_full_completed_task_hotloop",
                "runtime_python_artifact": "runtime_python",
                "runner_artifact": "hotloop_runner",
                "engine_artifact": "engine",
                "source_model_artifact": "source_onnx",
                "trtexec_artifact": "trtexec",
                "engine_build_receipt_artifact": "engine_build_receipt",
                "engine_build_receipt_status": trt_build_receipt_status,
                "quality_first_producer_identity_sha256":
                    quality_first_producer_sha,
                "quality_first_producer_artifact_match":
                    producer_artifact_match,
                "input_manifest_artifact": "input_manifest",
                "runtime_input_artifact": "runtime_input_tensor",
                "runtime_input_name": runtime_input_name,
                "runtime_input_shape":
                    input_manifest_payload.get("runtime_input_shape") or [],
                "runtime_input_dtype": str(
                    input_manifest_payload.get("runtime_input_dtype") or ""
                ),
                "runtime_input_layout": str(
                    (runtime_input_spec or {}).get("runtime_input_layout")
                    or ""
                ),
                "preprocess": dict(
                    (runtime_input_spec or {}).get("preprocess") or {}
                ),
                "task": "classification" if classification else "detection",
                "input_mode": "exact_semantic_dump_runtime_tensor",
                "input_image": input_image,
                "input_image_sha256": input_sha,
                "e2e_scope": "full_task_pipeline",
                "completed_task_stage": "classification_top1_top5" if classification else "decoded_nms",
                "completed_task_contract_family": "classification_top1_top5" if classification else "decoded_nms",
                "measurement_concurrency": 1,
                "postprocess_required": True,
                "postprocess_included": True,
                "host_postprocess_frozen":
                    trt_raw_completion_required,
                "normalization_frozen":
                    trt_direct_completion_required,
                "frozen_postprocess_implementation_bound":
                    trt_frozen_implementation_bound,
                "successful_run_completed_frames": completed_frames,
                "successful_run_postprocess_completed_frames":
                    postprocess_completed,
                "postprocess_completed_frames": postprocess_completed,
                "postprocess_completion_verified":
                    postprocess_binding_ok,
                "original_image_wh": original_wh,
                "source_endpoint_contract_hash": str(
                    row.get("endpoint_contract_hash") or ""
                ),
                "warmup": 0,
                "successful_run_warmup": int(ns.warmup),
            }
            if trt_raw_completion_required:
                energy_workload.update({
                    "frozen_postprocess_contract":
                        trt_frozen_contract,
                    "frozen_postprocess_contract_sha256": str(
                        trt_frozen_contract.get(
                            "contract_sha256"
                        )
                        or ""
                    ),
                    "completed_task_completion_mode":
                        "frozen_host_tail",
                })
            else:
                energy_workload.update(direct_workload_binding)
        else:
            energy_workload = {
                "available": common_trt_available,
                "status": (
                    "available"
                    if common_trt_available
                    else "full_energy_hotloop_unavailable"
                ),
                "kind": "tensorrt_full_hotloop",
                "trtexec_artifact": "trtexec",
                "engine_artifact": "engine",
                "source_model_artifact": "source_onnx",
                "engine_build_receipt_artifact":
                    "engine_build_receipt",
                "engine_build_receipt_status":
                    trt_build_receipt_status,
                "engine_build_receipt_sha256": str(
                    producer_receipt.get("sha256") or ""
                ),
                "engine_build_receipt_file_sha256": str(
                    quality_first_producer.get(
                        "engine_build_receipt_file_sha256"
                    ) or ""
                ),
                "trt_engine_build_receipt_sha256": str(
                    producer_inner_receipt.get("receipt_sha256") or ""
                ),
                "quality_first_producer_identity_sha256":
                    quality_first_producer_sha,
                "quality_first_producer_artifact_match":
                    producer_artifact_match,
                "input_manifest_artifact": "input_manifest",
                "runtime_input_artifact": "runtime_input_tensor",
                "runtime_input_name": runtime_input_name,
                "runtime_input_shape":
                    input_manifest_payload.get("runtime_input_shape") or [],
                "runtime_input_dtype": str(
                    input_manifest_payload.get("runtime_input_dtype") or ""
                ),
                "runtime_input_layout": str(
                    (runtime_input_spec or {}).get("runtime_input_layout")
                    or ""
                ),
                "preprocess": dict(
                    (runtime_input_spec or {}).get("preprocess") or {}
                ),
                "task": str(row.get("task") or ""),
                "input_mode": "exact_semantic_dump_runtime_tensor",
                "invariant_args": invariant_args,
                "warmup_ms": 0,
                "successful_run_warmup_ms": successful_warmup_ms,
                "exact_iteration_evidence":
                    "trtexec_export_times_record_count",
            }
    elif backend_name == "native_full_deepx":
        prepared = row.get("deepx_prepared_feed_benchmark")
        prepared = dict(prepared) if isinstance(prepared, Mapping) else {}
        (
            deepx_source_artifacts,
            deepx_source_binding_status,
        ) = _verified_benchmark_set_source_onnx_artifacts(
            benchmark_set, model,
        )
        artifacts.update(deepx_source_artifacts)
        dxnn_path = Path(str(row.get("dxnn_path") or "")).expanduser()
        deepx_runner = (ROOT / "scripts" / "native_deepx_full_energy_hotloop.py").resolve()
        runtime_python = str(row.get("runtime_python") or "")
        runtime_python_bound = _attach_python_artifact(
            artifacts, "runtime_python", runtime_python,
        )
        if dxnn_path.is_file():
            artifacts["dxnn"] = {
                "path": str(dxnn_path.resolve()), "sha256": _sha256_file(dxnn_path),
            }
        if deepx_runner.is_file():
            artifacts["hotloop_runner"] = {
                "path": str(deepx_runner), "sha256": _sha256_file(deepx_runner),
            }
        prepared_input_contract = prepared.get("input_contract")
        deepx_raw_postprocess_required = bool(
            prepared.get("host_postprocess_frozen") is True
            or (
                str(prepared.get("task") or row.get("task") or "").strip().lower()
                == "detection"
                and str(row.get("contract_family") or "").strip().lower()
                in {"raw_head", "decoded_pre_nms"}
            )
        )
        deepx_direct_normalization_required = bool(
            str(
                prepared.get("task") or row.get("task") or ""
            ).strip().lower()
            == "detection"
            and str(row.get("contract_family") or "").strip().lower()
            == "decoded_nms"
            and (
                prepared.get("normalization_frozen") is True
                or row.get("normalization_frozen") is True
            )
        )
        deepx_postprocess_required = bool(
            deepx_raw_postprocess_required
            or deepx_direct_normalization_required
        )
        deepx_frozen_contract = _verified_frozen_contract(
            prepared.get("frozen_host_postprocess_contract")
            or row.get("frozen_host_postprocess_contract")
        ) if deepx_raw_postprocess_required else {}
        try:
            deepx_direct_contract = (
                verify_frozen_decoded_nms_normalization_contract(
                    prepared.get(
                        "frozen_decoded_nms_normalization_contract"
                    )
                    or row.get(
                        "frozen_decoded_nms_normalization_contract"
                    )
                )
                if deepx_direct_normalization_required else {}
            )
        except (FrozenPostprocessError, TypeError, ValueError):
            deepx_direct_contract = {}
        deepx_completed_frames = int(prepared.get("completed_frames") or 0)
        deepx_postprocess_completed = int(
            prepared.get("postprocess_completed_frames") or 0
        )
        deepx_original_wh = prepared.get("original_image_wh")
        deepx_original_wh = (
            [int(value) for value in deepx_original_wh]
            if isinstance(deepx_original_wh, list)
            and len(deepx_original_wh) == 2
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value > 0
                for value in deepx_original_wh
            )
            else []
        )
        deepx_frozen_implementation_bound = bool(
            (deepx_frozen_contract or deepx_direct_contract)
            and _attach_frozen_implementation_artifacts(
                deepx_frozen_contract or deepx_direct_contract
            )
        )
        direct_workload_binding = (
            _verified_direct_completed_task_workload(
                row,
                deepx_direct_contract,
                completed_frames=deepx_completed_frames,
                postprocess_completed_frames=
                    deepx_postprocess_completed,
            )
            if deepx_direct_normalization_required
            and deepx_direct_contract else {}
        )
        deepx_raw_postprocess_binding_ok = bool(
            deepx_raw_postprocess_required
            and not deepx_direct_normalization_required
            and deepx_frozen_contract
            and prepared.get("host_postprocess_frozen") is True
            and row.get("normalization_frozen") is not True
            and str(
                row.get(
                    "frozen_host_postprocess_contract_sha256"
                ) or ""
            )
            == str(
                deepx_frozen_contract.get("contract_sha256") or ""
            )
        )
        deepx_postprocess_binding_ok = bool(
            not deepx_postprocess_required
            or (
                deepx_raw_postprocess_required
                != deepx_direct_normalization_required
                and bool(
                    deepx_frozen_contract
                    if deepx_raw_postprocess_required
                    else deepx_direct_contract
                )
                and deepx_frozen_implementation_bound
                and prepared.get("postprocess_included") is True
                and (
                    (
                        deepx_raw_postprocess_binding_ok
                        and prepared.get("host_postprocess_frozen")
                        is True
                    )
                    or (
                        bool(direct_workload_binding)
                        and prepared.get("normalization_frozen") is True
                        and prepared.get("host_postprocess_frozen")
                        is not True
                    )
                )
                and deepx_completed_frames > 0
                and deepx_postprocess_completed == deepx_completed_frames
                and bool(deepx_original_wh)
                and deepx_original_wh == list(
                    (
                        deepx_frozen_contract
                        or deepx_direct_contract
                    ).get("original_wh") or []
                )
            )
        )
        prepared_image = str(prepared.get("image") or input_image or "").strip()
        prepared_image_sha = ""
        prepared_image_path = Path(prepared_image).expanduser()
        if prepared_image_path.is_file():
            prepared_image = str(prepared_image_path.resolve())
            prepared_image_sha = _sha256_file(prepared_image_path)
        prepared_tensor_path = Path(
            str(prepared.get("prepared_input_file") or "")
        ).expanduser()
        prepared_tensor_identity_ok = bool(
            runtime_input_bound
            and prepared.get("prepared_input_binding_verified") is True
            and str(prepared.get("prepared_input_source") or "")
            == "sealed_semantic_dump_runtime_tensor"
            and _confined_regular_file(
                prepared_tensor_path,
                allowed_root=input_artifact_root,
                expected_path=runtime_input_path,
            )
            and str(prepared.get("prepared_input_sha256") or "").strip().lower()
            == declared_runtime_input_sha
            and str(prepared.get("prepared_input_file_sha256") or "").strip().lower()
            == declared_runtime_input_sha
            and _sha256_file(prepared_tensor_path) == declared_runtime_input_sha
            and int(prepared.get("prepared_input_bytes") or 0)
            == int((runtime_input_spec or {}).get("runtime_input_bytes") or 0)
            and str(prepared.get("prepared_input_name") or "")
            == runtime_input_name
            and list(prepared.get("prepared_input_shape") or [])
            == list((runtime_input_spec or {}).get("runtime_input_shape") or [])
            and str(prepared.get("prepared_input_dtype") or "").strip().lower()
            == str((runtime_input_spec or {}).get("runtime_input_dtype") or "").strip().lower()
            and str(prepared.get("prepared_input_layout") or "").strip().upper()
            == str((runtime_input_spec or {}).get("runtime_input_layout") or "").strip().upper()
            and dict(prepared.get("runtime_preprocessing_identity") or {})
            == dict((runtime_input_spec or {}).get("runtime_preprocessing_identity") or {})
            and str(prepared.get("runtime_preprocessing_sha256") or "").strip().lower()
            == str((runtime_input_spec or {}).get("runtime_preprocessing_sha256") or "")
            and dict(prepared.get("runtime_numeric_input_identity") or {})
            == dict((runtime_input_spec or {}).get("runtime_numeric_input_identity") or {})
            and str(prepared.get("runtime_numeric_input_sha256") or "").strip().lower()
            == str((runtime_input_spec or {}).get("runtime_numeric_input_sha256") or "")
            and str(
                prepared.get("prepared_input_source_image_id") or ""
            ) == Path(str((runtime_input_spec or {}).get("input_image") or "")).name
            and str(
                prepared.get("prepared_input_source_image_sha256") or ""
            ).strip().lower()
            == str(
                (runtime_input_spec or {}).get("input_image_sha256") or ""
            ).strip().lower()
            == input_sha
        )
        deepx_sealed_result = (
            prepared.get("frozen_decoded_nms_normalization_result")
            if deepx_direct_normalization_required
            else prepared.get("frozen_host_postprocess_result")
        )
        if deepx_postprocess_required and isinstance(
            deepx_sealed_result, Mapping,
        ):
            (
                deepx_completed_artifact_persisted,
                deepx_completed_artifact_status,
            ) = _completed_result_artifact_persistence_status(
                prepared, sealed_result=deepx_sealed_result,
                allowed_root=(
                    benchmark_set / "results"
                    / str(row.get("run_id") or "")
                ),
                expected_path=(
                    benchmark_set / "results"
                    / str(row.get("run_id") or "")
                    / "deepx_prepared_feed.completed_task_result_artifact.json"
                ),
            )
        else:
            deepx_completed_artifact_persisted = not deepx_postprocess_required
            deepx_completed_artifact_status = (
                "not_applicable" if not deepx_postprocess_required
                else "completed_task_result_artifact_missing"
            )
        prepared_contract_ok = bool(
            str(row.get("performance_benchmark_source") or "") == "dx_engine_prepared_feed"
            and str(row.get("prepared_feed_contract_version") or "")
            == DEEPX_PREPARED_FEED_CONTRACT_VERSION
            and str(prepared.get("prepared_feed_contract_version") or "")
            == DEEPX_PREPARED_FEED_CONTRACT_VERSION
            and isinstance(prepared_input_contract, Mapping)
            and int(row.get("completed_work_units") or 0) == int(row.get("frames") or 0)
            and prepared_tensor_identity_ok
            and deepx_completed_artifact_persisted
        )
        prepared_image_identity_ok = bool(
            re.fullmatch(r"[0-9a-f]{64}", input_sha) is not None
            and prepared_image_sha == input_sha
            and str(
                (runtime_input_spec or {}).get("input_image_sha256") or ""
            ).strip().lower() == input_sha
            and str(
                prepared.get("prepared_input_source_image_sha256") or ""
            ).strip().lower() == input_sha
        )
        available = bool(
            runtime_python_bound and command_python_bound
            and deepx_runner.is_file() and dxnn_path.is_file()
            and bool(deepx_source_artifacts)
            and prepared_contract_ok
            and runtime_input_bound
            and deepx_postprocess_binding_ok
        )
        energy_workload = {
            "available": available,
            "status": (
                "available" if available
                else "full_energy_prepared_tensor_input_mismatch"
                if not prepared_tensor_identity_ok
                else "full_energy_completed_result_persistence_missing"
                if not deepx_completed_artifact_persisted
                else "full_energy_source_model_binding_missing_or_invalid"
                if not deepx_source_artifacts
                else "full_energy_hotloop_unavailable"
            ),
            "kind": "deepx_full_prepared_feed_hotloop",
            "runtime_python_artifact": "runtime_python",
            "runner_artifact": "hotloop_runner",
            "dxnn_artifact": "dxnn",
            "source_model_artifact": "source_onnx",
            "benchmark_set_manifest_artifact": "benchmark_set_manifest",
            "source_model_binding_status": deepx_source_binding_status,
            "input_image": prepared_image,
            "input_image_sha256": prepared_image_sha,
            "input_image_matches_full_contract": prepared_image_identity_ok,
            "prepared_input_source_image_id": str(
                prepared.get("prepared_input_source_image_id") or ""
            ),
            "prepared_input_source_image_sha256": str(
                prepared.get("prepared_input_source_image_sha256") or ""
            ),
            "input_manifest_artifact": "input_manifest",
            "runtime_input_artifact": "runtime_input_tensor",
            "runtime_input_name": runtime_input_name,
            "runtime_input_shape": list(
                (runtime_input_spec or {}).get("runtime_input_shape") or []
            ),
            "runtime_input_dtype": str(
                (runtime_input_spec or {}).get("runtime_input_dtype") or ""
            ),
            "runtime_input_layout": str(
                (runtime_input_spec or {}).get("runtime_input_layout") or ""
            ),
            "runtime_input_bytes": int(
                (runtime_input_spec or {}).get("runtime_input_bytes") or 0
            ),
            "runtime_input_sha256": declared_runtime_input_sha,
            "runtime_input_mode": "exact_semantic_dump_runtime_tensor",
            "input_mode": "exact_semantic_dump_runtime_tensor",
            "runtime_preprocessing_identity": dict(
                (runtime_input_spec or {}).get(
                    "runtime_preprocessing_identity"
                ) or {}
            ),
            "runtime_preprocessing_sha256": str(
                (runtime_input_spec or {}).get(
                    "runtime_preprocessing_sha256"
                ) or ""
            ),
            "runtime_numeric_input_identity": dict(
                (runtime_input_spec or {}).get(
                    "runtime_numeric_input_identity"
                ) or {}
            ),
            "runtime_numeric_input_sha256": str(
                (runtime_input_spec or {}).get(
                    "runtime_numeric_input_sha256"
                ) or ""
            ),
            "runtime_input_binding_verified": prepared_tensor_identity_ok,
            "performance_completed_result_artifact_status": (
                deepx_completed_artifact_status
            ),
            "input_contract": dict(prepared_input_contract) if isinstance(prepared_input_contract, Mapping) else {},
            "prepared_feed_contract_version": str(
                prepared.get("prepared_feed_contract_version") or ""
            ),
            "warmup": 0,
            "successful_run_warmup": int(prepared.get("warmup_count") or 0),
            "task": str(prepared.get("task") or row.get("task") or "auto"),
            "e2e_scope": (
                "full_task_pipeline" if deepx_postprocess_required
                else "accelerator_output_endpoint"
            ),
            "postprocess_required": deepx_postprocess_required,
            "postprocess_included": deepx_postprocess_required,
            "host_postprocess_frozen":
                deepx_raw_postprocess_required,
            "normalization_frozen":
                deepx_direct_normalization_required,
            "frozen_postprocess_implementation_bound": deepx_frozen_implementation_bound,
            "successful_run_postprocess_completed_frames": deepx_postprocess_completed,
            "successful_run_completed_frames": deepx_completed_frames,
            "postprocess_completed_frames": deepx_postprocess_completed,
            "postprocess_completion_verified":
                deepx_postprocess_binding_ok,
            "original_image_wh": deepx_original_wh,
            "completed_work_units_source": str(
                row.get("completed_work_units_source") or ""
            ),
            "source_endpoint_contract_hash": str(
                row.get("endpoint_contract_hash") or ""
            ),
        }
        if deepx_raw_postprocess_required:
            energy_workload.update({
                "frozen_postprocess_contract":
                    deepx_frozen_contract,
                "frozen_postprocess_contract_sha256": str(
                    deepx_frozen_contract.get("contract_sha256") or ""
                ),
                "completed_task_completion_mode":
                    "frozen_host_tail",
            })
        elif deepx_direct_normalization_required:
            energy_workload.update(direct_workload_binding)
    if energy_workload.get("task") == "classification":
        energy_workload.update({
            "completed_task_stage": "classification_top1_top5",
            "completed_task_contract_family": "classification_top1_top5",
            "classification_postprocess_required": True,
            "postprocess_included": True,
            "e2e_scope": "full_task_pipeline",
        })
    source_model_artifact = (
        artifacts.get("source_onnx")
        if isinstance(artifacts.get("source_onnx"), Mapping) else {}
    )
    source_model_sha256 = str(
        source_model_artifact.get("sha256") or ""
    ).strip().lower()
    compiled_artifact_name = {
        "native_full_tensorrt": "engine",
        "native_full_deepx": "dxnn",
        "native_full_hailo8": "hef",
        "native_full_hailo10h": "hef",
    }.get(backend_name, "")
    compiled_artifact = (
        artifacts.get(compiled_artifact_name)
        if isinstance(artifacts.get(compiled_artifact_name), Mapping) else {}
    )
    sealed_quality_producer = row.get("quality_first_producer_identity")
    sealed_quality_producer = (
        dict(sealed_quality_producer)
        if isinstance(sealed_quality_producer, Mapping) else {}
    )
    sealed_quality_producer_sha = str(
        row.get("quality_first_producer_identity_sha256") or ""
    ).strip().lower()
    sealed_receipt_binding = (
        sealed_quality_producer.get("engine_build_receipt")
        if isinstance(
            sealed_quality_producer.get("engine_build_receipt"), Mapping
        ) else {}
    )
    sealed_receipt = (
        sealed_receipt_binding.get("receipt")
        if isinstance(sealed_receipt_binding.get("receipt"), Mapping) else {}
    )
    energy_workload["runtime_preprocessing_binding"] = dict(
        runtime_preprocessing_binding
    )
    contract: dict[str, Any] = {
        "schema": FULL_COMMAND_CONTRACT_SCHEMA,
        "schema_version": FULL_COMMAND_CONTRACT_VERSION,
        "backend": str(row.get("backend") or ""),
        "backend_arg": str(backend_arg or ""),
        "model": str(model or ""),
        "case": "full",
        "setup_id": str(row.get("setup_id") or ""),
        "comparison_backend": str(row.get("comparison_backend") or ""),
        "comparison_precision": str(row.get("comparison_precision") or ""),
        "legacy_comparison_precision": str(row.get("legacy_comparison_precision") or ""),
        "execution_precision": str(row.get("execution_precision") or ""),
        "full_runtime_precision": str(row.get("full_runtime_precision") or ""),
        "python_executable": str(sys.executable),
        "runner": "scripts/native_full_baseline_eval_runner.py",
        "runner_sha256": _sha256_file(runner_path),
        "root": str(root),
        "benchmark_set": str(benchmark_set),
        "input_case": input_case,
        "input_image": input_image,
        "input_image_sha256": input_sha,
        "source_model_sha256": source_model_sha256,
        "quality_first_producer_identity": sealed_quality_producer,
        "quality_first_producer_identity_sha256": sealed_quality_producer_sha,
        "quality_request_binding": dict(
            row.get("quality_request_binding") or {}
        ) if isinstance(row.get("quality_request_binding"), Mapping) else {},
        "quality_request_binding_sha256": str(
            row.get("quality_request_binding_sha256") or ""
        ),
        "quality_request_binding_set_sha256": str(
            row.get("quality_request_binding_set_sha256") or ""
        ),
        "runtime_preprocessing_binding": runtime_preprocessing_binding,
        "engine_build_receipt_path": str(
            sealed_receipt_binding.get("path") or ""
        ),
        "engine_build_receipt_sha256": str(
            sealed_receipt_binding.get("sha256") or ""
        ),
        "engine_build_receipt_file_sha256": str(
            sealed_quality_producer.get(
                "engine_build_receipt_file_sha256"
            ) or ""
        ),
        "trt_engine_build_receipt_sha256": str(
            sealed_receipt.get("receipt_sha256") or ""
        ),
        "engine_build_receipt_size_bytes": int(
            sealed_receipt_binding.get("size_bytes") or 0
        ),
        "engine_build_receipt_file_size_bytes": int(
            (artifacts.get("engine_build_receipt") or {}).get(
                "file_size_bytes"
            ) or 0
        ) if isinstance(artifacts.get("engine_build_receipt"), Mapping) else 0,
        "trt_engine_build_receipt": trt_build_receipt,
        "trt_engine_build_receipt_status": trt_build_receipt_status,
        "hailo_hef_build_receipt_status": str(
            row.get("hailo_hef_build_receipt_status") or ""
        ),
        "hailo_hef_build_receipt_path": str(
            row.get("hailo_hef_build_receipt_path") or ""
        ),
        "hailo_hef_build_receipt_file_sha256": str(
            row.get("hailo_hef_build_receipt_file_sha256") or ""
        ),
        "hailo_hef_build_receipt_sha256": str(
            row.get("hailo_hef_build_receipt_sha256") or ""
        ),
        "hailo_hef_build_receipt": dict(
            row.get("hailo_hef_build_receipt") or {}
        ) if isinstance(
            row.get("hailo_hef_build_receipt"), Mapping,
        ) else {},
        "hailo_hef_preprocessing_contract": dict(
            row.get("hailo_hef_preprocessing_contract") or {}
        ) if isinstance(
            row.get("hailo_hef_preprocessing_contract"), Mapping,
        ) else {},
        "hailo_hef_preprocessing_contract_sha256": str(
            row.get("hailo_hef_preprocessing_contract_sha256") or ""
        ),
        "model_binding": {
            "source_artifact": "source_onnx" if source_model_sha256 else "",
            "source_onnx_sha256": source_model_sha256,
            "compiled_artifact": compiled_artifact_name,
            "compiled_artifact_sha256": str(compiled_artifact.get("sha256") or ""),
            "status": (
                "verified_engine_build_receipt_bound"
                if (
                    backend_name == "native_full_tensorrt"
                    and source_model_sha256 and compiled_artifact
                    and trt_build_receipt
                )
                else "source_and_compiled_artifact_hash_bound"
                if (
                    backend_name != "native_full_tensorrt"
                    and source_model_sha256 and compiled_artifact
                )
                else "unavailable"
            ),
        },
        "runtime_options": runtime_options,
        "energy_workload": energy_workload,
        "artifacts": artifacts,
        "complete": bool(
            row.get("ok")
            and str(row.get("backend") or "")
            and str(model or "")
            and str(root)
            and str(benchmark_set)
            and str(backend_arg or "")
            and command_python_bound
        ),
    }
    contract["complete"] = bool(
        contract["complete"]
        and energy_workload.get("available") is True
        and (
            not isinstance(row.get("quality_request_binding"), Mapping)
            or not row.get("quality_request_binding")
            or runtime_preprocessing_binding.get("status")
            == "runtime_preprocessing_and_numeric_identity_verified_exact"
        )
    )
    contract["contract_sha256"] = _canonical_json_sha256(contract)
    return contract


def _sealed_trt_source_model_binding(contract: Mapping[str, Any]) -> bool:
    """Verify the sealed Source-ONNX -> TensorRT-engine hash relationship."""
    workload = contract.get("energy_workload")
    artifacts = contract.get("artifacts")
    if not isinstance(workload, Mapping) or not isinstance(artifacts, Mapping):
        return False
    if str(workload.get("kind") or "") not in {
        "tensorrt_full_hotloop",
        "tensorrt_full_completed_task_hotloop",
    }:
        return True
    binding = contract.get("model_binding")
    if not isinstance(binding, Mapping):
        return False
    source_key = str(workload.get("source_model_artifact") or "")
    engine_key = str(workload.get("engine_artifact") or "")
    source = artifacts.get(source_key)
    engine = artifacts.get(engine_key)
    trtexec = artifacts.get(str(workload.get("trtexec_artifact") or ""))
    receipt_artifact = artifacts.get(
        str(workload.get("engine_build_receipt_artifact") or "")
    )
    raw_receipt = contract.get("trt_engine_build_receipt")
    quality_producer = contract.get("quality_first_producer_identity")
    if (
        not isinstance(source, Mapping) or not isinstance(engine, Mapping)
        or not isinstance(trtexec, Mapping)
        or not isinstance(receipt_artifact, Mapping)
        or not isinstance(raw_receipt, Mapping)
        or not isinstance(quality_producer, Mapping)
    ):
        return False
    quality_producer = dict(quality_producer)
    quality_producer_sha = str(
        quality_producer.get("producer_identity_sha256") or ""
    ).strip().lower()
    unhashed_quality_producer = dict(quality_producer)
    unhashed_quality_producer.pop("producer_identity_sha256", None)
    quality_producer_receipt = quality_producer.get("engine_build_receipt")
    quality_producer_receipt = (
        dict(quality_producer_receipt)
        if isinstance(quality_producer_receipt, Mapping) else {}
    )
    quality_receipt_payload = quality_producer_receipt.get("receipt")
    quality_receipt_payload = (
        dict(quality_receipt_payload)
        if isinstance(quality_receipt_payload, Mapping) else {}
    )
    quality_source = quality_producer.get("source_onnx")
    quality_source = dict(quality_source) if isinstance(quality_source, Mapping) else {}
    quality_build = quality_producer.get("build_onnx")
    quality_build = dict(quality_build) if isinstance(quality_build, Mapping) else {}
    quality_engine = quality_producer.get("engine")
    quality_engine = dict(quality_engine) if isinstance(quality_engine, Mapping) else {}
    quality_trtexec = quality_producer.get("trtexec")
    quality_trtexec = dict(quality_trtexec) if isinstance(quality_trtexec, Mapping) else {}
    source_sha = str(source.get("sha256") or "").strip().lower()
    engine_sha = str(engine.get("sha256") or "").strip().lower()
    trtexec_sha = str(trtexec.get("sha256") or "").strip().lower()
    receipt_full = dict(raw_receipt)
    outer_receipt_sha = _canonical_json_sha256(receipt_full)
    receipt = dict(receipt_full)
    receipt_sha = str(receipt.pop("receipt_sha256", "") or "").strip().lower()
    if (
        re.fullmatch(r"[0-9a-f]{64}", receipt_sha) is None
        or _canonical_json_sha256(receipt) != receipt_sha
        or receipt.get("schema") != TRT_ENGINE_BUILD_RECEIPT_SCHEMA
        or receipt.get("schema_version") != TRT_ENGINE_BUILD_RECEIPT_VERSION
        or receipt.get("build_returncode") != 0
        or receipt.get("dry_run") is not False
    ):
        return False
    command = receipt.get("command")
    command = [str(value) for value in command] if isinstance(command, list) else []
    source_path = str(source.get("path") or "")
    engine_path = str(engine.get("path") or "")
    trtexec_path = str(trtexec.get("path") or "")
    if (
        not command or command[0] != trtexec_path
        or [value for value in command[1:] if value.startswith("--onnx=")]
        != [f"--onnx={source_path}"]
        or [value for value in command[1:] if value.startswith("--saveEngine=")]
        != [f"--saveEngine={engine_path}"]
    ):
        return False
    return bool(
        re.fullmatch(r"[0-9a-f]{64}", source_sha)
        and re.fullmatch(r"[0-9a-f]{64}", engine_sha)
        and re.fullmatch(r"[0-9a-f]{64}", trtexec_sha)
        and str(contract.get("source_model_sha256") or "").strip().lower() == source_sha
        and str(engine.get("compiled_from_source_onnx_sha256") or "").strip().lower() == source_sha
        and str(receipt.get("source_onnx") or "") == source_path
        and str(receipt.get("source_onnx_sha256") or "").strip().lower() == source_sha
        and str(receipt.get("engine") or "") == engine_path
        and str(receipt.get("engine_sha256") or "").strip().lower() == engine_sha
        and str(receipt.get("trtexec") or "") == trtexec_path
        and str(receipt.get("trtexec_sha256") or "").strip().lower() == trtexec_sha
        and re.fullmatch(
            r"[0-9a-f]{64}", str(receipt_artifact.get("sha256") or "").strip().lower()
        ) is not None
        and str(contract.get("engine_build_receipt_sha256") or "").strip().lower()
        == outer_receipt_sha
        and str(
            contract.get("engine_build_receipt_file_sha256") or ""
        ).strip().lower()
        == str(receipt_artifact.get("sha256") or "").strip().lower()
        and str(
            receipt_artifact.get("canonical_sha256") or ""
        ).strip().lower() == outer_receipt_sha
        and str(
            contract.get("trt_engine_build_receipt_sha256") or ""
        ).strip().lower() == receipt_sha
        and str(receipt_artifact.get("receipt_sha256") or "").strip().lower()
        == receipt_sha
        and int(contract.get("engine_build_receipt_size_bytes") or 0)
        == int(receipt_artifact.get("canonical_size_bytes") or -1)
        and re.fullmatch(r"[0-9a-f]{64}", quality_producer_sha) is not None
        and _canonical_json_sha256(unhashed_quality_producer)
        == quality_producer_sha
        and str(
            contract.get("quality_first_producer_identity_sha256") or ""
        ).strip().lower() == quality_producer_sha
        and str(
            workload.get("quality_first_producer_identity_sha256") or ""
        ).strip().lower() == quality_producer_sha
        and quality_receipt_payload == receipt_full
        and str(quality_producer_receipt.get("sha256") or "").strip().lower()
        == outer_receipt_sha
        and str(
            quality_producer.get("engine_build_receipt_file_sha256") or ""
        ).strip().lower()
        == str(receipt_artifact.get("sha256") or "").strip().lower()
        and str(quality_source.get("sha256") or "") == source_sha
        and str(quality_build.get("path") or "") == source_path
        and str(quality_build.get("sha256") or "") == source_sha
        and int(quality_build.get("size_bytes") or 0)
        == int(source.get("size_bytes") or -1)
        and str(quality_engine.get("sha256") or "") == engine_sha
        and int(quality_engine.get("size_bytes") or 0)
        == int(engine.get("size_bytes") or -1)
        and str(quality_trtexec.get("sha256") or "") == trtexec_sha
        and int(quality_trtexec.get("size_bytes") or 0)
        == int(trtexec.get("size_bytes") or -1)
        and int(quality_producer_receipt.get("size_bytes") or 0)
        == int(receipt_artifact.get("size_bytes") or -1)
        and str(binding.get("source_artifact") or "") == source_key
        and str(binding.get("source_onnx_sha256") or "").strip().lower() == source_sha
        and str(binding.get("compiled_artifact") or "") == engine_key
        and str(binding.get("compiled_artifact_sha256") or "").strip().lower() == engine_sha
        and str(binding.get("status") or "") == "verified_engine_build_receipt_bound"
        and str(contract.get("trt_engine_build_receipt_status") or "")
        == "engine_build_receipt_verified"
        and str(workload.get("engine_build_receipt_status") or "")
        == "engine_build_receipt_verified"
    )


def _sealed_deepx_source_model_binding(
    contract: Mapping[str, Any],
) -> bool:
    """Verify the sealed BenchmarkSet Source-ONNX -> DXNN relationship."""
    workload = contract.get("energy_workload")
    artifacts = contract.get("artifacts")
    if not isinstance(workload, Mapping) or not isinstance(artifacts, Mapping):
        return False
    if str(workload.get("kind") or "") != "deepx_full_prepared_feed_hotloop":
        return True
    model = str(contract.get("model") or "").strip()
    benchmark_set = _canonical_lexical_absolute_path(
        contract.get("benchmark_set")
    )
    source_key = str(workload.get("source_model_artifact") or "")
    manifest_key = str(
        workload.get("benchmark_set_manifest_artifact") or ""
    )
    dxnn_key = str(workload.get("dxnn_artifact") or "")
    source = artifacts.get(source_key)
    manifest = artifacts.get(manifest_key)
    dxnn = artifacts.get(dxnn_key)
    binding = contract.get("model_binding")
    if (
        not model
        or model != _safe_component(model)
        or Path(model).name != model
        or benchmark_set is None
        or source_key != "source_onnx"
        or manifest_key != "benchmark_set_manifest"
        or dxnn_key != "dxnn"
        or not isinstance(source, Mapping)
        or not isinstance(manifest, Mapping)
        or not isinstance(dxnn, Mapping)
        or not isinstance(binding, Mapping)
    ):
        return False
    source_sha = _strict_sha256_token(source.get("sha256"))
    manifest_sha = _strict_sha256_token(manifest.get("sha256"))
    dxnn_sha = _strict_sha256_token(dxnn.get("sha256"))
    source_path = _canonical_lexical_absolute_path(source.get("path"))
    manifest_path = _canonical_lexical_absolute_path(manifest.get("path"))
    expected_relative = f"models/{model}.onnx"
    return bool(
        source_sha and manifest_sha and dxnn_sha
        and source_path == benchmark_set / "models" / f"{model}.onnx"
        and manifest_path == benchmark_set / "benchmark_set.json"
        and _strict_positive_int(source.get("size_bytes")) is not None
        and _strict_positive_int(manifest.get("size_bytes")) is not None
        and str(source.get("role") or "")
        == "sealed_benchmark_set_source_onnx"
        and str(source.get("relative_path") or "") == expected_relative
        and _strict_sha256_token(
            source.get("benchmark_set_manifest_sha256")
        ) == manifest_sha
        and str(manifest.get("role") or "")
        == "sealed_benchmark_set_manifest"
        and str(workload.get("source_model_binding_status") or "")
        == "benchmark_set_source_onnx_verified_exact"
        and _strict_sha256_token(contract.get("source_model_sha256"))
        == source_sha
        and str(binding.get("source_artifact") or "") == source_key
        and _strict_sha256_token(binding.get("source_onnx_sha256"))
        == source_sha
        and str(binding.get("compiled_artifact") or "") == dxnn_key
        and _strict_sha256_token(
            binding.get("compiled_artifact_sha256")
        ) == dxnn_sha
        and str(binding.get("status") or "")
        == "source_and_compiled_artifact_hash_bound"
    )


def _sealed_frozen_postprocess_binding(contract: Mapping[str, Any]) -> bool:
    workload = contract.get("energy_workload")
    artifacts = contract.get("artifacts")
    if not isinstance(workload, Mapping) or not isinstance(artifacts, Mapping):
        return False
    if workload.get("postprocess_required") is not True:
        return True
    if workload.get("task") == "classification":
        completed = _strict_positive_int(workload.get("successful_run_completed_frames"))
        return bool(
            workload.get("kind") == "tensorrt_full_completed_task_hotloop"
            and workload.get("completed_task_stage") == "classification_top1_top5"
            and workload.get("postprocess_included") is True
            and workload.get("postprocess_completion_verified") is True
            and completed is not None
            and completed == _strict_positive_int(workload.get("successful_run_postprocess_completed_frames"))
            and not workload.get("host_postprocess_frozen")
            and not workload.get("normalization_frozen")
        )
    raw_mode = isinstance(
        workload.get("frozen_postprocess_contract"), Mapping,
    )
    direct_mode = isinstance(
        workload.get(
            "frozen_decoded_nms_normalization_contract"
        ),
        Mapping,
    )
    if raw_mode == direct_mode:
        return False
    if direct_mode:
        try:
            frozen = (
                verify_frozen_decoded_nms_normalization_contract(
                    workload.get(
                        "frozen_decoded_nms_normalization_contract"
                    )
                )
            )
            completed_frames = int(
                workload.get("successful_run_completed_frames") or 0
            )
            postprocess_frames = int(
                workload.get(
                    "successful_run_postprocess_completed_frames"
                )
                or 0
            )
            expected_completion = (
                build_normalized_detection_endpoint_attestation(
                    frozen,
                    workload.get(
                        "frozen_decoded_nms_normalization_result"
                    ),
                    completed_frames=completed_frames,
                    postprocess_completed_frames=postprocess_frames,
                )
            )
        except (
            FrozenPostprocessError, TypeError, ValueError,
        ):
            return False
        if (
            workload.get("normalization_frozen") is not True
            or workload.get("host_postprocess_frozen") is True
            or workload.get("postprocess_included") is not True
            or workload.get("postprocess_completion_verified")
            is not True
            or int(workload.get("postprocess_completed_frames") or 0)
            != postprocess_frames
            or completed_frames <= 0
            or postprocess_frames != completed_frames
            or str(
                workload.get("completed_task_completion_mode") or ""
            )
            != (
                "integrated_accelerator_plus_"
                "frozen_normalization"
            )
            or str(
                workload.get(
                    "frozen_decoded_nms_normalization_contract_sha256"
                )
                or ""
            ).strip().lower()
            != str(frozen.get("contract_sha256") or "")
            or str(
                workload.get("source_endpoint_contract_hash") or ""
            ).strip().lower()
            != str(
                frozen.get("source_endpoint_contract_hash") or ""
            )
            or str(workload.get("source_output_endpoint_id") or "")
            != str(frozen.get("source_output_endpoint_id") or "")
            or dict(
                workload.get("source_output_tensor_signature") or {}
            )
            != dict(
                frozen.get("source_output_tensor_signature") or {}
            )
            or str(
                workload.get(
                    "source_output_endpoint_attestation_sha256"
                )
                or ""
            )
            != str(
                frozen.get(
                    "source_output_endpoint_attestation_sha256"
                )
                or ""
            )
            or str(
                workload.get(
                    "letterbox_geometry_contract_sha256"
                )
                or ""
            )
            != str(
                frozen.get(
                    "letterbox_geometry_contract_sha256"
                )
                or ""
            )
            or dict(
                workload.get(
                    "completed_task_endpoint_attestation"
                )
                or {}
            )
            != dict(expected_completion)
        ):
            return False
        implementation = frozen.get("implementation_artifacts")
        if not isinstance(implementation, Mapping) or not implementation:
            return False
        return all(
            isinstance(raw, Mapping)
            and isinstance(
                artifacts.get(f"frozen_postprocess_{name}"),
                Mapping,
            )
            and str(
                (
                    artifacts.get(f"frozen_postprocess_{name}")
                    or {}
                ).get("sha256")
                or ""
            )
            == str(raw.get("sha256") or "")
            for name, raw in implementation.items()
        )
    try:
        frozen = verify_frozen_postprocess_contract(
            workload.get("frozen_postprocess_contract")
        )
    except (FrozenPostprocessError, TypeError, ValueError):
        return False
    declared_sha = str(
        workload.get("frozen_postprocess_contract_sha256") or ""
    ).strip().lower()
    original_wh = workload.get("original_image_wh")
    if (
        declared_sha != str(frozen.get("contract_sha256") or "")
        or workload.get("postprocess_included") is not True
        or workload.get("host_postprocess_frozen") is not True
        or workload.get("frozen_postprocess_implementation_bound") is not True
        or int(workload.get("successful_run_completed_frames") or 0) <= 0
        or int(workload.get("successful_run_postprocess_completed_frames") or 0)
        != int(workload.get("successful_run_completed_frames") or 0)
        or not isinstance(original_wh, list) or len(original_wh) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in original_wh
        )
        or list(original_wh) != list(frozen.get("original_wh") or [])
    ):
        return False
    implementation = frozen.get("implementation_artifacts")
    if not isinstance(implementation, Mapping) or not implementation:
        return False
    for name, raw in implementation.items():
        artifact = artifacts.get(f"frozen_postprocess_{name}")
        if (
            not isinstance(raw, Mapping)
            or not isinstance(artifact, Mapping)
            or str(artifact.get("sha256") or "") != str(raw.get("sha256") or "")
        ):
            return False
    return True


def _deepx_energy_artifact_role_paths_valid(
    contract: Mapping[str, Any], workload: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *, expected_root: Any,
) -> bool:
    """Reject hash-equal artifact rebinding to non-authoritative paths."""
    root = _absolute_without_resolving(
        Path(str(contract.get("root") or ""))
    )
    benchmark_set = _absolute_without_resolving(
        Path(str(contract.get("benchmark_set") or ""))
    )
    authoritative_root = _canonical_lexical_absolute_path(expected_root)
    declared_root = _canonical_lexical_absolute_path(contract.get("root"))
    declared_benchmark_set = _canonical_lexical_absolute_path(
        contract.get("benchmark_set")
    )
    model = str(contract.get("model") or "").strip()
    expected_benchmark_set = (
        authoritative_root / model
        / "benchmark_set"
        if authoritative_root is not None else None
    )
    if (
        authoritative_root is None
        or not model
        or model != _safe_component(model)
        or Path(model).name != model
        or declared_root != authoritative_root
        or declared_benchmark_set != expected_benchmark_set
        or _path_contains_symlink(root)
        or _path_contains_symlink(benchmark_set)
        or not root.is_dir()
        or not benchmark_set.is_dir()
    ):
        return False
    try:
        benchmark_set.resolve(strict=True).relative_to(
            root.resolve(strict=True)
        )
    except (OSError, RuntimeError, ValueError):
        return False
    semantic_root = (
        benchmark_set / "native_full_outputs"
        / f"model={_safe_component(contract.get('model'))}"
        / "backend=native_full_deepx"
        / f"setup={_safe_component(contract.get('setup_id') or 'unspecified')}"
        / f"comparison={_safe_component(contract.get('comparison_backend') or 'unspecified')}"
    )
    runtime_artifact = artifacts.get(
        str(workload.get("runtime_input_artifact") or "")
    )
    manifest_artifact = artifacts.get(
        str(workload.get("input_manifest_artifact") or "")
    )
    runner_artifact = artifacts.get(
        str(workload.get("runner_artifact") or "")
    )
    dxnn_artifact = artifacts.get(
        str(workload.get("dxnn_artifact") or "")
    )
    source_artifact = artifacts.get(
        str(workload.get("source_model_artifact") or "")
    )
    benchmark_manifest_artifact = artifacts.get(
        str(workload.get("benchmark_set_manifest_artifact") or "")
    )
    if not all(isinstance(value, Mapping) for value in (
        runtime_artifact, manifest_artifact, runner_artifact, dxnn_artifact,
        source_artifact, benchmark_manifest_artifact,
    )):
        return False
    if not _confined_regular_file(
        Path(str(runtime_artifact.get("path") or "")),
        allowed_root=semantic_root,
        expected_path=semantic_root / "runtime_input.bin",
    ):
        return False
    if not _confined_regular_file(
        Path(str(manifest_artifact.get("path") or "")),
        allowed_root=semantic_root,
        expected_path=semantic_root / "native_full_input_manifest.json",
    ):
        return False
    if not _confined_regular_file(
        Path(str(source_artifact.get("path") or "")),
        allowed_root=benchmark_set / "models",
        expected_path=benchmark_set / "models" / f"{model}.onnx",
    ):
        return False
    if not _confined_regular_file(
        Path(str(benchmark_manifest_artifact.get("path") or "")),
        allowed_root=benchmark_set,
        expected_path=benchmark_set / "benchmark_set.json",
    ):
        return False
    expected_runner = Path(__file__).resolve().parent / (
        "native_deepx_full_energy_hotloop.py"
    )
    if not _confined_regular_file(
        Path(str(runner_artifact.get("path") or "")),
        allowed_root=expected_runner.parent,
        expected_path=expected_runner,
    ):
        return False
    dxnn_path = Path(str(dxnn_artifact.get("path") or ""))
    if not _confined_regular_file(
        dxnn_path, allowed_root=benchmark_set,
    ):
        return False
    try:
        dxnn_relative = dxnn_path.resolve(strict=True).relative_to(
            benchmark_set.resolve(strict=True)
        )
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(
        dxnn_relative.parts
        and dxnn_relative.parts[0] == "deepx"
        and "full" in dxnn_relative.parts[:-1]
        and dxnn_relative.suffix.lower() == ".dxnn"
    )


def _verified_full_energy_contract(
    raw: Any, *, expected_root: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, Mapping):
        return None, "full_command_contract_missing"
    contract = dict(raw)
    declared = str(contract.pop("contract_sha256", "") or "").strip().lower()
    if len(declared) != 64 or _canonical_json_sha256(contract) != declared:
        return None, "full_command_contract_sha256_mismatch"
    contract["contract_sha256"] = declared
    if (
        contract.get("schema") != FULL_COMMAND_CONTRACT_SCHEMA
        or int(contract.get("schema_version") or 0) != FULL_COMMAND_CONTRACT_VERSION
        or contract.get("complete") is not True
    ):
        return None, "full_energy_hotloop_unavailable"
    if str(contract.get("runner_sha256") or "") != _sha256_file(Path(__file__).resolve()):
        return None, "full_runner_sha256_mismatch"
    workload = contract.get("energy_workload")
    if not isinstance(workload, Mapping) or workload.get("available") is not True:
        return None, "full_energy_hotloop_unavailable"
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None, "full_energy_artifacts_missing"
    if not _sealed_trt_source_model_binding(contract):
        return None, "full_energy_tensorrt_source_model_binding_invalid"
    if not _sealed_deepx_source_model_binding(contract):
        return None, "full_energy_deepx_source_model_binding_invalid"
    if not _sealed_frozen_postprocess_binding(contract):
        return None, "full_energy_frozen_postprocess_binding_invalid"
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            return None, f"full_energy_artifact_{name}_invalid"
        path = Path(str(artifact.get("path") or "")).expanduser()
        expected = str(artifact.get("sha256") or "").strip().lower()
        if not path.is_file() or len(expected) != 64 or _sha256_file(path) != expected:
            return None, f"full_energy_artifact_{name}_sha256_mismatch"
        if "python" in str(name):
            invocation = str(artifact.get("invocation_path") or "").strip()
            resolved = _resolved_executable(invocation)
            identity = artifact.get("interpreter_identity")
            if (
                resolved is None
                or str(resolved) != str(artifact.get("resolved_path") or "")
                or str(path.resolve()) != str(resolved)
                or not isinstance(identity, Mapping)
                or not all(str(identity.get(key) or "").strip() for key in (
                    "executable", "version", "implementation",
                ))
            ):
                return None, f"full_energy_artifact_{name}_interpreter_identity_mismatch"
    command_python = artifacts.get("command_python_executable")
    if (
        not isinstance(command_python, Mapping)
        or str(command_python.get("invocation_path") or "")
        != str(contract.get("python_executable") or "")
    ):
        return None, "full_energy_command_interpreter_binding_missing"
    kind = str(workload.get("kind") or "")
    if kind == "deepx_full_prepared_feed_hotloop":
        live_source_artifacts, live_source_status = (
            _verified_benchmark_set_source_onnx_artifacts(
                Path(str(contract.get("benchmark_set") or "")),
                str(contract.get("model") or ""),
            )
        )
        if (
            live_source_status
            != "benchmark_set_source_onnx_verified_exact"
            or any(
                dict(artifacts.get(name) or {}) != dict(expected)
                for name, expected in live_source_artifacts.items()
            )
            or set(live_source_artifacts)
            != {"benchmark_set_manifest", "source_onnx"}
        ):
            return None, "full_energy_deepx_source_model_metadata_invalid"
    if kind == "tensorrt_full_completed_task_hotloop" and (
        str(workload.get("e2e_scope") or "") != "full_task_pipeline"
        or str(workload.get("completed_task_stage") or "") != (
            "classification_top1_top5" if workload.get("task") == "classification" else "decoded_nms"
        )
        or int(workload.get("measurement_concurrency") or 0) != 1
    ):
        return None, "full_energy_completed_task_endpoint_contract_invalid"
    if kind in {
        "tensorrt_full_hotloop",
        "tensorrt_full_completed_task_hotloop",
    }:
        receipt_key = str(workload.get("engine_build_receipt_artifact") or "")
        receipt_artifact = artifacts.get(receipt_key)
        receipt_path = Path(str((receipt_artifact or {}).get("path") or "")).expanduser()
        persisted_receipt = _load_json(receipt_path) if receipt_path.is_file() else None
        if (
            not isinstance(persisted_receipt, Mapping)
            or dict(persisted_receipt) != dict(contract.get("trt_engine_build_receipt") or {})
        ):
            return None, "full_energy_tensorrt_build_receipt_content_mismatch"
    if kind in {
        "hailo_full_hotloop",
        "deepx_full_prepared_feed_hotloop",
        "tensorrt_full_completed_task_hotloop",
    }:
        runtime_key = str(workload.get("runtime_python_artifact") or "")
        if not runtime_key or not isinstance(artifacts.get(runtime_key), Mapping):
            return None, "full_energy_runtime_interpreter_binding_missing"
        if kind != "deepx_full_prepared_feed_hotloop" and (
            str(workload.get("input_image_sha256") or "").strip().lower()
            != str(contract.get("input_image_sha256") or "").strip().lower()
        ):
            return None, "full_energy_workload_input_identity_mismatch"
    if kind == "deepx_full_prepared_feed_hotloop":
        runtime_key = str(workload.get("runtime_input_artifact") or "")
        runtime_artifact = artifacts.get(runtime_key)
        semantic_identity = workload.get("runtime_preprocessing_identity")
        numeric_identity = workload.get("runtime_numeric_input_identity")
        semantic_sha = str(
            workload.get("runtime_preprocessing_sha256") or ""
        ).strip().lower()
        numeric_sha = str(
            workload.get("runtime_numeric_input_sha256") or ""
        ).strip().lower()
        numeric_errors = (
            runtime_numeric_input_identity_errors(
                numeric_identity, semantic_identity,
            )
            if isinstance(semantic_identity, Mapping)
            and isinstance(numeric_identity, Mapping)
            else ["runtime_numeric_input_identity_invalid"]
        )
        if not _deepx_energy_artifact_role_paths_valid(
            contract, workload, artifacts, expected_root=expected_root,
        ):
            return None, "full_energy_deepx_artifact_role_path_mismatch"
        if (
            workload.get("runtime_input_mode")
            != "exact_semantic_dump_runtime_tensor"
            or workload.get("runtime_input_binding_verified") is not True
            or not isinstance(runtime_artifact, Mapping)
            or str(runtime_artifact.get("sha256") or "").strip().lower()
            != str(workload.get("runtime_input_sha256") or "").strip().lower()
            or int(runtime_artifact.get("bytes") or 0)
            != int(workload.get("runtime_input_bytes") or 0)
            or not str(workload.get("runtime_input_name") or "").strip()
            or not isinstance(workload.get("runtime_input_shape"), list)
            or not workload.get("runtime_input_shape")
            or not str(workload.get("runtime_input_dtype") or "").strip()
            or not str(workload.get("runtime_input_layout") or "").strip()
            or not isinstance(semantic_identity, Mapping)
            or not isinstance(numeric_identity, Mapping)
            or semantic_identity.get("schema")
            != "onnx-splitpoint/image-preprocessing-contract"
            or int(semantic_identity.get("schema_version") or 0) != 2
            or numeric_identity.get("schema")
            != RUNTIME_NUMERIC_INPUT_SCHEMA
            or int(numeric_identity.get("schema_version") or 0)
            != RUNTIME_NUMERIC_INPUT_SCHEMA_VERSION
            or re.fullmatch(r"[0-9a-f]{64}", semantic_sha) is None
            or re.fullmatch(r"[0-9a-f]{64}", numeric_sha) is None
            or _canonical_json_sha256(semantic_identity) != semantic_sha
            or preprocessing_contract_sha256(semantic_identity)
            != semantic_sha
            or _canonical_json_sha256(numeric_identity) != numeric_sha
            or bool(numeric_errors)
            or str(
                numeric_identity.get("preprocessing_contract_sha256") or ""
            ).strip().lower() != semantic_sha
            or str(numeric_identity.get("backend") or "").strip().lower()
            != "native_full_deepx"
            or str(numeric_identity.get("runtime_input_name") or "")
            != str(workload.get("runtime_input_name") or "")
            or list(numeric_identity.get("runtime_input_shape") or [])
            != list(workload.get("runtime_input_shape") or [])
            or str(
                numeric_identity.get("runtime_input_dtype") or ""
            ).strip().lower()
            != str(workload.get("runtime_input_dtype") or "").strip().lower()
            or str(
                numeric_identity.get("runtime_input_layout") or ""
            ).strip().upper()
            != str(workload.get("runtime_input_layout") or "").strip().upper()
        ):
            return None, "full_energy_deepx_runtime_input_binding_invalid"
    input_image = str(workload.get("input_image") or "").strip()
    input_sha = str(workload.get("input_image_sha256") or "").strip().lower()
    if input_image and kind != "deepx_full_prepared_feed_hotloop":
        image = Path(input_image).expanduser()
        if not image.is_file() or len(input_sha) != 64 or _sha256_file(image) != input_sha:
            return None, "full_energy_input_image_sha256_mismatch"
    return contract, "hash_artifacts_input_and_hotloop_verified"


def _sealed_full_energy_contract(
    raw: Any, *, expected_root: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    """Verify only the small sealed contract object, never its artifact files."""
    if not isinstance(raw, Mapping):
        return None, "full_command_contract_missing"
    contract = dict(raw)
    declared = str(contract.pop("contract_sha256", "") or "").strip().lower()
    if len(declared) != 64 or _canonical_json_sha256(contract) != declared:
        return None, "full_command_contract_sha256_mismatch"
    contract["contract_sha256"] = declared
    workload = contract.get("energy_workload")
    if (
        contract.get("schema") != FULL_COMMAND_CONTRACT_SCHEMA
        or int(contract.get("schema_version") or 0) != FULL_COMMAND_CONTRACT_VERSION
        or contract.get("complete") is not True
        or not isinstance(workload, Mapping)
        or workload.get("available") is not True
    ):
        return None, "full_energy_hotloop_unavailable"
    if not _sealed_trt_source_model_binding(contract):
        return None, "full_energy_tensorrt_source_model_binding_invalid"
    if not _sealed_deepx_source_model_binding(contract):
        return None, "full_energy_deepx_source_model_binding_invalid"
    if not _sealed_frozen_postprocess_binding(contract):
        return None, "full_energy_frozen_postprocess_binding_invalid"
    kind = str(workload.get("kind") or "")
    if kind == "tensorrt_full_completed_task_hotloop" and (
        str(workload.get("e2e_scope") or "") != "full_task_pipeline"
        or str(workload.get("completed_task_stage") or "") != (
            "classification_top1_top5" if workload.get("task") == "classification" else "decoded_nms"
        )
        or int(workload.get("measurement_concurrency") or 0) != 1
    ):
        return None, "full_energy_completed_task_endpoint_contract_invalid"
    if kind in {
        "hailo_full_hotloop",
        "tensorrt_full_completed_task_hotloop",
    } and (
        str(workload.get("input_image_sha256") or "").strip().lower()
        != str(contract.get("input_image_sha256") or "").strip().lower()
    ):
        return None, "full_energy_workload_input_identity_mismatch"
    if kind == "deepx_full_prepared_feed_hotloop" and (
        _canonical_lexical_absolute_path(expected_root) is None
        or not str(contract.get("model") or "").strip()
        or str(contract.get("model") or "").strip()
        != _safe_component(contract.get("model"))
        or Path(str(contract.get("model") or "")).name
        != str(contract.get("model") or "")
        or _canonical_lexical_absolute_path(contract.get("root"))
        != _canonical_lexical_absolute_path(expected_root)
        or _canonical_lexical_absolute_path(contract.get("benchmark_set"))
        != (
            _canonical_lexical_absolute_path(expected_root)
            / str(contract.get("model") or "") / "benchmark_set"
        )
        or workload.get("runtime_input_mode")
        != "exact_semantic_dump_runtime_tensor"
        or workload.get("runtime_input_binding_verified") is not True
        or not str(workload.get("runtime_input_artifact") or "").strip()
        or str(workload.get("runtime_input_sha256") or "").strip().lower()
        != str(
            (
                contract.get("artifacts", {}).get(
                    str(workload.get("runtime_input_artifact") or ""), {}
                )
                if isinstance(contract.get("artifacts"), Mapping) else {}
            ).get("sha256") or ""
        ).strip().lower()
    ):
        return None, "full_energy_deepx_runtime_input_binding_invalid"
    return contract, "sealed_contract_verified"


def _energy_preflight_only(ns: argparse.Namespace) -> int:
    """Perform all heavyweight artifact checks before collector sampling starts."""
    try:
        if str(getattr(ns, "energy_command_contract_file", "") or "").strip():
            raw = _load_strict_json(
                Path(getattr(ns, "energy_command_contract_file")).expanduser()
            )
        else:
            raw = json.loads(str(ns.energy_command_contract_json or ""))
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "full_command_contract_json_invalid", "error": str(exc)}), file=sys.stderr)
        return 5
    contract, status = _verified_full_energy_contract(
        raw, expected_root=ns.root,
    )
    if contract is None:
        print(json.dumps({"ok": False, "status": status}), file=sys.stderr, flush=True)
        return 5
    nonce = str(ns.preflight_nonce or "").strip()
    output_text = str(ns.preflight_attestation_out or "").strip()
    if not nonce or not output_text:
        print(json.dumps({"ok": False, "status": "preflight_nonce_or_output_missing"}), file=sys.stderr)
        return 5
    now_ns = time.time_ns()
    max_age_s = max(1.0, float(ns.preflight_attestation_max_age_s or 300.0))
    artifacts = dict(contract.get("artifacts") or {})
    artifact_hashes = {
        str(name): str(value.get("sha256") or "").strip().lower()
        for name, value in artifacts.items() if isinstance(value, Mapping)
    }
    attestation: dict[str, Any] = {
        "schema": ENERGY_PREFLIGHT_ATTESTATION_SCHEMA,
        "schema_version": ENERGY_PREFLIGHT_ATTESTATION_VERSION,
        "ok": True,
        "nonce": nonce,
        "created_at_unix_ns": now_ns,
        "expires_at_unix_ns": now_ns + int(max_age_s * 1_000_000_000),
        "artifact_verification_status": "pass",
        "command_contract_sha256": str(contract["contract_sha256"]),
        "runner_sha256": str(contract.get("runner_sha256") or ""),
        "workload_kind": str((contract.get("energy_workload") or {}).get("kind") or ""),
        "verified_artifact_sha256": artifact_hashes,
        "verified_input_image_sha256": str(
            (contract.get("energy_workload") or {}).get("input_image_sha256") or ""
        ),
        "host": socket.gethostname(),
        "preflight_scope": "native_full_energy_hotloop_all_heavy_hashes",
    }
    attestation["attestation_sha256"] = _canonical_json_sha256(attestation)
    output = Path(output_text).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(attestation, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, output)
    print(
        "__SPLITPOINT_PREFLIGHT_ATTESTATION__="
        + json.dumps(attestation, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


def _verified_energy_preflight_attestation(
    raw_contract: Any, *, path_value: str, expected_nonce: str,
    max_age_s: float, expected_root: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    """Verify a small, fresh preflight seal without touching large artifacts."""
    contract, status = _sealed_full_energy_contract(
        raw_contract, expected_root=expected_root,
    )
    if contract is None:
        return None, status
    path = Path(str(path_value or "")).expanduser()
    try:
        if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size > 1024 * 1024:
            return None, "energy_preflight_attestation_missing_or_oversize"
        attestation = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "energy_preflight_attestation_invalid_json"
    if not isinstance(attestation, Mapping):
        return None, "energy_preflight_attestation_invalid"
    sealed = dict(attestation)
    declared = str(sealed.pop("attestation_sha256", "") or "").strip().lower()
    if len(declared) != 64 or _canonical_json_sha256(sealed) != declared:
        return None, "energy_preflight_attestation_sha256_mismatch"
    now_ns = time.time_ns()
    created_ns = int(attestation.get("created_at_unix_ns") or 0)
    expires_ns = int(attestation.get("expires_at_unix_ns") or 0)
    max_age_ns = int(max(1.0, float(max_age_s or 300.0)) * 1_000_000_000)
    expected_artifacts = {
        str(name): str(value.get("sha256") or "").strip().lower()
        for name, value in dict(contract.get("artifacts") or {}).items()
        if isinstance(value, Mapping)
    }
    expected_input_sha = str(
        (contract.get("energy_workload") or {}).get("input_image_sha256") or ""
    )
    if (
        attestation.get("schema") != ENERGY_PREFLIGHT_ATTESTATION_SCHEMA
        or int(attestation.get("schema_version") or 0) != ENERGY_PREFLIGHT_ATTESTATION_VERSION
        or attestation.get("ok") is not True
        or str(attestation.get("nonce") or "") != str(expected_nonce or "")
        or str(attestation.get("artifact_verification_status") or "") != "pass"
        or str(attestation.get("command_contract_sha256") or "") != str(contract["contract_sha256"])
        or str(attestation.get("runner_sha256") or "") != str(contract.get("runner_sha256") or "")
        or dict(attestation.get("verified_artifact_sha256") or {}) != expected_artifacts
        or str(attestation.get("verified_input_image_sha256") or "") != expected_input_sha
        or str(attestation.get("host") or "") != socket.gethostname()
        or created_ns <= 0 or expires_ns <= created_ns
        or created_ns > now_ns + 5_000_000_000
        or now_ns > expires_ns
        or now_ns - created_ns > max_age_ns
    ):
        return None, "energy_preflight_attestation_identity_or_freshness_mismatch"
    return contract, "fresh_preflight_attestation_verified"


def _trtexec_exported_iteration_count(path: Path) -> int:
    """Return the exact number of timing records exported by trtexec."""
    return int(_trtexec_exported_iteration_evidence(path).get("completed_work_units") or 0)


def _trtexec_exported_iteration_evidence(path: Path) -> dict[str, Any]:
    """Return exact count and measured trace span from ``--exportTimes``."""
    payload = _load_json(path) if path.is_file() else None
    candidates: list[Any] = []
    if isinstance(payload, list):
        candidates.append(payload)
    elif isinstance(payload, Mapping):
        for key in ("times", "iterations", "trace", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.append(value)
    for records in candidates:
        if records and all(isinstance(record, Mapping) for record in records):
            starts: list[float] = []
            ends: list[float] = []
            for record in records:
                for key in ("startH2dMs", "startEnqMs", "startComputeMs"):
                    try:
                        starts.append(float(record[key]))
                        break
                    except (KeyError, TypeError, ValueError):
                        continue
                for key in ("endD2hMs", "endComputeMs", "endEnqMs"):
                    try:
                        ends.append(float(record[key]))
                        break
                    except (KeyError, TypeError, ValueError):
                        continue
            span_s = (
                max(0.0, (max(ends) - min(starts)) / 1000.0)
                if starts and ends else None
            )
            return {
                "status": "ok", "completed_work_units": len(records),
                "measured_trace_duration_s": span_s,
            }
    return {
        "status": "missing_or_unsupported",
        "completed_work_units": 0,
        "measured_trace_duration_s": None,
    }


def _energy_workload_only(ns: argparse.Namespace) -> int:
    """Execute only a preverified Full hotloop, without build or discovery."""
    try:
        if str(getattr(ns, "energy_command_contract_file", "") or "").strip():
            raw = _load_strict_json(
                Path(getattr(ns, "energy_command_contract_file")).expanduser()
            )
        else:
            raw = json.loads(str(ns.energy_command_contract_json or ""))
    except Exception as exc:
        print(json.dumps({"ok": False, "status": "full_command_contract_json_invalid", "error": str(exc)}), file=sys.stderr)
        return 5
    contract, status = _verified_energy_preflight_attestation(
        raw,
        path_value=str(ns.preflight_attestation or ""),
        expected_nonce=str(ns.preflight_nonce or ""),
        max_age_s=float(ns.preflight_attestation_max_age_s or 300.0),
        expected_root=ns.root,
    )
    if contract is None:
        print(json.dumps({"ok": False, "status": status}), file=sys.stderr, flush=True)
        return 5
    out = Path(str(ns.out_dir or "")).expanduser().resolve()
    if not str(ns.out_dir or "").strip():
        print(json.dumps({"ok": False, "status": "fresh_output_root_missing"}), file=sys.stderr)
        return 5
    out.mkdir(parents=True, exist_ok=True)
    workload = dict(contract["energy_workload"])
    artifacts = dict(contract["artifacts"])
    frames = max(1, int(ns.frames))
    requested_duration_s = max(
        0.0, float(getattr(ns, "duration_s", 0.0) or 0.0),
    )
    report_path = out / "native_full_energy_hotloop.json"
    if report_path.exists():
        print(json.dumps({"ok": False, "status": "fresh_report_preexists"}), file=sys.stderr)
        return 5
    kind = str(workload.get("kind") or "")
    input_binding_verified = True
    if kind == "hailo_full_hotloop":
        runtime_contract = workload.get("runtime_input_contract")
        canonical_input_names = workload.get("canonical_input_slot_names")
        canonical_output_names = workload.get("canonical_output_slot_names")
        runtime_artifact_key = str(workload.get("runtime_input_artifact") or "")
        input_manifest_artifact_key = str(workload.get("input_manifest_artifact") or "")
        runtime_artifact = artifacts.get(runtime_artifact_key)
        input_manifest_artifact = artifacts.get(input_manifest_artifact_key)
        if (
            workload.get("runtime_input_mode") != "exact_semantic_dump_runtime_tensor"
            or not isinstance(runtime_contract, Mapping)
            or runtime_contract.get("schema") != _PREVERIFIED_RUNTIME_INPUT_SCHEMA
            or int(runtime_contract.get("schema_version") or 0) != _PREVERIFIED_RUNTIME_INPUT_VERSION
            or not isinstance(runtime_artifact, Mapping)
            or not isinstance(input_manifest_artifact, Mapping)
            or str(runtime_artifact.get("sha256") or "")
            != str(runtime_contract.get("runtime_input_sha256") or "")
            or int(runtime_artifact.get("bytes") or 0)
            != int(runtime_contract.get("runtime_input_bytes") or 0)
            or not isinstance(canonical_input_names, list)
            or canonical_input_names != [str(runtime_contract.get("runtime_input_name") or "")]
            or not isinstance(canonical_output_names, list)
            or not canonical_output_names
        ):
            print(json.dumps({"ok": False, "status": "full_energy_runtime_input_binding_invalid"}), file=sys.stderr)
            return 5
        runner = str(artifacts[str(workload["runner_artifact"])]["path"])
        hef = str(artifacts[str(workload["hef_artifact"])]["path"])
        runtime_input = str(runtime_artifact["path"])
        runtime_python = str(
            artifacts[str(workload["runtime_python_artifact"])]["invocation_path"]
        )
        cmd = [
            runtime_python, runner,
            "--hef", hef,
            "--hw-arch", str(workload["hw_arch"]),
            "--runtime-api", str(workload["runtime_api"]),
            "--throughput-mode",
            "--counted-hotloop-only",
            "--frames", str(frames),
            "--warmup", str(int(workload["warmup"])),
            "--duration-s", str(requested_duration_s),
            "--inflight", str(int(workload["inflight"])),
            "--runtime-input-bin", runtime_input,
            "--runtime-input-contract-json", json.dumps(
                runtime_contract, sort_keys=True, separators=(",", ":"),
            ),
            "--preverified-runtime-input-sha256", str(runtime_artifact["sha256"]),
            "--canonical-input-slot-names-json", json.dumps(
                canonical_input_names, separators=(",", ":"),
            ),
            "--canonical-output-slot-names-json", json.dumps(
                canonical_output_names, separators=(",", ":"),
            ),
            "--image", str(workload["input_image"]),
            "--preverified-input-image-sha256", str(workload["input_image_sha256"]),
            "--task", str(workload["task"]),
            "--backend-label", str(contract["backend"]),
            "--model", str(contract["model"]),
            "--setup-id", str(contract.get("setup_id") or ""),
            "--comparison-backend", str(contract.get("comparison_backend") or ""),
            "--json-out", str(report_path),
        ]
        if workload.get("postprocess_required") is True:
            if workload.get("normalization_frozen") is True:
                cmd += [
                    "--frozen-decoded-nms-normalization-contract-json",
                    json.dumps(
                        workload[
                            "frozen_decoded_nms_normalization_contract"
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            else:
                cmd += [
                    "--frozen-postprocess-contract-json", json.dumps(
                        workload["frozen_postprocess_contract"],
                        sort_keys=True, separators=(",", ":"),
                    ),
                ]
            cmd += [
                "--original-image-wh-json", json.dumps(
                    workload["original_image_wh"], separators=(",", ":"),
                ),
            ]
        for key, enabled in (
            ("quantized-inputs", workload.get("quantized_inputs")),
            ("quantized-outputs", workload.get("quantized_outputs")),
            ("persistent-activation", workload.get("persistent_activation")),
            ("hotloop", workload.get("hotloop")),
            ("copy-inputs", workload.get("copy_inputs")),
            ("copy-outputs", workload.get("copy_outputs")),
        ):
            cmd.append(f"--{key}" if bool(enabled) else f"--no-{key}")
        step = _run(cmd, timeout=ns.timeout, cwd=ROOT, env=_hailo_python_env(), label="native-full-energy-hotloop:hailo")
        payload = _load_json(report_path) if report_path.is_file() else {}
        input_binding_verified = bool(
            isinstance(payload, Mapping)
            and payload.get("runtime_input_binding_verified") is True
            and payload.get("runtime_input_source") == "preflight_bound_runtime_input_tensor"
            and str(payload.get("runtime_input_file") or "") == str(Path(runtime_input).expanduser().resolve())
            and str(payload.get("runtime_input_sha256") or "") == str(runtime_artifact["sha256"])
            and str(payload.get("runtime_input_dtype") or "") == str(runtime_contract.get("runtime_input_dtype") or "")
            and list(payload.get("runtime_input_shape") or []) == list(runtime_contract.get("runtime_input_shape") or [])
            and int(payload.get("runtime_input_bytes") or 0) == int(runtime_contract.get("runtime_input_bytes") or 0)
            and payload.get("image_decode_performed") is False
            and payload.get("preprocessing_performed") is False
            and payload.get("preprocessing_timed") is False
            and (
                workload.get("postprocess_required") is not True
                or (
                    payload.get("postprocess_included") is True
                    and int(payload.get("postprocess_completed_frames") or 0)
                    == int(payload.get("completed_frames") or 0)
                    and payload.get(
                        "completed_task_endpoint_attested"
                    ) is True
                    and str(
                        payload.get(
                            "completed_task_endpoint_attestation_status"
                        ) or ""
                    ).strip().lower() == "passed"
                    and payload.get(
                        "completed_task_result_artifact_saved"
                    ) is True
                    and (
                        (
                            workload.get("normalization_frozen") is True
                            and payload.get("normalization_frozen") is True
                            and payload.get("host_postprocess_frozen")
                            is not True
                            and str(
                                payload.get(
                                    "frozen_decoded_nms_normalization_contract_sha256"
                                ) or ""
                            ) == str(
                                workload.get(
                                    "frozen_decoded_nms_normalization_contract_sha256"
                                ) or ""
                            )
                        )
                        or (
                            workload.get("normalization_frozen") is not True
                            and payload.get("host_postprocess_frozen") is True
                            and str(
                                payload.get(
                                    "frozen_host_postprocess_contract_sha256"
                                ) or ""
                            ) == str(
                                workload.get(
                                    "frozen_postprocess_contract_sha256"
                                ) or ""
                            )
                        )
                    )
                )
            )
        )
        completed = (
            int((payload or {}).get("completed_frames") or 0)
            if isinstance(payload, Mapping) and input_binding_verified else 0
        )
    elif kind == "deepx_full_prepared_feed_hotloop":
        runner = str(artifacts[str(workload["runner_artifact"])]["path"])
        dxnn = str(artifacts[str(workload["dxnn_artifact"])]["path"])
        runtime_artifact = artifacts.get(
            str(workload.get("runtime_input_artifact") or "")
        )
        if (
            not isinstance(runtime_artifact, Mapping)
            or workload.get("runtime_input_mode")
            != "exact_semantic_dump_runtime_tensor"
            or workload.get("runtime_input_binding_verified") is not True
            or str(runtime_artifact.get("sha256") or "").strip().lower()
            != str(workload.get("runtime_input_sha256") or "").strip().lower()
            or int(runtime_artifact.get("bytes") or 0)
            != int(workload.get("runtime_input_bytes") or 0)
        ):
            print(json.dumps({
                "ok": False,
                "status": "full_energy_deepx_runtime_input_binding_invalid",
            }), file=sys.stderr)
            return 5
        benchmark_set = _absolute_without_resolving(
            Path(str(contract.get("benchmark_set") or ""))
        )
        semantic_root = (
            benchmark_set / "native_full_outputs"
            / f"model={_safe_component(contract.get('model'))}"
            / "backend=native_full_deepx"
            / f"setup={_safe_component(contract.get('setup_id') or 'unspecified')}"
            / f"comparison={_safe_component(contract.get('comparison_backend') or 'unspecified')}"
        )
        prepared_input = str(_absolute_without_resolving(
            Path(str(runtime_artifact["path"]))
        ))
        dxnn_root = benchmark_set / "deepx"
        expected_hotloop_runner = (
            Path(__file__).resolve().parent
            / "native_deepx_full_energy_hotloop.py"
        )
        runtime_python = str(
            artifacts[str(workload["runtime_python_artifact"])]["invocation_path"]
        )
        cmd = [
            runtime_python, runner,
            "--dxnn", dxnn,
            "--prepared-input-file", prepared_input,
            "--expected-prepared-input-path", str(
                semantic_root / "runtime_input.bin"
            ),
            "--expected-prepared-input-root", str(semantic_root),
            "--expected-prepared-input-sha256", str(
                workload["runtime_input_sha256"]
            ),
            "--expected-prepared-input-bytes", str(
                int(workload["runtime_input_bytes"])
            ),
            "--expected-prepared-input-name", str(
                workload["runtime_input_name"]
            ),
            "--expected-prepared-input-shape-json", json.dumps(
                workload["runtime_input_shape"], separators=(",", ":"),
            ),
            "--expected-prepared-input-dtype", str(
                workload["runtime_input_dtype"]
            ),
            "--expected-prepared-input-layout", str(
                workload["runtime_input_layout"]
            ),
            "--runtime-preprocessing-identity-json", json.dumps(
                workload["runtime_preprocessing_identity"],
                sort_keys=True, separators=(",", ":"),
            ),
            "--expected-runtime-preprocessing-sha256", str(
                workload["runtime_preprocessing_sha256"]
            ),
            "--runtime-numeric-input-identity-json", json.dumps(
                workload["runtime_numeric_input_identity"],
                sort_keys=True, separators=(",", ":"),
            ),
            "--expected-runtime-numeric-input-sha256", str(
                workload["runtime_numeric_input_sha256"]
            ),
            "--original-image-wh-json", json.dumps(
                workload["original_image_wh"], separators=(",", ":"),
            ),
            "--prepared-feed-contract-version", str(
                workload["prepared_feed_contract_version"]
            ),
            "--frames", str(frames),
            "--warmup", str(int(workload["warmup"])),
            "--duration-s", str(requested_duration_s),
            "--task", str(workload.get("task") or "auto"),
            "--json-out", str(report_path),
            "--expected-runner-sha256", str(
                artifacts[str(workload["runner_artifact"])]["sha256"]
            ),
            "--expected-runner-path", str(expected_hotloop_runner),
            "--expected-runner-root", str(expected_hotloop_runner.parent),
            "--expected-dxnn-sha256", str(
                artifacts[str(workload["dxnn_artifact"])]["sha256"]
            ),
            "--expected-dxnn-path", dxnn,
            "--expected-dxnn-root", str(dxnn_root),
            "--source-contract-sha256", str(contract["contract_sha256"]),
            "--preflight-attestation", str(ns.preflight_attestation),
            "--preflight-nonce", str(ns.preflight_nonce),
        ]
        if workload.get("postprocess_required") is True:
            contract_option = (
                "--frozen-decoded-nms-normalization-contract-json"
                if workload.get("normalization_frozen") is True
                else "--frozen-postprocess-contract-json"
            )
            contract_field = (
                "frozen_decoded_nms_normalization_contract"
                if workload.get("normalization_frozen") is True
                else "frozen_postprocess_contract"
            )
            cmd += [
                contract_option, json.dumps(
                    workload[contract_field],
                    sort_keys=True, separators=(",", ":"),
                ),
                "--source-endpoint-contract-hash",
                str(
                    workload.get("source_endpoint_contract_hash")
                    or ""
                ),
            ]
        step = _run(
            cmd, timeout=ns.timeout, cwd=out,
            env=_deepx_hotloop_env(),
            label="native-full-energy-hotloop:deepx",
        )
        payload = _load_json(report_path) if report_path.is_file() else {}
        sealed_completion_result = (
            payload.get("frozen_decoded_nms_normalization_result")
            if workload.get("normalization_frozen") is True
            else payload.get("frozen_postprocess_result")
        ) if isinstance(payload, Mapping) else None
        if (
            workload.get("postprocess_required") is True
            and isinstance(payload, Mapping)
            and isinstance(sealed_completion_result, Mapping)
        ):
            (
                completed_persistence_ok,
                completed_persistence_status,
            ) = _completed_result_artifact_persistence_status(
                payload, sealed_result=sealed_completion_result,
                allowed_root=out,
                expected_path=report_path.with_name(
                    f"{report_path.stem}.completed_task_result_artifact.json"
                ),
            )
        else:
            completed_persistence_ok = (
                workload.get("postprocess_required") is not True
            )
            completed_persistence_status = (
                "not_applicable" if completed_persistence_ok
                else "completed_task_result_artifact_missing"
            )
        input_binding_verified = bool(
            isinstance(payload, Mapping)
            and payload.get("runtime_input_binding_verified") is True
            and payload.get("runtime_input_source")
            == "preflight_bound_runtime_input_tensor"
            and str(payload.get("runtime_input_file") or "")
            == prepared_input
            and str(payload.get("runtime_input_sha256") or "").strip().lower()
            == str(workload.get("runtime_input_sha256") or "").strip().lower()
            and int(payload.get("runtime_input_bytes") or 0)
            == int(workload.get("runtime_input_bytes") or 0)
            and str(payload.get("runtime_input_name") or "")
            == str(workload.get("runtime_input_name") or "")
            and list(payload.get("runtime_input_shape") or [])
            == list(workload.get("runtime_input_shape") or [])
            and str(payload.get("runtime_input_dtype") or "").strip().lower()
            == str(workload.get("runtime_input_dtype") or "").strip().lower()
            and str(payload.get("runtime_input_layout") or "").strip().upper()
            == str(workload.get("runtime_input_layout") or "").strip().upper()
            and dict(payload.get("runtime_preprocessing_identity") or {})
            == dict(workload.get("runtime_preprocessing_identity") or {})
            and str(payload.get("runtime_preprocessing_sha256") or "").strip().lower()
            == str(workload.get("runtime_preprocessing_sha256") or "").strip().lower()
            and dict(payload.get("runtime_numeric_input_identity") or {})
            == dict(workload.get("runtime_numeric_input_identity") or {})
            and str(payload.get("runtime_numeric_input_sha256") or "").strip().lower()
            == str(workload.get("runtime_numeric_input_sha256") or "").strip().lower()
            and payload.get("image_decode_performed") is False
            and payload.get("preprocessing_performed") is False
            and payload.get("preprocessing_timed") is False
            and (
                workload.get("postprocess_required") is not True
                or (
                    payload.get("postprocess_included") is True
                    and (
                        (
                            workload.get("normalization_frozen")
                            is True
                            and payload.get("normalization_frozen")
                            is True
                            and payload.get(
                                "host_postprocess_frozen"
                            )
                            is not True
                            and str(
                                payload.get(
                                    "frozen_decoded_nms_"
                                    "normalization_contract_sha256"
                                )
                                or ""
                            )
                            == str(
                                workload.get(
                                    "frozen_decoded_nms_"
                                    "normalization_contract_sha256"
                                )
                                or ""
                            )
                            and str(
                                payload.get(
                                    "completed_task_completion_mode"
                                )
                                or ""
                            )
                            == (
                                "integrated_accelerator_plus_"
                                "frozen_normalization"
                            )
                        )
                        or (
                            workload.get("host_postprocess_frozen")
                            is True
                            and payload.get(
                                "host_postprocess_frozen"
                            )
                            is True
                            and str(
                                payload.get(
                                    "frozen_postprocess_contract_sha256"
                                )
                                or ""
                            )
                            == str(
                                workload.get(
                                    "frozen_postprocess_contract_sha256"
                                )
                                or ""
                            )
                            and str(
                                payload.get(
                                    "source_endpoint_contract_hash"
                                )
                                or ""
                            ).strip().lower()
                            == str(
                                workload.get(
                                    "source_endpoint_contract_hash"
                                )
                                or ""
                            ).strip().lower()
                        )
                    )
                    and int(payload.get("postprocess_completed_frames") or 0)
                    == int(payload.get("completed_work_units") or 0)
                    and list(payload.get("original_image_wh") or [])
                    == list(workload.get("original_image_wh") or [])
                    and payload.get("completed_task_endpoint_attested") is True
                    and str(
                        payload.get(
                            "completed_task_endpoint_attestation_status"
                        ) or ""
                    ).strip().lower() == "passed"
                    and completed_persistence_ok
                )
            )
        )
        completed = (
            int((payload or {}).get("completed_work_units") or 0)
            if isinstance(payload, Mapping) and input_binding_verified else 0
        )
    elif kind == "tensorrt_full_completed_task_hotloop":
        runner_artifact = artifacts[
            str(workload["runner_artifact"])
        ]
        engine_artifact = artifacts[
            str(workload["engine_artifact"])
        ]
        input_manifest_artifact = artifacts[
            str(workload["input_manifest_artifact"])
        ]
        runtime_input_artifact = artifacts[
            str(workload["runtime_input_artifact"])
        ]
        runtime_python = str(
            artifacts[
                str(workload["runtime_python_artifact"])
            ]["invocation_path"]
        )
        completion_contract_option = (
            "--frozen-decoded-nms-normalization-contract-json"
            if workload.get("normalization_frozen") is True
            else "--frozen-postprocess-contract-json"
        )
        completion_contract_field = (
            "frozen_decoded_nms_normalization_contract"
            if workload.get("normalization_frozen") is True
            else "frozen_postprocess_contract"
        )
        classification = workload.get("task") == "classification"
        cmd = [
            runtime_python,
            str(runner_artifact["path"]),
            "--engine", str(engine_artifact["path"]),
            "--input-manifest", str(input_manifest_artifact["path"]),
            *(["--classification"] if classification else [
                completion_contract_option, json.dumps(
                    workload[completion_contract_field],
                    sort_keys=True, separators=(",", ":"),
                ),
            ]),
            "--source-endpoint-contract-hash",
            str(workload.get("source_endpoint_contract_hash") or ""),
            "--quality-first-producer-identity-sha256",
            str(
                workload.get(
                    "quality_first_producer_identity_sha256"
                ) or ""
            ),
            "--frames", str(frames),
            "--warmup", str(int(workload.get("warmup") or 0)),
            "--duration-s", str(requested_duration_s),
            "--json-out", str(report_path),
            "--expected-runner-sha256",
            str(runner_artifact["sha256"]),
            "--expected-engine-sha256",
            str(engine_artifact["sha256"]),
            "--expected-input-manifest-sha256",
            str(input_manifest_artifact["sha256"]),
            "--expected-runtime-input-sha256",
            str(runtime_input_artifact["sha256"]),
            "--source-contract-sha256",
            str(contract["contract_sha256"]),
            "--preflight-attestation",
            str(ns.preflight_attestation),
            "--preflight-nonce", str(ns.preflight_nonce),
        ]
        step = _run(
            cmd, timeout=ns.timeout, cwd=out,
            label="native-full-energy-hotloop:tensorrt-completed-task",
        )
        payload = _load_json(report_path) if report_path.is_file() else {}
        input_binding_verified = bool(
            isinstance(payload, Mapping)
            and payload.get("ok") is True
            and payload.get("preflight_verified") is True
            and payload.get("runtime_input_binding_verified") is True
            and str(payload.get("engine_sha256") or "")
            == str(engine_artifact["sha256"])
            and str(payload.get("input_manifest_sha256") or "")
            == str(input_manifest_artifact["sha256"])
            and str(payload.get("runtime_input_sha256") or "")
            == str(runtime_input_artifact["sha256"])
            and payload.get("postprocess_included") is True
            and payload.get("postprocess_completion_verified") is True
            and (
                (
                    classification
                    and payload.get("task_complete") is True
                    and payload.get("completed_task_stage") == "classification_top1_top5"
                )
                or (
                    workload.get("normalization_frozen") is True
                    and payload.get("normalization_frozen") is True
                    and payload.get("host_postprocess_frozen")
                    is not True
                    and str(
                        payload.get(
                            "frozen_decoded_nms_"
                            "normalization_contract_sha256"
                        )
                        or ""
                    )
                    == str(
                        workload.get(
                            "frozen_decoded_nms_"
                            "normalization_contract_sha256"
                        )
                        or ""
                    )
                    and str(
                        payload.get(
                            "completed_task_completion_mode"
                        )
                        or ""
                    )
                    == (
                        "integrated_accelerator_plus_"
                        "frozen_normalization"
                    )
                )
                or (
                    payload.get("host_postprocess_frozen") is True
                    and str(
                        payload.get(
                            "frozen_host_postprocess_contract_sha256"
                        )
                        or ""
                    )
                    == str(
                        workload.get(
                            "frozen_postprocess_contract_sha256"
                        )
                        or ""
                    )
                )
            )
            and int(payload.get("postprocess_completed_frames") or 0)
            == int(payload.get("completed_work_units") or 0)
            and str(payload.get("completed_task_stage") or "")
            == ("classification_top1_top5" if classification else "decoded_nms")
        )
        completed = (
            int((payload or {}).get("completed_work_units") or 0)
            if isinstance(payload, Mapping) and input_binding_verified
            else 0
        )
    elif kind == "tensorrt_full_hotloop":
        trtexec = str(artifacts[str(workload["trtexec_artifact"])]["path"])
        runtime_input = str(
            artifacts[str(workload["runtime_input_artifact"])]["path"]
        )
        runtime_input_name = str(workload.get("runtime_input_name") or "").strip()
        if not runtime_input_name:
            print(json.dumps({"ok": False, "status": "full_energy_runtime_input_name_missing"}), file=sys.stderr)
            return 5
        invariant_args = [str(value) for value in list(workload.get("invariant_args") or [])]
        forbidden = ("--onnx=", "--saveEngine=", "--build", "--dump")
        if any(arg.startswith(forbidden) for arg in invariant_args):
            print(json.dumps({"ok": False, "status": "full_energy_hotloop_contains_forbidden_preparation"}), file=sys.stderr)
            return 5
        timing_trace = out / "trtexec_iteration_times.json"
        # Energy is duration-controlled.  ``--iterations=1`` is only the
        # mandatory lower bound; ``--exportTimes`` supplies the exact work
        # count completed during the same command window.  Performance runs
        # remain explicit duration=0 iteration runs below the regular path.
        trt_duration_s = int(math.ceil(requested_duration_s)) if requested_duration_s > 0.0 else 0
        trt_requested_iterations = 1 if trt_duration_s > 0 else frames
        cmd = [
            trtexec, *invariant_args,
            f"--loadInputs={runtime_input_name}:{runtime_input}",
            f"--iterations={trt_requested_iterations}",
            f"--duration={trt_duration_s}", "--warmUp=0",
            f"--exportTimes={timing_trace}",
        ]
        step = _run(cmd, timeout=ns.timeout, cwd=out, label="native-full-energy-hotloop:tensorrt")
        timing_evidence = (
            _trtexec_exported_iteration_evidence(timing_trace)
            if int(step.get("rc") or 0) == 0 else {
                "status": "trtexec_failed", "completed_work_units": 0,
                "measured_trace_duration_s": None,
            }
        )
        completed = int(timing_evidence.get("completed_work_units") or 0)
        trace_duration_s = timing_evidence.get("measured_trace_duration_s")
        minimum_duration_satisfied = bool(
            requested_duration_s <= 0.0
            or (
                isinstance(trace_duration_s, (int, float))
                and float(trace_duration_s) >= requested_duration_s * 0.99
            )
        )
        report_path.write_text(json.dumps({
            "ok": int(step.get("rc") or 0) == 0,
            "backend": contract.get("backend"),
            "model": contract.get("model"),
            "completed_work_units": completed,
            "completed_work_units_source": "trtexec_export_times_record_count",
            "requested_work_units": frames,
            "trtexec_minimum_iterations": trt_requested_iterations,
            "requested_duration_s": requested_duration_s,
            "trtexec_duration_s": trt_duration_s,
            "measured_trace_duration_s": trace_duration_s,
            "minimum_duration_satisfied": minimum_duration_satisfied,
            "measurement_control": (
                "trtexec_duration_with_exact_exported_count"
                if requested_duration_s > 0.0 else "exact_iterations_duration_zero"
            ),
            "timing_evidence": timing_evidence,
            "iteration_trace": str(timing_trace),
            "command": cmd,
            "stdout_tail": _tail(step.get("stdout_tail")),
            "stderr_tail": _tail(step.get("stderr_tail")),
        }, indent=2), encoding="utf-8")
    else:
        print(json.dumps({"ok": False, "status": "full_energy_hotloop_unavailable", "kind": kind}), file=sys.stderr)
        return 5
    # Re-read the final report for every backend.  The TensorRT branch writes
    # its report directly above, while the Hailo/DeepX helper processes write
    # theirs.  A single common read avoids backend-specific state leaking into
    # the completion gate.
    payload = _load_json(report_path) if report_path.is_file() else {}
    fresh = bool(
        report_path.is_file()
        and report_path.stat().st_size > 0
    )
    payload_map = payload if isinstance(payload, Mapping) else {}
    if workload.get("classification_postprocess_required") is True:
        input_binding_verified = bool(
            input_binding_verified
            and payload_map.get("task_complete") is True
            and payload_map.get("completed_task_stage") == "classification_top1_top5"
            and payload_map.get("postprocess_completion_verified") is True
            and int(payload_map.get("postprocess_completed_frames") or 0) == completed
        )
    minimum_work_units = 1 if kind == "tensorrt_full_hotloop" and requested_duration_s > 0.0 else frames
    minimum_duration_satisfied = bool(
        requested_duration_s <= 0.0
        or payload_map.get("minimum_duration_satisfied") is True
    )
    ok = bool(
        int(step.get("rc") or 0) == 0
        and completed >= minimum_work_units
        and minimum_duration_satisfied
        and fresh
        and input_binding_verified
    )
    result = {
        "ok": ok,
        "status": "ok" if ok else "full_energy_hotloop_failed",
        "backend": contract.get("backend"),
        "model": contract.get("model"),
        "completed_work_units": completed if ok else None,
        "task": workload.get("task"),
        "workload_kind": kind,
        **{key: payload_map.get(key) for key in (
            "task_complete", "completed_task_stage", "postprocess_included",
            "postprocess_completed_frames", "postprocess_completion_verified",
        )},
        "completed_work_units_source": "verified_full_energy_hotloop",
        "requested_work_units": frames,
        "minimum_requested_work_units": minimum_work_units,
        "requested_duration_s": requested_duration_s,
        "minimum_duration_satisfied": minimum_duration_satisfied,
        "fresh_report_verified": fresh,
        "runtime_input_binding_verified": input_binding_verified,
        "report": str(report_path),
        "source_contract_sha256": contract.get("contract_sha256"),
    }
    if ok:
        # Emit the exact counter directly from the sealed Full hotloop.  The
        # energy collector consumes these markers from the measured command's
        # stdout, so no unbound reporting wrapper has to run inside the command
        # window.
        print(f"__SPLITPOINT_WORK_UNITS__={completed}", flush=True)
        print(
            f"__SPLITPOINT_WORK_UNITS_SOURCE__=verified_full_energy_hotloop:{kind}",
            flush=True,
        )
        print("__SPLITPOINT_WORK_UNITS_EXACT__=1", flush=True)
    print(json.dumps(result), flush=True)
    return 0 if ok else 5


def _first_case(benchmark_set: Path) -> tuple[str, Path, Path] | None:
    payload = _load_json(benchmark_set / "benchmark_set.json") or {}
    case_ids: list[str] = []
    for item in list(payload.get("cases") or []):
        if isinstance(item, Mapping):
            value = item.get("case_id") or item.get("id") or item.get("case") or item.get("folder")
        else:
            value = item
        if value:
            case_ids.append(str(value))
    if not case_ids:
        case_ids = [path.name for path in sorted(benchmark_set.glob("b*")) if path.is_dir()]
    for case in case_ids:
        for case_dir in (benchmark_set / case, benchmark_set / "legacy_suite" / case):
            runner = case_dir / "run_split_onnxruntime.py"
            if runner.is_file():
                return case, case_dir, runner
    for runner in sorted(benchmark_set.rglob("run_split_onnxruntime.py")):
        if runner.parent.name.startswith("b"):
            return runner.parent.name, runner.parent, runner
    return None


def _parse_image_map(raw: str) -> dict[str, dict[str, str]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except Exception:
        path = Path(text).expanduser()
        obj = _load_json(path) if path.is_file() else {}
    out: dict[str, dict[str, str]] = {}
    if isinstance(obj, Mapping):
        for model, rows in obj.items():
            if isinstance(rows, Mapping):
                out[str(model)] = {str(case): str(value) for case, value in rows.items() if str(value or "").strip()}
    return out


def _resolve_image(benchmark_set: Path, model: str, case: str, image_map: Mapping[str, Mapping[str, str]]) -> tuple[Path | None, str]:
    model_images = image_map.get(model) or {}
    value = str(model_images.get("full") or model_images.get(case) or "").strip()
    if value:
        raw = Path(value).expanduser()
        candidates = [raw] if raw.is_absolute() else [benchmark_set / raw, benchmark_set / "legacy_suite" / raw]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve(), "generic_validation_image_map"
        name = raw.name
        matches = sorted(path for path in benchmark_set.rglob(name) if path.is_file())
        if matches:
            return matches[0].resolve(), "generic_validation_image_map_rebased"
    val_root = benchmark_set / "resources" / "validation"
    images = sorted(
        path for path in val_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ) if val_root.is_dir() else []
    if images:
        return images[0].resolve(), "materialised_validation_subset:first_sorted"
    test_images = sorted(
        path for path in benchmark_set.rglob("test_image*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if test_images:
        return test_images[0].resolve(), "suite_test_image"
    return None, "image_unavailable"


def _run_plan_meta(benchmark_set: Path, run_id: str, model: str) -> tuple[str, str]:
    plan = _load_json(benchmark_set / "benchmark_plan.json") or {}
    rows = list(plan.get("runs") or plan.get("planned_runs") or []) if isinstance(plan, Mapping) else []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        rid = str(row.get("id") or row.get("run_id") or "")
        if rid == run_id:
            task = str(row.get("benchmark_task") or row.get("task") or "auto")
            scale = str(row.get("image_scale") or "auto")
            return task, scale
    task = "classification" if any(token in model.lower() for token in ("resnet", "mobilenet", "efficientnet", "convnext", "vit")) else "detection" if "yolo" in model.lower() else "auto"
    return task, "auto"


def _deepx_original_full_failure(
    benchmark_set: Path, model: str, run_id: str, *, setup_id: str = "",
) -> dict[str, Any]:
    """Retain the prior canonical Full error as diagnostic context only."""
    if run_id != "deepx_m1_full":
        return {}
    compact = benchmark_set / "results" / run_id / "original_full_failure.json"
    if _confined_regular_file(compact, allowed_root=benchmark_set, expected_path=compact):
        context = _load_json(compact)
        if (
            isinstance(context, Mapping)
            and context.get("schema") == "onnx-splitpoint/deepx-full-failure-context"
            and context.get("schema_version") == 1
            and context.get("diagnostic_only") is True
            and context.get("model") == model
            and context.get("run_id") == run_id
            and context.get("backend") == "deepx_m1"
            and context.get("variant") == "full"
            and (not setup_id or context.get("setup_id") == setup_id)
            and context.get("status") in {"runtime_failed", "failed", "error"}
            and str(context.get("error") or "").strip()
        ):
            error = str(context["error"])
            return {
                "original_full_result_file": str(context.get("source_file") or ""),
                "original_full_failure_context_file": str(compact),
                "original_full_status": str(context["status"]),
                "original_full_error": error, "error": error,
                "status_detail": f"Original DeepX Full failure: {error}",
            }
    filename = "benchmark_results_deepx_m1_full_auto.json"
    roots = [benchmark_set / "results", benchmark_set]
    if benchmark_set.name == "legacy_suite":
        roots.append(benchmark_set.parent.parent / "benchmark_results")
    elif benchmark_set.name == "benchmark_set":
        roots.append(benchmark_set.parent / "benchmark_results")
    for root in roots:
        source = root / filename
        if not _confined_regular_file(
            source, allowed_root=root, expected_path=source,
        ):
            continue
        for row in _iter_result_rows(source):
            prepared = row.get("deepx_prepared_feed_benchmark")
            if (
                str(row.get("run_id") or "") != run_id
                or str(row.get("backend") or row.get("provider") or "")
                != "deepx_m1"
                or str(row.get("variant") or row.get("primary_variant") or "")
                != "full"
                or str(row.get("model_id") or row.get("model") or model) != model
                or (
                    setup_id
                    and str(row.get("setup_id") or row.get("hardware_target_id") or setup_id)
                    != setup_id
                )
                or row.get("runtime_ok") is True
                or not isinstance(prepared, Mapping)
                or str(prepared.get("status") or "") not in {
                    "runtime_failed", "failed", "error",
                }
            ):
                continue
            error = str(prepared.get("error") or "").strip()
            if not error:
                continue
            return {
                "original_full_result_file": str(source),
                "original_full_status": str(prepared.get("status") or ""),
                "original_full_error": error,
                "error": error,
                "status_detail": f"Original DeepX Full failure: {error}",
            }
    return {}


def _semantic_full_dump(
    benchmark_set: Path,
    model: str,
    backend: str,
    run_id: str,
    ns: argparse.Namespace,
    *,
    force: bool = False,
) -> dict[str, Any]:
    # Run one untimed exact-input Native Full inference and write manifests.
    if not force and not bool(getattr(ns, "dump_outputs", False)):
        return {"ok": False, "status": "disabled", "failure_reason": "semantic_output_dump_disabled"}
    info = _first_case(benchmark_set)
    if info is None:
        return {"ok": False, "status": "runner_missing", "failure_reason": "semantic_runner_missing"}
    case, _case_dir, _runner = info
    image, image_source = _resolve_image(
        benchmark_set, model, case, getattr(ns, "image_map_data", {}) or {}
    )
    if image is None:
        return {"ok": False, "status": "input_missing", "failure_reason": "semantic_input_image_missing"}
    task, _image_scale = _run_plan_meta(benchmark_set, run_id, model)
    backend_token = {
        "native_full_tensorrt": "tensorrt",
        "native_full_deepx": "deepx",
    }.get(backend, "")
    if not backend_token:
        return {"ok": False, "status": "unsupported", "failure_reason": "semantic_backend_not_supported_by_companion"}

    prepared_input_manifest: Path | None = None
    model_images = (getattr(ns, "image_map_data", {}) or {}).get(model) or {}
    if backend == "native_full_deepx" and model_images.get("full"):
        # New workflows bind one model-wide comparison image before fan-out.
        # A Generic Full prepared feed may precede that selection and refer to
        # a different image. Prepare the selected image with the same backend
        # contract, once before the semantic/performance/energy dispatches.
        # Preserve the earlier Generic evidence in its own result directory.
        if image_source not in {
            "generic_validation_image_map", "generic_validation_image_map_rebased",
        } or not _confined_regular_file(image, allowed_root=benchmark_set):
            return {
                "ok": False, "status": "prepared_input_source_image_unavailable",
                "failure_reason": "deepx_comparison_input_image_unavailable",
            }
        try:
            from scripts.native_full_semantic_dump import _find_deepx_contract
            from onnx_splitpoint_tool.runners.native_full_input import (
                prepare_and_seal_deepx_native_full_input,
            )
            _dxnn, input_contract = _find_deepx_contract(benchmark_set)
            prepared = prepare_and_seal_deepx_native_full_input(
                image_path=image, input_contract=input_contract, task=task,
                out_dir=_native_full_dump_dir(benchmark_set, model, backend, ns) / "prepared_input",
                model=model, setup_id=str(ns.setup_id or ""),
                comparison_backend=str(ns.comparison_backend or ""),
            )
            prepared_input_manifest = Path(prepared["manifest_path"])
        except Exception as exc:
            return {
                "ok": False, "status": "prepared_input_manifest_unavailable",
                "failure_reason": "deepx_comparison_input_preparation_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    elif backend == "native_full_deepx":
        run_component = str(run_id or "").strip()
        if (
            not run_component
            or Path(run_component).name != run_component
            or run_component in {".", ".."}
        ):
            return {
                "ok": False,
                "status": "prepared_input_manifest_unavailable",
                "failure_reason": "deepx_shared_prepared_input_run_id_invalid",
            }
        expected_candidate = (
            benchmark_set / "results" / run_component / "prepared_input"
            / "native_full_input_manifest.json"
        )
        if not _confined_regular_file(
            expected_candidate,
            allowed_root=benchmark_set,
            expected_path=expected_candidate,
        ):
            return {
                "ok": False,
                "status": "prepared_input_manifest_unavailable",
                "failure_reason": "deepx_shared_prepared_input_manifest_missing",
                **_deepx_original_full_failure(
                    benchmark_set, model, run_id,
                    setup_id=str(getattr(ns, "setup_id", "") or ""),
                ),
            }
        prepared_payload_raw = _load_json(expected_candidate)
        prepared_payload = (
            dict(prepared_payload_raw)
            if isinstance(prepared_payload_raw, Mapping) else {}
        )
        if (
            prepared_payload.get("schema")
            != "onnx-splitpoint/native-full-input-dump"
            or prepared_payload.get("schema_version") != 2
            or prepared_payload.get("backend") != "native_full_deepx"
            or str(prepared_payload.get("task") or "").strip().lower()
            != str(task or "").strip().lower()
            or str(prepared_payload.get("model") or "").strip()
            != str(model or "").strip()
            or str(prepared_payload.get("setup_id") or "").strip()
            != str(ns.setup_id or "").strip()
            or str(
                prepared_payload.get("comparison_backend") or ""
            ).strip() != str(ns.comparison_backend or "").strip()
        ):
            return {
                "ok": False,
                "status": "prepared_input_manifest_unavailable",
                "failure_reason": "deepx_shared_prepared_input_identity_mismatch",
            }
        prepared_input_manifest = expected_candidate
        prepared_image_sha = str(
            prepared_payload.get("input_image_sha256") or ""
        ).strip().lower()
        declared_image_text = str(
            prepared_payload.get("input_image") or ""
        ).strip()
        image_matches_prepared = bool(
            prepared_image_sha
            and _confined_regular_file(
                image, allowed_root=benchmark_set,
                expected_path=image,
            )
            and _sha256_file(image) == prepared_image_sha
        )
        if declared_image_text:
            declared_image = Path(declared_image_text).expanduser()
            if not declared_image.is_absolute():
                declared_image = prepared_input_manifest.parent / declared_image
            if (
                _confined_regular_file(
                    declared_image, allowed_root=benchmark_set,
                )
                and _sha256_file(declared_image) == prepared_image_sha
            ):
                image = declared_image.resolve(strict=True)
                image_matches_prepared = True
        if not image_matches_prepared:
            matching_images = [
                candidate.resolve()
                for candidate in sorted(benchmark_set.rglob("*"))
                if _confined_regular_file(
                    candidate, allowed_root=benchmark_set,
                    expected_path=candidate,
                )
                and candidate.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                and _sha256_file(candidate) == prepared_image_sha
            ]
            unique_images = {str(candidate): candidate for candidate in matching_images}
            if len(unique_images) != 1:
                return {
                    "ok": False,
                    "status": "prepared_input_source_image_unavailable",
                    "failure_reason": (
                        "deepx_shared_prepared_input_source_image_missing"
                        if not unique_images else
                        "deepx_shared_prepared_input_source_image_ambiguous"
                    ),
                    "candidate_count": len(unique_images),
                }
            image = next(iter(unique_images.values()))

    python, child_env, sites = _suite_python_env(ns, backend)
    python_path = str(python or sys.executable)
    existing_py = child_env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "scripts"), *sites] + ([existing_py] if existing_py else [])
    )
    child_env["PYTHONUNBUFFERED"] = "1"
    dump_dir = _native_full_dump_dir(benchmark_set, model, backend, ns)
    if _path_contains_symlink(dump_dir):
        return {
            "ok": False,
            "status": "semantic_output_path_invalid",
            "failure_reason": "native_full_output_path_contains_symlink",
        }
    report_path = dump_dir / "native_full_semantic_dump.json"
    command = [
        python_path,
        str(ROOT / "scripts" / "native_full_semantic_dump.py"),
        "--benchmark-set", str(benchmark_set),
        "--backend", backend_token,
        "--model", model,
        "--task", task,
        "--image", str(image),
        "--out-dir", str(dump_dir),
        "--trt-precision", str(ns.trt_precision),
        "--setup-id", str(ns.setup_id or ""),
        "--comparison-backend", str(ns.comparison_backend or ""),
        "--json-out", str(report_path),
    ]
    expected_producer_sha = ""
    if backend == "native_full_tensorrt":
        producer, producer_status, producer_file = (
            _quality_first_trt_producer_identity(benchmark_set, model, ns)
        )
        if producer is None:
            return {
                "ok": False,
                "status": "quality_first_identity_failed",
                "failure_reason": producer_status,
                "quality_first_producer_set": str(producer_file),
            }
        expected_producer_sha = str(
            producer.get("producer_identity_sha256") or ""
        )
        command += _trt_quality_identity_cli_args(
            producer, semantic_dump=True,
        )
        command += [
            "--quality-first-producer-identity-sha256",
            expected_producer_sha,
        ]
    if backend == "native_full_deepx" and bool(getattr(ns, "diagnostic_deepx_input_probes", False)):
        command.append("--diagnostic-deepx-input-probes")
    if backend == "native_full_deepx" and prepared_input_manifest is not None:
        command += [
            "--prepared-input-manifest", str(prepared_input_manifest),
        ]
    step = _run(
        command, timeout=min(int(ns.timeout), 1800), cwd=ROOT,
        env=child_env, label=f"native-full-dump:{backend}:{model}",
    )
    semantic_command_sha = _canonical_json_sha256(
        [str(value) for value in command]
    )
    expected_output_manifest = dump_dir / "native_full_outputs_manifest.json"
    expected_input_manifest = dump_dir / "native_full_input_manifest.json"
    report_path_ok = _confined_regular_file(
        report_path, allowed_root=dump_dir, expected_path=report_path,
    )
    payload = _load_json(report_path) if report_path_ok else {}
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    manifest_text = str(payload.get("output_manifest") or "").strip()
    manifest = Path(manifest_text).expanduser() if manifest_text else expected_output_manifest
    if not manifest.is_absolute():
        manifest = dump_dir / manifest
    input_manifest_text = str(payload.get("input_manifest") or "").strip()
    input_manifest = Path(input_manifest_text).expanduser() if input_manifest_text else expected_input_manifest
    if not input_manifest.is_absolute():
        input_manifest = dump_dir / input_manifest
    output_manifest_path_ok = _confined_regular_file(
        manifest, allowed_root=dump_dir,
        expected_path=expected_output_manifest,
    )
    input_manifest_path_ok = _confined_regular_file(
        input_manifest, allowed_root=dump_dir,
        expected_path=expected_input_manifest,
    )
    manifest_payload = (
        _load_json(manifest) if output_manifest_path_ok else {}
    )
    manifest_payload = dict(manifest_payload) if isinstance(manifest_payload, Mapping) else {}
    if output_manifest_path_ok:
        identity_ok, identity_reason = _native_full_manifest_identity_status(
            manifest,
            model=model,
            backend=backend,
            setup_id=str(ns.setup_id or ""),
            comparison_backend=str(ns.comparison_backend or ""),
        )
    else:
        identity_ok, identity_reason = (
            False, "native_full_output_manifest_path_invalid",
        )
    input_manifest_payload = (
        _load_json(input_manifest) if input_manifest_path_ok else {}
    )
    input_identity_ok = bool(
        isinstance(input_manifest_payload, Mapping)
        and input_manifest_payload.get("schema")
        == "onnx-splitpoint/native-full-input-dump"
        and input_manifest_payload.get("schema_version") == 2
        and str(input_manifest_payload.get("backend") or "").strip()
        == str(backend or "").strip()
        and str(input_manifest_payload.get("model") or "").strip()
        == str(model or "").strip()
        and str(input_manifest_payload.get("setup_id") or "").strip()
        == str(ns.setup_id or "").strip()
        and str(
            input_manifest_payload.get("comparison_backend") or ""
        ).strip() == str(ns.comparison_backend or "").strip()
        and str(input_manifest_payload.get("case") or "").strip() == "full"
        and str(input_manifest_payload.get("task") or "").strip().lower()
        == str(task or "").strip().lower()
    )
    if not input_identity_ok and identity_ok:
        identity_ok = False
        identity_reason = "native_full_input_manifest_identity_mismatch"
    ok = (
        step.get("rc") == 0
        and report_path_ok
        and output_manifest_path_ok
        and input_manifest_path_ok
        and bool(payload.get("ok", True))
        and manifest.is_file()
        and identity_ok
    )
    if backend == "native_full_tensorrt":
        semantic_producer_sha = str(
            payload.get("quality_first_producer_identity_sha256") or ""
        ).strip().lower()
        manifest_producer_sha = str(
            manifest_payload.get("quality_first_producer_identity_sha256") or ""
        ).strip().lower()
        producer_binding_ok = bool(
            semantic_producer_sha == expected_producer_sha
            and manifest_producer_sha == expected_producer_sha
        )
        if not producer_binding_ok:
            ok = False
            identity_ok = False
            identity_reason = "semantic_quality_first_producer_identity_mismatch"
    else:
        producer_binding_ok = True
    return {
        "ok": ok,
        "status": "ok" if ok else "failed",
        "failure_reason": "" if ok else (
            "semantic_output_dump_timeout" if step.get("timed_out")
            else identity_reason if not identity_ok
            else str(payload.get("failure_reason") or "semantic_output_dump_failed")
        ),
        "output_dump_manifest": str(manifest) if output_manifest_path_ok else "",
        "native_output_manifest": str(manifest) if output_manifest_path_ok else "",
        "input_manifest": str(input_manifest) if input_manifest_path_ok else "",
        "input_case": case,
        "input_image": str(image),
        "input_image_source": image_source,
        "input_image_sha256": _sha256_file(image),
        "task": str(manifest_payload.get("task") or task),
        "output_format": str(manifest_payload.get("output_format") or ""),
        "contract_family": str(manifest_payload.get("contract_family") or ""),
        "stage": str(manifest_payload.get("stage") or ""),
        "contract_source": str(manifest_payload.get("contract_source") or "native_full_semantic_dump"),
        "endpoint_contract_complete": manifest_payload.get("endpoint_contract_complete") is True,
        "endpoint_contract_hash": str(manifest_payload.get("endpoint_contract_hash") or ""),
        "output_endpoint_id": str(
            manifest_payload.get("output_endpoint_id") or ""
        ),
        "tensor_signature": manifest_payload.get("tensor_signature")
        if isinstance(manifest_payload.get("tensor_signature"), Mapping) else {},
        "output_endpoint_attestation": manifest_payload.get("output_endpoint_attestation")
        if isinstance(manifest_payload.get("output_endpoint_attestation"), Mapping) else {},
        "frozen_host_postprocess_contract": manifest_payload.get(
            "frozen_host_postprocess_contract"
        ) if isinstance(manifest_payload.get("frozen_host_postprocess_contract"), Mapping) else {},
        "frozen_host_postprocess_contract_sha256": str(
            manifest_payload.get("frozen_host_postprocess_contract_sha256") or ""
        ),
        "frozen_host_postprocess_result": manifest_payload.get(
            "frozen_host_postprocess_result"
        ) if isinstance(manifest_payload.get("frozen_host_postprocess_result"), Mapping) else {},
        "frozen_decoded_nms_normalization_contract": manifest_payload.get(
            "frozen_decoded_nms_normalization_contract"
        ) if isinstance(
            manifest_payload.get(
                "frozen_decoded_nms_normalization_contract"
            ),
            Mapping,
        ) else {},
        "frozen_decoded_nms_normalization_contract_sha256": str(
            manifest_payload.get(
                "frozen_decoded_nms_normalization_contract_sha256"
            ) or ""
        ),
        "frozen_decoded_nms_normalization_result": manifest_payload.get(
            "frozen_decoded_nms_normalization_result"
        ) if isinstance(
            manifest_payload.get(
                "frozen_decoded_nms_normalization_result"
            ),
            Mapping,
        ) else {},
        "host_postprocess_frozen": manifest_payload.get("host_postprocess_frozen") is True,
        "normalization_frozen": manifest_payload.get("normalization_frozen") is True,
        "postprocess_included": manifest_payload.get("postprocess_included") is True,
        "quality_first_producer_identity_sha256": expected_producer_sha,
        "quality_first_producer_identity_match": producer_binding_ok,
        "semantic_execution_command_sha256": semantic_command_sha,
        "python": python_path,
        "extra_sites": sites,
        "report": str(report_path),
        "step": step,
    }


def _attach_semantic_dump(
    row: dict[str, Any], benchmark_set: Path, model: str,
    backend: str, run_id: str, ns: argparse.Namespace,
    *,
    precomputed_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        precomputed_result is None
        and not bool(getattr(ns, "dump_outputs", False))
    ):
        row.setdefault("semantic_dump_status", "disabled")
        return row
    if not bool(row.get("ok")):
        row.setdefault("semantic_dump_status", "skipped_runtime_failed")
        return row
    performance_prepared = (
        dict(row.get("deepx_prepared_feed_benchmark") or {})
        if isinstance(row.get("deepx_prepared_feed_benchmark"), Mapping)
        else {}
    )
    # Keep independent evidence independent. Only the measured hotloop owns
    # completion counts, results and its frozen contracts. Explicit false/empty
    # values are evidence too and must never be hidden by a truthiness fallback.
    measured_fields = (
        "task_complete", "completed_task_stage", "classification_topk",
        "frozen_host_postprocess_contract", "frozen_host_postprocess_contract_sha256",
        "frozen_host_postprocess_result", "host_postprocess_frozen",
        "frozen_decoded_nms_normalization_contract",
        "frozen_decoded_nms_normalization_contract_sha256",
        "frozen_decoded_nms_normalization_result", "normalization_frozen",
        "postprocess_included", "postprocess_completed_frames",
        "postprocess_completion_verified", "completed_task_endpoint_attestation",
        "completed_task_result_artifact", "completed_task_result_artifact_path",
        "completed_task_result_artifact_sha256", "completed_task_result_artifact_file_sha256",
        "completed_task_result_artifact_saved",
    )
    performance = dict(row)
    projection_conflicts = []
    if backend == "native_full_deepx":
        for key in measured_fields + (
            "prepared_input_source_image_id", "prepared_input_source_image_sha256",
            "source_endpoint_contract_hash", "source_output_endpoint_attestation",
        ):
            if key in performance_prepared:
                if key in performance and performance[key] != performance_prepared[key]:
                    projection_conflicts.append(key)
                else:
                    performance[key] = performance_prepared[key]
                    row.setdefault(key, performance_prepared[key])
    def reject(reason: str) -> None:
        # Preserve the first concrete failure and every measured runtime fact.
        if row.get("ok") is True:
            row["failure_reason"] = reason
        row["ok"] = False
        row["status"] = "failed"
        row["semantic_dump_status"] = "failed"
        row["semantic_dump_failure_reason"] = str(row.get("failure_reason") or reason)
        for flag in ("claim_ok", "claim_eligible", "eligible_for_scientific_claim",
                     "scientific_claim_eligible", "performance_claim_eligible",
                     "energy_claim_eligible", "e2e_claim_eligible"):
            if flag in row:
                row[flag] = False

    performance_source_id = str(
        performance.get("prepared_input_source_image_id") or ""
    ).strip()
    performance_source_sha = str(
        performance.get("prepared_input_source_image_sha256") or ""
    ).strip().lower()
    existing = str(row.get("output_dump_manifest") or row.get("native_output_manifest") or "").strip()
    result: dict[str, Any] | None = (
        dict(precomputed_result)
        if isinstance(precomputed_result, Mapping) else None
    )
    expected_existing = (
        _native_full_dump_dir(benchmark_set, model, backend, ns)
        / "native_full_outputs_manifest.json"
    )
    existing_path_ok = bool(
        existing
        and _confined_regular_file(
            Path(existing).expanduser(),
            allowed_root=expected_existing.parent,
            expected_path=expected_existing,
        )
    )
    if existing and not existing_path_ok:
        row.setdefault("warnings", []).append("native_full_existing_manifest_path_invalid")
    # A path and matching identity do not prove NMS. Explicit precomputed
    # evidence always wins; otherwise perform the normal checked preparation.
    if result is None:
        result = _semantic_full_dump(
            benchmark_set, model, backend, run_id, ns,
        )
    row["semantic_dump_status"] = str(result.get("status") or ("ok" if result.get("ok") else "failed"))
    row["semantic_dump_failure_reason"] = str(result.get("failure_reason") or "")
    for key in (
        "output_dump_manifest", "native_output_manifest", "input_manifest",
        "input_case", "input_image", "input_image_source", "input_image_sha256", "task",
        "output_format", "contract_family", "stage", "contract_source",
        "endpoint_contract_complete", "endpoint_contract_hash", "tensor_signature",
        "output_endpoint_attestation", "quality_first_producer_identity_sha256",
        "quality_first_producer_identity_match", "semantic_execution_command_sha256",
        "frozen_host_postprocess_contract",
        "frozen_host_postprocess_contract_sha256",
        "frozen_host_postprocess_result", "host_postprocess_frozen",
        "frozen_decoded_nms_normalization_contract",
        "frozen_decoded_nms_normalization_contract_sha256",
        "frozen_decoded_nms_normalization_result",
        "normalization_frozen",
        "postprocess_included",
    ):
        if backend == "native_full_deepx" and key in measured_fields:
            continue
        if result.get(key) not in (None, ""):
            row[key] = result.get(key)
    row["semantic_dump_report"] = str(result.get("report") or "")
    input_manifest_text = str(row.get("input_manifest") or "").strip()
    input_payload = _load_json(Path(input_manifest_text).expanduser()) if input_manifest_text else {}
    input_payload = dict(input_payload) if isinstance(input_payload, Mapping) else {}
    preprocess = input_payload.get("preprocess")
    preprocess = dict(preprocess) if isinstance(preprocess, Mapping) else {}
    manifest_source_path = Path(
        str(input_payload.get("input_image") or "")
    ).expanduser()
    manifest_source_sha = str(
        input_payload.get("input_image_sha256") or ""
    ).strip().lower()
    semantic_source_sha = str(
        result.get("input_image_sha256") or ""
    ).strip().lower()
    semantic_source_id = Path(
        str(result.get("input_image") or "")
    ).name
    semantic_performance_source_ok = bool(
        backend != "native_full_deepx"
        or (
            re.fullmatch(r"[0-9a-f]{64}", performance_source_sha)
            is not None
            and performance_source_sha
            == manifest_source_sha
            == semantic_source_sha
            and performance_source_id
            == semantic_source_id
            == manifest_source_path.name
            and not _path_contains_symlink(manifest_source_path)
            and manifest_source_path.is_file()
            and _sha256_file(manifest_source_path)
            == performance_source_sha
        )
    )
    row["semantic_performance_source_binding_verified"] = (
        semantic_performance_source_ok
    )
    row["semantic_performance_source_image_id"] = performance_source_id
    row["semantic_performance_source_image_sha256"] = (
        performance_source_sha
    )
    if not semantic_performance_source_ok:
        row["ok"] = False
        row["status"] = "failed"
        row["failure_reason"] = (
            "semantic_performance_source_image_mismatch"
        )
        row["semantic_dump_status"] = "failed"
        row["semantic_dump_failure_reason"] = row["failure_reason"]
    input_fields = {
        "runtime_input_dtype": str(input_payload.get("runtime_input_dtype") or input_payload.get("input_dtype") or ""),
        "runtime_input_shape": input_payload.get("runtime_input_shape") or input_payload.get("input_shape") or [],
        "runtime_input_layout": str(input_payload.get("runtime_input_layout") or preprocess.get("layout") or ""),
        "runtime_preprocess_mode": str(input_payload.get("runtime_preprocess_mode") or preprocess.get("mode") or ""),
        "runtime_normalization": str(input_payload.get("runtime_normalization") or preprocess.get("normalization") or preprocess.get("ort_model_scale") or ""),
        "runtime_color_space": str(input_payload.get("runtime_color_space") or ("RGB" if preprocess.get("rgb") is True else "")),
        "runtime_preprocessing_identity": dict(
            input_payload.get("runtime_preprocessing_identity") or {}
        ) if isinstance(
            input_payload.get("runtime_preprocessing_identity"), Mapping,
        ) else {},
        "runtime_preprocessing_sha256": str(
            input_payload.get("runtime_preprocessing_sha256") or ""
        ),
        "runtime_numeric_input_identity": dict(
            input_payload.get("runtime_numeric_input_identity") or {}
        ) if isinstance(
            input_payload.get("runtime_numeric_input_identity"), Mapping,
        ) else {},
        "runtime_numeric_input_sha256": str(
            input_payload.get("runtime_numeric_input_sha256") or ""
        ),
    }
    input_conflicts = []
    for field, value in input_fields.items():
        if (backend == "native_full_deepx"
                and field not in {"runtime_preprocess_mode", "runtime_normalization"}
                and field in performance and performance[field] not in (None, "", [], {})):
            if performance[field] != value:
                input_conflicts.append(field)
        elif row.get(field) in (None, "", []):
            row[field] = value
    if result.get("step"):
        row.setdefault("steps", []).append(result["step"])
    performance_frozen_sha = str(
        performance.get("frozen_host_postprocess_contract_sha256") or ""
    ).strip().lower()
    semantic_frozen_sha = str(
        result.get("frozen_host_postprocess_contract_sha256") or ""
    ).strip().lower()
    frozen_binding_required = bool(
        performance.get("host_postprocess_frozen") is True
        or (backend == "native_full_deepx"
            and str(performance.get("task") or result.get("task") or "") == "detection"
            and str(result.get("contract_family") or "") in {"raw_head", "decoded_pre_nms"})
    )
    if frozen_binding_required and (
        not performance_frozen_sha or semantic_frozen_sha != performance_frozen_sha
    ):
        row["ok"] = False
        row["status"] = "failed"
        row["failure_reason"] = "semantic_performance_frozen_postprocess_contract_mismatch"
        row["semantic_dump_status"] = "failed"
        row["semantic_dump_failure_reason"] = row["failure_reason"]
    performance_source_hash = str(
        performance.get("source_endpoint_contract_hash") or ""
    ).strip().lower()
    performance_source_attestation = performance.get("source_output_endpoint_attestation")
    performance_source_attestation = dict(performance_source_attestation) if isinstance(performance_source_attestation, Mapping) else {}
    semantic_source_hash = str(
        result.get("endpoint_contract_hash") or ""
    ).strip().lower()
    semantic_source_attestation = (
        dict(result.get("output_endpoint_attestation") or {})
        if isinstance(
            result.get("output_endpoint_attestation"), Mapping,
        )
        else {}
    )
    deepx_raw_source_binding_required = bool(
        backend == "native_full_deepx"
        and frozen_binding_required

    )
    if deepx_raw_source_binding_required and (
        re.fullmatch(
            r"[0-9a-f]{64}", performance_source_hash,
        )
        is None
        or semantic_source_hash != performance_source_hash
        or semantic_source_attestation != performance_source_attestation
        or str(result.get("stage") or "") != str(performance_source_attestation.get("stage") or "")
        or str(result.get("contract_family") or "") != str(performance_source_attestation.get("stage") or "")
        or result.get("tensor_signature") != performance_source_attestation.get("tensor_signature")
        or performance_source_attestation.get("attested") is not True
        or str(
            performance_source_attestation.get("status") or ""
        ).strip().lower()
        != "passed"
        or str(
            performance_source_attestation.get(
                "endpoint_contract_hash"
            )
            or ""
        ).strip().lower()
        != performance_source_hash
        or semantic_source_attestation.get("attested") is not True
        or str(
            semantic_source_attestation.get("status") or ""
        ).strip().lower()
        != "passed"
        or str(
            semantic_source_attestation.get(
                "endpoint_contract_hash"
            )
            or ""
        ).strip().lower()
        != semantic_source_hash
    ):
        row["ok"] = False
        row["status"] = "failed"
        row["failure_reason"] = (
            "semantic_performance_raw_source_endpoint_mismatch"
        )
        row["semantic_dump_status"] = "failed"
        row["semantic_dump_failure_reason"] = row["failure_reason"]
    performance_direct_sha = str(
        performance.get("frozen_decoded_nms_normalization_contract_sha256") or ""
    ).strip().lower()
    semantic_direct_sha = str(
        result.get(
            "frozen_decoded_nms_normalization_contract_sha256"
        ) or ""
    ).strip().lower()
    direct_binding_required = performance.get("normalization_frozen") is True
    if direct_binding_required and (
        not performance_direct_sha
        or semantic_direct_sha != performance_direct_sha
    ):
        row["ok"] = False
        row["status"] = "failed"
        row["failure_reason"] = (
            "semantic_performance_direct_normalization_contract_mismatch"
        )
        row["semantic_dump_status"] = "failed"
        row["semantic_dump_failure_reason"] = row["failure_reason"]
    if backend == "native_full_deepx":
        if input_conflicts:
            row["semantic_performance_input_conflict_fields"] = sorted(input_conflicts)
            reject("semantic_performance_runtime_input_mismatch")
        if projection_conflicts:
            row["semantic_performance_projection_conflict_fields"] = sorted(projection_conflicts)
            reject("deepx_performance_evidence_projection_conflict")
        if frozen_binding_required:
            if (performance.get("host_postprocess_frozen") is not True
                    or performance.get("postprocess_included") is not True):
                reject("deepx_performance_frozen_postprocess_evidence_invalid")
            semantic_problem = _deepx_semantic_completion_failure(result, expected_model=model)
            if semantic_problem:
                row["semantic_dump_status_detail"] = semantic_problem
                row["status_detail"] = str(row.get("status_detail") or "") + ";semantic=" + semantic_problem
                reject(semantic_problem)
            if (performance.get("frozen_host_postprocess_contract")
                    != result.get("frozen_host_postprocess_contract")):
                reject("semantic_performance_frozen_postprocess_contract_mismatch")
        if direct_binding_required:
            direct_problem = _deepx_semantic_completion_failure(result, expected_model=model)
            if direct_problem:
                row["semantic_dump_status_detail"] = direct_problem
                row["status_detail"] = str(row.get("status_detail") or "") + ";semantic=" + direct_problem
                reject(direct_problem)
        if direct_binding_required and (
                result.get("normalization_frozen") is not True
                or result.get("postprocess_included") is not True
                or not isinstance(result.get("frozen_decoded_nms_normalization_result"), Mapping)
                or not result.get("frozen_decoded_nms_normalization_result")
                or performance.get("frozen_decoded_nms_normalization_contract")
                    != result.get("frozen_decoded_nms_normalization_contract")):
            reject("semantic_performance_direct_normalization_contract_mismatch")
    if result.get("ok") is not True:
        reject(str(result.get("failure_reason") or "semantic_output_dump_failed"))
    if not result.get("ok"):
        row.setdefault("warnings", []).append(
            str(result.get("failure_reason") or "semantic output dump failed")
        )
    if row.get("ok") is not True:
        reject(str(row.get("failure_reason") or "semantic_output_dump_failed"))
    return row


def _attach_trt_completed_task_hotloop(
    row: dict[str, Any],
    benchmark_set: Path,
    model: str,
    ns: argparse.Namespace,
) -> dict[str, Any]:
    """Attach completed-task timing using the existing prepared-input runtime."""
    if not bool(row.get("ok")):
        return row
    classification = str(row.get("task") or "").strip().lower() == "classification"
    if str(row.get("task") or "").strip().lower() not in {"detection", "classification"}:
        row.setdefault(
            "comparison_endpoint_stratum",
            str(row.get("contract_family") or "accelerator_output"),
        )
        row.setdefault("measurement_concurrency", 1)
        return row
    contract_family = str(
        row.get("contract_family") or ""
    ).strip().lower()
    completion_kind = ""
    try:
        if classification:
            completion_kind = "classification_top1_top5"
            completion_contract = {}
        elif contract_family in {"raw_head", "decoded_pre_nms"}:
            completion_kind = "raw_host_tail"
            completion_contract = verify_frozen_postprocess_contract(
                row.get("frozen_host_postprocess_contract")
            )
            if completion_contract["source_contract_family"] != contract_family:
                raise FrozenPostprocessError("tensorrt_full_completion_source_family_mismatch")
        elif contract_family == "decoded_nms":
            completion_kind = "direct_bn6_normalization"
            completion_contract = (
                verify_frozen_decoded_nms_normalization_contract(
                    row.get(
                        "frozen_decoded_nms_normalization_contract"
                    )
                )
            )
        else:
            row.setdefault(
                "comparison_endpoint_stratum",
                contract_family or "accelerator_output",
            )
            row.setdefault("measurement_concurrency", 1)
            return row
    except Exception as exc:
        row.update({
            "ok": False,
            "status": "failed",
            "failure_reason": (
                "tensorrt_full_frozen_postprocess_missing"
                if contract_family in {"raw_head", "decoded_pre_nms"}
                else "tensorrt_full_direct_normalization_missing"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        })
        return row
    input_manifest = Path(
        str(row.get("input_manifest") or "")
    ).expanduser().resolve()
    input_payload = (
        _load_json(input_manifest) if input_manifest.is_file() else {}
    )
    input_payload = (
        dict(input_payload) if isinstance(input_payload, Mapping) else {}
    )
    producer = row.get("quality_first_producer_identity")
    producer = dict(producer) if isinstance(producer, Mapping) else {}
    engine_payload = (
        producer.get("engine")
        if isinstance(producer.get("engine"), Mapping) else {}
    )
    engine = Path(str(engine_payload.get("path") or "")).expanduser().resolve()
    runner = (
        ROOT / "scripts" / "native_trt_full_completed_hotloop.py"
    ).resolve()
    report_path = (
        input_manifest.parent
        / "native_trt_full_completed_hotloop.json"
    )
    expected_engine_sha = str(
        engine_payload.get("sha256") or row.get("engine_sha256") or ""
    ).strip().lower()
    expected_runtime_sha = str(
        input_payload.get("runtime_input_sha256") or ""
    ).strip().lower()
    expected_input_manifest_sha = (
        _sha256_file(input_manifest) if input_manifest.is_file() else ""
    )
    expected_runner_sha = _sha256_file(runner) if runner.is_file() else ""
    command = [
        str(getattr(ns, "engine_python_selected", "") or sys.executable),
        str(runner),
        "--engine", str(engine),
        "--input-manifest", str(input_manifest),
        *(["--classification"] if classification else [
            "--frozen-postprocess-contract-json" if completion_kind == "raw_host_tail"
            else "--frozen-decoded-nms-normalization-contract-json",
            json.dumps(completion_contract, sort_keys=True, separators=(",", ":")),
        ]),
        "--source-endpoint-contract-hash",
        str(row.get("endpoint_contract_hash") or ""),
        "--quality-first-producer-identity-sha256",
        str(row.get("quality_first_producer_identity_sha256") or ""),
        "--frames", str(max(1, int(ns.frames))),
        "--warmup", str(max(0, int(ns.warmup))),
        "--duration-s",
        str(max(0.0, float(getattr(ns, "duration_s", 0.0) or 0.0))),
        "--json-out", str(report_path),
        "--expected-runner-sha256", expected_runner_sha,
        "--expected-engine-sha256", expected_engine_sha,
        "--expected-input-manifest-sha256",
        expected_input_manifest_sha,
        "--expected-runtime-input-sha256", expected_runtime_sha,
    ]
    accelerator_diagnostic = {
        key: row.get(key)
        for key in (
            "fps_makespan", "latency_mean_ms", "latency_p50_ms",
            "latency_p95_ms", "latency_semantics", "fps_source",
            "completed_work_units", "completed_work_units_source",
            "completed_work_units_status", "report", "log_path",
        )
    }
    step = _run(
        command,
        timeout=ns.timeout,
        cwd=ROOT,
        label=f"native-full:tensorrt-completed-task:{model}",
    )
    payload = _load_json(report_path) if report_path.is_file() else {}
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    completed = int(payload.get("completed_work_units") or 0)
    postprocess_completed = int(
        payload.get("postprocess_completed_frames") or 0
    )
    completed_attestation = (
        dict(payload.get("completed_task_endpoint_attestation") or {})
        if isinstance(
            payload.get("completed_task_endpoint_attestation"), Mapping,
        )
        else {}
    )
    (
        completed_endpoint_attested,
        completed_endpoint_attestation_status,
    ) = _completed_attestation_aliases(completed_attestation)
    completion_contract_sha_key = (
        "frozen_host_postprocess_contract_sha256"
        if completion_kind == "raw_host_tail"
        else "frozen_decoded_nms_normalization_contract_sha256"
    )
    expected_completion_mode = (
        "frozen_host_tail"
        if completion_kind == "raw_host_tail"
        else "integrated_accelerator_plus_frozen_normalization"
    )
    binding_ok = bool(
        step.get("rc") == 0
        and payload.get("ok") is True
        and str(payload.get("benchmark_kind") or "")
        == "tensorrt_full_completed_task_hotloop"
        and str(payload.get("engine_sha256") or "")
        == expected_engine_sha
        and str(payload.get("input_manifest_sha256") or "")
        == expected_input_manifest_sha
        and str(payload.get("runtime_input_sha256") or "")
        == expected_runtime_sha
        and str(payload.get(completion_contract_sha_key) or "")
        == str(completion_contract.get("contract_sha256") or "")
        and completed >= max(1, int(ns.frames))
        and postprocess_completed == completed
        and payload.get("postprocess_completion_verified") is True
        and (
            classification
            and payload.get("task_complete") is True
            and payload.get("completed_task_stage") == "classification_top1_top5"
            or not classification and (
            str(payload.get("completed_task_stage") or "")
            == "decoded_nms"
            and isinstance(
                payload.get("completed_task_endpoint_attestation"), Mapping
            )
            and (
                payload.get("completed_task_endpoint_attestation") or {}
            ).get("attested") is True
            and completed_endpoint_attested
            and str(
                payload.get("completed_task_completion_mode")
                or completed_attestation.get(
                    "completed_task_completion_mode"
                )
                or ""
            ) == expected_completion_mode
            )
        )
    )
    row.setdefault("steps", []).append(step)
    row["accelerator_only_diagnostic"] = accelerator_diagnostic
    if not binding_ok:
        row.update({
            "ok": False,
            "status": "failed",
            "failure_reason": "tensorrt_full_completed_task_hotloop_failed",
            "error": str(
                payload.get("error")
                or step.get("stderr_tail")
                or step.get("stdout_tail")
                or ""
            ),
            "completed_task_hotloop_report": str(report_path),
        })
        return row
    row.update({
        "ok": True,
        "status": "ok",
        "request_latency": payload.get("request_latency"),
        "producer_impl": "native_tensorrt_full_completed_task",
        "performance_benchmark_source": (
            "tensorrt_native_full_completed_task_hotloop"
        ),
        "comparison_endpoint_stratum": "decoded_nms",
        "measurement_concurrency": 1,
        "fps_makespan": _num(payload.get("fps_makespan")),
        "latency_mean_ms": _num(payload.get("latency_mean_ms")),
        "latency_p50_ms": _num(payload.get("latency_p50_ms")),
        "latency_p95_ms": _num(payload.get("latency_p95_ms")),
        "latency_semantics": str(
            payload.get("latency_semantics") or ""
        ),
        "fps_source": "native_trt_full_completed_hotloop",
        "measured_duration_s": _num(payload.get("measured_duration_s")),
        "measurement_endpoint": "completed_task",
        "measurement_boundary": "first_task_start_to_last_task_completion",

        "completed_frames": completed,
        "completed_work_units": completed,
        "completed_work_units_source": str(
            payload.get("completed_work_units_source") or ""
        ),
        "completed_work_units_status": "exact_runtime_counter",
        "e2e_scope": "full_task_pipeline",
        "host_postprocess_frozen": (
            completion_kind == "raw_host_tail"
        ),
        "normalization_frozen": (
            completion_kind == "direct_bn6_normalization"
        ),
        "postprocess_included": True,
        "postprocess_completed_frames": postprocess_completed,
        "postprocess_completion_verified": True,
        "frozen_host_postprocess_contract": (
            completion_contract
            if completion_kind == "raw_host_tail" else {}
        ),
        "frozen_host_postprocess_contract_sha256": str(
            completion_contract.get("contract_sha256") or ""
            if completion_kind == "raw_host_tail" else ""
        ),
        "frozen_host_postprocess_result": dict(
            payload.get("frozen_host_postprocess_result") or {}
        ),
        "frozen_decoded_nms_normalization_contract": (
            completion_contract
            if completion_kind == "direct_bn6_normalization" else {}
        ),
        "frozen_decoded_nms_normalization_contract_sha256": str(
            completion_contract.get("contract_sha256") or ""
            if completion_kind == "direct_bn6_normalization" else ""
        ),
        "frozen_decoded_nms_normalization_result": dict(
            payload.get(
                "frozen_decoded_nms_normalization_result"
            ) or {}
        ),
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_contract_hash": str(
            payload.get("completed_task_endpoint_contract_hash") or ""
        ),
        "completed_task_output_endpoint_id": str(
            payload.get("completed_task_output_endpoint_id") or ""
        ),
        "completed_task_comparison_endpoint_contract": dict(
            payload.get(
                "completed_task_comparison_endpoint_contract"
            )
            or completed_attestation.get(
                "completed_task_comparison_endpoint_contract"
            )
            or {}
        ),
        "completed_task_comparison_endpoint_contract_hash": str(
            payload.get(
                "completed_task_comparison_endpoint_contract_hash"
            )
            or completed_attestation.get(
                "completed_task_comparison_endpoint_contract_hash"
            )
            or ""
        ),
        "completed_task_comparison_output_endpoint_id": str(
            payload.get(
                "completed_task_comparison_output_endpoint_id"
            )
            or completed_attestation.get(
                "completed_task_comparison_output_endpoint_id"
            )
            or ""
        ),
        "completed_task_completion_mode": str(
            payload.get("completed_task_completion_mode")
            or completed_attestation.get(
                "completed_task_completion_mode"
            )
            or ""
        ),
        "completed_task_endpoint_attested": completed_endpoint_attested,
        "completed_task_endpoint_attestation_status":
            completed_endpoint_attestation_status,
        "completed_task_endpoint_attestation": completed_attestation,
        "completed_task_result_artifact_saved": bool(
            payload.get("completed_task_result_artifact_saved") is True
        ),
        "completed_task_result_artifact": dict(
            payload.get("completed_task_result_artifact") or {}
        ),
        "completed_task_result_artifact_sha256": str(
            payload.get("completed_task_result_artifact_sha256") or ""
        ),
        "completed_task_result_artifact_path": str(
            payload.get("completed_task_result_artifact_path") or ""
        ),
        "completed_task_result_artifact_file_sha256": str(
            payload.get(
                "completed_task_result_artifact_file_sha256"
            ) or ""
        ),
        "completed_task_hotloop_report": str(report_path),
    })
    if classification:
        for key in ("task_complete", "completed_task_stage", "postprocess_completed_frames",
                    "postprocess_completion_verified", "classification_topk"):
            row[key] = payload.get(key)
        row["completed_task_contract_family"] = "classification_top1_top5"
        row["comparison_endpoint_stratum"] = "classification_top1_top5"
        row["completed_task_completion_mode"] = "host_top1_top5"
    return row


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_strict_json(path: Path) -> Any:
    """Read JSON while rejecting duplicate object keys at every depth."""

    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
    )


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:
        return []


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _truth(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "ok", "pass", "passed", "success"}:
        return True
    if text in {"0", "false", "no", "failed", "fail", "error"}:
        return False
    return None


def _percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _bootstrap_median_ci(
    values: Iterable[float], *, confidence: float = 0.95,
    resamples: int = 4000, seed_key: str = "native-full",
) -> tuple[float | None, float | None, float | None]:
    """Deterministic percentile-bootstrap CI for independent-run medians."""
    vals = [float(value) for value in values if math.isfinite(float(value))]
    if not vals:
        return None, None, None
    median = float(statistics.median(vals))
    if len(vals) < 2:
        return median, None, None
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    n = len(vals)
    boot = [
        float(statistics.median(vals[rng.randrange(n)] for _ in range(n)))
        for _ in range(max(200, int(resamples)))
    ]
    alpha = max(0.0, min(1.0, 1.0 - float(confidence)))
    return median, _percentile(boot, alpha / 2.0), _percentile(boot, 1.0 - alpha / 2.0)


def _aggregate_full_repetitions(
    repeat_rows: list[dict[str, Any]], *, requested: int,
) -> dict[str, Any]:
    """Collapse independent runs to a median row without best-of selection."""
    if not repeat_rows:
        return {"ok": False, "status": "missing_repetitions", "repetition_records": []}
    valid = [
        row for row in repeat_rows
        if bool(row.get("ok")) and _num(row.get("fps_makespan")) not in (None, 0.0)
    ]
    representative = dict(valid[0] if valid else repeat_rows[0])
    identity = "|".join(str(representative.get(key) or "") for key in (
        "backend", "model", "case", "setup_id", "comparison_backend",
        "execution_precision", "output_format", "contract_family",
    ))
    fps_values = [float(row["fps_makespan"]) for row in valid]
    fps_med, fps_low, fps_high = _bootstrap_median_ci(
        fps_values, seed_key=f"{identity}|fps",
    )
    latency_values = [
        float(row["latency_mean_ms"]) for row in valid
        if _num(row.get("latency_mean_ms")) is not None
    ]
    lat_med, lat_low, lat_high = _bootstrap_median_ci(
        latency_values, seed_key=f"{identity}|latency",
    )
    requested_n = max(1, int(requested))
    complete = len(valid) == requested_n and len(repeat_rows) == requested_n
    runtime_success = bool(
        len(repeat_rows) == requested_n
        and all(row.get("runtime_success") is True for row in repeat_rows)
    )
    common_identity_fields = (
        "backend", "model", "case", "setup_id", "comparison_backend",
        "execution_precision", "full_runtime_precision",
    )
    claim_identity_fields = (
        "source_onnx_sha256", "build_onnx_sha256",
        "runtime_artifact_sha256", "engine_sha256", "trtexec_sha256",
        "engine_build_receipt_sha256",
        "engine_build_receipt_file_sha256",
        "trt_engine_build_receipt_sha256",
        "quality_first_producer_identity_sha256",
        "quality_first_full_command_identity_sha256",
        "quality_first_semantic_command_identity_sha256",
        "native_full_execution_command_sha256",
        "endpoint_contract_hash", "output_endpoint_id", "contract_family",
        "stage", "task", "model_sha256", "input_image_sha256",
        "comparison_endpoint_stratum", "measurement_concurrency",
        "frozen_host_postprocess_contract_sha256",
        "completed_task_endpoint_contract_hash",
        "completed_task_output_endpoint_id",
        "run_id", "prepared_feed_contract_version",
        "prepared_input_sha256", "prepared_input_bytes",
        "prepared_input_name", "prepared_input_shape",
        "prepared_input_dtype", "prepared_input_layout",
        "prepared_input_source_image_id",
        "prepared_input_source_image_sha256",
        "runtime_preprocessing_sha256",
        "runtime_numeric_input_sha256",
    )

    def _identity_value(row: Mapping[str, Any], field: str) -> str:
        value = row.get(field)
        if isinstance(value, (Mapping, list, tuple)):
            return json.dumps(
                value, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            )
        return str(value or "")

    identity_drift_fields: list[str] = []
    identity_missing_fields: list[str] = []
    for field in common_identity_fields:
        values = {_identity_value(row, field) for row in valid}
        if len(values) > 1:
            identity_drift_fields.append(field)
    for field in claim_identity_fields:
        values = [_identity_value(row, field) for row in valid]
        nonempty = {value for value in values if value}
        if len(nonempty) > 1:
            identity_drift_fields.append(field)

    trt_rows = bool(valid) and all(
        str(row.get("backend") or "") == "native_full_tensorrt"
        for row in valid
    )
    if trt_rows:
        required_trt_fields = (
            "model", "setup_id", "execution_precision",
            "source_onnx_sha256", "build_onnx_sha256",
            "runtime_artifact_sha256", "trtexec_sha256",
            "engine_build_receipt_sha256",
            "engine_build_receipt_file_sha256",
            "trt_engine_build_receipt_sha256",
            "quality_first_producer_identity_sha256",
            "quality_first_full_command_identity_sha256",
            "quality_first_semantic_command_identity_sha256",
            "native_full_execution_command_sha256",
            "endpoint_contract_hash",
        )
        for field in required_trt_fields:
            if any(not _identity_value(row, field) for row in valid):
                identity_missing_fields.append(field)
        if all(
            str(row.get("task") or "").strip().lower() == "detection"
            and str(row.get("contract_family") or "").strip().lower()
            in {"raw_head", "decoded_pre_nms"}
            for row in valid
        ):
            for field in (
                "comparison_endpoint_stratum",
                "measurement_concurrency",
                "frozen_host_postprocess_contract_sha256",
                "completed_task_endpoint_contract_hash",
                "completed_task_output_endpoint_id",
            ):
                if any(not _identity_value(row, field) for row in valid):
                    identity_missing_fields.append(field)
        for row in valid:
            producer = row.get("quality_first_producer_identity")
            if not isinstance(producer, Mapping):
                identity_missing_fields.append(
                    "quality_first_producer_identity"
                )
                continue
            producer = dict(producer)
            declared = str(
                producer.pop("producer_identity_sha256", "") or ""
            ).strip().lower()
            if (
                re.fullmatch(r"[0-9a-f]{64}", declared) is None
                or _canonical_json_sha256(producer) != declared
                or declared != str(
                    row.get("quality_first_producer_identity_sha256") or ""
                ).strip().lower()
            ):
                identity_missing_fields.append(
                    "quality_first_producer_identity_invalid"
                )
    deepx_rows = bool(valid) and all(
        str(row.get("backend") or "") == "native_full_deepx"
        for row in valid
    )
    if deepx_rows:
        required_deepx_fields = (
            "run_id", "prepared_feed_contract_version",
            "prepared_input_sha256", "prepared_input_bytes",
            "prepared_input_name", "prepared_input_shape",
            "prepared_input_dtype", "prepared_input_layout",
            "prepared_input_source_image_id",
            "prepared_input_source_image_sha256",
            "runtime_preprocessing_sha256",
            "runtime_numeric_input_sha256",
        )
        for field in required_deepx_fields:
            if any(not _identity_value(row, field) for row in valid):
                identity_missing_fields.append(field)
    identity_drift_fields = sorted(set(identity_drift_fields))
    identity_missing_fields = sorted(set(identity_missing_fields))
    identity_consistent = bool(
        complete and not identity_drift_fields and not identity_missing_fields
    )
    if not identity_consistent:
        complete = False
    runtime_instance_ids = [str(row.get("runtime_instance_id") or "") for row in repeat_rows]
    independence_verified = bool(
        complete and all(runtime_instance_ids)
        and len(set(runtime_instance_ids)) == len(runtime_instance_ids)
        and all(int(_num(row.get("completed_work_units") or row.get("completed_frames") or row.get("frames")) or 0) > 0 for row in repeat_rows)
        and len({int(_num(row.get("completed_work_units") or row.get("completed_frames") or row.get("frames")) or 0) for row in repeat_rows}) == 1
    )
    failed_repetition = next(
        (row for row in repeat_rows if not bool(row.get("ok"))), None,
    )
    primary_repetition_failure = (
        {
            "repetition_index": int(
                failed_repetition.get("repetition_index") or 1
            ),
            "status": str(failed_repetition.get("status") or ""),
            "failure_stage": str(failed_repetition.get("failure_stage") or ""),
            "primary_failure_reason": str(
                failed_repetition.get("primary_failure_reason")
                or failed_repetition.get("failure_reason") or ""
            ),
            "failure_reason": str(
                failed_repetition.get("failure_reason") or ""
            ),
            "status_detail": str(
                failed_repetition.get("status_detail") or ""
            ),
            "error": str(failed_repetition.get("error") or ""),
            # Preserve absence instead of silently converting it into a
            # successful execution receipt.  The terminal Full-line verifier
            # requires an explicit zero/false pair.
            "returncode": (
                int(_num(failed_repetition.get("returncode")))
                if _num(failed_repetition.get("returncode")) is not None
                and not isinstance(failed_repetition.get("returncode"), bool)
                else None
            ),
            "timed_out": (
                failed_repetition.get("timed_out")
                if isinstance(failed_repetition.get("timed_out"), bool)
                else None
            ),
            "stdout_tail": str(
                failed_repetition.get("stdout_tail") or ""
            ),
            "stderr_tail": str(
                failed_repetition.get("stderr_tail") or ""
            ),
            "report": str(failed_repetition.get("report") or ""),
        }
        if isinstance(failed_repetition, Mapping) else {}
    )
    representative.update({
        "ok": bool(complete and fps_med is not None),
        "runtime_success": runtime_success,
        "status": "ok" if complete and fps_med is not None else "partial_repetitions",
        "fps_makespan": fps_med,
        "fps_median": fps_med,
        "fps_makespan_median": fps_med,
        "fps_ci95_low": fps_low,
        "fps_ci95_high": fps_high,
        "fps_makespan_ci95_low": fps_low,
        "fps_makespan_ci95_high": fps_high,
        "latency_mean_ms": lat_med,
        "latency_median_ms": lat_med,
        "latency_ci95_low_ms": lat_low,
        "latency_ci95_high_ms": lat_high,
        "repetition_count_requested": requested_n,
        "repetition_count_attempted": len(repeat_rows),
        "repetition_count_valid": len(valid),
        "repetitions_requested": requested_n,
        "repetitions_completed": len(valid),
        "repetition_status": "complete" if complete else "partial",
        "repetition_aggregation": "median_with_deterministic_percentile_bootstrap_ci95",
        "repetition_runtime_scope": "fresh_process_per_repetition",
        "repetition_independence_verified": independence_verified,
        "repetition_identity_consistent": identity_consistent,
        "repetition_identity_drift_fields": identity_drift_fields,
        "repetition_identity_missing_fields": identity_missing_fields,
        "repetition_runtime_instance_ids": runtime_instance_ids,
        "fps_repetition_samples": fps_values,
        "latency_mean_repetition_samples_ms": latency_values,
        # Keep the aggregate status stable for downstream compatibility, but
        # also retain the concrete first child failure.  This makes a failed
        # Full line actionable without reopening remote terminal logs.
        "primary_repetition_failure": primary_repetition_failure,
        "primary_repetition_failure_reason": str(
            primary_repetition_failure.get("failure_reason") or ""
        ),
        "primary_repetition_status_detail": str(
            primary_repetition_failure.get("status_detail") or ""
        ),
        "primary_repetition_error": str(
            primary_repetition_failure.get("error") or ""
        ),
        "repetition_records": [
            {
                "repetition_index": int(row.get("repetition_index") or index),
                "ok": bool(row.get("ok")), "status": str(row.get("status") or ""),
                "runtime_success": row.get("runtime_success") is True,
                "fps_makespan": _num(row.get("fps_makespan")),
                "request_latency": row.get("request_latency"),
                "latency_mean_ms": _num(row.get("latency_mean_ms")),
                "latency_p50_ms": _num(row.get("latency_p50_ms")),
                "latency_p95_ms": _num(row.get("latency_p95_ms")),
                "frames": int(_num(row.get("completed_work_units") or row.get("completed_frames") or row.get("frames")) or 0),
                "completed_frames": _num(row.get("completed_frames")),
                "completed_work_units": _num(row.get("completed_work_units", row.get("completed_frames"))),
                "completed_work_units_status": row.get("completed_work_units_status"),
                **{key: row.get(key) for key in (
                    "task", "task_complete", "completed_task_stage", "measurement_endpoint", "measurement_boundary", "measured_duration_s", "makespan_ms",
                    "postprocess_completion_verified", "completed_task_endpoint_attested", "repetition_id",
                    "completed_task_endpoint_contract_hash", "completion_execution_contract_sha256",
                    "completed_work_units_source", "completed_work_units_status", "postprocess_completed_frames")},
                "repetition_id": str(row.get('repetition_id') or row.get('runtime_instance_id') or row.get('repetition_index') or index),

                "failure_reason": str(row.get("failure_reason") or ""),
                "failure_stage": str(row.get("failure_stage") or ""),
                "primary_failure_reason": str(
                    row.get("primary_failure_reason") or row.get("failure_reason") or ""
                ),
                "status_detail": str(row.get("status_detail") or ""),
                "error": str(row.get("error") or ""),
                "returncode": (
                    int(_num(row.get("returncode")))
                    if _num(row.get("returncode")) is not None
                    and not isinstance(row.get("returncode"), bool)
                    else None
                ),
                "timed_out": (
                    row.get("timed_out")
                    if isinstance(row.get("timed_out"), bool) else None
                ),
                "stdout_tail": str(row.get("stdout_tail") or ""),
                "stderr_tail": str(row.get("stderr_tail") or ""),
                "report": str(row.get("report") or ""),
                "runtime_instance_id": str(row.get("runtime_instance_id") or ""),
            }
            for index, row in enumerate(repeat_rows, start=1)
        ],
    })
    if identity_drift_fields:
        representative["status"] = "identity_drift"
        representative["failure_reason"] = "native_full_repetition_identity_drift"
    elif identity_missing_fields:
        representative["status"] = "identity_incomplete"
        representative["failure_reason"] = "native_full_repetition_identity_incomplete"
    elif not complete:
        representative["failure_reason"] = "native_full_repetition_set_incomplete"
    if not complete:
        representative["result_ok"] = False
        representative["performance_claim_eligible"] = False
    if primary_repetition_failure:
        # The first valid repetition may be the representative of a mixed
        # series.  Its metadata must not hide the concrete failed attempt.
        representative["primary_failure_reason"] = str(
            primary_repetition_failure.get("primary_failure_reason") or ""
        )
        if primary_repetition_failure.get("failure_stage"):
            representative["failure_stage"] = primary_repetition_failure["failure_stage"]
    representative["repetition_evidence"] = representative["repetition_records"]
    return representative


def _explicit_runtime_precision(*sources: Mapping[str, Any] | None) -> str:
    """Return only precision evidence explicitly scoped to the Full runtime.

    The historical ``precision`` field is intentionally excluded: for Native
    Full rows it stores the Split comparison stratum supplied through
    ``--comparison-precision`` and says nothing about Full execution.
    """
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("full_runtime_precision", "execution_precision", "runtime_precision"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return ""


def _model_roots(root: Path, models: list[str]) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    if models:
        for model in models:
            benchmark_set = root / model / "benchmark_set"
            if (benchmark_set / "benchmark_set.json").is_file():
                rows.append((model, benchmark_set))
        return rows
    for benchmark_set_json in root.glob("*/benchmark_set/benchmark_set.json"):
        rows.append((benchmark_set_json.parent.parent.name, benchmark_set_json.parent))
    return sorted(rows)


def _parse_trtexec_text(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    patterns = [
        ("fps_makespan", r"Throughput:\s*([0-9.]+)\s*qps"),
        ("fps_makespan", r"throughput(?:_qps|\s+fps)?\s*[:=]\s*([0-9.]+)"),
        ("latency_mean_ms", r"Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms"),
        ("gpu_compute_mean_ms", r"GPU Compute Time:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms"),
    ]
    for key, pattern in patterns:
        if key in out:
            continue
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            out[key] = float(match.group(1))
    latency_full = re.search(
        r"Latency:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        text,
        flags=re.IGNORECASE,
    )
    if latency_full:
        out.update(
            {
                "latency_min_ms": float(latency_full.group(1)),
                "latency_max_ms": float(latency_full.group(2)),
                "latency_mean_ms": float(latency_full.group(3)),
            }
        )
    return out


def _parse_trtexec_log(path: Path) -> dict[str, Any]:
    return _parse_trtexec_text(path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else "")


FPS_KEYS = (
    "pipeline_fps_selected",
    "full_backend_throughput_fps",
    "throughput_primary_fps",
    "full_fps",
    "fps_makespan",
    "throughput_fps",
    "throughput_qps",
    "fps",
)
LATENCY_KEYS = (
    "pipeline_cycle_selected_ms",
    "full_mean_ms",
    "full_latency_ms",
    "total_latency_ms",
    "latency_mean_ms",
    "latency_ms",
    "full_ms",
    "cycle_ms",
)


def _iter_result_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        yield from _read_csv(path)
        return
    payload = _load_json(path)
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, Mapping):
                yield dict(row)
    elif isinstance(payload, Mapping):
        rows = payload.get("rows") or payload.get("results")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    yield dict(row)
        else:
            yield dict(payload)


def _metrics_from_rows(paths: Iterable[Path]) -> dict[str, Any]:
    """Read one benchmark result without letting the lossy CSV shadow JSON.

    The CSV export stringifies nested mappings such as
    ``deepx_prepared_feed_benchmark``.  It is useful for tables, but it cannot
    carry the exact outer-makespan contract consumed below.  JSON is therefore
    the canonical machine-readable result whenever it is present.  CSV is only
    a compatibility fallback and a scalar cross-check; a conflicting scalar
    value fails closed instead of silently selecting either representation.
    """
    candidates: list[dict[str, Any]] = []
    json_paths_seen: list[str] = []
    for path in paths:
        if path.suffix.lower() == ".json":
            json_paths_seen.append(str(path))
        for row in _iter_result_rows(path):
            variant = str(row.get("variant") or row.get("primary_variant") or "full").strip().lower()
            if variant not in {"", "full"}:
                continue
            fps = next((_num(row.get(key)) for key in FPS_KEYS if _num(row.get(key)) not in (None, 0.0)), None)
            latency = next((_num(row.get(key)) for key in LATENCY_KEYS if _num(row.get(key)) not in (None, 0.0)), None)
            candidates.append({
                "path": path,
                "kind": "json" if path.suffix.lower() == ".json" else "csv",
                "row": row,
                "fps": fps,
                "latency": latency,
            })

    json_rows = [item for item in candidates if item["kind"] == "json"]
    if json_paths_seen and not json_rows:
        return {
            "source_consistent": False,
            "source_consistency_reason": "canonical_json_unreadable_or_empty",
            "canonical_json_paths": json_paths_seen,
        }
    eligible = json_rows or [item for item in candidates if item["kind"] == "csv"]
    if not eligible:
        return {}
    # Preserve caller ordering (newest invocation first), preferring a row with
    # an actual metric only within the already-selected canonical format.
    selected = next(
        (item for item in eligible if item["fps"] is not None or item["latency"] is not None),
        eligible[0],
    )
    selected_row = selected["row"]
    selected_run_id = str(selected_row.get("run_id") or "").strip()
    selected_variant = str(
        selected_row.get("variant") or selected_row.get("primary_variant") or "full"
    ).strip().lower()

    def same_identity(item: Mapping[str, Any]) -> bool:
        row = item.get("row") if isinstance(item.get("row"), Mapping) else {}
        run_id = str(row.get("run_id") or "").strip()
        variant = str(row.get("variant") or row.get("primary_variant") or "full").strip().lower()
        return (not selected_run_id or not run_id or run_id == selected_run_id) and variant == selected_variant

    def close_enough(left: float, right: float) -> bool:
        return abs(float(left) - float(right)) <= max(
            # Convenience CSV writers commonly round to six decimals.  That
            # representation loss is not a scientific disagreement; anything
            # outside this small serialization envelope still fails closed.
            5e-6, 1e-6 * max(abs(float(left)), abs(float(right)), 1.0),
        )

    conflicts: list[dict[str, Any]] = []
    if selected["kind"] == "json":
        # CSV may omit or stringify nested evidence.  Cross-check only scalar
        # values represented natively in both files.
        scalar_fields = (
            ("fps_makespan", FPS_KEYS),
            ("latency_mean_ms", LATENCY_KEYS),
            ("measured_makespan_s", ("measured_makespan_s",)),
            ("completed_frames", ("completed_frames",)),
        )
        for other in candidates:
            if other["kind"] != "csv" or not same_identity(other):
                continue
            other_row = other["row"]
            for field, aliases in scalar_fields:
                left = next((_num(selected_row.get(key)) for key in aliases if _num(selected_row.get(key)) is not None), None)
                right = next((_num(other_row.get(key)) for key in aliases if _num(other_row.get(key)) is not None), None)
                if left is not None and right is not None and not close_enough(left, right):
                    conflicts.append({
                        "field": field,
                        "canonical_json_value": left,
                        "csv_value": right,
                        "csv_source": str(other["path"]),
                    })
    return {
        "fps_makespan": selected["fps"],
        "latency_mean_ms": selected["latency"],
        "result_source": str(selected["path"]),
        "result_source_kind": selected["kind"],
        "result_row": selected_row,
        "source_consistent": not conflicts,
        "source_consistency_reason": (
            "canonical_json_csv_scalar_conflict" if conflicts else "ok"
        ),
        "source_conflicts": conflicts,
    }


def _deepx_prepared_feed_projection(
    metric_row: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project the exact DeepX prepared-feed contract without losing nesting."""

    nested_raw = metric_row.get("deepx_prepared_feed_benchmark")
    nested = dict(nested_raw) if isinstance(nested_raw, Mapping) else {}
    if (
        not nested.get("performance_benchmark_source")
        and str(nested.get("benchmark_kind") or "") == "prepared_feed_dx_engine"
    ):
        nested["performance_benchmark_source"] = "dx_engine_prepared_feed"
    aliases = {
        "fps_makespan": ("fps_makespan",),
        "makespan_s": ("measured_makespan_s", "makespan_s"),
        "completed_frames": ("completed_frames",),
        "completed_work_units": ("completed_work_units",),
        "completed_work_units_source": ("completed_work_units_source",),
        "completed_work_units_status": ("completed_work_units_status",),
        "performance_benchmark_source": ("performance_benchmark_source",),
        "prepared_feed_contract_version": ("prepared_feed_contract_version",),
        "postprocess_completed_frames": ("postprocess_completed_frames",),
        "postprocess_included": ("postprocess_included",),
        "host_postprocess_frozen": ("host_postprocess_frozen",),
        "normalization_frozen": ("normalization_frozen",),
        "postprocess_completion_verified": (
            "postprocess_completion_verified",
        ),
    }
    numeric = {
        "fps_makespan", "makespan_s", "completed_frames",
        "completed_work_units", "postprocess_completed_frames",
    }
    projection: dict[str, Any] = {}
    conflicts: list[dict[str, Any]] = []
    for field, top_names in aliases.items():
        nested_value = nested.get(field)
        top_value = next(
            (
                metric_row.get(name) for name in top_names
                if metric_row.get(name) is not None
            ),
            None,
        )
        # The suite's top-level projection is only a duplicate.  It must
        # never upgrade a missing/legacy nested prepared-feed contract to v3.
        chosen = (
            nested_value
            if field == "prepared_feed_contract_version"
            else nested_value if nested_value is not None else top_value
        )
        if chosen is not None:
            projection[field] = chosen
        if (
            field == "prepared_feed_contract_version"
            and nested_value is None
            and top_value is not None
        ):
            conflicts.append({
                "field": field,
                "nested_value": None,
                "top_level_value": top_value,
                "reason": "nested_contract_version_missing",
            })
            continue
        if nested_value is None or top_value is None:
            continue
        if field in numeric:
            left = _num(nested_value)
            right = _num(top_value)
            matches = bool(
                left is not None and right is not None
                and abs(float(left) - float(right)) <= max(
                    1e-9, 1e-6 * max(abs(float(left)), abs(float(right)), 1.0),
                )
            )
        else:
            matches = nested_value == top_value
        if not matches:
            conflicts.append({
                "field": field,
                "nested_value": nested_value,
                "top_level_value": top_value,
            })
    if (
        "completed_work_units" not in projection
        and projection.get("completed_frames") is not None
    ):
        projection["completed_work_units"] = projection["completed_frames"]
    if (
        "completed_work_units_status" not in projection
        and projection.get("performance_benchmark_source")
        == "dx_engine_prepared_feed"
        and projection.get("completed_work_units") is not None
    ):
        projection["completed_work_units_status"] = "exact_runtime_counter"
    projection["nested_payload"] = nested
    return projection, conflicts


def _diagnostic_fields(step: Mapping[str, Any], *, failure_reason: str = "", status_detail: str = "") -> dict[str, Any]:
    returncode = (
        step.get("rc")
        if step.get("rc") is not None
        else step.get("returncode")
    )
    timed_out = step.get("timed_out")
    return {
        "failure_reason": failure_reason,
        "status_detail": status_detail,
        "error": str(step.get("error") or ""),
        "timed_out": bool(timed_out) if timed_out is not None else None,
        "returncode": int(returncode) if returncode is not None else None,
        "stdout_tail": _tail(step.get("stdout_tail")),
        "stderr_tail": _tail(step.get("stderr_tail")),
    }


def _native_trt_full(benchmark_set: Path, model: str, ns: argparse.Namespace) -> dict[str, Any]:
    engine_python = str(getattr(ns, "engine_python_selected", "") or "")
    if not engine_python:
        step={"rc":127,"returncode":127,"timed_out":False,"stdout_tail":"","stderr_tail":"no ONNX-capable Python interpreter", "error":"no_onnx_capable_engine_build_python"}
        return {
            "backend":"native_full_tensorrt","producer_impl":"native_tensorrt_full",
            "model":model,"case":"full","execution_mode":"native_full_baseline",
            "duration_s":float(getattr(ns,"duration_s",0.0) or 0.0),"frames":int(ns.frames),
            "ok":False,"status":"failed","fps_makespan":None,"latency_mean_ms":None,
            "execution_precision": str(ns.trt_precision or ""),
            "full_runtime_precision": str(ns.trt_precision or ""),
            "runtime_precision_source": "trt_precision_cli",
            "engine_precision": str(ns.trt_precision or ""),
            "steps":[step],**_diagnostic_fields(step,failure_reason="no_onnx_capable_engine_build_python")
        }
    producer, producer_status, producer_file = _quality_first_trt_producer_identity(
        benchmark_set, model, ns,
    )
    if producer is None:
        step = {
            "rc": 4, "returncode": 4, "timed_out": False,
            "stdout_tail": "", "stderr_tail": producer_status,
            "error": producer_status,
        }
        return {
            "backend": "native_full_tensorrt",
            "producer_impl": "native_tensorrt_full",
            "model": model, "case": "full",
            "execution_mode": "native_full_baseline",
            "duration_s": float(getattr(ns, "duration_s", 0.0) or 0.0),
            "frames": int(ns.frames), "ok": False,
            "status": "quality_first_identity_failed",
            "fps_makespan": None, "latency_mean_ms": None,
            "execution_precision": str(ns.trt_precision or ""),
            "full_runtime_precision": str(ns.trt_precision or ""),
            "runtime_precision_source": "quality_first_producer",
            "engine_precision": str(ns.trt_precision or ""),
            "quality_first_producer_set": str(producer_file),
            "steps": [step],
            **_diagnostic_fields(step, failure_reason=producer_status),
        }
    producer_sha = str(producer["producer_identity_sha256"])
    engine_path = Path(str(producer["engine"]["path"])).resolve()
    cmd = [
        engine_python,
        str(ROOT / "scripts" / "native_trt_from_benchmarkset.py"),
        "--benchmark-set",
        str(benchmark_set),
        "--variants",
        "full",
        "--precision",
        ns.trt_precision,
        "--run-smoke",
        "--iterations",
        str(ns.frames),
        "--warmup-iterations",
        str(max(0, int(ns.warmup))),
        "--warmup-ms",
        "0",
        "--duration-s",
        "0",
        "--workspace-mb",
        str(ns.workspace_mb),
        "--workspace-mode",
        "auto",
        "--no-build",
    ]
    # Explicit mode never discovers or derives an engine path.  All child
    # outputs live beside the already sealed persistent Quality-FIRST engine.
    cmd += ["--out-dir", str(engine_path.parent)]
    cmd += _trt_quality_identity_cli_args(producer, semantic_dump=False)
    cmd += ["--quality-first-producer-identity-sha256", producer_sha]
    if ns.no_shapes:
        cmd.append("--no-shapes")
    step = _run(cmd, timeout=ns.timeout, label=f"native-full:tensorrt:{model}")
    execution_command_sha = _canonical_json_sha256([str(value) for value in cmd])
    outdir = engine_path.parent
    log_path = outdir / "run_trtexec.log"
    meta_path = outdir / "native_trt_meta.json"
    metrics = _parse_trtexec_log(log_path)
    meta = _load_json(meta_path) or {}
    if isinstance(meta, Mapping):
        for key in FPS_KEYS:
            value = _num(meta.get(key))
            if value and value > 0 and not _num(metrics.get("fps_makespan")):
                metrics["fps_makespan"] = value
                metrics["fps_source"] = f"native_trt_meta:{key}"
                break
        for key in LATENCY_KEYS:
            value = _num(meta.get(key))
            if value and value > 0 and not _num(metrics.get("latency_mean_ms")):
                metrics["latency_mean_ms"] = value
                break
        for key in ("latency_p50_ms", "latency_p95_ms"):
            if _num(meta.get(key)) is not None:
                metrics[key] = _num(meta.get(key))
        if (
            str(meta.get("completed_work_units_status") or "") == "exact_runtime_counter"
            and _num(meta.get("latency_mean_ms")) is not None
        ):
            metrics["latency_mean_ms"] = _num(meta.get("latency_mean_ms"))
            metrics["latency_source"] = "trtexec_export_times"
    build_ok = _truth(meta.get("build_ok")) if isinstance(meta, Mapping) else None
    run_ok = _truth(meta.get("run_ok")) if isinstance(meta, Mapping) else None
    quality_identity = (
        meta.get("quality_first_identity")
        if isinstance(meta, Mapping) else None
    )
    quality_identity_match = bool(
        isinstance(quality_identity, Mapping)
        and str(
            quality_identity.get("quality_first_producer_identity_sha256") or ""
        ).strip().lower() == producer_sha
        and str((quality_identity.get("paths") or {}).get("engine") or "")
        == str(engine_path)
        and str((quality_identity.get("hashes") or {}).get("engine") or "")
        == str(producer["engine"]["sha256"])
    )
    exact_count_ok = bool(
        isinstance(meta, Mapping)
        and str(meta.get("completed_work_units_status") or "") == "exact_runtime_counter"
        and int(_num(meta.get("completed_work_units")) or -1) == int(ns.frames)
    )
    # The trtexec process duration includes startup/warmup. Only its existing
    # measured exportTimes trace binds a classification throughput to time.
    trace = _load_json(Path(str((meta.get("trtexec_export_times") or {}).get("path") or "")))
    trace = trace if isinstance(trace, list) else next((trace[key] for key in
        ("times", "queries", "records", "results", "data")
        if isinstance(trace, Mapping) and isinstance(trace.get(key), list)), [])
    starts = [_num(r.get("startH2dMs")) for r in trace if isinstance(r, Mapping)]
    ends = [_num(r.get("endD2hMs")) for r in trace if isinstance(r, Mapping)]
    trace_duration_s = None
    if (exact_count_ok and len(trace) == int(ns.frames) and len(starts) == len(trace)
            and all(a is not None and b is not None and b >= a for a, b in zip(starts, ends))):
        trace_duration_s = (max(ends) - min(starts)) / 1000.0
        if trace_duration_s <= 0:
            trace_duration_s = None
    fps = _num(metrics.get("fps_makespan"))
    trace_rate = bool(str(producer.get("task")) == "classification" and trace_duration_s)
    if trace_rate:
        fps = int(meta["completed_work_units"]) / trace_duration_s
    ok = bool(
        step.get("rc") == 0 and build_ok is not False and run_ok is not False
        and quality_identity_match and exact_count_ok and fps is not None and fps > 0
    )
    reason = ""
    if step.get("timed_out"):
        reason = "native_tensorrt_timeout"
    elif step.get("rc") != 0:
        reason = "native_tensorrt_runner_failed"
    elif build_ok is False:
        reason = "native_tensorrt_build_failed"
    elif run_ok is False:
        reason = str(meta.get("run_failure_reason") or "native_tensorrt_execution_failed")
    elif not quality_identity_match:
        reason = "native_tensorrt_quality_first_identity_mismatch"
    elif not exact_count_ok:
        reason = "native_tensorrt_completed_work_unit_verification_failed"
    elif not fps:
        reason = "native_tensorrt_metrics_missing"
    return {
        "backend": "native_full_tensorrt",
        "producer_impl": "native_tensorrt_full",
        "model": model,
        "setup_id": str(producer["setup_id"]),
        "case": "full",
        "execution_mode": "native_full_baseline",
        "execution_precision": str(ns.trt_precision or ""),
        "full_runtime_precision": str(ns.trt_precision or ""),
        "runtime_precision_source": "trt_precision_cli",
        "engine_precision": str(ns.trt_precision or ""),
        "duration_s": float(getattr(ns, "duration_s", 0.0) or 0.0),
        "frames": int(ns.frames),
        "warmup": int(ns.warmup),
        "completed_work_units": int(meta.get("completed_work_units")) if exact_count_ok else None,
        "completed_work_units_source": str(meta.get("completed_work_units_source") or ""),
        "completed_work_units_status": str(meta.get("completed_work_units_status") or ""),
        "warmup_policy": str(meta.get("warmup_policy") or ""),
        "warmup_iterations": int(meta.get("warmup_iterations_requested") or 0),
        "warmup_iterations_completed": meta.get("warmup_iterations_completed"),
        "warmup_iterations_status": str(meta.get("warmup_iterations_status") or ""),
        "ok": ok,
        "status": "ok" if ok else "failed",
        "fps_makespan": fps,
        "latency_mean_ms": _num(metrics.get("latency_mean_ms")),
        "latency_p50_ms": _num(metrics.get("latency_p50_ms")),
        "latency_p95_ms": _num(metrics.get("latency_p95_ms")),
        "latency_semantics": "request_end_to_end_trtexec",
        "gpu_compute_mean_ms": _num(metrics.get("gpu_compute_mean_ms")),
        "fps_source": "trtexec_export_times_host_trace" if trace_rate else metrics.get("fps_source") or ("run_trtexec.log" if fps else ""),
        "measured_duration_s": trace_duration_s if trace_rate else None,
        "measurement_endpoint": "completed_task" if trace_rate else "",
        "measurement_boundary": "first_task_start_to_last_task_completion" if trace_rate else "",
        "classification_rate_source": "trtexec_export_times_host_trace" if trace_rate else "",
        "trtexec_reported_fps": _num(metrics.get("fps_makespan")),
        "report": str(meta_path),
        "log_path": str(log_path),
        "input_manifest": str(meta.get("input_manifest") or "") if isinstance(meta, Mapping) else "",
        "runtime_input_dtype": str(meta.get("runtime_input_dtype") or meta.get("input_dtype") or "") if isinstance(meta, Mapping) else "",
        "runtime_input_shape": (meta.get("runtime_input_shape") or meta.get("input_shape") or []) if isinstance(meta, Mapping) else [],
        "runtime_input_layout": str(meta.get("runtime_input_layout") or meta.get("input_layout") or "") if isinstance(meta, Mapping) else "",
        "runtime_preprocess_mode": str(meta.get("runtime_preprocess_mode") or meta.get("preprocess_mode") or "") if isinstance(meta, Mapping) else "",
        "runtime_normalization": str(meta.get("runtime_normalization") or meta.get("normalization") or "") if isinstance(meta, Mapping) else "",
        "runtime_color_space": str(meta.get("runtime_color_space") or meta.get("color_space") or "") if isinstance(meta, Mapping) else "",
        "quality_first_producer_identity": dict(producer),
        "quality_first_producer_identity_sha256": producer_sha,
        "quality_first_full_command_identity_sha256": producer_sha,
        "quality_first_semantic_command_identity_sha256": producer_sha,
        "native_full_execution_command_sha256": execution_command_sha,
        "source_onnx_sha256": str(producer["source_onnx"]["sha256"]),
        "build_onnx_sha256": str(producer["build_onnx"]["sha256"]),
        "runtime_artifact_sha256": str(producer["engine"]["sha256"]),
        "engine_sha256": str(producer["engine"]["sha256"]),
        "trtexec_sha256": str(producer["trtexec"]["sha256"]),
        "engine_build_receipt_sha256": str(
            producer["engine_build_receipt"]["sha256"]
        ),
        "engine_build_receipt_file_sha256": str(
            producer["engine_build_receipt_file_sha256"]
        ),
        "trt_engine_build_receipt_sha256": str(
            producer["engine_build_receipt"]["receipt"]["receipt_sha256"]
        ),
        "endpoint_contract_hash": str(producer["endpoint_contract_hash"]),
        "endpoint_contract_complete": producer["endpoint_contract_complete"] is True,
        "task": str(producer["task"]),
        "quality_first_producer_set": str(producer_file),
        "quality_first_identity_status": producer_status,
        "quality_first_runtime_identity_match": quality_identity_match,
        "steps": [step],
        **_diagnostic_fields(step, failure_reason=reason, status_detail=str(meta.get("status") or "") if isinstance(meta, Mapping) else ""),
    }


def _canonical_hailo_full_contract_backend(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    if token.startswith("native_full_"):
        token = token[len("native_full_"):]
    if token in {"hailo10", "hailo10h"}:
        return "hailo10"
    return token


def _prepare_verified_hailo_full_contract_overlay(
    benchmark_set: Path,
    *,
    dump_dir: Path,
    model: str,
    task: str,
    hw_arch: str,
    hef: Path,
    verified_receipt: Mapping[str, Any],
    onnx_python: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create and re-attest one isolated, receipt-bound Full declaration."""
    from onnx_splitpoint_tool.native_output_endpoint import (
        load_authoritative_output_contract,
    )
    from onnx_splitpoint_tool.hailo_full_contract_promotion import (
        promote_verified_hailo_full_contracts,
    )

    benchmark_path = benchmark_set / "benchmark_set.json"
    contracts_path = benchmark_set / "output_contracts.json"
    suite_bench_raw = _load_strict_json(benchmark_path)
    source_payload_raw = _load_strict_json(contracts_path)
    if not isinstance(suite_bench_raw, Mapping):
        raise ValueError("hailo_full_overlay_benchmark_set_invalid")
    if not isinstance(source_payload_raw, Mapping):
        raise ValueError("hailo_full_overlay_output_contracts_invalid")
    suite_bench = dict(suite_bench_raw)
    source_payload = dict(source_payload_raw)
    normalized_task = str(task or "").strip().lower()
    if normalized_task not in {"classification", "detection"}:
        raise ValueError("hailo_full_overlay_task_unsupported")
    if str(source_payload.get("model_id") or "") != str(model):
        raise ValueError("hailo_full_overlay_model_mismatch")
    if (
        str(source_payload.get("task") or "").strip().lower()
        != normalized_task
    ):
        raise ValueError("hailo_full_overlay_task_mismatch")
    raw_contracts = source_payload.get("contracts")
    if (
        not isinstance(raw_contracts, list)
        or not raw_contracts
        or any(not isinstance(row, Mapping) for row in raw_contracts)
    ):
        raise ValueError("hailo_full_overlay_contract_rows_invalid")
    contracts = [dict(row) for row in raw_contracts]
    target_backend = _canonical_hailo_full_contract_backend(hw_arch)
    promotions = promote_verified_hailo_full_contracts(
        suite_dir=benchmark_set,
        model_id=str(model),
        task=normalized_task,
        suite_bench=suite_bench,
        contracts=contracts,
        copied_verified={},
    )
    selected_promotions = [
        dict(row) for row in promotions
        if isinstance(row, Mapping)
        and _canonical_hailo_full_contract_backend(row.get("backend"))
        == target_backend
    ]
    matching_indexes = [
        index for index, row in enumerate(contracts)
        if _canonical_hailo_full_contract_backend(row.get("backend"))
        == target_backend
        and str(row.get("model_id") or "") == str(model)
        and str(row.get("variant") or "full").strip().lower() == "full"
    ]
    if len(selected_promotions) != 1 or len(matching_indexes) != 1:
        raise ValueError("hailo_full_overlay_exact_promotion_required")

    index = matching_indexes[0]
    contract = dict(contracts[index])
    artifact_text = str(
        contract.get("recorded_artifact_path")
        or contract.get("artifact_path") or ""
    ).strip()
    artifact = Path(artifact_text).expanduser()
    if not artifact.is_absolute():
        artifact = benchmark_set / artifact
    try:
        artifact = artifact.resolve(strict=True)
        expected_hef = hef.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("hailo_full_overlay_artifact_unavailable") from exc
    compiler_end_nodes = [
        str(value).strip()
        for value in list(verified_receipt.get("compiler_end_nodes") or [])
    ]
    contract_end_nodes = [
        str(value).strip()
        for value in list(contract.get("full_end_node_names") or [])
    ]
    raw_endpoint_origin = str(
        verified_receipt.get("raw_endpoint_origin") or ""
    ).strip()
    contract_raw_endpoint_origin = str(
        contract.get("raw_endpoint_origin") or ""
    ).strip()
    source_raw_head_attestation = (
        dict(verified_receipt.get("source_onnx_raw_head_attestation") or {})
        if isinstance(
            verified_receipt.get("source_onnx_raw_head_attestation"),
            Mapping,
        )
        else {}
    )
    source_raw_head_attestation_sha256 = _strict_sha256_token(
        verified_receipt.get(
            "source_onnx_raw_head_attestation_sha256"
        )
    )
    is_detection = normalized_task == "detection"
    compiler_cut_endpoint = bool(
        is_detection and raw_endpoint_origin == "compiler_end_nodes"
    )
    source_graph_endpoint = (
        is_detection
        and raw_endpoint_origin == "source_onnx_graph_outputs"
    )
    endpoint_boundary_verified = False
    if compiler_cut_endpoint:
        endpoint_boundary_verified = bool(
            compiler_end_nodes
            and contract_end_nodes == compiler_end_nodes
            and contract_raw_endpoint_origin == "compiler_end_nodes"
            and contract.get("source_onnx_multiscale_raw_head") is False
            and not source_raw_head_attestation
            and not source_raw_head_attestation_sha256
        )
    elif source_graph_endpoint:
        source_onnx_sha256 = _strict_sha256_token(
            verified_receipt.get("source_onnx_sha256")
        )
        compiler_onnx_sha256 = _strict_sha256_token(
            verified_receipt.get("compiler_onnx_sha256")
        )
        source_onnx_path = Path(
            str(verified_receipt.get("source_onnx_path") or "")
        )
        compiler_onnx_path = Path(
            str(verified_receipt.get("compiler_onnx_path") or "")
        )
        recomputed_attestation: dict[str, Any] = {}
        if (
            source_onnx_path.is_file()
            and compiler_onnx_path.is_file()
            and source_onnx_sha256
            and compiler_onnx_sha256
            and _sha256_file(source_onnx_path) == source_onnx_sha256
            and _sha256_file(compiler_onnx_path) == compiler_onnx_sha256
        ):
            overlay_attestation_diagnostics: dict[str, Any] = {}
            observed_attestation, attestation_status = (
                _onnx_multiscale_raw_head_attestation(
                    source_onnx_path,
                    source_onnx_sha256=source_onnx_sha256,
                    compiler_onnx_sha256=compiler_onnx_sha256,
                    onnx_python=onnx_python,
                    diagnostics_out=overlay_attestation_diagnostics,
                )
            )
            if isinstance(observed_attestation, Mapping):
                recomputed_attestation = dict(observed_attestation)
            else:
                exception_type = str(
                    overlay_attestation_diagnostics.get("exception_type")
                    or "RawHeadAttestationError"
                )
                exception_detail = str(
                    overlay_attestation_diagnostics.get("exception_detail")
                    or attestation_status
                )
                raise ValueError(
                    "hailo_full_overlay_source_raw_head_attestation_invalid:"
                    f"{attestation_status}:{exception_type}:"
                    f"{exception_detail}"
                )
        endpoint_boundary_verified = bool(
            not compiler_end_nodes
            and not contract_end_nodes
            and contract_raw_endpoint_origin
            == "source_onnx_graph_outputs"
            and contract.get("source_onnx_multiscale_raw_head") is True
            and verified_receipt.get("raw_end_nodes_required") is True
            and source_raw_head_attestation
            and source_raw_head_attestation.get("schema")
            == "onnx-splitpoint/hailo-source-raw-head-attestation"
            and source_raw_head_attestation.get("schema_version") == 1
            and source_raw_head_attestation.get("raw_endpoint_origin")
            == "source_onnx_graph_outputs"
            and _strict_sha256_token(
                source_raw_head_attestation.get("source_onnx_sha256")
            ) == source_onnx_sha256
            and _strict_sha256_token(
                source_raw_head_attestation.get("compiler_onnx_sha256")
            ) == compiler_onnx_sha256
            and source_raw_head_attestation_sha256
            == _strict_sha256_token(
                source_raw_head_attestation.get("attestation_sha256")
            )
            and recomputed_attestation == source_raw_head_attestation
        )
    receipt_path_text = str(
        contract.get("hailo_build_receipt_path")
        or contract.get("build_receipt_path") or ""
    ).strip()
    receipt_path = Path(receipt_path_text).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = benchmark_set / receipt_path
    try:
        receipt_path = receipt_path.resolve(strict=True)
        expected_receipt_path = Path(
            str(verified_receipt.get("path") or "")
        ).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("hailo_full_overlay_receipt_unavailable") from exc
    receipt_binding_verified = bool(
        artifact == expected_hef
        and str(
            contract.get("recorded_artifact_sha256") or ""
        ).strip().lower()
        == str(verified_receipt.get("hef_sha256") or "").strip().lower()
        and int(contract.get("recorded_artifact_size_bytes") or 0)
        == int(expected_hef.stat().st_size)
        and receipt_path == expected_receipt_path
        and str(
            contract.get("hailo_build_receipt_file_sha256")
            or contract.get("build_receipt_file_sha256") or ""
        ).strip().lower()
        == str(verified_receipt.get("file_sha256") or "").strip().lower()
        and str(
            contract.get("hailo_build_receipt_identity_sha256")
            or contract.get("build_receipt_identity_sha256") or ""
        ).strip().lower()
        == str(verified_receipt.get("receipt_sha256") or "").strip().lower()
        and contract.get("artifact_binding_status") == "verified"
        and contract.get("contract_status") == "recorded"
    )
    if not receipt_binding_verified:
        raise ValueError("hailo_full_overlay_receipt_binding_mismatch")
    if is_detection and (
        not endpoint_boundary_verified
        or str(contract.get("stage") or "").strip().lower() != "raw_head"
        or str(contract.get("contract_family") or "").strip().lower()
        != "raw_head"
        or str(contract.get("contract_reconciliation_status") or "")
        != "verified_suite_artifact_raw_head"
        or contract.get("host_tail_required") is not True
        or contract.get("postprocessing_required") is not True
        or contract.get("requires_external_postprocess") is not True
    ):
        raise ValueError("hailo_full_overlay_receipt_binding_mismatch")
    if not is_detection and (
        contract.get("host_tail_required") is not False
        or contract.get("postprocessing_required") is not False
        or contract.get("requires_external_postprocess") not in (None, False)
    ):
        raise ValueError("hailo_full_overlay_receipt_binding_mismatch")

    # The overlay is outside the suite root.  Make every file role absolute so
    # the public loader cannot accidentally resolve it below the overlay.
    contract["artifact_path"] = str(artifact)
    contract["recorded_artifact_path"] = str(artifact)
    contract["hailo_build_receipt_path"] = str(receipt_path)
    contract["build_receipt_path"] = str(receipt_path)
    if source_graph_endpoint:
        contract["source_onnx_raw_head_attestation"] = dict(
            source_raw_head_attestation
        )
        contract["source_onnx_raw_head_attestation_sha256"] = (
            source_raw_head_attestation_sha256
        )
    contracts[index] = contract
    overlay_dir = dump_dir / "contract_overlay"
    if _path_contains_symlink(overlay_dir):
        raise ValueError("hailo_full_overlay_path_contains_symlink")
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / "output_contracts.json"
    if overlay_path.is_symlink():
        raise ValueError("hailo_full_overlay_output_path_is_symlink")
    if overlay_path.exists() and not overlay_path.is_file():
        raise ValueError("hailo_full_overlay_output_path_not_regular")
    overlay_payload = {
        **source_payload,
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": int(source_payload.get("schema_version") or 1),
        "model_id": str(model),
        "task": normalized_task,
        "contracts": contracts,
        "runtime_overlay": True,
        "source_benchmark_set": str(benchmark_set),
        "source_benchmark_set_sha256": _sha256_file(benchmark_path),
        "source_output_contracts_sha256": _sha256_file(contracts_path),
    }
    _write_json(overlay_path, overlay_payload)
    declaration = load_authoritative_output_contract(
        overlay_path,
        backend=hw_arch,
        model_id=str(model),
        variant="full",
        task=normalized_task,
    )
    declaration_stage = str(declaration.get("stage") or "").strip().lower()
    authoritative_reload_verified = bool(
        declaration.get("contract_resolution_status") == "attested"
        and declaration.get("authoritative_output_contract") is True
        and Path(
            str(declaration.get("recorded_artifact_path") or "")
        ).resolve(strict=True) == expected_hef
    )
    if is_detection:
        authoritative_reload_verified = bool(
            authoritative_reload_verified
            and declaration_stage == "raw_head"
            and list(declaration.get("full_end_node_names") or [])
            == compiler_end_nodes
            and str(declaration.get("raw_endpoint_origin") or "")
            == raw_endpoint_origin
            and declaration.get("source_onnx_multiscale_raw_head")
            is source_graph_endpoint
            and not (
                source_graph_endpoint
                and (
                    dict(
                        declaration.get(
                            "source_onnx_raw_head_attestation"
                        ) or {}
                    ) != source_raw_head_attestation
                    or _strict_sha256_token(
                        declaration.get(
                            "source_onnx_raw_head_attestation_sha256"
                        )
                    ) != source_raw_head_attestation_sha256
                )
            )
        )
    else:
        authoritative_reload_verified = bool(
            authoritative_reload_verified
            and declaration_stage in {
                "classification_logits", "classification_probabilities",
            }
            and declaration.get("host_tail_required") is False
            and declaration.get("postprocessing_required") is False
            and declaration.get("requires_external_postprocess")
            in (None, False)
        )
    if not authoritative_reload_verified:
        raise ValueError("hailo_full_overlay_authoritative_reload_failed")
    return overlay_path, {
        "path": str(overlay_path),
        "sha256": _sha256_file(overlay_path),
        "source_output_contracts": str(contracts_path),
        "source_output_contracts_sha256": _sha256_file(contracts_path),
        "backend": target_backend,
        "task": normalized_task,
        "stage": declaration_stage,
        "raw_endpoint_origin": (
            raw_endpoint_origin if is_detection else "not_applicable"
        ),
        "full_end_node_names": (
            compiler_end_nodes if is_detection else []
        ),
        "source_onnx_raw_head_attestation_sha256": (
            source_raw_head_attestation_sha256 if is_detection else ""
        ),
        "promotion": selected_promotions[0],
        "resolution_status": "attested",
    }


def _native_hailo_full(
    benchmark_set: Path, model: str, hw_arch: str, ns: argparse.Namespace,
) -> dict[str, Any]:
    arch = str(hw_arch or "").strip().lower()
    is_hailo8 = arch.startswith("hailo8")
    backend = "native_full_hailo8" if is_hailo8 else "native_full_hailo10h"
    producer_impl = "hailo8_vstreams_full" if is_hailo8 else "hailo10_infermodel_async_full"
    run_id = "hailo8" if is_hailo8 else "hailo10"
    hef = _find_hailo_full_hef(benchmark_set, arch)
    if hef is None:
        deferred, detail = _full_artifact_was_deferred(benchmark_set, arch)
        reason = "deferred_cold_build" if deferred else "missing_hailo_full_hef"
        return {
            "backend": backend, "producer_impl": producer_impl, "model": model,
            "case": "full", "execution_mode": "native_full_baseline",
            "duration_s": float(getattr(ns, "duration_s", 0.0) or 0.0),
            "frames": int(ns.frames), "ok": False,
            "status": "deferred" if deferred else "missing_artifact",
            "fps_makespan": None, "latency_mean_ms": None,
            "failure_reason": reason, "status_detail": detail,
            "error": "", "timed_out": False, "returncode": 0 if deferred else 2,
            "stdout_tail": "", "stderr_tail": "", "steps": [],
            "semantic_dump_status": "skipped_runtime_failed",
        }
    expected_hef_sha = str(getattr(ns, "expected_hef_sha256", "") or "").strip().lower()
    actual_hef_sha = _sha256_file(hef)
    if expected_hef_sha and expected_hef_sha != actual_hef_sha:
        return {
            "backend": backend, "producer_impl": producer_impl, "model": model,
            "case": "full", "execution_mode": "native_full_baseline",
            "duration_s": float(getattr(ns, "duration_s", 0.0) or 0.0),
            "frames": int(ns.frames), "ok": False, "status": "failed",
            "fps_makespan": None, "latency_mean_ms": None,
            "failure_reason": "full_hef_sha256_mismatch",
            "expected_hef_sha256": expected_hef_sha,
            "actual_hef_sha256": actual_hef_sha,
            "error": "", "timed_out": False, "returncode": 4, "steps": [],
            "semantic_dump_status": "skipped_runtime_failed",
        }

    attestation_python = str(
        getattr(ns, "engine_python_selected", sys.executable) or ""
    )
    hef_receipt_diagnostics: dict[str, Any] = {}
    hef_receipt, hef_receipt_status = _verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model=model,
        hw_arch=arch,
        onnx_python=attestation_python,
        diagnostics_out=hef_receipt_diagnostics,
    )
    if hef_receipt is None:
        receipt_exception_type = str(
            hef_receipt_diagnostics.get("exception_type") or ""
        )
        receipt_exception_detail = str(
            hef_receipt_diagnostics.get("exception_detail") or ""
        )
        receipt_status_detail = ": ".join(
            value for value in (
                receipt_exception_type, receipt_exception_detail,
            ) if value
        )
        return {
            "backend": backend, "producer_impl": producer_impl,
            "model": model, "case": "full",
            "execution_mode": "native_full_baseline",
            "duration_s": float(
                getattr(ns, "duration_s", 0.0) or 0.0
            ),
            "frames": int(ns.frames), "ok": False,
            "status": "failed", "fps_makespan": None,
            "latency_mean_ms": None,
            "failure_reason": hef_receipt_status,
            "hailo_hef_build_receipt_status": hef_receipt_status,
            "hailo_hef_build_receipt_diagnostics": (
                hef_receipt_diagnostics
            ),
            "hef_path": str(hef),
            "status_detail": receipt_status_detail,
            "error": receipt_status_detail,
            "timed_out": False, "returncode": 4,
            "steps": [],
            "semantic_dump_status": "skipped_runtime_failed",
        }

    runtime_python, runtime_probe = _select_hailo_python(arch)
    if not runtime_python:
        step = {
            "rc": 127, "returncode": 127, "timed_out": False,
            "stdout_tail": "", "stderr_tail": json.dumps(runtime_probe, ensure_ascii=False),
            "error": "no_hailo_runtime_python",
        }
        return {
            "backend": backend, "producer_impl": producer_impl, "model": model,
            "case": "full", "execution_mode": "native_full_baseline",
            "duration_s": float(getattr(ns, "duration_s", 0.0) or 0.0),
            "frames": int(ns.frames), "ok": False, "status": "failed",
            "fps_makespan": None, "latency_mean_ms": None,
            "runtime_python_probe": runtime_probe, "steps": [step],
            **_diagnostic_fields(step, failure_reason="no_hailo_runtime_python"),
        }

    info = _first_case(benchmark_set)
    case = info[0] if info else "full"
    image, image_source = _resolve_image(
        benchmark_set, model, case, getattr(ns, "image_map_data", {}) or {}
    )
    expected_image_sha = str(
        getattr(ns, "expected_input_image_sha256", "") or ""
    ).strip().lower()
    actual_image_sha = _sha256_file(image) if image and image.is_file() else ""
    if expected_image_sha and expected_image_sha != actual_image_sha:
        return {
            "backend": backend, "producer_impl": producer_impl, "model": model,
            "case": "full", "execution_mode": "native_full_baseline",
            "duration_s": float(getattr(ns, "duration_s", 0.0) or 0.0),
            "frames": int(ns.frames), "ok": False, "status": "failed",
            "fps_makespan": None, "latency_mean_ms": None,
            "failure_reason": "full_input_image_sha256_mismatch",
            "expected_input_image_sha256": expected_image_sha,
            "actual_input_image_sha256": actual_image_sha,
            "error": "", "timed_out": False, "returncode": 4, "steps": [],
            "semantic_dump_status": "skipped_runtime_failed",
        }
    task, _scale = _run_plan_meta(benchmark_set, run_id, model)
    preprocess_mode = str(getattr(ns, "preprocess_mode", "auto") or "auto")
    letterbox_pad_value = int(getattr(ns, "letterbox_pad_value", 114))
    receipt_preprocessing = dict(
        hef_receipt.get("preprocessing_contract") or {}
    )
    effective_preprocess_mode = preprocess_mode.strip().lower()
    if effective_preprocess_mode == "auto":
        effective_preprocess_mode = (
            "letterbox" if str(task).strip().lower() == "detection"
            else "resize"
        )
    receipt_runtime_match = bool(
        str(receipt_preprocessing.get("task") or "").strip().lower()
        == str(task).strip().lower()
        and str(
            receipt_preprocessing.get("preprocess_mode") or ""
        ).strip().lower() == effective_preprocess_mode
        and (
            effective_preprocess_mode != "letterbox"
            or int(receipt_preprocessing.get("pad_value") or -1)
            == letterbox_pad_value
        )
        and str(
            receipt_preprocessing.get("color_space") or ""
        ).strip().upper() == "RGB"
        and str(
            receipt_preprocessing.get("input_domain") or ""
        ).strip().lower() == "uint8_0_255"
    )
    if not receipt_runtime_match:
        return {
            "backend": backend, "producer_impl": producer_impl,
            "model": model, "case": "full",
            "execution_mode": "native_full_baseline",
            "duration_s": float(
                getattr(ns, "duration_s", 0.0) or 0.0
            ),
            "frames": int(ns.frames), "ok": False,
            "status": "failed", "fps_makespan": None,
            "latency_mean_ms": None,
            "failure_reason": (
                "hailo_hef_build_receipt_runtime_preprocessing_mismatch"
            ),
            "hailo_hef_build_receipt_status": (
                "hailo_hef_build_receipt_runtime_preprocessing_mismatch"
            ),
            "hailo_hef_build_receipt_path": str(
                hef_receipt.get("path") or ""
            ),
            "hef_path": str(hef),
            "error": "", "timed_out": False, "returncode": 4,
            "steps": [],
            "semantic_dump_status": "skipped_runtime_failed",
        }
    dump_dir = _native_full_dump_dir(benchmark_set, model, backend, ns)
    report_path = dump_dir / "native_full_hailo_report.json"
    contract_overlay_path: Path | None = None
    contract_overlay: dict[str, Any] = {}
    if str(task).strip().lower() in {"classification", "detection"}:
        try:
            contract_overlay_path, contract_overlay = (
                _prepare_verified_hailo_full_contract_overlay(
                    benchmark_set,
                    dump_dir=dump_dir,
                    model=model,
                    task=task,
                    hw_arch=arch,
                    hef=hef,
                    verified_receipt=hef_receipt,
                    onnx_python=attestation_python,
                )
            )
        except Exception as exc:
            return {
                "backend": backend,
                "producer_impl": producer_impl,
                "model": model,
                "case": "full",
                "execution_mode": "native_full_baseline",
                "duration_s": float(
                    getattr(ns, "duration_s", 0.0) or 0.0
                ),
                "frames": int(ns.frames),
                "ok": False,
                "status": "failed",
                "fps_makespan": None,
                "latency_mean_ms": None,
                "failure_reason": "hailo_full_output_contract_overlay_invalid",
                "status_detail": f"{type(exc).__name__}: {exc}",
                "hef_path": str(hef),
                "hailo_hef_build_receipt_status": hef_receipt_status,
                "error": f"{type(exc).__name__}: {exc}",
                "timed_out": False,
                "returncode": 4,
                "steps": [],
                "semantic_dump_status": "skipped_contract_failed",
            }
    command = [
        runtime_python,
        str(ROOT / "scripts" / "smoke_hailo10_full_from_benchmarkset.py"),
        "--benchmark-set", str(benchmark_set),
        "--hw-arch", arch,
        "--model", model,
        "--runtime-api", "vstreams" if is_hailo8 else "infer_model",
        "--frames", str(max(1, int(ns.frames))),
        "--warmup", str(max(0, int(ns.warmup))),
        "--inflight", "1",
        "--task", task,
        "--preprocess-mode", preprocess_mode,
        "--letterbox-pad-value", str(letterbox_pad_value),
        "--backend-label", backend,
        "--setup-id", str(ns.setup_id or ""),
        "--comparison-backend", str(ns.comparison_backend or ""),
        "--json-out", str(report_path),
    ]
    if contract_overlay_path is not None:
        command += [
            "--declared-output-contract-json",
            str(contract_overlay_path),
        ]
    if image is not None:
        command += ["--image", str(image)]
    if bool(getattr(ns, "dump_outputs", False)):
        command += ["--dump-outputs", "--dump-dir", str(dump_dir)]
    step = _run(
        command, timeout=ns.timeout, cwd=ROOT, env=_hailo_python_env(),
        label=f"native-full:{arch}:{model}",
    )
    report = _load_json(report_path) if report_path.is_file() else {}
    report = dict(report) if isinstance(report, Mapping) else {}
    throughput = report.get("throughput") if isinstance(report.get("throughput"), Mapping) else {}
    fps = _num(throughput.get("fps"))
    if not fps:
        output_text = f"{step.get('stdout_tail', '')}\n{step.get('stderr_tail', '')}"
        match = re.search(r"throughput\s+fps=([0-9.]+)", output_text, flags=re.IGNORECASE)
        fps = float(match.group(1)) if match else None
    copy_outputs_verified = bool(
        report.get("copy_outputs") is True
        and report.get("claim_copy_outputs_verified") is True
    )
    requested_frames = int(_num(throughput.get("requested_frames")) or int(ns.frames))
    completed_frames = int(_num(throughput.get("completed_frames")) or 0)
    completion_count_verified = bool(
        completed_frames == requested_frames
        and str(throughput.get("completed_work_units_status") or "") == "exact_runtime_counter"
    )
    runtime_success = bool(
        step.get("rc") == 0
        and not bool(step.get("timed_out"))
        and fps is not None and fps > 0
        and completion_count_verified
    )
    runtime_endpoint_family = str(
        report.get("runtime_endpoint_contract_family") or ""
    ).strip().lower()
    host_postprocess_required = bool(
        report.get("host_postprocess_frozen") is True
        or runtime_endpoint_family == "raw_head"
    )
    direct_normalization_required = bool(
        task == "detection" and runtime_endpoint_family == "decoded_nms"
    )
    try:
        frozen_contract = verify_frozen_postprocess_contract(
            report.get("frozen_host_postprocess_contract")
        ) if host_postprocess_required else {}
    except (FrozenPostprocessError, TypeError, ValueError):
        frozen_contract = {}
    try:
        direct_normalization_contract = (
            verify_frozen_decoded_nms_normalization_contract(
                report.get(
                    "frozen_decoded_nms_normalization_contract"
                )
            )
            if direct_normalization_required
            and report.get("normalization_frozen") is True
            else {}
        )
    except (FrozenPostprocessError, TypeError, ValueError):
        direct_normalization_contract = {}
    postprocess_required = bool(
        host_postprocess_required or direct_normalization_required
    )
    postprocess_completed_frames = int(report.get("postprocess_completed_frames") or 0)
    completed_attestation = (
        dict(report.get("completed_task_endpoint_attestation") or {})
        if isinstance(
            report.get("completed_task_endpoint_attestation"), Mapping,
        )
        else {}
    )
    completed_attestation_verified = not postprocess_required
    completed_result_artifact_verified = not postprocess_required
    completed_result_artifact_status = (
        "not_applicable" if not postprocess_required else "missing"
    )
    sealed_postprocess_result: dict[str, Any] = {}
    if host_postprocess_required:
        sealed_postprocess_result = dict(
            report.get("frozen_host_postprocess_result") or {}
        )
    elif direct_normalization_required:
        sealed_postprocess_result = dict(
            report.get(
                "frozen_decoded_nms_normalization_result"
            ) or {}
        )
    if (
        postprocess_required
        and completed_attestation
        and (
            (host_postprocess_required and frozen_contract)
            or (
                direct_normalization_required
                and direct_normalization_contract
            )
        )
    ):
        try:
            if host_postprocess_required:
                completed_endpoint = completed_attestation.get(
                    "completed_endpoint_contract"
                )
                completed_endpoint = (
                    dict(completed_endpoint)
                    if isinstance(completed_endpoint, Mapping) else {}
                )
                expected_completed_attestation = (
                    build_completed_detection_endpoint_attestation(
                        frozen_contract,
                        sealed_postprocess_result,
                        completed_frames=completed_frames,
                        postprocess_completed_frames=(
                            postprocess_completed_frames
                        ),
                        source_endpoint_contract_hash=str(
                            completed_endpoint.get(
                                "source_endpoint_contract_hash"
                            ) or ""
                        ),
                    )
                )
            else:
                expected_completed_attestation = (
                    build_normalized_detection_endpoint_attestation(
                        direct_normalization_contract,
                        sealed_postprocess_result,
                        completed_frames=completed_frames,
                        postprocess_completed_frames=(
                            postprocess_completed_frames
                        ),
                    )
                )
            completed_attestation_verified = bool(
                completed_attestation == expected_completed_attestation
                and report.get("completed_task_endpoint_attested") is True
                and str(
                    report.get(
                        "completed_task_endpoint_attestation_status"
                    ) or ""
                ).strip().lower() == "passed"
            )
            (
                completed_result_artifact_verified,
                completed_result_artifact_status,
            ) = _completed_result_artifact_persistence_status(
                report,
                sealed_result=sealed_postprocess_result,
                allowed_root=dump_dir,
                expected_path=report_path.with_name(
                    f"{report_path.stem}.completed_task_result_artifact.json"
                ),
            )
        except (FrozenPostprocessError, TypeError, ValueError):
            completed_attestation_verified = False
            completed_result_artifact_verified = False
            completed_result_artifact_status = (
                "completed_task_attestation_invalid"
            )
    postprocess_completion_verified = bool(
        not postprocess_required
        or (
            (
                frozen_contract
                if host_postprocess_required
                else direct_normalization_contract
            )
            and report.get("postprocess_included") is True
            and postprocess_completed_frames == completed_frames
            and completed_attestation_verified
            and completed_result_artifact_verified
        )
    )
    ok = bool(
        step.get("rc") == 0 and bool(report.get("ok", True))
        and fps is not None and fps > 0
        and copy_outputs_verified and completion_count_verified
        and postprocess_completion_verified
    )
    if step.get("timed_out"):
        reason = "hailo_full_timeout"
    elif step.get("rc") != 0:
        reason = "hailo_full_runner_failed"
    elif not copy_outputs_verified:
        reason = "hailo_full_copy_outputs_not_verified"
    elif not completion_count_verified:
        reason = "hailo_full_completed_work_unit_verification_failed"
    elif not postprocess_completion_verified:
        reason = "hailo_full_completed_v2_postprocess_verification_failed"
    elif not fps:
        reason = "hailo_full_metrics_missing"
    else:
        reason = ""
    output_manifest = str(report.get("output_manifest") or "")
    input_manifest = str(report.get("input_manifest") or "")
    manifest_payload = _load_json(Path(output_manifest)) if output_manifest and Path(output_manifest).is_file() else {}
    manifest_payload = dict(manifest_payload) if isinstance(manifest_payload, Mapping) else {}
    direct_source_endpoint_binding_verified = (
        not direct_normalization_required
    )
    direct_source_endpoint_binding_status = (
        "not_applicable"
        if not direct_normalization_required else "missing"
    )
    if direct_normalization_required and direct_normalization_contract:
        source_attestation = manifest_payload.get(
            "output_endpoint_attestation"
        )
        source_attestation = (
            dict(source_attestation)
            if isinstance(source_attestation, Mapping) else {}
        )
        direct_source_hash = str(
            direct_normalization_contract.get(
                "source_endpoint_contract_hash"
            ) or ""
        ).strip().lower()
        direct_source_id = str(
            direct_normalization_contract.get(
                "source_output_endpoint_id"
            ) or ""
        ).strip()
        manifest_source_id = str(
            manifest_payload.get("output_endpoint_id") or ""
        ).strip()
        direct_source_endpoint_binding_verified = bool(
            source_attestation
            and _canonical_json_sha256(source_attestation)
            == str(
                direct_normalization_contract.get(
                    "source_output_endpoint_attestation_sha256"
                ) or ""
            ).strip().lower()
            and str(
                manifest_payload.get("endpoint_contract_hash") or ""
            ).strip().lower() == direct_source_hash
            and (
                not manifest_source_id
                or manifest_source_id == direct_source_id
            )
            and dict(
                manifest_payload.get("tensor_signature") or {}
            ) == dict(
                direct_normalization_contract.get(
                    "source_output_tensor_signature"
                ) or {}
            )
        )
        direct_source_endpoint_binding_status = (
            "verified_exact"
            if direct_source_endpoint_binding_verified
            else "direct_source_endpoint_manifest_mismatch"
        )
    if ok and not direct_source_endpoint_binding_verified:
        ok = False
        reason = "hailo_full_direct_source_endpoint_binding_failed"
    dump_requested = bool(getattr(ns, "dump_outputs", False))
    identity_ok, identity_reason = _native_full_manifest_identity_status(
        Path(output_manifest),
        model=model,
        backend=backend,
        setup_id=str(ns.setup_id or ""),
        comparison_backend=str(ns.comparison_backend or ""),
    ) if output_manifest else (False, "native_full_manifest_missing")
    semantic_ok = bool(
        output_manifest and Path(output_manifest).is_file() and identity_ok
    ) if dump_requested else False
    explicit_runtime_precision = _explicit_runtime_precision(report, manifest_payload)
    return {
        "backend": backend,
        "producer_impl": producer_impl,
        # Seal the logical Full invocation identity into the producer row.
        # Backend/setup labels alone are insufficient: a retained timing row
        # must not be rebound to another generated-suite run profile.
        "run_id": run_id,
        "model": model,
        "case": "full",
        "execution_mode": "native_full_baseline",
        "execution_precision": explicit_runtime_precision,
        "full_runtime_precision": explicit_runtime_precision,
        "runtime_precision_source": (
            "hailo_runtime_report" if explicit_runtime_precision else "unavailable"
        ),
        "engine_precision": "unavailable",
        "duration_s": float(getattr(ns, "duration_s", 0.0) or 0.0),
        "frames": int(ns.frames),
        "warmup": int(ns.warmup),
        "warmup_policy": "exact_untimed_frames_before_measured_loop",
        "runtime_success": runtime_success,
        "requested_frames": requested_frames,
        "completed_frames": completed_frames if completion_count_verified else None,
        "completed_work_units": completed_frames if completion_count_verified else None,
        "completed_work_units_source": str(throughput.get("completed_work_units_source") or ""),
        "completed_work_units_status": str(throughput.get("completed_work_units_status") or ""),
        "e2e_scope": "full_task_pipeline" if postprocess_required else "accelerator_output_endpoint",
        "comparison_endpoint_stratum": (
            "decoded_nms" if postprocess_required
            else str(manifest_payload.get("contract_family") or "")
        ),
        "measurement_concurrency": 1,
        "configured_inflight": max(1, int(ns.inflight)),
        "host_postprocess_frozen": bool(
            host_postprocess_required and frozen_contract
        ),
        "normalization_frozen": bool(
            direct_normalization_required
            and direct_normalization_contract
        ),
        "postprocess_included": bool(
            report.get("postprocess_included") is True
        ),
        "postprocess_location": str(
            report.get("postprocess_location") or ""
        ),
        "postprocess_completed_frames": postprocess_completed_frames,
        "postprocess_completion_verified": postprocess_completion_verified,
        "frozen_host_postprocess_contract": frozen_contract,
        "frozen_host_postprocess_contract_sha256": str(
            frozen_contract.get("contract_sha256") or ""
        ),
        "frozen_host_postprocess_result": dict(
            report.get("frozen_host_postprocess_result") or {}
        ),
        "frozen_decoded_nms_normalization_contract": dict(
            direct_normalization_contract
        ),
        "frozen_decoded_nms_normalization_contract_sha256": str(
            direct_normalization_contract.get("contract_sha256") or ""
        ),
        "frozen_decoded_nms_normalization_result": dict(
            report.get(
                "frozen_decoded_nms_normalization_result"
            ) or {}
        ),
        "source_endpoint_contract_hash": str(
            direct_normalization_contract.get(
                "source_endpoint_contract_hash"
            ) or ""
        ),
        "source_output_endpoint_id": str(
            direct_normalization_contract.get(
                "source_output_endpoint_id"
            ) or ""
        ),
        "source_output_tensor_signature": dict(
            direct_normalization_contract.get(
                "source_output_tensor_signature"
            ) or {}
        ),
        "source_output_endpoint_attestation_sha256": str(
            direct_normalization_contract.get(
                "source_output_endpoint_attestation_sha256"
            ) or ""
        ),
        "letterbox_geometry_contract_sha256": str(
            direct_normalization_contract.get(
                "letterbox_geometry_contract_sha256"
            ) or ""
        ),
        "direct_source_endpoint_binding_verified": (
            direct_source_endpoint_binding_verified
        ),
        "direct_source_endpoint_binding_status": (
            direct_source_endpoint_binding_status
        ),
        "completed_task_stage": str(
            completed_attestation.get("stage") or ""
        ),
        "completed_task_contract_family": (
            "decoded_nms"
            if completed_attestation.get("attested") is True else ""
        ),
        "completed_task_endpoint_contract_hash": str(
            completed_attestation.get("endpoint_contract_hash") or ""
        ),
        "completed_task_output_endpoint_id": str(
            completed_attestation.get("output_endpoint_id") or ""
        ),
        "completed_task_comparison_endpoint_contract": dict(
            completed_attestation.get(
                "completed_task_comparison_endpoint_contract"
            ) or {}
        ),
        "completed_task_comparison_endpoint_contract_hash": str(
            completed_attestation.get(
                "completed_task_comparison_endpoint_contract_hash"
            ) or ""
        ),
        "completed_task_comparison_output_endpoint_id": str(
            completed_attestation.get(
                "completed_task_comparison_output_endpoint_id"
            ) or ""
        ),
        "completed_task_completion_mode": str(
            completed_attestation.get(
                "completed_task_completion_mode"
            ) or ""
        ),
        "completed_task_endpoint_attested": (
            completed_attestation_verified
        ),
        "completed_task_endpoint_attestation_status": (
            "passed" if completed_attestation_verified else ""
        ),
        "completed_task_endpoint_attestation": completed_attestation,
        "completed_task_result_artifact_saved": bool(
            report.get("completed_task_result_artifact_saved") is True
        ),
        "completed_task_result_artifact": dict(
            report.get("completed_task_result_artifact") or {}
        ),
        "completed_task_result_artifact_sha256": str(
            report.get("completed_task_result_artifact_sha256") or ""
        ),
        "completed_task_result_artifact_path": str(
            report.get("completed_task_result_artifact_path") or ""
        ),
        "completed_task_result_artifact_file_sha256": str(
            report.get(
                "completed_task_result_artifact_file_sha256"
            ) or ""
        ),
        "completed_task_result_artifact_verification_status": (
            completed_result_artifact_status
        ),
        "original_image_wh": list(report.get("original_image_wh") or []),
        "copy_outputs": bool(report.get("copy_outputs")),
        "claim_copy_outputs_verified": copy_outputs_verified,
        "ok": ok,
        "status": "ok" if ok else "failed",
        "fps_makespan": fps,
        "measured_duration_s": _num(throughput.get("elapsed_s")),
        "measurement_endpoint": "completed_task",
        "measurement_boundary": "first_task_start_to_last_task_completion",
        "request_latency": throughput.get("request_latency") or report.get("request_latency"),
        # Async/inflight throughput cannot be inverted into request latency.
        # Preserve the reciprocal only under its correct service-interval name.
        "latency_mean_ms": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "latency_semantics": "not_measured_async_or_streaming_throughput",
        "completion_interval_mean_ms": (1000.0 / fps) if fps else None,
        "completion_interval_semantics": "reciprocal_steady_state_throughput",
        "report": str(report_path),
        "hef_path": str(hef),
        "runtime_python": runtime_python,
        "runtime_python_probe": runtime_probe,
        "input_image": str(report.get("input_image") or (image or "")),
        "input_case": case,
        "input_image_source": image_source if image else "image_unavailable",
        "input_image_sha256": str(report.get("input_image_sha256") or (_sha256_file(image) if image else "")),
        "output_dump_manifest": output_manifest,
        "native_output_manifest": output_manifest,
        "input_manifest": input_manifest,
        "runtime_input_dtype": str(report.get("runtime_input_dtype") or ""),
        "runtime_input_shape": report.get("runtime_input_shape") or [],
        "runtime_input_layout": str(report.get("runtime_input_layout") or ""),
        "runtime_preprocess_mode": str(report.get("runtime_preprocess_mode") or ""),
        "runtime_normalization": str(report.get("runtime_normalization") or ""),
        "runtime_color_space": str(report.get("runtime_color_space") or ""),
        "runtime_preprocessing_identity": dict(
            report.get("runtime_preprocessing_identity") or {}
        ) if isinstance(
            report.get("runtime_preprocessing_identity"), Mapping,
        ) else {},
        "runtime_preprocessing_sha256": str(
            report.get("runtime_preprocessing_sha256") or ""
        ),
        "runtime_numeric_input_identity": dict(
            report.get("runtime_numeric_input_identity") or {}
        ) if isinstance(
            report.get("runtime_numeric_input_identity"), Mapping,
        ) else {},
        "runtime_numeric_input_sha256": str(
            report.get("runtime_numeric_input_sha256") or ""
        ),
        "hailo_hef_build_receipt_status": hef_receipt_status,
        "hailo_hef_build_receipt_path": str(
            hef_receipt.get("path") or ""
        ),
        "hailo_hef_build_receipt_file_sha256": str(
            hef_receipt.get("file_sha256") or ""
        ),
        "hailo_hef_build_receipt_sha256": str(
            hef_receipt.get("receipt_sha256") or ""
        ),
        "hailo_hef_build_receipt": dict(
            hef_receipt.get("receipt") or {}
        ),
        "hailo_hef_source_onnx_path": str(
            hef_receipt.get("source_onnx_path") or ""
        ),
        "source_onnx_sha256": str(
            hef_receipt.get("source_onnx_sha256") or ""
        ),
        "hailo_hef_compiler_onnx_sha256": str(
            hef_receipt.get("compiler_onnx_sha256") or ""
        ),
        "hailo_hef_preprocessing_contract": dict(
            hef_receipt.get("preprocessing_contract") or {}
        ),
        "hailo_hef_preprocessing_contract_sha256": str(
            hef_receipt.get("preprocessing_contract_sha256") or ""
        ),
        "hailo_raw_endpoint_origin": str(
            hef_receipt.get("raw_endpoint_origin") or ""
        ),
        "hailo_source_onnx_raw_head_attestation": dict(
            hef_receipt.get("source_onnx_raw_head_attestation") or {}
        ),
        "hailo_source_onnx_raw_head_attestation_sha256": str(
            hef_receipt.get(
                "source_onnx_raw_head_attestation_sha256"
            ) or ""
        ),
        "hailo_output_contract_overlay": contract_overlay,
        "hailo_output_contract_overlay_path": str(
            contract_overlay_path or ""
        ),
        "hailo_output_contract_overlay_sha256": str(
            contract_overlay.get("sha256") or ""
        ),
        "semantic_dump_status": "ok" if semantic_ok else "failed" if dump_requested and ok else "disabled",
        "semantic_dump_failure_reason": "" if semantic_ok or not dump_requested else identity_reason,
        "task": str(manifest_payload.get("task") or task),
        "output_format": str(manifest_payload.get("output_format") or ""),
        "contract_family": str(manifest_payload.get("contract_family") or ""),
        "stage": str(manifest_payload.get("stage") or ""),
        "contract_source": str(manifest_payload.get("contract_source") or "hailo_native_full_runtime"),
        "endpoint_contract_complete": manifest_payload.get("endpoint_contract_complete") is True,
        "endpoint_contract_hash": str(manifest_payload.get("endpoint_contract_hash") or ""),
        "output_endpoint_id": str(
            manifest_payload.get("output_endpoint_id")
            or direct_normalization_contract.get(
                "source_output_endpoint_id"
            )
            or ""
        ),
        "tensor_signature": manifest_payload.get("tensor_signature")
        if isinstance(manifest_payload.get("tensor_signature"), Mapping) else {},
        "output_endpoint_attestation": manifest_payload.get("output_endpoint_attestation")
        if isinstance(manifest_payload.get("output_endpoint_attestation"), Mapping) else {},
        "steps": [step],
        **_diagnostic_fields(step, failure_reason=reason),
        **({key: report.get(key) for key in (
            "task_complete", "completed_task_stage", "classification_topk",
            "e2e_scope", "comparison_endpoint_stratum", "postprocess_location",
            "postprocess_included", "postprocess_completed_frames", "postprocess_completion_verified",
        )} if report.get("completed_task_stage") == "classification_top1_top5" else {}),
    }

def _generic_full_via_suite(
    benchmark_set: Path, model: str, backend: str, run_id: str,
    ns: argparse.Namespace, *,
    prepared_input_manifest: Path | None = None,
) -> dict[str, Any]:
    throughput_frames = (
        max(1, int(ns.frames))
        if backend == "native_full_deepx"
        else max(16, min(ns.frames, 256))
    )
    suite_python, child_env, _runtime_sites = _suite_python_env(ns, backend)
    if backend == "native_full_deepx" and bool(getattr(ns, "diagnostic_deepx_input_probes", False)):
        child_env["ONNX_SPLITPOINT_DEEPX_ALLOW_DIAGNOSTIC_FALLBACK"] = "1"
    expected_prepared_image: Path | None = None
    expected_prepared_image_sha256 = ""
    expected_runtime_input: dict[str, Any] = {}
    expected_runtime_input_status = "not_applicable"
    if backend == "native_full_deepx":
        if (
            prepared_input_manifest is None
            or not prepared_input_manifest.is_file()
        ):
            return {
                "backend": backend,
                "model": model,
                "case": "full",
                "execution_mode": "native_full_baseline",
                "ok": False,
                "status": "prepared_input_manifest_missing",
                "failure_reason": "deepx_prepared_input_manifest_missing",
                **_deepx_original_full_failure(
                    benchmark_set, model, run_id,
                    setup_id=str(getattr(ns, "setup_id", "") or ""),
                ),
            }
        expected_runtime_input_raw, expected_runtime_input_status = (
            _validated_runtime_input_manifest(
                prepared_input_manifest,
                allowed_root=prepared_input_manifest.parent,
            )
        )
        expected_runtime_input = (
            dict(expected_runtime_input_raw)
            if isinstance(expected_runtime_input_raw, Mapping) else {}
        )
        image_text = str(
            expected_runtime_input.get("input_image") or ""
        ).strip()
        image_sha = str(
            expected_runtime_input.get("input_image_sha256") or ""
        ).strip().lower()
        if image_text:
            image_candidate = Path(image_text).expanduser()
            if not image_candidate.is_absolute():
                image_candidate = prepared_input_manifest.parent / image_candidate
            if (
                not _path_contains_symlink(image_candidate)
                and image_candidate.is_file()
            ):
                expected_prepared_image = image_candidate.resolve(strict=True)
        if (
            expected_prepared_image is None
            or re.fullmatch(r"[0-9a-f]{64}", image_sha) is None
            or _sha256_file(expected_prepared_image) != image_sha
        ):
            return {
                "backend": backend,
                "model": model,
                "case": "full",
                "execution_mode": "native_full_baseline",
                "ok": False,
                "status": "prepared_input_source_image_unavailable",
                "failure_reason": "deepx_prepared_input_source_image_missing_or_mismatched",
                "prepared_input_binding_status": expected_runtime_input_status,
            }
        expected_prepared_image_sha256 = image_sha
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_benchmark_suite_from_set.py"),
        "--benchmark-set",
        str(benchmark_set),
        "--python",
        suite_python,
        "--no-refresh",
        "--",
        "--run-id",
        run_id,
        "--warmup",
        str(max(0, int(ns.warmup))),
        "--runs",
        str(max(3, min(50, ns.frames // 20 or 3))),
        "--throughput-frames",
        str(throughput_frames),
        "--throughput-warmup-frames",
        # Use the same run-mode warm-up effort as every other Native Full
        # backend.  The generated DeepX prepared-feed loop drains these exact
        # work units before opening its outer makespan interval.
        str(max(0, int(ns.warmup))),
        "--energy-measurement-only",
    ]
    if expected_prepared_image is not None:
        cmd.extend(["--prepared-feed-image", str(expected_prepared_image)])
    if backend == "native_full_deepx":
        cmd.extend([
            "--prepared-input-manifest",
            str(prepared_input_manifest.resolve()),
            "--quality-evidence-model-id",
            str(model),
            "--quality-evidence-setup-id",
            str(getattr(ns, "setup_id", "") or ""),
        ])
    before_results: dict[str, tuple[int, int]] = {}
    for suffix in ("json", "csv"):
        for path in benchmark_set.rglob(f"benchmark_results_{run_id}_*.{suffix}"):
            try:
                stat = path.stat()
                before_results[str(path.resolve())] = (int(stat.st_mtime_ns), int(stat.st_size))
            except OSError:
                continue
    step = _run(cmd, timeout=ns.timeout, env=child_env, label=f"native-full:{backend}:{model}")
    candidates: list[Path] = []
    # The generated suite writes the current result in the BenchmarkSet root.
    # Prefer that exact invocation scope; recursive copies are only a legacy
    # fallback and must not shadow a current root-level result.
    for suffix in ("json", "csv"):
        candidates.extend(sorted(benchmark_set.glob(f"benchmark_results_{run_id}_*.{suffix}")))
    if not candidates:
        for suffix in ("json", "csv"):
            candidates.extend(sorted(benchmark_set.rglob(f"benchmark_results_{run_id}_*.{suffix}")))
    # Deduplicate while preserving order.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    changed: list[Path] = []
    for path in unique:
        try:
            stat = path.stat()
            state = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            continue
        if before_results.get(str(path.resolve())) != state:
            changed.append(path)
    selected_results = changed or unique
    selected_results = sorted(
        selected_results,
        key=lambda path: path.stat().st_mtime_ns if path.is_file() else 0,
        reverse=True,
    )
    metrics = _metrics_from_rows(selected_results)
    metric_row = metrics.get("result_row") if isinstance(metrics.get("result_row"), Mapping) else {}
    prepared_projection, prepared_projection_conflicts = (
        _deepx_prepared_feed_projection(metric_row)
        if backend == "native_full_deepx" else ({}, [])
    )
    metric_source_consistent = bool(
        metrics.get("source_consistent") is not False
        and not prepared_projection_conflicts
    )
    prepared_feed = dict(prepared_projection.get("nested_payload") or {})
    prepared_contract = prepared_feed.get("input_contract")
    prepared_contract = dict(prepared_contract) if isinstance(prepared_contract, Mapping) else {}
    prepared_input = prepared_contract.get("input")
    prepared_input = dict(prepared_input) if isinstance(prepared_input, Mapping) else {}
    # ``expected_runtime_input`` was validated before the child command so the
    # timed runner and every post-run gate consume one identical sealed image
    # and tensor identity.  Non-DeepX rows retain the empty/not-applicable
    # values initialized above.
    semantic_artifact_root = _native_full_dump_dir(
        benchmark_set, model, backend, ns,
    )
    prepared_tensor_path = Path(
        str(prepared_feed.get("prepared_input_file") or "")
    )
    expected_semantic_tensor_path = semantic_artifact_root / "runtime_input.bin"
    prepared_input_binding_ok = bool(
        backend != "native_full_deepx"
        or (
            expected_runtime_input
            and prepared_feed.get("prepared_input_binding_verified") is True
            and str(prepared_feed.get("prepared_input_source") or "")
            == "sealed_semantic_dump_runtime_tensor"
            and _confined_regular_file(
                prepared_tensor_path,
                allowed_root=semantic_artifact_root,
                expected_path=expected_semantic_tensor_path,
            )
            and str(prepared_feed.get("prepared_input_sha256") or "").strip().lower()
            == str(expected_runtime_input.get("runtime_input_sha256") or "")
            and str(prepared_feed.get("prepared_input_file_sha256") or "").strip().lower()
            == str(expected_runtime_input.get("runtime_input_sha256") or "")
            and _sha256_file(prepared_tensor_path)
            == str(expected_runtime_input.get("runtime_input_sha256") or "")
            and int(prepared_feed.get("prepared_input_bytes") or 0)
            == int(expected_runtime_input.get("runtime_input_bytes") or 0)
            and str(prepared_feed.get("prepared_input_name") or "")
            == str(expected_runtime_input.get("runtime_input_name") or "")
            and list(prepared_feed.get("prepared_input_shape") or [])
            == list(expected_runtime_input.get("runtime_input_shape") or [])
            and str(prepared_feed.get("prepared_input_dtype") or "").strip().lower()
            == str(expected_runtime_input.get("runtime_input_dtype") or "").strip().lower()
            and str(prepared_feed.get("prepared_input_layout") or "").strip().upper()
            == str(expected_runtime_input.get("runtime_input_layout") or "").strip().upper()
            and dict(prepared_feed.get("runtime_preprocessing_identity") or {})
            == dict(expected_runtime_input.get("runtime_preprocessing_identity") or {})
            and str(prepared_feed.get("runtime_preprocessing_sha256") or "").strip().lower()
            == str(expected_runtime_input.get("runtime_preprocessing_sha256") or "")
            and dict(prepared_feed.get("runtime_numeric_input_identity") or {})
            == dict(expected_runtime_input.get("runtime_numeric_input_identity") or {})
            and str(prepared_feed.get("runtime_numeric_input_sha256") or "").strip().lower()
            == str(expected_runtime_input.get("runtime_numeric_input_sha256") or "")
            and str(
                prepared_feed.get("prepared_input_source_image_id") or ""
            ) == str(
                expected_prepared_image.name
                if expected_prepared_image is not None else ""
            )
            and str(
                prepared_feed.get(
                    "prepared_input_source_image_sha256"
                ) or ""
            ).strip().lower()
            == str(
                expected_runtime_input.get("input_image_sha256") or ""
            ).strip().lower()
            == expected_prepared_image_sha256
        )
    )
    prepared_image = str(
        prepared_feed.get("image") or metric_row.get("prepared_feed_image") or ""
    ).strip()
    prepared_image_path = Path(prepared_image).expanduser() if prepared_image else None
    prepared_image_sha256 = (
        _sha256_file(prepared_image_path)
        if prepared_image_path is not None and prepared_image_path.is_file()
        else ""
    )
    prepared_feed_image_binding_ok = bool(
        backend != "native_full_deepx"
        or (
            bool(expected_prepared_image_sha256)
            and prepared_image_sha256 == expected_prepared_image_sha256
            and str(
                expected_runtime_input.get("input_image_sha256") or ""
            ).strip().lower() == expected_prepared_image_sha256
            and str(
                prepared_feed.get(
                    "prepared_input_source_image_sha256"
                ) or ""
            ).strip().lower() == expected_prepared_image_sha256
            and str(
                prepared_feed.get("prepared_input_source_image_id") or ""
            ) == str(
                expected_prepared_image.name
                if expected_prepared_image is not None else ""
            )
        )
    )
    fps = _num(
        prepared_projection.get("fps_makespan")
        if backend == "native_full_deepx"
        else metrics.get("fps_makespan")
    )
    latency = _num(metrics.get("latency_mean_ms"))
    outer_makespan_verified = False
    measured_makespan_s = _num(
        prepared_projection.get("makespan_s")
        if backend == "native_full_deepx"
        else metric_row.get("measured_makespan_s")
    )
    prepared_completed_frames = int(
        _num(prepared_projection.get("completed_frames")) or 0
    )
    if backend == "native_full_deepx":
        # Do not use reciprocal mean request latency as throughput.  The
        # generated DeepX runner records the real outer measured interval.
        fps = None
        prepared_makespan_fps = _num(
            prepared_projection.get("fps_makespan")
        )
        if (
            prepared_makespan_fps is not None and prepared_makespan_fps > 0
            and measured_makespan_s is not None and measured_makespan_s > 0
            and prepared_completed_frames > 0
        ):
            observed_fps = float(prepared_completed_frames) / float(measured_makespan_s)
            relative_error = abs(observed_fps - float(prepared_makespan_fps)) / max(observed_fps, 1e-12)
            if relative_error <= 1e-6:
                fps = observed_fps
                outer_makespan_verified = True
        latency = _num(prepared_feed.get("mean_ms"))
        if latency is None:
            latency = _num(metrics.get("latency_mean_ms"))
    latency_p50 = _num(prepared_feed.get("p50_ms") or metric_row.get("latency_p50_ms"))
    latency_p95 = _num(prepared_feed.get("p95_ms") or metric_row.get("latency_p95_ms"))
    result_run_id = str(metric_row.get("run_id") or "").strip()
    result_runtime_ok = _truth(metric_row.get("runtime_ok"))
    result_identity_ok = bool(
        result_run_id == str(run_id)
        if backend == "native_full_deepx"
        else not result_run_id or result_run_id == str(run_id)
    )
    prepared_feed_contract_version = str(
        prepared_projection.get("prepared_feed_contract_version")
        or metric_row.get("prepared_feed_contract_version") or ""
    ).strip()
    prepared_feed_contract_binding_ok = bool(
        backend != "native_full_deepx"
        or (
            prepared_feed_contract_version == DEEPX_PREPARED_FEED_CONTRACT_VERSION
            and prepared_feed_image_binding_ok
            and prepared_input_binding_ok
        )
    )
    # Never turn a placeholder/failed DeepX row with a diagnostic FPS field
    # into successful runtime evidence.  Current generated rows attest
    # runtime_ok explicitly; the missing case remains accepted only for old
    # result schemas and can never satisfy the exact DeepX work-unit gate.
    if (
        result_runtime_ok is False
        or not result_identity_ok
        or not prepared_feed_contract_binding_ok
        or (backend == "native_full_deepx" and not metric_source_consistent)
    ):
        fps = None
        latency = None
    ok = bool(
        step.get("rc") == 0 and fps is not None and fps > 0
        and prepared_feed_contract_binding_ok
        and (backend != "native_full_deepx" or outer_makespan_verified)
    )
    completed_frames_raw = (
        prepared_projection.get("completed_frames")
        if backend == "native_full_deepx"
        else metric_row.get("completed_frames")
    )
    try:
        completed_frames = int(float(completed_frames_raw))
    except Exception:
        completed_frames = 0
    completed_work_units = int(
        _num(prepared_projection.get("completed_work_units")) or 0
    )
    performance_benchmark_source = str(
        prepared_projection.get("performance_benchmark_source")
        or metric_row.get("performance_benchmark_source") or ""
    )
    exact_deepx_counter = bool(
        backend == "native_full_deepx"
        and ok
        and performance_benchmark_source == "dx_engine_prepared_feed"
        and prepared_feed_contract_binding_ok
        and completed_frames == int(throughput_frames)
        and completed_work_units == int(throughput_frames)
        and str(
            prepared_projection.get("completed_work_units_status") or ""
        ) == "exact_runtime_counter"
    )
    try:
        deepx_frozen_contract = verify_frozen_postprocess_contract(
            prepared_feed.get("frozen_host_postprocess_contract")
        ) if prepared_feed.get("host_postprocess_frozen") is True else {}
    except (FrozenPostprocessError, TypeError, ValueError):
        deepx_frozen_contract = {}
    try:
        deepx_direct_contract = (
            verify_frozen_decoded_nms_normalization_contract(
                prepared_feed.get(
                    "frozen_decoded_nms_normalization_contract"
                )
            )
            if prepared_feed.get("normalization_frozen") is True
            else {}
        )
    except (FrozenPostprocessError, TypeError, ValueError):
        deepx_direct_contract = {}
    deepx_raw_postprocess_required = bool(
        prepared_projection.get("host_postprocess_frozen") is True
        or str(prepared_feed.get("runtime_endpoint_contract_family") or "").strip().lower()
        in {"raw_head", "decoded_pre_nms"}
    )
    deepx_direct_normalization_required = bool(
        prepared_projection.get("normalization_frozen") is True
    )
    deepx_postprocess_required = bool(
        deepx_raw_postprocess_required
        or deepx_direct_normalization_required
    )
    deepx_postprocess_completed = int(
        _num(prepared_projection.get("postprocess_completed_frames")) or 0
    )
    deepx_completion_attestation = (
        dict(prepared_feed.get("completed_task_endpoint_attestation") or {})
        if isinstance(
            prepared_feed.get("completed_task_endpoint_attestation"),
            Mapping,
        )
        else {}
    )
    (
        deepx_completion_attested,
        deepx_completion_attestation_status,
    ) = _completed_attestation_aliases(deepx_completion_attestation)
    deepx_raw_completion_binding = (
        _verified_raw_completed_task_attestation(
            prepared_feed,
            deepx_frozen_contract,
            completed_frames=completed_frames,
            postprocess_completed_frames=deepx_postprocess_completed,
        )
        if deepx_raw_postprocess_required
        and deepx_frozen_contract
        else {}
    )
    deepx_sealed_completion_result = (
        prepared_feed.get("frozen_decoded_nms_normalization_result")
        if deepx_direct_normalization_required
        else prepared_feed.get("frozen_host_postprocess_result")
    )
    if (
        deepx_postprocess_required
        and isinstance(deepx_sealed_completion_result, Mapping)
    ):
        (
            deepx_completed_artifact_persisted,
            deepx_completed_artifact_status,
        ) = _completed_result_artifact_persistence_status(
            prepared_feed,
            sealed_result=deepx_sealed_completion_result,
            allowed_root=benchmark_set / "results" / run_id,
            expected_path=(
                benchmark_set / "results" / run_id
                / "deepx_prepared_feed.completed_task_result_artifact.json"
            ),
        )
    else:
        deepx_completed_artifact_persisted = not deepx_postprocess_required
        deepx_completed_artifact_status = (
            "not_applicable" if not deepx_postprocess_required
            else "completed_task_result_artifact_missing"
        )
    expected_deepx_completion_mode = (
        "integrated_accelerator_plus_frozen_normalization"
        if deepx_direct_normalization_required
        else "frozen_host_tail"
        if deepx_raw_postprocess_required
        else ""
    )
    deepx_postprocess_completion_verified = bool(
        not deepx_postprocess_required
        or (
            (
                deepx_direct_contract
                if deepx_direct_normalization_required
                else deepx_frozen_contract
            )
            and prepared_projection.get("postprocess_included") is True
            and deepx_postprocess_completed == completed_frames
            and prepared_projection.get(
                "postprocess_completion_verified"
            ) is True
            and deepx_completion_attested
            and deepx_completed_artifact_persisted
            and (
                bool(deepx_raw_completion_binding)
                if deepx_raw_postprocess_required
                else True
            )
            and str(
                deepx_completion_attestation.get(
                    "completed_task_completion_mode"
                ) or prepared_feed.get(
                    "completed_task_completion_mode"
                ) or ""
            ) == expected_deepx_completion_mode
        )
    )
    exact_deepx_counter = bool(
        exact_deepx_counter and deepx_postprocess_completion_verified
    )
    if backend == "native_full_deepx" and not exact_deepx_counter:
        ok = False
        fps = None
    if step.get("timed_out"):
        reason = "native_full_suite_timeout"
    elif step.get("rc") != 0:
        reason = "native_full_suite_runner_failed"
    elif not prepared_feed_contract_binding_ok:
        reason = (
            "deepx_prepared_feed_expected_image_missing"
            if backend == "native_full_deepx" and not expected_prepared_image_sha256
            else "deepx_prepared_feed_image_mismatch"
            if not prepared_feed_image_binding_ok
            else "deepx_prepared_input_tensor_mismatch"
            if not prepared_input_binding_ok
            else "deepx_prepared_feed_contract_version_mismatch"
        )
    elif backend == "native_full_deepx" and not metric_source_consistent:
        reason = (
            "deepx_prepared_feed_top_level_conflict"
            if prepared_projection_conflicts
            else str(metrics.get("source_consistency_reason") or "deepx_result_source_inconsistent")
        )
    elif backend == "native_full_deepx" and not outer_makespan_verified:
        reason = "deepx_outer_makespan_missing_or_inconsistent"
    elif backend == "native_full_deepx" and not exact_deepx_counter:
        reason = (
            "deepx_completed_task_result_artifact_persistence_failed"
            if not deepx_completed_artifact_persisted
            else "deepx_frozen_postprocess_completion_failed"
            if not deepx_postprocess_completion_verified
            else "deepx_completed_work_unit_verification_failed"
        )
    elif not fps:
        reason = "native_full_metrics_missing"
    else:
        reason = ""
    status = "ok" if ok else (
        "timeout" if step.get("timed_out")
        else "prepared_feed_input_mismatch"
        if not prepared_feed_image_binding_ok or not prepared_input_binding_ok
        else "missing_fps" if step.get("rc") == 0
        else "failed"
    )
    explicit_runtime_precision = _explicit_runtime_precision(metric_row)
    return {
        "backend": backend,
        "producer_impl": f"{backend}_native_full_suite",
        "model": model,
        "case": "full",
        "execution_mode": "native_full_baseline",
        "execution_precision": explicit_runtime_precision,
        "full_runtime_precision": explicit_runtime_precision,
        "runtime_precision_source": (
            "deepx_prepared_feed_result" if explicit_runtime_precision else "unavailable"
        ),
        "engine_precision": "unavailable",
        "duration_s": float(getattr(ns, "duration_s", 0.0) or 0.0),
        "frames": int(throughput_frames),
        "warmup": int(ns.warmup),
        "warmup_policy": "exact_untimed_frames_before_measured_loop",
        "completed_frames": completed_frames if exact_deepx_counter else None,
        "completed_work_units": completed_work_units if exact_deepx_counter else None,
        "completed_work_units_source": str(
            prepared_projection.get("completed_work_units_source") or ""
        ) if exact_deepx_counter else "",
        "completed_work_units_status": "exact_runtime_counter" if exact_deepx_counter else "unavailable_or_count_mismatch",
        "ok": ok,
        "status": status,
        "fps_makespan": fps,
        "request_latency": prepared_feed.get("request_latency") or metric_row.get("request_latency"),
        "latency_mean_ms": latency,
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "latency_semantics": str(
            prepared_feed.get("latency_semantics") or metric_row.get("latency_semantics") or ""
        ),
        "measured_makespan_s": measured_makespan_s,
        "measured_duration_s": measured_makespan_s if outer_makespan_verified else None,
        "measurement_endpoint": "completed_task",
        "measurement_boundary": "first_task_start_to_last_task_completion",
        "fps_source": (
            "dx_engine_prepared_feed_outer_makespan" if outer_makespan_verified
            else ""
        ),
        "outer_makespan_verified": outer_makespan_verified,
        "report": str(metrics.get("result_source") or ""),
        "run_id": result_run_id,
        "result_source": str(metrics.get("result_source") or ""),
        "result_source_kind": str(metrics.get("result_source_kind") or ""),
        "result_source_consistent": bool(metric_source_consistent),
        "result_source_consistency_reason": str(metrics.get("source_consistency_reason") or ""),
        "result_source_conflicts": list(metrics.get("source_conflicts") or []),
        "deepx_prepared_feed_projection_conflicts": prepared_projection_conflicts,
        "runtime_python": suite_python,
        "runtime_python_sites": list(_runtime_sites),
        "dxnn_path": str(metric_row.get("dxnn_path") or ""),
        "deepx_prepared_feed_benchmark": prepared_feed,
        "e2e_scope": (
            "full_task_pipeline" if deepx_postprocess_required
            else "accelerator_output_endpoint"
        ),
        "comparison_endpoint_stratum": (
            "decoded_nms" if deepx_postprocess_required
            else str(metric_row.get("contract_family") or "")
        ),
        "measurement_concurrency": 1,
        "host_postprocess_frozen": deepx_raw_postprocess_required,
        "normalization_frozen": (
            deepx_direct_normalization_required
        ),
        "postprocess_included": deepx_postprocess_required,
        "postprocess_completed_frames": deepx_postprocess_completed,
        "postprocess_completion_verified": deepx_postprocess_completion_verified,
        "frozen_host_postprocess_contract": deepx_frozen_contract,
        "frozen_host_postprocess_contract_sha256": str(
            deepx_frozen_contract.get("contract_sha256") or ""
        ),
        "frozen_host_postprocess_result": dict(
            prepared_feed.get("frozen_host_postprocess_result") or {}
        ),
        "frozen_decoded_nms_normalization_contract": (
            deepx_direct_contract
        ),
        "frozen_decoded_nms_normalization_contract_sha256": str(
            deepx_direct_contract.get("contract_sha256") or ""
        ),
        "frozen_decoded_nms_normalization_result": dict(
            prepared_feed.get(
                "frozen_decoded_nms_normalization_result"
            ) or {}
        ),
        "completed_task_stage": str(
            deepx_completion_attestation.get("stage") or ""
        ),
        "completed_task_contract_family": (
            "decoded_nms"
            if deepx_completion_attestation.get("attested") is True
            else ""
        ),
        "completed_task_endpoint_contract_hash": str(
            deepx_completion_attestation.get(
                "endpoint_contract_hash"
            ) or ""
        ),
        "completed_task_output_endpoint_id": str(
            deepx_completion_attestation.get("output_endpoint_id") or ""
        ),
        "completed_task_comparison_endpoint_contract": dict(
            deepx_completion_attestation.get(
                "completed_task_comparison_endpoint_contract"
            ) or {}
        ),
        "completed_task_comparison_endpoint_contract_hash": str(
            deepx_completion_attestation.get(
                "completed_task_comparison_endpoint_contract_hash"
            ) or ""
        ),
        "completed_task_comparison_output_endpoint_id": str(
            deepx_completion_attestation.get(
                "completed_task_comparison_output_endpoint_id"
            ) or ""
        ),
        "completed_task_completion_mode": str(
            deepx_completion_attestation.get(
                "completed_task_completion_mode"
            ) or ""
        ),
        "completed_task_endpoint_attested": deepx_completion_attested,
        "completed_task_endpoint_attestation_status":
            deepx_completion_attestation_status,
        "completed_task_endpoint_attestation": (
            deepx_completion_attestation
        ),
        "source_endpoint_contract_hash": str(
            prepared_feed.get("source_endpoint_contract_hash") or ""
        ),
        "source_output_endpoint_attestation": dict(
            prepared_feed.get("source_output_endpoint_attestation")
            or {}
        ),
        "original_image_wh": list(prepared_feed.get("original_image_wh") or []),
        "input_image": str(prepared_image_path.resolve()) if prepared_image_path and prepared_image_path.is_file() else prepared_image,
        "input_image_sha256": _sha256_file(prepared_image_path) if prepared_image_path and prepared_image_path.is_file() else "",
        "input_case": "full",
        "input_manifest": str(metric_row.get("input_manifest") or ""),
        "runtime_input_dtype": str(prepared_input.get("dtype") or ""),
        "runtime_input_shape": prepared_input.get("shape") or [],
        "runtime_input_layout": str(prepared_input.get("layout") or ""),
        "runtime_preprocess_mode": str(prepared_input.get("preprocess_mode") or ""),
        "runtime_normalization": str(prepared_input.get("normalization") or ""),
        "runtime_color_space": str(prepared_input.get("color_space") or ""),
        "task": str(prepared_feed.get("task") or metric_row.get("task") or "auto"),
        "performance_benchmark_source": performance_benchmark_source,
        "prepared_feed_contract_version": prepared_feed_contract_version,
        "expected_prepared_feed_contract_version": (
            DEEPX_PREPARED_FEED_CONTRACT_VERSION
            if backend == "native_full_deepx" else ""
        ),
        "prepared_feed_contract_binding_ok": prepared_feed_contract_binding_ok,
        "prepared_feed_image_binding_ok": prepared_feed_image_binding_ok,
        "prepared_input_binding_verified": prepared_input_binding_ok,
        "prepared_input_binding_status": (
            "verified_exact" if prepared_input_binding_ok
            else expected_runtime_input_status
        ),
        "prepared_input_file": str(
            prepared_feed.get("prepared_input_file") or ""
        ),
        "prepared_input_sha256": str(
            prepared_feed.get("prepared_input_sha256") or ""
        ),
        "prepared_input_file_sha256": str(
            prepared_feed.get("prepared_input_file_sha256") or ""
        ),
        "prepared_input_bytes": int(
            prepared_feed.get("prepared_input_bytes") or 0
        ),
        "prepared_input_name": str(
            prepared_feed.get("prepared_input_name") or ""
        ),
        "prepared_input_shape": list(
            prepared_feed.get("prepared_input_shape") or []
        ),
        "prepared_input_dtype": str(
            prepared_feed.get("prepared_input_dtype") or ""
        ),
        "prepared_input_layout": str(
            prepared_feed.get("prepared_input_layout") or ""
        ),
        "prepared_input_source_image_id": str(
            prepared_feed.get("prepared_input_source_image_id") or ""
        ),
        "prepared_input_source_image_sha256": str(
            prepared_feed.get("prepared_input_source_image_sha256") or ""
        ),
        "runtime_preprocessing_identity": dict(
            prepared_feed.get("runtime_preprocessing_identity") or {}
        ),
        "runtime_preprocessing_sha256": str(
            prepared_feed.get("runtime_preprocessing_sha256") or ""
        ),
        "runtime_numeric_input_identity": dict(
            prepared_feed.get("runtime_numeric_input_identity") or {}
        ),
        "runtime_numeric_input_sha256": str(
            prepared_feed.get("runtime_numeric_input_sha256") or ""
        ),
        "completed_task_result_artifact_saved": bool(
            prepared_feed.get("completed_task_result_artifact_saved") is True
        ),
        "completed_task_result_artifact": dict(
            prepared_feed.get("completed_task_result_artifact") or {}
        ),
        "completed_task_result_artifact_sha256": str(
            prepared_feed.get("completed_task_result_artifact_sha256") or ""
        ),
        "completed_task_result_artifact_path": str(
            prepared_feed.get("completed_task_result_artifact_path") or ""
        ),
        "completed_task_result_artifact_file_sha256": str(
            prepared_feed.get(
                "completed_task_result_artifact_file_sha256"
            ) or ""
        ),
        "completed_task_result_artifact_verification_status": (
            deepx_completed_artifact_status
        ),
        "expected_prepared_feed_image": str(expected_prepared_image or ""),
        "expected_prepared_feed_image_sha256": expected_prepared_image_sha256,
        "actual_prepared_feed_image_sha256": prepared_image_sha256,
        "performance_input_contract_mode": (
            "explicit" if performance_benchmark_source == "dx_engine_prepared_feed"
            else "diagnostic_fallback"
        ) if backend == "native_full_deepx" else "not_applicable",
        "steps": [step],
        **_diagnostic_fields(
            step, failure_reason=reason,
            status_detail=(
                f"examined_result_files={len(unique)};current_result_files={len(changed)};result_run_id={result_run_id or 'legacy_missing'};"
                f"result_runtime_ok={result_runtime_ok};result_identity_ok={result_identity_ok};"
                f"prepared_feed_contract_version={prepared_feed_contract_version or 'missing'};"
                f"prepared_feed_contract_binding_ok={prepared_feed_contract_binding_ok};"
                f"prepared_feed_image_binding_ok={prepared_feed_image_binding_ok};"
                f"prepared_input_binding_ok={prepared_input_binding_ok};"
                f"completed_result_persistence={deepx_completed_artifact_status}"
            ),
        ),
        **({key: prepared_feed.get(key) for key in (
            "task_complete", "completed_task_stage", "classification_topk",
            "e2e_scope", "comparison_endpoint_stratum", "postprocess_location",
            "postprocess_included", "postprocess_completed_frames", "postprocess_completion_verified",
        )} if prepared_feed.get("completed_task_stage") == "classification_top1_top5" else {}),
    }


def _deepx_semantic_completion_failure(
    semantic: Mapping[str, Any], *, expected_model: str = "",
) -> str:
    """Check the already executed semantic NMS, without inventing a hotloop."""
    stage = str(semantic.get("contract_family") or semantic.get("stage") or "").strip().lower()
    if str(semantic.get("task") or "").strip().lower() != "detection":
        return ""
    if stage == "decoded_nms" and semantic.get("normalization_frozen") is True:
        try:
            verified = verify_frozen_decoded_nms_normalization_contract(
                semantic.get("frozen_decoded_nms_normalization_contract"))
            if (semantic.get("postprocess_included") is not True
                    or str(semantic.get("frozen_decoded_nms_normalization_contract_sha256") or "") != str(verified.get("contract_sha256") or "")):
                return "deepx_semantic_direct_normalization_contract_binding_mismatch"
            build_normalized_detection_endpoint_attestation(
                verified, semantic.get("frozen_decoded_nms_normalization_result"),
                completed_frames=1, postprocess_completed_frames=1,
            )
        except (FrozenPostprocessError, TypeError, ValueError) as exc:
            return str(exc) or "deepx_semantic_direct_normalization_invalid"
        return ""
    if stage not in {"raw_head", "decoded_pre_nms"}:
        return ""
    if str(semantic.get("stage") or "") != stage:
        return "deepx_semantic_endpoint_stage_mismatch"
    if semantic.get("endpoint_contract_complete") is not True:
        return "deepx_semantic_endpoint_contract_incomplete"
    contract = semantic.get("frozen_host_postprocess_contract")
    if not isinstance(contract, Mapping) or not contract:
        return "deepx_semantic_frozen_postprocess_contract_missing"
    try:
        verified = verify_frozen_postprocess_contract(contract)
        if expected_model and str(verified.get("model_id") or "") != str(expected_model):
            return "deepx_semantic_frozen_postprocess_model_mismatch"
        source = semantic.get("output_endpoint_attestation")
        signature = verified.get("raw_output_tensor_signature")
        if (not isinstance(source, Mapping)
                or source.get("attested") is not True or source.get("status") != "passed"
                or source.get("stage") != stage
                or source.get("endpoint_contract_hash") != semantic.get("endpoint_contract_hash")
                or source.get("tensor_signature") != signature
                or semantic.get("tensor_signature") != signature):
            return "deepx_semantic_frozen_postprocess_source_endpoint_mismatch"
        if (str(verified.get("source_contract_family") or "") != stage
                or str(semantic.get("frozen_host_postprocess_contract_sha256") or "")
                    != str(verified.get("contract_sha256") or "")):
            return "deepx_semantic_frozen_postprocess_contract_binding_mismatch"
        result = semantic.get("frozen_host_postprocess_result")
        if (semantic.get("host_postprocess_frozen") is not True
                or semantic.get("postprocess_included") is not True
                or not isinstance(result, Mapping) or not result):
            return "deepx_semantic_frozen_postprocess_result_missing"
        build_completed_detection_endpoint_attestation(
            verified, result, completed_frames=1, postprocess_completed_frames=1,
            source_endpoint_contract_hash=str(semantic.get("endpoint_contract_hash") or ""),
        )
    except (FrozenPostprocessError, TypeError, ValueError) as exc:
        return str(exc) or "deepx_semantic_frozen_postprocess_invalid"
    return ""


def _deepx_full_preparation_failure(
    semantic: Mapping[str, Any], model: str,
) -> dict[str, Any] | None:
    manifest = str(semantic.get("input_manifest") or "").strip()
    problem = _deepx_semantic_completion_failure(semantic, expected_model=model) if semantic.get("ok") is True else ""
    if semantic.get("ok") is True and manifest and Path(manifest).is_file() and not problem:
        return None
    reason = str(semantic.get("failure_reason") or problem or "deepx_semantic_prepared_input_missing")
    return {
        "backend": "native_full_deepx", "model": model, "case": "full",
        "execution_mode": "native_full_baseline", "ok": False,
        "status": "semantic_prepared_input_failed",
        "semantic_dump_status": str(semantic.get("status") or "failed"),
        "semantic_dump_failure_reason": reason, "failure_reason": reason,
        **{key: semantic[key] for key in (
            "original_full_result_file", "original_full_failure_context_file",
            "original_full_status", "original_full_error", "error", "status_detail",
        ) if key in semantic},
    }


def _deepx_full_series_preflight(
    benchmark_set: Path, model: str, ns: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve the exact Full input once, outside all timed repetitions.

    This may run one untimed semantic inference. A failure is one preparation
    attempt and zero performance attempts, without fabricated child receipts.
    """
    try:
        semantic = _semantic_full_dump(
            benchmark_set, model, "native_full_deepx", "deepx_m1_full", ns, force=True,
        )
    except Exception as exc:
        semantic = {"ok": False, "status": "exception",
                    "failure_reason": "deepx_full_preparation_exception",
                    "error": f"{type(exc).__name__}: {exc}"}
    failure = _deepx_full_preparation_failure(semantic, model)
    if failure is not None:
        requested = max(1, int(ns.repetitions))
        failure.update({
            "status": "blocked_before_repetitions", "runtime_success": False,
            "preparation_count_attempted": 1,
            "preparation_evidence": dict(semantic),
            "repetition_count_requested": requested, "repetitions_requested": requested,
            "repetition_count_attempted": 0, "repetition_count_valid": 0,
            "repetitions_completed": 0, "repetition_status": "blocked",
            "repetition_records": [], "repetition_evidence": [],
            "repetition_runtime_instance_ids": [],
            "repetition_independence_verified": False,
            "fps_makespan": None, "frames": 0, "completed_frames": 0,
            "completed_work_units": 0,
        })
    return dict(semantic), failure


def _row_for_backend(benchmark_set: Path, model: str, backend: str, ns: argparse.Namespace) -> dict[str, Any]:
    if backend == "tensorrt":
        runtime_row = _native_trt_full(benchmark_set, model, ns)
        runtime_row["runtime_success"] = bool(
            runtime_row.get("ok") is True
            and (_num(runtime_row.get("fps_makespan")) or 0.0) > 0.0
            and not bool(runtime_row.get("timed_out"))
        )
        row = _attach_semantic_dump(
            runtime_row, benchmark_set, model,
            "native_full_tensorrt", "ort_tensorrt", ns,
        )
        return _attach_trt_completed_task_hotloop(
            row, benchmark_set, model, ns,
        )
    if backend == "hailo10h":
        return _native_hailo_full(benchmark_set, model, "hailo10h", ns)
    if backend == "hailo8":
        return _native_hailo_full(benchmark_set, model, "hailo8", ns)
    if backend == "deepx":
        # DeepX Performance and Energy replay the exact numeric input produced
        # by this untimed semantic execution.  Generate and verify it first;
        # source-image equality alone is not a numeric-input contract.
        semantic = getattr(ns, "deepx_full_precomputed_semantic", None)
        if not isinstance(semantic, Mapping):
            semantic = _semantic_full_dump(
                benchmark_set, model, "native_full_deepx", "deepx_m1_full",
                ns, force=True,
            )
        failure = _deepx_full_preparation_failure(semantic, model)
        if failure is not None:
            return failure
        input_manifest = Path(str(semantic["input_manifest"])).expanduser()
        runtime_row = _generic_full_via_suite(
            benchmark_set, model, "native_full_deepx", "deepx_m1_full",
            ns, prepared_input_manifest=input_manifest,
        )
        runtime_row["runtime_success"] = bool(
            runtime_row.get("ok") is True
            and (_num(runtime_row.get("fps_makespan")) or 0.0) > 0.0
            and not bool(runtime_row.get("timed_out"))
        )
        return _attach_semantic_dump(
            runtime_row,
            benchmark_set, model, "native_full_deepx", "deepx_m1_full", ns,
            precomputed_result=semantic,
        )
    return {
        "backend": backend, "model": model, "case": "full",
        "execution_mode": "native_full_baseline", "ok": False,
        "status": "unsupported_backend", "failure_reason": "unsupported_backend",
        "steps": [],
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Staged native_fifo_evalsets root")
    parser.add_argument(
        "--remote-contract-preflight",
        action="store_true",
        help=(
            "Import and invoke the remote-safe Hailo Full contract promoter "
            "without starting a hardware workload."
        ),
    )
    parser.add_argument("--models", default="")
    parser.add_argument("--backends", default="tensorrt,hailo8,hailo10h,deepx")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument(
        "--repetitions", type=int, default=1,
        help=(
            "Independent executions per model/backend. The exported primary row "
            "uses the median and a deterministic percentile-bootstrap 95%% CI."
        ),
    )
    parser.add_argument("--deepx-classification-profile-json", default="", help="Selected workflow preprocessing profile; checked before any DeepX Full preparation or repetition.")
    parser.add_argument("--setup-id", default="")
    parser.add_argument("--comparison-backend", default="")
    parser.add_argument("--comparison-precision", default="", help="Legacy Split-boundary comparison stratum. This is not the Full runtime precision.")
    parser.add_argument("--inflight", type=int, default=8)
    parser.add_argument("--trt-precision", default="fp16")
    parser.add_argument(
        "--trt-quality-producer-json", default="",
        help=(
            "Local/remote JSON producer set for this physical setup. TensorRT "
            "Full rows fail closed unless the requested model has exactly one "
            "signed Quality-FIRST producer in producers_by_model."
        ),
    )
    parser.add_argument(
        "--quality-request-binding-set", default="",
        help=(
            "Hash-sealed setup-local Central-Quality request provenance for "
            "vendor Native Full rows. Only exact backend/model keys are used."
        ),
    )
    parser.add_argument("--workspace-mb", type=int, default=4096)
    parser.add_argument("--engine-build-python", default="auto", help="ONNX/NumPy/ORT-capable Python used for TensorRT builds and as an extra-site source for vendor full runners.")
    parser.add_argument("--image-map", default="", help="JSON object or JSON file mapping model -> case -> exact validation image.")
    parser.add_argument("--preprocess-mode", default="auto", choices=("auto", "resize", "letterbox"))
    parser.add_argument("--letterbox-pad-value", type=int, default=114)
    parser.add_argument("--dump-outputs", action=argparse.BooleanOptionalAction, default=False, help="Generate exact-input semantic output dumps for successful Native Full rows.")
    parser.add_argument(
        "--diagnostic-deepx-input-probes", action="store_true",
        help="Permit diagnostic DeepX preprocessing/layout probes. Omit for strict/final evidence.",
    )
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--no-shapes", action="store_true")
    parser.add_argument("--out-dir", default="", help="Fresh evidence root for an energy replay; canonical BenchmarkSet artifacts remain untouched.")
    parser.add_argument("--expected-runner-sha256", default="", help="Fail closed unless this runner file matches the successful Full command contract.")
    parser.add_argument("--expected-runner-path", default="")
    parser.add_argument("--expected-runner-root", default="")
    parser.add_argument("--expected-hef-sha256", default="", help="Optional Hailo Full artifact binding from the successful command contract.")
    parser.add_argument("--expected-input-image-sha256", default="", help="Optional exact Hailo Full input binding from the successful command contract.")
    parser.add_argument("--energy-workload-only", action="store_true", help="Run only a sealed, preverified Full inference hotloop. No build, discovery, semantic dump or capability probe.")
    parser.add_argument("--energy-preflight-only", action="store_true", help="Verify all Full workload artifacts and emit a short-lived attestation before collector sampling.")
    energy_contract_source = parser.add_mutually_exclusive_group()
    energy_contract_source.add_argument(
        "--energy-command-contract-json", default="",
        help="Legacy inline sealed Native Full command contract.",
    )
    energy_contract_source.add_argument(
        "--energy-command-contract-file", default="",
        help=(
            "Hash-staged remote JSON file containing the sealed Native Full "
            "command contract. Preferred for preflight/workload replay."
        ),
    )
    parser.add_argument("--preflight-nonce", default="")
    parser.add_argument("--preflight-attestation", default="", help="Small preflight attestation consumed by --energy-workload-only.")
    parser.add_argument("--preflight-attestation-out", default="", help="Remote attestation path written by --energy-preflight-only.")
    parser.add_argument("--preflight-attestation-max-age-s", type=float, default=300.0)
    ns = parser.parse_args()
    if ns.remote_contract_preflight:
        from onnx_splitpoint_tool.hailo_full_contract_promotion import (
            remote_import_preflight,
        )

        print(json.dumps(remote_import_preflight(), sort_keys=True), flush=True)
        return 0
    if ns.energy_workload_only and ns.energy_preflight_only:
        print(json.dumps({"ok": False, "status": "energy_mode_mutually_exclusive"}), file=sys.stderr)
        return 4
    expected_runner_sha = str(ns.expected_runner_sha256 or "").strip().lower()
    actual_runner_sha = _sha256_file(Path(__file__).resolve())
    if (
        (ns.energy_preflight_only or ns.energy_workload_only)
        and re.fullmatch(r"[0-9a-f]{64}", expected_runner_sha) is None
    ):
        print(json.dumps({
            "ok": False,
            "status": "expected_full_runner_sha256_missing_or_invalid",
        }, indent=2), file=sys.stderr, flush=True)
        return 4
    if ns.energy_preflight_only or ns.energy_workload_only:
        runner_lexical = _absolute_without_resolving(Path(__file__))
        if (
            not str(ns.expected_runner_path or "").strip()
            or not str(ns.expected_runner_root or "").strip()
            or not _confined_regular_file(
                runner_lexical,
                allowed_root=Path(ns.expected_runner_root),
                expected_path=Path(ns.expected_runner_path),
            )
        ):
            print(json.dumps({
                "ok": False,
                "status": "full_runner_role_path_mismatch",
            }, indent=2), file=sys.stderr, flush=True)
            return 4
    if expected_runner_sha and expected_runner_sha != actual_runner_sha:
        print(json.dumps({
            "ok": False,
            "status": "full_runner_sha256_mismatch",
            "expected_runner_sha256": expected_runner_sha,
            "actual_runner_sha256": actual_runner_sha,
        }, indent=2), file=sys.stderr, flush=True)
        return 4
    # Both energy paths return before normal engine-Python selection and Full
    # discovery.  Heavy artifact hashing is confined to preflight; workload-only
    # validates only the small, fresh attestation.
    if ns.energy_preflight_only:
        return _energy_preflight_only(ns)
    if ns.energy_workload_only:
        return _energy_workload_only(ns)
    (
        ns.quality_request_binding_set_data,
        quality_request_binding_set_status,
    ) = _load_full_quality_binding_set(ns)
    ns.quality_request_binding_set_load_status = (
        quality_request_binding_set_status
    )
    if quality_request_binding_set_status not in {
        "not_requested", "quality_request_binding_set_verified",
        "quality_request_binding_set_verified_partial",
    }:
        print(json.dumps({
            "ok": True,
            "status": quality_request_binding_set_status,
            "runtime_execution_continues": True,
            "claim_eligible": False,
            "reason": "quality_request_binding_set_invalid",
        }, indent=2), file=sys.stderr, flush=True)
        ns.quality_request_binding_set_data = {}
    selected_python, selected_meta = _select_engine_python(str(ns.engine_build_python or "auto"))
    ns.engine_python_selected = selected_python
    ns.engine_python_sites = _site_packages_for_python(selected_python)
    ns.image_map_data = _parse_image_map(str(ns.image_map or ""))
    print(f"[native-full] engine_build_python={selected_python or 'unavailable'}", flush=True)

    root = Path(ns.root).expanduser().resolve()
    models = [item.strip() for item in ns.models.split(",") if item.strip()]
    backends = [item.strip().lower() for item in ns.backends.split(",") if item.strip()]
    model_roots = _model_roots(root, models)
    selected_models = {model for model, _ in model_roots}
    selection_errors: list[str] = []
    if not root.is_dir():
        selection_errors.append("root_not_found")
    if models:
        missing_models = [model for model in models if model not in selected_models]
        if missing_models:
            selection_errors.append(
                "benchmark_set_index_missing:" + ",".join(missing_models)
            )
    if not model_roots:
        selection_errors.append("requested_or_discovered_benchmark_sets_empty")
    if not backends:
        selection_errors.append("requested_backends_empty")
    if selection_errors:
        print(json.dumps({
            "ok": False,
            "status": "selection_invalid",
            "failure_reason": "requested_benchmark_set_or_backend_unavailable",
            "selection_errors": selection_errors,
            "root": str(root),
            "models": models,
            "backends": backends,
            "rows": 0,
        }, indent=2), flush=True)
        return 3
    vendor_backend_names = {
        "hailo8": "native_full_hailo8",
        "hailo10h": "native_full_hailo10h",
        "deepx": "native_full_deepx",
    }
    requested_vendor_keys = {
        f"{vendor_backend_names[backend]}|{model}"
        for backend in backends
        if backend in vendor_backend_names
        for model in models
    }
    if requested_vendor_keys:
        binding_payload = ns.quality_request_binding_set_data
        binding_keys = set(
            str(key) for key in (
                binding_payload.get("bindings_by_backend_model") or {}
            )
        ) if isinstance(binding_payload, Mapping) else set()
        required_binding_keys = set(
            str(key) for key in (
                binding_payload.get("required_binding_keys") or []
            )
        ) if isinstance(binding_payload, Mapping) else set()
        binding_set_applicable = bool(
            quality_request_binding_set_status in {
                "quality_request_binding_set_verified",
                "quality_request_binding_set_verified_partial",
            }
            and required_binding_keys == requested_vendor_keys
            and binding_keys.issubset(requested_vendor_keys)
        )
        if not binding_set_applicable:
            print(json.dumps({
                "ok": True,
                "status": "quality_request_binding_set_incomplete",
                "runtime_execution_continues": True,
                "claim_eligible": False,
                "reason": (
                    "requested_vendor_full_quality_binding_unavailable"
                ),
                "requested_binding_keys": sorted(requested_vendor_keys),
                "available_binding_keys": sorted(binding_keys),
                "required_binding_keys": sorted(required_binding_keys),
            }, indent=2), flush=True)
            ns.quality_request_binding_set_data = {}
    rows: list[dict[str, Any]] = []
    for model_index, (model, benchmark_set) in enumerate(model_roots, 1):
        for backend in backends:
            from onnx_splitpoint_tool.native_job_identity import planned_native_identity, attach_identity_without_conflicts
            planned_identity = planned_native_identity({"backend":"native_full_" + ("hailo10h" if backend == "hailo10" else backend),
                                                       "model":model, "case":"full", "setup_id":ns.setup_id,
                                                       "comparison_backend":ns.comparison_backend,"precision":ns.comparison_precision})
            repeat_rows: list[dict[str, Any]] = []
            repetition_count = max(1, int(ns.repetitions))
            deepx_semantic = None
            blocked_row = None
            admission = None
            if backend == "deepx" and str(ns.deepx_classification_profile_json or "").strip():
                from onnx_splitpoint_tool.deepx.config import classification_profile_admission, declared_deepx_task
                from onnx_splitpoint_tool.native_job_identity import planned_native_identity, failed_native_result
                try:
                    profile = json.loads(ns.deepx_classification_profile_json)
                    if not isinstance(profile, dict):
                        raise ValueError("profile must be an object")
                    model_contract = _load_json(benchmark_set / "benchmark_set.json")
                    if not isinstance(model_contract, dict): model_contract = {}
                    full_output = _load_json(benchmark_set / "deepx/deepx_m1/full/output_contract.json")
                    # Empty model argument disables the legacy filename heuristic.
                    plan_task, _ = _run_plan_meta(benchmark_set, "deepx_m1_full", "")
                    declared_task = declared_deepx_task(model_contract, full_output, plan_task=plan_task)
                    admission = classification_profile_admission(profile, {"task": declared_task})
                    reason = str(admission.get("reason") or "") if not admission["allowed"] else ""
                    if declared_task == "classification" and admission["allowed"]:
                        actual_mode = str(full_output.get("classification_preprocessing") or "") if isinstance(full_output, dict) else ""
                        if actual_mode != str(admission["classification_preprocessing"]):
                            reason = "deepx_classification_preprocessing_artifact_mismatch:expected=" + str(admission["classification_preprocessing"]) + ",observed=" + (actual_mode or "missing")
                    if reason:
                        identity = planned_native_identity({"backend":"native_full_deepx", "model":model, "case":"full", "setup_id":ns.setup_id, "comparison_backend":ns.comparison_backend, "precision":ns.comparison_precision})
                        blocked_row = failed_native_result(identity, failure_stage="classification_preprocessing_admission", failure_reason=reason,
                                                           repetition_count_requested=repetition_count, repetition_records=[], repetition_evidence=[],
                                                           scientific_claim_exclusion_reason=reason, execution_mode="native_full_baseline")
                except (ValueError, TypeError) as exc:
                    from onnx_splitpoint_tool.native_job_identity import failed_native_result
                    blocked_row = failed_native_result({"backend":"native_full_deepx", "model":model, "case":"full", "setup_id":ns.setup_id, "comparison_backend":ns.comparison_backend, "precision":ns.comparison_precision},
                                                      failure_stage="classification_preprocessing_admission", failure_reason="deepx_classification_profile_invalid:" + str(exc), repetition_count_requested=repetition_count)
            if backend == "deepx" and blocked_row is None:
                print(f"[native-full] {model}/{backend} preparing exact Full input", flush=True)
                deepx_semantic, blocked_row = _deepx_full_series_preflight(
                    benchmark_set, model, ns,
                )
            repetition_indices = () if blocked_row is not None else range(1, repetition_count + 1)
            for repetition_index in repetition_indices:
                print(
                    f"[native-full] Modell {model_index}/{len(model_roots)} · {model} · {ns.setup_id}/{backend} Full · repetition={repetition_index}/{repetition_count} · {ns.frames} Frames + {ns.warmup} Warmup · Performance",
                    flush=True,
                )
                rep_ns = argparse.Namespace(**vars(ns))
                rep_ns.full_repetition_index = repetition_index
                if deepx_semantic is not None:
                    rep_ns.deepx_full_precomputed_semantic = deepx_semantic
                # One semantic dump is sufficient for an otherwise identical
                # measurement series and remains outside all performance loops.
                rep_ns.dump_outputs = bool(
                    ns.dump_outputs
                    and (
                        repetition_index == 1
                        or backend == "tensorrt"
                    )
                )
                try:
                    row = _row_for_backend(benchmark_set, model, backend, rep_ns)
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                    row = {
                        # This is our own exception observation, so its job
                        # identity is already known. Vendor aliases describe
                        # Split children and must not be applied to this row.
                        **planned_identity,
                        "execution_mode": "native_full_baseline",
                        "ok": False,
                        "result_ok": False,
                        "runtime_success": False,
                        "status": "exception",
                        "failure_stage": "native_full_backend_call",
                        "failure_reason": "native_full_exception",
                        "primary_failure_reason": "native_full_exception",
                        "status_detail": error_text,
                        "error": error_text,
                        "performance_claim_eligible": False,
                        "steps": [],
                    }
                row["repetition_index"] = repetition_index
                runtime_invocation_token = hashlib.sha256(
                    (
                        f"native-full-child-process:{model}:{backend}:"
                        f"{repetition_index}:{time.time_ns()}"
                    ).encode("utf-8")
                ).hexdigest()
                row["runtime_instance_id"] = f"fresh_process:{runtime_invocation_token}"
                row["repetition_runtime_scope"] = "fresh_process_per_repetition"
                row.setdefault("setup_id", str(ns.setup_id or ""))
                row.setdefault("comparison_backend", str(ns.comparison_backend or ""))
                legacy_comparison_precision = str(
                    row.get("comparison_precision")
                    or row.get("legacy_comparison_precision")
                    or row.get("precision")
                    or ns.comparison_precision
                    or ""
                )
                # Keep ``precision`` for old readers, but label its actual role
                # explicitly.  Never infer Full execution precision from it.
                row["precision"] = legacy_comparison_precision
                row["comparison_precision"] = legacy_comparison_precision
                row["legacy_comparison_precision"] = legacy_comparison_precision
                explicit_runtime_precision = _explicit_runtime_precision(row)
                row["execution_precision"] = explicit_runtime_precision
                row["full_runtime_precision"] = explicit_runtime_precision
                row.setdefault(
                    "runtime_precision_source",
                    "explicit_backend_evidence" if explicit_runtime_precision else "unavailable",
                )
                row = attach_identity_without_conflicts(planned_identity, row)
                repeat_rows.append(row)
            row = blocked_row if blocked_row is not None else _aggregate_full_repetitions(
                repeat_rows, requested=repetition_count,
            )
            if blocked_row is not None:
                row["setup_id"] = str(ns.setup_id or "")
                row["comparison_backend"] = str(ns.comparison_backend or "")
                row["comparison_precision"] = str(ns.comparison_precision or "")
                row["legacy_comparison_precision"] = str(ns.comparison_precision or "")
                row["precision"] = str(ns.comparison_precision or "")
                row["execution_precision"] = ""
                row["full_runtime_precision"] = ""
                row["runtime_precision_source"] = "unavailable"
            row = attach_identity_without_conflicts(planned_identity, row)
            full_contract = _full_command_contract(
                row=row, root=root, benchmark_set=benchmark_set,
                model=model, backend_arg=backend, ns=ns,
            )
            row, quality_request_binding_status = (
                _attach_full_quality_request_binding(
                    row, full_contract, model=model, ns=ns,
                )
            )
            row.setdefault(
                "quality_request_binding_status",
                quality_request_binding_status,
            )
            if quality_request_binding_status not in {
                "not_requested", "not_available_for_identity",
            }:
                # Seal the verified provenance into the successful workload
                # contract (or its explicit failed counterpart).
                full_contract = _full_command_contract(
                    row=row, root=root, benchmark_set=benchmark_set,
                    model=model, backend_arg=backend, ns=ns,
                )
            row["full_command_contract"] = full_contract
            row["full_command_contract_sha256"] = str(
                full_contract.get("contract_sha256") or ""
            )
            if row.get("backend") == "native_full_tensorrt":
                row["quality_first_producer_identity"] = dict(
                    full_contract.get("quality_first_producer_identity") or {}
                )
                row["quality_first_producer_identity_sha256"] = str(
                    full_contract.get(
                        "quality_first_producer_identity_sha256"
                    ) or ""
                )
                for receipt_field in (
                    "engine_build_receipt_path",
                    "engine_build_receipt_sha256",
                    "engine_build_receipt_file_sha256",
                    "trt_engine_build_receipt_sha256",
                    "engine_build_receipt_size_bytes",
                    "engine_build_receipt_file_size_bytes",
                ):
                    row[receipt_field] = full_contract.get(receipt_field)
            if str(full_contract.get("source_model_sha256") or ""):
                row["source_onnx_sha256"] = str(
                    full_contract["source_model_sha256"]
                )
                row["model_sha256"] = str(full_contract["source_model_sha256"])
            row["workload_contract_sha256"] = row["full_command_contract_sha256"]
            for records_key in ("repetition_records", "repetition_evidence"):
                records = row.get(records_key)
                if isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            record["workload_contract_sha256"] = row["workload_contract_sha256"]
            if admission is not None:
                row["deepx_classification_admission"] = admission
                if not admission.get("claim_eligible", True):
                    row.update({"claim_eligible":False,"performance_claim_eligible":False,"energy_claim_eligible":False,
                                "scientific_claim_eligible":False,"counts_as_benchmark":False,"diagnostic_only":True,
                                "scientific_claim_exclusion_reason":admission["scientific_claim_exclusion_reason"]})
            rows.append(row)

    outdir = (
        Path(ns.out_dir).expanduser().resolve() / "analysis_tables"
        if str(ns.out_dir or "").strip()
        else root / "analysis_tables"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    overall_ok = bool(rows and all(row.get("ok") is True for row in rows))
    data = {
        "schema": "onnx-splitpoint/native-full-baseline-eval",
        "schema_version": 5,
        "ok": overall_ok,
        "status": "ok" if overall_ok else "failed",
        "root": str(root),
        "rows": rows,
        "row_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("ok")),
        "failed_count": sum(1 for row in rows if not row.get("ok")),
        "warmup": int(ns.warmup),
        "repetitions": int(ns.repetitions),
        "engine_build_python": selected_meta,
        "engine_python_sites": list(getattr(ns, "engine_python_sites", []) or []),
    }
    json_path = _write_json(outdir / "native_full_baseline_eval.json", data)
    fields = [
        "backend",
        "producer_impl",
        "model",
        "case",
        "setup_id",
        "comparison_backend",
        "precision",
        "comparison_precision",
        "legacy_comparison_precision",
        "execution_precision",
        "full_runtime_precision",
        "runtime_precision_source",
        "engine_precision",
        "full_command_contract_sha256",
        "execution_mode",
        "duration_s",
        "frames",
        "warmup",
        "warmup_policy",
        "warmup_iterations_completed",
        "warmup_iterations_status",
        "comparison_endpoint_stratum",
        "measurement_concurrency",
        "configured_inflight",
        "e2e_scope",
        "completed_task_stage",
        "completed_task_contract_family",
        "completed_task_endpoint_contract_hash",
        "completed_task_output_endpoint_id",
        "completed_task_endpoint_attested",
        "completed_task_endpoint_attestation_status",
        "completed_task_endpoint_attestation",
        "postprocess_completion_verified",
        "frozen_host_postprocess_contract_sha256",
        "completed_frames",
        "completed_work_units",
        "completed_work_units_source",
        "completed_work_units_status",
        "ok",
        "status",
        "fps_makespan",
        "fps_median",
        "fps_ci95_low",
        "fps_ci95_high",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_ci95_low_ms",
        "latency_ci95_high_ms",
        "latency_semantics",
        "completion_interval_mean_ms",
        "completion_interval_semantics",
        "measured_makespan_s",
        "preparation_count_attempted",
        "repetition_count_requested",
        "repetition_count_attempted",
        "repetition_count_valid",
        "repetition_status",
        "repetition_aggregation",
        "gpu_compute_mean_ms",
        "fps_source",
        "result_source",
        "performance_benchmark_source",
        "performance_input_contract_mode",
        "report",
        "failure_reason",
        "status_detail",
        "error",
        "timed_out",
        "returncode",
        "stdout_tail",
        "stderr_tail",
        "semantic_dump_status",
        "semantic_dump_failure_reason",
        "output_dump_manifest",
        "native_output_manifest",
        "input_manifest",
        "runtime_input_dtype",
        "runtime_input_shape",
        "runtime_input_layout",
        "runtime_preprocess_mode",
        "runtime_normalization",
        "runtime_color_space",
        "runtime_preprocessing_identity",
        "runtime_preprocessing_sha256",
        "runtime_numeric_input_identity",
        "runtime_numeric_input_sha256",
        "quality_request_binding_status",
        "quality_request_binding_sha256",
        "quality_request_binding_set_sha256",
        "source_request_sha256",
        "model_sha256",
        "validation_dataset_sha256",
        "validation_dataset_image_ids_sha256",
        "validation_dataset_ground_truth_sha256",
        "task_quality_policy_sha256",
        "runtime_quality_gate_policy_sha256",
        "quality_contract_sha256",
        "preprocessing_contract_sha256",
        "decoder_contract_sha256",
        "nms_contract_sha256",
        "quality_record_endpoint_contract_sha256",
        "central_quality_result_sha256",
        "input_image",
        "input_image_source",
        "input_image_sha256",
        "task",
        "output_format",
        "contract_family",
        "stage",
        "contract_source",
        "endpoint_contract_complete",
        "endpoint_contract_hash",
        "tensor_signature",
        "output_endpoint_attestation",
        "runtime_python",
        "runtime_python_probe",
        "hef_path",
        "hailo_hef_build_receipt_status",
        "hailo_hef_build_receipt_path",
        "hailo_hef_build_receipt_file_sha256",
        "hailo_hef_build_receipt_sha256",
        "hailo_hef_preprocessing_contract_sha256",
    ]
    with (outdir / "native_full_baseline_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    markdown = [
        "# Native full baseline eval",
        "",
        f"Root: `{root}`",
        "",
        "| backend | model | ok | median FPS | FPS 95% CI | latency ms | repeats | status | failure reason |",
        "|---|---|---:|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        markdown.append(
            f"| {row.get('backend', '')} | {row.get('model', '')} | {row.get('ok')} | "
            f"{row.get('fps_makespan', '')} | "
            f"[{row.get('fps_ci95_low', '')}, {row.get('fps_ci95_high', '')}] | "
            f"{row.get('latency_mean_ms', '')} | "
            f"{row.get('repetition_count_valid', '')}/{row.get('repetition_count_requested', '')} | "
            f"{row.get('status', '')} | {row.get('failure_reason', '')} |"
        )
    (outdir / "native_full_baseline_eval.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": overall_ok,
                "status": data["status"],
                "rows": len(rows),
                "ok_count": data["ok_count"],
                "json": str(json_path),
                "md": str(outdir / "native_full_baseline_eval.md"),
            },
            indent=2,
        )
    )
    return 0 if overall_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
