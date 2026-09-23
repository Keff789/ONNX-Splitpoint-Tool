#!/usr/bin/env python3
"""Update an existing EvaluationRun/BenchmarkSets for native-producer execution.

This helper is intentionally practical: it lets an already generated EvalRun be
"upgraded" to the current runner/templates without regenerating HEFs/DXNNs.  It
can refresh generated BenchmarkSet suites, write a native_producers block into
profile.yaml, and optionally execute the native-producer stage directly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_THIS_FILE = Path(__file__).resolve()
ROOT = _THIS_FILE.parents[1] if _THIS_FILE.parent.name == "scripts" else _THIS_FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.workflow.native_transfer import (  # noqa: E402
    build_native_transfer_inventory,
    classify_native_transfer_failure,
    required_remote_bytes,
    write_rsync_files_from,
)
from onnx_splitpoint_tool.quality_service import (  # noqa: E402
    _validate_candidate_execution_contract,
)
from onnx_splitpoint_tool.trt_quality_chain import (  # noqa: E402
    PRODUCER_SET_SCHEMA,
    SPLIT_BINDING_SET_SCHEMA,
    TensorRTQualityChainError,
)
from onnx_splitpoint_tool.native_split_quality import (  # noqa: E402
    canonical_native_split_backend,
    canonical_json_sha256,
    validate_central_native_split_quality_selection,
    validate_native_split_quality_binding,
)
from onnx_splitpoint_tool.native_split_quality_authority import (  # noqa: E402
    native_split_quality_required_for_row,
    resolve_native_split_quality_authority,
)
from onnx_splitpoint_tool.remote_runtime_closure import (  # noqa: E402
    native_remote_package_closure,
)
from onnx_splitpoint_tool.native_execution_contract import (  # noqa: E402
    NATIVE_EXECUTION_FIELDS,
    build_native_execution_contract,
    verify_native_execution_contract,
)
from onnx_splitpoint_tool.validation.accuracy_gates import (  # noqa: E402
    AccuracyGatePolicy,
)

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _script(name: str) -> Path:
    source = ROOT / "scripts" / name
    if source.is_file():
        return source
    packaged = _THIS_FILE.parent / name
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(name)


def _load_yaml(p: Path) -> dict[str, Any]:
    if not p.is_file():
        return {}
    txt = p.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(txt) or {}
        return data if isinstance(data, dict) else {}
    # Fallback: only supports JSON-shaped YAML.
    try:
        data = json.loads(txt)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_yaml(p: Path, data: Mapping[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        p.write_text(yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _strict_json(p: Path) -> Any:
    """Read identity-bearing JSON and reject duplicate keys."""

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise TensorRTQualityChainError(
                    f"duplicate JSON key in TensorRT quality producer set: {key!r}"
                )
            out[key] = value
        return out

    try:
        return json.loads(p.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except TensorRTQualityChainError:
        raise
    except Exception as exc:
        raise TensorRTQualityChainError(
            f"TensorRT quality producer set is not valid JSON: {p}: {exc}"
        ) from exc


def _write_json(p: Path, data: Mapping[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    os.replace(temporary, p)


def _run(cmd: list[str], *, timeout: int | float | None = None, cwd: Path | None = None, label: str = "native-child") -> dict[str, Any]:
    from onnx_splitpoint_tool.native_progress import stream_command
    from onnx_splitpoint_tool.remote.process_lease import (
        REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
        cancel_journaled_remote_processes_from_environment,
        journaled_ssh_wrapper_argv,
    )
    progress_root = Path(os.environ.get("ONNX_SPLITPOINT_NATIVE_PROGRESS_DIR", "")) if os.environ.get("ONNX_SPLITPOINT_NATIVE_PROGRESS_DIR") else None
    # The variant coordinator is a nested Python process, so workflow remote
    # lease ownership arrives via inherited environment.  One boundary here
    # covers all direct SSH split/full/preflight workloads without changing
    # standalone behaviour when no lease scope is configured.
    routed_cmd = journaled_ssh_wrapper_argv(
        cmd, label=label, timeout_s=float(timeout) if timeout is not None else None,
    )
    # Let the broker own the requested remote timeout.  The outer streaming
    # wrapper gets only a bounded cleanup allowance so exact remote TERM/KILL
    # finishes before it stops the local broker/SSH process tree.
    stream_timeout = (
        float(timeout) + REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S
        if timeout is not None and routed_cmd != list(cmd)
        else timeout
    )
    before_terminate = (
        lambda: cancel_journaled_remote_processes_from_environment(grace_s=3.0)
    ) if routed_cmd != list(cmd) else None
    from onnx_splitpoint_tool.process_control import controller_local_activity
    from contextlib import nullcontext
    admission = controller_local_activity("transfer") if cmd and Path(cmd[0]).name in {"rsync", "scp"} else nullcontext()
    with admission:
        return stream_command(routed_cmd, timeout=stream_timeout, cwd=cwd, label=label, heartbeat_s=float(os.environ.get("ONNX_SPLITPOINT_NATIVE_HEARTBEAT_S", "15")), progress_jsonl=(progress_root / "native_progress.jsonl" if progress_root else None), progress_json=(progress_root / "native_progress.json" if progress_root else None), before_terminate=before_terminate)


def _q(s: str | Path) -> str:
    return shlex.quote(str(s))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_namespace(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if (
        len(text) > 160
        or text in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None
    ):
        raise TensorRTQualityChainError(
            "--artifact-namespace must be one safe, non-empty path component"
        )
    return text


def _final_report_remote_context_args(
    backend_results: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Serialize trusted dispatch context for the offline DeepX verifier."""
    records: list[dict[str, Any]] = []
    scopes: dict[tuple[str, str], str] = {}
    for result in backend_results:
        if str(result.get("backend") or "").strip().lower() != "deepx":
            continue
        setup_id = str(result.get("setup_id") or "").strip()
        remote_root = str(result.get("remote_root") or "").strip()
        remote_tool_dir = str(result.get("remote_tool_dir") or "").strip()
        if not setup_id or not remote_root or not remote_tool_dir:
            continue
        scope = (setup_id, remote_root)
        previous = scopes.get(scope)
        if previous is not None and previous != remote_tool_dir:
            raise TensorRTQualityChainError(
                "DeepX reporter remote context drift for "
                f"setup={setup_id!r} root={remote_root!r}"
            )
        scopes[scope] = remote_tool_dir
        record = {
            "schema": (
                "onnx-splitpoint/"
                "native-final-report-remote-execution-context"
            ),
            "schema_version": 1,
            "setup_id": setup_id,
            "remote_root": remote_root,
            "remote_tool_dir": remote_tool_dir,
        }
        if record not in records:
            records.append(record)
    args: list[str] = []
    for record in records:
        args.extend([
            "--remote-execution-context-json",
            json.dumps(record, sort_keys=True, separators=(",", ":")),
        ])
    return args


def _sync_remote_script_v60i(ssh: str, remote_tool_dir: str, script_name: str, required_tokens: tuple[str, ...] = (), timeout: int = 300) -> list[dict[str, Any]]:
    """Copy and capability-check the exact local remote runner."""
    local = _script(script_name)
    if not local.is_file():
        raise FileNotFoundError(f"local runner missing: {local}")
    remote_scripts = f"{str(remote_tool_dir).rstrip('/')}/scripts"
    remote = f"{remote_scripts}/{local.name}"
    steps: list[dict[str, Any]] = []
    mkdir = _run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, f"mkdir -p {_q(remote_scripts)}"], timeout=min(timeout, 120))
    steps.append({"name": f"mkdir_remote_{local.stem}", **mkdir})
    if mkdir.get("rc") != 0:
        raise RuntimeError(f"remote script directory creation failed: {mkdir.get('stderr_tail')}")
    copy = _run(["rsync", "-a", "--checksum", str(local), f"{ssh}:{remote}"], timeout=timeout)
    steps.append({"name": f"sync_{local.stem}", **copy})
    if copy.get("rc") != 0:
        raise RuntimeError(f"remote script sync failed: {copy.get('stderr_tail')}")
    local_sha = hashlib.sha256(local.read_bytes()).hexdigest()
    token_literal = json.dumps(list(required_tokens), separators=(",", ":"))
    verify_py = (
        "import hashlib,json,pathlib,sys;"
        f"p=pathlib.Path({remote!r});"
        "t=p.read_text(encoding='utf-8',errors='replace');"
        f"want={local_sha!r};tokens=json.loads({token_literal!r});"
        "got=hashlib.sha256(p.read_bytes()).hexdigest();"
        "missing=[x for x in tokens if x not in t];"
        "print(json.dumps({'sha256':got,'expected':want,'missing_tokens':missing}));"
        "sys.exit(0 if got==want and not missing else 7)"
    )
    verify = _run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, f"python -c {_q(verify_py)}"], timeout=min(timeout, 120))
    steps.append({"name": f"verify_{local.stem}", **verify, "expected_sha256": local_sha})
    if verify.get("rc") != 0:
        raise RuntimeError(f"remote script capability verification failed: {verify.get('stdout_tail') or verify.get('stderr_tail')}")
    return steps


def _sync_remote_package_asset_v263(
    ssh: str,
    remote_tool_dir: str,
    relative_path: str,
    required_tokens: tuple[str, ...] = (),
    timeout: int = 300,
) -> list[dict[str, Any]]:
    """Stage and byte-verify an imported package module, not just scripts."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe package asset path: {relative_path}")
    local = ROOT / relative
    if not local.is_file():
        raise FileNotFoundError(f"local package asset missing: {local}")
    remote = f"{str(remote_tool_dir).rstrip('/')}/{relative.as_posix()}"
    steps: list[dict[str, Any]] = []
    mkdir = _run([
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh,
        f"mkdir -p {_q(str(Path(remote).parent))}",
    ], timeout=min(timeout, 120))
    steps.append({"name": "mkdir_remote_package_asset", **mkdir, "relative_path": relative.as_posix()})
    if mkdir.get("rc") != 0:
        raise RuntimeError(f"remote package asset directory creation failed: {ssh}:{remote}: {mkdir.get('stderr_tail')}")
    copy = _run(["rsync", "-a", "--checksum", str(local), f"{ssh}:{remote}"], timeout=timeout)
    steps.append({"name": "sync_remote_package_asset", **copy, "relative_path": relative.as_posix()})
    if copy.get("rc") != 0:
        raise RuntimeError(f"remote package asset sync failed: {ssh}:{remote}: {copy.get('stderr_tail')}")
    expected = hashlib.sha256(local.read_bytes()).hexdigest()
    tokens_json = json.dumps(list(required_tokens), separators=(",", ":"))
    verify_py = (
        "import hashlib,json,pathlib,sys;"
        f"p=pathlib.Path({remote!r});expected={expected!r};tokens=json.loads({tokens_json!r});"
        "b=p.read_bytes();text=b.decode('utf-8','replace');got=hashlib.sha256(b).hexdigest();"
        "missing=[x for x in tokens if x not in text];"
        "print(json.dumps({'sha256':got,'expected':expected,'missing_tokens':missing}));"
        "sys.exit(0 if got==expected and not missing else 7)"
    )
    verify = _run([
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh,
        f"python -c {_q(verify_py)}",
    ], timeout=min(timeout, 120))
    steps.append({
        "name": "verify_remote_package_asset", **verify,
        "relative_path": relative.as_posix(), "expected_sha256": expected,
    })
    if verify.get("rc") != 0:
        raise RuntimeError(
            f"remote package asset SHA/token verification failed: {ssh}:{remote}: "
            f"{verify.get('stdout_tail') or verify.get('stderr_tail')}"
        )
    return steps


def _verify_remote_module_binding_v263(
    ssh: str, remote_tool_dir: str, remote_env: str,
    module_name: str, relative_path: str, expected_sha256: str,
    timeout: int = 180,
) -> dict[str, Any]:
    expected_path = f"{str(remote_tool_dir).rstrip('/')}/{relative_path}"
    code = (
        "import hashlib,importlib,json,pathlib,sys;"
        f"m=importlib.import_module({module_name!r});"
        "p=pathlib.Path(m.__file__).resolve();"
        f"want=pathlib.Path({expected_path!r}).resolve();expected={expected_sha256!r};"
        "got=hashlib.sha256(p.read_bytes()).hexdigest();ok=(p==want and got==expected);"
        "print(json.dumps({'ok':ok,'imported_path':str(p),'expected_path':str(want),'sha256':got,'expected_sha256':expected}));"
        "sys.exit(0 if ok else 8)"
    )
    prefix = (str(remote_env).strip() + " && ") if str(remote_env).strip() else ""
    remote = (
        prefix + f"cd {_q(remote_tool_dir)} && "
        + f"PYTHONPATH={_q(remote_tool_dir)}:${{PYTHONPATH:-}} python -c {_q(code)}"
    )
    result = _run([
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, remote,
    ], timeout=min(timeout, 180))
    result.update({
        "name": "verify_remote_module_binding_" + module_name.replace(".", "_"),
        "module": module_name,
        "expected_path": expected_path,
        "expected_sha256": expected_sha256,
    })
    if result.get("rc") != 0:
        raise RuntimeError(
            f"remote module binding verification failed: {ssh}:{expected_path}: "
            f"{result.get('stdout_tail') or result.get('stderr_tail')}"
        )
    return result




def _native_full_failure_reason(record: Mapping[str, Any]) -> str:
    text = "\n".join(str(record.get(k) or "") for k in ("stderr", "stderr_tail", "stdout", "stdout_tail", "error"))
    low = text.lower()
    if "onnx_splitpoint_tool.native_progress" in low or ("modulenotfounderror" in low and "onnx_splitpoint_tool" in low):
        return "remote_runner_import_failed"
    if "no space left on device" in low or "errno 28" in low:
        return "remote_disk_insufficient"
    if bool(record.get("timed_out")) or "timeoutexpired" in low:
        return "native_full_timeout"
    if "no_onnx_capable_engine_build_python" in low:
        return "no_onnx_capable_engine_build_python"
    return "native_full_runtime_failed"

def _setup_id_for_backend(backend: str) -> str:
    return {
        "hailo8": "orin_nx_hailo8_01",
        "hailo10h": "orin_nx_hailo10_01",
        "deepx": "orin_nx_deepx_m1_01",
    }.get(str(backend), "")


def _resolved_setup_id(backend: str, remote_config: Mapping[str, Any] | None) -> str:
    """Use the configured physical setup identity with a legacy fallback."""
    configured = (
        str(remote_config.get("setup_id") or "").strip()
        if isinstance(remote_config, Mapping)
        else ""
    )
    return configured or _setup_id_for_backend(backend)


def _native_full_backends_for(
    native_backend: str,
    override: Any,
    by_producer: Mapping[str, Any],
) -> list[str]:
    aliases = {
        "trt": "tensorrt", "ort_tensorrt": "tensorrt",
        "hailo10": "hailo10h", "hailo_10": "hailo10h",
        "deepx_m1": "deepx", "dx_m1": "deepx",
    }
    source = by_producer.get(native_backend) if native_backend in by_producer else override
    defaults = {
        "hailo8": ["hailo8", "tensorrt"],
        "hailo10h": ["hailo10h", "tensorrt"],
        "deepx": ["deepx", "tensorrt"],
    }.get(native_backend, [])
    if not source:
        return list(defaults)
    requested: list[str] = []
    for raw in _parse_list(source):
        token = str(raw or "").strip().lower().replace("-", "_")
        token = aliases.get(token, token)
        if token in defaults and token not in requested:
            requested.append(token)
    return requested


def _parse_trt_quality_producer_sets(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        raw = value
    else:
        text = str(value or "").strip()
        if not text:
            return {}
        try:
            raw = json.loads(text)
        except Exception as exc:
            raise TensorRTQualityChainError(
                f"--trt-quality-producer-sets is not valid JSON: {exc}"
            ) from exc
    if not isinstance(raw, Mapping):
        raise TensorRTQualityChainError(
            "--trt-quality-producer-sets must be a setup-id to JSON-path object"
        )
    out: dict[str, str] = {}
    for key, value_path in raw.items():
        setup_id = str(key or "").strip()
        path_text = str(value_path or "").strip()
        if not setup_id or not path_text or setup_id in out:
            raise TensorRTQualityChainError(
                "TensorRT quality producer-set mapping contains an empty or duplicate entry"
            )
        out[setup_id] = str(Path(path_text).expanduser().resolve())
    return out


def _parse_native_split_quality_binding_sets(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        raw = value
    else:
        text = str(value or "").strip()
        if not text:
            return {}
        try:
            raw = json.loads(text)
        except Exception as exc:
            raise TensorRTQualityChainError(
                f"--native-split-quality-binding-sets is not valid JSON: {exc}"
            ) from exc
    if not isinstance(raw, Mapping):
        raise TensorRTQualityChainError(
            "--native-split-quality-binding-sets must be a setup-id to JSON-path object"
        )
    out: dict[str, str] = {}
    for key, value_path in raw.items():
        setup_id = str(key or "").strip()
        path_text = str(value_path or "").strip()
        if not setup_id or not path_text or setup_id in out:
            raise TensorRTQualityChainError(
                "Native split binding-set mapping contains an empty or duplicate entry"
            )
        out[setup_id] = str(Path(path_text).expanduser().resolve())
    return out


def _validate_trt_quality_producer_set(
    path: Path,
    *,
    eval_run_id: str,
    setup_id: str,
    model_ids: list[str],
) -> dict[str, Any]:
    """Validate exact coverage and producer identity before any remote run."""

    if not path.is_file():
        raise TensorRTQualityChainError(
            f"TensorRT quality producer set is missing: {path}"
        )
    payload = _strict_json(path)
    if not isinstance(payload, Mapping):
        raise TensorRTQualityChainError("TensorRT quality producer set root is not an object")
    payload = dict(payload)
    producers = payload.get("producers_by_model")
    if (
        payload.get("schema") != PRODUCER_SET_SCHEMA
        or int(payload.get("schema_version") or 0) != 1
        or str(payload.get("eval_run_id") or "") != eval_run_id
        or str(payload.get("setup_id") or "") != setup_id
        or not isinstance(producers, Mapping)
        or set(str(key) for key in producers) != set(model_ids)
    ):
        raise TensorRTQualityChainError(
            f"TensorRT quality producer set identity/model coverage mismatch: "
            f"setup={setup_id!r} models={model_ids!r}"
        )
    for model_id in model_ids:
        producer = producers.get(model_id)
        if not isinstance(producer, Mapping):
            raise TensorRTQualityChainError(
                f"TensorRT quality producer missing for model={model_id!r}"
            )
        try:
            validated, producer_sha = _validate_candidate_execution_contract(
                producer,
                role="Native Full TensorRT coordinator hand-off",
                task=str(producer.get("task") or "").strip().lower(),
            )
        except Exception as exc:
            raise TensorRTQualityChainError(
                f"TensorRT quality producer invalid for model={model_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        exact = {
            "eval_run_id": eval_run_id,
            "setup_id": setup_id,
            "model_id": model_id,
            "source_run_id": "native_full_tensorrt",
            "case_id": "full",
            "execution_role": "full_quality_only",
            "backend": "native_tensorrt",
            "variant": "full",
            "performance_claims_emitted": False,
        }
        if any(validated.get(key) != value for key, value in exact.items()):
            raise TensorRTQualityChainError(
                f"TensorRT quality producer role/identity mismatch for model={model_id!r}"
            )
        if str(validated.get("producer_identity_sha256") or "").lower() != str(producer_sha).lower():
            raise TensorRTQualityChainError(
                f"TensorRT quality producer SHA mismatch for model={model_id!r}"
            )
    return payload


def _split_backend_for_physical(backend: str) -> str:
    aliases = {
        "hailo8": "hailo8_to_trt",
        "hailo10h": "hailo10h_to_trt",
        "deepx": "deepx_to_trt",
    }
    try:
        return aliases[str(backend).strip().lower()]
    except KeyError as exc:
        raise TensorRTQualityChainError(
            f"unsupported Native split backend={backend!r}"
        ) from exc


def _benchmark_task_for_set(benchmark_set: Path, model_id: str) -> str:
    payload = _read_json(benchmark_set / "benchmark_set.json") or {}
    for value in (
        payload.get("benchmark_task"), payload.get("task"),
        payload.get("model_task"),
        (payload.get("model") or {}).get("task")
        if isinstance(payload.get("model"), Mapping) else "",
    ):
        task = str(value or "").strip().lower()
        if task in {"classification", "detection"}:
            return task
    model = str(model_id).strip().lower()
    if model.startswith("resnet"):
        return "classification"
    if model.startswith("yolo"):
        return "detection"
    raise TensorRTQualityChainError(
        f"benchmark task is missing for Native split model={model_id!r}"
    )


def _native_split_selections(
    *,
    benchmark_sets: Mapping[str, Path],
    case_map: Mapping[str, Any],
    backend: str,
    precision: str,
) -> list[dict[str, str]]:
    case_map_active = bool(case_map)
    selections: list[dict[str, str]] = []
    source_backend = _split_backend_for_physical(backend)
    unknown_models = sorted(set(str(key) for key in case_map).difference(benchmark_sets))
    if unknown_models:
        raise TensorRTQualityChainError(
            f"Native split case map contains unknown models: {unknown_models!r}"
        )
    for model_id, benchmark_set in sorted(benchmark_sets.items()):
        if case_map_active and model_id not in case_map:
            continue
        available = sorted(
            path.name for path in benchmark_set.iterdir()
            if path.is_dir() and path.name.startswith("b")
        )
        raw_cases = case_map.get(model_id) if case_map_active else available
        cases = [str(case).strip().lower() for case in (raw_cases or [])]
        if not cases or len(cases) != len(set(cases)):
            raise TensorRTQualityChainError(
                f"Native split cases missing or duplicated for model={model_id!r}"
            )
        if any(case not in set(available) for case in cases):
            raise TensorRTQualityChainError(
                f"Native split case map selects an unavailable case for "
                f"model={model_id!r}"
            )
        task = _benchmark_task_for_set(benchmark_set, model_id)
        for case_id in cases:
            selections.append({
                "model_id": model_id,
                "case_id": case_id,
                "backend": source_backend,
                "precision": str(precision).strip(),
                "task": task,
            })
    if not selections:
        raise TensorRTQualityChainError(
            f"Native split selection is empty for backend={backend!r}"
        )
    return selections


def _validate_native_split_quality_binding_set(
    path: Path,
    *,
    eval_run_id: str,
    setup_id: str,
    selections: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate exact set and binding identities before any remote command."""

    if not path.is_file():
        raise TensorRTQualityChainError(
            f"Native split quality binding set is missing: {path}"
        )
    payload = _strict_json(path)
    if not isinstance(payload, Mapping):
        raise TensorRTQualityChainError(
            "Native split quality binding set root is not an object"
        )
    payload = dict(payload)
    bindings = payload.get("bindings_by_model_case_backend")
    selection_by_key = {
        "|".join((
            str(row.get("model_id") or "").strip(),
            str(row.get("case_id") or "").strip().lower(),
            str(row.get("backend") or "").strip(),
        )): row
        for row in selections
    }
    if (
        payload.get("schema") != SPLIT_BINDING_SET_SCHEMA
        or int(payload.get("schema_version") or 0) != 2
        or str(payload.get("eval_run_id") or "") != eval_run_id
        or str(payload.get("setup_id") or "") != setup_id
        or not isinstance(bindings, Mapping)
        or set(str(key) for key in bindings) != set(selection_by_key)
    ):
        raise TensorRTQualityChainError(
            f"Native split binding-set identity/coverage mismatch: "
            f"setup={setup_id!r}"
        )
    declared_set_sha = str(payload.get("binding_set_sha256") or "").strip().lower()
    unhashed_set = dict(payload)
    unhashed_set.pop("binding_set_sha256", None)
    if (
        len(declared_set_sha) != 64
        or canonical_json_sha256(unhashed_set) != declared_set_sha
        or len(str(payload.get("central_quality_summary_sha256") or "").strip())
        != 64
    ):
        raise TensorRTQualityChainError(
            f"Native split binding-set seal mismatch: setup={setup_id!r}"
        )
    for key, selection in sorted(selection_by_key.items()):
        raw_binding = bindings.get(key)
        validated, status = validate_native_split_quality_binding(
            raw_binding,
            expected_identity={
                "model": selection.get("model_id"),
                "case": selection.get("case_id"),
                "backend": selection.get("backend"),
                "precision": selection.get("precision"),
                "task": selection.get("task"),
                "setup_id": setup_id,
            },
            verification_mode="portable",
        )
        if validated is None:
            raise TensorRTQualityChainError(
                f"Native split binding invalid for {key!r}: {status}"
            )
        central_selection, selection_status = (
            validate_central_native_split_quality_selection(
                validated, required=True,
            )
        )
        if central_selection is None:
            raise TensorRTQualityChainError(
                f"Native split Central selection invalid for {key!r}: "
                f"{selection_status}"
            )
        preselection = validated.get("preselection")
        if not isinstance(preselection, Mapping):
            raise TensorRTQualityChainError(
                f"Native split preselection missing for {key!r}"
            )
        exact = {
            "eval_run_id": (validated.get("eval_run_id"), eval_run_id),
            "source_run_id": (
                canonical_native_split_backend(
                    central_selection.get("source_run_id"), setup_id,
                ),
                canonical_native_split_backend(selection.get("backend"), setup_id),
            ),
            "setup_id": (preselection.get("setup_id"), setup_id),
        }
        drift = [name for name, values in exact.items() if values[0] != values[1]]
        if drift:
            raise TensorRTQualityChainError(
                f"Native split exact identity mismatch for {key!r}: {drift!r}"
            )
    return payload


def _stage_remote_trt_quality_producer_set(
    *,
    local_path: Path,
    ssh: str,
    remote_root: str,
    timeout: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Copy one verified set and byte-verify it on the physical setup."""

    remote_dir = f"{str(remote_root).rstrip('/')}/quality_first"
    remote_path = f"{remote_dir}/tensorrt_quality_producer_set.json"
    expected_sha = hashlib.sha256(local_path.read_bytes()).hexdigest()
    steps: list[dict[str, Any]] = []
    mkdir = _run(
        [
            "ssh", "-o", "BatchMode=yes", "-o",
            "StrictHostKeyChecking=accept-new", ssh,
            f"mkdir -p {_q(remote_dir)}",
        ],
        timeout=min(timeout, 120),
    )
    steps.append({"name": "mkdir_remote_trt_quality_producer_set", **mkdir})
    if mkdir.get("rc") != 0:
        raise TensorRTQualityChainError(
            f"remote TensorRT producer-set directory creation failed: {mkdir.get('stderr_tail')}"
        )
    copied = _run(
        ["rsync", "-a", "--checksum", str(local_path), f"{ssh}:{remote_path}"],
        timeout=timeout,
    )
    steps.append({
        "name": "sync_trt_quality_producer_set", **copied,
        "expected_sha256": expected_sha,
    })
    if copied.get("rc") != 0:
        raise TensorRTQualityChainError(
            f"remote TensorRT producer-set sync failed: {copied.get('stderr_tail')}"
        )
    verify_script = (
        "import hashlib,json,pathlib,sys;"
        f"p=pathlib.Path({remote_path!r});want={expected_sha!r};"
        "got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else '';"
        "print(json.dumps({'path':str(p),'expected':want,'sha256':got}));"
        "sys.exit(0 if got==want else 9)"
    )
    verified = _run(
        [
            "ssh", "-o", "BatchMode=yes", "-o",
            "StrictHostKeyChecking=accept-new", ssh,
            f"python -c {_q(verify_script)}",
        ],
        timeout=min(timeout, 120),
    )
    steps.append({"name": "verify_remote_trt_quality_producer_set", **verified})
    if verified.get("rc") != 0:
        raise TensorRTQualityChainError(
            "remote TensorRT producer-set SHA verification failed"
        )
    return remote_path, steps


def _stage_remote_native_split_quality_binding_set(
    *,
    local_path: Path,
    expected_sha256: str,
    ssh: str,
    remote_root: str,
    timeout: int,
    remote_filename: str = "native_split_quality_binding_set.json",
) -> tuple[str, list[dict[str, Any]]]:
    """Transfer the locally preflighted set and verify identical remote bytes."""

    observed_local_sha = _sha256_file(local_path) if local_path.is_file() else ""
    if not expected_sha256 or observed_local_sha != expected_sha256:
        raise TensorRTQualityChainError(
            "Native split binding set changed after local preflight"
        )
    remote_dir = f"{str(remote_root).rstrip('/')}/.quality_first"
    if remote_filename not in {"native_split_quality_binding_set.json", "vendor_full_quality_request_binding_set.json"}:
        raise ValueError("unrecognized quality binding role")
    if remote_filename == "vendor_full_quality_request_binding_set.json":
        remote_dir = f"{str(remote_root).rstrip('/')}/quality_first"
    remote_path = f"{remote_dir}/{remote_filename}"
    steps: list[dict[str, Any]] = []
    mkdir = _run([
        "ssh", "-o", "BatchMode=yes", "-o",
        "StrictHostKeyChecking=accept-new", ssh,
        f"mkdir -p {_q(remote_dir)}",
    ], timeout=min(timeout, 120))
    steps.append({"name": "mkdir_remote_native_split_quality_set", **mkdir})
    if mkdir.get("rc") != 0:
        raise TensorRTQualityChainError(
            "remote Native split binding-set directory creation failed"
        )
    copied = _run([
        "rsync", "-a", "--checksum", str(local_path), f"{ssh}:{remote_path}",
    ], timeout=min(timeout, 300))
    steps.append({
        "name": "sync_remote_native_split_quality_set", **copied,
        "local_sha256": expected_sha256,
    })
    if copied.get("rc") != 0:
        raise TensorRTQualityChainError(
            f"remote Native split binding-set sync failed: "
            f"{copied.get('stderr_tail')}"
        )
    verify_script = (
        "import hashlib,json,pathlib,sys;"
        f"p=pathlib.Path({remote_path!r});want={expected_sha256!r};"
        "got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else '';"
        "print(json.dumps({'path':str(p),'expected':want,'sha256':got}));"
        "sys.exit(0 if got==want else 9)"
    )
    verified = _run([
        "ssh", "-o", "BatchMode=yes", "-o",
        "StrictHostKeyChecking=accept-new", ssh,
        f"python -c {_q(verify_script)}",
    ], timeout=min(timeout, 120))
    steps.append({
        "name": "verify_remote_native_split_quality_set", **verified,
        "expected_sha256": expected_sha256,
    })
    if verified.get("rc") != 0:
        raise TensorRTQualityChainError(
            "remote Native split binding-set SHA verification failed"
        )
    return remote_path, steps


def _performance_repetitions(cfg: Mapping[str, Any]) -> int:
    """Return the independent performance-repeat count (never Energy runs)."""
    try:
        return max(1, int(cfg.get("repetitions") or 1))
    except (TypeError, ValueError):
        return 1


def _scaled_performance_timeout(timeout: int, repetitions: int) -> int:
    """Scale an outer command timeout while keeping the per-repeat budget."""
    return max(1, int(timeout)) * max(1, int(repetitions))


def _telemetry_label(value: Any) -> str:
    raw = str(value or "manual").strip() or "manual"
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return safe[:96] or "manual"


def _capture_remote_host_telemetry(
    *, ssh: str, env: str, remote_tool_dir: str, remote_root: str,
    backend: str, setup_id: str, run_id: str, phase: str, label: str,
    timeout: int,
) -> dict[str, Any]:
    """Best-effort remote capture; callers must never gate execution on it."""
    output = f"{str(remote_root).rstrip('/')}/host_telemetry/{_telemetry_label(label)}_{phase}.json"
    args = [
        "python", "-u", "scripts/native_host_telemetry.py",
        "--output", _q(output),
        "--phase", _q(phase),
        "--backend", _q(backend),
        "--setup-id", _q(setup_id),
        "--run-id", _q(run_id),
        "--capture-group", _q(_telemetry_label(label)),
    ]
    shell = " ".join(
        ([env, "&&"] if str(env).strip() else [])
        + ["cd", _q(remote_tool_dir), "&&", "mkdir", "-p", _q(str(Path(output).parent)), "&&"]
        + args
    )
    try:
        record = _run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, shell],
            timeout=min(max(30, int(timeout)), 180),
            label=f"host-telemetry-{phase}:{backend}",
        )
    except Exception as exc:  # pragma: no cover - defensive around transport layer
        record = {"rc": -1, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "name": f"capture_host_telemetry_{phase}",
        **record,
        "phase": phase,
        "backend": backend,
        "setup_id": setup_id,
        "capture_group": _telemetry_label(label),
        "remote_output": output,
        "non_blocking": True,
        "cmd_shell": shell,
    }


def _summarize_host_telemetry(run_dir: Path, reports: Path) -> dict[str, Any]:
    """Create the local cross-host pre/post summary without gating the run."""
    output = reports / "native_host_telemetry_summary.json"
    try:
        record = _run([
            sys.executable, "-u", str(_script("native_host_telemetry.py")),
            "--output", str(output),
            "--compare-root", str(run_dir / "native_producers"),
        ], timeout=180, cwd=ROOT, label="host-telemetry-summary")
    except Exception as exc:  # pragma: no cover - defensive around local process layer
        record = {"rc": -1, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "name": "summarize_host_telemetry",
        **record,
        "output": str(output),
        "non_blocking": True,
    }


def _collect_failure_host_telemetry(
    *, run_dir: Path, ssh: str, remote_root: str, backend: str, timeout: int,
    local_backend_root: Path | None = None,
) -> dict[str, Any]:
    """Best-effort recovery of pre/post evidence after a remote run failure."""
    backend_root = (
        local_backend_root
        if local_backend_root is not None
        else run_dir / "native_producers" / str(backend)
    )
    local_dir = backend_root / "host_telemetry"
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        record = _run(
            [
                "rsync", "-a",
                f"{ssh}:{str(remote_root).rstrip('/')}/host_telemetry/",
                str(local_dir) + "/",
            ],
            timeout=min(max(30, int(timeout)), 180),
            label=f"collect-failure-host-telemetry:{backend}",
        )
    except Exception as exc:  # pragma: no cover - defensive around transport layer
        record = {"rc": -1, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "name": "collect_failure_host_telemetry",
        **record,
        "backend": backend,
        "remote_root": str(remote_root),
        "local_directory": str(local_dir),
        "non_blocking": True,
    }


def _model_from_benchmark_path(run_dir: Path, p: Path) -> str:
    try:
        rel = p.relative_to(run_dir).parts
        if len(rel) >= 2 and rel[0] == "models":
            return rel[1]
    except Exception:
        pass
    # Common copied/nested cases.
    parts = p.parts
    if "models" in parts:
        i = parts.index("models")
        if i + 1 < len(parts):
            return parts[i + 1]
    if p.name == "legacy_suite" and p.parent.parent.name:
        return p.parent.parent.name
    if p.parent.name == "benchmark_set" and p.parent.parent.name:
        return p.parent.parent.name
    return p.name


def _has_native_case_dirs(p: Path) -> bool:
    """Return true if p looks like an executable legacy/native BenchmarkSet root.

    Some EvalRun debug/analysis trees contain a light-weight `benchmark_set/`
    directory with plans/tables only, while the actual runnable suite lives under
    `benchmark_results/remote_diagnostics/**/lean_bundle`.  Native producer
    staging must copy the runnable root, otherwise all downstream native smoke
    matrices find zero cases.
    """
    if not p.is_dir():
        return False
    for b in p.glob('b*'):
        if not b.is_dir():
            continue
        if (b / 'split_manifest.json').is_file():
            return True
        if any(b.glob('*_part1_b*.onnx')) or any(b.glob('*_part2_b*.onnx')):
            return True
        if (b / 'hailo').exists() or (b / 'deepx').exists() or (b / 'native_pipeline').exists():
            return True
    return False


def _find_benchmark_sets(run_dir: Path) -> dict[str, Path]:
    """Find the best runnable BenchmarkSet root per model.

    Priority:
      1. `benchmark_set/legacy_suite` if present;
      2. remote diagnostics `lean_bundle` roots with b*/ case dirs;
      3. any benchmark_set.json root with b*/ case dirs;
      4. model `benchmark_set` as a last resort.
    """
    candidates: list[Path] = []
    models = run_dir / "models"
    if models.is_dir():
        patterns = [
            "*/benchmark_set/legacy_suite",
            "*/benchmark_results/remote_diagnostics/lean_bundle",
            "*/benchmark_results/remote_diagnostics/*/lean_bundle",
            "*/benchmark_set",
        ]
        for pat in patterns:
            for p in models.glob(pat):
                if (p / "benchmark_set.json").is_file():
                    candidates.append(p)
    if (run_dir / "benchmark_set.json").is_file():
        candidates.append(run_dir)
    if not candidates:
        for js in run_dir.rglob("benchmark_set.json"):
            sp = str(js)
            if any(x in sp for x in ("/energy/", "/collector_storage/", "/processed/")):
                continue
            candidates.append(js.parent)

    # Deduplicate exact paths.
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp not in seen:
            seen.add(rp); uniq.append(p)

    def score(p: Path) -> tuple[int, int, int]:
        s = 0
        if p.name == "legacy_suite": s += 100
        if "remote_diagnostics" in str(p) and p.name == "lean_bundle": s += 80
        if _has_native_case_dirs(p): s += 1000
        if (p / "benchmark_suite.py").exists(): s += 20
        # Prefer shorter paths after feature scoring.
        return (s, -len(str(p)), -len(list(p.glob('b*'))))

    by_model: dict[str, Path] = {}
    for p in uniq:
        m = _model_from_benchmark_path(run_dir, p)
        old = by_model.get(m)
        if old is None or score(p) > score(old):
            by_model[m] = p
    return dict(sorted(by_model.items()))

def _parse_list(s: Any) -> list[str]:
    """Parse comma/semicolon separated backend lists robustly.

    v59br stored ``backends`` as a real list in native_producer_stage_config.json,
    then parsed that list again via ``str(list)`` which produced tokens such as
    ``['hailo8'`` and ``'deepx']``.  Accept lists/tuples/sets natively so
    update_evalset_native_producers.py can be re-run on existing EvalRuns.
    """
    if s is None:
        return []
    if isinstance(s, (list, tuple, set)):
        return [str(x).strip().strip("'\"[]") for x in s if str(x).strip()]
    if isinstance(s, str):
        raw = s.replace(";", ",")
        return [x.strip().strip("'\"[]") for x in raw.split(",") if x.strip().strip("'\"[]")]
    return [str(s).strip().strip("'\"[]")] if str(s).strip() else []


def _parse_case_map(s: str) -> dict[str, list[str]]:
    if not str(s or "").strip():
        return {}
    data = json.loads(s)
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
        else:
            out[str(k)] = [str(v)]
    return out


def _native_cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    remotes: dict[str, dict[str, str]] = {}
    if args.hailo8_ssh:
        remotes.setdefault("hailo8", {})["ssh"] = args.hailo8_ssh
    if args.hailo10_ssh:
        remotes.setdefault("hailo10h", {})["ssh"] = args.hailo10_ssh
    if args.deepx_ssh:
        remotes.setdefault("deepx", {})["ssh"] = args.deepx_ssh
    if args.hailo8_env:
        remotes.setdefault("hailo8", {})["env"] = args.hailo8_env
    if args.hailo10_env:
        remotes.setdefault("hailo10h", {})["env"] = args.hailo10_env
    if args.deepx_env:
        remotes.setdefault("deepx", {})["env"] = args.deepx_env
    for backend, attr in (
        ("hailo8", "hailo8_remote_base_dir"),
        ("hailo10h", "hailo10_remote_base_dir"),
        ("deepx", "deepx_remote_base_dir"),
    ):
        remote_base_dir = str(getattr(args, attr, "") or "").strip()
        if remote_base_dir:
            remotes.setdefault(backend, {})[
                "remote_base_dir"
            ] = remote_base_dir
    for backend, attr in (
        ("hailo8", "hailo8_setup_id"),
        ("hailo10h", "hailo10_setup_id"),
        ("deepx", "deepx_setup_id"),
    ):
        setup_id = str(getattr(args, attr, "") or "").strip()
        if setup_id:
            remotes.setdefault(backend, {})["setup_id"] = setup_id
    cfg: dict[str, Any] = {
        "enabled": True,
        "artifact_policy": str(
            getattr(args, "artifact_policy", "normal") or "normal"
        ).strip().lower().replace("-", "_"),
        "cache_verify_only": str(
            getattr(args, "artifact_policy", "normal") or "normal"
        ).strip().lower().replace("-", "_") == "cache_verify_only",
        "backends": _parse_list(args.backends),
        "case_policy": args.case_policy,
        "precision": args.precision,
        "frames": args.frames,
        "warmup": args.warmup,
        "repetitions": max(1, int(getattr(args, "repetitions", 1) or 1)),
        "queue_depth": args.queue_depth,
        "inflight": args.inflight,
        "hailo_format": args.hailo_format,
        "native_letterbox_pad_value": int(getattr(args, "native_letterbox_pad_value", 0) or 0),
        "native_force_rebuild_engines": bool(getattr(args, "native_force_rebuild_engines", False)),
        "native_dequant_scale": float(getattr(args, "native_dequant_scale", 0.0) or 0.0),
        "native_dequant_zero_point": float(getattr(args, "native_dequant_zero_point", 0.0) or 0.0),
        "native_boundary_layout": str(getattr(args, "native_boundary_layout", "as_input") or "as_input"),
        "remote_root": args.remote_root,
        "remote_tool_dir": args.remote_tool_dir,
        "dump_outputs": bool(args.dump_outputs),
        "dump_boundary": bool(getattr(args, "native_boundary_debug", False)),
        "build_missing_engines": not bool(args.no_build_missing_engines),
        "engine_build_python": args.engine_build_python,
        "copy_benchmarksets": not bool(args.no_copy),
        "strict_supported_only": True,
        "telemetry_label": _telemetry_label(getattr(args, "native_telemetry_label", "manual")),
        "central_quality_summary": str(getattr(args, "central_quality_summary", "") or ""),
        "vendor_full_quality_binding_sets_by_setup": json.loads(getattr(args, "vendor_full_quality_binding_sets", "") or "{}"),
        "artifact_namespace": _artifact_namespace(
            getattr(args, "artifact_namespace", "")
        ),
        "native_split_quality_required": bool(
            getattr(args, "native_split_quality_required", False)
        ),
        "native_split_quality_applicable": not bool(
            getattr(args, "native_split_quality_not_applicable", False)
        ),
        "native_split_quality_binding_sets_by_setup": (
            _parse_native_split_quality_binding_sets(
                getattr(args, "native_split_quality_binding_sets", "")
            )
        ),
        "smoke_diagnostic_quality_continue": bool(
            getattr(args, "smoke_diagnostic_quality_continue", False)
        ),
        "remotes": remotes,
    }
    raw_execution_contract = str(
        getattr(args, "native_execution_contract_json", "") or ""
    ).strip()
    if raw_execution_contract:
        try:
            parsed_execution_contract = json.loads(raw_execution_contract)
            if not isinstance(parsed_execution_contract, Mapping):
                raise ValueError("contract root is not an object")
            execution_contract = verify_native_execution_contract(
                parsed_execution_contract
            )
        except Exception as exc:
            raise TensorRTQualityChainError(
                f"native_execution_contract_invalid:{exc}"
            ) from exc
        for field in NATIVE_EXECUTION_FIELDS:
            if int(cfg[field]) != int(execution_contract[field]):
                raise TensorRTQualityChainError(
                    "native_execution_contract_cli_mismatch:"
                    f"{field}:{cfg[field]}!={execution_contract[field]}"
                )
    else:
        execution_contract = build_native_execution_contract(cfg)
    cfg["_native_execution_contract"] = execution_contract
    cfg["native_execution_contract_sha256"] = str(
        execution_contract["contract_sha256"]
    )
    raw_quality_gate = str(
        getattr(args, "quality_gate_json", "") or ""
    ).strip()
    if raw_quality_gate:
        try:
            parsed_quality_gate = json.loads(raw_quality_gate)
        except Exception as exc:
            raise TensorRTQualityChainError(
                f"quality_gate_json_invalid:{exc}"
            ) from exc
        if not isinstance(parsed_quality_gate, Mapping):
            raise TensorRTQualityChainError(
                "quality_gate_json_invalid:root_not_object"
            )
        cfg["quality_gate_policy"] = dict(parsed_quality_gate)
    if getattr(args, "native_full_baselines", False):
        cfg["full_baselines"] = {"enabled": True}
        native_full_backends = str(getattr(args, "native_full_backends", "") or "").strip()
        # "auto" means use per-native-backend defaults, not a literal backend named "auto".
        if native_full_backends and native_full_backends.lower() != "auto":
            cfg["full_baselines"]["backends"] = _parse_list(native_full_backends)
        raw_map = str(getattr(args, "native_full_backends_by_producer", "") or "").strip()
        if raw_map:
            try:
                parsed = json.loads(raw_map)
                if isinstance(parsed, Mapping):
                    cfg["full_baselines"]["backends_by_producer"] = {str(k): _parse_list(v) for k, v in parsed.items()}
            except Exception:
                pass
        cfg["trt_quality_producer_sets_by_setup"] = _parse_trt_quality_producer_sets(
            getattr(args, "trt_quality_producer_sets", "")
        )
    if getattr(args, "native_energy", False) or getattr(args, "native_energy_mode", None):
        mode = str(getattr(args, "native_energy_mode", "") or "plan").strip().lower()
        if mode not in {"plan", "measure"}:
            mode = "plan"
        energy_cfg = {
            "enabled": True,
            "mode": mode,
            "duration_source": "tool_config",
            "timeout": int(getattr(args, "native_energy_timeout", 0) or 900),
            "include_split_rows": True,
            "include_full_baselines": True,
            "repeat_override": max(1, int(getattr(args, "native_energy_runs", 1) or 1)),
            "allow_unpaired": bool(
                getattr(args, "native_energy_allow_unpaired", False)
                or getattr(args, "smoke_diagnostic_quality_continue", False)
            ),
        }
        dur = float(getattr(args, "native_energy_duration_s", 0.0) or 0.0)
        if dur > 0:
            energy_cfg["duration_s"] = dur
            energy_cfg["duration_source"] = "cli_override"
        cfg["energy"] = energy_cfg
    if getattr(args, "native_validation", False):
        cfg["validation"] = {
            "enabled": True,
            "mode": str(getattr(args, "native_validation_mode", "") or "dump_and_visual"),
            "topk": int(getattr(args, "native_validation_topk", 5) or 5),
        }
        cfg["dump_outputs"] = True
        # v59dz: native semantic validation relies on Full-ONNX self-reference
        # when possible.  Hailo8 can provide the exact native preprocessed input
        # via the boundary dump; request it automatically for validation runs
        # unless the user already set --native-boundary-debug explicitly.
        cfg["dump_boundary"] = True
    cm = _parse_case_map(args.case_map)
    if cm:
        cfg["case_map"] = cm
    return cfg


def _bind_evalrun_quality_gate_policy(
    run_dir: Path, cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Inherit and verify the EvalRun policy before any remote dispatch."""

    effective = dict(cfg)
    if bool(effective.get("cache_verify_only")):
        return effective
    profile = _load_yaml(run_dir / "profile.yaml")
    profile_policy = (
        dict(profile.get("quality_gate"))
        if isinstance(profile.get("quality_gate"), Mapping) else {}
    )
    configured = (
        dict(effective.get("quality_gate_policy"))
        if isinstance(effective.get("quality_gate_policy"), Mapping) else {}
    )
    validation = (
        dict(effective.get("validation"))
        if isinstance(effective.get("validation"), Mapping) else {}
    )
    required = bool(
        validation.get("enabled")
        or effective.get("native_split_quality_required")
    )
    if profile_policy and configured:
        if (
            AccuracyGatePolicy.from_mapping(profile_policy).sha256()
            != AccuracyGatePolicy.from_mapping(configured).sha256()
        ):
            raise TensorRTQualityChainError(
                "quality_gate_policy_drift_from_evalrun_profile"
            )
        effective["quality_gate_policy"] = profile_policy
        return effective
    if profile_policy:
        effective["quality_gate_policy"] = profile_policy
        return effective
    if configured or required:
        raise TensorRTQualityChainError(
            "evalrun_profile_quality_gate_missing"
        )
    return effective


def _refresh_suites(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, bs in _find_benchmark_sets(run_dir).items():
        script = _script("run_benchmark_suite_from_set.py")
        cmd = [sys.executable, str(script), "--benchmark-set", str(bs), "--refresh-only"]
        rec = {"model": model, "benchmark_set": str(bs)}
        try:
            rec.update(_run(cmd, timeout=300))
        except Exception as exc:
            rec.update({"rc": -1, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(rec)
    return rows


def _write_profile_config(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    p = run_dir / "profile.yaml"
    data = _load_yaml(p)
    old = data.get("native_producers") if isinstance(data, dict) else None
    data["native_producers"] = cfg
    _write_yaml(p, data)
    return {"profile": str(p), "old_native_producers": old, "new_native_producers": cfg}




def _extract_ref_image_from_report(report: Path) -> str:
    try:
        j=json.loads(report.read_text(encoding='utf-8'))
    except Exception:
        return ''
    for path in [
        ('run_cfg','image'), ('viz','image'), ('benchmark_input_policy','image'),
    ]:
        cur=j
        ok=True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur=cur[k]
            else:
                ok=False; break
        if ok and cur:
            return Path(str(cur)).name
    return ''

def _build_validation_image_map(run_dir: Path, models: list[str], case_map: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for m in models:
        for c in case_map.get(m, []):
            pats=[
                run_dir/'models'/m/'benchmark_results'/'remote_diagnostics'/'case_reports'/'results'/c/'results_ort_cpu'/'validation_report.json',
                run_dir/'models'/m/'benchmark_results'/'remote_diagnostics'/'lean_bundle'/c/'results_ort_cpu'/'validation_report.json',
            ]
            img=''
            for p in pats:
                if p.is_file():
                    img=_extract_ref_image_from_report(p)
                    if img: break
            if img:
                out.setdefault(m,{})[c]=img
    return out


def _reports_dir_for_cfg(run_dir: Path, cfg: Mapping[str, Any]) -> Path:
    namespace = _artifact_namespace(cfg.get("artifact_namespace"))
    if namespace:
        return run_dir / "reports" / "native_producer_variants" / namespace
    return run_dir / "reports"


def _local_backend_root(
    run_dir: Path, cfg: Mapping[str, Any], backend: str,
) -> Path:
    namespace = _artifact_namespace(cfg.get("artifact_namespace"))
    if namespace:
        return run_dir / "native_producers" / "variants" / namespace / backend
    return run_dir / "native_producers" / backend


def _run_native_producers(run_dir: Path, cfg: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    # Direct API users receive the same policy as the CLI, before directories,
    # remote transfers or compiler processes are created.
    for force_key in ("force_rebuild_engines", "native_force_rebuild_engines", "force_rebuild_native_engines"):
        value = cfg.get(force_key, False)
        if type(value) is not bool or value:
            raise TensorRTQualityChainError("productive_force_build_disabled:" + force_key)
    cache_verify_only = bool(cfg.get("cache_verify_only"))
    if cache_verify_only:
        if bool(cfg.get("build_missing_engines", True)):
            raise TensorRTQualityChainError(
                "cache_verify_only forbids build_missing_engines"
            )
        if bool(cfg.get("native_force_rebuild_engines", False)):
            raise TensorRTQualityChainError(
                "cache_verify_only forbids native_force_rebuild_engines"
            )
        os.environ["ONNX_SPLITPOINT_ARTIFACT_POLICY"] = "cache_verify_only"
    reports = _reports_dir_for_cfg(run_dir, cfg); reports.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    canonical_stage_path = run_dir / "reports" / "native_producer_stage.json"
    split_quality_authority = resolve_native_split_quality_authority(
        run_manifest_path=run_dir / "run_manifest.json",
        stage_path=canonical_stage_path,
    )
    bsets = _find_benchmark_sets(run_dir)
    case_policy = str(cfg.get("case_policy") or "all_accepted")
    case_map = cfg.get("case_map") if isinstance(cfg.get("case_map"), dict) else {}
    # For validation-aligned native dumps we need the exact reference image per model/case.
    # When case_policy=all_accepted, derive the cases from the copied BenchmarkSets.
    def _local_cases(bs: Path) -> list[str]:
        try:
            return sorted([p.name for p in bs.iterdir() if p.is_dir() and p.name.startswith('b')])
        except Exception:
            return []
    if not bsets:
        raise TensorRTQualityChainError(
            "no runnable BenchmarkSets were discovered for Native execution"
        )
    if case_policy == "case_map_only":
        if not case_map:
            raise TensorRTQualityChainError(
                "case_map_only requires a non-empty exact case map"
            )
        unknown_models = sorted(set(case_map).difference(bsets))
        if unknown_models:
            raise TensorRTQualityChainError(
                f"case map selects unknown BenchmarkSets: {unknown_models!r}"
            )
        effective_case_map: dict[str, list[str]] = {}
        for model_id, raw_cases in sorted(case_map.items()):
            cases = [str(case).strip() for case in list(raw_cases or []) if str(case).strip()]
            available = set(_local_cases(bsets[model_id]))
            if (not cases and cfg.get("native_split_quality_applicable", True)) or len(cases) != len(set(cases)):
                raise TensorRTQualityChainError(
                    f"case map contains empty/duplicate cases for model={model_id!r}"
                )
            missing = [case for case in cases if case not in available]
            if missing:
                raise TensorRTQualityChainError(
                    f"case map selects unavailable cases for model={model_id!r}: {missing!r}"
                )
            effective_case_map[model_id] = cases
        models = sorted(effective_case_map)
        validation_case_map = effective_case_map
    else:
        effective_case_map = {}
        models = sorted(bsets)
        validation_case_map = {
            model_id: _local_cases(benchmark_set)
            for model_id, benchmark_set in bsets.items()
        }
        empty_models = sorted(
            model_id for model_id, cases in validation_case_map.items()
            if not cases
        )
        if empty_models:
            raise TensorRTQualityChainError(
                f"Native case discovery is empty for models: {empty_models!r}"
            )
    # Discovery validates selection against the complete inventory; only the
    # exact authorized leaf models may enter transfer or remote dispatch.
    bsets = {model: bsets[model] for model in models}
    case_map_json = json.dumps(effective_case_map)
    vendor_sets = {}
    prepared_bindings = []
    for setup, raw_path in dict(cfg.get("vendor_full_quality_binding_sets_by_setup") or {}).items():
        from onnx_splitpoint_tool.workflow.runner import _native_full_quality_binding_set_v275
        path = Path(raw_path)
        supplied = _read_json(path) or {}
        summary = Path(str(cfg.get("central_quality_summary") or ""))
        expected, errors = _native_full_quality_binding_set_v275(summary, eval_run_id=run_id,
            setup_id=setup, producer=str(supplied.get("comparison_backend") or ""),
            full_backends=sorted({key.split("|")[0] for key in supplied.get("required_binding_keys") or []}),
            model_ids=models, case_release_run_root=run_dir if cfg.get("artifact_namespace", "").startswith("case") else None)
        if errors or expected != supplied or supplied.get("complete") is not True:
            raise TensorRTQualityChainError("vendor Full binding changed or incomplete:" + ";".join(errors))
        vendor_sets[setup] = path
        prepared_bindings.extend(supplied["bindings_by_backend_model"].values())
    from onnx_splitpoint_tool.workflow.native_transfer import build_native_validation_image_map
    validation_image_map, _ = build_native_validation_image_map(run_dir, models, validation_case_map,
        bsets, prepared_input_bindings=prepared_bindings)
    validation_image_map_json = json.dumps(validation_image_map)
    backends = _parse_list(cfg.get("backends") or ["hailo8"])
    remotes = cfg.get("remotes") if isinstance(cfg.get("remotes"), dict) else {}
    if cache_verify_only:
        profile = _load_yaml(run_dir / "profile.yaml")
        guard = profile.get("execution_guard") if isinstance(profile.get("execution_guard"), Mapping) else {}
        expected = guard.get("expected_plan") if isinstance(guard.get("expected_plan"), Mapping) else {}
        setup_ids = sorted({
            _resolved_setup_id(str(backend), remotes.get(str(backend), {}) if isinstance(remotes, Mapping) else {})
            for backend in backends
        })
        observed_case_map = {
            str(model): sorted(_local_cases(benchmark_set))
            for model, benchmark_set in sorted(bsets.items())
        }
        actual_core = {
            "models": sorted(bsets),
            "native_backends": backends,
            "native_case_map": observed_case_map,
            "hardware_setup_ids": setup_ids,
            "native_full_backends": (
                sorted(str(value) for value in list((cfg.get("full_baselines") or {}).get("backends") or []))
                if isinstance(cfg.get("full_baselines"), Mapping)
                and bool((cfg.get("full_baselines") or {}).get("enabled"))
                else []
            ),
            "native_energy_enabled": bool(
                isinstance(cfg.get("energy"), Mapping)
                and (cfg.get("energy") or {}).get("enabled")
            ),
        }
        mismatches = [
            field for field, value in actual_core.items()
            if value != expected.get(field)
        ]
        requested_case_map = {
            str(model): sorted(str(case) for case in list(cases or []))
            for model, cases in sorted(case_map.items())
        }
        if requested_case_map != observed_case_map:
            mismatches.append("requested_case_map_vs_generated_suites")
        if mismatches:
            raise TensorRTQualityChainError(
                "cache_verify_only exact matrix mismatch before remote dispatch: "
                + ", ".join(mismatches)
            )
    remote_root_base = str(cfg.get("remote_root") or "/home/nx/native_fifo_evalsets")
    remote_tool_dir_default = str(cfg.get("remote_tool_dir") or "/home/nx/ONNX-Splitpoint-Tool")
    precision = str(cfg.get("precision") or "uint8_cast_fp16")
    execution_contract = verify_native_execution_contract(
        cfg.get("_native_execution_contract")
        if isinstance(cfg.get("_native_execution_contract"), Mapping)
        else build_native_execution_contract(cfg)
    )
    frames = int(execution_contract["frames"])
    warmup = int(execution_contract["warmup"])
    repetitions = int(execution_contract["repetitions"])
    performance_timeout = _scaled_performance_timeout(timeout, repetitions)
    queue_depth = int(execution_contract["queue_depth"])
    inflight = int(execution_contract["inflight"])
    hailo_format = str(cfg.get("hailo_format") or "uint8")
    native_letterbox_pad_value = int(cfg.get("native_letterbox_pad_value") or 0)
    native_force_rebuild_requested = bool(
        cfg.get("native_force_rebuild_engines", False)
    )
    native_dequant_scale = float(cfg.get("native_dequant_scale") or 0.0)
    native_dequant_zero_point = float(cfg.get("native_dequant_zero_point") or 0.0)
    native_boundary_layout = str(cfg.get("native_boundary_layout") or "as_input")
    build_missing_requested = bool(cfg.get("build_missing_engines", True))
    engine_build_python = str(cfg.get("engine_build_python") or "auto")
    copy_sets = bool(cfg.get("copy_benchmarksets", True))
    dump_outputs = bool(cfg.get("dump_outputs", False))
    dump_boundary = bool(cfg.get("dump_boundary", False))
    full_cfg = cfg.get("full_baselines") if isinstance(cfg.get("full_baselines"), dict) else {}
    native_full_enabled = bool(full_cfg.get("enabled") or cfg.get("native_full_baselines_enabled"))
    native_full_backend_override = full_cfg.get("backends") if isinstance(full_cfg, dict) else None
    native_full_by_producer = full_cfg.get("backends_by_producer") if isinstance(full_cfg, dict) and isinstance(full_cfg.get("backends_by_producer"), Mapping) else {}
    trt_quality_sets = (
        cfg.get("trt_quality_producer_sets_by_setup")
        if isinstance(cfg.get("trt_quality_producer_sets_by_setup"), Mapping)
        else {}
    )
    split_quality_sets = (
        cfg.get("native_split_quality_binding_sets_by_setup")
        if isinstance(cfg.get("native_split_quality_binding_sets_by_setup"), Mapping)
        else {}
    )
    smoke_diagnostic = bool(cfg.get("smoke_diagnostic_quality_continue"))
    split_quality_applicable = bool(
        cfg.get("native_split_quality_applicable", True)
    )
    authority_requires_split_quality = bool(
        not cache_verify_only
        and native_split_quality_required_for_row(
            {"backend": "hailo8_to_trt"}, split_quality_authority,
        )
    )
    split_quality_required = bool(
        not cache_verify_only
        and split_quality_applicable
        and (
            cfg.get("native_split_quality_required")
            or split_quality_sets
            or authority_requires_split_quality
        )
    )
    build_missing = bool(build_missing_requested and not split_quality_required)
    native_force_rebuild_engines = bool(
        native_force_rebuild_requested and not split_quality_required
    )
    artifact_namespace = _artifact_namespace(cfg.get("artifact_namespace"))
    stage: dict[str, Any] = {
        "schema": "onnx-splitpoint/native-producer-stage",
        "schema_version": 2,
        "enabled": True,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id,
        "workflow_version": str(
            split_quality_authority.get("workflow_version") or ""
        ),
        "tool_version": str(split_quality_authority.get("tool_version") or ""),
        "profile_start_snapshot_sha256": str(
            split_quality_authority.get("profile_start_snapshot_sha256") or ""
        ),
        "profile_selection_snapshot_sha256": str(
            split_quality_authority.get("profile_selection_snapshot_sha256") or ""
        ),
        "profile_selection_fingerprint": str(
            split_quality_authority.get("profile_selection_fingerprint") or ""
        ),
        "artifact_namespace": artifact_namespace,
        "models": models,
        "case_policy": case_policy,
        "case_map": case_map,
        "case_map_effective": effective_case_map,
        "validation_image_map": validation_image_map,
        "strict_supported_only": True,
        "performance_repetitions": repetitions,
        "performance_timeout_s": performance_timeout,
        "energy_repetitions_independent": True,
        "native_full_baselines": {
            "enabled": native_full_enabled,
            "backends": (native_full_backend_override or "auto"),
            "backends_by_producer": dict(native_full_by_producer or {}),
            "quality_first_producer_sets_by_setup": dict(trt_quality_sets or {}),
        },
        "native_split_quality_first": {
            "required": authority_requires_split_quality,
            "applicable": split_quality_applicable,
            "status": (
                "not_applicable_cache_verify_only"
                if cache_verify_only else
                "not_applicable_full_only"
                if not split_quality_applicable else
                "required" if split_quality_required else "not_requested"
            ),
            "binding_sets_by_setup": dict(split_quality_sets or {}),
            "engine_rebuild_allowed": False if split_quality_required else None,
            "failure_policy": (
                "partial_continue_diagnostic" if smoke_diagnostic else "hard_fail"
            ),
        },
        "diagnostic_only": smoke_diagnostic,
        "claim_eligible": False if smoke_diagnostic else None,
        "artifact_lifecycle": {
            "remote_root_scope": "variant_isolated" if artifact_namespace else "evalrun",
            "local_root_scope": "variant_isolated_append_only" if artifact_namespace else "backend",
            "cleanup": "deferred_to_parent_after_combined_semantics_and_energy"
            if artifact_namespace else "manual_owner",
        },
        "config": cfg,
        "backend_results": [],
    }
    stage["native_split_quality_first"].update({
        "build_missing_requested": build_missing_requested,
        "force_rebuild_requested": native_force_rebuild_requested,
        "build_or_force_flags_forwarded": False if split_quality_required else None,
    })

    # Publish the run-bound policy before Final/Validation/Energy (and before
    # the first remote command).  For a variant namespace the coordinator owns
    # the canonical checkpoint; the updater writes only its isolated stage.
    checkpoint_path = (
        reports / "native_producer_stage.json"
        if artifact_namespace else canonical_stage_path
    )
    _write_json(checkpoint_path, stage)
    split_quality_authority = resolve_native_split_quality_authority(
        run_manifest_path=run_dir / "run_manifest.json",
        stage_path=canonical_stage_path,
    )
    stage["native_split_quality_authority"] = split_quality_authority
    split_quality_authority_invalid = bool(
        split_quality_applicable
        and split_quality_required
        and split_quality_authority.get("valid") is not True
    )
    if (
        split_quality_authority_invalid
    ):
        authority_error = ";".join(
            str(value)
            for value in list(split_quality_authority.get("errors") or [])
        ) or "native_split_quality_authority_invalid"
        if smoke_diagnostic:
            stage.setdefault("warnings", []).append(
                "Central Native split authority is unavailable; split transfers "
                "are blocked and setup-local Full diagnostics continue"
            )
            stage["upstream_quality_error"] = {
                "failure_class": "upstream_quality_evidence",
                "failure_reason": "upstream_central_quality_binding_missing",
                "upstream_stage": "central_quality",
                "transfer_attempted": False,
                "error": authority_error,
            }
        else:
            stage.update({
                "status": "failed",
                "failure_class": "upstream_quality_evidence",
                "failure_reason": "upstream_central_quality_binding_missing",
                "upstream_stage": "central_quality",
                "transfer_attempted": False,
                "preflight_errors_by_backend": {"authority": authority_error},
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            _write_json(checkpoint_path, stage)
            return stage

    # Validate every identity-bearing set before the first SSH/rsync command.
    # A bad second host must not leave a partially executed first host behind.
    preflight_by_backend: dict[str, dict[str, Any]] = {}
    preflight_errors: dict[str, str] = {}
    canonical_backends: list[str] = []
    for raw_backend in backends:
        backend = str(raw_backend).lower().strip()
        if backend in {
            "hailo10", "hailo10_to_trt", "hailo10h_to_trt",
            "hailo10_to_tensorrt", "hailo10h_to_tensorrt",
        }:
            backend = "hailo10h"
        if backend in {"hailo8_to_trt", "hailo8_to_tensorrt"}:
            backend = "hailo8"
        if backend in {
            "deepx_m1", "deepx_to_trt", "deepx_to_tensorrt",
            "deepx_m1_to_trt", "deepx_m1_to_tensorrt",
        }:
            backend = "deepx"
        if not backend or backend in canonical_backends:
            preflight_errors[backend or "<empty>"] = (
                "empty or duplicate physical backend alias"
            )
            continue
        canonical_backends.append(backend)
        remote_cfg = remotes.get(backend, {}) if isinstance(remotes, dict) else {}
        if not isinstance(remote_cfg, Mapping):
            remote_cfg = {}
        setup_id = _resolved_setup_id(backend, remote_cfg)
        full_backends = (
            _native_full_backends_for(
                backend, native_full_backend_override, native_full_by_producer,
            )
            if native_full_enabled else []
        )
        record: dict[str, Any] = {
            "setup_id": setup_id,
            "full_backends": full_backends,
            "split_quality_required": False,
            "split_execution_allowed": bool(
                split_quality_applicable and not split_quality_required
            ),
            "split_quality_applicable": split_quality_applicable,
            "diagnostic_skip_native_split": False,
        }
        split_error = ""
        trt_error = ""
        if split_quality_required:
            try:
                if split_quality_authority_invalid:
                    raise TensorRTQualityChainError(
                        "central Native split quality authority is invalid"
                    )
                if native_force_rebuild_requested:
                    raise TensorRTQualityChainError(
                        "Quality-FIRST Native split forbids force-rebuild flags"
                    )
                configured = str(split_quality_sets.get(setup_id) or "").strip()
                if not configured:
                    raise TensorRTQualityChainError(
                        f"Quality-FIRST Native split requested without central "
                        f"binding set for setup={setup_id!r}"
                    )
                split_path = Path(configured).expanduser().resolve()
                selections = _native_split_selections(
                    benchmark_sets=bsets,
                    case_map=effective_case_map,
                    backend=backend,
                    precision=precision,
                )
                _validate_native_split_quality_binding_set(
                    split_path,
                    eval_run_id=run_id,
                    setup_id=setup_id,
                    selections=selections,
                )
                record["split_quality"] = {
                    "local_path": str(split_path),
                    "sha256": _sha256_file(split_path),
                    "selection_count": len(selections),
                    "selections": selections,
                }
                record["split_quality_required"] = True
                record["split_execution_allowed"] = True
            except Exception as exc:
                split_error = f"{type(exc).__name__}: {exc}"
                if smoke_diagnostic:
                    record["diagnostic_skip_native_split"] = True
                    record["split_quality_error"] = split_error
                else:
                    preflight_errors[backend] = split_error
        if "tensorrt" in full_backends:
            try:
                configured = str(trt_quality_sets.get(setup_id) or "").strip()
                if not configured:
                    raise TensorRTQualityChainError(
                        f"Native Full TensorRT requested without central-quality "
                        f"producer set for setup={setup_id!r}"
                    )
                trt_path = Path(configured).expanduser().resolve()
                _validate_trt_quality_producer_set(
                    trt_path,
                    eval_run_id=run_id,
                    setup_id=setup_id,
                    model_ids=models,
                )
                record["trt_quality"] = {
                    "local_path": str(trt_path),
                    "sha256": _sha256_file(trt_path),
                }
            except Exception as exc:
                trt_error = f"{type(exc).__name__}: {exc}"
                if smoke_diagnostic:
                    record["full_backends"] = [
                        value for value in full_backends if value != "tensorrt"
                    ]
                    record["trt_quality_error"] = trt_error
                else:
                    existing = preflight_errors.get(backend)
                    preflight_errors[backend] = "; ".join(
                        value for value in (existing, trt_error) if value
                    )
        if split_error or trt_error:
            record["upstream_quality_error"] = {
                "failure_class": "upstream_quality_evidence",
                "failure_reason": "upstream_central_quality_binding_missing",
                "upstream_stage": "central_quality",
                "transfer_attempted": False,
                "split_error": split_error,
                "tensorrt_full_error": trt_error,
            }
        preflight_by_backend[backend] = record

    if preflight_errors:
        for backend in canonical_backends or list(preflight_errors):
            stage["backend_results"].append({
                "backend": backend,
                "setup_id": str(
                    (preflight_by_backend.get(backend) or {}).get("setup_id")
                    or _resolved_setup_id(
                        backend,
                        remotes.get(backend, {}) if isinstance(remotes, dict) else {},
                    )
                ),
                "ok": False,
                "status": "failed",
                "failure_class": "upstream_quality_evidence",
                "failure_reason": "upstream_central_quality_binding_missing",
                "upstream_stage": "central_quality",
                "transfer_attempted": False,
                "claim_eligible": False,
                "error": preflight_errors.get(backend)
                or "another physical setup failed the all-host preflight",
                "steps": [],
            })
        stage["status"] = "failed"
        stage["failure_class"] = "upstream_quality_evidence"
        stage["failure_reason"] = "upstream_central_quality_binding_missing"
        stage["upstream_stage"] = "central_quality"
        stage["transfer_attempted"] = False
        stage["preflight_errors_by_backend"] = preflight_errors
        stage["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(reports / "native_producer_stage.json", stage)
        return stage

    collected: list[Path] = []
    seen_physical_backends: set[str] = set()
    for backend in backends:
        b = backend.lower().strip()
        if b in {"hailo10", "hailo10_to_trt", "hailo10h_to_trt", "hailo10_to_tensorrt", "hailo10h_to_tensorrt"}: b = "hailo10h"
        if b in {"hailo8_to_trt", "hailo8_to_tensorrt"}: b = "hailo8"
        if b in {"deepx_m1", "deepx_to_trt", "deepx_to_tensorrt", "deepx_m1_to_trt", "deepx_m1_to_tensorrt"}: b = "deepx"
        if not b or b in seen_physical_backends:
            stage["backend_results"].append({
                "backend": b,
                "ok": False,
                "status": "partial",
                "failure_reason": "duplicate_physical_backend_alias",
                "error": "empty or duplicate physical backend alias",
                "steps": [],
            })
            continue
        seen_physical_backends.add(b)
        rcfg = remotes.get(b, {}) if isinstance(remotes, dict) else {}
        if not isinstance(rcfg, Mapping):
            rcfg = {}
        ssh = str(rcfg.get("ssh") or "")
        env = str(rcfg.get("env") or "")
        rtool = str(rcfg.get("remote_tool_dir") or remote_tool_dir_default)
        rroot = str(
            rcfg.get("remote_root")
            or f"{remote_root_base.rstrip('/')}/{run_id}"
        ).rstrip("/")
        if artifact_namespace:
            rroot = f"{rroot}/variants/{artifact_namespace}"
            if artifact_namespace.startswith("case"):
                rroot += "/" + run_id
        setup_id = _resolved_setup_id(b, rcfg)
        full_backends = (
            _native_full_backends_for(
                b, native_full_backend_override, native_full_by_producer,
            )
            if native_full_enabled else []
        )
        result: dict[str, Any] = {
            "backend": b,
            "ssh": ssh,
            "setup_id": setup_id,
            "remote_root": rroot,
            "remote_tool_dir": rtool,
            "native_full_backends": full_backends,
            "artifact_namespace": artifact_namespace,
            "steps": [],
            "ok": False,
            "transfer_attempted": False,
            "diagnostic_only": smoke_diagnostic,
            "claim_eligible": False if smoke_diagnostic else None,
        }
        telemetry_pre_attempted = False
        telemetry_post_attempted = False
        failure_telemetry_recovered = False
        split_runner_failed = False
        if not ssh:
            result.update({"status": "skipped", "error": "missing remote ssh"}); stage["backend_results"].append(result); continue
        try:
            preflight = preflight_by_backend.get(b) or {}
            full_backends = list(preflight.get("full_backends") or full_backends)
            result["native_full_backends"] = full_backends
            backend_split_quality_required = bool(
                preflight.get("split_quality_required")
            )
            backend_split_execution_allowed = bool(
                preflight.get("split_execution_allowed")
            )
            diagnostic_skip_native_split = bool(
                preflight.get("diagnostic_skip_native_split")
            )
            upstream_quality_error = preflight.get("upstream_quality_error")
            if isinstance(upstream_quality_error, Mapping):
                result["upstream_quality_error"] = dict(upstream_quality_error)
                result.update({
                    "failure_class": "upstream_quality_evidence",
                    "failure_reason": "upstream_central_quality_binding_missing",
                    "upstream_stage": "central_quality",
                })
                result.setdefault("warnings", []).append(
                    "Quality-FIRST preflight blocked the unbound Native path; "
                    "remaining setup-local Full diagnostics continue"
                )
            if diagnostic_skip_native_split and not full_backends:
                result.update({
                    "status": "partial",
                    "error": str(preflight.get("split_quality_error") or (
                        "Native split blocked by missing Central binding"
                    )),
                })
                stage["backend_results"].append(result)
                continue
            split_quality = (
                preflight.get("split_quality")
                if isinstance(preflight.get("split_quality"), Mapping) else {}
            )
            split_quality_set_local = (
                Path(str(split_quality.get("local_path"))).expanduser().resolve()
                if split_quality else None
            )
            split_quality_set_sha = str(split_quality.get("sha256") or "")
            split_quality_set_remote = ""
            trt_quality_set_local: Path | None = None
            trt_quality_set_remote = ""
            vendor_quality_set_remote = ""
            vendor_quality_set_local = vendor_sets.get(setup_id)
            if "tensorrt" in full_backends:
                trt_quality = (
                    preflight.get("trt_quality")
                    if isinstance(preflight.get("trt_quality"), Mapping) else {}
                )
                trt_quality_set_local = Path(
                    str(trt_quality.get("local_path") or "")
                ).expanduser().resolve()
                result["trt_quality_producer_set"] = {
                    "local_path": str(trt_quality_set_local),
                    "sha256": str(trt_quality.get("sha256") or ""),
                    "model_count": len(models),
                    "status": "verified_local",
                }
            if split_quality_set_local is not None:
                result["native_split_quality_binding_set"] = {
                    "local_path": str(split_quality_set_local),
                    "sha256": split_quality_set_sha,
                    "selection_count": int(
                        split_quality.get("selection_count") or 0
                    ),
                    "status": "verified_local",
                }
            result["transfer_attempted"] = True
            if b == "hailo8":
                sync_name, sync_tokens = "native_fifo_eval_runner.py", (
                    "--case-map", "--image-map", "--dump-boundary",
                    "--precision", "--repetitions", "--setup-id",
                    "--mixed-runtime-python",
                    "--native-split-quality-binding-set",
                )
            else:
                sync_name, sync_tokens = "native_producer_e2e_eval_runner.py", (
                    "--backend", "--case-map", "--image-map",
                    "--dump-boundary", "--repetitions", "--setup-id",
                    "--native-split-quality-binding-set",
                )
            result["steps"].extend(_sync_remote_script_v60i(ssh, rtool, sync_name, sync_tokens, timeout=min(timeout, 600)))
            result["steps"].extend(_sync_remote_script_v60i(
                ssh, rtool, "native_host_telemetry.py",
                ("--phase", "--capture-group", "--compare-root", "nvpmodel", "tegrastats"),
                timeout=min(timeout, 600),
            ))
            if b == "hailo8":
                for dependency_name, dependency_tokens in (
                    ("native_fifo_smoke_matrix.py", ("native_fifo_capability_report_nonzero_exit", "--native-split-quality-binding-set", "--setup-id", "--repetitions", "--mixed-runtime-python", "SPLITPOINT_EXTRA_SITES")),
                    ("native_fifo_capability_report.py", ("--benchmark-set", "native_fifo_capability_report.json")),
                    ("validate_output_dumps.py", ("--candidate", "output_dump_validation.json")),
                    ("native_hailo_trt_fifo_from_benchmarkset.py", ("--result-json", "--native-split-quality-binding", "--expected-runner-sha256", "native_command_contract", "--repetitions")),
                    ("native_hailo10_trt_e2e_from_benchmarkset.py", ("class NativeTRT", "--expected-runner-sha256")),
                    ("native_trt_from_benchmarkset.py", ("--benchmark-set", "--precision", "--duration")),
                    (
                        "materialize_cache_verify_native_split_binding.py",
                        (
                            "--engine-cache-root",
                            "cache_verify_native_split_binding_ready",
                            "compiler_dispatched",
                        ),
                    ),
                ):
                    result["steps"].extend(_sync_remote_script_v60i(
                        ssh, rtool, dependency_name, dependency_tokens,
                        timeout=min(timeout, 600),
                    ))
            elif b == "hailo10h":
                result["steps"].extend(_sync_remote_script_v60i(
                    ssh, rtool, "native_hailo10_trt_e2e_from_benchmarkset.py",
                    ("--out-dir", "--native-split-quality-binding", "--expected-runner-sha256", "native_command_contract", "--repetitions"),
                    timeout=min(timeout, 600),
                ))
            elif b == "deepx":
                # DeepX imports NativeTRT from the Hailo10 sibling, so both
                # exact source files are part of its staged runtime closure.
                for dependency_name, dependency_tokens in (
                    ("native_deepx_trt_e2e_from_benchmarkset.py", ("--out-dir", "--native-split-quality-binding", "--expected-dxnn-sha256", "native_command_contract", "--repetitions")),
                    ("native_hailo10_trt_e2e_from_benchmarkset.py", ("class NativeTRT", "--expected-runner-sha256")),
                ):
                    result["steps"].extend(_sync_remote_script_v60i(
                        ssh, rtool, dependency_name, dependency_tokens,
                        timeout=min(timeout, 600),
                    ))
            pending_module_checks = []
            for asset_relative, module_name, asset_tokens in native_remote_package_closure():
                module_steps = _sync_remote_package_asset_v263(
                    ssh, rtool, asset_relative, asset_tokens,
                    timeout=min(timeout, 600),
                )
                result["steps"].extend(module_steps)
                module_hash = hashlib.sha256((ROOT / asset_relative).read_bytes()).hexdigest()
                pending_module_checks.append((module_name, asset_relative, module_hash))
            # Import only after the entire package inventory passed transfer.
            for module_name, asset_relative, module_hash in pending_module_checks:
                result["steps"].append(_verify_remote_module_binding_v263(
                    ssh, rtool, env, module_name, asset_relative, module_hash,
                    timeout=min(timeout, 180),
                ))
            result["steps"].extend(_sync_remote_script_v60i(ssh, rtool, "run_and_report_work_units.py", ("__SPLITPOINT_WORK_UNITS__",), timeout=min(timeout, 600)))
            result["steps"].extend(_sync_remote_script_v60i(
                ssh, rtool, "native_yolo_full_self_reference_probe.py",
                ("--benchmark-set", "_input_dump_feed_from_manifest", "native_fifo_boundary"),
                timeout=min(timeout, 600),
            ))
            # The self-reference probe imports these siblings at process
            # start. Synchronise the exact matching implementations instead of
            # relying on an arbitrary remote checkout version.
            result["steps"].extend(_sync_remote_script_v60i(
                ssh, rtool, "native_producer_validate_visualize.py",
                ("_decode_layout_candidates", "_expected_detection_contract", "_match_detections"),
                timeout=min(timeout, 600),
            ))
            result["steps"].extend(_sync_remote_script_v60i(
                ssh, rtool, "validate_output_dumps.py",
                ("def load_dump", "def summarize"),
                timeout=min(timeout, 600),
            ))
            result["steps"].extend(_sync_remote_script_v60i(
                ssh, rtool, "validate_classification_output_dump.py",
                ("def _topk", "def _select_logits"),
                timeout=min(timeout, 600),
            ))
            if native_full_enabled:
                result["steps"].extend(_sync_remote_script_v60i(ssh, rtool, "native_progress.py", ("stream_command",), timeout=min(timeout, 600)))
                result["steps"].extend(_sync_remote_script_v60i(ssh, rtool, "native_full_baseline_eval_runner.py", ("--setup-id", "--comparison-backend", "--comparison-precision", "--repetitions", "--trt-quality-producer-json"), timeout=min(timeout, 600)))
                for helper_name, helper_tokens in (
                    ("smoke_hailo10_full_from_benchmarkset.py", ("--benchmark-set", "--dump-outputs", "hailo8")),
                    ("smoke_hailo10_hef_runner.py", ("--dump-outputs", "native_full_outputs_manifest.json")),
                    ("native_full_semantic_dump.py", ("--backend", "native_full_outputs_manifest.json")),
                    ("native_hailo10_trt_e2e_from_benchmarkset.py", ("class NativeTRT",)),
                    ("native_trt_full_completed_hotloop.py", ("tensorrt_full_completed_task_hotloop", "--frozen-postprocess-contract-json")),
                ):
                    result["steps"].extend(_sync_remote_script_v60i(ssh, rtool, helper_name, helper_tokens, timeout=min(timeout, 600)))
                if b == "deepx":
                    result["steps"].extend(_sync_remote_script_v60i(
                        ssh, rtool, "native_deepx_full_energy_hotloop.py",
                        ("--expected-runner-sha256", "--expected-dxnn-sha256", "deepx_full_energy_hotloop_failed"),
                        timeout=min(timeout, 600),
                    ))
            if copy_sets:
                transfer_dir = reports / 'native_transfer'
                transfer_dir.mkdir(parents=True, exist_ok=True)
                for m, bs in bsets.items():
                    remote_bs = f"{rroot}/{m}/benchmark_set"
                    inventory = build_native_transfer_inventory(bs)
                    files_from = write_rsync_files_from(inventory, transfer_dir / f'{b}_{m}_native_transfer_files.txt')
                    _write_json(transfer_dir / f'{b}_{m}_native_transfer_manifest.json', {
                        **inventory, 'backend': b, 'model': m, 'remote': ssh,
                        'remote_benchmark_set': remote_bs, 'transfer_policy': 'lean_native_files_from_v60y',
                    })
                    required = required_remote_bytes(int(inventory.get('total_bytes') or 0))
                    preflight_shell = (
                        f"mkdir -p {_q(rroot)}; printf '%s\\n' {_q(run_id)} > {_q(rroot + '/.onnx_splitpoint_native_evalset')}; "
                        f"old=$(du -sk {_q(remote_bs)} 2>/dev/null | awk '{{print $1}}'); old=${{old:-0}}; "
                        f"avail=$(df -Pk {_q(str(Path(rroot).parent))} | tail -1 | awk '{{print $4}}'); "
                        f"printf 'available_kb=%s\\nexisting_target_kb=%s\\n' \"$avail\" \"$old\""
                    )
                    pre = _run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, preflight_shell], timeout=120)
                    avail = old = 0
                    for line in str(pre.get('stdout_tail') or '').splitlines():
                        if line.startswith('available_kb='):
                            try: avail = int(line.split('=',1)[1]) * 1024
                            except Exception: avail = 0
                        elif line.startswith('existing_target_kb='):
                            try: old = int(line.split('=',1)[1]) * 1024
                            except Exception: old = 0
                    result['steps'].append({'name': f'remote_storage_preflight_{m}', **pre, 'required_bytes': required, 'available_bytes': avail, 'existing_target_bytes': old})
                    if pre['rc'] != 0:
                        raise RuntimeError(f"remote storage preflight failed for {m}: {pre.get('stderr_tail')}")
                    if avail and avail + old < required:
                        result['failure_reason'] = 'remote_disk_insufficient'
                        raise RuntimeError(f'remote_disk_insufficient backend={b} model={m} required_bytes={required} available_after_cleanup_bytes={avail+old}')
                    reset = _run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, f"rm -rf {_q(remote_bs)} && mkdir -p {_q(remote_bs)}"], timeout=120)
                    result['steps'].append({'name': f'reset_remote_benchmark_set_{m}', **reset})
                    if reset['rc'] != 0:
                        raise RuntimeError(f"remote reset failed for {m}: {reset.get('stderr_tail')}")
                    rsync = ["rsync", "-a", "--files-from", str(files_from), "--relative", str(bs) + "/", f"{ssh}:{remote_bs}/"]
                    rr = _run(rsync, timeout=timeout)
                    rr['failure_reason'] = classify_native_transfer_failure(rr.get('stderr_tail',''), rr.get('stdout_tail','')) if rr['rc'] != 0 else ''
                    result["steps"].append({"name": f"rsync_{m}", **rr, 'file_count': inventory.get('file_count'), 'payload_bytes': inventory.get('total_bytes')})
                    if rr["rc"] != 0:
                        result['failure_reason'] = rr['failure_reason']
                        raise RuntimeError(f"{rr['failure_reason']}: rsync failed for {m}: {rr.get('stderr_tail')}")
            if vendor_quality_set_local is not None:
                vendor_quality_set_remote, vendor_steps = _stage_remote_native_split_quality_binding_set(
                    local_path=vendor_quality_set_local, expected_sha256=_sha256_file(vendor_quality_set_local),
                    ssh=ssh, remote_root=rroot, timeout=timeout,
                    remote_filename="vendor_full_quality_request_binding_set.json")
                result["steps"].extend(vendor_steps)
            if trt_quality_set_local is not None:
                trt_quality_set_remote, quality_set_steps = _stage_remote_trt_quality_producer_set(
                    local_path=trt_quality_set_local,
                    ssh=ssh,
                    remote_root=rroot,
                    timeout=timeout,
                )
                result["steps"].extend(quality_set_steps)
                result["trt_quality_producer_set"].update({
                    "remote_path": trt_quality_set_remote,
                    "status": "verified_remote",
                })
            if split_quality_set_local is not None:
                (
                    split_quality_set_remote,
                    split_quality_set_steps,
                ) = _stage_remote_native_split_quality_binding_set(
                    local_path=split_quality_set_local,
                    expected_sha256=split_quality_set_sha,
                    ssh=ssh,
                    remote_root=rroot,
                    timeout=timeout,
                )
                result["steps"].extend(split_quality_set_steps)
                result["native_split_quality_binding_set"].update({
                    "remote_path": split_quality_set_remote,
                    "remote_sha256": split_quality_set_sha,
                    "status": "verified_remote",
                })
            if cache_verify_only:
                cache_rows = [
                    (str(model), str(case))
                    for model, cases in sorted(effective_case_map.items())
                    for case in list(cases or [])
                ]
                if b != "hailo8" or len(cache_rows) != 1:
                    raise TensorRTQualityChainError(
                        "cache_verify_only Native split replay requires exactly "
                        "one Hailo-8 model/case row"
                    )
                cache_model, cache_case = cache_rows[0]
                remote_base_dir = str(
                    rcfg.get("remote_base_dir") or "~/splitpoint_runs"
                ).rstrip("/")
                remote_engine_cache_root = (
                    remote_base_dir
                    + "/_onnx_splitpoint_cache/tensorrt"
                )
                split_quality_set_remote = (
                    f"{rroot}/cache_verify_native_split/{setup_id}/"
                    "native_split_quality_binding_set.json"
                )
                remote_bs = f"{rroot}/{cache_model}/benchmark_set"
                replay_args = [
                    "python",
                    "scripts/materialize_cache_verify_native_split_binding.py",
                    "--benchmark-set", _q(remote_bs),
                    "--model-id", _q(cache_model),
                    "--case", _q(cache_case),
                    "--setup-id", _q(setup_id),
                    "--backend", "hailo8_to_trt",
                    "--eval-run-id", _q(run_id),
                    "--engine-cache-root", _q(remote_engine_cache_root),
                    "--output", _q(split_quality_set_remote),
                ]
                replay_shell = " ".join(
                    ([env, "&&"] if str(env).strip() else [])
                    + ["cd", _q(rtool), "&&"]
                    + ["ONNX_SPLITPOINT_ARTIFACT_POLICY=cache_verify_only"]
                    + replay_args
                )
                replay_step = _run(
                    [
                        "ssh", "-o", "BatchMode=yes", "-o",
                        "StrictHostKeyChecking=accept-new", ssh, replay_shell,
                    ],
                    timeout=min(max(300, int(timeout)), 1800),
                    label="cache-verify-native-split-replay:hailo8",
                )
                result["steps"].append({
                    "name": "materialize_cache_verify_native_split_binding",
                    **replay_step,
                    "hardware_started": False,
                    "compiler_dispatch_allowed": False,
                    "remote_engine_cache_root": remote_engine_cache_root,
                    "remote_binding_set": split_quality_set_remote,
                })
                if int(replay_step.get("rc") or 0) != 0:
                    result["failure_reason"] = "cache_miss_blocked"
                    raise TensorRTQualityChainError(
                        "cache_miss_blocked before Native hardware start: "
                        + str(
                            replay_step.get("stderr_tail")
                            or replay_step.get("stdout_tail") or ""
                        )[-3000:]
                    )
                backend_split_quality_required = True
                backend_split_execution_allowed = True
                result["cache_verify_native_split_engine"] = {
                    "status": "verified_remote_replay",
                    "diagnostic_only": True,
                    "claim_eligible": False,
                    "compiler_dispatch_allowed": False,
                    "hardware_started_during_probe": False,
                    "remote_engine_cache_root": remote_engine_cache_root,
                    "remote_binding_set": split_quality_set_remote,
                    "model": cache_model,
                    "case": cache_case,
                    "setup_id": setup_id,
                }
                result["native_split_quality_binding_set"] = {
                    "remote_path": split_quality_set_remote,
                    "status": "verified_remote_cache_replay",
                    "selection_count": 1,
                }
            models_arg = ",".join(models)
            telemetry_label = _telemetry_label(cfg.get("telemetry_label") or precision)
            pre_telemetry = _capture_remote_host_telemetry(
                ssh=ssh, env=env, remote_tool_dir=rtool, remote_root=rroot,
                backend=b, setup_id=setup_id, run_id=run_id,
                phase="pre", label=telemetry_label, timeout=timeout,
            )
            telemetry_pre_attempted = True
            result["steps"].append(pre_telemetry)
            if int(pre_telemetry.get("rc") or 0) != 0:
                result.setdefault("warnings", []).append(
                    "pre-run host telemetry unavailable; performance continues because telemetry is non-blocking"
                )
            if b == "hailo8":
                args = ["python", "scripts/native_fifo_eval_runner.py", "--root", _q(rroot), "--models", _q(models_arg)]
                if case_policy == "case_map_only":
                    args += ["--case-map", _q(case_map_json)]
                args += ["--hw-arch", "hailo8", "--precision", _q(precision), "--frames", str(frames), "--warmup", str(warmup), "--repetitions", str(repetitions), "--queue-depth", str(queue_depth), "--hailo-format", _q(hailo_format), "--letterbox-pad-value", str(native_letterbox_pad_value), "--mixed-runtime-python", "/usr/bin/python3"]
                if build_missing: args.append("--build-missing-engines")
                if native_force_rebuild_engines: args.append("--force-rebuild-engines")
                if precision == "uint8_dequant_fp16":
                    if native_dequant_scale > 0.0: args += ["--dequant-scale", str(native_dequant_scale)]
                    args += ["--dequant-zero-point", str(native_dequant_zero_point)]
                if precision in {"uint8_dequant_fp16", "float32_layout_fp16"}:
                    if native_boundary_layout != "as_input": args += ["--boundary-layout", _q(native_boundary_layout)]
                if dump_outputs: args.append("--dump-outputs")
                if dump_boundary: args.append("--dump-boundary")
                if validation_image_map:
                    args += ["--image-map", _q(validation_image_map_json)]
            elif b == "hailo10h":
                args = ["python", "scripts/native_producer_e2e_eval_runner.py", "--root", _q(rroot), "--backend", "hailo10h", "--models", _q(models_arg)]
                if case_policy == "case_map_only":
                    args += ["--case-map", _q(case_map_json)]
                args += ["--precision", _q(precision), "--frames", str(frames), "--warmup", str(warmup), "--repetitions", str(repetitions), "--queue-depth", str(queue_depth), "--inflight", str(inflight), "--hailo-format", _q(hailo_format)]
                if build_missing: args.append("--build-missing-engine")
                if native_force_rebuild_engines: args.append("--force-rebuild-engine")
                if engine_build_python: args += ["--engine-build-python", _q(engine_build_python)]
                if precision == "uint8_dequant_fp16":
                    if native_dequant_scale > 0.0: args += ["--dequant-scale", str(native_dequant_scale)]
                    args += ["--dequant-zero-point", str(native_dequant_zero_point)]
                if precision in {"uint8_dequant_fp16", "float32_layout_fp16"}:
                    if native_boundary_layout != "as_input": args += ["--boundary-layout", _q(native_boundary_layout)]
                if dump_outputs: args.append("--dump-outputs")
                if dump_boundary: args.append("--dump-boundary")
                if validation_image_map:
                    args += ["--image-map", _q(validation_image_map_json)]
            elif b == "deepx":
                args = ["python", "scripts/native_producer_e2e_eval_runner.py", "--root", _q(rroot), "--backend", "deepx", "--models", _q(models_arg)]
                if case_policy == "case_map_only":
                    args += ["--case-map", _q(case_map_json)]
                args += ["--precision", _q(precision), "--frames", str(frames), "--warmup", str(warmup), "--repetitions", str(repetitions), "--queue-depth", str(queue_depth)]
                if build_missing: args.append("--build-missing-engine")
                if engine_build_python: args += ["--engine-build-python", _q(engine_build_python)]
                if precision in {"float32_layout_fp16", "uint8_dequant_fp16"}:
                    args += ["--boundary-layout", _q(native_boundary_layout)]
                if dump_outputs: args.append("--dump-outputs")
                if dump_boundary: args.append("--dump-boundary")
                if validation_image_map:
                    args += ["--image-map", _q(validation_image_map_json)]
            else:
                raise RuntimeError(f"unsupported backend {b}")
            if backend_split_quality_required:
                if not split_quality_set_remote:
                    raise TensorRTQualityChainError(
                        f"verified remote Native split binding set missing for "
                        f"setup={setup_id!r}"
                    )
                args += [
                    "--setup-id", _q(setup_id),
                    "--native-split-quality-binding-set",
                    _q(split_quality_set_remote),
                ]
            qf_env = ([
                "ONNX_SPLITPOINT_ARTIFACT_POLICY=cache_verify_only",
            ] if cache_verify_only else []) + ([
                "ONNX_SPLITPOINT_NATIVE_SPLIT_BINDING_SET="
                + _q(split_quality_set_remote),
                "ONNX_SPLITPOINT_SETUP_ID=" + _q(setup_id),
            ] if backend_split_quality_required else [])
            if backend_split_execution_allowed:
                inner = " ".join(
                    ([env, "&&"] if env else [])
                    + ["cd", _q(rtool), "&&"] + qf_env + args
                )
                rr = _run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, inner], timeout=performance_timeout, label=f"split:{b}")
                result["steps"].append({"name": "run_native_producer", **rr})
                split_runner_failed = bool(rr["rc"] != 0)
                child_checkpoint = reports / "native_child_checkpoints" / (
                    _artifact_namespace(str(cfg.get("artifact_namespace") or "default"))
                    + f"_{b}.json"
                )
                _write_json(child_checkpoint, {
                    "schema": "onnx-splitpoint/native-child-checkpoint",
                    "schema_version": 1,
                    "backend": b,
                    "setup_id": setup_id,
                    "models": models,
                    "terminal": True,
                    "runner_returncode": int(rr["rc"]),
                    "runner_status": (
                        "completed" if rr["rc"] == 0 else "failed"
                    ),
                    "failure_reason": (
                        "" if rr["rc"] == 0 else "native_runner_failed"
                    ),
                    "native_execution_contract_sha256": str(
                        execution_contract["contract_sha256"]
                    ),
                    "collection_status": "pending",
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "stdout_tail": str(rr.get("stdout_tail") or "")[-4000:],
                    "stderr_tail": str(rr.get("stderr_tail") or "")[-4000:],
                })
                result["native_child_checkpoint"] = str(child_checkpoint)
                if split_runner_failed:
                    result["failure_reason"] = "native_runner_failed"
                    result.setdefault("warnings", []).append(
                        "Native child failed; Full baselines and result "
                        "collection continue for this setup"
                    )
            else:
                if not split_quality_applicable:
                    result["steps"].append({
                        "name": "run_native_producer",
                        "status": "not_applicable_full_only",
                        "failure_reason": "",
                        "execution_attempted": False,
                        "diagnostic_continue": False,
                    })
                else:
                    result["steps"].append({
                        "name": "run_native_producer",
                        "status": "blocked_upstream_quality",
                        "failure_class": "upstream_quality_evidence",
                        "failure_reason": "upstream_central_quality_binding_missing",
                        "upstream_stage": "central_quality",
                        "transfer_attempted": False,
                        "execution_attempted": False,
                        "diagnostic_continue": True,
                    })

            # v59dt: Generate YOLO Full-ONNX self-reference sidecars on the remote
            # accelerator host before collecting results.  The local collector may
            # not have onnxruntime, but the remote benchmark_set already has the
            # exact boundary/input dumps and Full ONNX model.
            try:
                vcfg_probe = cfg.get("validation") if isinstance(cfg.get("validation"), dict) else {}
                probe_case_map = validation_case_map or effective_case_map or {}
                probe_yolo_requested = any(
                    "yolo" in str(model).lower()
                    and bool(cases)
                    for model, cases in (probe_case_map or {}).items()
                )
                if backend_split_execution_allowed and b in {"hailo8", "hailo10h", "deepx"} and bool(vcfg_probe.get("enabled") or cfg.get("dump_outputs")) and dump_boundary and probe_yolo_requested:
                    py = """
import json, subprocess, sys
from pathlib import Path
rroot = Path(%(rroot_json)s)
case_map = %(case_map_json)s
precision = %(precision_json)s
backend = %(backend_json)s
quality_gate_json = %(quality_gate_json)s
subdir = 'hailo10h_to_trt' if backend == 'hailo10h' else ('deepx_to_trt' if backend == 'deepx' else 'hailo_to_trt')
rows = []
for model, cases in sorted((case_map or {}).items()):
    if 'yolo' not in str(model).lower():
        continue
    bs = rroot / str(model) / 'benchmark_set'
    for case in cases or []:
        case = str(case)
        work = bs / 'native_pipeline' / case / subdir / precision
        man = work / 'native_fifo_boundary' / 'native_fifo_boundary_manifest.json'
        out = work / 'native_yolo_full_self_reference_probe.json'
        report_names = (
            ('native_fifo_results.json',) if backend == 'hailo8' else
            ('hailo10_native_fifo_e2e_results.json',) if backend == 'hailo10h' else
            ('deepx_native_fifo_e2e_results.json',)
        )
        native_report = next(
            (work / name for name in report_names if (work / name).is_file()),
            None,
        )
        report_payload = None
        if native_report is not None:
            try:
                loaded_report = json.loads(native_report.read_text(encoding='utf-8'))
                report_payload = loaded_report if isinstance(loaded_report, dict) else None
            except Exception:
                report_payload = None
        manifest_keys = (
            'native_fifo_output_manifest', 'native_output_manifest',
            'output_manifest', 'output_dump_manifest',
        )
        declared_manifests = []
        if isinstance(report_payload, dict):
            for key in manifest_keys:
                value = str(report_payload.get(key) or '').strip()
                if not value:
                    continue
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = work / path
                declared_manifests.append(path)
        exact_candidates = tuple(path for path in (
            work / 'native_fifo_outputs' / 'native_fifo_output_manifest.json',
            work / 'native_fifo_outputs' / 'native_fifo_outputs_manifest.json',
            work / 'native_outputs' / 'native_outputs_manifest.json',
        ) if path.is_file())
        manifest_paths = declared_manifests or list(exact_candidates)
        resolved_manifests = []
        manifests_valid = bool(manifest_paths)
        for path in manifest_paths:
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(work.resolve())
                resolved_manifests.append(resolved)
            except Exception:
                manifests_valid = False
        unique_manifests = {str(path) for path in resolved_manifests}
        manifests_valid = bool(
            manifests_valid and len(unique_manifests) == 1
            and (declared_manifests or len(exact_candidates) == 1)
        )
        native_manifest = (
            Path(next(iter(unique_manifests))) if manifests_valid else None
        )
        report_valid = bool(
            native_report is not None
            and isinstance(report_payload, dict)
            and report_payload.get('ok') is True
        )
        rec = {
            'model': str(model), 'case': case, 'precision': precision,
            'boundary_manifest': str(man),
            'native_output_manifest': str(native_manifest or ''),
            'native_report': str(native_report or ''), 'out': str(out),
        }
        if not man.is_file():
            rec.update({'ok': False, 'status': 'missing_boundary_manifest'})
            rows.append(rec); continue
        if not report_valid or native_manifest is None:
            rec.update({'ok': False, 'status': 'native_report_or_manifest_identity_invalid'})
            rows.append(rec); continue
        cmd = [
            sys.executable, 'scripts/native_yolo_full_self_reference_probe.py',
            '--benchmark-set', str(bs), '--case', case,
            '--boundary-manifest', str(man),
            '--native-output-manifest', str(native_manifest),
            '--out', str(out),
        ]
        cmd += ['--native-report', str(native_report)]
        if quality_gate_json:
            cmd += ['--quality-gate-json', quality_gate_json]
        cp = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        rec.update({'ok': cp.returncode == 0, 'rc': cp.returncode, 'stdout_tail': cp.stdout[-2000:], 'stderr_tail': cp.stderr[-2000:]})
        rows.append(rec)
ok = bool(rows) and all(r.get('ok') is True for r in rows)
print(json.dumps({'ok': ok, 'rows': rows}, indent=2))
if not rows:
    sys.exit(3)
sys.exit(0 if ok else 4)
""" % {
                        'rroot_json': json.dumps(rroot),
                        'case_map_json': json.dumps(probe_case_map),
                        'precision_json': json.dumps(precision),
                        'backend_json': json.dumps(b),
                        'quality_gate_json': json.dumps(
                            json.dumps(
                                dict(cfg.get('quality_gate_policy') or {}),
                                sort_keys=True,
                                separators=(',', ':'),
                            )
                            if isinstance(
                                cfg.get('quality_gate_policy'), Mapping,
                            )
                            else ''
                        ),
                    }
                    probe_inner = " ".join(([env, "&&"] if env else []) + ["cd", _q(rtool), "&&", "python", "-c", _q(py)])
                    sr = _run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, probe_inner], timeout=max(120, 30 * sum(len(v or []) for v in (probe_case_map or {}).values())))
                    result["steps"].append({"name": "remote_yolo_self_reference_sidecars", **sr, "cmd_shell": probe_inner})
            except Exception as _probe_exc:
                result.setdefault("warnings", []).append(f"remote self-reference sidecar generation failed: {type(_probe_exc).__name__}: {_probe_exc}")

            if native_full_enabled:
                fb_list = list(full_backends)
                if fb_list:
                    fargs = ["python", "-u", "scripts/native_full_baseline_eval_runner.py", "--root", _q(rroot), "--models", _q(models_arg), "--backends", _q(",".join(fb_list)), "--frames", str(frames), "--warmup", str(warmup), "--repetitions", str(repetitions), "--inflight", str(inflight), "--setup-id", _q(setup_id), "--comparison-backend", _q(b), "--comparison-precision", _q(precision), "--engine-build-python", _q(engine_build_python or "auto")]
                    if vendor_quality_set_remote:
                        fargs += ["--quality-request-binding-set", _q(vendor_quality_set_remote)]
                    if "tensorrt" in fb_list:
                        if not trt_quality_set_remote:
                            raise TensorRTQualityChainError(
                                f"verified remote TensorRT quality producer set missing "
                                f"for setup={setup_id!r}"
                            )
                        fargs += [
                            "--trt-quality-producer-json",
                            _q(trt_quality_set_remote),
                        ]
                    if validation_image_map:
                        fargs += ["--image-map", _q(validation_image_map_json)]
                    fargs.append("--dump-outputs")
                    py_path = f"PYTHONPATH={_q(rtool)}:${{PYTHONPATH:-}}"
                    preflight_program = (
                        f"{py_path} python -u "
                        "scripts/native_full_baseline_eval_runner.py "
                        "--root . --remote-contract-preflight && "
                        f"{py_path} python -u "
                        "scripts/native_full_semantic_dump.py --help"
                    )
                    preflight_shell = " ".join(
                        ([env, "&&"] if env else [])
                        + ["cd", _q(rtool), "&&", preflight_program]
                    )
                    pf = _run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, preflight_shell], timeout=min(timeout, 120), label=f"full-preflight:{b}")
                    result["steps"].append({"name": "native_full_remote_import_preflight", **pf, "cmd_shell": preflight_shell})
                    if pf.get("rc") != 0:
                        fr = dict(pf)
                        fr["failure_reason"] = "remote_runner_import_failed"
                    else:
                        shell_full = " ".join(([env, "&&"] if env else []) + ["cd", _q(rtool), "&&", py_path] + fargs)
                        fr = _run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", ssh, shell_full], timeout=performance_timeout, label=f"full:{b}")
                        result["steps"].append({"name": "run_native_full_baselines", **fr, "cmd_shell": shell_full})
                    result["native_full_baseline_available"] = fr.get("rc") == 0
                    result["native_full_baseline_rc"] = fr.get("rc")
                    if fr.get("rc") != 0:
                        reason = str(fr.get("failure_reason") or _native_full_failure_reason(fr))
                        result["native_full_failure_reason"] = reason
                        result.setdefault("warnings", []).append(f"native full baseline failed reason={reason} rc={fr.get('rc')}: {fr.get('stderr_tail') or fr.get('stdout_tail')}")
            post_telemetry = _capture_remote_host_telemetry(
                ssh=ssh, env=env, remote_tool_dir=rtool, remote_root=rroot,
                backend=b, setup_id=setup_id, run_id=run_id,
                phase="post", label=telemetry_label, timeout=timeout,
            )
            telemetry_post_attempted = True
            result["steps"].append(post_telemetry)
            if int(post_telemetry.get("rc") or 0) != 0:
                result.setdefault("warnings", []).append(
                    "post-run host telemetry unavailable; performance continues because telemetry is non-blocking"
                )
            if split_runner_failed:
                # A failed child remains a partial result so setup-local Full
                # diagnostics and result collection can continue.  Recover its
                # telemetry explicitly as well: the exception-only recovery
                # path is not entered for this intentional continuation.
                recovery_args: dict[str, Any] = {
                    "run_dir": run_dir,
                    "ssh": ssh,
                    "remote_root": rroot,
                    "backend": b,
                    "timeout": timeout,
                }
                if artifact_namespace:
                    recovery_args["local_backend_root"] = _local_backend_root(
                        run_dir, cfg, b,
                    )
                recovered_telemetry = _collect_failure_host_telemetry(
                    **recovery_args,
                )
                failure_telemetry_recovered = True
                result["steps"].append(recovered_telemetry)
                if int(recovered_telemetry.get("rc") or 0) != 0:
                    result.setdefault("warnings", []).append(
                        "failure-time host telemetry could not be copied locally; "
                        "the original benchmark failure is retained"
                    )
            local_root = _local_backend_root(run_dir, cfg, b)
            if local_root.exists():
                import shutil as _shutil
                _shutil.rmtree(local_root)
            local_root.mkdir(parents=True, exist_ok=True)
            cr = _run(["rsync", "-a", "--delete", f"{ssh}:{rroot}/", str(local_root) + "/"], timeout=timeout)
            result["steps"].append({"name": "collect_results", **cr})
            if cr["rc"] != 0: raise RuntimeError(f"collect failed: {cr.get('stderr_tail')}")
            if result.get("native_child_checkpoint"):
                child_checkpoint = Path(
                    str(result["native_child_checkpoint"])
                )
                checkpoint_payload = _read_json(child_checkpoint) or {}
                if isinstance(checkpoint_payload, Mapping):
                    _write_json(child_checkpoint, {
                        **dict(checkpoint_payload),
                        "collection_status": "collected",
                        "collection_returncode": 0,
                        "collection_finished_at": time.strftime(
                            "%Y-%m-%dT%H:%M:%S%z"
                        ),
                    })
            collected.append(local_root)
            result_status = (
                "partial" if (
                    split_runner_failed
                    or
                    result.get("native_full_baseline_available") is False
                    or isinstance(result.get("upstream_quality_error"), Mapping)
                ) else "ok"
            )
            result.update({"ok": True, "status": result_status, "local_root": str(local_root)})
        except Exception as exc:
            if telemetry_pre_attempted and not telemetry_post_attempted:
                post_telemetry = _capture_remote_host_telemetry(
                    ssh=ssh, env=env, remote_tool_dir=rtool, remote_root=rroot,
                    backend=b, setup_id=setup_id, run_id=run_id,
                    phase="post", label=_telemetry_label(cfg.get("telemetry_label") or precision),
                    timeout=timeout,
                )
                telemetry_post_attempted = True
                result["steps"].append(post_telemetry)
            if telemetry_pre_attempted and not failure_telemetry_recovered:
                recovery_args: dict[str, Any] = {
                    "run_dir": run_dir,
                    "ssh": ssh,
                    "remote_root": rroot,
                    "backend": b,
                    "timeout": timeout,
                }
                if artifact_namespace:
                    recovery_args["local_backend_root"] = _local_backend_root(
                        run_dir, cfg, b,
                    )
                recovered_telemetry = _collect_failure_host_telemetry(
                    **recovery_args,
                )
                result["steps"].append(recovered_telemetry)
                if int(recovered_telemetry.get("rc") or 0) != 0:
                    result.setdefault("warnings", []).append(
                        "failure-time host telemetry could not be copied locally; "
                        "the original benchmark failure is retained"
                    )
            try:
                failure_root = _local_backend_root(run_dir, cfg, b)
                failure_root.mkdir(parents=True, exist_ok=True)
                recovered_results = _run(
                    [
                        "rsync", "-a", f"{ssh}:{rroot}/",
                        str(failure_root) + "/",
                    ],
                    timeout=timeout,
                )
                result["steps"].append({
                    "name": "collect_failure_results",
                    **recovered_results,
                })
                if recovered_results.get("rc") == 0:
                    collected.append(failure_root)
            except Exception as collection_exc:
                result.setdefault("warnings", []).append(
                    "failure-time result collection unavailable: "
                    f"{type(collection_exc).__name__}: {collection_exc}"
                )
            result.update({"ok": False, "status": "partial", "error": f"{type(exc).__name__}: {exc}"})
        stage["backend_results"].append(result)
    telemetry_summary = _summarize_host_telemetry(run_dir, reports)
    stage["host_telemetry_summary"] = telemetry_summary
    if int(telemetry_summary.get("rc") or 0) != 0:
        stage.setdefault("warnings", []).append(
            "host telemetry summary unavailable; native performance results were retained"
        )
    # combined report
    if collected:
        cmd = [sys.executable, str(_script("native_producer_final_report.py"))]
        for r in collected: cmd += ["--root", str(r)]
        cmd += ["--recursive", "--out-dir", str(reports)]
        cmd += _final_report_remote_context_args(stage["backend_results"])
        pr = _run(cmd, timeout=300); stage["final_report"] = pr
        for base in ("native_producer_combined_summary",):
            for ext in ("json", "csv", "md"):
                src = reports / f"{base}.{ext}"
                if src.exists():
                    dst = reports / f"native_producer_summary.{ext}"
                    if src != dst:
                        dst.write_bytes(src.read_bytes())

    # Native Validation / Visuals annotate the technically admitted Energy plan.
    try:
        vcfg = cfg.get("validation") if isinstance(cfg.get("validation"), dict) else {}
        if bool(vcfg.get("enabled")):
            summary_json = reports / "native_producer_combined_summary.json"
            if not summary_json.exists():
                summary_json = reports / "native_producer_summary.json"
            validation_record = {"enabled": True, "status": "skipped"}
            if summary_json.exists():
                vout = reports / "native_validation"
                roots = list(collected)
                cmd = [sys.executable, str(_script("native_producer_validate_visualize.py")), "--summary", str(summary_json), "--out-dir", str(vout), "--topk", str(int(vcfg.get("topk") or 5))]
                if isinstance(cfg.get("quality_gate_policy"), Mapping):
                    cmd += ["--quality-gate-json", json.dumps(dict(cfg.get("quality_gate_policy") or {}), sort_keys=True, separators=(",", ":"))]
                central_quality_summary = (Path(str(cfg["central_quality_summary"])) if cfg.get("central_quality_summary")
                    else run_dir / "quality_management" / "central_quality_summary.json")
                if central_quality_summary.is_file():
                    cmd += [
                        "--central-quality-summary",
                        str(central_quality_summary),
                    ]
                for r in roots:
                    cmd += ["--root", str(r)]
                vr = _run(cmd, timeout=600)
                validation_payload = _read_json(
                    vout / "native_producer_validation_summary.json"
                ) or {}
                validation_record.update({"cmd": cmd, "rc": vr.get("rc"), "stdout_tail": vr.get("stdout_tail"), "stderr_tail": vr.get("stderr_tail"), "out_dir": str(vout), "status": "ok" if vr.get("rc") == 0 else "failed", "technical_error_count": int(validation_payload.get("technical_error_count") or 0)})
            else:
                validation_record.update({"status": "failed", "error": "native_producer_summary.json missing"})
            stage["native_validation"] = validation_record
        else:
            stage["native_validation"] = {
                "enabled": False,
                "requested": False,
                "status": "not_applicable",
                "rc": 0,
                "complete": True,
                "reason": "not_requested",
            }
    except Exception as exc:
        stage["native_validation"] = {"enabled": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    # v2.68: the first report is the input to semantic validation.  Rebuild it
    # afterwards so precision/performance claim gates are bound to that exact
    # validation summary.  Missing or failed validation deliberately leaves the
    # first report fail-closed.
    quality_summary = reports / "native_validation" / "native_producer_validation_summary.json"
    validation_ok = str((stage.get("native_validation") or {}).get("status") or "") == "ok"
    if collected and validation_ok and quality_summary.is_file():
        qcmd = [sys.executable, str(_script("native_producer_final_report.py"))]
        for root in collected:
            qcmd += ["--root", str(root)]
        qcmd += [
            "--recursive", "--out-dir", str(reports),
            "--quality-summary", str(quality_summary),
        ]
        qcmd += _final_report_remote_context_args(stage["backend_results"])
        qreport = _run(qcmd, timeout=300, label="quality-gated-final-report")
        stage["quality_gated_final_report"] = qreport
        if qreport.get("rc") == 0:
            for ext in ("json", "csv", "md"):
                src = reports / f"native_producer_combined_summary.{ext}"
                dst = reports / f"native_producer_summary.{ext}"
                if src.is_file():
                    dst.write_bytes(src.read_bytes())
    else:
        stage["quality_gated_final_report"] = {
            "rc": None,
            "status": (
                "not_applicable"
                if (stage.get("native_validation") or {}).get("enabled")
                is False else "skipped_fail_closed"
            ),
            "reason": (
                "native_validation_not_requested"
                if (stage.get("native_validation") or {}).get("enabled")
                is False
                else "native_validation_not_successful_or_summary_missing"
            ),
            "quality_summary": str(quality_summary),
        }


    # v60i: Native Energy runs only after semantic validation. Mirrors the EvalRunner
    # stage and uses time-based duration from ToolConfig unless overridden.
    try:
        ecfg = cfg.get("energy") if isinstance(cfg.get("energy"), dict) else {}
        if bool(ecfg.get("enabled")):
            mode = str(ecfg.get("mode") or "plan").strip().lower()
            if mode not in {"plan", "measure"}:
                mode = "plan"
            try:
                from onnx_splitpoint_tool.energy.config import load_energy_defaults as _load_energy_defaults
                default_duration_s = float(getattr(_load_energy_defaults(), "native_energy_duration_s", 60.0) or 60.0)
            except Exception:
                default_duration_s = 60.0
            duration_s = float(ecfg.get("duration_s") or default_duration_s)
            duration_s = max(1.0, duration_s)
            etimeout = int(ecfg.get("timeout") or 900)
            summary_json = reports / "native_producer_summary.json"
            validation_summary_json = reports / "native_validation" / "native_producer_validation_summary.json"
            energy_record = {"enabled": True, "mode": mode, "duration_s": duration_s, "status": "skipped", "validation_summary": str(validation_summary_json), "semantic_gate_required": False, "quality_claim_gate_required": True, "measure_all_runtime_successful": True, "measurement_admission_policy": "all_runtime_successful_constructible_native_rows", "diagnostic_only": smoke_diagnostic, "claim_eligible": False if smoke_diagnostic else None, "energy_claim_eligible": False if smoke_diagnostic else None}
            if summary_json.exists():
                if mode == "measure":
                    eout = reports / "native_energy_measurements"
                    cmd = [sys.executable, str(_script("run_native_producer_energy_from_summary.py")), "--summary", str(summary_json), "--validation-summary", str(validation_summary_json), "--out-dir", str(eout), "--duration-s", str(duration_s), "--hailo8-ssh", str((cfg.get("remotes") or {}).get("hailo8", {}).get("ssh", "")), "--hailo10-ssh", str((cfg.get("remotes") or {}).get("hailo10h", {}).get("ssh", "")), "--deepx-ssh", str((cfg.get("remotes") or {}).get("deepx", {}).get("ssh", "")), "--remote-tool-dir", str(cfg.get("remote_tool_dir") or "/home/nx/ONNX-Splitpoint-Tool"), "--remote-root", str(cfg.get("remote_root") or "/home/nx/native_fifo_evalsets").rstrip("/") + "/" + run_id, "--timeout", str(etimeout), "--runs", str(max(1, int(ecfg.get("repeat_override") or 1)))]
                    cmd.append("--measure-all-runtime-successful")
                    if smoke_diagnostic or bool(ecfg.get("allow_unpaired")):
                        cmd.append("--allow-unpaired")
                    if smoke_diagnostic:
                        cmd.append("--smoke-diagnostic")
                    er = _run(cmd, timeout=max(1800, 300 + len(stage.get("backend_results", [])) * (etimeout + 420)))
                else:
                    eout = reports / "native_energy_plan"
                    cmd = [sys.executable, str(_script("native_producer_energy_plan.py")), "--summary", str(summary_json), "--validation-summary", str(validation_summary_json), "--out-dir", str(eout), "--duration-s", str(duration_s), "--hailo8-ssh", str((cfg.get("remotes") or {}).get("hailo8", {}).get("ssh", "")), "--hailo10-ssh", str((cfg.get("remotes") or {}).get("hailo10h", {}).get("ssh", "")), "--deepx-ssh", str((cfg.get("remotes") or {}).get("deepx", {}).get("ssh", "")), "--remote-tool-dir", str(cfg.get("remote_tool_dir") or "/home/nx/ONNX-Splitpoint-Tool"), "--remote-root", str(cfg.get("remote_root") or "/home/nx/native_fifo_evalsets").rstrip("/") + "/" + run_id, "--timeout", str(etimeout)]
                    cmd.append("--measure-all-runtime-successful")
                    if smoke_diagnostic or bool(ecfg.get("allow_unpaired")):
                        cmd.append("--allow-unpaired")
                    if smoke_diagnostic:
                        cmd.append("--smoke-diagnostic")
                    er = _run(cmd, timeout=180)
                plan_path = (
                    eout / "plan" / "native_producer_energy_plan.json"
                    if mode == "measure"
                    else eout / "native_producer_energy_plan.json"
                )
                plan_payload = (
                    _read_json(plan_path)
                    if plan_path.is_file() else {}
                ) or {}
                plan_preflight_status = str(
                    plan_payload.get("preflight_status") or ""
                )
                energy_status = (
                    plan_preflight_status
                    if plan_preflight_status
                    and plan_preflight_status != "passed"
                    else "ok" if er.get("rc") == 0
                    else "failed"
                )
                energy_record.update({"cmd": cmd, "rc": er.get("rc"), "stdout_tail": er.get("stdout_tail"), "stderr_tail": er.get("stderr_tail"), "out_dir": str(eout), "status": energy_status, "plan_preflight_status": plan_preflight_status})
            else:
                energy_record.update({"status": "failed", "error": "native_producer_summary.json missing"})
            stage["native_energy"] = energy_record
        else:
            stage["native_energy"] = {
                "enabled": False,
                "requested": False,
                "status": "not_applicable",
                "ok": True,
                "complete": True,
                "strict_requested": False,
                "strict_failure": False,
                "reason": "not_requested",
            }
    except Exception as exc:
        stage["native_energy"] = {"enabled": True, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    # v60l: legacy accuracy-gate report retired.  The canonical Scientific
    # Report consumes runtime task_quality_gate blocks after native rows have
    # been attached.
    stage["accuracy_gates"] = {
        "enabled": False,
        "status": "retired_replaced_by_scientific_report",
        "replacement": "reports/scientific/task_quality.csv",
    }

    stage["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ok_count = sum(1 for r in stage["backend_results"] if r.get("ok"))
    stage["ok_backends"] = ok_count
    has_partial = any(
        str(row.get("status") or "") == "partial"
        for row in stage["backend_results"]
    )
    native_validation_technical_errors = int(
        (stage.get("native_validation") or {}).get("technical_error_count") or 0
    )
    native_validation_failed = bool(
        (stage.get("native_validation") or {}).get("enabled") is True
        and (
            int((stage.get("native_validation") or {}).get("rc") or 0) != 0
            or str((stage.get("native_validation") or {}).get("status") or "")
            == "failed"
        )
    )
    stage["status"] = (
        "failed" if (
            (native_validation_technical_errors > 0 or native_validation_failed)
            and not smoke_diagnostic
        )
        else "partial" if (
            has_partial or native_validation_technical_errors > 0
            or native_validation_failed
        )
        else "ok" if ok_count == len(backends)
        else "partial" if ok_count
        else "failed"
    )
    _write_json(reports / "native_producer_stage.json", stage)
    return stage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-run-dir", required=True)
    ap.add_argument("--refresh-suites", action="store_true", help="Refresh generated benchmark_suite.py / case runner templates inside every model BenchmarkSet.")
    ap.add_argument("--write-profile-config", action="store_true", help="Write native_producers config into <EvalRun>/profile.yaml.")
    ap.add_argument("--run-native-producers", action="store_true", help="Execute native producer stage against the existing EvalRun without regenerating BenchmarkSets.")
    ap.add_argument("--backends", default="hailo8,hailo10h,deepx")
    ap.add_argument("--case-policy", default="all_accepted", choices=["all_accepted", "case_map_only", "preferred_then_backfill"])
    ap.add_argument("--case-map", default="")
    ap.add_argument("--hailo8-ssh", default="")
    ap.add_argument("--hailo10-ssh", default="")
    ap.add_argument("--deepx-ssh", default="")
    ap.add_argument("--hailo8-setup-id", default="")
    ap.add_argument("--hailo10-setup-id", default="")
    ap.add_argument("--deepx-setup-id", default="")
    ap.add_argument("--hailo8-env", default="")
    ap.add_argument("--hailo10-env", default="export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate")
    ap.add_argument("--deepx-env", default="source ~/venvs/deepx-runtime/bin/activate")
    ap.add_argument("--hailo8-remote-base-dir", default="")
    ap.add_argument("--hailo10-remote-base-dir", default="")
    ap.add_argument("--deepx-remote-base-dir", default="")
    ap.add_argument("--remote-root", default="/home/nx/native_fifo_evalsets")
    ap.add_argument("--remote-tool-dir", default="/home/nx/ONNX-Splitpoint-Tool")
    ap.add_argument("--central-quality-summary", default="")
    ap.add_argument("--vendor-full-quality-binding-sets", default="")
    ap.add_argument(
        "--artifact-namespace", default="",
        help=(
            "Safe per-variant namespace. Remote and local results are isolated "
            "below this component and retained for combined semantics/Energy."
        ),
    )
    ap.add_argument(
        "--artifact-policy",
        default=os.environ.get("ONNX_SPLITPOINT_ARTIFACT_POLICY", "normal"),
        choices=["normal", "cache_verify_only"],
        help="Fail closed on every compiler/engine cache miss.",
    )
    ap.add_argument("--precision", default="uint8_cast_fp16")
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--repetitions", type=int, default=1, help="Independent performance repetitions for Split and Full rows. Energy remains a separate one-pass workload unless --native-energy-runs says otherwise.")
    ap.add_argument("--queue-depth", type=int, default=3)
    ap.add_argument("--inflight", type=int, default=8)
    ap.add_argument(
        "--native-execution-contract-json", default="",
        help=(
            "Exact parent-sealed Native effort contract. When supplied, all "
            "five CLI effort values must match before remote dispatch."
        ),
    )
    ap.add_argument(
        "--quality-gate-json", default="",
        help=(
            "Exact profile quality_gate JSON. It must match profile.yaml and "
            "is propagated to remote self-reference and local validation."
        ),
    )
    ap.add_argument("--hailo-format", default="uint8")
    ap.add_argument("--dump-outputs", action="store_true")
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument("--engine-build-python", default="auto")
    ap.add_argument("--no-build-missing-engines", action="store_true")
    ap.add_argument("--native-full-baselines", action="store_true", help="Run native full-model baselines as separate native_full_baseline rows.")
    ap.add_argument("--native-full-backends", default="", help="Optional comma list overriding full baseline backends.")
    ap.add_argument("--native-full-backends-by-producer", default="", help="JSON mapping producer to local full backends; e.g. hailo8->[hailo8,tensorrt].")
    ap.add_argument(
        "--trt-quality-producer-sets", default="",
        help=(
            "JSON mapping physical setup_id to the one central-quality-verified "
            "TensorRT producer-set file. Required for every requested Native "
            "Full TensorRT backend."
        ),
    )
    ap.add_argument(
        "--native-split-quality-binding-sets", default="",
        help=(
            "JSON mapping physical setup_id to its exact central-quality "
            "Native split binding-set file."
        ),
    )
    ap.add_argument(
        "--native-split-quality-required", action="store_true",
        help=(
            "Fail before remote execution unless every selected split row has "
            "one exact setup-local Quality-FIRST binding."
        ),
    )
    ap.add_argument(
        "--native-split-quality-not-applicable", action="store_true",
        help=(
            "The sealed Native matrix is Full-only. Keep the current Split "
            "Quality policy required, but do not synthesize or execute Split "
            "rows."
        ),
    )
    ap.add_argument(
        "--smoke-diagnostic-quality-continue", action="store_true",
        help=(
            "Smoke-only: block unbound Native split/TensorRT paths before "
            "transfer, continue setup-local vendor Full diagnostics, and keep "
            "all resulting evidence ineligible for claims."
        ),
    )
    ap.add_argument("--native-energy", action="store_true", help="Enable native-producer energy integration for the updated EvalRun.")
    ap.add_argument("--native-energy-mode", default="", choices=["", "plan", "measure"], help="Native energy mode. Use plan for smoke tests; measure executes u.RECS commands.")
    ap.add_argument("--native-energy-duration-s", type=float, default=0.0, help="Optional override. Default uses ToolConfig energy_defaults.native_energy_duration_s.")
    ap.add_argument("--native-energy-frames", type=int, default=0, help="Legacy compatibility only; Native Energy is time based.")
    ap.add_argument("--native-energy-warmup", type=int, default=0, help="Legacy compatibility only; Native Energy defaults to 0 command warmup.")
    ap.add_argument("--native-energy-timeout", type=int, default=900)
    ap.add_argument("--native-energy-runs", type=int, default=1)
    ap.add_argument("--native-energy-allow-unpaired", action="store_true")
    ap.add_argument("--native-validation", action="store_true", help="Enable native output dump validation / visual artifacts for update/debug runs.")
    ap.add_argument("--native-boundary-debug", action="store_true", help="For Hailo8 native FIFO, also dump raw Part1 boundary payloads for interface/dequant diagnostics.")
    ap.add_argument("--native-letterbox-pad-value", type=int, default=0, help="Native Hailo8 input letterbox pad value. Use 114 to align YOLO native preprocessing with the generic harness.")
    ap.add_argument("--native-force-rebuild-engines", action="store_true", help="Force rebuild native TensorRT engines, needed when changing dequant/layout bridge parameters.")
    ap.add_argument("--native-dequant-scale", type=float, default=0.0, help="Explicit Hailo boundary uint8 dequant scale for Hailo8->TRT uint8_dequant_fp16.")
    ap.add_argument("--native-dequant-zero-point", type=float, default=0.0, help="Explicit Hailo boundary uint8 dequant zero point for Hailo8->TRT uint8_dequant_fp16.")
    ap.add_argument("--native-boundary-layout", default="as_input", choices=["as_input", "memory_nwc_to_ncw", "memory_nhwc_to_nchw", "memory_hwcn_to_nchw", "memory_nwhc_to_nchw", "memory_ncwh_to_nchw", "memory_chwn_to_nchw"], help="Raw Hailo boundary memory layout before the Part2 input. Use memory_nhwc_to_nchw when validator ranks that layout best.")
    ap.add_argument("--native-validation-mode", default="dump_and_visual")
    ap.add_argument("--native-validation-topk", type=int, default=5)
    ap.add_argument("--native-telemetry-label", default="manual", help="Stable label used for non-blocking pre/post host telemetry evidence files.")
    ap.add_argument("--timeout", type=int, default=7200)
    ns = ap.parse_args()
    if ns.native_force_rebuild_engines:
        ap.error("productive_force_build_disabled: --native-force-rebuild-engines is disabled; missing engines remain buildable")
    if ns.repetitions < 1:
        ap.error("--repetitions must be >= 1")
    run_dir = Path(ns.eval_run_dir).expanduser().resolve()
    cfg = _bind_evalrun_quality_gate_policy(
        run_dir, _native_cfg_from_args(ns),
    )
    if cfg.get("cache_verify_only"):
        if not bool(ns.no_build_missing_engines):
            ap.error("cache_verify_only requires --no-build-missing-engines")
        if bool(ns.native_force_rebuild_engines):
            ap.error("cache_verify_only forbids --native-force-rebuild-engines")
        if bool(ns.refresh_suites):
            ap.error("cache_verify_only forbids --refresh-suites")
        os.environ["ONNX_SPLITPOINT_ARTIFACT_POLICY"] = "cache_verify_only"
    reports = _reports_dir_for_cfg(run_dir, cfg)
    reports.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"ok": True, "eval_run_dir": str(run_dir), "artifact_namespace": str(cfg.get("artifact_namespace") or ""), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps": []}
    if ns.refresh_suites:
        summary["steps"].append({"name": "refresh_suites", "results": _refresh_suites(run_dir)})
    if ns.write_profile_config:
        summary["steps"].append({"name": "write_profile_config", **_write_profile_config(run_dir, cfg)})
    _write_json(reports / "native_producer_stage_config.json", cfg)
    native_stage: dict[str, Any] | None = None
    if ns.run_native_producers:
        try:
            native_stage = _run_native_producers(run_dir, cfg, timeout=ns.timeout)
        except Exception as exc:
            native_stage = {
                "schema": "onnx-splitpoint/native-producer-stage",
                "schema_version": 2,
                "enabled": True,
                "status": "failed",
                "orchestration_status": "failed",
                "evidence_status": "empty",
                "failure_class": "native_selection_preflight",
                "failure_reason": "requested_benchmark_set_or_case_unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "transfer_attempted": False,
                "started_remote_count": 0,
                "started_performance_count": 0,
                "technical_quality_failure": True,
            }
            _write_json(reports / "native_producer_stage.json", native_stage)
        summary["steps"].append({"name": "run_native_producers", "stage": native_stage})
    native_status = str((native_stage or {}).get("status") or "ok")
    summary["status"] = native_status
    summary["ok"] = native_status == "ok"
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_json(reports / "update_evalset_native_producers.json", summary)
    print(json.dumps({"ok": summary["ok"], "status": native_status, "eval_run_dir": str(run_dir), "summary": str(reports / "update_evalset_native_producers.json"), "native_stage_config": str(reports / "native_producer_stage_config.json")}, indent=2))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
