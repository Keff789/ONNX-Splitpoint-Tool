from __future__ import annotations

"""Append-only Hailo compiler-attempt receipts.

Every invocation receives its own immutable start and terminal receipt.  A
small atomic pointer selects the chronologically last terminal attempt, even
when that attempt failed or timed out.  Earlier parser errors and later
fallback timeouts therefore remain simultaneously visible.
"""

import dataclasses
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "onnx-splitpoint/hailo-build-attempt-receipt"


def _sha256_file(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        path = Path(path)
    except Exception:
        return ""
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        return path
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    return path


def _result_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    if dataclasses.is_dataclass(result):
        try:
            return dataclasses.asdict(result)
        except Exception:
            pass
    out: dict[str, Any] = {}
    for name in (
        "ok", "status", "semantic_status", "elapsed_s", "error", "hef_path",
        "timed_out", "timeout_kind", "last_stage", "compiler_phase",
        "failure_kind", "unsupported_reason", "returncode", "rc", "details",
        "fixed_onnx_path", "compiler_onnx_path", "subprocess_stdout_tail",
        "subprocess_stderr_tail",
    ):
        if hasattr(result, name):
            try:
                out[name] = getattr(result, name)
            except Exception:
                pass
    return out


def semantic_status(result: Any = None, error: BaseException | None = None) -> str:
    if error is not None:
        text = f"{type(error).__name__}:{error}".lower()
        if isinstance(error, TimeoutError) or "timeout" in text:
            return "timeout"
        if "unsupported" in text or "not supported" in text:
            return "unsupported"
        return "failed"
    row = _result_mapping(result)
    status = str(row.get("semantic_status") or row.get("status") or "").strip().lower().replace("-", "_")
    if bool(row.get("timed_out")) or str(row.get("timeout_kind") or "") or status in {"timeout", "timed_out", "hard_timeout", "idle_timeout"}:
        return "timeout"
    if str(row.get("unsupported_reason") or "") or status in {"unsupported", "not_supported"}:
        return "unsupported"
    rc = row.get("returncode", row.get("rc"))
    try:
        if rc not in (None, "") and int(rc) != 0:
            return "failed"
    except (TypeError, ValueError, OverflowError):
        return "failed"
    if row.get("ok") is False or status in {"failed", "fail", "error", "rejected"}:
        return "failed"
    if str(row.get("error") or "").strip():
        return "failed"
    if row.get("ok") is True or status in {"ok", "success", "successful", "completed", "pass", "passed"}:
        return "success"
    return "failed" if result is None or result is False else "success"


def _normalise_nodes(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value or "").strip() else []


def begin_hailo_attempt(*, outdir: Path, bound: Mapping[str, Any]) -> dict[str, Any]:
    outdir = Path(outdir)
    started = time.time()
    attempt_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(started))}_{uuid.uuid4().hex}"
    source_text = str(bound.get("onnx_path") or bound.get("source_onnx") or "")
    source = Path(source_text).expanduser() if source_text else None
    compiler_text = str(
        bound.get("compiler_onnx") or bound.get("compiler_onnx_path")
        or bound.get("activation_part1_onnx") or source_text
    )
    compiler = Path(compiler_text).expanduser() if compiler_text else None
    receipts = outdir / "hailo_attempt_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "schema_version": 2,
        "attempt_id": attempt_id,
        "started_at_epoch_s": started,
        "invocation_status": "running",
        "semantic_status": "running",
        "terminal": False,
        "outdir": str(outdir),
        "hw_arch": str(bound.get("hw_arch") or bound.get("target_arch") or bound.get("arch") or ""),
        "backend": str(bound.get("backend") or ""),
        "net_name": str(bound.get("net_name") or ""),
        "attempt_kind": str(bound.get("attempt_kind") or bound.get("endpoint") or "unspecified"),
        "endpoint": str(bound.get("endpoint") or bound.get("output_endpoint") or ""),
        "start_nodes": _normalise_nodes(bound.get("start_nodes") or bound.get("start_node_names")),
        "end_nodes": _normalise_nodes(bound.get("end_nodes") or bound.get("end_node_names")),
        "source_onnx": str(source or ""),
        "source_onnx_sha256": _sha256_file(source),
        "compiler_onnx": str(compiler or ""),
        "compiler_onnx_sha256": _sha256_file(compiler),
        "timeout_policy": {
            "requested_timeout_s": bound.get("timeout_s", bound.get("requested_timeout_s")),
            "hard_timeout_s": bound.get("hard_timeout_s"),
            "idle_timeout_s": bound.get("idle_timeout_s"),
            "hard_timeout_enabled": bound.get("hard_timeout_enabled"),
            "heartbeat_enabled": bound.get("heartbeat_enabled", True),
            "manual_abort_enabled": bound.get("manual_abort_enabled", True),
        },
        "metadata": dict(bound.get("metadata") or {}) if isinstance(bound.get("metadata"), Mapping) else {},
    }
    start_path = receipts / f"attempt_{attempt_id}.started.json"
    _atomic_json(start_path, payload, exclusive=True)
    payload["start_receipt"] = str(start_path)
    return payload


def _select_terminal(receipts: Path) -> Path | None:
    rows: list[tuple[float, Path, dict[str, Any]]] = []
    for path in receipts.glob("attempt_*.json"):
        if path.name.endswith(".started.json") or path.name == "terminal_attempt.json":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(value, Mapping) or value.get("terminal") is not True:
            continue
        rows.append((float(value.get("ended_at_epoch_s") or 0.0), path, dict(value)))
    if not rows:
        return None
    _, selected_path, selected = sorted(rows, key=lambda item: (item[0], item[1].name))[-1]
    selected["immutable_receipt"] = str(selected_path)
    selected["immutable_receipt_sha256"] = _sha256_file(selected_path)
    _atomic_json(receipts / "terminal_attempt.json", selected)
    return selected_path


def finalize_hailo_attempt(
    *, attempt: Mapping[str, Any], result: Any = None,
    error: BaseException | None = None, compiler_phase: str = "",
    stdout_tail: str = "", stderr_tail: str = "", returned: bool = True,
) -> Path:
    payload = dict(attempt)
    row = _result_mapping(result)
    ended = time.time()
    fixed_onnx_text = str(
        row.get("fixed_onnx_path") or row.get("compiler_onnx_path")
        or payload.get("compiler_onnx") or ""
    )
    fixed_onnx = Path(fixed_onnx_text).expanduser() if fixed_onnx_text else None
    hef_text = str(row.get("hef_path") or "")
    hef = Path(hef_text).expanduser() if hef_text else None
    phase = str(
        compiler_phase or row.get("compiler_phase") or row.get("last_stage") or ""
    )
    semantic = semantic_status(result, error)
    try:
        rc = row.get("returncode", row.get("rc"))
        rc = None if rc in (None, "") else int(rc)
    except Exception:
        rc = row.get("returncode", row.get("rc"))
    elapsed = row.get("elapsed_s")
    try:
        elapsed_value = float(elapsed)
    except Exception:
        elapsed_value = max(0.0, ended - float(payload.get("started_at_epoch_s") or ended))
    payload.update({
        "ended_at_epoch_s": ended,
        "duration_s": elapsed_value,
        "elapsed_s": elapsed_value,
        "invocation_status": "returned" if returned and error is None else "raised",
        "call_status": "returned" if returned and error is None else "raised",
        "semantic_status": semantic,
        "terminal": True,
        "returncode": rc,
        "timed_out": bool(row.get("timed_out") or semantic == "timeout"),
        "timeout_kind": str(row.get("timeout_kind") or ("hard_timeout" if semantic == "timeout" else "")),
        "last_active_stage": phase,
        "compiler_phase": phase,
        "failure_kind": str(row.get("failure_kind") or ""),
        "unsupported_reason": str(row.get("unsupported_reason") or ""),
        "error_class": type(error).__name__ if error is not None else str(
            row.get("failure_kind") or row.get("timeout_kind") or row.get("error_class") or ""
        ),
        "error": str(error) if error is not None else str(row.get("error") or ""),
        "compiler_onnx": str(fixed_onnx or payload.get("compiler_onnx") or ""),
        "compiler_onnx_sha256": _sha256_file(fixed_onnx),
        "hef_path": str(hef or ""),
        "hef_sha256": _sha256_file(hef),
        "stdout_tail": str(stdout_tail or row.get("subprocess_stdout_tail") or "")[-16000:],
        "stderr_tail": str(stderr_tail or row.get("subprocess_stderr_tail") or row.get("error") or "")[-16000:],
        "details": row.get("details") if isinstance(row.get("details"), Mapping) else {},
        "result_summary": row,
    })
    outdir = Path(str(payload.get("outdir") or "."))
    receipts = outdir / "hailo_attempt_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    immutable = receipts / f"attempt_{payload['attempt_id']}.json"
    _atomic_json(immutable, payload, exclusive=True)
    _select_terminal(receipts)
    return immutable


@dataclass
class HailoAttempt:
    payload: dict[str, Any]

    @property
    def root(self) -> Path:
        return Path(str(self.payload["outdir"])) / "hailo_attempt_receipts"

    @property
    def attempt_id(self) -> str:
        return str(self.payload["attempt_id"])

    def heartbeat(self, *, compiler_phase: str, detail: str = "") -> Path:
        value = {
            "schema": "onnx-splitpoint/hailo-build-attempt-heartbeat",
            "schema_version": 1,
            "attempt_id": self.attempt_id,
            "compiler_phase": str(compiler_phase),
            "detail": str(detail),
            "updated_at_epoch_s": time.time(),
        }
        return _atomic_json(self.root / f"attempt_{self.attempt_id}.heartbeat.json", value)

    def finish(
        self, *, value: Any = None, error: BaseException | None = None,
        compiler_phase: str = "", stderr_tail: str = "", stdout_tail: str = "",
        returned: bool = True,
    ) -> Path:
        return finalize_hailo_attempt(
            attempt=self.payload, result=value, error=error,
            compiler_phase=compiler_phase, stderr_tail=stderr_tail,
            stdout_tail=stdout_tail, returned=returned,
        )


def start_hailo_attempt(
    output_dir: Path, *, attempt_kind: str, endpoint: str,
    source_onnx: Path | None = None, compiler_onnx: Path | None = None,
    end_nodes: Sequence[str] = (), timeout_policy: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HailoAttempt:
    timeout = dict(timeout_policy or {})
    payload = begin_hailo_attempt(outdir=Path(output_dir), bound={
        "attempt_kind": attempt_kind,
        "endpoint": endpoint,
        "onnx_path": str(source_onnx or ""),
        "compiler_onnx": str(compiler_onnx or source_onnx or ""),
        "end_node_names": list(end_nodes),
        "timeout_s": timeout.get("requested_timeout_s", timeout.get("hard_timeout_s")),
        "hard_timeout_s": timeout.get("hard_timeout_s"),
        "idle_timeout_s": timeout.get("idle_timeout_s"),
        "hard_timeout_enabled": timeout.get("hard_timeout_enabled"),
        "heartbeat_enabled": timeout.get("heartbeat_enabled", True),
        "manual_abort_enabled": timeout.get("manual_abort_enabled", True),
        "metadata": dict(metadata or {}),
    })
    return HailoAttempt(payload)


def select_terminal_attempt(root: Path, _candidate: Path | None = None) -> Path | None:
    root = Path(root)
    receipts = root if root.name == "hailo_attempt_receipts" else root / "hailo_attempt_receipts"
    return _select_terminal(receipts)
