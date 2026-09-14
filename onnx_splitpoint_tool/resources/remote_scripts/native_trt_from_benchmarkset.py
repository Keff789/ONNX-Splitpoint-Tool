#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TRT_ENGINE_BUILD_RECEIPT_SCHEMA = "onnx-splitpoint/tensorrt-engine-build-receipt"
TRT_ENGINE_BUILD_RECEIPT_VERSION = 1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import onnx  # type: ignore
except Exception as exc:  # pragma: no cover - target host dependency
    onnx = None  # type: ignore
    _ONNX_IMPORT_ERROR = exc
else:
    _ONNX_IMPORT_ERROR = None


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    elem_type: str
    has_dynamic: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_receipt_path(work_dir: Path) -> Path:
    return work_dir / "engine_build_receipt.json"


def _write_engine_build_receipt(
    *, work_dir: Path, command: list[str], source_onnx: Path,
    engine: Path, trtexec: str, returncode: int, dry_run: bool,
) -> dict[str, Any] | None:
    trtexec_path = Path(str(trtexec)).expanduser()
    if (
        dry_run or int(returncode) != 0 or not source_onnx.is_file()
        or not engine.is_file() or not trtexec_path.is_file()
    ):
        return None
    canonical_command: list[str] = []
    for index, value in enumerate(command):
        text = str(value)
        if index == 0:
            text = str(trtexec_path.resolve())
        elif text.startswith("--onnx="):
            text = f"--onnx={source_onnx.resolve()}"
        elif text.startswith("--saveEngine="):
            text = f"--saveEngine={engine.resolve()}"
        canonical_command.append(text)
    payload: dict[str, Any] = {
        "schema": TRT_ENGINE_BUILD_RECEIPT_SCHEMA,
        "schema_version": TRT_ENGINE_BUILD_RECEIPT_VERSION,
        "build_returncode": 0,
        "dry_run": False,
        "command": canonical_command,
        "source_onnx": str(source_onnx.resolve()),
        "source_onnx_sha256": _sha256_file(source_onnx),
        "engine": str(engine.resolve()),
        "engine_sha256": _sha256_file(engine),
        "trtexec": str(trtexec_path.resolve()),
        "trtexec_sha256": _sha256_file(trtexec_path),
    }
    payload["receipt_sha256"] = _canonical_json_sha256(payload)
    path = _build_receipt_path(work_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _verify_engine_build_receipt(
    raw: Any, *, source_onnx: Path, engine: Path, trtexec: str,
) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(raw, dict):
        return None, "receipt_missing"
    receipt = dict(raw)
    declared = str(receipt.pop("receipt_sha256", "") or "").strip().lower()
    if (
        len(declared) != 64 or any(ch not in "0123456789abcdef" for ch in declared)
        or _canonical_json_sha256(receipt) != declared
    ):
        return None, "receipt_integrity_mismatch"
    receipt["receipt_sha256"] = declared
    if (
        receipt.get("schema") != TRT_ENGINE_BUILD_RECEIPT_SCHEMA
        or receipt.get("schema_version") != TRT_ENGINE_BUILD_RECEIPT_VERSION
        or receipt.get("build_returncode") != 0
        or receipt.get("dry_run") is not False
    ):
        return None, "receipt_schema_or_status_invalid"
    trtexec_path = Path(str(trtexec)).expanduser()
    if not source_onnx.is_file():
        return None, "source_onnx_missing"
    if not engine.is_file():
        return None, "engine_not_found"
    if int(engine.stat().st_size) <= 0:
        return None, "engine_empty"
    if not trtexec_path.is_file():
        return None, "builder_not_found"
    if str(receipt.get("source_onnx") or "") != str(source_onnx.resolve()):
        return None, "source_onnx_path_mismatch"
    if str(receipt.get("source_onnx_sha256") or "") != _sha256_file(source_onnx):
        return None, "source_onnx_mismatch"
    if str(receipt.get("engine") or "") != str(engine.resolve()):
        return None, "engine_path_mismatch"
    if str(receipt.get("engine_sha256") or "") != _sha256_file(engine):
        return None, "engine_sha256_mismatch"
    if str(receipt.get("trtexec") or "") != str(trtexec_path.resolve()):
        return None, "builder_path_mismatch"
    if str(receipt.get("trtexec_sha256") or "") != _sha256_file(trtexec_path):
        return None, "builder_abi_mismatch"
    command = receipt.get("command")
    if not isinstance(command, list) or not command:
        return None, "receipt_command_missing"
    argv = [str(value) for value in command]
    if str(Path(argv[0]).expanduser().resolve()) != str(trtexec_path.resolve()):
        return None, "receipt_command_builder_mismatch"
    onnx_args = [value for value in argv[1:] if value.startswith("--onnx=")]
    engine_args = [value for value in argv[1:] if value.startswith("--saveEngine=")]
    if onnx_args != [f"--onnx={source_onnx.resolve()}"]:
        return None, "receipt_command_source_mismatch"
    if engine_args != [f"--saveEngine={engine.resolve()}"]:
        return None, "receipt_command_engine_mismatch"
    return receipt, "engine_build_receipt_verified"


def _command_build_contract(command: list[Any]) -> dict[str, Any] | None:
    """Extract engine-affecting options without invoking ``trtexec --help``."""
    if not isinstance(command, list) or not command:
        return None
    precision: list[str] = []
    shapes: list[str] = []
    workspace: list[tuple[str, int]] = []
    extra: list[str] = []
    for raw in command[1:]:
        value = str(raw)
        if value.startswith(("--onnx=", "--saveEngine=", "--timingCacheFile=")):
            continue
        if value in {"--fp16", "--int8"}:
            precision.append(value)
            continue
        if value.startswith("--shapes="):
            shapes.append(value.split("=", 1)[1])
            continue
        if value.startswith("--workspace="):
            try:
                workspace.append(("workspace", int(value.split("=", 1)[1])))
            except ValueError:
                return None
            continue
        if value.startswith("--memPoolSize=workspace:"):
            try:
                workspace.append(("mempool", int(value.rsplit(":", 1)[1])))
            except ValueError:
                return None
            continue
        # Verbosity does not change the serialized engine.
        if value == "--verbose":
            continue
        extra.append(value)
    if len(shapes) > 1 or len(workspace) > 1:
        return None
    return {
        "precision_flags": precision,
        "shapes": shapes[0] if shapes else "",
        "workspace": workspace[0] if workspace else None,
        "extra": extra,
    }


def _receipt_matches_build_request(
    receipt: dict[str, Any], *, precision: str, shapes: str,
    workspace_mb: int, workspace_mode: str, retry_without_shapes: bool,
    retry_workspace_alt: bool, extra_build_args: list[str],
) -> bool:
    """Check that a verified receipt represents the requested engine contract."""
    actual = _command_build_contract(list(receipt.get("command") or []))
    if actual is None:
        return False
    if actual["precision_flags"] != _precision_flags(precision):
        return False

    allowed_shapes = {str(shapes)}
    if bool(retry_without_shapes) and shapes:
        allowed_shapes.add("")
    if str(actual["shapes"]) not in allowed_shapes:
        return False

    mode = str(workspace_mode).strip().lower()
    expected_workspace: set[tuple[str, int] | None]
    if mode == "none":
        expected_workspace = {None}
    elif mode == "auto":
        expected_workspace = {
            ("workspace", int(workspace_mb)),
            ("mempool", int(workspace_mb)),
        }
    elif mode == "workspace":
        expected_workspace = {("workspace", int(workspace_mb))}
        if retry_workspace_alt:
            expected_workspace.add(("mempool", int(workspace_mb)))
    elif mode == "mempool":
        expected_workspace = {("mempool", int(workspace_mb))}
        if retry_workspace_alt:
            expected_workspace.add(("workspace", int(workspace_mb)))
    else:
        return False
    if actual["workspace"] not in expected_workspace:
        return False

    requested_extra = [
        str(value) for value in extra_build_args if str(value) != "--verbose"
    ]
    return actual["extra"] == requested_extra


def _verify_engine_cache_candidate(
    *, source_onnx: Path, engine: Path, receipt_path: Path, trtexec: str,
    precision: str, shapes: str, workspace_mb: int, workspace_mode: str,
    retry_without_shapes: bool, retry_workspace_alt: bool,
    extra_build_args: list[str], force_rebuild: bool = False,
    enforce_build_contract: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    if force_rebuild:
        return None, "forced_rebuild"
    if not engine.is_file() and not receipt_path.is_file():
        return None, "not_found"
    if not engine.is_file():
        return None, "engine_not_found"
    if not receipt_path.is_file():
        return None, "receipt_missing"
    raw = _load_json_safe(receipt_path)
    verified, reason = _verify_engine_build_receipt(
        raw, source_onnx=source_onnx, engine=engine, trtexec=trtexec,
    )
    if verified is None:
        return None, reason
    if enforce_build_contract and not _receipt_matches_build_request(
        verified,
        precision=precision,
        shapes=shapes,
        workspace_mb=workspace_mb,
        workspace_mode=workspace_mode,
        retry_without_shapes=retry_without_shapes,
        retry_workspace_alt=retry_workspace_alt,
        extra_build_args=extra_build_args,
    ):
        return None, "build_contract_mismatch"
    return verified, "verified"


def _benchmark_model_id(root: Path, source_onnx: Path) -> str:
    for candidate in (root / "benchmark_set.json", root / "benchmark_plan.json"):
        raw = _load_json_safe(candidate)
        if not isinstance(raw, dict):
            continue
        for key in ("model_id", "model_name", "model"):
            value = str(raw.get(key) or "").strip()
            if value:
                return value
    name = source_onnx.stem
    for suffix in ("_part1_native", "_part2_native", "_part1", "_part2"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name or "unknown"


def _log_trt_cache(
    status: str, *, role: str, model: str, case: str | None,
    reason: str, engine: Path, identity: str = "",
) -> None:
    fields = [
        f"[trt-cache] {status.upper()}",
        f"role={role}",
        f"model={model}",
        f"case={case or '-'}",
        f"reason={reason}",
    ]
    if identity:
        fields.append(f"identity={identity}")
    fields.append(f"engine={engine}")
    print(" ".join(fields), flush=True)


def _existing_cache_identity(engine: Path, receipt: Mapping[str, Any]) -> str:
    for part in reversed(engine.resolve().parts):
        if re.fullmatch(r"[0-9a-f]{64}", part.lower()):
            return part.lower()
    return str(receipt.get("receipt_sha256") or "")


def _probe_engine_candidates(
    *, role: str, case_id: str, cache_roots: Sequence[Path], precision: str,
) -> list[tuple[Path, str]]:
    role_norm = str(role).strip().lower()
    variant = "full" if role_norm in {"full", "trt_full"} else "part2"
    filename = f"{variant}_{precision}.engine"
    case_norm = _norm_case(case_id) if case_id else ""
    candidates: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for index, raw_root in enumerate(cache_roots):
        root = Path(raw_root).expanduser().resolve()
        namespace = "current" if index == 0 else "legacy"
        direct: list[Path] = []
        if root.name == filename:
            direct.append(root)
        elif variant == "full":
            direct.extend([
                root / "full" / precision / filename,
                *root.glob(f"full/*/{precision}/{filename}"),
                *root.glob(f"**/full/{precision}/{filename}"),
                *root.glob(f"**/full/*/{precision}/{filename}"),
            ])
        else:
            direct.extend([
                root / case_norm / "part2" / precision / filename,
                root / "splits" / case_norm / "part2" / precision / filename,
                *root.glob(
                    f"splits/{case_norm}/part2/*/{precision}/{filename}"
                ),
                *root.glob(f"{case_norm}/part2/*/{precision}/{filename}"),
                *root.glob(f"**/{case_norm}/part2/{precision}/{filename}"),
                *root.glob(f"**/{case_norm}/part2/*/{precision}/{filename}"),
            ])
        for engine in direct:
            resolved = engine.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append((resolved, namespace))
    return candidates


def probe_trt_cache(
    *, role: str, model_id: str, case_id: str = "", source_onnx: Path,
    cache_roots: Sequence[Path], precision: str = "fp16",
    expected_builder_abi: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Read-only receipt/engine/source/ABI probe used by cache preflight."""
    role_norm = str(role).strip().lower()
    if role_norm in {"full", "trt_full"}:
        canonical_role = "trt_full"
    elif role_norm in {"part2", "p2", "trt_p2"}:
        canonical_role = "trt_p2"
    else:
        return {
            "model_id": str(model_id),
            "role": "trt_p2",
            "item_id": str(case_id or ""),
            "status": "UNKNOWN",
            "reason": "probe_unavailable",
            "artifact_path": "",
            "receipt_path": "",
            "identity": "",
            "evidence": {"verification_reason": "unsupported_role"},
        }
    item_id = "full" if canonical_role == "trt_full" else _norm_case(case_id)
    base: dict[str, Any] = {
        "model_id": str(model_id),
        "role": canonical_role,
        "item_id": item_id,
        "status": "MISS",
        "reason": "not_found",
        "artifact_path": "",
        "receipt_path": "",
        "identity": "",
    }
    if canonical_role == "trt_p2" and not item_id:
        return {**base, "status": "UNKNOWN", "reason": "probe_unavailable"}
    source = Path(source_onnx).expanduser().resolve()
    if not source.is_file():
        return {
            **base,
            "status": "MISS",
            "reason": "source_onnx_mismatch",
            "evidence": {"verification_reason": "source_onnx_missing"},
        }

    candidates = _probe_engine_candidates(
        role=canonical_role,
        case_id=item_id,
        cache_roots=cache_roots,
        precision=str(precision),
    )
    failure_evidence: list[dict[str, Any]] = []
    for engine, namespace in candidates:
        receipt_path = _build_receipt_path(engine.parent)
        raw = _load_json_safe(receipt_path)
        declared_builder = str(
            (expected_builder_abi or {}).get("path")
            or (expected_builder_abi or {}).get("trtexec")
            or (expected_builder_abi or {}).get("trtexec_path")
            or ""
        ).strip()
        if not declared_builder and isinstance(raw, dict):
            declared_builder = str(raw.get("trtexec") or "").strip()
        if not declared_builder:
            failure_evidence.append({
                "artifact_path": str(engine),
                "receipt_path": str(receipt_path),
                "verification_reason": "builder_not_declared",
            })
            continue
        verified, verification_reason = _verify_engine_build_receipt(
            raw, source_onnx=source, engine=engine, trtexec=declared_builder,
        )
        if verified is not None and expected_builder_abi:
            expected_sha = str(
                expected_builder_abi.get("sha256")
                or expected_builder_abi.get("trtexec_sha256")
                or ""
            ).strip().lower()
            if expected_sha and expected_sha != str(
                verified.get("trtexec_sha256") or ""
            ).strip().lower():
                verified = None
                verification_reason = "builder_abi_mismatch"
        if verified is not None:
            return {
                **base,
                "status": "HIT",
                "reason": "compatible_receipt",
                "artifact_path": str(engine),
                "receipt_path": str(receipt_path),
                "identity": _existing_cache_identity(engine, verified),
                "source_namespace": namespace,
                "evidence": {
                    "verification_reason": "engine_build_receipt_verified",
                    "engine_sha256": str(verified.get("engine_sha256") or ""),
                    "receipt_sha256": str(verified.get("receipt_sha256") or ""),
                },
            }
        stable_reason = (
            "probe_unavailable"
            if verification_reason in {"builder_not_found", "builder_not_declared"}
            else "builder_abi_mismatch"
            if verification_reason in {
                "builder_abi_mismatch", "builder_path_mismatch",
                "receipt_command_builder_mismatch",
            }
            else "source_onnx_mismatch"
            if verification_reason in {
                "source_onnx_mismatch", "source_onnx_path_mismatch",
                "receipt_command_source_mismatch", "source_onnx_missing",
            }
            else "not_found"
        )
        failure_evidence.append({
            "artifact_path": str(engine),
            "receipt_path": str(receipt_path),
            "verification_reason": verification_reason,
            "stable_reason": stable_reason,
            "source_namespace": namespace,
        })

    reason = "legacy_candidate_not_found" if len(cache_roots) > 1 else "not_found"
    for preferred in ("builder_abi_mismatch", "source_onnx_mismatch"):
        if any(row.get("stable_reason") == preferred for row in failure_evidence):
            reason = preferred
            break
    status = "MISS"
    if failure_evidence and all(
        row.get("stable_reason") == "probe_unavailable"
        for row in failure_evidence
    ):
        status = "UNKNOWN"
        reason = "probe_unavailable"
    first = failure_evidence[0] if failure_evidence else {}
    result = {
        **base,
        "status": status,
        "reason": reason,
        "artifact_path": str(first.get("artifact_path") or ""),
        "receipt_path": str(first.get("receipt_path") or ""),
        "evidence": {"candidates": failure_evidence},
    }
    if first.get("source_namespace"):
        result["source_namespace"] = first["source_namespace"]
    return result


def _norm_case(case: str) -> str:
    case = str(case).strip()
    if not case:
        return case
    if case.startswith("b"):
        digits = case[1:]
    else:
        digits = case
    if digits.isdigit():
        return f"b{int(digits):03d}"
    return case


def _find_trtexec(user_path: str = "") -> str:
    if user_path:
        p = Path(user_path).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        raise FileNotFoundError(f"trtexec not executable: {p}")
    found = shutil.which("trtexec")
    if found:
        return found
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    for c in candidates:
        if Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError("trtexec not found in PATH or standard TensorRT locations")




def _trtexec_help(trtexec: str) -> str:
    try:
        proc = subprocess.run([trtexec, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return proc.stdout or ""
    except Exception:
        return ""


def _workspace_args(trtexec: str, workspace_mb: int, mode: str = "auto") -> list[str]:
    if workspace_mb <= 0:
        return []
    mode = (mode or "auto").lower()
    if mode == "none":
        return []
    if mode == "workspace":
        return [f"--workspace={workspace_mb}"]
    if mode == "mempool":
        return [f"--memPoolSize=workspace:{workspace_mb}"]
    # TRT 10 prefers --memPoolSize, older Jetson TRT 8 commonly accepts --workspace.
    help_text = _trtexec_help(trtexec)
    if "--memPoolSize" in help_text:
        return [f"--memPoolSize=workspace:{workspace_mb}"]
    return [f"--workspace={workspace_mb}"]


def _log_tail(path: Path, lines: int = 120) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    parts = txt.splitlines()
    return "\n".join(parts[-lines:])


def _failure_hint(text: str) -> str:
    t = text.lower()
    if "unknown option" in t or "unrecognized option" in t or "invalid argument" in t and "workspace" in t:
        return "trtexec argument mismatch; try --workspace-mode mempool or workspace."
    if "could not parse onnx" in t or "failed to parse onnx" in t or "model importer" in t:
        return "ONNX parse/import failure; inspect unsupported operator, plugin requirement, or external data."
    if "no importer registered" in t:
        return "TensorRT unsupported ONNX operator or missing plugin."
    if "static model does not take explicit shapes" in t:
        return "TensorRT static-shape ONNX: omit --shapes/--minShapes/--optShapes/--maxShapes."
    if "profile" in t and "shape" in t:
        return "TensorRT shape/profile issue; try --no-shapes or explicit --shape-override."
    if "out of memory" in t or "insufficient memory" in t:
        return "TensorRT build memory issue; reduce workspace or close GPU processes."
    return "see build_trtexec.log tail."

def _onnx_inputs(path: Path) -> list[TensorSpec]:
    if onnx is None:
        raise RuntimeError(f"onnx import failed: {_ONNX_IMPORT_ERROR}")
    model = onnx.load(str(path), load_external_data=False)
    initializer_names = {init.name for init in model.graph.initializer}
    specs: list[TensorSpec] = []
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        t = value.type.tensor_type
        dims: list[int] = []
        has_dynamic = False
        for d in t.shape.dim:
            if d.dim_value and int(d.dim_value) > 0:
                dims.append(int(d.dim_value))
            else:
                # Keep native TensorRT smoke tests static. Unknown dims are usually batch.
                dims.append(1)
                has_dynamic = True
        elem = onnx.TensorProto.DataType.Name(t.elem_type) if t.elem_type else "UNKNOWN"
        specs.append(TensorSpec(value.name, dims, elem, has_dynamic))
    return specs


def _onnx_outputs(path: Path) -> list[TensorSpec]:
    if onnx is None:
        raise RuntimeError(f"onnx import failed: {_ONNX_IMPORT_ERROR}")
    model = onnx.load(str(path), load_external_data=False)
    specs: list[TensorSpec] = []
    for value in model.graph.output:
        t = value.type.tensor_type
        dims: list[int] = []
        has_dynamic = False
        for d in t.shape.dim:
            if d.dim_value and int(d.dim_value) > 0:
                dims.append(int(d.dim_value))
            else:
                dims.append(1)
                has_dynamic = True
        elem = onnx.TensorProto.DataType.Name(t.elem_type) if t.elem_type else "UNKNOWN"
        specs.append(TensorSpec(value.name, dims, elem, has_dynamic))
    return specs




def _make_uint8_cast_bridge_onnx(src: Path, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    if onnx is None:
        raise RuntimeError(f"onnx import failed: {_ONNX_IMPORT_ERROR}")
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_uint8_cast_bridge.onnx"
    meta_path = out_dir / "uint8_cast_bridge_meta.json"
    source_sha256 = _sha256_file(src)
    try:
        src_mtime = float(src.stat().st_mtime)
        if out.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta.get("source") == str(src)
                and str(meta.get("source_sha256") or "") == source_sha256
                and out.is_file()
                and str(meta.get("bridge_sha256") or "") == _sha256_file(out)
            ):
                return out, meta
    except Exception:
        pass
    model = onnx.load(str(src), load_external_data=True)
    init_names = {i.name for i in model.graph.initializer}
    inputs = [v for v in model.graph.input if v.name not in init_names]
    if not inputs:
        raise RuntimeError(f"cannot create uint8 cast bridge: no graph input in {src}")
    inp = inputs[0]
    input_name = str(inp.name)
    cast_out = input_name + "__uint8_cast_to_float"
    replaced = 0
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == input_name:
                node.input[i] = cast_out
                replaced += 1
    if replaced <= 0:
        raise RuntimeError(f"cannot create uint8 cast bridge: input {input_name!r} is not consumed")
    inp.type.tensor_type.elem_type = onnx.TensorProto.UINT8
    cast_node = onnx.helper.make_node("Cast", inputs=[input_name], outputs=[cast_out], name=input_name + "__cast_uint8_to_float", to=int(onnx.TensorProto.FLOAT))
    model.graph.node.insert(0, cast_node)
    try:
        model.producer_name = (model.producer_name or "kmd-onnx-split") + "+uint8_bridge"
    except Exception:
        pass
    onnx.save(model, str(out))
    meta = {
        "schema": "onnx-splitpoint/uint8-cast-bridge",
        "schema_version": 1,
        "source": str(src),
        "source_sha256": source_sha256,
        "source_mtime": float(src.stat().st_mtime),
        "bridge": str(out),
        "bridge_sha256": _sha256_file(out),
        "input_name": input_name,
        "input_dtype": "UINT8",
        "cast_output": cast_out,
        "cast_to": "FLOAT",
        "replaced_uses": int(replaced),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta


def _load_json_safe(path: Path) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _walk_numbers(obj: Any):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k).lower(), v
            yield from _walk_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_numbers(v)


def _infer_dequant_params(root: Path, case: str | None, input_name: str, explicit_scale: float = 0.0, explicit_zp: float = 0.0) -> dict[str, Any]:
    # Conservative best-effort extractor.  Prefer explicit scale/zero_point in
    # output_contract/quant metadata.  Otherwise use activation min/max stats.
    if explicit_scale and explicit_scale > 0:
        return {"scale": float(explicit_scale), "zero_point": float(explicit_zp), "source": "explicit"}
    cdir = root / str(case or "") if case else root
    candidates = []
    for pat in ["**/output_contract.json", "**/*quant*.json", "**/activation_calibration/stats.json", "**/stats.json"]:
        candidates.extend(sorted(cdir.glob(pat)))
    best = {"scale": None, "zero_point": None, "source": "fallback_1_over_255"}
    mins=[]; maxs=[]
    for p in candidates:
        j=_load_json_safe(p)
        if j is None: continue
        # Direct scale/zp fields anywhere near the tensor name.
        txt = json.dumps(j)[:200000]
        if input_name and input_name in txt:
            # Try nested fields by scanning keys.
            vals = dict(_walk_numbers(j))
            scale = None; zp = None
            for k,v in vals.items():
                if k in ("scale","quant_scale","output_scale","dequant_scale"):
                    try: scale=float(v)
                    except Exception: pass
                if k in ("zero_point","zeropoint","zp","quant_zero_point","output_zero_point"):
                    try: zp=float(v)
                    except Exception: pass
            if scale and scale > 0:
                return {"scale": scale, "zero_point": 0.0 if zp is None else zp, "source": str(p)}
        for k,v in _walk_numbers(j):
            if k in ("min","minimum","amin"):
                try: mins.append(float(v))
                except Exception: pass
            if k in ("max","maximum","amax"):
                try: maxs.append(float(v))
                except Exception: pass
    if mins and maxs:
        mn=min(mins); mx=max(maxs)
        if mx > mn:
            # symmetricish uint8 activation approximation
            scale=(mx-mn)/255.0
            zp=round(max(0.0, min(255.0, -mn/scale))) if scale > 0 else 0.0
            return {"scale": float(scale), "zero_point": float(zp), "source": "activation_stats_minmax", "min": mn, "max": mx}
    return {"scale": 1.0/255.0, "zero_point": 0.0, "source": "fallback_1_over_255"}



def _value_info_static_shape(value: Any) -> list[int] | None:
    """Return static tensor shape from an ONNX ValueInfoProto, or None."""
    try:
        dims = value.type.tensor_type.shape.dim
    except Exception:
        return None
    out: list[int] = []
    for d in dims:
        try:
            v = int(getattr(d, "dim_value", 0) or 0)
        except Exception:
            v = 0
        if v <= 0:
            return None
        out.append(v)
    return out or None


def _normalise_boundary_layout(layout: str) -> str:
    v = str(layout or "").strip().lower().replace("-", "_")
    aliases = {
        "": "as_input",
        "auto": "as_input",
        "none": "as_input",
        "identity": "as_input",
        "nchw": "as_input",
        "as_target_shape": "as_input",
        "as_manifest_shape": "as_input",
        "nhwc_to_nchw": "memory_nhwc_to_nchw",
        "nwc_to_ncw": "memory_nwc_to_ncw",
        "hwcn_to_nchw": "memory_hwcn_to_nchw",
        "nwhc_to_nchw": "memory_nwhc_to_nchw",
        "ncwh_to_nchw": "memory_ncwh_to_nchw",
        "chwn_to_nchw": "memory_chwn_to_nchw",
    }
    return aliases.get(v, v)


def _make_boundary_layout_nodes(input_name: str, tensor_name: str, input_shape: list[int] | None, layout: str) -> tuple[str, list[Any], list[Any], dict[str, Any]]:
    """Create ONNX nodes that reinterpret raw Hailo boundary memory before Part2.

    The native FIFO copies the Hailo VStream byte buffer directly into the TensorRT
    input binding.  For many Hailo outputs the byte order is HWC/NHWC-like even
    when the ONNX Part2 input is canonical NCHW.  A Reshape followed by Transpose
    converts that raw memory contract into the original ONNX tensor contract.
    """
    layout_n = _normalise_boundary_layout(layout)
    meta: dict[str, Any] = {"requested": str(layout or "as_input"), "effective": layout_n, "applied": False}
    if layout_n == "as_input":
        return tensor_name, [], [], meta
    import numpy as _np

    if layout_n == "memory_nwc_to_ncw":
        if not input_shape or len(input_shape) != 3 or min(input_shape) <= 0:
            raise RuntimeError(f"boundary layout {layout!r} requires a positive static 3D Part2 input shape, got {input_shape}")
        n, c, w = [int(x) for x in input_shape]
        if c == w:
            raise RuntimeError("native_split_quality_boundary_layout_shape_ambiguous")
        mem_shape = [n, w, c]
        perm = [0, 2, 1]
    else:
        if not input_shape or len(input_shape) != 4:
            raise RuntimeError(f"boundary layout {layout!r} requires a static 4D Part2 input shape, got {input_shape}")
        n, c, h, w = [int(x) for x in input_shape]
        if min(n, c, h, w) <= 0:
            raise RuntimeError(f"boundary layout {layout!r} requires positive static dims, got {input_shape}")
        if layout_n == "memory_nhwc_to_nchw":
            mem_shape = [n, h, w, c]
            perm = [0, 3, 1, 2]
        elif layout_n == "memory_hwcn_to_nchw":
            mem_shape = [h, w, c, n]
            perm = [3, 2, 0, 1]
        elif layout_n == "memory_nwhc_to_nchw":
            mem_shape = [n, w, h, c]
            perm = [0, 3, 2, 1]
        elif layout_n == "memory_ncwh_to_nchw":
            mem_shape = [n, c, w, h]
            perm = [0, 1, 3, 2]
        elif layout_n == "memory_chwn_to_nchw":
            mem_shape = [c, h, w, n]
            perm = [3, 0, 1, 2]
        else:
            raise RuntimeError(f"unsupported boundary layout: {layout!r}")

    shape_name = input_name + "__boundary_mem_shape"
    reshape_out = input_name + "__boundary_mem_" + layout_n.replace("memory_", "")
    transpose_out = input_name + ("__boundary_to_ncw" if len(perm) == 3 else "__boundary_to_nchw")
    shape_init = onnx.numpy_helper.from_array(_np.asarray(mem_shape, dtype=_np.int64), name=shape_name)
    nodes = [
        onnx.helper.make_node("Reshape", inputs=[tensor_name, shape_name], outputs=[reshape_out], name=input_name + "__boundary_memory_reshape"),
        onnx.helper.make_node("Transpose", inputs=[reshape_out], outputs=[transpose_out], name=input_name + "__boundary_memory_to_nchw", perm=perm),
    ]
    meta.update({"applied": True, "memory_shape": mem_shape, "perm": perm, "output": transpose_out})
    return transpose_out, nodes, [shape_init], meta

def _make_uint8_dequant_bridge_onnx(src: Path, out_dir: Path, *, root: Path, case: str | None, scale: float = 0.0, zero_point: float = 0.0, boundary_layout: str = "as_input") -> tuple[Path, dict[str, Any]]:
    if onnx is None:
        raise RuntimeError(f"onnx import failed: {_ONNX_IMPORT_ERROR}")
    import numpy as np
    src = Path(src); out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_uint8_dequant_bridge.onnx"
    meta_path = out_dir / "uint8_dequant_bridge_meta.json"
    model = onnx.load(str(src), load_external_data=True)
    init_names = {i.name for i in model.graph.initializer}
    inputs = [v for v in model.graph.input if v.name not in init_names]
    if not inputs:
        raise RuntimeError(f"cannot create uint8 dequant bridge: no graph input in {src}")
    inp=inputs[0]; input_name=str(inp.name)
    input_shape = _value_info_static_shape(inp)
    params=_infer_dequant_params(root, case, input_name, scale, zero_point)
    sc=float(params.get('scale') or (1.0/255.0)); zp=float(params.get('zero_point') or 0.0)
    cast_out=input_name+'__uint8_cast_to_float'
    sub_out=input_name+'__dequant_sub_zp'
    deq_out=input_name+'__dequant_float'
    bridge_out, layout_nodes, layout_inits, layout_meta = _make_boundary_layout_nodes(input_name, deq_out, input_shape, boundary_layout)
    replaced=0
    for node in model.graph.node:
        for i,name in enumerate(node.input):
            if name == input_name:
                node.input[i]=bridge_out; replaced += 1
    if replaced <= 0:
        raise RuntimeError(f"cannot create uint8 dequant bridge: input {input_name!r} is not consumed")
    inp.type.tensor_type.elem_type = onnx.TensorProto.UINT8
    scale_name=input_name+'__dequant_scale_const'; zp_name=input_name+'__dequant_zp_const'
    scale_init=onnx.numpy_helper.from_array(np.array(sc, dtype=np.float32), name=scale_name)
    zp_init=onnx.numpy_helper.from_array(np.array(zp, dtype=np.float32), name=zp_name)
    model.graph.initializer.extend([scale_init, zp_init] + list(layout_inits))
    nodes=[
        onnx.helper.make_node('Cast', inputs=[input_name], outputs=[cast_out], name=input_name+'__cast_uint8_to_float', to=int(onnx.TensorProto.FLOAT)),
        onnx.helper.make_node('Sub', inputs=[cast_out, zp_name], outputs=[sub_out], name=input_name+'__sub_zero_point'),
        onnx.helper.make_node('Mul', inputs=[sub_out, scale_name], outputs=[deq_out], name=input_name+'__mul_scale'),
    ] + list(layout_nodes)
    for n in reversed(nodes):
        model.graph.node.insert(0, n)
    try: model.producer_name=(model.producer_name or 'kmd-onnx-split') + '+uint8_dequant_bridge'
    except Exception: pass
    onnx.save(model, str(out))
    meta={"schema":"onnx-splitpoint/uint8-dequant-bridge","schema_version":2,"source":str(src),"source_sha256":_sha256_file(src),"bridge":str(out),"bridge_sha256":_sha256_file(out),"input_name":input_name,"input_dtype":"UINT8","input_shape":input_shape,"dequant_output":deq_out,"bridge_output":bridge_out,"boundary_layout":layout_meta,"scale":sc,"zero_point":zp,"params_source":params.get('source'),"params":params,"replaced_uses":int(replaced)}
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return out, meta




def _make_float32_layout_bridge_onnx(src: Path, out_dir: Path, *, boundary_layout: str = "as_input") -> tuple[Path, dict[str, Any]]:
    """Create a FLOAT32-input Part2 bridge that only fixes raw boundary memory layout.

    This is the safest diagnostic variant for HailoRT FLOAT32 output: HailoRT
    performs the output dequantization, then this bridge reinterprets the raw
    vstream memory order (for example NHWC/HWC) and transposes back to the
    canonical ONNX Part2 NCHW contract. It deliberately does not apply any
    manual scale/zero-point.
    """
    if onnx is None:
        raise RuntimeError(f"onnx import failed: {_ONNX_IMPORT_ERROR}")
    src = Path(src); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_float32_layout_bridge.onnx"
    meta_path = out_dir / "float32_layout_bridge_meta.json"
    model = onnx.load(str(src), load_external_data=True)
    init_names = {i.name for i in model.graph.initializer}
    inputs = [v for v in model.graph.input if v.name not in init_names]
    if not inputs:
        raise RuntimeError(f"cannot create float32 layout bridge: no graph input in {src}")
    inp = inputs[0]
    input_name = str(inp.name)
    input_shape = _value_info_static_shape(inp)
    bridge_out, layout_nodes, layout_inits, layout_meta = _make_boundary_layout_nodes(input_name, input_name, input_shape, boundary_layout)
    replaced = 0
    if layout_meta.get("applied"):
        for node in model.graph.node:
            for i, name in enumerate(node.input):
                if name == input_name:
                    node.input[i] = bridge_out
                    replaced += 1
        if replaced <= 0:
            raise RuntimeError(f"cannot create float32 layout bridge: input {input_name!r} is not consumed")
        model.graph.initializer.extend(list(layout_inits))
        for n in reversed(list(layout_nodes)):
            model.graph.node.insert(0, n)
        try:
            model.producer_name = (model.producer_name or 'kmd-onnx-split') + '+float32_layout_bridge'
        except Exception:
            pass
    onnx.save(model, str(out))
    meta = {
        "schema": "onnx-splitpoint/float32-layout-bridge",
        "schema_version": 1,
        "source": str(src),
        "source_sha256": _sha256_file(src),
        "bridge": str(out),
        "bridge_sha256": _sha256_file(out),
        "input_name": input_name,
        "input_dtype": "FLOAT",
        "input_shape": input_shape,
        "bridge_output": bridge_out,
        "boundary_layout": layout_meta,
        "params_source": "hailort_float32_vstream",
        "replaced_uses": int(replaced),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return out, meta

def _shape_arg(specs: list[TensorSpec], override: str = "") -> str:
    if override:
        return override
    return ",".join(f"{s.name}:{'x'.join(str(int(v)) for v in s.shape)}" for s in specs if s.shape)


def _all_inputs_static(specs: list[TensorSpec]) -> bool:
    return bool(specs) and all(not bool(getattr(s, "has_dynamic", False)) for s in specs)


def _find_full_onnx(root: Path) -> Path | None:
    models = root / "models"
    if models.is_dir():
        cands = sorted(models.glob("*.onnx"))
        if cands:
            return cands[0]
    cands = [p for p in sorted(root.glob("*.onnx")) if "part" not in p.name.lower()]
    return cands[0] if cands else None


def _find_case_dir(root: Path, case: str) -> Path:
    norm = _norm_case(case)
    candidates = [root / norm]
    if norm.startswith("b"):
        candidates.append(root / f"b{int(norm[1:])}")
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"case directory not found for {case} below {root}")


def _find_part_onnx(case_dir: Path, part: str) -> Path:
    part = part.lower()
    patterns = [f"*_{part}_*.onnx", f"*{part}*.onnx"]
    for pat in patterns:
        cands = sorted(case_dir.glob(pat))
        if cands:
            return cands[0]
    cands = sorted(case_dir.glob("*.onnx"))
    for c in cands:
        if part in c.name.lower():
            return c
    raise FileNotFoundError(f"{part} ONNX not found in {case_dir}")


def _target_path(root: Path, case: str | None, variant: str, precision: str, out_dir: Path | None) -> tuple[Path, Path]:
    if out_dir is None:
        out_dir = root / "native_trt"
    if case:
        d = out_dir / _norm_case(case) / variant / precision
    else:
        d = out_dir / "full" / precision
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{variant}_{precision}.engine", d


def _precision_flags(precision: str) -> list[str]:
    p = precision.lower()
    if p == "fp32":
        return []
    if p in {"fp16", "uint8_cast_fp16", "uint8_dequant_fp16", "float32_layout_fp16"}:
        return ["--fp16"]
    if p == "int8":
        # Works only for explicit-Q/DQ ONNX or TRT setups that support int8 without external calibrator.
        return ["--int8"]
    raise ValueError(f"unsupported precision: {precision}")


def _strict_warm_trt_build_forbidden(kind: str, *, case_id: str = "", generic_part2: bool = False) -> bool:
    if not str(os.environ.get("ONNX_SPLITPOINT_TRT_BUILD_GUARD") or "").strip():
        return False
    try:
        from splitpoint_runners.native_split_quality_runtime import trt_build_forbidden
    except ImportError:
        from onnx_splitpoint_tool.runners.native_split_quality_runtime import trt_build_forbidden
    return trt_build_forbidden(kind, case_id=case_id, generic_part2=generic_part2)


def _run(cmd: list[str], *, cwd: Path, log_path: Path, dry_run: bool, artifact_role: str = "", case_id: str = "", generic_part2: bool = False) -> dict[str, Any]:
    meta: dict[str, Any] = {"cmd": cmd, "cwd": str(cwd), "log_path": str(log_path), "dry_run": dry_run}
    policy = str(os.environ.get("ONNX_SPLITPOINT_ARTIFACT_POLICY") or "normal").strip().lower().replace("-", "_")
    build_tokens = ("--onnx", "--saveEngine", "--buildOnly")
    if policy == "cache_verify_only" and any(
        str(arg).startswith(build_tokens) for arg in cmd
    ):
        message = (
            "cache_miss_blocked[tensorrt_build]: cache_verify_only forbids "
            "trtexec build commands"
        )
        log_path.write_text(message + "\n", encoding="utf-8")
        meta.update({
            "returncode": 78,
            "elapsed_s": 0.0,
            "status": "cache_miss_blocked",
            "error": message,
        })
        return meta
    if any(str(arg).startswith(build_tokens) for arg in cmd) and _strict_warm_trt_build_forbidden(artifact_role, case_id=case_id, generic_part2=generic_part2):
        message = "cache_miss_blocked:artifact_policy=strict_warm_cache:compiler=trtexec"
        log_path.write_text(message + "\n", encoding="utf-8")
        meta.update({"returncode": 78, "elapsed_s": 0.0, "status": "cache_miss_blocked", "error": message, "compiler_dispatched": False})
        return meta
    if dry_run:
        log_path.write_text("DRY RUN\n" + " ".join(cmd) + "\n", encoding="utf-8")
        meta.update({"returncode": 0, "elapsed_s": 0.0})
        return meta
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    log_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
    meta.update({"returncode": int(proc.returncode), "elapsed_s": float(elapsed)})
    # Common trtexec parse hints.
    text = proc.stdout or ""
    m = re.search(r"Throughput:\s*([0-9.]+)\s*qps", text)
    if m:
        meta["throughput_qps"] = float(m.group(1))
    m = re.search(r"Latency:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms", text)
    if m:
        meta["latency_ms"] = {"min": float(m.group(1)), "max": float(m.group(2)), "mean": float(m.group(3))}
    return meta


def _engine_build_cmd(
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    shapes: str,
    timing_cache: Path,
    workspace_mb: int,
    workspace_mode: str,
    extra: list[str],
) -> list[str]:
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    cmd.extend(_precision_flags(precision))
    if shapes:
        cmd.append(f"--shapes={shapes}")
    if timing_cache:
        cmd.append(f"--timingCacheFile={timing_cache}")
    cmd.extend(_workspace_args(trtexec, int(workspace_mb), workspace_mode))
    cmd.extend(extra)
    return cmd


def _engine_run_cmd(trtexec: str, engine_path: Path, shapes: str, iterations: int, warmup_ms: int, duration_s: int, extra: list[str]) -> list[str]:
    cmd = [trtexec, f"--loadEngine={engine_path}"]
    if shapes:
        cmd.append(f"--shapes={shapes}")
    if iterations > 0:
        cmd.append(f"--iterations={iterations}")
    cmd.append(f"--warmUp={max(0, int(warmup_ms))}")
    # Pin this explicitly even for iteration-controlled measurements.  Without
    # ``--duration=0`` trtexec may continue until its default minimum duration
    # is reached, making ``--iterations=N`` only a lower bound.
    cmd.append(f"--duration={max(0, int(duration_s))}")
    cmd.extend(extra)
    return cmd


def _same_process_warmup_contract(
    warmup_iterations: int, warmup_ms: int,
) -> dict[str, Any]:
    """Resolve the only warmup supported by one trtexec process/context."""
    requested_iterations = max(0, int(warmup_iterations))
    effective_ms = max(0, int(warmup_ms))
    source = "explicit_warmup_ms"
    if effective_ms <= 0 and requested_iterations > 0:
        # Iterations and milliseconds are not equivalent units.  trtexec only
        # exposes an internal time-window warmup in the same process/context,
        # so use at least its conventional 200 ms default and disclose that no
        # cross-backend work-count equivalence can be established.
        effective_ms = 200
        source = "requested_iterations_trigger_conservative_trtexec_200ms_minimum"
    return {
        "warmup_iterations_requested": requested_iterations,
        "warmup_iterations_completed": None,
        "warmup_iterations_status": (
            "not_countable_trtexec_internal_time_window"
            if effective_ms > 0 else "disabled"
        ),
        "warmup_ms_effective": effective_ms,
        "warmup_budget_source": source,
        "warmup_work_equivalence": "cross_backend_not_iteration_equivalent",
        "warmup_iterations_observed": False,
        "warmup_attestation_basis": "trtexec_single_invocation_cli_contract",
        "warmup_policy": (
            "same_trtexec_process_internal_time_window_drained_before_measurement"
            if effective_ms > 0 else "disabled"
        ),
        "warmup_and_measurement_same_process": True,
        "warmup_and_measurement_same_execution_context": True,
    }


def _percentile(values: list[float], q: float) -> float | None:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, float(q))) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _trtexec_export_times_evidence(path: Path) -> dict[str, Any]:
    """Read exact measured-query evidence emitted by ``--exportTimes``.

    TensorRT releases have emitted either a top-level list or a mapping holding
    that list.  Both are accepted, but the requested iteration count is never
    substituted for missing runtime evidence.
    """
    result: dict[str, Any] = {
        "path": str(path), "status": "missing",
        "completed_work_units": None, "latency_samples_ms": [],
    }
    if not path.is_file():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.update({"status": "invalid_json", "error": f"{type(exc).__name__}: {exc}"})
        return result
    records: list[Any] | None = payload if isinstance(payload, list) else None
    if isinstance(payload, dict):
        for key in ("times", "queries", "records", "results", "data"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
    if records is None:
        result["status"] = "unsupported_schema"
        return result
    latency_samples: list[float] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        value = next((record.get(key) for key in (
            "latencyMs", "latency_ms", "latency", "endToEndMs", "end_to_end_ms",
        ) if record.get(key) not in (None, "")), None)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric >= 0.0:
            latency_samples.append(numeric)
    result.update({
        "status": "ok",
        "completed_work_units": len(records),
        "latency_samples_ms": latency_samples,
        "latency_sample_count": len(latency_samples),
        "latency_mean_ms": statistics.fmean(latency_samples) if latency_samples else None,
        "latency_p50_ms": _percentile(latency_samples, 0.50),
        "latency_p95_ms": _percentile(latency_samples, 0.95),
    })
    return result


def _artifact_specs(root: Path, cases: list[str], variants: list[str]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for variant in variants:
        v = variant.lower()
        if v == "full":
            full = _find_full_onnx(root)
            if full is None:
                raise FileNotFoundError(f"full ONNX not found below {root}")
            artifacts.append({"case": None, "variant": "full", "onnx": full})
        elif v in {"part1", "part2"}:
            for case in cases:
                cdir = _find_case_dir(root, case)
                artifacts.append({"case": _norm_case(case), "variant": v, "onnx": _find_part_onnx(cdir, v)})
        else:
            raise ValueError(f"unsupported variant: {variant}")
    return artifacts


def _explicit_full_quality_identity(args: argparse.Namespace, trtexec: str) -> dict[str, Any] | None:
    """Verify every quality-sealed artifact before an explicit Full reuse."""
    option_names = (
        "explicit_full_source_onnx", "explicit_full_build_onnx",
        "explicit_full_engine", "explicit_full_build_receipt",
        "expected_source_onnx_sha256", "expected_build_onnx_sha256",
        "expected_engine_sha256", "expected_trtexec_sha256",
        "expected_engine_build_receipt_sha256",
        "expected_engine_build_receipt_file_sha256",
        "expected_trt_engine_build_receipt_sha256",
        "quality_first_producer_identity_sha256",
    )
    supplied = [bool(str(getattr(args, name, "") or "").strip()) for name in option_names]
    if not any(supplied):
        return None
    if not all(supplied):
        raise RuntimeError("explicit Full TensorRT quality identity is incomplete")
    if bool(args.build) or [str(value).strip().lower() for value in str(args.variants).split(",") if str(value).strip()] != ["full"]:
        raise RuntimeError("explicit Full TensorRT reuse requires --no-build and --variants full")

    paths = {
        "source_onnx": Path(args.explicit_full_source_onnx).expanduser().resolve(),
        "build_onnx": Path(args.explicit_full_build_onnx).expanduser().resolve(),
        "engine": Path(args.explicit_full_engine).expanduser().resolve(),
        "trtexec": Path(str(trtexec)).expanduser().resolve(),
        "engine_build_receipt": Path(args.explicit_full_build_receipt).expanduser().resolve(),
    }
    expected_hashes = {
        "source_onnx": str(args.expected_source_onnx_sha256).strip().lower(),
        "build_onnx": str(args.expected_build_onnx_sha256).strip().lower(),
        "engine": str(args.expected_engine_sha256).strip().lower(),
        "trtexec": str(args.expected_trtexec_sha256).strip().lower(),
        "engine_build_receipt": str(args.expected_engine_build_receipt_file_sha256).strip().lower(),
    }
    expected_sizes = {
        "source_onnx": int(args.expected_source_onnx_size_bytes or 0),
        "build_onnx": int(args.expected_build_onnx_size_bytes or 0),
        "engine": int(args.expected_engine_size_bytes or 0),
        "trtexec": int(args.expected_trtexec_size_bytes or 0),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"explicit Full TensorRT {name} is missing: {path}")
        expected_hash = expected_hashes[name]
        if len(expected_hash) != 64 or _sha256_file(path) != expected_hash:
            raise RuntimeError(f"explicit Full TensorRT {name} SHA-256 mismatch")
        if name in expected_sizes and (
            expected_sizes[name] <= 0 or int(path.stat().st_size) != expected_sizes[name]
        ):
            raise RuntimeError(f"explicit Full TensorRT {name} size mismatch")
    if expected_hashes["source_onnx"] != expected_hashes["build_onnx"]:
        raise RuntimeError("explicit Full source/build ONNX hash mismatch")
    producer_sha = str(
        args.quality_first_producer_identity_sha256 or ""
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", producer_sha) is None:
        raise RuntimeError("explicit Full TensorRT producer identity SHA-256 is invalid")

    receipt = _load_json_safe(paths["engine_build_receipt"])
    verified_receipt, receipt_status = _verify_engine_build_receipt(
        receipt,
        source_onnx=paths["build_onnx"],
        engine=paths["engine"],
        trtexec=str(paths["trtexec"]),
    )
    if verified_receipt is None:
        raise RuntimeError(f"explicit Full TensorRT receipt invalid: {receipt_status}")
    inner_sha = str(verified_receipt.get("receipt_sha256") or "").strip().lower()
    outer_sha = _canonical_json_sha256(verified_receipt)
    canonical_receipt = json.dumps(
        verified_receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    if inner_sha != str(args.expected_trt_engine_build_receipt_sha256).strip().lower():
        raise RuntimeError("explicit Full TensorRT inner receipt SHA-256 mismatch")
    if outer_sha != str(args.expected_engine_build_receipt_sha256).strip().lower():
        raise RuntimeError("explicit Full TensorRT outer receipt SHA-256 mismatch")
    if (
        int(args.expected_engine_build_receipt_size_bytes or 0) <= 0
        or len(canonical_receipt) != int(args.expected_engine_build_receipt_size_bytes)
    ):
        raise RuntimeError("explicit Full TensorRT canonical receipt size mismatch")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": expected_hashes,
        "sizes": expected_sizes,
        "engine_build_receipt_sha256": outer_sha,
        "engine_build_receipt_file_sha256": expected_hashes["engine_build_receipt"],
        "trt_engine_build_receipt_sha256": inner_sha,
        "engine_build_receipt_size_bytes": len(canonical_receipt),
        "quality_first_producer_identity_sha256": producer_sha,
        "receipt": verified_receipt,
        "status": "quality_first_identity_verified_exact",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build and optionally smoke native TensorRT engines for a Splitpoint benchmark set."
    )
    ap.add_argument("--benchmark-set", required=True, help="Benchmark set root, e.g. .../yolov7_paper_benchmark_20260625_160757")
    ap.add_argument("--case", action="append", default=[], help="Case id such as b066. Can be repeated. Required for part1/part2.")
    ap.add_argument("--variants", default="full,part2", help="Comma-separated: full,part1,part2")
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8", "uint8_cast_fp16", "uint8_dequant_fp16", "float32_layout_fp16"], help="TensorRT build precision/tag. uint8_cast_fp16 creates a UINT8-input Cast-to-FLOAT bridge ONNX for part2; uint8_dequant_fp16 adds (x-zero_point)*scale before part2; float32_layout_fp16 expects HailoRT FLOAT32 boundary and only applies layout reshape/transpose.")
    ap.add_argument("--out-dir", default="", help="Output directory. Default: <benchmark-set>/native_trt")
    ap.add_argument("--dequant-scale", type=float, default=0.0, help="Explicit scale for uint8_dequant_fp16; 0 means infer from calibration stats or use 1/255 fallback.")
    ap.add_argument("--dequant-zero-point", type=float, default=0.0, help="Explicit zero point for uint8_dequant_fp16.")
    ap.add_argument("--boundary-layout", default="as_input", choices=["as_input", "memory_nwc_to_ncw", "memory_nhwc_to_nchw", "memory_hwcn_to_nchw", "memory_nwhc_to_nchw", "memory_ncwh_to_nchw", "memory_chwn_to_nchw"], help="Raw Hailo boundary memory layout before the Part2 ONNX input. Use memory_nhwc_to_nchw for Hailo VStream HWC/NHWC bytes feeding canonical NCHW Part2 input.")
    ap.add_argument("--trtexec", default="", help="Path to trtexec. Default: auto-detect.")
    ap.add_argument("--explicit-full-source-onnx", default="", help="Quality-sealed suite source ONNX for explicit Full reuse.")
    ap.add_argument("--explicit-full-build-onnx", default="", help="Quality-sealed ONNX used by the Full engine build receipt.")
    ap.add_argument("--explicit-full-engine", default="", help="Quality-sealed Full engine; disables all engine discovery.")
    ap.add_argument("--explicit-full-build-receipt", default="", help="Quality-sealed Full engine build receipt.")
    ap.add_argument("--expected-source-onnx-sha256", default="")
    ap.add_argument("--expected-build-onnx-sha256", default="")
    ap.add_argument("--expected-engine-sha256", default="")
    ap.add_argument("--expected-trtexec-sha256", default="")
    ap.add_argument("--expected-engine-build-receipt-sha256", default="", help="Canonical SHA-256 of the complete receipt including its inner receipt_sha256.")
    ap.add_argument("--expected-engine-build-receipt-file-sha256", default="", help="SHA-256 of the receipt file bytes.")
    ap.add_argument("--expected-trt-engine-build-receipt-sha256", default="", help="Inner receipt.receipt_sha256 value.")
    ap.add_argument("--expected-source-onnx-size-bytes", type=int, default=0)
    ap.add_argument("--expected-build-onnx-size-bytes", type=int, default=0)
    ap.add_argument("--expected-engine-size-bytes", type=int, default=0)
    ap.add_argument("--expected-trtexec-size-bytes", type=int, default=0)
    ap.add_argument("--expected-engine-build-receipt-size-bytes", type=int, default=0, help="Canonical JSON byte length of the complete receipt.")
    ap.add_argument("--quality-first-producer-identity-sha256", default="", help="Signed central Quality-FIRST producer identity consumed by this exact run.")
    ap.add_argument("--workspace-mb", type=int, default=4096, help="TensorRT workspace in MiB.")
    ap.add_argument("--workspace-mode", default="auto", choices=["auto", "workspace", "mempool", "none"], help="TensorRT workspace flag style. auto chooses --memPoolSize for TRT10 and --workspace for older trtexec.")
    ap.add_argument("--no-shapes", action="store_true", help="Do not pass --shapes to trtexec. Useful for static-shape ONNX parse troubleshooting.")
    ap.add_argument("--shape-policy", default="auto", choices=["auto", "always", "never"], help="When to pass --shapes. auto omits shapes for static ONNX inputs and passes shapes only for dynamic inputs or explicit --shape-override.")
    ap.add_argument("--retry-without-shapes", action=argparse.BooleanOptionalAction, default=True, help="If the first build fails and shapes were passed, retry without --shapes.")
    ap.add_argument("--retry-workspace-alt", action=argparse.BooleanOptionalAction, default=True, help="If workspace flag style may be wrong, retry with the other TensorRT workspace flag.")
    ap.add_argument("--verbose-build", action="store_true", help="Pass --verbose to trtexec build.")
    ap.add_argument("--shape-override", default="", help="Override --shapes value for all builds/runs.")
    ap.add_argument(
        "--build", action=argparse.BooleanOptionalAction, default=True,
        help="Allow building on a verified cache miss; valid cached engines are reused.",
    )
    ap.add_argument(
        "--force-rebuild", action="store_true",
        help=(
            "Ignore an otherwise valid engine receipt and rebuild. By default "
            "--build means build-on-cache-miss, never rebuild-on-every-run."
        ),
    )
    ap.add_argument(
        "--artifact-policy",
        default=os.environ.get("ONNX_SPLITPOINT_ARTIFACT_POLICY", "normal"),
        choices=["normal", "cache_verify_only"],
        help="cache_verify_only permits receipt-verified engine reuse and loadEngine only.",
    )
    ap.add_argument("--run-smoke", action=argparse.BooleanOptionalAction, default=True, help="Run trtexec smoke after build.")
    ap.add_argument("--iterations", type=int, default=100, help="trtexec iterations for smoke runs.")
    ap.add_argument("--warmup-ms", type=int, default=200, help="trtexec warmup ms for smoke runs.")
    ap.add_argument(
        "--warmup-iterations", type=int, default=0,
        help=(
            "Requested warmup budget. trtexec exposes an internal millisecond "
            "warmup only; it is executed in the same process/context as the "
            "measured duration=0 iteration run and is never reported as an "
            "exact warmup-query count."
        ),
    )
    ap.add_argument("--duration-s", type=int, default=0, help="trtexec duration seconds for smoke runs; 0 uses iterations.")
    ap.add_argument("--extra-build-arg", action="append", default=[], help="Extra argument passed to trtexec during build. Repeatable.")
    ap.add_argument("--extra-run-arg", action="append", default=[], help="Extra argument passed to trtexec during run. Repeatable.")
    ap.add_argument("--dry-run", action="store_true", help="Write commands only; do not execute trtexec.")
    ap.add_argument("--json-out", default="", help="Optional aggregate JSON report path.")
    args = ap.parse_args()
    if args.force_rebuild:
        ap.error("productive_force_build_disabled: --force-rebuild is disabled; missing or incompatible engines remain buildable")

    artifact_policy = str(args.artifact_policy or "normal").strip().lower().replace("-", "_")
    if artifact_policy == "cache_verify_only":
        if bool(args.build):
            raise SystemExit(
                "cache_miss_blocked[tensorrt_build]: cache_verify_only requires --no-build"
            )
        if list(args.extra_build_arg or []) or bool(args.verbose_build):
            raise SystemExit(
                "cache_miss_blocked[tensorrt_build]: build arguments conflict with cache_verify_only"
            )
        if bool(args.force_rebuild):
            raise SystemExit(
                "cache_miss_blocked[tensorrt_build]: --force-rebuild conflicts with cache_verify_only"
            )
        os.environ["ONNX_SPLITPOINT_ARTIFACT_POLICY"] = "cache_verify_only"

    root = Path(args.benchmark_set).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    variants = [v.strip().lower() for v in str(args.variants).split(",") if v.strip()]
    cases = [_norm_case(c) for c in args.case]
    if any(v in {"part1", "part2"} for v in variants) and not cases:
        raise SystemExit("--case is required when variants include part1/part2")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    trtexec = _find_trtexec(args.trtexec) if not args.dry_run else (args.trtexec or "trtexec")
    explicit_full_identity = _explicit_full_quality_identity(args, trtexec)
    artifacts = (
        [{
            "case": None, "variant": "full",
            "onnx": Path(explicit_full_identity["paths"]["build_onnx"]),
        }]
        if explicit_full_identity is not None
        else _artifact_specs(root, cases, variants)
    )

    report: dict[str, Any] = {
        "benchmark_set": str(root),
        "trtexec": str(trtexec),
        "precision": str(args.precision),
        "artifact_policy": artifact_policy,
        "variants": variants,
        "cases": cases,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifacts": [],
    }

    for art in artifacts:
        onnx_path = Path(art["onnx"])
        bridge_meta = None
        precision_tag = str(args.precision)
        # v59an: uint8_cast_fp16 is meaningful only for split part2 boundary bridges.
        # Full models have normal image inputs; build/reuse fp16 full engines instead.
        effective_precision_tag = "fp16" if precision_tag in ("uint8_cast_fp16", "uint8_dequant_fp16", "float32_layout_fp16") and str(art.get("variant")) == "full" else precision_tag
        # Diagnostic/paper-style bridge: engine input becomes UINT8 and the first
        # TensorRT layer casts to FLOAT.  This avoids Python CPU uint8->float32 casts
        # while keeping the existing float part2 graph semantics.
        if effective_precision_tag == "uint8_cast_fp16" and str(art.get("variant")) == "part2":
            tmp_engine_path, tmp_work_dir = _target_path(root, art.get("case"), art["variant"], effective_precision_tag, out_dir)
            onnx_path, bridge_meta = _make_uint8_cast_bridge_onnx(onnx_path, tmp_work_dir)
        elif effective_precision_tag == "uint8_dequant_fp16" and str(art.get("variant")) == "part2":
            tmp_engine_path, tmp_work_dir = _target_path(root, art.get("case"), art["variant"], effective_precision_tag, out_dir)
            onnx_path, bridge_meta = _make_uint8_dequant_bridge_onnx(onnx_path, tmp_work_dir, root=root, case=art.get("case"), scale=args.dequant_scale, zero_point=args.dequant_zero_point, boundary_layout=args.boundary_layout)
        elif effective_precision_tag == "float32_layout_fp16" and str(art.get("variant")) == "part2":
            tmp_engine_path, tmp_work_dir = _target_path(root, art.get("case"), art["variant"], effective_precision_tag, out_dir)
            onnx_path, bridge_meta = _make_float32_layout_bridge_onnx(onnx_path, tmp_work_dir, boundary_layout=args.boundary_layout)
        inputs = _onnx_inputs(onnx_path)
        outputs = _onnx_outputs(onnx_path)
        if bool(args.no_shapes) or str(args.shape_policy).lower() == "never":
            shapes = ""
            shapes_policy_effective = "never" if str(args.shape_policy).lower() == "never" else "no_shapes_flag"
        elif str(args.shape_policy).lower() == "auto" and _all_inputs_static(inputs) and not args.shape_override:
            # TensorRT 10 rejects --shapes for static-shape ONNX models:
            # "Static model does not take explicit shapes".  Omit shapes by default.
            shapes = ""
            shapes_policy_effective = "auto_static_omit"
        else:
            shapes = _shape_arg(inputs, args.shape_override)
            shapes_policy_effective = "always_or_dynamic"
        if explicit_full_identity is not None:
            engine_path = Path(explicit_full_identity["paths"]["engine"])
            work_dir = engine_path.parent
        else:
            engine_path, work_dir = _target_path(root, art.get("case"), art["variant"], effective_precision_tag, out_dir)
        timing_cache = work_dir / "timing.cache"
        build_receipt_path = (
            Path(explicit_full_identity["paths"]["engine_build_receipt"])
            if explicit_full_identity is not None else _build_receipt_path(work_dir)
        )
        entry: dict[str, Any] = {
            "schema": "onnx-splitpoint/native-trt-meta",
            "schema_version": 1,
            "case": art.get("case"),
            "variant": art["variant"],
            "onnx": str(onnx_path),
            "engine": str(engine_path),
            "inputs": [s.__dict__ for s in inputs],
            "outputs": [s.__dict__ for s in outputs],
            "shapes_arg": shapes,
            "shapes_policy_effective": shapes_policy_effective,
            "inputs_static": _all_inputs_static(inputs),
            "precision": str(effective_precision_tag),
            "requested_precision": str(args.precision),
            "uint8_cast_bridge": bridge_meta,
            "quality_first_identity": dict(explicit_full_identity or {}),
            "source_onnx": (
                str(explicit_full_identity["paths"]["source_onnx"])
                if explicit_full_identity is not None else str(onnx_path)
            ),
        }
        extra_build_args = list(args.extra_build_arg)
        if bool(args.verbose_build) and "--verbose" not in extra_build_args:
            extra_build_args.append("--verbose")
        model_id = _benchmark_model_id(root, onnx_path)
        verified_cache, cache_reason = _verify_engine_cache_candidate(
            source_onnx=onnx_path,
            engine=engine_path,
            receipt_path=build_receipt_path,
            trtexec=str(trtexec),
            precision=str(args.precision),
            shapes=shapes,
            workspace_mb=int(args.workspace_mb),
            workspace_mode=str(args.workspace_mode),
            retry_without_shapes=bool(args.retry_without_shapes),
            retry_workspace_alt=bool(args.retry_workspace_alt),
            extra_build_args=extra_build_args,
            force_rebuild=bool(args.force_rebuild),
            enforce_build_contract=explicit_full_identity is None,
        )
        cache_hit = verified_cache is not None
        cache_identity = (
            str(verified_cache.get("receipt_sha256") or "")
            if verified_cache is not None else ""
        )
        entry["trt_cache"] = {
            "status": "HIT" if cache_hit else "MISS",
            "reason": "compatible_receipt" if cache_hit else cache_reason,
            "role": str(art["variant"]),
            "model_id": model_id,
            "case_id": str(art.get("case") or ""),
            "artifact_path": str(engine_path),
            "receipt_path": str(build_receipt_path),
            "identity": cache_identity,
        }
        _log_trt_cache(
            "HIT" if cache_hit else "MISS",
            role=str(art["variant"]),
            model=model_id,
            case=art.get("case"),
            reason="compatible_receipt" if cache_hit else cache_reason,
            engine=engine_path,
            identity=cache_identity,
        )

        if cache_hit:
            assert verified_cache is not None
            entry["build_ok"] = True
            entry["engine_build_receipt"] = verified_cache
            entry["engine_build_receipt_path"] = str(build_receipt_path)
            entry["engine_build_receipt_status"] = "engine_build_receipt_verified"
            entry["build"] = {
                "cmd": list(verified_cache["command"]),
                "returncode": int(verified_cache["build_returncode"]),
                "dry_run": False,
                "evidence_source": "verified_engine_build_receipt",
                "compiler_dispatched": False,
                "cache_hit": True,
            }
            entry["build_attempts"] = []
            entry["trt_cache"]["outcome"] = "reused"
            entry["trt_cache"]["compiler_dispatched"] = False
        elif args.build and _strict_warm_trt_build_forbidden(str(art["variant"]), case_id=str(art.get("case") or ""), generic_part2=str(args.precision) in {"fp16", "fp32", "int8"}):
            # Preserve the rejected engine/receipt for diagnosis; the guard
            # runs before invalidation, retries or compiler preparation.
            entry["build_ok"] = False
            entry["build_attempts"] = []
            entry["build"] = {"returncode": 78, "status": "cache_miss_blocked", "compiler_dispatched": False,
                              "error": "cache_miss_blocked:artifact_policy=strict_warm_cache:compiler=trtexec"}
            entry["trt_cache"].update({"outcome": "cache_miss_blocked", "compiler_dispatched": False})
        elif args.build:
            if build_receipt_path.exists() and not args.dry_run:
                try:
                    build_receipt_path.unlink()
                except OSError:
                    pass
            build_attempts: list[dict[str, Any]] = []
            attempt_specs = [(shapes, str(args.workspace_mode), "initial")]
            if bool(args.retry_workspace_alt):
                wm = str(args.workspace_mode).lower()
                if wm in {"auto", "mempool"}:
                    attempt_specs.append((shapes, "workspace", "retry_workspace_flag"))
                elif wm == "workspace":
                    attempt_specs.append((shapes, "mempool", "retry_mempool_flag"))
            if bool(args.retry_without_shapes) and shapes:
                attempt_specs.append(("", str(args.workspace_mode), "retry_without_shapes"))
            seen_attempts: set[tuple[str, str]] = set()
            build_ok = False
            build_meta: dict[str, Any] | None = None
            for attempt_idx, (attempt_shapes, attempt_workspace_mode, attempt_reason) in enumerate(attempt_specs, start=1):
                key = (attempt_shapes, attempt_workspace_mode)
                if key in seen_attempts:
                    continue
                seen_attempts.add(key)
                if engine_path.exists() and not args.dry_run:
                    try:
                        engine_path.unlink()
                    except Exception:
                        pass
                cmd = _engine_build_cmd(
                    trtexec,
                    onnx_path,
                    engine_path,
                    str(args.precision),
                    attempt_shapes,
                    timing_cache,
                    int(args.workspace_mb),
                    attempt_workspace_mode,
                    extra_build_args,
                )
                log_path = work_dir / ("build_trtexec.log" if attempt_idx == 1 else f"build_trtexec_attempt{attempt_idx}.log")
                build_meta = _run(cmd, cwd=work_dir, log_path=log_path, dry_run=bool(args.dry_run), artifact_role=str(art["variant"]), case_id=str(art.get("case") or ""), generic_part2=str(args.precision) in {"fp16", "fp32", "int8"})
                tail = _log_tail(log_path)
                build_meta["attempt"] = attempt_idx
                build_meta["reason"] = attempt_reason
                build_meta["shapes_arg"] = attempt_shapes
                build_meta["workspace_mode"] = attempt_workspace_mode
                build_meta["compiler_dispatched"] = not bool(args.dry_run) and build_meta.get("status") != "cache_miss_blocked"
                build_meta["log_tail"] = tail[-6000:]
                build_meta["failure_hint"] = _failure_hint(tail)
                build_attempts.append(build_meta)
                build_ok = bool(build_meta.get("returncode", 1) == 0 and (engine_path.exists() or args.dry_run))
                if build_ok:
                    entry["build_success_shapes_arg"] = attempt_shapes
                    entry["build_success_workspace_mode"] = attempt_workspace_mode
                    receipt = _write_engine_build_receipt(
                        work_dir=work_dir, command=cmd,
                        source_onnx=onnx_path, engine=engine_path,
                        trtexec=str(trtexec),
                        returncode=int(build_meta.get("returncode") or 0),
                        dry_run=bool(args.dry_run),
                    )
                    if receipt is not None:
                        entry["engine_build_receipt"] = receipt
                        entry["engine_build_receipt_path"] = str(build_receipt_path)
                        entry["engine_build_receipt_status"] = "engine_build_receipt_verified"
                    break
            if build_ok and not args.dry_run:
                verified_built, built_receipt_status = _verify_engine_cache_candidate(
                    source_onnx=onnx_path,
                    engine=engine_path,
                    receipt_path=build_receipt_path,
                    trtexec=str(trtexec),
                    precision=str(args.precision),
                    shapes=shapes,
                    workspace_mb=int(args.workspace_mb),
                    workspace_mode=str(args.workspace_mode),
                    retry_without_shapes=bool(args.retry_without_shapes),
                    retry_workspace_alt=bool(args.retry_workspace_alt),
                    extra_build_args=extra_build_args,
                )
                if verified_built is None:
                    build_ok = False
                    entry["engine_build_receipt_status"] = built_receipt_status
                else:
                    entry["engine_build_receipt"] = verified_built
                    entry["engine_build_receipt_path"] = str(build_receipt_path)
                    entry["engine_build_receipt_status"] = (
                        "engine_build_receipt_verified"
                    )
                    entry["trt_cache"]["identity"] = str(
                        verified_built.get("receipt_sha256") or ""
                    )
            entry["build"] = build_meta or {"returncode": 999, "reason": "no_attempt"}
            entry["build_attempts"] = build_attempts
            entry["build_ok"] = build_ok
            entry["trt_cache"]["outcome"] = "built" if build_ok else "build_failed"
            entry["trt_cache"]["compiler_dispatched"] = any(
                bool(attempt.get("compiler_dispatched"))
                for attempt in build_attempts
            )
            if not build_ok:
                print("[native-trt][build-failed]", art.get("case"), art["variant"], (build_attempts[-1].get("failure_hint") if build_attempts else "no attempt"), file=sys.stderr)
                if build_attempts:
                    print(build_attempts[-1].get("log_tail", "")[-2000:], file=sys.stderr)
        else:
            # ``--no-build`` is a strict cache-only request.  A bare engine is
            # not enough: it must have a receipt bound to the exact source and
            # current builder ABI.
            entry["build_ok"] = False
            entry["engine_build_receipt_status"] = cache_reason
            entry["build"] = {
                "skipped": True,
                "reason": f"cache_miss:{cache_reason}",
                "compiler_dispatched": False,
                "cache_hit": False,
            }
            entry["build_attempts"] = []
        if args.run_smoke:
            if not bool(entry.get("build_ok")) and not args.dry_run:
                entry["run_smoke"] = {
                    "skipped": True,
                    "reason": "engine_cache_or_build_not_valid",
                }
                entry["run_ok"] = False
            else:
                run_shapes = str(entry.get("build_success_shapes_arg", shapes))
                warmup_iterations = max(0, int(args.warmup_iterations))
                # trtexec has no exact warmup-query CLI.  Use its internal
                # warmup window so deserialization, warmup and measurement all
                # share one engine and execution context.  A numeric
                # ``--warmup-iterations`` value is treated only as a minimum
                # millisecond budget when no explicit --warmup-ms was given;
                # never claim it as an observed query count.
                same_process_warmup = _same_process_warmup_contract(
                    warmup_iterations, int(args.warmup_ms),
                )
                effective_warmup_ms = int(same_process_warmup["warmup_ms_effective"])
                for stale_warmup in (
                    work_dir / "run_trtexec_warmup.log",
                    work_dir / "run_trtexec_warmup_times.json",
                ):
                    if stale_warmup.exists() and not args.dry_run:
                        try:
                            stale_warmup.unlink()
                        except OSError:
                            pass
                entry.update(same_process_warmup)
                export_times_path = work_dir / "run_trtexec_times.json"
                if export_times_path.exists() and not args.dry_run:
                    try:
                        export_times_path.unlink()
                    except OSError:
                        pass
                measured_extra = [
                    str(arg) for arg in args.extra_run_arg
                    if not str(arg).startswith("--exportTimes=")
                ]
                measured_extra.append(f"--exportTimes={export_times_path}")
                cmd = _engine_run_cmd(
                    trtexec,
                    engine_path,
                    run_shapes,
                    int(args.iterations),
                    effective_warmup_ms,
                    int(args.duration_s),
                    measured_extra,
                )
                entry["run_shapes_arg"] = run_shapes
                run_meta = _run(
                    cmd, cwd=work_dir,
                    log_path=work_dir / "run_trtexec.log",
                    dry_run=bool(args.dry_run),
                )
                entry["run_smoke"] = run_meta
                timing_evidence = _trtexec_export_times_evidence(export_times_path)
                entry["trtexec_export_times"] = timing_evidence
                exact_iteration_mode = bool(int(args.duration_s) <= 0 and int(args.iterations) > 0)
                actual_work_units = timing_evidence.get("completed_work_units")
                exact_count_ok = bool(
                    exact_iteration_mode
                    and timing_evidence.get("status") == "ok"
                    and actual_work_units == int(args.iterations)
                )
                entry["completed_work_units"] = actual_work_units
                entry["completed_work_units_source"] = (
                    "trtexec_export_times" if timing_evidence.get("status") == "ok" else ""
                )
                entry["completed_work_units_status"] = (
                    "exact_runtime_counter" if exact_count_ok
                    else "dry_run_unverified" if args.dry_run
                    else "count_mismatch" if timing_evidence.get("status") == "ok"
                    else "runtime_evidence_missing"
                )
                for key in ("latency_mean_ms", "latency_p50_ms", "latency_p95_ms"):
                    if timing_evidence.get(key) is not None:
                        entry[key] = timing_evidence[key]
                entry["run_ok"] = bool(
                    run_meta.get("returncode", 1) == 0
                    and (exact_count_ok or bool(args.dry_run) or not exact_iteration_mode)
                )
                if run_meta.get("returncode", 1) == 0 and exact_iteration_mode and not exact_count_ok:
                    entry["run_failure_reason"] = "trtexec_completed_work_unit_verification_failed"
        (work_dir / "native_trt_meta.json").write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"case": entry.get("case"), "variant": entry["variant"], "engine": entry["engine"], "build_ok": entry.get("build_ok"), "run_ok": entry.get("run_ok")}, sort_keys=True))
        report["artifacts"].append(entry)

    report["ok"] = all(bool(a.get("build_ok", True)) and bool(a.get("run_ok", True)) for a in report["artifacts"])
    json_out = Path(args.json_out).expanduser().resolve() if args.json_out else (out_dir or (root / "native_trt")) / "native_trt_summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[native-trt] wrote {json_out}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
