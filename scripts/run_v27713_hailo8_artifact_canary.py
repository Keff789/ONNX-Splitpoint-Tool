#!/usr/bin/env python3
"""Run a small, optional Hailo-8 structural canary on existing Full artifacts.

The canary never compiles a model and never mutates an input BenchmarkSet.  It
revalidates the existing HEF/build-receipt/output-contract chain, selects one
to sixteen real validation images, and delegates every inference to the
existing ``smoke_hailo10_full_from_benchmarkset.py`` runtime runner.  PASS
attests artifact identity, runtime input/preprocess binding, finite dumped
outputs and structural postprocess completion.  It does not attest numerical
correctness, accuracy, or parity with another backend.

Local usage::

    python -B scripts/run_v27713_hailo8_artifact_canary.py \
      --benchmark-set /path/to/benchmark_set/legacy_suite \
      --out /path/to/new/canary-output

Remote Hailo-8 usage::

    python -B scripts/run_v27713_hailo8_artifact_canary.py \
      --benchmark-set /path/to/model-a/benchmark_set/legacy_suite \
      --benchmark-set /path/to/model-b/benchmark_set/legacy_suite \
      --hardware-matrix /path/to/hardware_matrix.json \
      --setup-id my-hailo8-setup --max-images 16 \
      --out /path/to/new/canary-output

``SKIP`` exits zero unless ``--require-hardware`` is set.  An integrity error
or any failure after local/remote runtime admission is always ``FAIL``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.preprocessing_contract import (  # noqa: E402
    preprocessing_contract_sha256,
)
from onnx_splitpoint_tool.native_output_endpoint import (  # noqa: E402
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.remote.ssh_transport import (  # noqa: E402
    HostConfig,
    SSHTransport,
)
from onnx_splitpoint_tool.remote.process_lease import (  # noqa: E402
    RemoteProcessLeaseRegistry,
    resolve_remote_process_lease_scope,
)


SCHEMA = "onnx-splitpoint/v27713-hailo8-artifact-canary-result"
SCHEMA_VERSION = 1
RECEIPT_SCHEMA = "onnx-splitpoint/hailo-hef-build-receipt/v2"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
EXCLUDED_TREE_NAMES = {
    ".git", ".pytest_cache", ".venv", "__pycache__", "build", "dist",
}
HEX64 = re.compile(r"[0-9a-f]{64}")
ARTIFACT_POLICY = "cache_verify_only"
ARTIFACT_POLICY_ENV = "ONNX_SPLITPOINT_ARTIFACT_POLICY"
PAYLOAD_SCHEMA = "onnx-splitpoint/v27713-hailo8-structural-canary-payload"
PAYLOAD_SCHEMA_VERSION = 1
PAYLOAD_MANIFEST_NAME = "canary_payload_manifest.json"
PAYLOAD_VERIFIER = ROOT / "scripts/verify_v27713_hailo8_canary_payload.py"
RUNTIME_ENTRY_POINT = "scripts/smoke_hailo10_full_from_benchmarkset.py"
REMOTE_KILL_AFTER_S = 15
REMOTE_TRANSPORT_GRACE_S = 30
MAX_RESULT_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_RESULT_MEMBER_BYTES = 512 * 1024 * 1024
MAX_RESULT_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_RESULT_MEMBER_COUNT = 20_000


class CanaryError(RuntimeError):
    """A fail-closed input, integrity or execution error."""


class CanarySkip(RuntimeError):
    """Optional material or hardware is unavailable before admission."""


@dataclass(frozen=True)
class ImagePlan:
    source: Path
    sha256: str


@dataclass(frozen=True)
class BenchmarkPlan:
    benchmark_set: Path
    model_id: str
    task: str
    hef: Path
    hef_sha256: str
    receipt: Path
    output_contracts: Path
    benchmark_set_json: Path
    onnx: Path | None
    onnx_sha256: str
    receipt_source_onnx_sha256: str
    receipt_compiler_onnx_sha256: str
    preprocessing_contract: Mapping[str, Any]
    preprocessing_contract_sha256: str
    output_contract_resolution: Mapping[str, Any]
    images: tuple[ImagePlan, ...]


@dataclass(frozen=True)
class PayloadBundle:
    path: Path
    sha256: str
    size_bytes: int
    manifest: Mapping[str, Any]
    manifest_sha256: str
    verifier: Path
    verifier_sha256: str
    verifier_size_bytes: int


@dataclass(frozen=True)
class RemoteSetup:
    setup_id: str
    host: str
    user: str
    port: int
    remote_base_dir: str
    remote_venv: str
    ssh_extra_args: str


@dataclass(frozen=True)
class CanaryConfig:
    benchmark_sets: tuple[Path, ...]
    out_dir: Path
    hardware_matrix: Path | None = None
    setup_id: str = ""
    max_images: int = 1
    require_hardware: bool = False
    timeout_s: int = 300


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    return token if HEX64.fullmatch(token) else ""


def _json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, indent=2, sort_keys=True, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _lexical_absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _load_json(path: Path) -> dict[str, Any]:
    _require_regular_file(path, label="json")
    duplicate = False

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        )
    except Exception as exc:
        raise CanaryError(f"invalid_json:{path}:{type(exc).__name__}") from exc
    if duplicate or not isinstance(value, dict):
        raise CanaryError(f"invalid_or_duplicate_json:{path}")
    return value


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise CanaryError(f"missing_{label}:{path}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise CanaryError(f"unsafe_{label}:{path}")


def _require_safe_tree_file(root: Path, path: Path, *, label: str) -> Path:
    root = _lexical_absolute(root)
    path = _lexical_absolute(path)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CanaryError(f"{label}_escapes_benchmark_set:{path}") from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as exc:
            raise CanaryError(f"missing_{label}:{cursor}") from exc
        if stat.S_ISLNK(mode):
            raise CanaryError(f"unsafe_symlink_{label}:{cursor}")
    _require_regular_file(path, label=label)
    return path


def _safe_relative(root: Path, value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    relative = Path(text)
    if (
        not text or relative.is_absolute() or ".." in relative.parts
        or relative == Path(".")
    ):
        raise CanaryError(f"unsafe_{label}_relative_path:{text}")
    return _require_safe_tree_file(root, root / relative, label=label)


def _prepare_output(path: Path, *, inputs: Sequence[Path]) -> Path:
    output = _lexical_absolute(path)
    for source in inputs:
        source_abs = _lexical_absolute(source)
        if output == source_abs or source_abs in output.parents:
            raise CanaryError("output_must_not_be_inside_benchmark_set")
    cursor = Path(output.anchor)
    missing: list[Path] = []
    for part in output.parts[1:]:
        cursor /= part
        if os.path.lexists(cursor):
            info = cursor.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise CanaryError(f"unsafe_output_parent:{cursor}")
            if missing:
                raise CanaryError(f"output_parent_identity_invalid:{cursor}")
        else:
            missing.append(cursor)
    for directory in missing:
        directory.mkdir(mode=0o755)
    if output.is_symlink() or not output.is_dir():
        raise CanaryError(f"unsafe_output:{output}")
    check = Path(output.anchor)
    for part in output.parts[1:]:
        check /= part
        info = check.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise CanaryError(f"unsafe_output_component:{check}")
    if any(output.iterdir()):
        raise CanaryError(f"output_must_be_empty:{output}")
    return output


def _require_safe_output_file(root: Path, path: Path, *, label: str) -> Path:
    root = _lexical_absolute(root)
    path = _lexical_absolute(path)
    root_info = root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise CanaryError(f"unsafe_{label}_root:{root}")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise CanaryError(f"{label}_escapes_output:{path}") from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        try:
            info = cursor.lstat()
        except FileNotFoundError as exc:
            raise CanaryError(f"missing_{label}:{cursor}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise CanaryError(f"unsafe_symlink_{label}:{cursor}")
    _require_regular_file(path, label=label)
    return path


def _task(payload: Mapping[str, Any], model_id: str) -> str:
    for key in ("benchmark_task", "model_task", "task"):
        value = str(payload.get(key) or "").strip().lower()
        if value in {"classification", "detection"}:
            return value
    low = model_id.lower()
    return "detection" if "yolo" in low or "detect" in low else "classification"


def _model_id(payload: Mapping[str, Any], benchmark_set: Path) -> str:
    value = str(payload.get("model_id") or payload.get("model_name") or "").strip()
    if value:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            raise CanaryError(f"unsafe_model_id:{value}")
        return value
    parent = benchmark_set.parent
    inferred = parent.parent.name if parent.name == "benchmark_set" else benchmark_set.name
    if not re.fullmatch(r"[A-Za-z0-9._-]+", inferred):
        raise CanaryError(f"unsafe_model_id:{inferred}")
    return inferred


def _find_hef(benchmark_set: Path) -> Path:
    exact = [
        benchmark_set / "hailo/hailo8/full/compiled.hef",
        benchmark_set / "hailo/hailo8/full/model.hef",
    ]
    matches = [path for path in exact if os.path.lexists(path)]
    if not matches:
        matches = sorted(
            path for path in benchmark_set.glob("**/hailo/hailo8/full/*.hef")
            if ".hailo-generations" not in path.parts
        )
    unique = list(dict.fromkeys(_lexical_absolute(path) for path in matches))
    if not unique:
        raise CanarySkip(f"hailo8_full_hef_unavailable:{benchmark_set}")
    if len(unique) != 1:
        raise CanaryError(f"hailo8_full_hef_ambiguous:{benchmark_set}")
    return _require_safe_tree_file(
        benchmark_set, unique[0], label="hailo8_full_hef",
    )


def _find_full_onnx(benchmark_set: Path) -> Path | None:
    candidates: list[Path] = []
    models = benchmark_set / "models"
    if models.is_dir() and not models.is_symlink():
        candidates.extend(sorted(models.glob("*.onnx")))
    candidates.extend(sorted(benchmark_set.glob("**/models/*.onnx")))
    unique: list[Path] = []
    for raw in candidates:
        name = raw.name.lower()
        if any(token in name for token in ("part1", "part2", "split")):
            continue
        path = _lexical_absolute(raw)
        if path not in unique:
            unique.append(path)
    if not unique:
        return None
    # This mirrors the existing runner's deterministic first-Full-ONNX
    # selection.  Only that selected source is copied into the isolated
    # payload, so local and remote discovery cannot diverge.
    return _require_safe_tree_file(
        benchmark_set, unique[0], label="full_onnx",
    )


def _portable_output_contract_resolution(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(value)
    declaration = str(result.pop("declaration_source", "") or "")
    if Path(declaration).name != "output_contracts.json":
        raise CanaryError("output_contract_declaration_source_invalid")
    result["declaration_source_name"] = "output_contracts.json"
    return result


def _find_images(benchmark_set: Path, *, maximum: int) -> tuple[ImagePlan, ...]:
    roots = [
        benchmark_set / "resources/validation",
        benchmark_set / "validation",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for current, directories, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            unsafe_dirs = [
                name for name in directories
                if (current_path / name).is_symlink()
            ]
            if unsafe_dirs:
                raise CanaryError(
                    f"validation_image_tree_contains_symlink:{current_path}"
                )
            for filename in filenames:
                path = current_path / filename
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    candidates.append(path)
    unique = sorted(dict.fromkeys(_lexical_absolute(path) for path in candidates))
    if not unique:
        raise CanarySkip(f"validation_images_unavailable:{benchmark_set}")
    selected = unique[:maximum]
    if len({path.name for path in selected}) != len(selected):
        raise CanaryError("selected_validation_image_names_not_unique")
    plans: list[ImagePlan] = []
    for path in selected:
        safe = _require_safe_tree_file(
            benchmark_set, path, label="validation_image",
        )
        plans.append(ImagePlan(source=safe, sha256=_sha256(safe)))
    return tuple(plans)


def _attest_benchmark_set(path: Path, *, max_images: int) -> BenchmarkPlan:
    benchmark_set = _lexical_absolute(path)
    try:
        mode = benchmark_set.lstat().st_mode
    except FileNotFoundError as exc:
        raise CanaryError(f"benchmark_set_missing:{benchmark_set}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CanaryError(f"unsafe_benchmark_set:{benchmark_set}")
    if benchmark_set.resolve(strict=True) != benchmark_set:
        raise CanaryError(f"noncanonical_benchmark_set:{benchmark_set}")

    benchmark_json = _require_safe_tree_file(
        benchmark_set, benchmark_set / "benchmark_set.json",
        label="benchmark_set_json",
    )
    payload = _load_json(benchmark_json)
    model_id = _model_id(payload, benchmark_set)
    task = _task(payload, model_id)
    hef = _find_hef(benchmark_set)
    receipt = _require_safe_tree_file(
        benchmark_set, hef.parent / "hailo_hef_build_receipt.json",
        label="hailo_build_receipt",
    )
    output_contracts = _require_safe_tree_file(
        benchmark_set, benchmark_set / "output_contracts.json",
        label="output_contracts",
    )
    receipt_payload = _load_json(receipt)
    output_payload = _load_json(output_contracts)
    hef_sha = _sha256(hef)
    hef_size = int(hef.stat().st_size)
    preprocessing = receipt_payload.get("preprocessing_contract")
    preprocessing_sha = _sha_token(
        receipt_payload.get("preprocessing_contract_sha256")
    )
    if (
        receipt_payload.get("schema") != RECEIPT_SCHEMA
        or str(receipt_payload.get("hw_arch") or "").strip().lower()
        != "hailo8"
        or _sha_token(receipt_payload.get("hef_sha256")) != hef_sha
        or int(receipt_payload.get("hef_size_bytes") or -1) != hef_size
        or not isinstance(preprocessing, Mapping)
        or str(preprocessing.get("task") or "").strip().lower() != task
        or preprocessing_sha != preprocessing_contract_sha256(preprocessing)
    ):
        raise CanaryError(f"hailo_build_receipt_attestation_failed:{model_id}")
    if (
        not isinstance(output_payload.get("contracts"), list)
        or not output_payload["contracts"]
    ):
        raise CanaryError(f"output_contract_container_invalid:{model_id}")
    matching_contracts = [
        row for row in output_payload["contracts"]
        if isinstance(row, Mapping)
        and str(row.get("model_id") or "").strip() == model_id
        and str(row.get("variant") or "").strip().lower() == "full"
        and str(row.get("task") or "").strip().lower() == task
        and str(row.get("backend") or "").strip().lower().startswith("hailo8")
    ]
    if len(matching_contracts) != 1:
        raise CanaryError(f"authoritative_output_contract_not_unique:{model_id}")
    declared_artifact_sha = _sha_token(
        matching_contracts[0].get("recorded_artifact_sha256")
        or matching_contracts[0].get("artifact_sha256")
    )
    if not declared_artifact_sha or declared_artifact_sha != hef_sha:
        raise CanaryError(f"output_contract_hef_sha256_mismatch:{model_id}")
    artifact_path_text = str(
        matching_contracts[0].get("recorded_artifact_path")
        or matching_contracts[0].get("artifact_path")
        or ""
    ).strip()
    artifact_relative = Path(artifact_path_text)
    if (
        not artifact_path_text
        or artifact_relative.is_absolute()
        or ".." in artifact_relative.parts
        or _lexical_absolute(benchmark_set / artifact_relative) != hef
    ):
        raise CanaryError(f"output_contract_hef_path_mismatch:{model_id}")
    try:
        declared_size = int(
            matching_contracts[0].get("recorded_artifact_size_bytes")
            or matching_contracts[0].get("artifact_size_bytes")
            or -1
        )
    except (TypeError, ValueError) as exc:
        raise CanaryError(
            f"output_contract_hef_size_invalid:{model_id}"
        ) from exc
    if declared_size != hef_size:
        raise CanaryError(f"output_contract_hef_size_mismatch:{model_id}")
    resolved_contract = load_authoritative_output_contract(
        benchmark_set,
        backend="hailo8",
        model_id=model_id,
        variant="full",
        task=task,
    )
    if (
        resolved_contract.get("contract_resolution_status") != "attested"
        or resolved_contract.get("authoritative_output_contract") is not True
    ):
        raise CanaryError(f"output_contract_resolution_failed:{model_id}")
    portable_resolution = _portable_output_contract_resolution(
        resolved_contract
    )
    onnx = _find_full_onnx(benchmark_set)
    onnx_sha = _sha256(onnx) if onnx is not None else ""
    receipt_source_onnx_sha = _sha_token(
        receipt_payload.get("source_onnx_sha256")
    )
    receipt_compiler_onnx_sha = _sha_token(
        receipt_payload.get("compiler_onnx_sha256")
    )
    receipt_claims_onnx = any(
        str(receipt_payload.get(key) or "").strip()
        for key in ("source_onnx_sha256", "compiler_onnx_sha256")
    )
    if onnx is None and receipt_claims_onnx:
        raise CanaryError(f"receipt_source_onnx_missing:{model_id}")
    if onnx is not None and (
        not receipt_source_onnx_sha
        or receipt_source_onnx_sha != onnx_sha
        or not receipt_compiler_onnx_sha
    ):
        raise CanaryError(f"receipt_source_onnx_binding_failed:{model_id}")
    images = _find_images(benchmark_set, maximum=max_images)
    return BenchmarkPlan(
        benchmark_set=benchmark_set,
        model_id=model_id,
        task=task,
        hef=hef,
        hef_sha256=hef_sha,
        receipt=receipt,
        output_contracts=output_contracts,
        benchmark_set_json=benchmark_json,
        onnx=onnx,
        onnx_sha256=onnx_sha,
        receipt_source_onnx_sha256=receipt_source_onnx_sha,
        receipt_compiler_onnx_sha256=receipt_compiler_onnx_sha,
        preprocessing_contract=dict(preprocessing),
        preprocessing_contract_sha256=preprocessing_sha,
        output_contract_resolution=portable_resolution,
        images=images,
    )


def _quote_remote_source_path(value: str) -> str:
    path = str(value or "")
    if not path or any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise CanaryError("remote_venv_path_invalid")
    if path.startswith("~/"):
        relative = path[2:]
        if not relative:
            raise CanaryError("remote_venv_path_invalid")
        return '"$HOME"/' + shlex.quote(relative)
    if path.startswith("~"):
        raise CanaryError("remote_venv_tilde_user_not_supported")
    return shlex.quote(path)


def _normalise_remote_venv(value: Any) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        raise CanaryError("remote_venv_must_be_string")
    raw = value.strip()
    if not raw:
        return ""
    if any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise CanaryError("remote_venv_contains_control_character")
    if not any(char.isspace() for char in raw):
        return f"source {_quote_remote_source_path(raw)}"

    # A whitespace-bearing value is the legacy shell-snippet form used by the
    # benchmark runner.  This canary accepts only activation and simple export
    # statements; arbitrary commands, substitutions, pipes and redirects are
    # rejected before any SSH operation.
    lexer = shlex.shlex(raw, posix=True, punctuation_chars=";&|<>()`$")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise CanaryError("remote_venv_snippet_parse_failed") from exc
    if not tokens:
        return ""
    commands: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token == ";":
            if not current:
                raise CanaryError("remote_venv_snippet_empty_command")
            commands.append(current)
            current = []
        elif token in {"&", "&&", "|", "||", "<", ">", "(", ")", "`", "$", "$("}:
            raise CanaryError("remote_venv_snippet_shell_operator_forbidden")
        else:
            current.append(token)
    if current:
        commands.append(current)
    if not commands:
        raise CanaryError("remote_venv_snippet_empty")

    normalised: list[str] = []
    for command in commands:
        head = command[0]
        if head in {"source", "."} and len(command) == 2:
            normalised.append(
                f"source {_quote_remote_source_path(command[1])}"
            )
            continue
        if head == "export" and len(command) >= 2:
            exports: list[str] = []
            for assignment in command[1:]:
                if "=" not in assignment:
                    raise CanaryError("remote_venv_export_assignment_required")
                name, assigned = assignment.split("=", 1)
                if (
                    not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                    or name == ARTIFACT_POLICY_ENV
                ):
                    raise CanaryError("remote_venv_export_name_forbidden")
                exports.append(f"{name}={shlex.quote(assigned)}")
            normalised.append("export " + " ".join(exports))
            continue
        raise CanaryError("remote_venv_snippet_command_forbidden")
    return "; ".join(normalised)


def _remote_policy_activation_prefix(activation: str) -> str:
    lines = [
        f"export {ARTIFACT_POLICY_ENV}={ARTIFACT_POLICY}",
        f"readonly {ARTIFACT_POLICY_ENV}",
    ]
    if activation:
        lines.append(activation)
    lines.append(
        f'test "${{{ARTIFACT_POLICY_ENV}}}" = {shlex.quote(ARTIFACT_POLICY)}'
    )
    return "; ".join(lines) + "; "


def _select_remote_setup(path: Path, requested: str) -> RemoteSetup:
    payload = _load_json(_lexical_absolute(path))
    targets = payload.get("hardware_targets")
    if not isinstance(targets, list):
        raise CanaryError("hardware_matrix_targets_invalid")
    admitted: list[RemoteSetup] = []
    for raw in targets:
        if not isinstance(raw, Mapping):
            continue
        setup_id = str(raw.get("id") or "").strip()
        accelerator = str(raw.get("accelerator") or raw.get("provider") or "").strip().lower()
        if raw.get("enabled") is False or not accelerator.startswith("hailo8"):
            continue
        if requested and setup_id != requested:
            continue
        runtime_raw = raw.get("remote") or raw.get("runtime")
        runtime = dict(runtime_raw) if isinstance(runtime_raw, Mapping) else {}
        host = str(runtime.get("host") or "").strip()
        user = str(runtime.get("user") or "").strip()
        if not host or runtime.get("enabled") is False:
            continue
        port_raw = runtime.get("port", 22)
        if isinstance(port_raw, bool):
            raise CanaryError(f"remote_port_invalid:{setup_id}")
        try:
            port = int(port_raw)
        except (TypeError, ValueError) as exc:
            raise CanaryError(f"remote_port_invalid:{setup_id}") from exc
        if (
            not 1 <= port <= 65535
            or not re.fullmatch(r"[A-Za-z0-9._-]+", host)
            or host.startswith("-")
            or (user and not re.fullmatch(r"[A-Za-z0-9._-]+", user))
            or user.startswith("-")
        ):
            raise CanaryError(f"remote_host_invalid:{setup_id}")
        ssh_extra_args = str(runtime.get("ssh_extra_args") or "").strip()
        if ssh_extra_args:
            # HostConfig accepts arbitrary OpenSSH options; ProxyCommand and
            # LocalCommand can execute local programs.  This read-only source
            # canary therefore admits no per-matrix SSH option string and
            # relies on the already selected host/user/port and normal user
            # SSH configuration.
            raise CanaryError(
                f"remote_ssh_extra_args_forbidden:{setup_id}"
            )
        admitted.append(RemoteSetup(
            setup_id=setup_id,
            host=host,
            user=user,
            port=port,
            remote_base_dir=str(
                runtime.get("remote_base_dir") or "~/splitpoint_runs"
            ),
            remote_venv=_normalise_remote_venv(
                runtime.get("remote_venv") or runtime.get("venv") or ""
            ),
            ssh_extra_args="",
        ))
    if not admitted:
        suffix = f":{requested}" if requested else ""
        raise CanarySkip(f"configured_hailo8_remote_unavailable{suffix}")
    if len(admitted) != 1:
        raise CanaryError("multiple_hailo8_remotes_require_setup_id")
    return admitted[0]


def _attested_source_paths(plans: Sequence[BenchmarkPlan]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for plan in plans:
        paths.update({
            plan.hef, plan.receipt, plan.output_contracts,
            plan.benchmark_set_json,
        })
        if plan.onnx is not None:
            paths.add(plan.onnx)
        paths.update(image.source for image in plan.images)
    paths.update(_iter_package_files())
    paths.update({
        ROOT / "scripts/smoke_hailo10_hef_runner.py",
        ROOT / "scripts/smoke_hailo10_full_from_benchmarkset.py",
        PAYLOAD_VERIFIER,
    })
    return tuple(sorted(_lexical_absolute(path) for path in paths))


def _source_snapshot(paths: Sequence[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        _require_regular_file(path, label="attested_source")
        result[str(path)] = _sha256(path)
    return result


def _source_snapshot_after(
    before: Mapping[str, str],
) -> tuple[dict[str, str], list[str]]:
    after: dict[str, str] = {}
    changed: list[str] = []
    for raw_path, digest in before.items():
        path = Path(raw_path)
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                observed = "unsafe_or_nonregular"
            else:
                observed = _sha256(path)
        except Exception as exc:
            observed = f"unreadable:{type(exc).__name__}"
        after[raw_path] = observed
        if observed != digest:
            changed.append(raw_path)
    return after, changed


def _add_tar_bytes(
    archive: tarfile.TarFile, *, name: str, raw: bytes, mode: int = 0o644,
) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(raw)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    import io
    archive.addfile(info, io.BytesIO(raw))


def _open_regular_payload_source(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CanaryError(f"unsafe_payload_source:{path}") from exc
    try:
        opened = os.fstat(fd)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise CanaryError(f"unsafe_payload_source:{path}")
        return fd, opened
    except BaseException:
        os.close(fd)
        raise


def _payload_source_identity(path: Path) -> tuple[int, str, int]:
    fd, before = _open_regular_payload_source(path)
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(fd)
        current = path.lstat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(before, key) != getattr(after, key) for key in stable_fields)
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or size != after.st_size
        ):
            raise CanaryError(f"payload_source_changed_during_identity:{path}")
        mode = 0o755 if after.st_mode & 0o111 else 0o644
        return size, digest.hexdigest(), mode
    finally:
        os.close(fd)


def _add_tar_regular_file(
    archive: tarfile.TarFile,
    *,
    source: Path,
    name: str,
    expected_size: int,
    expected_mode: int,
) -> None:
    """Materialize one safe source as REGTYPE, even for local hardlinks."""
    fd, before = _open_regular_payload_source(source)
    try:
        if before.st_size != expected_size:
            raise CanaryError(f"payload_source_size_drift:{name}")
        info = tarfile.TarInfo(name=name)
        info.type = tarfile.REGTYPE
        info.size = expected_size
        info.mode = expected_mode
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        with os.fdopen(fd, "rb", closefd=False) as handle:
            archive.addfile(info, handle)
        after = os.fstat(fd)
        current = source.lstat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(before, key) != getattr(after, key) for key in stable_fields)
            or stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise CanaryError(f"payload_source_changed_during_archive:{name}")
    finally:
        os.close(fd)


def _iter_package_files() -> Iterable[Path]:
    package = ROOT / "onnx_splitpoint_tool"
    for path in sorted(package.rglob("*")):
        if any(part in EXCLUDED_TREE_NAMES for part in path.parts):
            continue
        if path.is_symlink():
            raise CanaryError(f"source_package_symlink:{path}")
        if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
            yield path


def _build_payload(
    plans: Sequence[BenchmarkPlan], out_dir: Path,
) -> PayloadBundle:
    bundle = out_dir / "hailo8_artifact_canary_payload.tar.gz"
    sources: list[tuple[Path, str, str, str]] = []
    seen_relatives: set[str] = set()

    def add(source: Path, relative: str, kind: str, model_id: str = "") -> None:
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not relative_path.parts
            or relative in seen_relatives
        ):
            raise CanaryError(f"payload_relative_path_invalid_or_duplicate:{relative}")
        _require_regular_file(source, label="payload_source")
        seen_relatives.add(relative)
        sources.append((source, relative, kind, model_id))

    for path in _iter_package_files():
        add(
            path, path.relative_to(ROOT).as_posix(), "runtime_source"
        )
    for script_name in (
        "smoke_hailo10_hef_runner.py",
        "smoke_hailo10_full_from_benchmarkset.py",
        PAYLOAD_VERIFIER.name,
    ):
        add(
            ROOT / "scripts" / script_name,
            f"scripts/{script_name}",
            "runtime_entry_point" if script_name != PAYLOAD_VERIFIER.name
            else "payload_verifier",
        )

    models: list[dict[str, Any]] = []
    for index, plan in enumerate(plans):
        relative_root = f"benchmark_sets/{index:02d}_{plan.model_id}"
        hef_relative = f"{relative_root}/hailo/hailo8/full/compiled.hef"
        receipt_relative = (
            f"{relative_root}/hailo/hailo8/full/"
            "hailo_hef_build_receipt.json"
        )
        output_contract_relative = f"{relative_root}/output_contracts.json"
        benchmark_json_relative = f"{relative_root}/benchmark_set.json"
        add(plan.benchmark_set_json, benchmark_json_relative, "benchmark_contract", plan.model_id)
        add(plan.output_contracts, output_contract_relative, "output_contract", plan.model_id)
        add(plan.hef, hef_relative, "precompiled_hef", plan.model_id)
        add(plan.receipt, receipt_relative, "build_receipt", plan.model_id)
        onnx_relative = ""
        if plan.onnx is not None:
            onnx_relative = f"{relative_root}/models/{plan.onnx.name}"
            add(plan.onnx, onnx_relative, "source_onnx", plan.model_id)
        image_rows: list[dict[str, Any]] = []
        for image_index, image_plan in enumerate(plan.images):
            image_name = f"images/{image_index:02d}_{image_plan.source.name}"
            image_relative = f"{relative_root}/{image_name}"
            add(image_plan.source, image_relative, "validation_image", plan.model_id)
            image_rows.append({
                "relative_path": image_name,
                "payload_path": image_relative,
                "source_name": image_plan.source.name,
                "sha256": image_plan.sha256,
                "size_bytes": image_plan.source.stat().st_size,
            })
        models.append({
            "index": index,
            "model_id": plan.model_id,
            "task": plan.task,
            "benchmark_set_relative": relative_root,
            "benchmark_set_json_path": benchmark_json_relative,
            "benchmark_set_json_sha256": _sha256(plan.benchmark_set_json),
            "output_contracts_path": output_contract_relative,
            "output_contracts_sha256": _sha256(plan.output_contracts),
            "hef_path": hef_relative,
            "hef_sha256": plan.hef_sha256,
            "hef_size_bytes": plan.hef.stat().st_size,
            "receipt_path": receipt_relative,
            "receipt_sha256": _sha256(plan.receipt),
            "onnx_path": onnx_relative,
            "onnx_sha256": plan.onnx_sha256,
            "receipt_source_onnx_sha256": plan.receipt_source_onnx_sha256,
            "receipt_compiler_onnx_sha256": plan.receipt_compiler_onnx_sha256,
            "preprocessing_contract_sha256": plan.preprocessing_contract_sha256,
            "output_contract_resolution_sha256": _json_sha256(
                plan.output_contract_resolution
            ),
            "images": image_rows,
        })

    file_rows = []
    source_identities: dict[str, tuple[int, str, int]] = {}
    for source, relative, kind, model_id in sorted(
        sources, key=lambda row: row[1]
    ):
        size_bytes, sha256, archive_mode = _payload_source_identity(source)
        source_identities[relative] = (size_bytes, sha256, archive_mode)
        file_rows.append({
            "path": relative,
            "kind": kind,
            "model_id": model_id,
            "size_bytes": size_bytes,
            "sha256": sha256,
        })
    manifest = {
        "schema": PAYLOAD_SCHEMA,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "artifact_policy": ARTIFACT_POLICY,
        "runtime_entry_point": RUNTIME_ENTRY_POINT,
        "compile_or_build_entry_points_allowed": [],
        "files": file_rows,
        "models": models,
    }
    manifest_raw = _json_bytes(manifest)
    with tarfile.open(bundle, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for source, relative, _kind, _model_id in sorted(
            sources, key=lambda row: row[1]
        ):
            size_bytes, _sha256_value, archive_mode = source_identities[relative]
            _add_tar_regular_file(
                archive,
                source=source,
                name=f"payload/{relative}",
                expected_size=size_bytes,
                expected_mode=archive_mode,
            )
        _add_tar_bytes(
            archive,
            name=f"payload/{PAYLOAD_MANIFEST_NAME}",
            raw=manifest_raw,
        )
    _require_regular_file(PAYLOAD_VERIFIER, label="payload_verifier")
    payload = PayloadBundle(
        path=bundle,
        sha256=_sha256(bundle),
        size_bytes=bundle.stat().st_size,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        verifier=PAYLOAD_VERIFIER,
        verifier_sha256=_sha256(PAYLOAD_VERIFIER),
        verifier_size_bytes=PAYLOAD_VERIFIER.stat().st_size,
    )
    _verify_payload_locally(payload, out_dir)
    return payload


def _expected_payload_attestation(payload: PayloadBundle) -> dict[str, Any]:
    return {
        "status": "PASS",
        "archive_sha256": payload.sha256,
        "archive_size_bytes": payload.size_bytes,
        "manifest_sha256": payload.manifest_sha256,
        "file_count": len(list(payload.manifest.get("files") or [])),
    }


def _verify_payload_locally(payload: PayloadBundle, out_dir: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".payload_verify_", dir=out_dir,
    ) as temporary:
        destination = Path(temporary) / "payload"
        process = subprocess.run(
            [
                sys.executable, "-B", str(payload.verifier),
                "--archive", str(payload.path),
                "--destination", str(destination),
                "--expected-archive-sha256", payload.sha256,
                "--expected-archive-size", str(payload.size_bytes),
                "--expected-manifest-sha256", payload.manifest_sha256,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
            env={**os.environ, ARTIFACT_POLICY_ENV: ARTIFACT_POLICY},
        )
        lines = [line for line in process.stdout.splitlines() if line.strip()]
        try:
            observed = json.loads(lines[-1]) if lines else {}
        except Exception as exc:
            raise CanaryError("local_payload_verifier_output_invalid") from exc
        expected = _expected_payload_attestation(payload)
        if (
            process.returncode != 0
            or not isinstance(observed, Mapping)
            or any(observed.get(key) != value for key, value in expected.items())
        ):
            detail = (process.stderr or process.stdout or "")[-2000:]
            raise CanaryError(f"local_payload_verification_failed:{detail}")


def _safe_extract(archive_path: Path, destination: Path) -> None:
    _require_regular_file(archive_path, label="result_archive")
    if archive_path.stat().st_size > MAX_RESULT_ARCHIVE_BYTES:
        raise CanaryError("result_archive_size_exceeded")
    destination = _lexical_absolute(destination)
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve(strict=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_RESULT_MEMBER_COUNT:
            raise CanaryError("result_archive_member_count_exceeded")
        names: set[str] = set()
        normalised_members: list[tuple[tarfile.TarInfo, PurePosixPath]] = []
        total = 0
        for member in members:
            relative = PurePosixPath(member.name)
            normalised_name = relative.as_posix()
            if (
                normalised_name in names
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[:1] != ("results",)
                or not (member.isfile() or member.isdir())
                or member.size < 0
                or member.size > MAX_RESULT_MEMBER_BYTES
            ):
                raise CanaryError(f"unsafe_result_archive_member:{member.name}")
            names.add(normalised_name)
            normalised_members.append((member, relative))
            total += int(member.size)
            if total > MAX_RESULT_TOTAL_BYTES:
                raise CanaryError("result_archive_total_bytes_exceeded")
            target = (destination / relative).resolve()
            try:
                target.relative_to(base)
            except ValueError as exc:
                raise CanaryError(
                    f"result_archive_path_escape:{member.name}"
                ) from exc
        for member, relative in normalised_members:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            cursor = destination
            for part in target.parent.relative_to(destination).parts:
                cursor /= part
                info = cursor.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise CanaryError(
                        f"unsafe_result_extract_parent:{member.name}"
                    )
            if member.isdir():
                if os.path.lexists(target):
                    info = target.lstat()
                    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                        raise CanaryError(
                            f"result_extract_directory_unsafe:{member.name}"
                        )
                else:
                    target.mkdir()
                continue
            if os.path.lexists(target):
                raise CanaryError(f"result_extract_target_exists:{member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise CanaryError(f"result_archive_member_unreadable:{member.name}")
            with target.open("xb") as output:
                written = 0
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    written += len(block)
                    if written > member.size:
                        raise CanaryError(
                            f"result_archive_member_expanded_size_drift:{member.name}"
                        )
                    output.write(block)
                if written != member.size:
                    raise CanaryError(
                        f"result_archive_member_size_mismatch:{member.name}"
                    )


def _local_runtime_probe() -> tuple[bool, str]:
    code = (
        "import numpy; from PIL import Image; "
        "\ntry:\n import hailo_platform\n"
        "except Exception:\n import hailort\n"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-c", code], text=True, capture_output=True,
            timeout=15, check=False,
            env={**os.environ, ARTIFACT_POLICY_ENV: ARTIFACT_POLICY},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"local_hailo_runtime_probe_failed:{type(exc).__name__}"
    detail = (process.stderr or process.stdout or "").strip()[-1000:]
    return process.returncode == 0, detail


def _smoke_command(
    *, python: str, benchmark_set: str, model_id: str, task: str,
    image: str, result_dir: str, setup_id: str,
) -> list[str]:
    return [
        python,
        "scripts/smoke_hailo10_full_from_benchmarkset.py",
        "--benchmark-set", benchmark_set,
        "--hw-arch", "hailo8",
        "--model", model_id,
        "--task", task,
        "--frames", "1",
        "--warmup", "0",
        "--inflight", "1",
        "--runtime-api", "vstreams",
        "--image", image,
        "--dump-outputs",
        "--dump-dir", f"{result_dir}/dump",
        "--artifacts-dir", f"{result_dir}/artifacts",
        "--diagnostic-only",
        "--backend-label", "native_full_hailo8",
        "--setup-id", setup_id,
        "--comparison-backend", "hailo8",
        "--json-out", f"{result_dir}/report.json",
    ]


def _execute_local(
    plans: Sequence[BenchmarkPlan], out_dir: Path, timeout_s: int,
) -> dict[str, Any]:
    ok, detail = _local_runtime_probe()
    if not ok:
        return {
            "status": "SKIP", "reason": "local_hailo8_runtime_unavailable",
            "detail": detail, "runtime_admitted": False,
        }
    results = out_dir / "results"
    rows: list[dict[str, Any]] = []
    failed = False
    for model_index, plan in enumerate(plans):
        for image_index, image in enumerate(plan.images):
            result_dir = results / f"{model_index:02d}_{plan.model_id}" / f"{image_index:02d}"
            result_dir.mkdir(parents=True, exist_ok=False)
            command = _smoke_command(
                python=sys.executable,
                benchmark_set=str(plan.benchmark_set),
                model_id=plan.model_id,
                task=plan.task,
                image=str(image.source),
                result_dir=str(result_dir),
                setup_id="local_hailo8",
            )
            process = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True,
                timeout=timeout_s, check=False,
                env={**os.environ, ARTIFACT_POLICY_ENV: ARTIFACT_POLICY},
            )
            (result_dir / "stdout.txt").write_text(
                process.stdout or "", encoding="utf-8",
            )
            (result_dir / "stderr.txt").write_text(
                process.stderr or "", encoding="utf-8",
            )
            rows.append({
                "model_id": plan.model_id, "image": image.source.name,
                "image_sha256": image.sha256,
                "returncode": int(process.returncode),
                "result_relative": result_dir.relative_to(out_dir).as_posix(),
                "expected_hef": str(plan.hef),
                "expected_hef_sha256": plan.hef_sha256,
                "artifact_policy": ARTIFACT_POLICY,
                "invocation_kind": "existing_hef_structural_runtime",
                "runtime_entry_point": RUNTIME_ENTRY_POINT,
            })
            failed = failed or process.returncode != 0
    return {
        "status": "FAIL" if failed else "COMPLETED",
        "reason": "local_structural_runtime_canary_failed" if failed else "",
        "runtime_admitted": True, "invocations": rows,
    }


def _remote_transport(
    setup: RemoteSetup, *, session_id: str,
) -> tuple[SSHTransport, RemoteProcessLeaseRegistry]:
    registry = RemoteProcessLeaseRegistry()
    scope = resolve_remote_process_lease_scope(
        registry=registry,
        fallback_run_id=f"v27713-hailo8-structural-canary-{setup.setup_id}",
        workflow_session_id=session_id,
    )
    return SSHTransport(HostConfig(
        id=setup.setup_id,
        label=setup.setup_id,
        host=setup.host,
        user=setup.user,
        port=setup.port,
        remote_base_dir=setup.remote_base_dir,
        ssh_extra_args=setup.ssh_extra_args,
    ), remote_lease_scope=scope, remote_lease_registry=registry), registry


def _validate_remote_root(value: str) -> PurePosixPath:
    root = PurePosixPath(str(value or ""))
    if (
        not root.is_absolute()
        or root == PurePosixPath("/")
        or len(root.parts) < 3
        or not re.fullmatch(r"/[A-Za-z0-9._/-]+", str(root))
        or ".." in root.parts
    ):
        raise CanaryError(f"unsafe_remote_base_dir:{value}")
    return root


def _execute_remote(
    plans: Sequence[BenchmarkPlan], out_dir: Path, timeout_s: int,
    setup: RemoteSetup, payload: PayloadBundle,
) -> dict[str, Any]:
    session = uuid.uuid4().hex
    transport, lease_registry = _remote_transport(
        setup, session_id=session,
    )
    ok, connection = transport.test_connection(timeout_s=min(timeout_s, 15))
    if not ok:
        return {
            "status": "SKIP", "reason": "remote_hailo8_unreachable",
            "detail": connection[-2000:], "runtime_admitted": False,
        }
    rc, probe_out = transport.run_read_only(
        "command -v python3 >/dev/null 2>&1 && "
        "command -v timeout >/dev/null 2>&1",
        timeout_s=min(timeout_s, 15),
    )
    if rc != 0:
        return {
            "status": "SKIP", "reason": "remote_python_or_timeout_unavailable",
            "detail": probe_out[-2000:], "runtime_admitted": False,
        }
    try:
        resolved_base = transport.resolve_path_read_only(
            setup.remote_base_dir, timeout_s=min(timeout_s, 15),
        )
    except Exception as exc:
        return {
            "status": "SKIP", "reason": "remote_base_resolution_failed",
            "detail": str(exc)[-2000:], "runtime_admitted": False,
        }
    base = _validate_remote_root(resolved_base)
    remote_root = base / f"v27713_hailo8_artifact_canary_{session}"
    quoted_root = shlex.quote(str(remote_root))
    activation = str(setup.remote_venv or "")
    policy_prefix = _remote_policy_activation_prefix(activation)
    runtime_admitted = False
    workspace_created = False
    payload_verified = False
    payload_attestation: dict[str, Any] = {}
    result_archive_attestation: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    cleanup_error = ""
    cleanup_unproven = False
    workspace_quarantined = False
    skip_reason = ""
    try:
        rc, output = transport.run(
            f"umask 077; mkdir -- {quoted_root}", timeout_s=min(timeout_s, 30),
        )
        if rc != 0:
            return {
                "status": "FAIL", "reason": "remote_workspace_create_failed",
                "detail": output[-2000:], "runtime_admitted": False,
            }
        workspace_created = True
        rc, output = transport.scp_upload(
            str(payload.path), f"{remote_root}/payload.tar.gz", timeout_s=timeout_s,
        )
        if rc != 0:
            failures.append(f"payload_upload_failed:{output[-1000:]}")
        if not failures:
            rc, output = transport.scp_upload(
                str(payload.verifier), f"{remote_root}/verify_payload.py",
                timeout_s=min(timeout_s, 120),
            )
            if rc != 0:
                failures.append(f"payload_verifier_upload_failed:{output[-1000:]}")
        if not failures:
            verifier_check_code = (
                "import hashlib,os,stat,sys; p=sys.argv[1]; s=os.lstat(p); "
                "assert stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode); "
                "assert s.st_size==int(sys.argv[3]); "
                "assert hashlib.sha256(open(p,'rb').read()).hexdigest()==sys.argv[2]"
            )
            verifier_check = " ".join(shlex.quote(value) for value in (
                "python3", "-c", verifier_check_code,
                f"{remote_root}/verify_payload.py",
                payload.verifier_sha256,
                str(payload.verifier_size_bytes),
            ))
            rc, output = transport.run(
                "set -e; "
                + _remote_policy_activation_prefix("")
                + verifier_check,
                timeout_s=min(timeout_s, 30),
            )
            if rc != 0:
                failures.append(f"payload_verifier_identity_failed:{output[-1000:]}")
        if not failures:
            verifier_argv = [
                "python3", "-B", f"{remote_root}/verify_payload.py",
                "--archive", f"{remote_root}/payload.tar.gz",
                "--destination", f"{remote_root}/payload",
                "--expected-archive-sha256", payload.sha256,
                "--expected-archive-size", str(payload.size_bytes),
                "--expected-manifest-sha256", payload.manifest_sha256,
            ]
            rc, output = transport.run(
                "set -e; "
                + _remote_policy_activation_prefix("")
                + " ".join(
                    shlex.quote(value) for value in verifier_argv
                ),
                timeout_s=max(timeout_s, 120),
            )
            lines = [line for line in output.splitlines() if line.strip()]
            try:
                observed = json.loads(lines[-1]) if lines else {}
            except Exception:
                observed = {}
            expected_payload = _expected_payload_attestation(payload)
            if (
                rc != 0
                or not isinstance(observed, Mapping)
                or any(
                    observed.get(key) != value
                    for key, value in expected_payload.items()
                )
            ):
                failures.append(f"payload_remote_verification_failed:{output[-1000:]}")
            else:
                payload_verified = True
                payload_attestation = dict(observed)
        if not failures:
            runtime_probe = (
                "set -e; " + policy_prefix
                + "python3 -c "
                + shlex.quote(
                    "import numpy; from PIL import Image\n"
                    "try:\n import hailo_platform\n"
                    "except Exception:\n import hailort\n"
                )
            )
            rc, output = transport.run(
                runtime_probe, timeout_s=min(timeout_s, 30),
            )
            if rc != 0:
                skip_reason = "remote_hailo8_runtime_unavailable"
                probe_out = output
            else:
                runtime_admitted = True
        if not failures and not skip_reason:
            inventory = list(payload.manifest.get("models") or [])
            for model in inventory:
                if not isinstance(model, Mapping):
                    failures.append("payload_model_inventory_invalid")
                    break
                model_id = str(model["model_id"])
                task = str(model["task"])
                benchmark_set = (
                    f"{remote_root}/payload/{model['benchmark_set_relative']}"
                )
                for image_index, image_row in enumerate(model["images"]):
                    result_rel = f"{int(model['index']):02d}_{model_id}/{image_index:02d}"
                    result_dir = f"{remote_root}/results/{result_rel}"
                    image = f"{benchmark_set}/{image_row['relative_path']}"
                    expected_hef = (
                        f"{remote_root}/payload/{model['hef_path']}"
                    )
                    argv = _smoke_command(
                        python="python3", benchmark_set=benchmark_set,
                        model_id=model_id, task=task, image=image,
                        result_dir=result_dir, setup_id=setup.setup_id,
                    )
                    command = (
                        "set -e; "
                        + policy_prefix
                        + f"cd {quoted_root}/payload; "
                        + f"mkdir -p -- {shlex.quote(result_dir)}; "
                        + "export PYTHONPATH=$PWD; "
                        + "timeout --signal=TERM "
                        + f"--kill-after={REMOTE_KILL_AFTER_S}s "
                        + f"{int(timeout_s)}s "
                        + " ".join(shlex.quote(value) for value in argv)
                    )
                    rc, output = transport.run(
                        command,
                        timeout_s=int(timeout_s) + REMOTE_TRANSPORT_GRACE_S,
                    )
                    log_path = out_dir / "remote_logs" / f"{result_rel.replace('/', '_')}.txt"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(output or "", encoding="utf-8")
                    rows.append({
                        "model_id": model_id,
                        "image": str(image_row["source_name"]),
                        "image_sha256": str(image_row["sha256"]),
                        "returncode": int(rc),
                        "result_relative": f"results/{result_rel}",
                        "log_relative": log_path.relative_to(out_dir).as_posix(),
                        "expected_hef": expected_hef,
                        "expected_hef_sha256": str(model["hef_sha256"]),
                        "artifact_policy": ARTIFACT_POLICY,
                        "invocation_kind": "existing_hef_structural_runtime",
                        "runtime_entry_point": RUNTIME_ENTRY_POINT,
                        "remote_timeout_s": int(timeout_s),
                        "remote_kill_after_s": REMOTE_KILL_AFTER_S,
                        "outer_transport_timeout_s": (
                            int(timeout_s) + REMOTE_TRANSPORT_GRACE_S
                        ),
                    })
                    if rc != 0:
                        failures.append(
                            f"remote_structural_runtime_canary_failed:"
                            f"{model_id}:{image_index}:rc={rc}"
                        )
                        if rc == 70 or "cleanup unproven" in output.lower():
                            cleanup_unproven = True
                            break
                if cleanup_unproven:
                    break
        if runtime_admitted and not cleanup_unproven:
            pack = (
                "set -e; "
                + _remote_policy_activation_prefix("")
                + f"tar -czf {quoted_root}/results.tar.gz "
                f"-C {quoted_root} results"
            )
            rc, output = transport.run(
                pack, timeout_s=min(max(timeout_s, 60), 600),
            )
            if rc == 70 or "cleanup unproven" in output.lower():
                cleanup_unproven = True
                failures.append("remote_result_pack_cleanup_unproven")
            elif rc != 0:
                failures.append(f"remote_result_pack_failed:{output[-1000:]}")
            else:
                archive_identity_code = (
                    "import hashlib,json,os,stat,sys; p=sys.argv[1]; "
                    "s=os.lstat(p); "
                    "assert stat.S_ISREG(s.st_mode) and not stat.S_ISLNK(s.st_mode); "
                    "assert 0 < s.st_size <= int(sys.argv[2]); "
                    "h=hashlib.sha256(); f=open(p,'rb'); "
                    "[h.update(b) for b in iter(lambda:f.read(1048576),b'')]; "
                    "print(json.dumps({'sha256':h.hexdigest(),'size_bytes':s.st_size},sort_keys=True))"
                )
                identity_argv = (
                    "python3", "-c", archive_identity_code,
                    f"{remote_root}/results.tar.gz",
                    str(MAX_RESULT_ARCHIVE_BYTES),
                )
                rc, output = transport.run(
                    "set -e; "
                    + _remote_policy_activation_prefix("")
                    + " ".join(shlex.quote(value) for value in identity_argv),
                    timeout_s=min(max(timeout_s, 120), 600),
                )
                lines = [line for line in output.splitlines() if line.strip()]
                try:
                    identity = json.loads(lines[-1]) if lines else {}
                except Exception:
                    identity = {}
                if (
                    rc != 0
                    or not isinstance(identity, Mapping)
                    or not HEX64.fullmatch(
                        str(identity.get("sha256") or "").lower()
                    )
                    or type(identity.get("size_bytes")) is not int
                    or not 0 < int(identity["size_bytes"]) <= MAX_RESULT_ARCHIVE_BYTES
                ):
                    failures.append(
                        f"remote_result_archive_identity_failed:{output[-1000:]}"
                    )
                else:
                    result_archive_attestation = dict(identity)
                    local_archive = out_dir / "remote_results.tar.gz"
                    rc, output = transport.scp_download(
                        f"{remote_root}/results.tar.gz", str(local_archive),
                        timeout_s=max(timeout_s, 120),
                    )
                if rc != 0:
                    failures.append(f"remote_result_download_failed:{output[-1000:]}")
                elif result_archive_attestation and (
                    local_archive.stat().st_size
                    != int(result_archive_attestation["size_bytes"])
                    or _sha256(local_archive)
                    != str(result_archive_attestation["sha256"])
                ):
                    failures.append("remote_result_archive_download_identity_mismatch")
                elif result_archive_attestation:
                    _safe_extract(local_archive, out_dir)
    finally:
        if workspace_created:
            if cleanup_unproven or lease_registry.active_count() != 0:
                workspace_quarantined = True
                failures.append("remote_workspace_cleanup_withheld_unproven_process_state")
            else:
                rc, output = transport.run(
                    f"rm -rf -- {quoted_root}", timeout_s=min(timeout_s, 60),
                )
                if rc != 0:
                    cleanup_error = output[-2000:]
                    failures.append("remote_workspace_cleanup_failed")
    if skip_reason and not failures:
        return {
            "status": "SKIP",
            "reason": skip_reason,
            "detail": probe_out[-2000:],
            "runtime_admitted": False,
            "payload_verified": payload_verified,
            "payload_attestation": payload_attestation,
        }
    return {
        "status": "FAIL" if failures else "COMPLETED",
        "reason": failures[0] if failures else "",
        "runtime_admitted": runtime_admitted,
        "payload_verified": payload_verified,
        "payload_attestation": payload_attestation,
        "result_archive_attestation": result_archive_attestation,
        "invocations": rows,
        "failures": failures,
        "cleanup_error": cleanup_error,
        "remote_workspace_quarantined": workspace_quarantined,
    }


def _finite_output_summary(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for row in value.values():
        if not isinstance(row, Mapping) or not isinstance(row.get("shape"), list):
            return False
        if not row["shape"] or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in row["shape"]
        ):
            return False
        for key in ("min", "max"):
            number = row.get(key)
            if not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                return False
    return True


def _validate_dump_artifacts(
    result_dir: Path, *, plan: BenchmarkPlan, setup_id: str,
    image_sha256: str, output_summary: Any,
) -> bool:
    dump_dir = result_dir / "dump"
    output_path = dump_dir / "native_full_outputs_manifest.json"
    input_path = dump_dir / "native_full_input_manifest.json"
    output = _load_json(
        _require_safe_output_file(
            result_dir, output_path, label="output_dump_manifest",
        )
    )
    input_payload = _load_json(
        _require_safe_output_file(
            result_dir, input_path, label="input_dump_manifest",
        )
    )
    resolution = output.get("authoritative_output_contract_resolution")
    entries = output.get("outputs")
    if (
        output.get("schema") != "onnx-splitpoint/runner-output-dump"
        or int(output.get("schema_version") or 0) != 4
        or str(output.get("model") or "") != plan.model_id
        or str(output.get("backend") or "") != "native_full_hailo8"
        or str(output.get("setup_id") or "") != setup_id
        or str(output.get("input_image_sha256") or "").lower()
        != image_sha256
        or not isinstance(resolution, Mapping)
        or resolution.get("contract_resolution_status") != "attested"
        or resolution.get("authoritative_output_contract") is not True
        or _portable_output_contract_resolution(resolution)
        != plan.output_contract_resolution
        or not isinstance(entries, list)
        or not entries
        or input_payload.get("schema")
        != "onnx-splitpoint/native-full-input-dump"
        or int(input_payload.get("schema_version") or 0) != 2
        or str(input_payload.get("model") or "") != plan.model_id
        or str(input_payload.get("task") or "") != plan.task
        or str(input_payload.get("setup_id") or "") != setup_id
        or str(input_payload.get("input_image_sha256") or "").lower()
        != image_sha256
        or input_payload.get("preprocessing_contract")
        != plan.preprocessing_contract
        or _sha_token(input_payload.get("preprocessing_contract_sha256"))
        != plan.preprocessing_contract_sha256
        or preprocessing_contract_sha256(
            input_payload.get("preprocessing_contract")
        ) != plan.preprocessing_contract_sha256
        or not isinstance(output_summary, Mapping)
    ):
        return False
    entry_names = [
        str(entry.get("name") or "")
        for entry in entries if isinstance(entry, Mapping)
    ]
    if (
        len(entry_names) != len(entries)
        or not all(entry_names)
        or len(set(entry_names)) != len(entry_names)
        or set(output_summary) != set(entry_names)
    ):
        return False
    try:
        import numpy as np
    except Exception:
        return False
    for entry in entries:
        if not isinstance(entry, Mapping):
            return False
        relative = Path(str(entry.get("file") or ""))
        if (
            relative.is_absolute() or ".." in relative.parts
            or len(relative.parts) != 1
        ):
            return False
        tensor = dump_dir / relative
        try:
            tensor = _require_safe_output_file(
                result_dir, tensor, label="output_tensor",
            )
        except CanaryError:
            return False
        if (
            int(entry.get("bytes") or -1) != tensor.stat().st_size
            or _sha_token(entry.get("sha256")) != _sha256(tensor)
            or not isinstance(entry.get("shape"), list)
            or not entry.get("shape")
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in entry["shape"]
            )
        ):
            return False
        summary = output_summary.get(str(entry.get("name") or ""))
        if not isinstance(summary, Mapping):
            return False
        try:
            dtype = np.dtype(str(entry.get("dtype") or ""))
            shape = tuple(int(dim) for dim in entry["shape"])
            expected_items = math.prod(shape)
            array = np.fromfile(tensor, dtype=dtype)
            if array.size != expected_items:
                return False
            array = array.reshape(shape)
            if (
                not bool(np.isfinite(array).all())
                or summary.get("shape") != list(shape)
                or str(summary.get("dtype") or "") != str(dtype)
                or float(summary.get("min")) != float(np.min(array))
                or float(summary.get("max")) != float(np.max(array))
            ):
                return False
        except Exception:
            return False
    return True


def _validate_results(
    plans: Sequence[BenchmarkPlan], out_dir: Path,
    execution: Mapping[str, Any], setup_id: str,
) -> list[dict[str, Any]]:
    invocations = execution.get("invocations")
    if not isinstance(invocations, list):
        raise CanaryError("execution_invocations_missing")
    if execution.get("runtime_admitted") is not True:
        raise CanaryError("completed_execution_without_runtime_admission")
    expected_keys = {
        (plan.model_id, image.sha256)
        for plan in plans for image in plan.images
    }
    if len(invocations) != len(expected_keys):
        raise CanaryError(
            "execution_invocation_count_mismatch:"
            f"{len(invocations)}:{len(expected_keys)}"
        )
    plan_by_model = {plan.model_id: plan for plan in plans}
    preflight_keys: list[tuple[str, str]] = []
    preflight_result_paths: list[str] = []
    for invocation in invocations:
        if not isinstance(invocation, Mapping):
            raise CanaryError("execution_invocation_invalid")
        model_id = str(invocation.get("model_id") or "")
        image_sha256 = _sha_token(invocation.get("image_sha256"))
        relative = Path(str(invocation.get("result_relative") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative in {Path(""), Path(".")}
        ):
            raise CanaryError("unsafe_execution_result_relative_path")
        preflight_keys.append((model_id, image_sha256))
        preflight_result_paths.append(relative.as_posix())
    if (
        set(preflight_keys) != expected_keys
        or len(set(preflight_keys)) != len(preflight_keys)
    ):
        raise CanaryError("execution_invocation_exact_set_mismatch")
    if len(set(preflight_result_paths)) != len(preflight_result_paths):
        raise CanaryError("execution_result_paths_not_unique")

    observed_keys: set[tuple[str, str]] = set()
    observed_result_paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    for invocation, invocation_key, relative_text in zip(
        invocations, preflight_keys, preflight_result_paths,
    ):
        model_id, image_sha256 = invocation_key
        plan = plan_by_model.get(model_id)
        if plan is None:
            raise CanaryError(f"unexpected_execution_model:{model_id}")
        if (
            invocation_key not in expected_keys
            or invocation_key in observed_keys
        ):
            raise CanaryError(
                f"execution_invocation_identity_invalid:{model_id}:{image_sha256}"
            )
        observed_keys.add(invocation_key)
        relative = Path(relative_text)
        if relative_text in observed_result_paths:
            raise CanaryError("unsafe_execution_result_relative_path")
        observed_result_paths.add(relative_text)
        result_dir = out_dir / relative
        report = _load_json(_require_safe_output_file(
            out_dir, result_dir / "report.json", label="runtime_report",
        ))
        image_name = str(invocation.get("image") or "")
        image_matches = [
            image for image in plan.images
            if image.sha256 == image_sha256
            and image.source.name == image_name
        ]
        if len(image_matches) != 1:
            raise CanaryError(f"execution_image_identity_invalid:{model_id}:{image_name}")
        image = image_matches[0]
        dump_ok = _validate_dump_artifacts(
            result_dir,
            plan=plan,
            setup_id=setup_id,
            image_sha256=image.sha256,
            output_summary=report.get("output_summary"),
        )
        common_ok = bool(
            type(invocation.get("returncode")) is int
            and invocation.get("returncode") == 0
            and invocation.get("artifact_policy") == ARTIFACT_POLICY
            and invocation.get("invocation_kind")
            == "existing_hef_structural_runtime"
            and invocation.get("runtime_entry_point") == RUNTIME_ENTRY_POINT
            and str(invocation.get("expected_hef") or "")
            == str(report.get("hef") or "")
            and _sha_token(invocation.get("expected_hef_sha256"))
            == plan.hef_sha256
            and report.get("ok") is True
            and report.get("diagnostic_only") is True
            and report.get("claim_eligible") is False
            and str(report.get("backend") or "") == "native_full_hailo8"
            and str(report.get("comparison_backend") or "") == "hailo8"
            and str(report.get("model") or "") == model_id
            and str(report.get("task") or "") == plan.task
            and str(report.get("setup_id") or "") == setup_id
            and int(report.get("completed_frames") or 0) == 1
            and str(report.get("completed_work_units_status") or "")
            == "exact_runtime_counter"
            and report.get("runtime_input_binding_verified") is True
            and report.get("preprocessing_contract_attested") is True
            and report.get("preprocessing_contract")
            == plan.preprocessing_contract
            and _sha_token(report.get("preprocessing_contract_sha256"))
            == plan.preprocessing_contract_sha256
            and preprocessing_contract_sha256(
                report.get("preprocessing_contract")
            ) == plan.preprocessing_contract_sha256
            and str(report.get("input_image_sha256") or "").lower()
            == image.sha256
            and _finite_output_summary(report.get("output_summary"))
            and dump_ok
        )
        detection_ok = True
        if plan.task == "detection":
            artifact_path = result_dir / "report.completed_task_result_artifact.json"
            try:
                artifact_path = _require_safe_output_file(
                    out_dir, artifact_path,
                    label="completed_task_result_artifact",
                )
                artifact_safe = True
            except CanaryError:
                artifact_safe = False
            detection_ok = bool(
                report.get("postprocess_included") is True
                and report.get("postprocess_completion_verified") is True
                and int(report.get("postprocess_completed_frames") or 0) == 1
                and report.get("completed_task_endpoint_attested") is True
                and str(report.get("completed_task_endpoint_attestation_status") or "")
                == "passed"
                and report.get("completed_task_result_artifact_saved") is True
                and artifact_safe
                and _sha256(artifact_path)
                == str(report.get("completed_task_result_artifact_file_sha256") or "")
            )
        status = "PASS" if common_ok and detection_ok else "FAIL"
        rows.append({
            "model_id": model_id,
            "task": plan.task,
            "image": image_name,
            "image_sha256": image.sha256,
            "hef_sha256": plan.hef_sha256,
            "status": status,
            "structural_preprocessing_binding_attested": bool(
                report.get("preprocessing_contract")
                == plan.preprocessing_contract
                and _sha_token(report.get("preprocessing_contract_sha256"))
                == plan.preprocessing_contract_sha256
            ),
            "structural_output_dump_attested": dump_ok,
            "structural_postprocess_completed": (
                report.get("completed_task_endpoint_attested") is True
                if plan.task == "detection" else None
            ),
            "numerical_correctness_attested": False,
            "backend_parity_attested": False,
        })
    if observed_keys != expected_keys:
        raise CanaryError("execution_invocation_exact_set_mismatch")
    if any(row["status"] != "PASS" for row in rows):
        raise CanaryError("structural_artifact_runtime_validation_failed")
    return rows


ExecuteFn = Callable[
    [Sequence[BenchmarkPlan], Path, int, RemoteSetup | None, PayloadBundle],
    dict[str, Any],
]


def _default_execute(
    plans: Sequence[BenchmarkPlan], out_dir: Path, timeout_s: int,
    setup: RemoteSetup | None, payload: PayloadBundle,
) -> dict[str, Any]:
    if setup is None:
        return _execute_local(plans, out_dir, timeout_s)
    return _execute_remote(
        plans, out_dir, timeout_s, setup, payload,
    )


def _base_result(config: CanaryConfig) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "status": "FAIL",
        "reason": "",
        "benchmark_sets": [str(path) for path in config.benchmark_sets],
        "out_dir": str(config.out_dir),
        "max_images": config.max_images,
        "require_hardware": config.require_hardware,
        "canary_scope": "structural_artifact_and_runtime_only",
        "numerical_correctness_attested": False,
        "backend_parity_attested": False,
        "artifact_policy": ARTIFACT_POLICY,
        "runtime_entry_point": RUNTIME_ENTRY_POINT,
        "compile_or_build_entry_points_allowed": [],
        "attested_source_files_modified": None,
        "attested_source_files_checked": [],
        "model_results": [],
    }


def run_canary(
    config: CanaryConfig,
    *,
    execute: ExecuteFn = _default_execute,
    validate: Callable[[Sequence[BenchmarkPlan], Path, Mapping[str, Any], str], list[dict[str, Any]]] = _validate_results,
) -> tuple[dict[str, Any], int]:
    result = _base_result(config)
    returncode = 1
    before: dict[str, str] = {}
    try:
        if not 1 <= int(config.max_images) <= 16:
            raise CanaryError("max_images_must_be_between_1_and_16")
        if int(config.timeout_s) <= 0:
            raise CanaryError("timeout_must_be_positive")
        if not config.benchmark_sets:
            raise CanaryError("at_least_one_benchmark_set_is_required")
        if config.setup_id and config.hardware_matrix is None:
            raise CanaryError("setup_id_requires_hardware_matrix")
        plans = tuple(
            _attest_benchmark_set(path, max_images=config.max_images)
            for path in config.benchmark_sets
        )
        out_dir = _prepare_output(
            config.out_dir,
            inputs=tuple(plan.benchmark_set for plan in plans),
        )
        model_ids = [plan.model_id for plan in plans]
        if len(set(model_ids)) != len(model_ids):
            raise CanaryError("benchmark_set_model_ids_must_be_unique")
        source_paths = list(_attested_source_paths(plans))
        if config.hardware_matrix is not None:
            source_paths.append(_lexical_absolute(config.hardware_matrix))
        source_paths = sorted(set(source_paths))
        before = _source_snapshot(source_paths)
        result["attested_source_files_checked"] = sorted(before)
        result["attested_source_file_count"] = len(before)
        result["source_snapshot_before_sha256"] = _json_sha256(before)
        setup = (
            _select_remote_setup(config.hardware_matrix, config.setup_id)
            if config.hardware_matrix is not None else None
        )
        payload = _build_payload(plans, out_dir)
        expected_payload = _expected_payload_attestation(payload)
        result["payload_bundle_attestation"] = {
            **expected_payload,
            "local_verification": "PASS",
            "verifier_sha256": payload.verifier_sha256,
            "manifest_model_count": len(
                list(payload.manifest.get("models") or [])
            ),
        }
        execution = execute(
            plans, out_dir, config.timeout_s, setup, payload,
        )
        result["runtime"] = dict(execution)
        status = str(execution.get("status") or "FAIL").upper()
        if status == "SKIP":
            if execution.get("runtime_admitted") is not False:
                raise CanaryError("skip_after_runtime_admission_forbidden")
            result["status"] = "FAIL" if config.require_hardware else "SKIP"
            result["reason"] = str(execution.get("reason") or "hardware_unavailable")
            returncode = 2 if config.require_hardware else 0
        elif status != "COMPLETED":
            raise CanaryError(
                str(execution.get("reason") or "hardware_execution_failed")
            )
        else:
            if execution.get("runtime_admitted") is not True:
                raise CanaryError(
                    "completed_execution_without_runtime_admission"
                )
            if setup is not None:
                observed_payload = execution.get("payload_attestation")
                if (
                    execution.get("payload_verified") is not True
                    or not isinstance(observed_payload, Mapping)
                    or any(
                        observed_payload.get(key) != value
                        for key, value in expected_payload.items()
                    )
                ):
                    raise CanaryError(
                        "remote_payload_attestation_not_bound_to_local_payload"
                    )
            setup_id = setup.setup_id if setup is not None else "local_hailo8"
            model_results = validate(plans, out_dir, execution, setup_id)
            result.update({
                "status": "PASS",
                "reason": "",
                "setup_id": setup_id,
                "model_results": model_results,
                "validated_invocation_count": len(model_results),
            })
            returncode = 0
    except CanarySkip as exc:
        result["status"] = "FAIL" if config.require_hardware else "SKIP"
        result["reason"] = str(exc)
        returncode = 2 if config.require_hardware else 0
    except Exception as exc:
        result["status"] = "FAIL"
        result["reason"] = f"{type(exc).__name__}:{exc}"
        returncode = 1
    finally:
        if before:
            after, changed = _source_snapshot_after(before)
            result["source_snapshot_after_sha256"] = _json_sha256(after)
            result["attested_source_files_modified"] = bool(changed)
            result["attested_source_files_changed"] = changed
            if changed:
                prior = str(result.get("reason") or "")
                result["status"] = "FAIL"
                result["reason"] = (
                    "attested_source_files_changed_during_canary"
                    + (f"; prior={prior}" if prior else "")
                )
                returncode = 1
    return result, returncode


def _write_result(result: Mapping[str, Any], out_dir: Path) -> None:
    path = _lexical_absolute(out_dir) / "canary_result.json"
    raw = _json_bytes(result)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(path.parent, directory_flags)
    except OSError as exc:
        raise CanaryError("canary_result_parent_unavailable_or_unsafe") from exc
    created = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        created = True
        try:
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise
        info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_size != len(raw):
            raise CanaryError("canary_result_receipt_identity_failed")
        read_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        read_fd = os.open(path.name, read_flags, dir_fd=parent_fd)
        with os.fdopen(read_fd, "rb", closefd=True) as handle:
            observed = handle.read()
        if observed != raw:
            raise CanaryError("canary_result_receipt_verification_failed")
        os.fsync(parent_fd)
    except BaseException:
        if created:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(parent_fd)


def _parse_args(argv: Sequence[str] | None = None) -> CanaryConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-set", action="append", required=True,
        help="Existing generated BenchmarkSet root; repeat for more models.",
    )
    parser.add_argument("--out", required=True, help="New or empty output directory.")
    parser.add_argument(
        "--hardware-matrix", default="",
        help="Optional hardware_matrix.json for remote Hailo-8 execution.",
    )
    parser.add_argument(
        "--setup-id", default="",
        help="Exact Hailo-8 setup ID when the matrix contains multiple targets.",
    )
    parser.add_argument(
        "--max-images", type=int, default=1,
        help="Real validation images per model (default 1, maximum 16).",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=300,
        help="Timeout for each probe, transfer, or structural runtime invocation.",
    )
    parser.add_argument(
        "--require-hardware", action="store_true",
        help="Promote a pre-admission hardware SKIP to FAIL (exit 2).",
    )
    ns = parser.parse_args(argv)
    return CanaryConfig(
        benchmark_sets=tuple(Path(value) for value in ns.benchmark_set),
        out_dir=Path(ns.out),
        hardware_matrix=(Path(ns.hardware_matrix) if ns.hardware_matrix else None),
        setup_id=str(ns.setup_id or ""),
        max_images=int(ns.max_images),
        require_hardware=bool(ns.require_hardware),
        timeout_s=int(ns.timeout_s),
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_args(argv)
    result, returncode = run_canary(config)
    try:
        result["result_receipt_written"] = True
        _write_result(result, config.out_dir)
    except Exception as exc:
        result["result_receipt_written"] = False
        if result.get("status") == "PASS":
            result["status"] = "FAIL"
            result["reason"] = (
                "canary_result_receipt_write_failed:"
                f"{type(exc).__name__}:{exc}"
            )
            returncode = 1
    display_result = dict(result)
    checked = display_result.pop("attested_source_files_checked", None)
    if isinstance(checked, list):
        display_result["attested_source_files_checked_omitted_from_stdout"] = len(
            checked
        )
    print(json.dumps(display_result, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"V27713_HAILO8_ARTIFACT_CANARY={result['status']}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
