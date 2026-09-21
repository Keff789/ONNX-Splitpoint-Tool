#!/usr/bin/env python3
"""Concurrent Hailo-8 -> TensorRT -> post-processing normal-runner adapter.

The implementation vendors the hardware-proven YOLOv7 Three-Stage canary
runtime and exposes it through the normal benchmark-set runner contract.
Performance and the quality oracle remain separated:

    timed:      P1 -> P2 -> fast contract-bound post-processing
    postflight: fast result -> frozen oracle parity

The helper is intentionally fail-closed.  It currently admits the exact
Hailo-8 YOLOv7 anchor-multiscale contract validated by the v2.79 canaries.
Other model/producer families remain on their existing normal-runner paths
until their concurrent integration is admitted independently.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import site
import subprocess
import sys
import traceback
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "onnx-splitpoint/native-dual-endpoint-result"
from onnx_splitpoint_tool.release_identity import BUILD_ID
DIAGNOSTIC_SCOPE = "diagnostic_single_image"
CLAIM_SCOPE = "claim_gate_32"
CLAIM_ITEM_COUNT = 32
DIAGNOSTIC_ITEM_COUNT = 1
DATASET_MANIFEST_SHA256 = "2de36f0f8949e4f1fcbd0eefbda1f4d18b411208fa91e2dd2bf985a56d0f22e2"
REFERENCE_REPORT_SHA256 = "4f4aff24452f5833bdcec8de31fa1f3173ccabc2a78372744f8bf59cd5ee0cc9"


@dataclass(frozen=True)
class ThreeStageInvocation:
    """Fully resolved, immutable input contract for one canary invocation."""

    benchmark_set: Path
    case_id: str
    setup_id: str
    corpus_manifest: Path
    dataset_manifest: Path
    reference_report: Path
    quality_binding: Path
    out_root: Path
    expected_item_count: int
    model_id: str
    precision: str
    execution_scope: str
    claim_eligible: bool
    corpus_manifest_sha256: str
    dataset_manifest_sha256: str
    reference_report_sha256: str
    quality_binding_sha256: str
    out_root_binding_sha256: str


@dataclass(frozen=True)
class ThreeStageChildResult:
    returncode: int
    console: Path
    command: tuple[str, ...]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _add_extra_sites() -> None:
    for raw in os.environ.get("SPLITPOINT_EXTRA_SITES", "").split(os.pathsep):
        raw = raw.strip()
        if raw:
            site.addsitedir(raw)


def _artifact_path(binding: Mapping[str, Any], key: str) -> Path:
    artifacts = binding.get("artifacts") or {}
    item = artifacts.get(key) or {}
    path = item.get("path") if isinstance(item, Mapping) else None
    if not path:
        raise RuntimeError(f"native_split_quality_binding_missing_artifact:{key}")
    p = Path(str(path)).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"native_split_quality_binding_artifact_missing:{key}:{p}")
    expected = str(item.get("sha256") or "")
    if expected.startswith("sha256:"):
        expected = expected[7:]
    if expected and _sha256(p) != expected:
        raise RuntimeError(f"native_split_quality_binding_artifact_sha256_mismatch:{key}:{p}")
    return p


def _normalise_status(value: Any) -> str:
    return str(value or "").strip().upper()


def _pick_number(payload: Any, *keys: str) -> float | None:
    """Recursively find the first finite numeric value for one of *keys*."""
    wanted = set(keys)
    queue = [payload]
    seen: set[int] = set()
    while queue:
        value = queue.pop(0)
        ident = id(value)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(value, Mapping):
            for key in keys:
                candidate = value.get(key)
                if isinstance(candidate, (int, float)):
                    return float(candidate)
            for k, v in value.items():
                if k in wanted and isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, (Mapping, list, tuple)):
                    queue.append(v)
        elif isinstance(value, (list, tuple)):
            queue.extend(v for v in value if isinstance(v, (Mapping, list, tuple)))
    return None


def _pick_mapping(payload: Any, *keys: str) -> Mapping[str, Any] | None:
    queue = [payload]
    seen: set[int] = set()
    while queue:
        value = queue.pop(0)
        ident = id(value)
        if ident in seen:
            continue
        seen.add(ident)
        if isinstance(value, Mapping):
            for key in keys:
                candidate = value.get(key)
                if isinstance(candidate, Mapping):
                    return candidate
            queue.extend(v for v in value.values() if isinstance(v, (Mapping, list, tuple)))
        elif isinstance(value, (list, tuple)):
            queue.extend(v for v in value if isinstance(v, (Mapping, list, tuple)))
    return None


def _package_version_from_source(tool_root: Path) -> str:
    """Read the source-tree package version without importing the package."""
    package = tool_root / "onnx_splitpoint_tool"

    def literal(path: Path, variable: str) -> str | None:
        if not path.is_file():
            return None
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise RuntimeError(f"source_snapshot_package_identity_unreadable:{path}") from exc
        for node in tree.body:
            value_node: ast.expr | None = None
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == variable
                for target in node.targets
            ):
                value_node = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == variable
            ):
                value_node = node.value
            if value_node is None:
                continue
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise RuntimeError("source_snapshot_package_version_not_literal") from exc
            return value.strip() if isinstance(value, str) else None
        return None

    # Current releases keep one authoritative literal in release_identity.py.
    # The __init__ literal is retained only as a compatibility fallback for
    # older source trees used by the provenance regression fixture.
    release_identity = package / "release_identity.py"
    legacy_identity = package / "__init__.py"
    version = literal(release_identity, "VERSION")
    if not version:
        version = literal(legacy_identity, "__version__")

    if not version:
        raise RuntimeError(
            f"source_snapshot_package_version_missing:{release_identity}:{legacy_identity}"
        )
    if version in {".", ".."} or any(char in version for char in ("/", "\\", "\0")):
        raise RuntimeError(f"source_snapshot_package_version_unsafe:{version!r}")
    return version


def _source_snapshot_prefix(tool_root: Path) -> str:
    return f"ONNX-Splitpoint-Tool_v{_package_version_from_source(tool_root)}"


def _create_source_snapshot(tool_root: Path, output: Path) -> Path:
    prefix = _source_snapshot_prefix(tool_root)
    members = (
        "onnx_splitpoint_tool/__init__.py",
        "onnx_splitpoint_tool/release_identity.py",
        "onnx_splitpoint_tool/native_three_stage.py",
        "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py",
    )
    missing = [rel for rel in members if not (tool_root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            "source_snapshot_required_members_missing:" + ",".join(missing)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for rel in members:
            path = tool_root / rel
            zf.write(path, arcname=f"{prefix}/{rel}")
    return output


def _strip_sha256(value: Any) -> str:
    digest = str(value or "").strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[7:]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError(f"invalid_sha256:{value!r}")
    return digest


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _out_root_binding_sha256(
    out_root: Path, *, execution_scope: str, model_id: str, case_id: str, setup_id: str
) -> str:
    return _canonical_payload_sha256(
        {
            "out_root": str(out_root.resolve()),
            "execution_scope": execution_scope,
            "model_id": model_id,
            "case_id": case_id,
            "setup_id": setup_id,
        }
    )


def _invocation_receipt(inv: ThreeStageInvocation) -> dict[str, Any]:
    payload = asdict(inv)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    receipt: dict[str, Any] = {
        "schema": "onnx-splitpoint/three-stage-invocation",
        "schema_version": 1,
        **payload,
    }
    receipt["invocation_sha256"] = _canonical_payload_sha256(receipt)
    return receipt


def _prepare_out_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and (not path.is_dir() or path.is_symlink()):
        raise RuntimeError(f"three_stage_out_root_not_directory:{path}")
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".three_stage_write_probe"
    try:
        probe.write_text("write-probe\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"three_stage_out_root_not_writable:{path}") from exc
    return path


def _find_report_only_below(out_root: Path) -> Path | None:
    root = out_root.resolve()
    reports = sorted(
        root.rglob("three_stage_canary_report.json"),
        key=lambda candidate: candidate.stat().st_mtime_ns,
        reverse=True,
    )
    for report in reports:
        if report.is_symlink():
            continue
        resolved = report.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _child_failure_details(report: Mapping[str, Any]) -> tuple[str, str]:
    failure_class = str(
        report.get("failure_class")
        or report.get("error_class")
        or report.get("exception_type")
        or ""
    )
    failure_reason = str(
        report.get("failure_reason")
        or report.get("error")
        or report.get("reason")
        or ""
    )
    if not failure_class and ":" in failure_reason:
        prefix = failure_reason.split(":", 1)[0].strip()
        if prefix and prefix.replace("_", "").isalnum():
            failure_class = prefix
    return failure_class, failure_reason


def _write_out_root_index(
    *,
    out_root: Path,
    out_root_binding_sha256: str,
    result_json: Path,
    invocation: ThreeStageInvocation | None,
    report_path: Path | None,
    console: Path | None,
    failure_receipt: Path | None,
) -> Path:
    def artifact(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            return None
        return {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}

    invocation_path = out_root / "three_stage_invocation_receipt.json"
    command_path = out_root / "three_stage_remote_command_receipt.json"
    payload = {
        "schema": "onnx-splitpoint/three-stage-artifact-index",
        "schema_version": 1,
        "out_root": str(out_root),
        "out_root_binding_sha256": out_root_binding_sha256,
        "invocation_sha256": (
            _invocation_receipt(invocation)["invocation_sha256"] if invocation else ""
        ),
        "artifacts": {
            "invocation_receipt": artifact(invocation_path),
            "remote_command_receipt": artifact(command_path),
            "canonical_result": artifact(result_json),
            "child_report": artifact(report_path),
            "child_console": artifact(console),
            "failure_receipt": artifact(failure_receipt),
        },
    }
    index = out_root / "three_stage_artifact_index.json"
    _write_json(index, payload)
    return index


def _require_regular_json(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".json" or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{role}_not_regular_json:{path}")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{role}_json_root_not_object:{path}")
    return payload, _sha256(path)


def _verify_expected_file_sha256(
    path: Path, *, role: str, expected: str, required: bool
) -> str:
    actual = _sha256(path)
    if expected:
        wanted = _strip_sha256(expected)
        if actual != wanted:
            raise RuntimeError(
                f"{role}_sha256_mismatch:expected={wanted}:actual={actual}:{path}"
            )
    elif required:
        raise RuntimeError(f"{role}_expected_sha256_missing:{path}")
    return actual


def _normalised_model_id(value: str) -> str:
    return "yolov7_paper" if value in {"", "yolov7", "yolov7_paper"} else value


def _validate_quality_binding(
    *,
    binding: Mapping[str, Any],
    model_id: str,
    case_id: str,
    setup_id: str,
    precision: str,
    claim_mode: bool,
) -> str:
    if binding.get("schema") != "onnx-splitpoint/native-split-quality-binding":
        raise RuntimeError("quality_binding_schema_mismatch")
    identities = [
        value
        for value in (
            binding.get("preselection"),
            binding.get("central_quality_selection"),
            binding.get("boundary_metadata_payload"),
        )
        if isinstance(value, Mapping)
    ]
    if not identities:
        raise RuntimeError("quality_binding_identity_missing")
    resolved_setup = ""
    for identity in identities:
        candidate_model = str(identity.get("model_id") or "")
        candidate_case = str(identity.get("case_id") or "")
        candidate_setup = str(identity.get("setup_id") or "")
        candidate_precision = str(identity.get("precision") or "")
        if candidate_model and _normalised_model_id(candidate_model) != model_id:
            raise RuntimeError(
                f"quality_binding_model_mismatch:{candidate_model}:{model_id}"
            )
        if candidate_case and candidate_case != case_id:
            raise RuntimeError(f"quality_binding_case_mismatch:{candidate_case}:{case_id}")
        if candidate_setup:
            if resolved_setup and resolved_setup != candidate_setup:
                raise RuntimeError("quality_binding_setup_conflict")
            resolved_setup = candidate_setup
            if setup_id and candidate_setup != setup_id:
                raise RuntimeError(
                    f"quality_binding_setup_mismatch:{candidate_setup}:{setup_id}"
                )
        if candidate_precision and candidate_precision != precision:
            raise RuntimeError(
                f"quality_binding_precision_mismatch:{candidate_precision}:{precision}"
            )
    if claim_mode and not setup_id:
        raise RuntimeError("claim_gate_setup_id_required")
    if claim_mode and binding.get("quality_completed") is not True:
        raise RuntimeError("claim_gate_quality_binding_not_completed")
    return setup_id or resolved_setup


def _validate_dataset_manifest(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], str]:
    payload, actual_sha = _require_regular_json(path, role="dataset_manifest")
    if payload.get("schema") != "onnx-splitpoint/dataset-manifest":
        raise RuntimeError("dataset_manifest_schema_mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or int(payload.get("item_count") or -1) != len(items):
        raise RuntimeError("dataset_manifest_item_count_mismatch")
    if payload.get("dataset_id") != "coco2017-val" or len(items) != 5000:
        raise RuntimeError("dataset_manifest_not_frozen_coco2017_val")
    _verify_expected_file_sha256(
        path,
        role="dataset_manifest",
        expected=expected_sha256,
        required=True,
    )
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise RuntimeError("dataset_manifest_item_not_object")
        image_id = int(item.get("image_id") or -1)
        if image_id < 0 or image_id in seen:
            raise RuntimeError(f"dataset_manifest_image_id_invalid:{image_id}")
        seen.add(image_id)
        relative = Path(str(item.get("relative_path") or ""))
        if not relative.name or relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"dataset_manifest_relative_path_invalid:{relative}")
        _strip_sha256(item.get("sha256"))
        if int(item.get("size_bytes") or 0) <= 0:
            raise RuntimeError(f"dataset_manifest_size_invalid:{image_id}")
    return payload, actual_sha


def _validate_reference_report(
    path: Path,
    *,
    expected_sha256: str,
    dataset_manifest_sha256: str,
    expected_item_count: int,
) -> tuple[dict[str, Any], str]:
    payload, actual_sha = _require_regular_json(path, role="reference_report")
    _verify_expected_file_sha256(
        path,
        role="reference_report",
        expected=expected_sha256,
        required=True,
    )
    if payload.get("schema") != "onnx-splitpoint/yolov7-multi-image-fast-decode-parity-canary":
        raise RuntimeError("reference_report_schema_mismatch")
    if not _normalise_status(payload.get("status")).startswith("PASS"):
        raise RuntimeError(f"reference_report_status_not_pass:{payload.get('status')}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_item_count:
        raise RuntimeError(
            f"reference_report_item_count_mismatch:{len(rows) if isinstance(rows, list) else -1}:{expected_item_count}"
        )
    corpus = payload.get("corpus") or {}
    if not isinstance(corpus, Mapping):
        raise RuntimeError("reference_report_corpus_missing")
    bound_dataset_sha = str(corpus.get("dataset_manifest_sha256") or "")
    if not bound_dataset_sha:
        raise RuntimeError("reference_report_dataset_manifest_sha256_missing")
    if _strip_sha256(bound_dataset_sha) != dataset_manifest_sha256:
        raise RuntimeError("reference_report_dataset_manifest_sha256_mismatch")
    return payload, actual_sha


def _validate_corpus_manifest(
    path: Path,
    *,
    dataset: Mapping[str, Any],
    dataset_manifest_sha256: str,
    reference: Mapping[str, Any],
    expected_item_count: int,
) -> tuple[dict[str, Any], str]:
    payload, actual_sha = _require_regular_json(path, role="corpus_manifest")
    items = payload.get("items")
    requested = int(payload.get("requested_count") or -1)
    selected = int(payload.get("selected_count") or -1)
    if (
        not isinstance(items, list)
        or requested != expected_item_count
        or selected != expected_item_count
        or len(items) != expected_item_count
    ):
        raise RuntimeError(
            f"corpus_manifest_item_count_mismatch:{len(items) if isinstance(items, list) else -1}:{expected_item_count}"
        )
    expected_schema = (
        "onnx-splitpoint/yolov7-fast-decode-parity-corpus"
        if expected_item_count == CLAIM_ITEM_COUNT
        else "onnx-splitpoint/yolov7-product-smoke-corpus"
    )
    if payload.get("schema") != expected_schema:
        raise RuntimeError("corpus_manifest_schema_mismatch")
    bound_dataset_sha = str(payload.get("dataset_manifest_sha256") or "")
    if not bound_dataset_sha or _strip_sha256(bound_dataset_sha) != dataset_manifest_sha256:
        raise RuntimeError("corpus_manifest_dataset_manifest_sha256_mismatch")

    dataset_by_id = {
        int(item["image_id"]): item
        for item in list(dataset.get("items") or [])
        if isinstance(item, Mapping)
    }
    reference_rows = list(reference.get("rows") or [])
    reference_ids = [int(row.get("image_id") or -1) for row in reference_rows]
    corpus_ids: list[int] = []
    manifest_root = path.parent.resolve()
    for index, item in enumerate(items):
        if (
            not isinstance(item, Mapping)
            or "index" not in item
            or int(item["index"]) != index
        ):
            raise RuntimeError(f"corpus_manifest_order_invalid:{index}")
        image_id = int(item.get("image_id") or -1)
        corpus_ids.append(image_id)
        dataset_item = dataset_by_id.get(image_id)
        if dataset_item is None:
            raise RuntimeError(f"corpus_manifest_unknown_image_id:{image_id}")
        if str(item.get("relative_path") or "") != str(dataset_item.get("relative_path") or ""):
            raise RuntimeError(f"corpus_manifest_relative_path_mismatch:{image_id}")
        expected_source_sha = _strip_sha256(dataset_item.get("sha256"))
        staged_sha = _strip_sha256(item.get("staged_sha256") or item.get("sha256"))
        if staged_sha != expected_source_sha:
            raise RuntimeError(f"corpus_manifest_source_sha256_mismatch:{image_id}")
        staged_relative = Path(str(item.get("staged_file") or ""))
        if not staged_relative.name or staged_relative.is_absolute() or ".." in staged_relative.parts:
            raise RuntimeError(f"corpus_manifest_staged_path_invalid:{image_id}")
        staged_path = (manifest_root / staged_relative).resolve()
        try:
            staged_path.relative_to(manifest_root)
        except ValueError as exc:
            raise RuntimeError(f"corpus_manifest_staged_path_escape:{image_id}") from exc
        if not staged_path.is_file() or staged_path.is_symlink():
            raise RuntimeError(f"corpus_manifest_staged_file_missing:{staged_path}")
        if _sha256(staged_path) != staged_sha:
            raise RuntimeError(f"corpus_manifest_staged_sha256_mismatch:{image_id}")
    if corpus_ids != reference_ids:
        raise RuntimeError("corpus_manifest_reference_order_mismatch")
    return payload, actual_sha


def _stage_claim_corpus(
    *,
    resource: Path,
    dataset_manifest: Path,
    corpus_dir: Path,
    images_root: str,
    annotations: str,
    work_dir: Path,
) -> Path:
    """Run the versioned v1 stager with an explicit, fail-closed CLI."""
    script = resource / "stage_corpus.py"
    if not script.is_file():
        raise FileNotFoundError(f"stage_corpus_runtime_missing:{script}")
    command = [
        sys.executable,
        str(script),
        "--dataset-manifest", str(dataset_manifest),
        "--out", str(corpus_dir),
        "--count", str(CLAIM_ITEM_COUNT),
    ]
    if images_root:
        command.extend(["--images-root", str(Path(images_root).expanduser().resolve())])
    if annotations:
        command.extend(["--annotations", str(Path(annotations).expanduser().resolve())])
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    console = work_dir / "stage_corpus_console.log"
    console.write_text(completed.stdout or "", encoding="utf-8")
    _write_json(
        work_dir / "stage_corpus_command_receipt.json",
        {
            "adapter": "yolov7_stage_corpus_v1_explicit",
            "argv": command,
            "dataset_manifest": str(dataset_manifest),
            "corpus_manifest": str(corpus_dir / "corpus_manifest.json"),
            "expected_item_count": CLAIM_ITEM_COUNT,
            "returncode": int(completed.returncode),
            "console": str(console),
            "console_sha256": _sha256(console),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(f"stage_corpus_rc:{completed.returncode}:{console}")
    manifest = corpus_dir / "corpus_manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"stage_corpus_manifest_missing:{manifest}")
    return manifest



def _prepare_one_image_smoke_corpus(
    *,
    resource: Path,
    image: Path,
    corpus_dir: Path,
    work_dir: Path,
    dataset_manifest: Path | None = None,
    dataset_manifest_sha256: str = "",
    reference_source: Path | None = None,
) -> tuple[Path, Path]:
    """Create a one-image sentinel corpus and matching frozen reference subset.

    The full 32-image architecture canary has already passed independently.  The
    normal-runner hardware smoke only needs to prove that the product dispatch,
    concurrent P1/P2/Post pipeline, and out-of-timing oracle execute together.
    """
    dataset_manifest = dataset_manifest or (
        resource / "reference" / "dataset_detection_validation.json"
    )
    dataset_manifest_sha256 = dataset_manifest_sha256 or _sha256(dataset_manifest)
    dataset = _load_json(dataset_manifest)
    dataset_items = {
        int(item["image_id"]): item
        for item in list(dataset.get("items") or [])
        if isinstance(item, Mapping)
    }
    reference_source = reference_source or (
        resource / "reference" / "multi_image_fast_decode_canary_report.json"
    )
    reference = _load_json(reference_source)
    rows = list(reference.get("rows") or [])
    try:
        image_id = int(image.stem)
    except ValueError as exc:
        raise RuntimeError(f"sentinel_image_name_not_numeric:{image.name}") from exc
    dataset_item = dataset_items.get(image_id)
    if dataset_item is None:
        raise RuntimeError(f"sentinel_not_in_dataset_manifest:{image_id}")
    image_sha = _sha256(image)
    if image_sha != _strip_sha256(dataset_item.get("sha256")):
        raise RuntimeError(f"sentinel_dataset_sha256_mismatch:{image_id}")
    if image.stat().st_size != int(dataset_item.get("size_bytes") or -1):
        raise RuntimeError(f"sentinel_dataset_size_mismatch:{image_id}")
    matches = [row for row in rows if int(row.get("image_id") or -1) == image_id]
    if len(matches) != 1:
        raise RuntimeError(f"sentinel_reference_row_count:{image_id}:{len(matches)}")
    row = dict(matches[0])

    images_dir = corpus_dir / "images"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    target = images_dir / f"00_{image.name}"
    shutil.copy2(image, target)
    staged_sha = _sha256(target)
    item = {
        "index": 0,
        "image_id": image_id,
        "relative_path": str(dataset_item["relative_path"]),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "aspect_ratio": float(row.get("aspect_ratio") or (int(row["width"]) / float(int(row["height"])) )),
        "annotation_count": int(row.get("annotation_count") or 0),
        "noncrowd_annotation_count": int(row.get("noncrowd_annotation_count") or 0),
        "category_count": int(row.get("category_count") or 0),
        "selection_bin": "product_smoke_sentinel",
        "selection_reason": "exact_prevalidated_b066_sentinel",
        "staged_file": f"images/{target.name}",
        "staged_sha256": staged_sha,
        "sha256": staged_sha,
        "size_bytes": target.stat().st_size,
    }
    manifest = corpus_dir / "corpus_manifest.json"
    _write_json(
        manifest,
        {
            "schema": "onnx-splitpoint/yolov7-product-smoke-corpus",
            "schema_version": 1,
            "selection_scope": "exact_prevalidated_sentinel_only",
            "selection_uses_model_predictions": False,
            "claim_eligible": False,
            "execution_scope": DIAGNOSTIC_SCOPE,
            "dataset_manifest": str(dataset_manifest.resolve()),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_id": dataset.get("dataset_id"),
            "requested_count": 1,
            "selected_count": 1,
            "items": [item],
        },
    )
    subset = work_dir / "reference_report_subset.json"
    subset_payload = dict(reference)
    subset_payload["scope"] = "product_smoke_single_prevalidated_sentinel"
    subset_payload["rows"] = [row]
    subset_payload["corpus"] = {
        **dict(reference.get("corpus") or {}),
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "requested_count": 1,
        "selected_count": 1,
        "items": [item],
    }
    _write_json(subset, subset_payload)
    return manifest, subset


def build_three_stage_command(
    inv: ThreeStageInvocation,
    *,
    script: Path,
    tool_root: Path,
    expected_artifacts: Path,
    repetitions: int,
    frames: int,
    warmup: int,
    p1_queue_depth: int,
    post_queue_depth: int,
    device_id: str = "",
) -> list[str]:
    """Build the explicit CLI for the vendored YOLOv7 canary v1 parser."""
    command = [
        sys.executable,
        str(script),
        "--tool-root", str(tool_root),
        "--corpus", str(inv.corpus_manifest),
        "--expected", str(expected_artifacts),
        "--reference-report", str(inv.reference_report),
        "--out-root", str(inv.out_root),
        "--expected-corpus-count", str(inv.expected_item_count),
        "--repetitions", str(repetitions),
        "--frames", str(frames),
        "--warmup", str(warmup),
        "--p1-queue-depth", str(p1_queue_depth),
        "--post-queue-depth", str(post_queue_depth),
    ]
    if device_id:
        command.extend(["--device-id", device_id])
    return command


def _run_vendored_canary_explicit(
    *,
    inv: ThreeStageInvocation,
    script: Path,
    tool_root: Path,
    expected_artifacts: Path,
    repetitions: int,
    frames: int,
    warmup: int,
    p1_queue_depth: int,
    post_queue_depth: int,
    timeout_s: float,
    device_id: str = "",
) -> ThreeStageChildResult:
    inv.out_root.mkdir(parents=True, exist_ok=True)
    command = build_three_stage_command(
        inv,
        script=script,
        tool_root=tool_root,
        expected_artifacts=expected_artifacts,
        repetitions=repetitions,
        frames=frames,
        warmup=warmup,
        p1_queue_depth=p1_queue_depth,
        post_queue_depth=post_queue_depth,
        device_id=device_id,
    )
    invocation_payload = _invocation_receipt(inv)
    _write_json(inv.out_root / "three_stage_invocation_receipt.json", invocation_payload)
    _write_json(
        inv.out_root / "three_stage_remote_command_receipt.json",
        {
            "schema": "onnx-splitpoint/three-stage-remote-command-receipt",
            "schema_version": 1,
            "adapter": "yolov7_three_stage_canary_v1_explicit",
            "argv": command,
            "invocation_sha256": invocation_payload["invocation_sha256"],
            "out_root": str(inv.out_root),
            "out_root_binding_sha256": inv.out_root_binding_sha256,
        },
    )
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(tool_root) + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
        timeout=timeout_s if timeout_s > 0 else None,
    )
    console = inv.out_root / "vendored_canary_console.log"
    console.write_text(completed.stdout or "", encoding="utf-8")
    if completed.stdout:
        print(completed.stdout, end="")
    return ThreeStageChildResult(
        returncode=int(completed.returncode),
        console=console,
        command=tuple(command),
    )


def _v2793_stage_timings(report: Mapping[str, Any]) -> dict[str, Any]:
    """Project P1/P2/Post timing evidence from the real canary report."""
    repetitions = list(report.get("repetitions") or [])
    rows: list[dict[str, Any]] = []
    for item in repetitions:
        if not isinstance(item, Mapping):
            continue
        metrics = item.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            continue
        rows.append({
            "repetition": item.get("repetition"),
            "p1_stage_mean_ms": metrics.get("p1_stage_mean_ms"),
            "p2_stage_mean_ms": metrics.get("p2_stage_mean_ms"),
            "post_mean_ms": metrics.get("post_mean_ms"),
            "post_median_ms": metrics.get("post_median_ms"),
            "post_p95_ms": metrics.get("post_p95_ms"),
            "post_max_ms": metrics.get("post_max_ms"),
            "raw_fps": metrics.get("raw_fps"),
            "completed_fps": metrics.get("completed_fps"),
            "completed_raw_ratio": metrics.get("completed_raw_ratio"),
            "theoretical_raw_fps": metrics.get("theoretical_raw_fps"),
            "theoretical_three_stage_fps": metrics.get("theoretical_three_stage_fps"),
            "callback_failures": metrics.get("callback_failures"),
        })
    aggregate = report.get("aggregate") or {}
    measurement = report.get("measurement_contract") or {}
    return {
        "schema": "onnx-splitpoint/native-three-stage-timing-projection",
        "schema_version": 1,
        "P1": {
            "semantics": measurement.get("P1"),
            "mean_ms_by_repetition": [row.get("p1_stage_mean_ms") for row in rows],
        },
        "P2": {
            "semantics": measurement.get("P2"),
            "mean_ms_by_repetition": [row.get("p2_stage_mean_ms") for row in rows],
        },
        "Post": {
            "semantics": measurement.get("postprocessing"),
            "location": measurement.get("postprocess_location"),
            "mean_ms_by_repetition": [row.get("post_mean_ms") for row in rows],
            "median_ms_by_repetition": [row.get("post_median_ms") for row in rows],
            "p95_ms_by_repetition": [row.get("post_p95_ms") for row in rows],
            "max_ms_by_repetition": [row.get("post_max_ms") for row in rows],
            "aggregate_p95_guard_ms": aggregate.get("post_p95_guard_ms"),
            "aggregate_max_guard_ms": aggregate.get("post_max_guard_ms"),
        },
        "repetitions": rows,
        "aggregate": aggregate,
    }


def _v2793_oracle_parity(report: Mapping[str, Any]) -> dict[str, Any]:
    postflight = report.get("postflight_quality_oracle") or {}
    repetitions = []
    for item in list(report.get("repetitions") or []):
        if not isinstance(item, Mapping):
            continue
        parity = item.get("measurement_result_parity") or {}
        if isinstance(parity, Mapping):
            repetitions.append({
                "repetition": item.get("repetition"),
                "requested_images": parity.get("requested_images"),
                "exact_images": parity.get("exact_images"),
                "all_exact": parity.get("all_exact"),
            })
    return {
        "schema": "onnx-splitpoint/native-three-stage-oracle-parity",
        "schema_version": 1,
        "status": "passed" if bool(postflight.get("all_exact")) else "failed",
        "inside_performance_timing": False,
        "postflight": dict(postflight) if isinstance(postflight, Mapping) else {},
        "measurement_repetitions": repetitions,
    }

def _project_result(
    *,
    report: Mapping[str, Any],
    args: argparse.Namespace,
    binding: Mapping[str, Any],
    report_path: Path,
    resource: Path,
    invocation: ThreeStageInvocation | None = None,
    child_returncode: int = 0,
    child_console: Path | None = None,
) -> dict[str, Any]:
    status = _normalise_status(report.get("status"))
    p2_fps = _pick_number(
        report,
        "raw_fps_median",
        "p2_output_fps",
        "raw_model_outputs_fps",
        "fps_raw_median",
    )
    completed_fps = _pick_number(
        report,
        "completed_fps_median",
        "completed_detection_fps",
        "application_throughput_fps",
        "fps_completed_median",
    )
    ratio = _pick_number(
        report,
        "completed_raw_ratio_median",
        "completed_to_p2_ratio",
        "completed_raw_ratio",
    )
    if ratio is None and p2_fps and completed_fps:
        ratio = completed_fps / p2_fps

    latency_repeats = [{"repetition_id": f"three_stage:{r.get('repetition')}",
                        "request_latency": (r.get("metrics") or {}).get("request_latency") or {}}
                       for r in report.get("repetitions", [])]
    stage_timings = _v2793_stage_timings(report)
    parity = _v2793_oracle_parity(report)
    parity_passed = str(parity.get("status") or "").strip().lower() == "passed"
    direct_measured = bool(p2_fps and completed_fps)
    ok = (
        child_returncode == 0
        and status.startswith("PASS")
        and bool(p2_fps)
        and bool(completed_fps)
        and parity_passed
    )
    child_failure_class, child_failure_reason = _child_failure_details(report)
    if child_returncode != 0 and not child_failure_class:
        child_failure_class = "ThreeStageChildProcessError"
    if child_returncode != 0 and not child_failure_reason:
        child_failure_reason = f"three_stage_child_returncode:{child_returncode}"
    contract = (
        binding.get("output_contract")
        or binding.get("endpoint_contract")
        or {}
    )
    p2_family = str(
        contract.get("family")
        or binding.get("p2_output_contract_family")
        or "yolov7_anchor_multiscale_raw"
    )
    return {
        "ok": ok,
        "schema": SCHEMA,
        "schema_version": 2,
        "mode": "native_hailort_tensorrt_concurrent_three_stage_fifo",
        "build_id": BUILD_ID,
        "performance_endpoint": "p2_output",
        "application_performance_endpoint": "completed_detection",
        "endpoint_execution_policy": "concurrent_three_stage_single_invocation",
        "repetition_records": latency_repeats,
        "repetition_count_requested": (report.get("measurement_contract") or {}).get("repetitions", len(latency_repeats)),
        "three_stage_concurrency_directly_measured": direct_measured,
        "three_stage_hardware_integration_status": (
            "directly_measured" if direct_measured else "failed_before_direct_measurement"
        ),
        "p2_output_contract_family": p2_family,
        "postprocess_adapter_id": "yolov7_anchor_multiscale_sparse",
        "postprocess_location": "host_cpu_numpy_inprocess_callback",
        "quality_oracle_location": "outside_performance_timing",
        "quality_oracle_status": "passed" if parity_passed else "failed",
        "p2_output_fps": p2_fps,
        "completed_detection_fps": completed_fps,
        "completed_to_p2_ratio": ratio,
        "throughput_primary_fps": p2_fps,
        "application_throughput_fps": completed_fps,
        "endpoint_relation_verified": bool(ok),
        "directly_measured": direct_measured,
        "execution_scope": invocation.execution_scope if invocation else DIAGNOSTIC_SCOPE,
        "claim_eligible": bool(invocation and invocation.claim_eligible and ok),
        "expected_item_count": (
            invocation.expected_item_count if invocation else DIAGNOSTIC_ITEM_COUNT
        ),
        "child_returncode": child_returncode,
        "child_failure_class": child_failure_class,
        "child_failure_reason": child_failure_reason,
        "vendored_runtime_rc": child_returncode,
        "vendored_runtime_console": str(child_console) if child_console else "",
        "stage_timings": dict(stage_timings or {}),
        "oracle_parity": dict(parity or {}),
        "canary_status": report.get("status"),
        "canary_report": str(report_path),
        "resource_runtime": str(resource),
        "out_root": str(invocation.out_root) if invocation else str(report_path.parent),
        "out_root_binding_sha256": (
            invocation.out_root_binding_sha256 if invocation else ""
        ),
        "invocation": _invocation_receipt(invocation) if invocation else {},
        "identity": {
            "model_id": invocation.model_id if invocation else args.model_id,
            "case_id": args.case,
            "setup_id": invocation.setup_id if invocation else args.setup_id,
            "eval_run_id": args.eval_run_id,
            "source_run_id": args.source_run_id,
            "precision": args.precision,
            "binding": str(args.native_split_quality_binding),
            "binding_sha256": (
                invocation.quality_binding_sha256
                if invocation
                else _sha256(Path(args.native_split_quality_binding))
            ),
        },
        "phases": {
            "concurrent_three_stage": {
                "directly_measured": direct_measured,
                "report": str(report_path),
            },
            "quality_oracle": {
                "inside_performance_timing": False,
                "status": "passed" if parity_passed else "failed",
            },
        },
        "failure_class": "" if ok else (
            child_failure_class or "ThreeStageContractFailure"
        ),
        "failure_reason": "" if ok else (
            child_failure_reason
            or (
                f"oracle_parity_status:{parity.get('status')}"
                if not parity_passed
                else f"three_stage_status:{report.get('status')}"
            )
        ),
    }


def _build_three_stage_invocation(
    *,
    args: argparse.Namespace,
    resource: Path,
    binding_path: Path,
    binding: Mapping[str, Any],
    image: Path,
    work_dir: Path,
) -> ThreeStageInvocation:
    claim_mode = args.three_stage_scope == CLAIM_SCOPE
    benchmark_set = Path(args.benchmark_set).expanduser().resolve()
    if not benchmark_set.exists():
        raise FileNotFoundError(f"benchmark_set_missing:{benchmark_set}")

    model_id = _normalised_model_id(args.model_id)
    setup_id = _validate_quality_binding(
        binding=binding,
        model_id=model_id,
        case_id=args.case,
        setup_id=args.setup_id,
        precision=args.precision,
        claim_mode=claim_mode,
    )
    _, quality_binding_sha = _require_regular_json(
        binding_path, role="quality_binding"
    )
    _verify_expected_file_sha256(
        binding_path,
        role="quality_binding",
        expected=args.expected_quality_binding_sha256,
        required=claim_mode,
    )

    bundled_dataset = resource / "reference" / "dataset_detection_validation.json"
    dataset_manifest = (
        Path(args.dataset_manifest).expanduser().resolve()
        if args.dataset_manifest
        else bundled_dataset.resolve()
    )
    dataset_expected_sha = str(args.expected_dataset_manifest_sha256 or "")
    if not dataset_expected_sha and dataset_manifest == bundled_dataset.resolve():
        dataset_expected_sha = DATASET_MANIFEST_SHA256
    dataset, dataset_sha = _validate_dataset_manifest(
        dataset_manifest, expected_sha256=dataset_expected_sha
    )

    bundled_reference = (
        resource / "reference" / "multi_image_fast_decode_canary_report.json"
    ).resolve()
    reference_source = (
        Path(args.reference_report).expanduser().resolve()
        if args.reference_report
        else bundled_reference
    )
    reference_expected_sha = str(args.expected_reference_report_sha256 or "")
    if not reference_expected_sha and reference_source == bundled_reference:
        reference_expected_sha = REFERENCE_REPORT_SHA256

    corpus_dir = work_dir / "corpus"
    if claim_mode:
        if not args.three_stage_out_root:
            raise RuntimeError("claim_gate_explicit_three_stage_out_root_required")
        expected_item_count = CLAIM_ITEM_COUNT
        reference, reference_sha = _validate_reference_report(
            reference_source,
            expected_sha256=reference_expected_sha,
            dataset_manifest_sha256=dataset_sha,
            expected_item_count=expected_item_count,
        )
        corpus_manifest = _stage_claim_corpus(
            resource=resource,
            dataset_manifest=dataset_manifest,
            corpus_dir=corpus_dir,
            images_root=args.images_root,
            annotations=args.annotations,
            work_dir=work_dir,
        )
        reference_report = reference_source
    else:
        expected_item_count = DIAGNOSTIC_ITEM_COUNT
        # Verify the immutable 32-item reference before deriving the exact
        # single-sentinel non-claim subset consumed by the diagnostic canary.
        _validate_reference_report(
            reference_source,
            expected_sha256=reference_expected_sha,
            dataset_manifest_sha256=dataset_sha,
            expected_item_count=CLAIM_ITEM_COUNT,
        )
        corpus_manifest, reference_report = _prepare_one_image_smoke_corpus(
            resource=resource,
            image=image,
            corpus_dir=corpus_dir,
            work_dir=work_dir,
            dataset_manifest=dataset_manifest,
            dataset_manifest_sha256=dataset_sha,
            reference_source=reference_source,
        )
        reference, reference_sha = _validate_reference_report(
            reference_report,
            expected_sha256=_sha256(reference_report),
            dataset_manifest_sha256=dataset_sha,
            expected_item_count=expected_item_count,
        )

    _, corpus_sha = _validate_corpus_manifest(
        corpus_manifest,
        dataset=dataset,
        dataset_manifest_sha256=dataset_sha,
        reference=reference,
        expected_item_count=expected_item_count,
    )
    if args.expected_corpus_manifest_sha256:
        _verify_expected_file_sha256(
            corpus_manifest,
            role="corpus_manifest",
            expected=args.expected_corpus_manifest_sha256,
            required=False,
        )

    out_root_raw = (
        Path(args.three_stage_out_root)
        if args.three_stage_out_root
        else work_dir / "three_stage_result"
    )
    out_root = _prepare_out_root(out_root_raw)
    out_binding_sha = _out_root_binding_sha256(
        out_root,
        execution_scope=args.three_stage_scope,
        model_id=model_id,
        case_id=args.case,
        setup_id=setup_id,
    )
    return ThreeStageInvocation(
        benchmark_set=benchmark_set,
        case_id=args.case,
        setup_id=setup_id,
        corpus_manifest=corpus_manifest.resolve(),
        dataset_manifest=dataset_manifest,
        reference_report=reference_report.resolve(),
        quality_binding=binding_path,
        out_root=out_root,
        expected_item_count=expected_item_count,
        model_id=model_id,
        precision=args.precision,
        execution_scope=args.three_stage_scope,
        claim_eligible=claim_mode,
        corpus_manifest_sha256=corpus_sha,
        dataset_manifest_sha256=dataset_sha,
        reference_report_sha256=reference_sha,
        quality_binding_sha256=quality_binding_sha,
        out_root_binding_sha256=out_binding_sha,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-set", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--hw-arch", required=True)
    p.add_argument("--precision", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--queue-depth", type=int, default=3)
    p.add_argument("--post-queue-depth", type=int, default=4)
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--setup-id", default="")
    p.add_argument("--eval-run-id", default="")
    p.add_argument("--source-run-id", default="")
    p.add_argument("--model-id", default="")
    p.add_argument("--native-split-quality-binding", required=True)
    p.add_argument(
        "--three-stage-scope",
        choices=(DIAGNOSTIC_SCOPE, CLAIM_SCOPE),
        default=DIAGNOSTIC_SCOPE,
        help="One-image non-claim diagnostic or strict 32-item claim gate.",
    )
    p.add_argument("--dataset-manifest", default="")
    p.add_argument("--expected-dataset-manifest-sha256", default="")
    p.add_argument("--images-root", default="")
    p.add_argument("--annotations", default="")
    p.add_argument("--reference-report", default="")
    p.add_argument("--expected-reference-report-sha256", default="")
    p.add_argument("--expected-quality-binding-sha256", default="")
    p.add_argument("--expected-corpus-manifest-sha256", default="")
    p.add_argument("--three-stage-out-root", default="")
    p.add_argument("--expected-image-sha256", default="")
    p.add_argument("--expected-hef-sha256", default="")
    p.add_argument("--expected-engine-sha256", default="")
    p.add_argument("--expected-boundary-layout", default="")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--result-json", required=True)
    p.add_argument("--timeout-s", type=float, default=3600.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    result_json = Path(args.result_json).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    invocation: ThreeStageInvocation | None = None
    child: ThreeStageChildResult | None = None
    report_path: Path | None = None
    failure_receipt: Path | None = None
    try:
        if args.hw_arch != "hailo8":
            raise RuntimeError("concurrent_three_stage_currently_requires_hailo8")
        if args.case != "b066":
            raise RuntimeError("concurrent_three_stage_yolov7_reference_requires_b066")
        if args.model_id not in {"", "yolov7", "yolov7_paper"}:
            raise RuntimeError("concurrent_three_stage_yolov7_model_identity_mismatch")
        if args.precision != "uint8_dequant_fp16":
            raise RuntimeError("concurrent_three_stage_requires_uint8_dequant_fp16")

        _add_extra_sites()
        import numpy as np  # noqa: F401
        import tensorrt as trt  # noqa: F401
        import hailo_platform  # noqa: F401

        binding_path = Path(args.native_split_quality_binding).expanduser().resolve()
        binding, _ = _require_regular_json(binding_path, role="quality_binding")
        hef = _artifact_path(binding, "part1_runtime")
        engine = _artifact_path(binding, "engine")
        image = Path(args.image).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"input_image_missing:{image}")

        for actual, expected, role in (
            (_sha256(image), args.expected_image_sha256, "image"),
            (_sha256(hef), args.expected_hef_sha256, "hef"),
            (_sha256(engine), args.expected_engine_sha256, "engine"),
        ):
            expected_digest = str(expected)
            if expected_digest.startswith("sha256:"):
                expected_digest = expected_digest[7:]
            if expected and actual != expected_digest:
                raise RuntimeError(f"{role}_sha256_mismatch:{actual}")

        tool_root = Path(__file__).resolve().parents[1]
        resource = (
            tool_root
            / "onnx_splitpoint_tool"
            / "resources"
            / "native_concurrent_three_stage_yolov7"
        )
        canary_script = resource / "three_stage_canary.py"
        if not canary_script.is_file():
            raise FileNotFoundError(f"vendored_three_stage_runtime_missing:{canary_script}")

        invocation = _build_three_stage_invocation(
            args=args,
            resource=resource,
            binding_path=binding_path,
            binding=binding,
            image=image,
            work_dir=work_dir,
        )
        source_snapshot_prefix = _source_snapshot_prefix(tool_root)
        source_zip = _create_source_snapshot(tool_root, work_dir / "source_snapshot.zip")

        child = _run_vendored_canary_explicit(
            inv=invocation,
            script=canary_script,
            tool_root=tool_root,
            expected_artifacts=resource / "expected_artifacts.json",
            repetitions=args.repetitions,
            frames=args.frames,
            warmup=args.warmup,
            p1_queue_depth=args.queue_depth,
            post_queue_depth=args.post_queue_depth,
            timeout_s=args.timeout_s,
        )
        report_path = _find_report_only_below(invocation.out_root)
        if report_path is None:
            raise RuntimeError(
                f"three_stage_report_missing_after_rc:{child.returncode}:out_root={invocation.out_root}"
            )
        report = _load_json(report_path)
        if not isinstance(report, Mapping):
            raise RuntimeError(f"three_stage_report_root_not_object:{report_path}")
        result = _project_result(
            report=report,
            args=args,
            binding=binding,
            report_path=report_path,
            resource=resource,
            invocation=invocation,
            child_returncode=child.returncode,
            child_console=child.console,
        )
        result["product_execution_context"] = {
            "source_snapshot": str(source_zip),
            "source_snapshot_sha256": _sha256(source_zip),
            "source_snapshot_prefix": source_snapshot_prefix,
            "performance_corpus_manifest": str(invocation.corpus_manifest),
            "performance_corpus_manifest_sha256": invocation.corpus_manifest_sha256,
            "performance_corpus_count": invocation.expected_item_count,
            "dataset_manifest": str(invocation.dataset_manifest),
            "dataset_manifest_sha256": invocation.dataset_manifest_sha256,
            "postflight_reference_report": str(invocation.reference_report),
            "postflight_reference_report_sha256": invocation.reference_report_sha256,
            "quality_binding": str(invocation.quality_binding),
            "quality_binding_sha256": invocation.quality_binding_sha256,
            "canary_out_root": str(invocation.out_root),
            "canary_out_root_binding_sha256": invocation.out_root_binding_sha256,
            "execution_scope": invocation.execution_scope,
            "claim_eligible": bool(invocation.claim_eligible and result["ok"]),
            "quality_oracle_inside_performance_timing": False,
        }
        if result["ok"] and invocation.claim_eligible:
            result["three_stage_hardware_integration_status"] = "passed"
        _write_json(result_json, result)
        if not result["ok"]:
            failure_receipt = invocation.out_root / "three_stage_failure_receipt.json"
            _write_json(
                failure_receipt,
                {
                    "schema": "onnx-splitpoint/three-stage-failure-receipt",
                    "schema_version": 1,
                    "out_root": str(invocation.out_root),
                    "out_root_binding_sha256": invocation.out_root_binding_sha256,
                    "invocation_sha256": _invocation_receipt(invocation)["invocation_sha256"],
                    "child_returncode": child.returncode,
                    "failure_class": result.get("failure_class"),
                    "failure_reason": result.get("failure_reason"),
                    "child_failure_class": result.get("child_failure_class"),
                    "child_failure_reason": result.get("child_failure_reason"),
                    "child_report": str(report_path),
                    "child_report_sha256": _sha256(report_path),
                    "child_console": str(child.console),
                    "child_console_sha256": _sha256(child.console),
                    "stderr_tail": child.console.read_text(
                        encoding="utf-8", errors="replace"
                    )[-4000:],
                },
            )
        _write_out_root_index(
            out_root=invocation.out_root,
            out_root_binding_sha256=invocation.out_root_binding_sha256,
            result_json=result_json,
            invocation=invocation,
            report_path=report_path,
            console=child.console,
            failure_receipt=failure_receipt,
        )
        return 0 if result["ok"] else 1
    except Exception as exc:
        fallback_out_root = (
            invocation.out_root
            if invocation is not None
            else Path(args.three_stage_out_root or (work_dir / "three_stage_result"))
            .expanduser()
            .resolve()
        )
        out_binding_sha = (
            invocation.out_root_binding_sha256
            if invocation is not None
            else _out_root_binding_sha256(
                fallback_out_root,
                execution_scope=args.three_stage_scope,
                model_id=_normalised_model_id(args.model_id),
                case_id=args.case,
                setup_id=args.setup_id,
            )
        )
        failure = {
            "ok": False,
            "schema": SCHEMA,
            "schema_version": 2,
            "mode": "native_hailort_tensorrt_concurrent_three_stage_fifo",
            "build_id": BUILD_ID,
            "performance_endpoint": "p2_output",
            "application_performance_endpoint": "completed_detection",
            "endpoint_execution_policy": "concurrent_three_stage_single_invocation",
            "three_stage_concurrency_directly_measured": False,
            "three_stage_hardware_integration_status": "failed_before_direct_measurement",
            "quality_oracle_location": "outside_performance_timing",
            "execution_scope": args.three_stage_scope,
            "claim_eligible": False,
            "out_root": str(fallback_out_root),
            "out_root_binding_sha256": out_binding_sha,
            "invocation": _invocation_receipt(invocation) if invocation else {},
            "child_returncode": child.returncode if child else None,
            "vendored_runtime_rc": child.returncode if child else None,
            "child_failure_class": type(exc).__name__,
            "child_failure_reason": str(exc),
            "failure_class": type(exc).__name__,
            "failure_reason": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
        _write_json(result_json, failure)
        try:
            fallback_out_root = _prepare_out_root(fallback_out_root)
            failure_receipt = fallback_out_root / "three_stage_failure_receipt.json"
            _write_json(
                failure_receipt,
                {
                    "schema": "onnx-splitpoint/three-stage-failure-receipt",
                    "schema_version": 1,
                    "out_root": str(fallback_out_root),
                    "out_root_binding_sha256": out_binding_sha,
                    "invocation_sha256": (
                        _invocation_receipt(invocation)["invocation_sha256"]
                        if invocation else ""
                    ),
                    "child_returncode": child.returncode if child else None,
                    "failure_class": type(exc).__name__,
                    "failure_reason": str(exc),
                    "child_report": str(report_path) if report_path else "",
                    "child_console": str(child.console) if child else "",
                    "stderr_tail": (
                        child.console.read_text(encoding="utf-8", errors="replace")[-4000:]
                        if child and child.console.is_file()
                        else ""
                    ),
                },
            )
            _write_out_root_index(
                out_root=fallback_out_root,
                out_root_binding_sha256=out_binding_sha,
                result_json=result_json,
                invocation=invocation,
                report_path=report_path,
                console=child.console if child else None,
                failure_receipt=failure_receipt,
            )
        except Exception as index_error:
            failure["artifact_index_error"] = f"{type(index_error).__name__}: {index_error}"
            _write_json(result_json, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
