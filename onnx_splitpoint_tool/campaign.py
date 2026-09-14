"""Final-campaign manifests, readiness checks, and immutable freeze bundles.

This module turns the dissertation's pre-registration requirements into tool
artefacts.  It deliberately does not claim that a dataset is final or that a
model is genuinely unseen merely because a YAML flag exists.  Instead, it
creates content-addressed manifests, verifies calibration/validation
separation, requires an explicit hold-out attestation, and packages the
prospective prediction freezes before measurements are interpreted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from . import __version__ as TOOL_PACKAGE_VERSION
from .protocol_freeze import (
    CONFIRMATORY_HOLDOUT_ROLE,
    create_protocol_freeze,
    is_confirmatory_holdout,
    normalize_evaluation_role,
    verify_configured_protocol_freeze,
    verify_protocol_freeze,
)
from .source_integrity import SOURCE_INTEGRITY_BINDING_SCHEMA
from .energy.config import resolve_effective_energy_config
from .preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)
from .workflow.artifacts import now_iso, sha256_file, sha256_json, write_csv, write_json, write_text

DATASET_MANIFEST_SCHEMA = "onnx-splitpoint/dataset-manifest"
PIPELINE_CONTRACT_SCHEMA = "onnx-splitpoint/pipeline-contract-manifest"
HOLDOUT_REGISTRY_SCHEMA = "onnx-splitpoint/holdout-registry"
CANDIDATE_UNIVERSE_SCHEMA = "onnx-splitpoint/candidate-universe-manifest"
CAMPAIGN_READINESS_SCHEMA = "onnx-splitpoint/campaign-readiness"
CAMPAIGN_FREEZE_SCHEMA = "onnx-splitpoint/campaign-freeze"
ENERGY_CALIBRATION_SCHEMA = "onnx-splitpoint/energy-calibration-manifest"
PREDICTION_FREEZE_APPROVAL_SCHEMA = "onnx-splitpoint/prediction-freeze-approval"
RANKING_MODEL_BUNDLE_SCHEMA = "onnx-splitpoint/ranking-model-bundle"
CAMPAIGN_SCHEMA_VERSION = 1

DIRECT_ENERGY_CALIBRATION_MODE = "direct_calibration"
INHERITED_VALIDATED_ENERGY_METHOD_MODE = "inherited_validated_method"
ENERGY_EVIDENCE_MODES = {
    DIRECT_ENERGY_CALIBRATION_MODE,
    INHERITED_VALIDATED_ENERGY_METHOD_MODE,
}

FINAL_MATRIX_SETUP_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "setup_id": "orin_nx_hailo8_01",
        "urecs_address": "192.168.0.197",
        "channel": 0,
        "sample_rate_hz": 2000,
        "scope": "FS",
        "measurement_point": "complete_system_input",
    },
    {
        "setup_id": "orin_nx_hailo10_01",
        "urecs_address": "192.168.0.176",
        "channel": 0,
        "sample_rate_hz": 2000,
        "scope": "FS",
        "measurement_point": "complete_system_input",
    },
    {
        "setup_id": "orin_nx_deepx_m1_01",
        "urecs_address": "192.168.0.185",
        "channel": 0,
        "sample_rate_hz": 2000,
        "scope": "FS",
        "measurement_point": "complete_system_input",
    },
)

WACHSMUTH_ENERGY_METHOD_REFERENCE = {
    "reference_id": "wachsmuth2026masterarbeit",
    "author": "Wachsmuth, Joris",
    "title": (
        "Entwicklung und Validierung eines Messsystems zur energetischen "
        "Bewertung eingebetteter KI-Beschleuniger"
    ),
    "institution": "Bielefeld University",
    "work_type": "Master's thesis",
    "internal_identifier": "M99",
    "year": 2026,
}

_PLACEHOLDER_PATTERN = re.compile(
    r"(?:\bTODO\b|REPLACE_WITH|replace_campaign_endpoint_adapter_placeholder)",
    flags=re.IGNORECASE,
)

EVALUATED_MATRIX_CLAIM_SCOPE = "evaluated_matrix"
RANKING_GENERALIZATION_CLAIM_SCOPE = "ranking_generalization"
CAMPAIGN_CLAIM_SCOPES = {
    EVALUATED_MATRIX_CLAIM_SCOPE,
    RANKING_GENERALIZATION_CLAIM_SCOPE,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _normalise_sha256(value: Any) -> str:
    """Return a lowercase bare SHA-256 digest for tolerant comparisons.

    Older artefacts mix ``sha256:<hex>`` and ``<hex>`` representations.  The
    digest is identical; only the serialization differs.  Normalising avoids
    expensive false-negative rechecks while preserving strict content checks.
    """
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


def _stable_sample_indices(count: int, sample_count: int) -> list[int]:
    """Deterministic, coverage-oriented sample indices for relaxed verification."""
    count = max(0, int(count))
    sample_count = max(0, int(sample_count))
    if count <= 0 or sample_count <= 0:
        return []
    if sample_count >= count:
        return list(range(count))
    if sample_count == 1:
        return [0]
    # Evenly cover the complete manifest rather than checking only its prefix.
    return sorted({round(i * (count - 1) / (sample_count - 1)) for i in range(sample_count)})


def _normalize_sha256(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        raw = raw.split(":", 1)[1]
    return raw


def _sha256_equal(expected: Any, actual: Any) -> bool:
    a = _normalize_sha256(expected)
    b = _normalize_sha256(actual)
    return bool(a and b and a == b)


def _load_structured(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8")
    try:
        if p.suffix.lower() == ".json":
            value = json.loads(text)
        elif yaml is not None:
            value = yaml.safe_load(text)
        else:
            value = json.loads(text)
    except Exception:
        return {}
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _placeholder_paths(value: Any, *, prefix: str = "") -> list[str]:
    """Return paths whose values still contain a shipped placeholder token."""

    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            findings.extend(_placeholder_paths(item, prefix=child))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]" if prefix else f"[{index}]"
            findings.extend(_placeholder_paths(item, prefix=child))
    elif isinstance(value, str) and _PLACEHOLDER_PATTERN.search(value):
        findings.append(prefix or "<root>")
    return findings


def _write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write campaign YAML files")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def _resolve_ref(value: Any, base_dir: str | Path | None = None) -> Path:
    p = Path(str(value or "")).expanduser()
    if not p.is_absolute() and base_dir:
        p = Path(base_dir) / p
    try:
        return p.resolve()
    except Exception:
        return p


def _file_entry(path: Path, root: Path, *, hash_mode: str = "content", sample_id: str = "") -> dict[str, Any]:
    try:
        rel = str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        rel = path.name
    entry: dict[str, Any] = {
        "sample_id": sample_id or rel,
        "relative_path": rel,
        "size_bytes": int(path.stat().st_size),
    }
    if hash_mode == "content":
        entry["sha256"] = sha256_file(path) or ""
    return entry


def _iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _select_dataset_items(
    items: Sequence[Mapping[str, Any]],
    *,
    max_items: int,
    strategy: str,
    seed: int,
    task: str,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in items if isinstance(row, Mapping)]
    if max_items <= 0 or max_items >= len(rows):
        return rows
    strategy_l = str(strategy or "deterministic_hash").strip().lower().replace("-", "_")

    def order_key(row: Mapping[str, Any]) -> str:
        identity = str(row.get("sample_id") or row.get("relative_path") or "")
        return hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).hexdigest()

    if strategy_l in {"class_stratified", "stratified"} and task == "classification":
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get("class_name") or ""), []).append(row)
        for class_rows in groups.values():
            class_rows.sort(key=order_key)
        selected: list[dict[str, Any]] = []
        class_names = sorted(groups)
        cursor = {name: 0 for name in class_names}
        while len(selected) < max_items:
            made_progress = False
            for name in class_names:
                idx = cursor[name]
                if idx >= len(groups[name]):
                    continue
                selected.append(groups[name][idx])
                cursor[name] += 1
                made_progress = True
                if len(selected) >= max_items:
                    break
            if not made_progress:
                break
        return selected
    if strategy_l in {"sorted", "sorted_first", "first"}:
        return rows[:max_items]
    # Default: deterministic, path/ID-based hash sampling.  It is independent
    # of predictions, labels and measured task quality.
    return sorted(rows, key=order_key)[:max_items]


def create_dataset_manifest(
    *,
    task: str,
    role: str,
    dataset_id: str,
    split: str,
    root: str | Path,
    output: str | Path,
    annotations: str | Path | None = None,
    labels: str | Path | None = None,
    hash_mode: str = "content",
    max_items: int = 0,
    selection_strategy: str = "deterministic_hash",
    selection_seed: int = 20260710,
) -> Path:
    """Create a content-addressed classification or COCO-style manifest."""
    task_l = str(task).strip().lower()
    role_l = str(role).strip().lower()
    if task_l not in {"classification", "detection"}:
        raise ValueError("task must be classification or detection")
    if role_l not in {"calibration", "validation", "screening"}:
        raise ValueError("role must be calibration, validation, or screening")
    if hash_mode not in {"content", "paths"}:
        raise ValueError("hash_mode must be content or paths")

    root_path = _resolve_ref(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root_path}")
    ann_path = _resolve_ref(annotations) if annotations else None
    labels_path = _resolve_ref(labels) if labels else None

    items: list[dict[str, Any]] = []
    source_kind = "directory_scan"
    if task_l == "detection" and ann_path and ann_path.is_file():
        source_kind = "coco_annotations"
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        images = list(payload.get("images") or []) if isinstance(payload, Mapping) else []
        for image in sorted((x for x in images if isinstance(x, Mapping)), key=lambda x: int(x.get("id") or 0)):
            file_name = str(image.get("file_name") or "").strip()
            if not file_name:
                continue
            path = root_path / file_name
            if not path.is_file():
                raise FileNotFoundError(f"COCO image referenced by annotations is missing: {path}")
            entry = _file_entry(path, root_path, hash_mode=hash_mode, sample_id=str(image.get("id") or file_name))
            entry.update({
                "image_id": image.get("id"),
                "width": image.get("width"),
                "height": image.get("height"),
            })
            items.append(entry)
    else:
        for path in _iter_images(root_path):
            entry = _file_entry(path, root_path, hash_mode=hash_mode)
            if task_l == "classification":
                entry["class_name"] = path.parent.name
            items.append(entry)

    if not items:
        raise ValueError(f"no dataset items found under {root_path}")
    population_count = len(items)
    items = _select_dataset_items(
        items,
        max_items=max_items,
        strategy=selection_strategy,
        seed=selection_seed,
        task=task_l,
    )
    identity_rows = [
        {
            "sample_id": row.get("sample_id"),
            "relative_path": row.get("relative_path"),
            # Stable across old/new serializers that use either sha256:<hex>
            # or a bare digest.
            "sha256": _normalize_sha256(row.get("sha256", "")),
            "class_name": row.get("class_name", ""),
        }
        for row in items
    ]
    manifest = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "created_at": now_iso(),
        "dataset_id": dataset_id,
        "task": task_l,
        "role": role_l,
        "split": split,
        "source_kind": source_kind,
        "root": str(root_path),
        "root_name": root_path.name,
        "hash_mode": hash_mode,
        "item_count": len(items),
        "population_count": population_count,
        "selection": {
            "strategy": ("all" if max_items <= 0 or max_items >= population_count else str(selection_strategy)),
            "seed": int(selection_seed),
            "requested_max_items": int(max_items),
            "selection_uses_model_predictions": False,
        },
        "items": items,
        "items_identity_sha256": _sha256_bytes(_canonical_json(identity_rows)),
        "annotations": {
            "path": str(ann_path) if ann_path else "",
            "sha256": sha256_file(ann_path) if ann_path and ann_path.is_file() else "",
        },
        "labels": {
            "path": str(labels_path) if labels_path else "",
            "sha256": sha256_file(labels_path) if labels_path and labels_path.is_file() else "",
        },
        "final_use_note": "A manifest is content-addressed evidence. Final status still depends on the campaign profile and a disjointness check.",
    }
    manifest["manifest_payload_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"})
    return write_json(output, manifest)


def dataset_manifest_identity(manifest: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    content: set[str] = set()
    sample_ids: set[str] = set()
    for item in list(manifest.get("items") or []):
        if not isinstance(item, Mapping):
            continue
        sha = str(item.get("sha256") or "").strip()
        sid = str(item.get("sample_id") or item.get("relative_path") or "").strip()
        if sha:
            content.add(sha)
        if sid:
            sample_ids.add(sid)
    return content, sample_ids


def _normalize_sha256(value: Any) -> str:
    """Return the bare lower-case SHA-256 hex digest for tolerant comparisons."""
    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        raw = raw.split(":", 1)[1]
    return raw


def _deterministic_manifest_sample(items: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    rows = [dict(x) for x in items if isinstance(x, Mapping)]
    if count <= 0 or count >= len(rows):
        return rows
    def key(row: Mapping[str, Any]) -> str:
        identity = str(row.get("sample_id") or row.get("relative_path") or "")
        return hashlib.sha256(("campaign-verify|" + identity).encode("utf-8")).hexdigest()
    return sorted(rows, key=key)[:count]


def verify_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    verify_files: bool = True,
    sample_size: int = 0,
    verification_mode: str | None = None,
) -> dict[str, Any]:
    """Verify dataset-manifest integrity with mode-aware file checks.

    ``verification_mode`` may be ``manifest_only``, ``sampled`` or ``full``.
    The legacy ``verify_files`` flag is retained.  Development runs use a
    deterministic sample so tens of thousands of images are not re-hashed at
    every start; final/strict runs verify every recorded file.
    """
    payload = dict(manifest or {})
    payload_hash = _normalize_sha256(payload.get("manifest_payload_sha256"))
    recomputed_payload_hash = _normalize_sha256(sha256_json({k: v for k, v in payload.items() if k != "manifest_payload_sha256"}))
    payload_ok = bool(payload_hash and payload_hash == recomputed_payload_hash)
    schema_ok = payload.get("schema") == DATASET_MANIFEST_SCHEMA
    root = _resolve_ref(payload.get("root")) if payload.get("root") else Path()
    items = [dict(x) for x in list(payload.get("items") or []) if isinstance(x, Mapping)]
    item_count_ok = int(payload.get("item_count") or 0) == len(items) and bool(items)
    content_required = str(payload.get("hash_mode") or "") == "content"

    mode = str(verification_mode or "").strip().lower().replace("-", "_")
    if not mode:
        if not verify_files:
            mode = "manifest_only"
        elif int(sample_size or 0) > 0:
            mode = "sampled"
        else:
            mode = "full"
    if mode in {"none", "metadata", "manifest", "manifest_only", "fast"}:
        mode = "manifest_only"
    elif mode in {"sample", "sampled", "relaxed"}:
        mode = "sampled"
    else:
        mode = "full"

    checked_items = items if mode == "full" else (_deterministic_manifest_sample(items, max(1, int(sample_size or 24))) if mode == "sampled" else [])
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    if mode != "manifest_only":
        if not root.is_dir():
            missing.append(str(root))
        else:
            for item in checked_items:
                rel = str(item.get("relative_path") or "")
                path = root / rel
                if not path.is_file():
                    missing.append(rel)
                    continue
                expected_size = item.get("size_bytes")
                if expected_size not in (None, "") and int(expected_size) != int(path.stat().st_size):
                    mismatches.append({
                        "relative_path": rel,
                        "expected_size_bytes": int(expected_size),
                        "actual_size_bytes": int(path.stat().st_size),
                    })
                    continue
                expected = _normalize_sha256(item.get("sha256"))
                if content_required:
                    actual_raw = sha256_file(path) or ""
                    actual = _normalize_sha256(actual_raw)
                    if not expected or actual != expected:
                        mismatches.append({"relative_path": rel, "expected": expected, "actual": actual})
        # Annotation and label maps are small and always worth checking whenever
        # file verification is requested.
        for key in ("annotations", "labels"):
            ref = payload.get(key) if isinstance(payload.get(key), Mapping) else {}
            ref_path = _resolve_ref(ref.get("path")) if ref and ref.get("path") else None
            expected = _normalize_sha256(ref.get("sha256")) if ref else ""
            if ref_path:
                if not ref_path.is_file():
                    missing.append(str(ref_path))
                elif expected:
                    actual = _normalize_sha256(sha256_file(ref_path))
                    if actual != expected:
                        mismatches.append({"artifact": key, "path": str(ref_path), "expected": expected, "actual": actual})

    identity_rows = [
        {
            "sample_id": row.get("sample_id"),
            "relative_path": row.get("relative_path"),
            "sha256": _normalize_sha256(row.get("sha256", "")),
            "class_name": row.get("class_name", ""),
        }
        for row in items
    ]
    legacy_identity_rows = [
        {
            "sample_id": row.get("sample_id"),
            "relative_path": row.get("relative_path"),
            "sha256": row.get("sha256", ""),
            "class_name": row.get("class_name", ""),
        }
        for row in items
    ]
    identity_expected = _normalize_sha256(payload.get("items_identity_sha256"))
    identity_actual = _normalize_sha256(_sha256_bytes(_canonical_json(identity_rows)))
    identity_actual_legacy = _normalize_sha256(_sha256_bytes(_canonical_json(legacy_identity_rows)))
    identity_ok = bool(identity_expected and identity_expected in {identity_actual, identity_actual_legacy})
    ok = bool(schema_ok and payload_ok and item_count_ok and identity_ok and not missing and not mismatches)
    return {
        "schema": "onnx-splitpoint/dataset-manifest-verification",
        "schema_version": 2,
        "created_at": now_iso(),
        "ok": ok,
        "schema_ok": schema_ok,
        "payload_hash_ok": payload_ok,
        "identity_hash_ok": identity_ok,
        "item_count_ok": item_count_ok,
        "verify_files": mode != "manifest_only",
        "verification_mode": mode,
        "manifest_item_count": len(items),
        "checked_item_count": len(checked_items),
        "sample_size_requested": int(sample_size or 0),
        "missing_count": len(missing),
        "mismatch_count": len(mismatches),
        "missing_preview": missing[:20],
        "mismatch_preview": mismatches[:20],
    }


def validate_calibration_validation_separation(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(x) for x in manifests if isinstance(x, Mapping)]
    checks: list[dict[str, Any]] = []
    ok = True
    for task in ("classification", "detection"):
        calibration = [x for x in rows if str(x.get("task")) == task and str(x.get("role")) == "calibration"]
        validation = [x for x in rows if str(x.get("task")) == task and str(x.get("role")) == "validation"]
        if not calibration or not validation:
            checks.append({"task": task, "status": "missing", "content_overlap": None, "sample_id_overlap": None})
            ok = False
            continue
        c_content: set[str] = set()
        c_ids: set[str] = set()
        v_content: set[str] = set()
        v_ids: set[str] = set()
        for item in calibration:
            a, b = dataset_manifest_identity(item)
            c_content |= a
            c_ids |= b
        for item in validation:
            a, b = dataset_manifest_identity(item)
            v_content |= a
            v_ids |= b
        content_overlap = sorted(c_content & v_content)
        id_overlap = sorted(c_ids & v_ids)
        status = "pass" if not content_overlap and not id_overlap else "fail"
        if status != "pass":
            ok = False
        checks.append({
            "task": task,
            "status": status,
            "calibration_count": sum(int(x.get("item_count") or 0) for x in calibration),
            "validation_count": sum(int(x.get("item_count") or 0) for x in validation),
            "content_overlap_count": len(content_overlap),
            "sample_id_overlap_count": len(id_overlap),
            "content_overlap_preview": content_overlap[:20],
            "sample_id_overlap_preview": id_overlap[:20],
        })
    return {
        "schema": "onnx-splitpoint/dataset-separation-report",
        "schema_version": 1,
        "created_at": now_iso(),
        "ok": ok,
        "checks": checks,
    }


def create_pipeline_contract_manifest(*, spec: str | Path, output: str | Path) -> Path:
    spec_path = _resolve_ref(spec)
    output_path = _resolve_ref(output)
    output_base = output_path.parent
    raw = _load_structured(spec_path)
    portable_paths = bool(raw.get("portable_paths"))

    def _stored_path(path: Path) -> str:
        if portable_paths:
            try:
                return path.relative_to(output_base).as_posix()
            except ValueError:
                pass
        return str(path)

    entries = list(raw.get("contracts") or raw.get("artifacts") or [])
    if not entries:
        raise ValueError("contract spec must contain a non-empty contracts list")
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        path = _resolve_ref(entry.get("path"), spec_path.parent)
        if not path.is_file():
            raise FileNotFoundError(f"contract artifact not found: {path}")
        normalized_sha = ""
        structured = _load_structured(path)
        if path.suffix.lower() in {".json", ".yaml", ".yml"} and not structured:
            raise ValueError(
                f"structured contract artifact is empty or invalid: {path}"
            )
        if structured:
            normalized_sha = sha256_json(structured)
        implementation_sources: list[dict[str, Any]] = []
        raw_sources = entry.get("implementation_sources")
        if raw_sources is None:
            raw_sources = entry.get("sources")
        source_values = (
            list(raw_sources)
            if isinstance(raw_sources, Sequence)
            and not isinstance(raw_sources, (str, bytes, bytearray))
            else []
        )
        for raw_source in source_values:
            source = _resolve_ref(raw_source, spec_path.parent)
            if not source.is_file():
                raise FileNotFoundError(
                    f"contract implementation source not found: {source}"
                )
            implementation_sources.append({
                "path": _stored_path(source),
                "sha256": sha256_file(source),
                "size_bytes": int(source.stat().st_size),
            })
        unresolved_placeholders = _placeholder_paths(structured)
        for source in implementation_sources:
            source_path = _resolve_ref(
                source.get("path"), output_base if portable_paths else None
            )
            try:
                source_text = source_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                source_text = ""
            if "replace_campaign_endpoint_adapter_placeholder" in source_text:
                unresolved_placeholders.append(
                    f"implementation_sources:{source_path.name}"
                )
        out.append({
            "id": str(entry.get("id") or path.stem),
            "kind": str(entry.get("kind") or "configuration"),
            "task": str(entry.get("task") or "all"),
            "path": _stored_path(path),
            "sha256": sha256_file(path),
            "normalized_sha256": normalized_sha,
            "locked": bool(structured.get("locked")) if structured else bool(entry.get("locked")),
            "size_bytes": int(path.stat().st_size),
            "implementation_sources": implementation_sources,
            "placeholder_free": not unresolved_placeholders,
            "unresolved_placeholder_paths": sorted(
                set(unresolved_placeholders)
            ),
        })
    if not out:
        raise ValueError("no valid contract artifacts found")
    manifest = {
        "schema": PIPELINE_CONTRACT_SCHEMA,
        "schema_version": 2,
        "created_at": now_iso(),
        "claim_scope": str(raw.get("claim_scope") or ""),
        "model_ids": [str(value) for value in list(raw.get("model_ids") or [])],
        "path_resolution": (
            "manifest_relative" if portable_paths else "absolute"
        ),
        "source_spec": _stored_path(spec_path),
        "source_spec_sha256": sha256_file(spec_path),
        "contracts": out,
        "contract_set_sha256": sha256_json(out),
        "required_kinds": ["preprocessing", "decoder", "nms"],
    }
    manifest["manifest_payload_sha256"] = sha256_json(manifest)
    return write_json(output_path, manifest)


def verify_pipeline_contract_manifest(
    manifest: Mapping[str, Any],
    *,
    require_locked: bool = True,
    expected_claim_scope: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = dict(manifest or {})
    manifest_base = (
        _resolve_ref(manifest_path).parent if manifest_path else None
    )
    rows = [dict(x) for x in list(payload.get("contracts") or []) if isinstance(x, Mapping)]
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    placeholder_rows: list[dict[str, Any]] = []
    kinds = {str(row.get("kind") or "") for row in rows}
    for row in rows:
        path = (
            _resolve_ref(row.get("path"), manifest_base)
            if row.get("path")
            else None
        )
        if not path or not path.is_file():
            missing.append(str(path or row.get("path") or ""))
            continue
        expected = str(row.get("sha256") or "")
        actual = sha256_file(path) or ""
        if not expected or not _sha256_equal(expected, actual):
            mismatches.append({"id": row.get("id"), "path": str(path), "expected": expected, "actual": actual})
        structured = _load_structured(path)
        if path.suffix.lower() in {".json", ".yaml", ".yml"} and not structured:
            mismatches.append({
                "id": row.get("id"),
                "path": str(path),
                "kind": "structured_contract_invalid",
            })
        elif structured and bool(row.get("locked")) != (
            structured.get("locked") is True
        ):
            mismatches.append({
                "id": row.get("id"),
                "path": str(path),
                "kind": "structured_lock_mismatch",
                "manifest_locked": row.get("locked"),
                "artifact_locked": structured.get("locked"),
            })
        unresolved_placeholders = _placeholder_paths(structured)
        normalized_expected = str(row.get("normalized_sha256") or "")
        if normalized_expected and structured:
            normalized_actual = sha256_json(structured)
            if normalized_expected != normalized_actual:
                mismatches.append({
                    "id": row.get("id"),
                    "path": str(path),
                    "kind": "normalized_sha256",
                    "expected": normalized_expected,
                    "actual": normalized_actual,
                })
        for source in list(row.get("implementation_sources") or []):
            if not isinstance(source, Mapping):
                mismatches.append({
                    "id": row.get("id"),
                    "kind": "implementation_source_invalid",
                })
                continue
            source_path = (
                _resolve_ref(source.get("path"), manifest_base)
                if source.get("path")
                else None
            )
            if not source_path or not source_path.is_file():
                missing.append(str(source_path or source.get("path") or ""))
                continue
            expected_source = str(source.get("sha256") or "")
            actual_source = sha256_file(source_path) or ""
            if not expected_source or not _sha256_equal(
                expected_source, actual_source,
            ):
                mismatches.append({
                    "id": row.get("id"),
                    "path": str(source_path),
                    "kind": "implementation_source_sha256",
                    "expected": expected_source,
                    "actual": actual_source,
                })
            try:
                source_text = source_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                source_text = ""
            if "replace_campaign_endpoint_adapter_placeholder" in source_text:
                unresolved_placeholders.append(
                    f"implementation_sources:{source_path.name}"
                )
        declared_placeholder_free = row.get("placeholder_free")
        if unresolved_placeholders or declared_placeholder_free is False:
            placeholder_rows.append({
                "id": str(row.get("id") or ""),
                "paths": sorted(set(unresolved_placeholders))
                or list(row.get("unresolved_placeholder_paths") or []),
            })
    required_kinds = {"preprocessing", "decoder", "nms"}
    coverage_ok = required_kinds <= kinds or "all" in kinds
    locked_ok = bool(rows) and all(row.get("locked") is True for row in rows)
    adapter_rows = [
        row for row in rows if str(row.get("kind") or "") == "adapter"
    ]
    adapter_source_coverage_ok = bool(adapter_rows) and all(
        bool(list(row.get("implementation_sources") or []))
        for row in adapter_rows
    )
    set_hash_ok = str(payload.get("contract_set_sha256") or "") == sha256_json(rows)
    schema_ok = payload.get("schema") == PIPELINE_CONTRACT_SCHEMA
    try:
        schema_version = int(payload.get("schema_version") or 1)
    except (TypeError, ValueError):
        schema_version = 0
    manifest_claim_scope = str(payload.get("claim_scope") or "").strip()
    declared_model_ids = [
        str(value) for value in list(payload.get("model_ids") or [])
    ]
    declared_payload_hash = str(payload.get("manifest_payload_sha256") or "")
    manifest_payload_hash_required = bool(
        manifest_claim_scope or declared_model_ids
    )
    manifest_payload_hash_ok = bool(
        (not manifest_payload_hash_required and not declared_payload_hash)
        or declared_payload_hash
        == sha256_json(
            {
                key: value
                for key, value in payload.items()
                if key != "manifest_payload_sha256"
            }
        )
    )
    placeholder_free_ok = not placeholder_rows
    expected_scope = str(expected_claim_scope or "").strip()
    claim_scope_binding_ok = bool(
        not expected_scope or manifest_claim_scope == expected_scope
    )
    semantic_scope = expected_scope or manifest_claim_scope
    matrix_model_ids_ok = bool(
        semantic_scope != EVALUATED_MATRIX_CLAIM_SCOPE
        or declared_model_ids
        == ["resnet50", "yolo26s", "yolov7_paper"]
    )
    matrix_schema_version_ok = bool(
        semantic_scope != EVALUATED_MATRIX_CLAIM_SCOPE
        or schema_version == 2
    )
    evaluated_matrix_semantics = (
        _pipeline_contract_semantics(rows, base_dir=manifest_base)
        if semantic_scope == EVALUATED_MATRIX_CLAIM_SCOPE
        else {
            "ok": True,
            "status": "not_applicable",
        }
    )
    ok = bool(
        schema_ok
        and rows
        and coverage_ok
        and set_hash_ok
        and manifest_payload_hash_ok
        and not missing
        and not mismatches
        and placeholder_free_ok
        and claim_scope_binding_ok
        and matrix_model_ids_ok
        and matrix_schema_version_ok
        and evaluated_matrix_semantics.get("ok")
        and (locked_ok or not require_locked)
    )
    return {
        "schema": "onnx-splitpoint/pipeline-contract-verification",
        "schema_version": 2,
        "created_at": now_iso(),
        "ok": ok,
        "schema_ok": schema_ok,
        "contract_count": len(rows),
        "required_kind_coverage_ok": coverage_ok,
        "locked_ok": locked_ok,
        "adapter_contract_count": len(adapter_rows),
        "adapter_implementation_source_count": sum(
            len(list(row.get("implementation_sources") or []))
            for row in adapter_rows
        ),
        "adapter_source_coverage_ok": adapter_source_coverage_ok,
        "placeholder_free_ok": placeholder_free_ok,
        "placeholder_row_count": len(placeholder_rows),
        "placeholder_rows": placeholder_rows[:20],
        "evaluated_matrix_semantics": evaluated_matrix_semantics,
        "contract_set_hash_ok": set_hash_ok,
        "manifest_payload_hash_ok": manifest_payload_hash_ok,
        "manifest_payload_hash_required": manifest_payload_hash_required,
        "manifest_claim_scope": manifest_claim_scope,
        "expected_claim_scope": expected_scope,
        "claim_scope_binding_ok": claim_scope_binding_ok,
        "matrix_model_ids_ok": matrix_model_ids_ok,
        "matrix_schema_version_ok": matrix_schema_version_ok,
        "declared_model_ids": declared_model_ids,
        "missing_count": len(missing),
        "mismatch_count": len(mismatches),
        "missing_preview": missing[:20],
        "mismatch_preview": mismatches[:20],
        "kinds": sorted(kinds),
    }


def create_energy_calibration_manifest(*, spec: str | Path, output: str | Path) -> Path:
    """Create hash-verifiable full-system measurement evidence.

    ``direct_calibration`` retains the original coefficient/residual contract.
    ``inherited_validated_method`` records that the already validated
    u.RECS implementation is reused unchanged.  The latter deliberately does
    not manufacture a second set of coefficients, residuals, or uncertainty
    values; it binds method provenance, executable/source hashes, and each
    concrete setup/IP/channel instead.
    """
    spec_path = _resolve_ref(spec)
    output_path = _resolve_ref(output)
    raw = _load_structured(spec_path)
    if not raw:
        raise ValueError(f"could not load energy calibration spec: {spec_path}")
    unresolved = _placeholder_paths(raw)
    if unresolved:
        raise ValueError(
            "energy evidence spec contains unresolved placeholders: "
            + ", ".join(unresolved)
        )
    evidence_mode = str(
        raw.get("evidence_mode") or DIRECT_ENERGY_CALIBRATION_MODE
    ).strip().lower()
    if evidence_mode not in ENERGY_EVIDENCE_MODES:
        raise ValueError(
            f"unsupported energy evidence_mode: {evidence_mode!r}"
        )
    artifacts: list[dict[str, Any]] = []
    for entry in list(raw.get("artifacts") or []):
        if not isinstance(entry, Mapping):
            continue
        raw_path = str(entry.get("path") or "").strip()
        executable = str(
            entry.get("executable") or entry.get("command") or ""
        ).strip()
        if raw_path:
            path = _resolve_ref(raw_path, spec_path.parent)
            resolution = "path"
        elif executable:
            resolved = shutil.which(executable)
            if not resolved:
                raise FileNotFoundError(
                    f"energy method executable not found: {executable}"
                )
            path = _resolve_ref(resolved)
            resolution = "executable"
        else:
            raise ValueError(
                "energy evidence artifact needs path or executable"
            )
        if not path.is_file():
            raise FileNotFoundError(f"calibration artifact not found: {path}")
        artifact = {
            "id": str(entry.get("id") or path.stem),
            "kind": str(
                entry.get("kind")
                or (
                    "measurement_implementation"
                    if evidence_mode
                    == INHERITED_VALIDATED_ENERGY_METHOD_MODE
                    else "calibration_data"
                )
            ),
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        # Claim-bearing inherited manifests have one deliberately exact row
        # shape. Resolution diagnostics are useful for legacy/direct evidence,
        # but are not admissible aliases in the exact-reuse contract.
        if evidence_mode != INHERITED_VALIDATED_ENERGY_METHOD_MODE:
            artifact["requested_executable"] = executable
            artifact["resolution"] = resolution
        artifacts.append(artifact)
    calibration = dict(raw.get("calibration") or {}) if isinstance(raw.get("calibration"), Mapping) else {}
    verification = dict(raw.get("verification") or {}) if isinstance(raw.get("verification"), Mapping) else {}
    uncertainty = dict(raw.get("uncertainty") or {}) if isinstance(raw.get("uncertainty"), Mapping) else {}
    manifest = {
        "schema": ENERGY_CALIBRATION_SCHEMA,
        "schema_version": (
            2
            if evidence_mode == INHERITED_VALIDATED_ENERGY_METHOD_MODE
            else 1
        ),
        "created_at": now_iso(),
        "evidence_mode": evidence_mode,
        "channel_id": str(raw.get("channel_id") or ""),
        "scope": str(raw.get("scope") or "").upper(),
        "measurement_point": str(raw.get("measurement_point") or ""),
        "sensor_chain": raw.get("sensor_chain") or {},
        "sample_rate_hz": raw.get("sample_rate_hz"),
        "locked": bool(raw.get("locked")),
        "calibration": calibration,
        "verification": verification,
        "uncertainty": uncertainty,
        "method": dict(raw.get("method") or {})
        if isinstance(raw.get("method"), Mapping)
        else {},
        "validation_reference": dict(raw.get("validation_reference") or {})
        if isinstance(raw.get("validation_reference"), Mapping)
        else {},
        "reuse_attestation": dict(raw.get("reuse_attestation") or {})
        if isinstance(raw.get("reuse_attestation"), Mapping)
        else {},
        "source_release_integrity": copy.deepcopy(
            dict(raw.get("source_release_integrity") or {})
        )
        if isinstance(raw.get("source_release_integrity"), Mapping)
        else {},
        "channel_bindings": [
            dict(row)
            for row in list(raw.get("channel_bindings") or [])
            if isinstance(row, Mapping)
        ],
        "artifacts": artifacts,
        "source_spec": (
            spec_path.relative_to(output_path.parent).as_posix()
            if raw.get("portable_paths")
            and spec_path.parent == output_path.parent
            else str(spec_path)
        ),
        "source_spec_sha256": sha256_file(spec_path),
    }
    manifest["channel_binding_set_sha256"] = sha256_json(
        manifest["channel_bindings"]
    )
    manifest["artifact_set_sha256"] = sha256_json(artifacts)
    manifest["manifest_payload_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"})
    return write_json(output_path, manifest)


def verify_energy_calibration_manifest(
    manifest: Mapping[str, Any],
    *,
    require_final: bool = True,
    expected_channel_bindings: Sequence[Mapping[str, Any]] | None = None,
    require_source_integrity_binding: bool = False,
) -> dict[str, Any]:
    payload = dict(manifest or {})
    rows = [dict(x) for x in list(payload.get("artifacts") or []) if isinstance(x, Mapping)]
    missing: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for row in rows:
        path = _resolve_ref(row.get("path")) if row.get("path") else None
        if not path or not path.is_file():
            missing.append(str(path or row.get("path") or ""))
            continue
        actual = sha256_file(path) or ""
        expected = str(row.get("sha256") or "")
        if not expected or _normalise_sha256(actual) != _normalise_sha256(expected):
            mismatches.append({"id": row.get("id"), "path": str(path), "expected": expected, "actual": actual})
    calibration = payload.get("calibration") if isinstance(payload.get("calibration"), Mapping) else {}
    verification = payload.get("verification") if isinstance(payload.get("verification"), Mapping) else {}
    uncertainty = payload.get("uncertainty") if isinstance(payload.get("uncertainty"), Mapping) else {}
    coefficients = calibration.get("coefficients") if isinstance(calibration.get("coefficients"), Mapping) else {}
    coefficient_values = [_number(value) for value in coefficients.values()]
    calibration_ok = bool(
        str(calibration.get("model") or "")
        and coefficients
        and coefficient_values
        and all(value is not None for value in coefficient_values)
    )
    verification_ok = bool(
        verification.get("independent") is True
        and verification.get("pass") is True
        and (_number(verification.get("sample_count")) or 0.0) > 0
    )
    uncertainty_ok = bool(
        _number(uncertainty.get("combined_relative_percent")) is not None
        or _number(uncertainty.get("expanded_relative_percent")) is not None
    )
    evidence_mode = str(
        payload.get("evidence_mode") or DIRECT_ENERGY_CALIBRATION_MODE
    ).strip().lower()
    evidence_mode_ok = evidence_mode in ENERGY_EVIDENCE_MODES
    schema_version = _number(payload.get("schema_version"))
    schema_version_ok = bool(
        schema_version == (
            2.0
            if evidence_mode == INHERITED_VALIDATED_ENERGY_METHOD_MODE
            else 1.0
        )
    )
    placeholder_paths = _placeholder_paths(payload)
    placeholder_free_ok = not placeholder_paths
    bindings = [
        dict(row)
        for row in list(payload.get("channel_bindings") or [])
        if isinstance(row, Mapping)
    ]
    binding_ids = [str(row.get("setup_id") or "").strip() for row in bindings]
    binding_rows_valid = bool(bindings) and all(
        str(row.get("setup_id") or "").strip()
        and str(row.get("urecs_address") or "").strip()
        and not isinstance(row.get("channel"), bool)
        and isinstance(row.get("channel"), int)
        and int(row.get("channel")) >= 0
        and (_number(row.get("sample_rate_hz")) or 0.0) > 0
        and str(row.get("scope") or "").strip().upper() == "FS"
        and str(row.get("measurement_point") or "").strip()
        == "complete_system_input"
        for row in bindings
    )
    binding_unique_ok = bool(binding_ids) and (
        len(binding_ids) == len(set(binding_ids))
    )
    binding_set_hash_ok = str(
        payload.get("channel_binding_set_sha256") or ""
    ) == sha256_json(bindings)

    def _binding_projection(
        source: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted(
            [
                {
                    "setup_id": str(row.get("setup_id") or "").strip(),
                    "urecs_address": str(
                        row.get("urecs_address") or ""
                    ).strip(),
                    "channel": row.get("channel"),
                    "sample_rate_hz": row.get("sample_rate_hz"),
                    "scope": str(row.get("scope") or "").strip().upper(),
                    "measurement_point": str(
                        row.get("measurement_point") or ""
                    ).strip(),
                }
                for row in source
            ],
            key=lambda row: str(row.get("setup_id") or ""),
        )

    expected_bindings = [
        dict(row)
        for row in list(expected_channel_bindings or [])
        if isinstance(row, Mapping)
    ]
    expected_bindings_ok = bool(
        not expected_bindings
        or _binding_projection(bindings)
        == _binding_projection(expected_bindings)
    )
    method = (
        dict(payload.get("method") or {})
        if isinstance(payload.get("method"), Mapping)
        else {}
    )
    reference = (
        dict(payload.get("validation_reference") or {})
        if isinstance(payload.get("validation_reference"), Mapping)
        else {}
    )
    attestation = (
        dict(payload.get("reuse_attestation") or {})
        if isinstance(payload.get("reuse_attestation"), Mapping)
        else {}
    )
    source_integrity = (
        dict(payload.get("source_release_integrity") or {})
        if isinstance(payload.get("source_release_integrity"), Mapping)
        else {}
    )
    source_integrity_report = (
        dict(source_integrity.get("verification_report") or {})
        if isinstance(source_integrity.get("verification_report"), Mapping)
        else {}
    )

    def _strict_lower_sha256(value: Any) -> str:
        text = value if isinstance(value, str) else ""
        return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""

    source_integrity_binding_present = bool(source_integrity)
    source_integrity_binding_shape_ok = bool(
        set(source_integrity)
        == {
            "schema",
            "schema_version",
            "package_version",
            "build_id",
            "manifest_path",
            "manifest_sha256",
            "sha256sums_path",
            "sha256sums_sha256",
            "verification_report",
        }
        and source_integrity.get("schema") == SOURCE_INTEGRITY_BINDING_SCHEMA
        and type(source_integrity.get("schema_version")) is int
        and source_integrity.get("schema_version") == 1
        and isinstance(source_integrity.get("package_version"), str)
        and bool(source_integrity.get("package_version"))
        and isinstance(source_integrity.get("build_id"), str)
        and bool(source_integrity.get("build_id"))
        and isinstance(source_integrity.get("manifest_path"), str)
        and Path(source_integrity.get("manifest_path") or "").is_absolute()
        and bool(_strict_lower_sha256(source_integrity.get("manifest_sha256")))
        and isinstance(source_integrity.get("sha256sums_path"), str)
        and Path(source_integrity.get("sha256sums_path") or "").is_absolute()
        and bool(_strict_lower_sha256(source_integrity.get("sha256sums_sha256")))
        and source_integrity_report.get("ok") is True
        and source_integrity_report.get("status") == "verified"
        and all(
            source_integrity_report.get(field) == source_integrity.get(field)
            for field in (
                "package_version",
                "build_id",
                "manifest_path",
                "manifest_sha256",
                "sha256sums_path",
                "sha256sums_sha256",
            )
        )
    )
    inherited_method_ok = bool(
        str(payload.get("channel_id") or "").strip()
        and str(payload.get("measurement_point") or "").strip()
        == "complete_system_input"
        and _number(payload.get("sample_rate_hz")) == 2000.0
        and method.get("collector") == "urecs-data-collector"
        and method.get("postprocessor") == "power_calculations"
        and method.get("collector_mode") == "fast_firmware"
        and _number(method.get("sample_rate_hz")) == 2000.0
        and method.get("output_semantics")
        == "calibrated_input_energy_unsubtracted"
    )
    expected_method_identity_ok = bool(
        not expected_bindings
        or (
            str(payload.get("channel_id") or "").strip()
            == "urecs_fs_input_channel_0"
            and method.get("implementation_policy")
            == "exact_validated_implementation_reuse"
        )
    )
    inherited_reference_ok = bool(
        str(reference.get("reference_id") or "").strip()
        == str(WACHSMUTH_ENERGY_METHOD_REFERENCE["reference_id"])
        and str(reference.get("author") or "").strip()
        == str(WACHSMUTH_ENERGY_METHOD_REFERENCE["author"])
        and str(reference.get("title") or "").strip()
        == str(WACHSMUTH_ENERGY_METHOD_REFERENCE["title"])
        and str(reference.get("internal_identifier") or "").strip()
        == "M99"
        and str(reference.get("institution") or "").strip()
        == str(WACHSMUTH_ENERGY_METHOD_REFERENCE["institution"])
        and str(reference.get("work_type") or "").strip()
        == str(WACHSMUTH_ENERGY_METHOD_REFERENCE["work_type"])
        and _number(reference.get("year")) == 2026.0
    )
    inherited_attestation_ok = bool(
        attestation.get("validated_method_accepted") is True
        and attestation.get("exact_implementation_reused") is True
        and attestation.get("new_calibration_required") is False
        and str(attestation.get("attested_by") or "").strip()
        and str(attestation.get("attested_at") or "").strip()
    )
    implementation_artifacts_ok = bool(rows) and any(
        str(row.get("kind") or "") == "measurement_implementation"
        for row in rows
    )
    artifact_ids = {str(row.get("id") or "").strip() for row in rows}
    evaluated_matrix_required_artifact_ids = {
        "tool_source_collector",
        "tool_source_config",
        "tool_source_metrics",
        "tool_source_energy_measurement_cli",
        "tool_source_native_producer_energy_plan",
        "tool_source_run_native_producer_energy_from_summary",
        "urecs_data_collector_binary",
        "power_calculations_binary",
    }
    expected_implementation_artifacts_ok = bool(
        not expected_bindings
        or evaluated_matrix_required_artifact_ids <= artifact_ids
    )
    payload_hash_ok = str(payload.get("manifest_payload_sha256") or "") == sha256_json({k: v for k, v in payload.items() if k != "manifest_payload_sha256"})
    artifact_hash_ok = str(payload.get("artifact_set_sha256") or "") == sha256_json(rows)
    direct_final_ok = bool(
        payload.get("scope") == "FS"
        and payload.get("locked") is True
        and calibration_ok
        and verification_ok
        and uncertainty_ok
    )
    inherited_final_ok = bool(
        payload.get("scope") == "FS"
        and payload.get("locked") is True
        and inherited_method_ok
        and expected_method_identity_ok
        and inherited_reference_ok
        and inherited_attestation_ok
        and implementation_artifacts_ok
        and expected_implementation_artifacts_ok
        and binding_rows_valid
        and binding_unique_ok
        and binding_set_hash_ok
        and expected_bindings_ok
        and (
            source_integrity_binding_shape_ok
            or not require_source_integrity_binding
        )
    )
    final_ok = (
        inherited_final_ok
        if evidence_mode == INHERITED_VALIDATED_ENERGY_METHOD_MODE
        else direct_final_ok
    )
    ok = bool(
        payload.get("schema") == ENERGY_CALIBRATION_SCHEMA
        and schema_version_ok
        and evidence_mode_ok
        and placeholder_free_ok
        and payload_hash_ok
        and artifact_hash_ok
        and not missing
        and not mismatches
        and (final_ok or not require_final)
    )
    return {
        "schema": "onnx-splitpoint/energy-calibration-verification",
        "schema_version": 2,
        "created_at": now_iso(),
        "ok": ok,
        "evidence_mode": evidence_mode,
        "evidence_mode_ok": evidence_mode_ok,
        "schema_ok": payload.get("schema") == ENERGY_CALIBRATION_SCHEMA,
        "schema_version_ok": schema_version_ok,
        "payload_hash_ok": payload_hash_ok,
        "artifact_set_hash_ok": artifact_hash_ok,
        "final_scope_and_lock_ok": final_ok,
        "direct_calibration_final_ok": direct_final_ok,
        "inherited_validated_method_final_ok": inherited_final_ok,
        "calibration_coefficients_ok": calibration_ok,
        "independent_verification_ok": verification_ok,
        "uncertainty_budget_ok": uncertainty_ok,
        "inherited_method_contract_ok": inherited_method_ok,
        "expected_method_identity_ok": expected_method_identity_ok,
        "inherited_validation_reference_ok": inherited_reference_ok,
        "inherited_reuse_attestation_ok": inherited_attestation_ok,
        "source_integrity_binding_required": bool(
            require_source_integrity_binding
        ),
        "source_integrity_binding_present": (
            source_integrity_binding_present
        ),
        "source_integrity_binding_shape_ok": (
            source_integrity_binding_shape_ok
        ),
        "implementation_artifacts_ok": implementation_artifacts_ok,
        "expected_implementation_artifacts_ok": (
            expected_implementation_artifacts_ok
        ),
        "implementation_artifact_ids": sorted(artifact_ids),
        "channel_bindings_ok": bool(
            binding_rows_valid
            and binding_unique_ok
            and binding_set_hash_ok
            and expected_bindings_ok
        ),
        "channel_binding_count": len(bindings),
        "channel_binding_ids": sorted(binding_ids),
        "channel_binding_set_hash_ok": binding_set_hash_ok,
        "expected_channel_bindings_ok": expected_bindings_ok,
        "placeholder_free_ok": placeholder_free_ok,
        "placeholder_paths": placeholder_paths[:20],
        "missing_count": len(missing),
        "mismatch_count": len(mismatches),
        "missing_preview": missing[:20],
        "mismatch_preview": mismatches[:20],
    }


def validate_ranking_model_bundle(
    bundle: Mapping[str, Any],
    *,
    development_model_ids: Sequence[str] = (),
    holdout_model_ids: Sequence[str] = (),
    require_native_handover: bool = False,
) -> dict[str, Any]:
    """Validate that a frozen ranking bundle was fit only from declared development rows.

    Final-campaign bundles are deliberately stricter than legacy fitting helpers:
    the payload hash is mandatory, hold-out rows must be excluded, and blank or
    screening roles must not have been silently interpreted as development data.
    """
    payload = dict(bundle or {})
    dev_ids = {str(x) for x in development_model_ids if str(x)}
    holdout_ids = {str(x) for x in holdout_model_ids if str(x)}
    training_ids = {str(x) for x in list(payload.get("development_model_ids") or []) if str(x)}
    stage_models = payload.get("stage_time_models") if isinstance(payload.get("stage_time_models"), Mapping) else {}
    handover_models = payload.get("handover_models") if isinstance(payload.get("handover_models"), Mapping) else {}
    fitted_stage = [row for row in stage_models.values() if isinstance(row, Mapping) and row.get("status") == "fitted"]
    fitted_native = [
        row
        for row in list((handover_models.get("native_fifo") or {}).values())
        if isinstance(row, Mapping) and row.get("status") == "fitted"
    ] if isinstance(handover_models.get("native_fifo"), Mapping) else []
    fit_policy = payload.get("fit_policy") if isinstance(payload.get("fit_policy"), Mapping) else {}
    holdout_excluded = bool(fit_policy.get("holdout_rows_excluded") is True)
    explicit_development_required = bool(fit_policy.get("explicit_development_role_required") is True)
    unlabeled_treated_as_development = bool(fit_policy.get("unlabeled_rows_treated_as_development") is True)
    role_policy_ok = explicit_development_required and not unlabeled_treated_as_development
    disjoint = not bool(training_ids & holdout_ids)
    subset_ok = not dev_ids or training_ids <= dev_ids
    payload_hash = str(payload.get("bundle_payload_sha256") or "")
    payload_hash_ok = bool(payload_hash) and _normalise_sha256(payload_hash) == _normalise_sha256(sha256_json({k: v for k, v in payload.items() if k != "bundle_payload_sha256"}))
    ok = bool(
        payload.get("schema") == RANKING_MODEL_BUNDLE_SCHEMA
        and holdout_excluded
        and role_policy_ok
        and disjoint
        and subset_ok
        and fitted_stage
        and (fitted_native or not require_native_handover)
        and payload_hash_ok
    )
    return {
        "schema": "onnx-splitpoint/ranking-model-bundle-verification",
        "schema_version": 1,
        "created_at": now_iso(),
        "ok": ok,
        "schema_ok": payload.get("schema") == RANKING_MODEL_BUNDLE_SCHEMA,
        "payload_hash_ok": payload_hash_ok,
        "holdout_rows_excluded": holdout_excluded,
        "explicit_development_role_required": explicit_development_required,
        "unlabeled_rows_treated_as_development": unlabeled_treated_as_development,
        "role_policy_ok": role_policy_ok,
        "training_holdout_disjoint": disjoint,
        "training_ids_subset_of_development": subset_ok,
        "training_model_ids": sorted(training_ids),
        "excluded_non_development_row_count": int(payload.get("excluded_non_development_row_count") or 0),
        "fitted_stage_model_count": len(fitted_stage),
        "fitted_native_handover_model_count": len(fitted_native),
    }


def apply_ranking_model_bundle(profile: Mapping[str, Any], *, profile_path: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a frozen model bundle into the runtime profile.

    Profiles keep the bundle as a separate immutable artefact.  This helper
    materialises its profile patch in memory before candidate predictions are
    frozen, while retaining an audit record of the source hash.
    """
    resolved = dict(profile or {})
    refs = campaign_references(resolved, profile_path)
    bundle_path = refs.get("ranking_model_bundle")
    if not bundle_path or not bundle_path.is_file():
        return resolved, {"status": "not_configured", "path": str(bundle_path or "")}
    bundle = _load_structured(bundle_path)
    patch_root = bundle.get("profile_patch") if isinstance(bundle.get("profile_patch"), Mapping) else {}
    patch = patch_root.get("ranking_validation") if isinstance(patch_root.get("ranking_validation"), Mapping) else {}
    if not patch:
        return resolved, {"status": "invalid_bundle", "path": str(bundle_path), "sha256": sha256_file(bundle_path)}
    ranking = dict(resolved.get("ranking_validation") or {}) if isinstance(resolved.get("ranking_validation"), Mapping) else {}
    for section in ("cycle_time_no_handover", "cycle_time_with_handover"):
        incoming = patch.get(section) if isinstance(patch.get(section), Mapping) else {}
        current = dict(ranking.get(section) or {}) if isinstance(ranking.get(section), Mapping) else {}
        current.update(dict(incoming))
        ranking[section] = current
    resolved["ranking_validation"] = ranking
    campaign = dict(resolved.get("campaign") or {}) if isinstance(resolved.get("campaign"), Mapping) else {}
    campaign["ranking_model_bundle_resolved_sha256"] = sha256_file(bundle_path)
    campaign["ranking_model_bundle_loaded"] = True
    resolved["campaign"] = campaign
    return resolved, {
        "status": "loaded",
        "path": str(bundle_path),
        "sha256": sha256_file(bundle_path),
        "bundle_payload_sha256": bundle.get("bundle_payload_sha256", ""),
    }


def _model_entries(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    suite = profile.get("model_suite") if isinstance(profile.get("model_suite"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for tier in ("primary", "reserve"):
        for item in list(suite.get(tier) or []):
            row = dict(item) if isinstance(item, Mapping) else {"id": str(item)}
            # Reserve/stress entries can be documented in a campaign profile
            # without becoming part of the active execution, registry, or
            # readiness projection.  Opt-in must be explicit.
            if isinstance(item, Mapping) and item.get("enabled") is False:
                continue
            if row.get("id"):
                row.setdefault("suite_tier", tier)
                rows.append(row)
    return rows


def _holdout_registry_models(profile: Mapping[str, Any], profile_path: Path) -> list[dict[str, Any]]:
    """Return the model-role projection bound by the hold-out registry.

    The registry intentionally binds only model identity, role, candidate-universe
    declaration and human attestation.  Ranking coefficients and other unrelated
    profile sections may be patched later without invalidating the hold-out
    decision, while changing a model file, role or attestation invalidates it.
    """
    models: list[dict[str, Any]] = []
    for row in _model_entries(profile):
        model_ref = row.get("path") or row.get("model_path") or row.get("onnx")
        model_path = _resolve_ref(model_ref, profile_path.parent) if model_ref else None
        models.append({
            "id": str(row.get("id")),
            "family": str(row.get("family") or ""),
            "family_id": str(row.get("family_id") or ""),
            "task": str(row.get("task") or "auto"),
            "suite_tier": str(row.get("suite_tier") or "primary"),
            "evaluation_role": str(row.get("evaluation_role") or ""),
            "generalization_scope": str(row.get("generalization_scope") or ""),
            "validation_tier": str(row.get("validation_tier") or ""),
            "holdout_group": str(row.get("holdout_group") or ""),
            "model_path": str(model_path) if model_path else "",
            "model_sha256": sha256_file(model_path) if model_path and model_path.is_file() else "",
            "candidate_universe": dict(row.get("candidate_universe") or {}) if isinstance(row.get("candidate_universe"), Mapping) else {},
            "unseen_attestation": dict(row.get("unseen_attestation") or {}) if isinstance(row.get("unseen_attestation"), Mapping) else {},
        })
    return models


def create_holdout_registry(*, profile: str | Path, output: str | Path) -> Path:
    profile_path = _resolve_ref(profile)
    payload = _load_structured(profile_path)
    models = _holdout_registry_models(payload, profile_path)
    registry = {
        "schema": HOLDOUT_REGISTRY_SCHEMA,
        "schema_version": 2,
        "created_at": now_iso(),
        "profile": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "model_role_projection_sha256": sha256_json(models),
        "models": models,
        "rule": "A hold-out must be selected before its measurements are inspected. The tool requires an explicit human attestation and cannot infer historical exposure automatically.",
    }
    registry["registry_sha256"] = sha256_json({k: v for k, v in registry.items() if k != "registry_sha256"})
    return write_json(output, registry)


def verify_holdout_registry(
    registry: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
    profile_path: str | Path | None = None,
    require_attestation: bool = True,
) -> dict[str, Any]:
    payload = dict(registry or {})
    registry_rows = {str(row.get("id") or ""): dict(row) for row in list(payload.get("models") or []) if isinstance(row, Mapping) and row.get("id")}
    profile_rows = {str(row.get("id") or ""): dict(row) for row in _model_entries(profile) if row.get("id")}
    missing = sorted(set(profile_rows) - set(registry_rows))
    extra = sorted(set(registry_rows) - set(profile_rows))
    mismatches: list[dict[str, Any]] = []
    attestation_failures: list[str] = []
    model_identity_failures: list[str] = []
    for model_id, row in profile_rows.items():
        reg = registry_rows.get(model_id) or {}
        role_profile = normalize_evaluation_role(row.get("evaluation_role"))
        role_registry = normalize_evaluation_role(reg.get("evaluation_role"))
        for field, profile_value, registry_value in (
            ("evaluation_role", role_profile, role_registry),
            ("family_id", str(row.get("family_id") or ""), str(reg.get("family_id") or "")),
            ("generalization_scope", str(row.get("generalization_scope") or ""), str(reg.get("generalization_scope") or "")),
            ("validation_tier", str(row.get("validation_tier") or ""), str(reg.get("validation_tier") or "")),
        ):
            if profile_value != registry_value:
                mismatches.append({"model_id": model_id, "field": field, "profile": profile_value, "registry": registry_value})
        if str(row.get("holdout_group") or "") != str(reg.get("holdout_group") or ""):
            mismatches.append({"model_id": model_id, "field": "holdout_group", "profile": row.get("holdout_group", ""), "registry": reg.get("holdout_group", "")})
        if is_confirmatory_holdout(role_profile) and require_attestation:
            att = reg.get("unseen_attestation") if isinstance(reg.get("unseen_attestation"), Mapping) else {}
            if not bool(att.get("unseen") is True and str(att.get("attested_by") or "").strip() and str(att.get("attested_at") or "").strip()):
                attestation_failures.append(model_id)
        model_path = str(reg.get("model_path") or "").strip()
        expected_model_hash = str(reg.get("model_sha256") or "").strip()
        if require_attestation and (not model_path or not expected_model_hash):
            model_identity_failures.append(model_id)
        if model_path and expected_model_hash:
            path = _resolve_ref(model_path)
            actual = sha256_file(path) if path.is_file() else ""
            if actual != expected_model_hash:
                mismatches.append({"model_id": model_id, "field": "model_sha256", "expected": expected_model_hash, "actual": actual})

    registry_hash_ok = str(payload.get("registry_sha256") or "") == sha256_json({k: v for k, v in payload.items() if k != "registry_sha256"})
    source_profile_hash_ok = True
    source_profile_actual = ""
    projection_expected = str(payload.get("model_role_projection_sha256") or "")
    projection_actual = ""
    projection_ok = False
    if profile_path:
        source = _resolve_ref(profile_path)
        if source.is_file():
            source_profile_actual = sha256_file(source) or ""
            source_profile_hash_ok = str(payload.get("profile_sha256") or "") == source_profile_actual
            expected_models = _holdout_registry_models(profile, source)
            projection_actual = sha256_json(expected_models)
            projection_ok = bool(projection_expected and projection_expected == projection_actual)
    else:
        # In-memory verification cannot reliably resolve relative model paths.
        # Compare the stored model rows directly when no profile path is known.
        projection_actual = sha256_json([registry_rows[key] for key in profile_rows if key in registry_rows])
        projection_ok = bool(projection_expected and projection_expected == projection_actual)

    # v60.1 registries use the narrow model-role projection.  Older registries
    # fall back to the full source-profile hash for backwards compatibility.
    profile_binding_ok = projection_ok if projection_expected else source_profile_hash_ok
    ok = bool(
        payload.get("schema") == HOLDOUT_REGISTRY_SCHEMA
        and registry_hash_ok
        and profile_binding_ok
        and not missing
        and not extra
        and not mismatches
        and not attestation_failures
        and not model_identity_failures
    )
    return {
        "schema": "onnx-splitpoint/holdout-registry-verification",
        "schema_version": 1,
        "created_at": now_iso(),
        "ok": ok,
        "schema_ok": payload.get("schema") == HOLDOUT_REGISTRY_SCHEMA,
        "registry_hash_ok": registry_hash_ok,
        "profile_binding_ok": profile_binding_ok,
        "model_role_projection_hash_ok": projection_ok,
        "model_role_projection_sha256": projection_actual,
        "source_profile_hash_ok": source_profile_hash_ok,
        "source_profile_sha256": source_profile_actual,
        "missing_model_ids": missing,
        "extra_model_ids": extra,
        "mismatches": mismatches[:50],
        "attestation_failures": attestation_failures,
        "model_identity_failures": model_identity_failures,
    }


def _number(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def _case_id(row: Mapping[str, Any], index: int) -> str:
    value = str(row.get("case_id") or row.get("case") or "").strip()
    if value:
        return value
    boundary = row.get("boundary", row.get("split_index"))
    try:
        return f"b{int(boundary):03d}"
    except Exception:
        return f"candidate_{index:04d}"


def stable_candidate_identity(model_id: str, row: Mapping[str, Any], index: int = 0) -> dict[str, str]:
    """Return a score- and measurement-independent candidate identity.

    ``case_id`` alone is only unique inside one model.  The qualified ID is
    therefore the stable join key, while the digest catches accidental changes
    to the graph boundary represented by that ID.
    """
    case_id = _case_id(row, index)
    boundary = row.get("boundary", row.get("split_index", row.get("boundary_index")))
    try:
        boundary = int(boundary)
    except Exception:
        boundary = None
    model_hash = str(row.get("model_sha256") or row.get("model_hash") or "").strip()
    qualified = f"{str(model_id).strip()}::{case_id}"
    core = {
        "model_id": str(model_id).strip(),
        "model_sha256": model_hash,
        "case_id": case_id,
        "boundary": boundary,
    }
    core_digest = sha256_json(core)
    stratum = {
        "direction": str(row.get("direction") or row.get("pipeline_direction") or "").strip(),
        "runner_regime": str(row.get("runner_regime") or row.get("runner") or "").strip(),
        "setup_id": str(row.get("setup_id") or row.get("hardware_setup_id") or row.get("setup") or "").strip(),
        "backend": str(row.get("backend") or row.get("producer_backend") or "").strip(),
        "precision": str(row.get("precision") or row.get("dtype") or "").strip(),
        "contract_id": str(row.get("contract_id") or row.get("contract") or "").strip(),
        "contract_sha256": str(row.get("contract_sha256") or row.get("contract_hash") or "").strip(),
    }
    execution_payload = {"candidate_core_sha256": core_digest, **{k: v for k, v in stratum.items() if v}}
    execution_digest = sha256_json(execution_payload)
    return {
        "candidate_id": qualified,
        "candidate_identity_sha256": core_digest,
        "execution_candidate_id": f"{qualified}::{execution_digest.replace('sha256:', '')[:16]}",
        "execution_candidate_identity_sha256": execution_digest,
    }


def _static_candidate_row(row: Mapping[str, Any], index: int, *, model_id: str = "") -> dict[str, Any]:
    result = {
        "case_id": _case_id(row, index),
        "boundary": row.get("boundary", row.get("split_index")),
        "cut_bytes": row.get("cut_bytes", row.get("cost_bytes", row.get("crossing_bytes"))),
        "cut_mib": row.get("cut_mib_val", row.get("cut_mib")),
        "imbalance": row.get("imbalance_val", row.get("imbalance")),
        "n_cut_tensors": row.get("n_cut_tensors", row.get("crossing_tensor_count")),
        "strict_ok": row.get("strict_ok"),
        "exclude_reason": row.get("exclude_reason", ""),
    }
    if model_id:
        result.update(stable_candidate_identity(model_id, {**dict(row), **result}, index))
    return result


def _static_candidate_order_key(row: Mapping[str, Any]) -> tuple[float, str, str, str]:
    """Canonical, score-independent ordering for candidate-universe artefacts.

    Predictor output is normally score-sorted, but the deterministic audit must
    not inherit that ordering when two static extrema tie.  Keeping the same
    order in the JSON/CSV payload also makes the universe identity independent
    of the caller's candidate-list order.
    """
    boundary = _number(row.get("boundary", row.get("split_index", row.get("boundary_index"))))
    return (
        float(boundary) if boundary is not None else math.inf,
        str(row.get("case_id") or row.get("case") or ""),
        str(row.get("candidate_id") or ""),
        str(row.get("candidate_identity_sha256") or ""),
    )


def deterministic_audit_cases(candidates: Sequence[Mapping[str, Any]], *, model_id: str, size: int, seed: int) -> list[str]:
    """Select a score-independent audit universe from static graph features.

    Selection covers feature extrema and evenly spaced graph-position windows;
    ties and backfill use a deterministic SHA-256 order.  No predicted score or
    measured runtime is used.
    """
    rows = sorted(
        (
            _static_candidate_row(row, idx, model_id=model_id)
            for idx, row in enumerate(candidates, start=1)
            if row.get("strict_ok") is not False
        ),
        key=_static_candidate_order_key,
    )
    if not rows:
        return []
    size = max(1, min(int(size or len(rows)), len(rows)))
    selected: list[str] = []

    def add(row: Mapping[str, Any]) -> None:
        cid = str(row.get("case_id") or "")
        if cid and cid not in selected and len(selected) < size:
            selected.append(cid)

    # Static extrema improve coverage of communication and balance regimes.
    for key in ("cut_bytes", "cut_mib", "imbalance", "n_cut_tensors"):
        valid = [row for row in rows if _number(row.get(key)) is not None]
        if valid:
            add(min(valid, key=lambda row: (float(_number(row.get(key)) or 0.0), _static_candidate_order_key(row))))
            add(min(valid, key=lambda row: (-float(_number(row.get(key)) or 0.0), _static_candidate_order_key(row))))

    ordered = sorted(rows, key=lambda row: (int(_number(row.get("boundary")) or 0), str(row.get("case_id"))))
    remaining = max(0, size - len(selected))
    if remaining:
        for window_index in range(remaining):
            lo = math.floor(window_index * len(ordered) / remaining)
            hi = max(lo + 1, math.floor((window_index + 1) * len(ordered) / remaining))
            window = ordered[lo:hi]
            if not window:
                continue
            chosen = min(
                window,
                key=lambda row: hashlib.sha256(f"{seed}|{model_id}|{row.get('case_id')}".encode("utf-8")).hexdigest(),
            )
            add(chosen)

    for row in sorted(rows, key=lambda row: hashlib.sha256(f"{seed}|{model_id}|{row.get('case_id')}".encode("utf-8")).hexdigest()):
        add(row)
        if len(selected) >= size:
            break
    return selected


def create_candidate_universe_manifest(
    *,
    model_id: str,
    candidates: Sequence[Mapping[str, Any]],
    mode: str,
    output_dir: str | Path,
    audit_size: int = 0,
    minimum_valid_candidates: int = 10,
    seed: int = 20260710,
    source_prediction_sha256: str = "",
    identity_context: Mapping[str, Any] | None = None,
    write_artifacts: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    mode_l = str(mode or "declared_shortlist").strip().lower().replace("-", "_")
    allowed_modes = {"all_feasible", "deterministic_audit", "audit", "audit_universe", "declared_shortlist"}
    if mode_l not in allowed_modes:
        raise ValueError(f"unsupported candidate-universe mode: {mode_l!r}")
    identity_context = dict(identity_context or {})
    feasible = sorted(
        (
            _static_candidate_row({**dict(row), **identity_context}, idx, model_id=model_id)
            for idx, row in enumerate(candidates, start=1)
            if row.get("strict_ok") is not False
        ),
        key=_static_candidate_order_key,
    )
    min_valid = max(1, int(minimum_valid_candidates or 10))
    if mode_l == "all_feasible":
        selected = [str(row.get("case_id")) for row in feasible]
        claim_scope = "complete_feasible_universe"
        declared_complete = True
    elif mode_l in {"deterministic_audit", "audit", "audit_universe"}:
        selected = deterministic_audit_cases(feasible, model_id=model_id, size=(audit_size or min(20, len(feasible))), seed=seed)
        claim_scope = "predeclared_audit_universe"
        declared_complete = True
        mode_l = "deterministic_audit"
    else:
        selected = [str(row.get("case_id")) for row in feasible]
        claim_scope = "development_shortlist"
        declared_complete = False
        mode_l = "declared_shortlist"
    selected_set = set(selected)
    for row in feasible:
        row["selected_for_measurement"] = str(row.get("case_id")) in selected_set
        row["candidate_role"] = "audit" if row["selected_for_measurement"] and mode_l in {"all_feasible", "deterministic_audit"} else "not_in_audit"
    identity_by_case = {
        str(row.get("case_id") or ""): str(row.get("candidate_id") or "")
        for row in feasible
    }
    selected_identities = [identity_by_case.get(case_id, "") for case_id in selected]
    feasible_identities = [str(row.get("candidate_id") or "") for row in feasible]
    payload = {
        "schema": CANDIDATE_UNIVERSE_SCHEMA,
        "schema_version": 1,
        "created_at": now_iso(),
        "model_id": model_id,
        "model_sha256": str(identity_context.get("model_sha256") or identity_context.get("model_hash") or ""),
        "mode": mode_l,
        "claim_scope": claim_scope,
        "declared_complete": declared_complete,
        "source_prediction_sha256": source_prediction_sha256,
        "feasible_candidate_count": len(feasible),
        "selected_candidate_count": len(selected),
        "selected_case_ids": selected,
        "selected_candidate_ids": selected_identities,
        "feasible_candidate_identity_sha256": sha256_json(feasible_identities),
        "selected_candidate_identity_sha256": sha256_json(selected_identities),
        "audit": {
            "requested_size": int(audit_size or (20 if mode_l == "deterministic_audit" else 0)),
            "selected_size": len(selected) if mode_l in {"all_feasible", "deterministic_audit"} else 0,
            "minimum_valid_candidates": min_valid,
            "candidate_count_gate_satisfiable": len(selected) >= min_valid if mode_l in {"all_feasible", "deterministic_audit"} else False,
            "seed": int(seed),
            "selection_uses_predictions": False,
            "selection_uses_measurements": False,
        },
        "candidates": feasible,
    }
    # The identity hash deliberately excludes the wall-clock timestamp.  A
    # resume of an unchanged analysis must reproduce the same universe hash;
    # otherwise the prospective prediction freeze would falsely conflict.
    payload["universe_sha256"] = sha256_json({k: v for k, v in payload.items() if k not in {"universe_sha256", "created_at"}})
    out = Path(output_dir)
    p_json = out / "candidate_universe_manifest.json"
    p_csv = out / "candidate_universe.csv"
    if write_artifacts:
        p_json = write_json(p_json, payload)
        p_csv = write_csv(p_csv, feasible)
    return p_json, p_csv, payload


def _prediction_freeze_paths(run_dir: str | Path, model_id: str) -> tuple[Path, Path]:
    analysis_dir = _resolve_ref(run_dir) / "models" / str(model_id) / "analysis"
    return analysis_dir, analysis_dir / "prediction_freeze_manifest.json"


def _verify_prediction_freeze_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_structured(manifest_path)
    base = manifest_path.parent
    mismatches: list[dict[str, Any]] = []
    conflict_path = base / "prediction_freeze_conflict.json"
    rejection_path = base / "prediction_freeze_rejection.json"
    if conflict_path.is_file():
        mismatches.append({
            "artifact": "prediction_freeze_conflict",
            "path": str(conflict_path),
            "reason": "prediction_freeze_conflict_present",
        })
    if rejection_path.is_file():
        rejection = _load_structured(rejection_path)
        if rejection.get("approval_eligible") is False:
            mismatches.append({
                "artifact": "prediction_freeze_rejection",
                "path": str(rejection_path),
                "reason": str(rejection.get("status") or "prediction_freeze_rejected"),
            })
    expected_files = {
        "prediction_json": (str(manifest.get("prediction_json") or "prediction.json"), str(manifest.get("prediction_sha256") or "")),
        "prediction_csv": (str(manifest.get("prediction_csv") or "predictions_frozen.csv"), str(manifest.get("prediction_csv_sha256") or "")),
        "ranking_prediction_csv": (str(manifest.get("ranking_prediction_csv") or "ranking_predictions_frozen.csv"), str(manifest.get("ranking_prediction_csv_sha256") or "")),
        "candidate_universe_manifest": (str(manifest.get("candidate_universe_manifest") or "candidate_universe_manifest.json"), ""),
    }
    universe_csv_name = str(manifest.get("candidate_universe_csv") or "").strip()
    universe_csv_expected = str(manifest.get("candidate_universe_csv_sha256") or "").strip()
    if universe_csv_name or universe_csv_expected:
        expected_files["candidate_universe_csv"] = (
            universe_csv_name or "candidate_universe.csv",
            universe_csv_expected,
        )
    resolved: dict[str, str] = {}
    for key, (name, expected) in expected_files.items():
        path = base / name
        resolved[key] = str(path)
        if not path.is_file():
            mismatches.append({"artifact": key, "path": str(path), "reason": "missing"})
            continue
        actual = sha256_file(path) or ""
        if expected and _normalise_sha256(actual) != _normalise_sha256(expected):
            mismatches.append({"artifact": key, "path": str(path), "expected": expected, "actual": actual})
    universe = _load_structured(base / expected_files["candidate_universe_manifest"][0])
    universe_hash = str(universe.get("universe_sha256") or "")
    universe_self_hash = sha256_json({
        key: value
        for key, value in universe.items()
        if key not in {"universe_sha256", "created_at"}
    }) if universe else ""
    expected_universe_hash = str(manifest.get("candidate_universe_sha256") or "")
    if not universe_hash or not universe_self_hash or _normalise_sha256(universe_hash) != _normalise_sha256(universe_self_hash):
        mismatches.append({
            "artifact": "candidate_universe_manifest",
            "reason": "universe_self_sha256_mismatch",
            "stored_universe_sha256": universe_hash,
            "computed_universe_sha256": universe_self_hash,
        })
    if not expected_universe_hash or _normalise_sha256(universe_hash) != _normalise_sha256(expected_universe_hash):
        mismatches.append({
            "artifact": "candidate_universe_manifest",
            "reason": "freeze_manifest_universe_sha256_mismatch",
            "expected_universe_sha256": expected_universe_hash,
            "actual_universe_sha256": universe_hash,
        })
    prospective = bool(manifest.get("prospective"))
    holdout_valid = bool(manifest.get("valid_for_holdout"))
    return {
        "ok": bool(manifest and prospective and holdout_valid and not mismatches),
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
        "prospective": prospective,
        "valid_for_holdout": holdout_valid,
        "universe_sha256": universe_hash,
        "universe_self_sha256": universe_self_hash,
        "mismatches": mismatches,
        "resolved_files": resolved,
    }


def create_prediction_freeze_approval(
    *,
    run_dir: str | Path,
    model_id: str,
    signer: str,
    output: str | Path | None = None,
    private_key: str | Path | None = None,
    public_key: str | Path | None = None,
) -> Path:
    """Approve a prospective model-level prediction freeze before benchmarks.

    The approval is a human attestation bound to the exact freeze manifest and
    candidate-universe hash.  A detached OpenSSL signature is optional; the
    unsigned JSON still records the named approver and its checksum.
    """
    signer = str(signer or "").strip()
    if not signer:
        raise ValueError("signer must be a non-empty person or role identifier")
    analysis_dir, manifest_path = _prediction_freeze_paths(run_dir, model_id)
    verification = _verify_prediction_freeze_manifest(manifest_path)
    if not verification.get("ok"):
        raise ValueError(f"prediction freeze is not valid for hold-out approval: {verification.get('mismatches')}")
    manifest = verification.get("manifest") if isinstance(verification.get("manifest"), Mapping) else {}
    target = _resolve_ref(output) if output else (analysis_dir / "prediction_freeze_approval.json")
    payload = {
        "schema": PREDICTION_FREEZE_APPROVAL_SCHEMA,
        "schema_version": 1,
        "created_at": now_iso(),
        "approved": True,
        "signer": signer,
        "model_id": str(model_id),
        "run_id": _resolve_ref(run_dir).name,
        "prediction_freeze_manifest": str(manifest_path),
        "prediction_freeze_manifest_sha256": verification.get("manifest_sha256"),
        "prediction_sha256": manifest.get("prediction_sha256", ""),
        "ranking_prediction_csv_sha256": manifest.get("ranking_prediction_csv_sha256", ""),
        "candidate_universe_sha256": verification.get("universe_sha256", ""),
        "freeze_status": manifest.get("freeze_status", ""),
        "prospective": bool(manifest.get("prospective")),
        "valid_for_holdout": bool(manifest.get("valid_for_holdout")),
        "attestation": "The signer reviewed this prospective prediction and candidate-universe freeze before opening or executing the corresponding hold-out benchmark measurements.",
        "signature_mode": "openssl_sha256" if private_key else "named_checksum_attestation",
    }
    payload["approval_payload_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "approval_payload_sha256"})
    approval_path = write_json(target, payload)
    write_text(approval_path.with_suffix(approval_path.suffix + ".sha256"), f"{str(sha256_file(approval_path)).replace('sha256:', '')}  {approval_path.name}\n")
    if private_key:
        key = _resolve_ref(private_key)
        if not key.is_file():
            raise FileNotFoundError(f"private key not found: {key}")
        sig = approval_path.with_suffix(approval_path.suffix + ".sig")
        subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(key), "-out", str(sig), str(approval_path)], check=True)
        if public_key:
            pub = _resolve_ref(public_key)
            if not pub.is_file():
                raise FileNotFoundError(f"public key not found: {pub}")
            shutil.copy2(pub, approval_path.with_name("prediction_freeze_public_key.pem"))
    return approval_path


def verify_prediction_freeze_approval(
    approval: str | Path,
    *,
    public_key: str | Path | None = None,
    require_cryptographic_signature: bool = False,
) -> dict[str, Any]:
    approval_path = _resolve_ref(approval)
    payload = _load_structured(approval_path)
    # Approvals may be copied into a standalone BenchmarkSet or campaign-freeze
    # archive.  The original approval intentionally records the absolute source
    # path for provenance, but that path is not portable.  Prefer it when it
    # still exists and otherwise bind to the co-located manifest.  The recorded
    # manifest SHA-256 remains the decisive identity check in both cases.
    recorded_manifest = _resolve_ref(payload.get("prediction_freeze_manifest")) if payload.get("prediction_freeze_manifest") else None
    local_manifest = approval_path.parent / "prediction_freeze_manifest.json"
    manifest_path = recorded_manifest if recorded_manifest and recorded_manifest.is_file() else local_manifest
    freeze = _verify_prediction_freeze_manifest(manifest_path)
    payload_hash_ok = str(payload.get("approval_payload_sha256") or "") == sha256_json({k: v for k, v in payload.items() if k != "approval_payload_sha256"})
    binding_ok = bool(
        str(payload.get("prediction_freeze_manifest_sha256") or "") == str(freeze.get("manifest_sha256") or "")
        and str(payload.get("candidate_universe_sha256") or "") == str(freeze.get("universe_sha256") or "")
        and payload.get("model_id") == (freeze.get("manifest") or {}).get("model_id")
    )
    signature_status = "not_present"
    sig = approval_path.with_suffix(approval_path.suffix + ".sig")
    pub = _resolve_ref(public_key) if public_key else approval_path.with_name("prediction_freeze_public_key.pem")
    if sig.is_file() and pub.is_file():
        cp = subprocess.run(
            ["openssl", "dgst", "-sha256", "-verify", str(pub), "-signature", str(sig), str(approval_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        signature_status = "valid" if cp.returncode == 0 else "invalid"
    signature_ok = signature_status == "valid" if require_cryptographic_signature else signature_status != "invalid"
    ok = bool(
        payload.get("schema") == PREDICTION_FREEZE_APPROVAL_SCHEMA
        and payload.get("approved") is True
        and str(payload.get("signer") or "").strip()
        and payload_hash_ok
        and binding_ok
        and freeze.get("ok")
        and signature_ok
    )
    return {
        "schema": "onnx-splitpoint/prediction-freeze-approval-verification",
        "schema_version": 1,
        "created_at": now_iso(),
        "ok": ok,
        "approval": str(approval_path),
        "payload_hash_ok": payload_hash_ok,
        "binding_ok": binding_ok,
        "prediction_freeze_ok": bool(freeze.get("ok")),
        "signature_status": signature_status,
        "require_cryptographic_signature": bool(require_cryptographic_signature),
        "signer": payload.get("signer", ""),
        "model_id": payload.get("model_id", ""),
        "freeze_mismatches": freeze.get("mismatches", []),
    }


def _check(checks: list[dict[str, Any]], check_id: str, ok: bool, detail: str, *, severity: str = "required", status: str | None = None, evidence: Any = None) -> None:
    checks.append({
        "id": check_id,
        "status": status or ("pass" if ok else "fail"),
        "severity": severity,
        "detail": detail,
        "evidence": evidence,
    })


def _campaign_block(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = profile.get("campaign")
    return dict(value or {}) if isinstance(value, Mapping) else {}


def resolve_campaign_claim_scope(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the scientific claim made by a campaign.

    ``evaluated_matrix`` covers only the explicitly evaluated workload and
    hardware matrix.  It deliberately does not claim that a ranking model
    generalises to unseen models.  ``ranking_generalization`` retains the
    prospective hold-out, frozen-prediction, and fitted-ranking contracts.

    Profiles predating 2.75.27 have no explicit scope.  They are interpreted
    conservatively from their existing intent: an enabled ranking validation
    or an active hold-out model retains the strict generalisation scope;
    otherwise the profile describes an evaluated matrix.  The inference source
    is reported so archived profiles are never silently presented as explicit.
    """

    campaign = _campaign_block(profile)
    explicit_scope = "claim_scope" in campaign
    raw = str(campaign.get("claim_scope") or "").strip()
    if explicit_scope:
        # Explicit values form a scientific contract, not a convenience
        # option.  Accept only the two schema spellings so a typo can never
        # silently reduce the claimed evidence boundary.
        scope = raw
        return {
            "scope": scope,
            "valid": scope in CAMPAIGN_CLAIM_SCOPES,
            "explicit": True,
            "source": "campaign.claim_scope",
            "raw": raw,
        }

    ranking = (
        profile.get("ranking_validation")
        if isinstance(profile.get("ranking_validation"), Mapping)
        else {}
    )
    has_holdout = any(
        is_confirmatory_holdout(row.get("evaluation_role"))
        for row in _model_entries(profile)
    )
    ranking_intent = bool(ranking.get("enabled") or has_holdout)
    return {
        "scope": (
            RANKING_GENERALIZATION_CLAIM_SCOPE
            if ranking_intent
            else EVALUATED_MATRIX_CLAIM_SCOPE
        ),
        "valid": True,
        "explicit": False,
        "source": (
            "legacy_ranking_or_holdout_intent"
            if ranking_intent
            else "legacy_evaluated_matrix_intent"
        ),
        "raw": "",
    }


def campaign_references(profile: Mapping[str, Any], profile_path: str | Path | None = None) -> dict[str, Path]:
    base = Path(profile_path).parent if profile_path else Path.cwd()
    campaign = _campaign_block(profile)
    refs: dict[str, Path] = {}
    datasets = campaign.get("dataset_manifests") if isinstance(campaign.get("dataset_manifests"), Mapping) else {}
    for task, block in datasets.items():
        if not isinstance(block, Mapping):
            continue
        for role, value in block.items():
            if value:
                refs[f"dataset_{task}_{role}"] = _resolve_ref(value, base)
    for key in (
        "pipeline_contract_manifest",
        "holdout_registry",
        "ranking_model_bundle",
        "energy_calibration_manifest",
        "campaign_freeze_artifact",
    ):
        value = campaign.get(key)
        if value:
            refs[key] = _resolve_ref(value, base)
    protocol = campaign.get("protocol_freeze") if isinstance(campaign.get("protocol_freeze"), Mapping) else {}
    protocol_artifact = protocol.get("artifact") or protocol.get("path")
    if protocol_artifact:
        refs["protocol_freeze_artifact"] = _resolve_ref(protocol_artifact, base)
    protocol_manifests = protocol.get("manifests") if isinstance(protocol.get("manifests"), Mapping) else {}
    for kind in ("candidate", "dag", "prediction", "policy", "energy"):
        value = protocol_manifests.get(kind) or protocol_manifests.get(f"{kind}_manifest") or protocol.get(f"{kind}_manifest")
        if isinstance(value, Mapping):
            value = value.get("path") or value.get("artifact")
        if value:
            refs[f"protocol_{kind}_manifest"] = _resolve_ref(value, base)
    return refs


def build_campaign_readiness(profile: Mapping[str, Any], *, profile_path: str | Path | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    campaign = _campaign_block(profile)
    final_mode = str(campaign.get("mode") or "development").lower() == "final"
    enforcement = str(campaign.get("enforcement") or ("strict" if final_mode else "warn")).lower()
    claim_scope = resolve_campaign_claim_scope(profile)
    claim_scope_value = str(claim_scope.get("scope") or "")
    campaign_id = str(
        campaign.get("id")
        or campaign.get("campaign_id")
        or profile.get("name")
        or ""
    ).strip()
    thesis_evaluated_matrix_contract_required = bool(
        claim_scope.get("explicit")
        and claim_scope_value == EVALUATED_MATRIX_CLAIM_SCOPE
        and campaign_id == "thesis_final_evaluated_matrix_v1"
    )
    # An invalid scope must not accidentally disable the stronger gates.
    ranking_generalization_required = bool(
        not claim_scope.get("valid")
        or claim_scope_value == RANKING_GENERALIZATION_CLAIM_SCOPE
    )
    integrity = profile.get("integrity_policy") if isinstance(profile.get("integrity_policy"), Mapping) else {}
    integrity_mode = str(integrity.get("effective_mode") or integrity.get("mode") or ("strict" if final_mode else "fast")).lower()
    strict_content_verification = bool(final_mode or integrity_mode == "strict")
    relaxed_sample_size = 0 if strict_content_verification else max(1, int(integrity.get("dataset_sample_size") or integrity.get("dataset_sample_count") or 24))
    _check(checks, "campaign_mode", bool(final_mode), "Campaign mode is final." if final_mode else "Campaign remains development/screening.", severity="informational" if not final_mode else "required", status="pass" if final_mode else "development")

    def _final_requirement(check_id: str, condition: bool, detail: str, *, evidence: Any = None) -> None:
        """Record a final-campaign requirement without pretending it passed in development mode."""
        if final_mode:
            _check(checks, check_id, bool(condition), detail, severity="required", evidence=evidence)
        else:
            _check(
                checks,
                check_id,
                bool(condition),
                detail,
                severity="required_for_final",
                status="pass" if condition else "deferred",
                evidence=evidence,
            )

    def _not_applicable(check_id: str, detail: str, *, evidence: Any = None) -> None:
        _check(
            checks,
            check_id,
            True,
            detail,
            severity="informational",
            status="not_applicable",
            evidence=evidence,
        )

    _final_requirement(
        "claim_scope_valid",
        bool(claim_scope.get("valid")),
        "Campaign claim_scope is evaluated_matrix or ranking_generalization.",
        evidence=dict(claim_scope),
    )
    _check(
        checks,
        "claim_scope",
        bool(claim_scope.get("valid")),
        (
            "The final claim is limited to the explicitly evaluated workload/hardware matrix."
            if claim_scope_value == EVALUATED_MATRIX_CLAIM_SCOPE
            else "The campaign additionally claims prospective ranking generalisation to held-out models."
            if claim_scope_value == RANKING_GENERALIZATION_CLAIM_SCOPE
            else f"Unknown campaign claim scope: {claim_scope_value or '<empty>'}."
        ),
        severity="informational",
        status=("pass" if claim_scope.get("valid") else "fail"),
        evidence=dict(claim_scope),
    )

    frozen = bool(campaign.get("frozen_before_final_campaign") or (profile.get("quality_gate") or {}).get("frozen_before_final_campaign"))
    _final_requirement("profile_frozen", frozen, "Profile and task-quality policy are marked frozen before the final campaign.")

    protocol_cfg = campaign.get("protocol_freeze") if isinstance(campaign.get("protocol_freeze"), Mapping) else {}
    protocol_required = bool(campaign.get("require_protocol_freeze")) or bool(
        ranking_generalization_required and claim_scope.get("explicit")
    )
    protocol_configured = bool(protocol_cfg.get("artifact") or protocol_cfg.get("path"))
    if protocol_required or protocol_configured:
        protocol_check = verify_configured_protocol_freeze(profile, profile_path=profile_path)
        _final_requirement(
            "protocol_freeze_integrity",
            bool(protocol_check.get("ok")),
            "The versioned protocol freeze matches model roles and the candidate, DAG, prediction, policy, and energy manifests.",
            evidence=protocol_check,
        )
        _final_requirement(
            "protocol_freeze_version",
            bool(str(protocol_cfg.get("version") or "").strip() and str(protocol_cfg.get("amendment") or "").strip()),
            "The protocol freeze declares an explicit version and amendment identifier.",
            evidence={"version": protocol_cfg.get("version"), "amendment": protocol_cfg.get("amendment")},
        )
    elif ranking_generalization_required:
        _check(
            checks,
            "protocol_freeze_integrity",
            True,
            "Legacy inferred ranking-generalization profile retains its historical contract; write an explicit claim_scope to opt into the v2.75.27 five-manifest protocol gate.",
            severity="informational",
            status="legacy_compatible",
            evidence=dict(claim_scope),
        )
    else:
        _not_applicable(
            "protocol_freeze_integrity",
            "A prospective five-manifest protocol freeze is optional for evaluated_matrix unless explicitly requested.",
        )

    refs = campaign_references(profile, profile_path)
    freeze_path = refs.get("campaign_freeze_artifact")
    freeze_required = bool(campaign.get("require_campaign_freeze"))
    freeze_configured = bool(freeze_path)
    if freeze_configured and freeze_path:
        freeze_verification = verify_campaign_freeze(freeze_path)
        _check(
            checks,
            "campaign_freeze_integrity",
            bool(freeze_verification.get("ok")),
            "The configured post-run campaign archive is complete and hash-consistent.",
            severity="post_run_required",
            status=("completed" if freeze_verification.get("ok") else "post_run_invalid"),
            evidence=freeze_verification,
        )
    elif freeze_required:
        # create_campaign_freeze() calls this preflight before it can write the
        # archive.  Therefore the archive is a completion requirement, not a
        # circular pre-run gate.
        _check(
            checks,
            "campaign_freeze_artifact",
            True,
            "A signed campaign archive must be created after the run; it is recorded as a completion requirement and is not a preflight prerequisite.",
            severity="post_run_required",
            status="post_run_required",
            evidence={"claim_scope": claim_scope_value},
        )
    else:
        _not_applicable(
            "campaign_freeze_artifact",
            "No separate post-run campaign archive is required by this profile.",
            evidence={"claim_scope": claim_scope_value},
        )
    dataset_verification_mode = "full" if strict_content_verification else ("manifest_only" if integrity_mode in {"off", "none", "manifest_only"} else "sampled")
    manifests: list[dict[str, Any]] = []
    for task in ("classification", "detection"):
        for role in ("calibration", "validation"):
            key = f"dataset_{task}_{role}"
            path = refs.get(key)
            exists = bool(path and path.is_file())
            _final_requirement(key, exists, f"{task} {role} dataset manifest {'found' if exists else 'missing'}: {path or ''}")
            if exists and path:
                data = _load_structured(path)
                schema_ok = data.get("schema") == DATASET_MANIFEST_SCHEMA and str(data.get("task")) == task and str(data.get("role")) == role
                content_ok = str(data.get("hash_mode")) == "content" and bool(data.get("items_identity_sha256")) and int(data.get("item_count") or 0) > 0
                verification = verify_dataset_manifest(data, verify_files=True, sample_size=relaxed_sample_size, verification_mode=dataset_verification_mode)
                _final_requirement(key + "_schema", schema_ok, "Dataset manifest task/role/schema match.")
                _final_requirement(key + "_content_hash", content_ok, "Final manifests use per-item content hashes and a non-empty identity hash.")
                _final_requirement(
                    key + "_current_files",
                    bool(verification.get("ok")),
                    f"Dataset manifest verification passed in {dataset_verification_mode} mode.",
                    evidence=verification,
                )
                manifests.append(data)
    if manifests:
        separation = validate_calibration_validation_separation(manifests)
        _final_requirement("calibration_validation_disjoint", bool(separation.get("ok")), "Calibration and validation sets are content- and ID-disjoint.", evidence=separation)

    contract_path = refs.get("pipeline_contract_manifest")
    contract_ok = bool(contract_path and contract_path.is_file())
    contract_verification: dict[str, Any] = {}
    _final_requirement("pipeline_contract_manifest", contract_ok, f"Preprocessing/decoder/NMS contract manifest {'found' if contract_ok else 'missing'}: {contract_path or ''}")
    if contract_ok and contract_path:
        contract = _load_structured(contract_path)
        verification = verify_pipeline_contract_manifest(
            contract,
            require_locked=final_mode,
            expected_claim_scope=(
                EVALUATED_MATRIX_CLAIM_SCOPE
                if thesis_evaluated_matrix_contract_required
                else None
            ),
            manifest_path=contract_path,
        )
        contract_verification = dict(verification)
        _final_requirement(
            "pipeline_contract_hashes",
            bool(verification.get("ok")),
            "Pipeline contract files still match their hashes and cover preprocessing, decoder, and NMS.",
            evidence=verification,
        )
        _final_requirement("pipeline_contract_locked", bool(verification.get("locked_ok")), "All preprocessing, decoder, and NMS configurations are explicitly locked before the final campaign.")

    models = _model_entries(profile)
    if thesis_evaluated_matrix_contract_required:
        _final_requirement(
            "evaluated_matrix_exact_model_ids",
            [str(row.get("id") or "") for row in models]
            == ["resnet50", "yolo26s", "yolov7_paper"],
            "The thesis evaluated matrix contains exactly resnet50, yolo26s and yolov7_paper in the frozen order.",
            evidence={
                "model_ids": [str(row.get("id") or "") for row in models]
            },
        )
    development = [x for x in models if normalize_evaluation_role(x.get("evaluation_role")) == "development"]
    holdout = [x for x in models if is_confirmatory_holdout(x.get("evaluation_role"))]
    _final_requirement("development_models_present", bool(development), "At least one development model is declared.")
    if ranking_generalization_required:
        _final_requirement("holdout_models_present", bool(holdout), "At least one model-level hold-out is declared.")
        _final_requirement(
            "holdout_adapter_sources_frozen",
            bool(
                not holdout
                or contract_verification.get("adapter_source_coverage_ok")
            ),
            "Confirmatory hold-outs require at least one locked adapter contract whose implementation source files are content-hashed before outcomes are opened.",
            evidence={
                "holdout_model_ids": [str(row.get("id") or "") for row in holdout],
                "adapter_contract_count": contract_verification.get(
                    "adapter_contract_count", 0,
                ),
                "adapter_implementation_source_count": contract_verification.get(
                    "adapter_implementation_source_count", 0,
                ),
            },
        )
    else:
        _not_applicable(
            "holdout_models_present",
            "No model-level hold-out is required because the claim is restricted to the evaluated matrix.",
            evidence={"active_holdout_model_ids": [str(row.get("id") or "") for row in holdout]},
        )
        _not_applicable(
            "holdout_adapter_sources_frozen",
            "Prospective hold-out adapter freezing belongs to ranking_generalization, not evaluated_matrix.",
        )
    ids_dev = {str(x.get("id")) for x in development}
    ids_ho = {str(x.get("id")) for x in holdout}
    _check(checks, "model_role_disjoint", not bool(ids_dev & ids_ho), "Development and hold-out model IDs are disjoint.", evidence={"overlap": sorted(ids_dev & ids_ho)})
    model_ids = [str(x.get("id") or "") for x in models]
    _final_requirement("model_ids_unique", len(model_ids) == len(set(model_ids)), "Every campaign model has a unique model ID.", evidence={"model_ids": model_ids})
    profile_base = Path(profile_path).parent if profile_path else Path.cwd()
    dev_family_ids = {str(x.get("family_id") or "").strip() for x in development if str(x.get("family_id") or "").strip()}
    ranking_cfg_for_audit = profile.get("ranking_validation") if isinstance(profile.get("ranking_validation"), Mapping) else {}
    global_min_valid = max(1, int(ranking_cfg_for_audit.get("minimum_valid_audit_candidates") or 10))
    for row in models:
        model_id = str(row.get("id") or "")
        role = normalize_evaluation_role(row.get("evaluation_role"))
        family_id = str(row.get("family_id") or "").strip()
        scope = str(row.get("generalization_scope") or "").strip().lower()
        validation_tier = str(row.get("validation_tier") or "").strip().lower()
        _final_requirement(
            f"model_role_explicit_{model_id}",
            role in {"development", CONFIRMATORY_HOLDOUT_ROLE},
            f"Model {model_id} explicitly declares evaluation_role=development or confirmatory_holdout.",
        )
        if ranking_generalization_required:
            _final_requirement(f"model_family_id_explicit_{model_id}", bool(family_id), f"Model {model_id} explicitly declares a stable family_id.")
            _final_requirement(f"model_generalization_scope_explicit_{model_id}", bool(scope), f"Model {model_id} explicitly declares its generalization_scope.")
        else:
            _not_applicable(
                f"model_family_id_explicit_{model_id}",
                f"Model-family transfer is outside the evaluated_matrix claim for {model_id}.",
            )
            _not_applicable(
                f"model_generalization_scope_explicit_{model_id}",
                f"A generalisation scope is not consumed by the evaluated_matrix claim for {model_id}.",
            )
        _final_requirement(f"model_validation_tier_explicit_{model_id}", validation_tier in {"screening", "final"}, f"Model {model_id} explicitly declares validation_tier.")
        _final_requirement(f"model_validation_tier_final_{model_id}", validation_tier == "final", f"Model {model_id} uses the final validation tier in a final campaign.")

        if role == "development" and ranking_generalization_required:
            development_scope_ok = scope == "development" or (
                str(row.get("suite_tier") or "").strip().lower() == "reserve" and scope == "stress_test"
            )
            _final_requirement(
                f"development_scope_{model_id}",
                development_scope_ok,
                f"Development model {model_id} declares generalization_scope=development; an enabled reserve may instead declare stress_test.",
            )
        elif role == CONFIRMATORY_HOLDOUT_ROLE and ranking_generalization_required:
            valid_holdout_scope = scope in {"model_family_holdout", "within_family_transfer"}
            _final_requirement(f"holdout_scope_{model_id}", valid_holdout_scope, f"Hold-out {model_id} declares model_family_holdout or within_family_transfer.")
            if scope == "model_family_holdout":
                _final_requirement(
                    f"family_holdout_disjoint_{model_id}",
                    bool(family_id and family_id not in dev_family_ids),
                    f"Family hold-out {model_id} must use a family_id absent from all development models.",
                    evidence={"family_id": family_id, "development_family_ids": sorted(dev_family_ids)},
                )
            elif scope == "within_family_transfer":
                _final_requirement(
                    f"within_family_anchor_{model_id}",
                    bool(family_id and family_id in dev_family_ids),
                    f"Within-family hold-out {model_id} must have a development anchor with the same family_id.",
                    evidence={"family_id": family_id, "development_family_ids": sorted(dev_family_ids)},
                )
            att = row.get("unseen_attestation") if isinstance(row.get("unseen_attestation"), Mapping) else {}
            attested = bool(att.get("unseen") is True and str(att.get("attested_by") or "").strip() and str(att.get("attested_at") or "").strip())
            _final_requirement(f"holdout_attestation_{model_id}", attested, f"Hold-out {model_id} has an explicit prospective attestation.")

        model_ref = row.get("path") or row.get("model_path") or row.get("onnx")
        model_path = _resolve_ref(model_ref, profile_base) if model_ref else None
        actual_model_hash = sha256_file(model_path) if model_path and model_path.is_file() else ""
        declared_model_hash = str(row.get("model_sha256") or "").strip()
        identity_ok = bool(model_path and model_path.is_file() and actual_model_hash)
        if declared_model_hash:
            identity_ok = identity_ok and _sha256_equal(declared_model_hash, actual_model_hash)
        _final_requirement(
            f"model_identity_{model_id}",
            identity_ok,
            f"Model {model_id} resolves to an exact local model artefact with a content hash.",
            evidence={
                "model_path": str(model_path or ""),
                "model_sha256": actual_model_hash,
                "declared_model_sha256": declared_model_hash,
                "exists": bool(model_path and model_path.is_file()),
            },
        )

        universe = row.get("candidate_universe") if isinstance(row.get("candidate_universe"), Mapping) else {}
        mode = str(universe.get("mode") or "").lower()
        configured = mode in {"all_feasible", "deterministic_audit", "audit_universe", "declared_shortlist"}
        _final_requirement(f"candidate_universe_{model_id}", configured, f"Model {model_id} explicitly declares its candidate-universe mode.", evidence=dict(universe))
        if role == CONFIRMATORY_HOLDOUT_ROLE and ranking_generalization_required:
            auditable = mode in {"all_feasible", "deterministic_audit", "audit_universe"}
            min_valid = max(1, int(universe.get("minimum_valid_candidates") or global_min_valid))
            try:
                audit_size = int(universe.get("audit_size") or 0)
            except Exception:
                audit_size = 0
            size_contract_ok = mode == "all_feasible" or audit_size >= min_valid
            _final_requirement(f"holdout_candidate_universe_{model_id}", auditable, f"Hold-out {model_id} declares all_feasible or deterministic_audit candidate coverage.", evidence=dict(universe))
            _final_requirement(
                f"holdout_audit_size_{model_id}",
                bool(auditable and size_contract_ok),
                f"Hold-out {model_id} audit can yield at least {min_valid} valid candidates.",
                evidence={"mode": mode, "audit_size": audit_size, "minimum_valid_candidates": min_valid},
            )

    registry_path = refs.get("holdout_registry")
    registry_ok = bool(registry_path and registry_path.is_file())
    if ranking_generalization_required:
        _final_requirement("holdout_registry", registry_ok, f"Hold-out registry {'found' if registry_ok else 'missing'}: {registry_path or ''}")
        if registry_ok and registry_path:
            verification = verify_holdout_registry(
                _load_structured(registry_path),
                profile=profile,
                profile_path=profile_path,
                require_attestation=final_mode,
            )
            _final_requirement(
                "holdout_registry_integrity",
                bool(verification.get("ok")),
                "Hold-out registry matches the current source profile, model hashes, roles and attestations.",
                evidence=verification,
            )
    else:
        _not_applicable(
            "holdout_registry",
            "A hold-out registry is outside the evaluated_matrix claim scope.",
            evidence={"configured_path": str(registry_path or "")},
        )
        _not_applicable(
            "holdout_registry_integrity",
            "No hold-out registry is consumed for evaluated_matrix claims.",
        )

    ranking_cfg = profile.get("ranking_validation") if isinstance(profile.get("ranking_validation"), Mapping) else {}
    ranking_enabled = (
        bool(ranking_cfg.get("enabled"))
        if "enabled" in ranking_cfg
        else bool(ranking_generalization_required and not claim_scope.get("explicit"))
    )
    model_bundle = refs.get("ranking_model_bundle")
    require_stage_fit = bool(campaign.get("require_fitted_stage_time", True if final_mode and ranking_generalization_required else False))
    require_native_fit = bool(campaign.get("require_native_handover_model", True if final_mode and ranking_generalization_required else False))
    no_handover_cfg = ranking_cfg.get("cycle_time_no_handover") if isinstance(ranking_cfg.get("cycle_time_no_handover"), Mapping) else {}
    stage_inline = bool(no_handover_cfg.get("backend_throughput_gops") or no_handover_cfg.get("stage_time_models"))
    bundle_ok = bool(model_bundle and model_bundle.is_file())
    bundle_verification: dict[str, Any] = {}
    if bundle_ok and model_bundle:
        bundle_verification = validate_ranking_model_bundle(
            _load_structured(model_bundle),
            development_model_ids=sorted(ids_dev),
            holdout_model_ids=sorted(ids_ho),
            require_native_handover=require_native_fit,
        )
    if ranking_generalization_required:
        _final_requirement(
            "ranking_validation_enabled",
            ranking_enabled,
            "ranking_generalization requires ranking_validation.enabled=true.",
        )
        _final_requirement(
            "ranking_model_bundle_integrity",
            bool(bundle_verification.get("ok")),
            "The frozen ranking-model bundle is development-only, hold-out-disjoint and contains fitted parameters.",
            evidence=bundle_verification,
        )
        _final_requirement("stage_time_model", (stage_inline or bool(bundle_verification.get("fitted_stage_model_count")) or not require_stage_fit), "Development-fitted stage-time parameters are present inline or in a verified frozen model bundle.")
    else:
        _not_applicable(
            "ranking_validation_enabled",
            "Ranking validation is optional diagnostic output for evaluated_matrix and cannot enlarge the final claim.",
            evidence={"ranking_validation_enabled": ranking_enabled},
        )
        _not_applicable(
            "ranking_model_bundle_integrity",
            "A fitted ranking bundle is required only for ranking_generalization.",
            evidence={"configured_path": str(model_bundle or "")},
        )
        _not_applicable(
            "stage_time_model",
            "A development-fitted ranking cost model is not a blocker for measured evaluated_matrix results.",
        )
    handovers = ((ranking_cfg.get("cycle_time_with_handover") or {}).get("handover_models")) if isinstance(ranking_cfg.get("cycle_time_with_handover"), Mapping) else {}
    native_model = bool(isinstance(handovers, Mapping) and isinstance(handovers.get("native_fifo"), Mapping) and handovers.get("native_fifo"))
    if ranking_generalization_required:
        _final_requirement("native_handover_model", (native_model or bool(bundle_verification.get("fitted_native_handover_model_count")) or not require_native_fit), "A development-fitted Native FIFO handover model is present or explicitly not required.")
    else:
        _not_applicable(
            "native_handover_model",
            "Measured Native FIFO performance remains required, but a fitted handover predictor is not needed for evaluated_matrix claims.",
        )

    native = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    native_enabled = bool(native.get("enabled"))
    native_full = native.get("full_baselines") if isinstance(native.get("full_baselines"), Mapping) else {}
    native_full_ok = bool(native_full.get("enabled")) and bool(native_full.get("same_runtime_contract", True))
    _final_requirement("native_full_baselines", (not native_enabled) or native_full_ok, "Native full baselines are enabled and use the same runtime contract whenever Native FIFO claims are enabled.")

    top_energy_declared = isinstance(profile.get("energy"), Mapping) and bool(profile.get("energy"))
    native_energy_declared = isinstance(native.get("energy"), Mapping) and bool(native.get("energy"))
    if top_energy_declared or native_energy_declared:
        effective_energy = resolve_effective_energy_config(profile)
        energy_lifecycle = effective_energy.get("lifecycle") if isinstance(effective_energy.get("lifecycle"), Mapping) else {}
        _final_requirement(
            "effective_energy_configuration",
            bool(energy_lifecycle.get("requested") and energy_lifecycle.get("configured") and not effective_energy.get("configuration_errors")),
            "One effective energy request is configured without conflicting generic/native enablement.",
            evidence=effective_energy,
        )
        energy_ab = effective_energy.get("window_method_ab") if isinstance(effective_energy.get("window_method_ab"), Mapping) else {}
        _final_requirement(
            "energy_window_method_ab",
            bool(
                energy_ab.get("enabled")
                and energy_ab.get("valid")
                and energy_ab.get("primary_method") == "command_marker_window"
                and energy_ab.get("shadow_method") == "chapter4_legacy_window"
                and energy_ab.get("same_raw_capture")
                and energy_ab.get("mode") == "shadow"
                and not energy_ab.get("auto_switch")
                and not energy_ab.get("requires_picoscope")
            ),
            "The command-marker window is frozen as the scientific primary. The Chapter-4 legacy window is a same-trace, non-blocking sensitivity shadow; it never auto-switches, cannot invalidate the primary claim and requires no PicoScope rerun.",
            evidence=energy_ab,
        )
    else:
        _check(
            checks,
            "effective_energy_configuration",
            True,
            "No explicit effective energy block is declared; legacy measurement_campaign checks remain authoritative.",
            severity="informational",
            status="legacy_compatible",
        )

    measurement = profile.get("measurement_campaign") if isinstance(profile.get("measurement_campaign"), Mapping) else {}
    system_power = measurement.get("system_power") if isinstance(measurement.get("system_power"), Mapping) else {}
    scope_ok = str(system_power.get("scope") or "").upper() == "FS"
    raw_primary = bool(system_power.get("raw_primary"))
    window_value = str(system_power.get("window") or system_power.get("measurement_window") or "").strip().lower().replace("-", "_")
    window_ok = window_value in {"command", "command_energy", "command_window"}
    repeats = int(system_power.get("repeats") or 0)
    confidence_level = float(system_power.get("confidence_level") or 0.0)
    randomize_run_order = bool(system_power.get("randomize_run_order"))
    randomization_seed_value = _number(system_power.get("randomization_seed"))
    randomization_seed_ok = bool(randomization_seed_value is not None and float(randomization_seed_value).is_integer())
    idle_norm = system_power.get("idle_normalization") if isinstance(system_power.get("idle_normalization"), Mapping) else {}
    idle_ok = (not bool(idle_norm.get("enabled"))) or (bool(idle_norm.get("report_raw_and_normalized")) and str(idle_norm.get("method") or "") in {"paired_fs_rail_on_off", "physical_accelerator_off_run"})
    _final_requirement("full_system_power_scope", scope_ok and raw_primary, "Final energy campaign uses raw full-system input energy as the primary metric.")
    _final_requirement("energy_command_window", window_ok, "Final full-system energy is bound to the command window.", evidence={"window": system_power.get("window") or system_power.get("measurement_window")})
    _final_requirement("energy_repetitions", repeats >= 3, "Full-system energy uses at least three independent repetitions.", evidence={"repeats": repeats})
    _final_requirement("energy_confidence_interval", confidence_level >= 0.95, "Repeated energy windows use a pre-declared confidence level of at least 95%.", evidence={"confidence_level": confidence_level})
    _final_requirement("energy_run_order", randomize_run_order and randomization_seed_ok, "Final energy target blocks use a recorded deterministic random order with an explicit integer seed.", evidence={"randomize_run_order": randomize_run_order, "randomization_seed": system_power.get("randomization_seed")})
    _final_requirement("idle_normalization_policy", idle_ok, "Idle normalization reports raw and normalized values and uses a scope-compatible paired method.")
    calibration_path = refs.get("energy_calibration_manifest")
    calibration_ok = bool(calibration_path and calibration_path.is_file())
    _final_requirement("energy_calibration_manifest", calibration_ok, f"Full-system input-channel calibration manifest {'found' if calibration_ok else 'missing'}: {calibration_path or ''}")
    if calibration_ok and calibration_path:
        verification = verify_energy_calibration_manifest(
            _load_structured(calibration_path),
            require_final=final_mode,
            expected_channel_bindings=(
                FINAL_MATRIX_SETUP_BINDINGS
                if claim_scope_value == EVALUATED_MATRIX_CLAIM_SCOPE
                else None
            ),
        )
        _final_requirement(
            "energy_calibration_integrity",
            bool(verification.get("ok")),
            "Full-system input energy is bound either to a locked direct calibration or to the unchanged, validated Wachsmuth measurement method; setup, IP, channel, sample rate and implementation hashes are consistent.",
            evidence=verification,
        )

    failures = [x for x in checks if x.get("status") == "fail" and x.get("severity") == "required"]
    deferred = [x for x in checks if x.get("status") == "deferred"]
    post_run_pending = [
        x for x in checks
        if x.get("status") in {"post_run_required", "post_run_invalid"}
    ]
    if final_mode:
        status = "final_ready" if not failures else ("blocked" if enforcement == "strict" else "incomplete")
    else:
        status = "development_ready"
    return {
        "schema": CAMPAIGN_READINESS_SCHEMA,
        "schema_version": 1,
        "created_at": now_iso(),
        "campaign_id": str(campaign.get("id") or campaign.get("campaign_id") or profile.get("name") or "campaign"),
        "mode": "final" if final_mode else "development",
        "enforcement": enforcement,
        "claim_scope": claim_scope_value,
        "claim_scope_valid": bool(claim_scope.get("valid")),
        "claim_scope_explicit": bool(claim_scope.get("explicit")),
        "claim_scope_source": str(claim_scope.get("source") or ""),
        "ranking_generalization_required": ranking_generalization_required,
        "status": status,
        # `ready` is intentionally reserved for a final, frozen campaign.
        "ready": status == "final_ready",
        "final_ready": status == "final_ready",
        "development_ready": status == "development_ready",
        "required_failure_count": len(failures),
        "deferred_final_requirement_count": len(deferred),
        "post_run_requirement_count": len(post_run_pending),
        "checks": checks,
        "references": {key: str(value) for key, value in refs.items()},
        "profile_sha256": sha256_json(profile),
    }


def readiness_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Final campaign readiness",
        "",
        f"- Campaign: `{report.get('campaign_id')}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Enforcement: `{report.get('enforcement')}`",
        f"- Claim scope: `{report.get('claim_scope', 'unknown')}`",
        f"- Status: **{report.get('status')}**",
        f"- Required failures: **{report.get('required_failure_count', 0)}**",
        f"- Deferred final requirements: **{report.get('deferred_final_requirement_count', 0)}**",
        f"- Pending post-run requirements: **{report.get('post_run_requirement_count', 0)}**",
        "",
        "| Check | Status | Severity | Detail |",
        "|---|---|---|---|",
    ]
    for row in list(report.get("checks") or []):
        if not isinstance(row, Mapping):
            continue
        detail = str(row.get("detail") or "").replace("|", "\\|")
        lines.append(f"| `{row.get('id')}` | {row.get('status')} | {row.get('severity')} | {detail} |")
    lines.extend(["", "## Interpretation", "", "`development_ready` means the development workflow can run while final-only requirements remain explicitly deferred. `final_ready` validates the frozen software-side protocol and referenced hashes. Neither status manufactures missing measurements, proves that a human hold-out attestation is truthful, or replaces the final hardware campaign.", ""])
    return "\n".join(lines)


def materialize_campaign_inputs(profile: Mapping[str, Any], *, profile_path: str | Path | None, output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for key, source in campaign_references(profile, profile_path).items():
        if not source.is_file():
            continue
        dst = out / f"{key}{source.suffix.lower()}"
        shutil.copy2(source, dst)
        records.append({"id": key, "source": str(source), "copied_path": dst.name, "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size})
    payload = {
        "schema": "onnx-splitpoint/campaign-inputs",
        "schema_version": 1,
        "created_at": now_iso(),
        "profile_sha256": sha256_json(profile),
        "artifacts": records,
        "artifact_set_sha256": sha256_json(records),
    }
    write_json(out / "campaign_inputs_manifest.json", payload)
    return payload


def _zip_tree(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(source).as_posix())


def create_campaign_freeze(
    *,
    profile: str | Path,
    output: str | Path,
    run_dir: str | Path | None = None,
    signer: str = "",
    private_key: str | Path | None = None,
    public_key: str | Path | None = None,
) -> Path:
    profile_path = _resolve_ref(profile)
    profile_payload = _load_structured(profile_path)
    if not profile_payload:
        raise ValueError(f"could not load profile: {profile_path}")
    out = _resolve_ref(output)
    with tempfile.TemporaryDirectory(prefix="splitpoint_campaign_freeze_") as tmp:
        root = Path(tmp)
        inputs = root / "campaign_inputs"
        inputs.mkdir()
        shutil.copy2(profile_path, root / "profile.yaml")
        materialize_campaign_inputs(profile_payload, profile_path=profile_path, output_dir=inputs)
        readiness = build_campaign_readiness(profile_payload, profile_path=profile_path)
        write_json(root / "campaign_readiness.json", readiness)
        write_text(root / "campaign_readiness.md", readiness_markdown(readiness))
        final_mode = str((_campaign_block(profile_payload).get("mode") or "development")).lower() == "final"
        if final_mode and not readiness.get("ready"):
            raise ValueError(
                f"final campaign freeze refused because preflight is not ready: "
                f"{readiness.get('required_failure_count', 0)} required failures"
            )
        if final_mode and not str(signer or "").strip():
            raise ValueError("final campaign freeze requires a non-empty --signer attestation")
        campaign_cfg = _campaign_block(profile_payload)
        claim_scope = resolve_campaign_claim_scope(profile_payload)
        generalization_claim = bool(
            claim_scope.get("valid")
            and claim_scope.get("scope") == RANKING_GENERALIZATION_CLAIM_SCOPE
        )
        declared_holdouts = (
            [row for row in _model_entries(profile_payload) if is_confirmatory_holdout(row.get("evaluation_role"))]
            if generalization_claim
            else []
        )
        prediction_approval_required = bool(
            generalization_claim
            and campaign_cfg.get("require_prediction_freeze_approval")
        )
        if final_mode and declared_holdouts and not run_dir:
            raise ValueError("final campaign freeze requires --run-dir so that every declared hold-out prediction freeze can be archived")
        included_predictions: list[dict[str, Any]] = []
        prediction_checks: list[dict[str, Any]] = []
        if run_dir:
            rr = _resolve_ref(run_dir)
            for manifest in sorted(rr.glob("models/*/analysis/prediction_freeze_manifest.json")):
                model_id = manifest.parents[1].name
                dst_dir = root / "prediction_freezes" / model_id
                dst_dir.mkdir(parents=True, exist_ok=True)
                copied: list[dict[str, Any]] = []
                data = _load_structured(manifest)
                freeze_check = _verify_prediction_freeze_manifest(manifest)
                approval_required = bool(
                    is_confirmatory_holdout(data.get("evaluation_role"))
                    and prediction_approval_required
                )
                approval_path = manifest.parent / "prediction_freeze_approval.json"
                approval_check = verify_prediction_freeze_approval(
                    approval_path,
                    public_key=(manifest.parent / "prediction_freeze_public_key.pem") if (manifest.parent / "prediction_freeze_public_key.pem").is_file() else None,
                    require_cryptographic_signature=bool(_campaign_block(profile_payload).get("require_cryptographic_prediction_signature")),
                ) if approval_path.is_file() else {"ok": not approval_required, "signature_status": "missing"}
                prediction_checks.append({
                    "model_id": model_id,
                    "evaluation_role": data.get("evaluation_role", ""),
                    "freeze_ok": bool(freeze_check.get("ok")),
                    "approval_required": approval_required,
                    "approval_ok": bool(approval_check.get("ok")),
                    "signature_status": approval_check.get("signature_status", ""),
                    "mismatches": freeze_check.get("mismatches", []),
                })
                names = [
                    manifest.name,
                    str(data.get("prediction_csv") or "predictions_frozen.csv"),
                    str(data.get("ranking_prediction_csv") or "ranking_predictions_frozen.csv"),
                    "candidate_universe_manifest.json",
                    "candidate_universe.csv",
                    "prediction_freeze_approval.json",
                    "prediction_freeze_approval.json.sha256",
                    "prediction_freeze_approval.json.sig",
                    "prediction_freeze_public_key.pem",
                    "prediction_freeze_approval_verification.json",
                ]
                for name in names:
                    src = manifest.parent / name
                    if src.is_file():
                        dst = dst_dir / src.name
                        shutil.copy2(src, dst)
                        copied.append({"path": f"prediction_freezes/{model_id}/{dst.name}", "sha256": sha256_file(dst)})
                included_predictions.append({"model_id": model_id, "files": copied, "valid_for_holdout": bool(data.get("valid_for_holdout")), "freeze_status": data.get("freeze_status")})
        if final_mode:
            checked_holdout_ids = {str(row.get("model_id") or "") for row in prediction_checks if is_confirmatory_holdout(row.get("evaluation_role"))}
            missing_holdout_freezes = sorted(str(row.get("id") or "") for row in declared_holdouts if str(row.get("id") or "") not in checked_holdout_ids)
            if missing_holdout_freezes:
                raise ValueError(f"final campaign freeze refused because prediction freezes are missing for hold-outs: {missing_holdout_freezes}")
            invalid_holdouts = [
                row for row in prediction_checks
                if is_confirmatory_holdout(row.get("evaluation_role"))
                and (not row.get("freeze_ok") or (row.get("approval_required") and not row.get("approval_ok")))
            ]
            if invalid_holdouts:
                raise ValueError(f"final campaign freeze refused because hold-out prediction approval is incomplete: {invalid_holdouts}")
        files = []
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"campaign_freeze_manifest.json", "SHA256SUMS.txt"}):
            files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        protocol_verification = verify_configured_protocol_freeze(profile_payload, profile_path=profile_path)
        protocol_manifest = protocol_verification.get("manifest") if isinstance(protocol_verification.get("manifest"), Mapping) else {}
        protocol_projection = protocol_manifest.get("projection") if isinstance(protocol_manifest.get("projection"), Mapping) else {}
        manifest = {
            "schema": CAMPAIGN_FREEZE_SCHEMA,
            "schema_version": 1,
            "created_at": now_iso(),
            "tool_version": TOOL_PACKAGE_VERSION,
            "signer": signer,
            "profile_sha256": sha256_file(root / "profile.yaml"),
            "claim_scope": str(claim_scope.get("scope") or ""),
            "claim_scope_source": str(claim_scope.get("source") or ""),
            "protocol_freeze": {
                "configured": bool(protocol_verification.get("configured")),
                "verified": bool(protocol_verification.get("ok")),
                "protocol_version": protocol_projection.get("protocol_version", ""),
                "amendment": protocol_projection.get("amendment", ""),
                "projection_sha256": protocol_manifest.get("projection_sha256", ""),
            },
            "included_prediction_freezes": included_predictions,
            "prediction_freeze_checks": prediction_checks,
            "files": files,
            "files_sha256": sha256_json(files),
            "attestation": (
                "The signer confirms that the referenced hold-out choices and prediction freezes were fixed before inspecting the corresponding hold-out measurements."
                if generalization_claim
                else "The signer confirms that the archived identities, policies, contracts, measurements, and report artefacts correspond to the explicitly evaluated workload and hardware matrix."
            ),
        }
        manifest["manifest_payload_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"})
        manifest_path = write_json(root / "campaign_freeze_manifest.json", manifest)
        sums = [f"{str(row['sha256']).replace('sha256:', '')}  {row['path']}" for row in files]
        sums.append(f"{str(sha256_file(manifest_path)).replace('sha256:', '')}  campaign_freeze_manifest.json")
        write_text(root / "SHA256SUMS.txt", "\n".join(sums) + "\n")

        if private_key:
            key = _resolve_ref(private_key)
            if not key.is_file():
                raise FileNotFoundError(f"private key not found: {key}")
            signature = root / "campaign_freeze_manifest.sig"
            subprocess.run(["openssl", "dgst", "-sha256", "-sign", str(key), "-out", str(signature), str(manifest_path)], check=True)
            if public_key:
                pub = _resolve_ref(public_key)
                if pub.is_file():
                    shutil.copy2(pub, root / "campaign_freeze_public_key.pem")
        _zip_tree(root, out)
    return out


def verify_campaign_freeze(path: str | Path, *, public_key: str | Path | None = None) -> dict[str, Any]:
    """Verify a campaign-freeze archive without trusting ZIP paths or metadata."""
    src = _resolve_ref(path)
    if not src.is_file():
        return {
            "schema": "onnx-splitpoint/campaign-freeze-verification",
            "schema_version": 1,
            "created_at": now_iso(),
            "archive": str(src),
            "ok": False,
            "error": "archive_missing",
            "mismatches": [],
            "signature_status": "not_present",
        }

    with tempfile.TemporaryDirectory(prefix="splitpoint_campaign_verify_") as tmp:
        root = Path(tmp)
        unsafe_names: list[str] = []
        zip_test_result = ""
        try:
            with zipfile.ZipFile(src, "r") as zf:
                zip_test_result = str(zf.testzip() or "")
                seen_names: set[str] = set()
                for info in zf.infolist():
                    name = str(info.filename or "")
                    pure = PurePosixPath(name)
                    unix_mode = (int(info.external_attr) >> 16) & 0o170000
                    if pure.is_absolute() or ".." in pure.parts or not name or name in seen_names or unix_mode == 0o120000:
                        unsafe_names.append(name)
                    seen_names.add(name)
                if not unsafe_names and not zip_test_result:
                    zf.extractall(root)
        except (OSError, zipfile.BadZipFile) as exc:
            return {
                "schema": "onnx-splitpoint/campaign-freeze-verification",
                "schema_version": 1,
                "created_at": now_iso(),
                "archive": str(src),
                "ok": False,
                "error": f"invalid_zip:{type(exc).__name__}",
                "mismatches": [],
                "signature_status": "not_present",
            }

        if unsafe_names or zip_test_result:
            return {
                "schema": "onnx-splitpoint/campaign-freeze-verification",
                "schema_version": 1,
                "created_at": now_iso(),
                "archive": str(src),
                "ok": False,
                "error": "unsafe_or_corrupt_zip",
                "unsafe_names": unsafe_names,
                "zip_test_failure": zip_test_result,
                "mismatches": [],
                "signature_status": "not_present",
            }

        manifest_path = root / "campaign_freeze_manifest.json"
        manifest = _load_structured(manifest_path)
        schema_ok = manifest.get("schema") == CAMPAIGN_FREEZE_SCHEMA
        manifest_payload_hash_ok = str(manifest.get("manifest_payload_sha256") or "") == sha256_json({k: v for k, v in manifest.items() if k != "manifest_payload_sha256"})
        rows = [dict(row) for row in list(manifest.get("files") or []) if isinstance(row, Mapping)]
        file_list_hash_ok = str(manifest.get("files_sha256") or "") == sha256_json(rows)
        mismatches: list[dict[str, Any]] = []
        for row in rows:
            rel = str(row.get("path") or "")
            pure = PurePosixPath(rel)
            if pure.is_absolute() or ".." in pure.parts or not rel:
                mismatches.append({"path": rel, "reason": "unsafe_manifest_path"})
                continue
            p = root / pure.as_posix()
            actual = sha256_file(p) if p.is_file() else ""
            expected = str(row.get("sha256") or "")
            if not _sha256_equal(expected, actual):
                mismatches.append({"path": rel, "expected": expected, "actual": actual})
            expected_size = row.get("size_bytes")
            if p.is_file() and expected_size not in (None, "") and int(expected_size) != int(p.stat().st_size):
                mismatches.append({"path": rel, "expected_size": expected_size, "actual_size": p.stat().st_size})

        profile_path = root / "profile.yaml"
        profile_hash_ok = bool(profile_path.is_file() and str(manifest.get("profile_sha256") or "") == str(sha256_file(profile_path) or ""))

        sums_path = root / "SHA256SUMS.txt"
        sums_mismatches: list[dict[str, Any]] = []
        sums_entries = 0
        if sums_path.is_file():
            for line in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                parts = line.split(None, 1)
                if len(parts) != 2:
                    sums_mismatches.append({"line": line, "reason": "invalid_format"})
                    continue
                expected_hex, rel = parts[0].strip(), parts[1].strip().lstrip("*")
                pure = PurePosixPath(rel)
                if pure.is_absolute() or ".." in pure.parts or not rel:
                    sums_mismatches.append({"path": rel, "reason": "unsafe_path"})
                    continue
                target = root / pure.as_posix()
                actual = str(sha256_file(target) or "").replace("sha256:", "") if target.is_file() else ""
                sums_entries += 1
                if actual != expected_hex.replace("sha256:", ""):
                    sums_mismatches.append({"path": rel, "expected": expected_hex, "actual": actual})
        else:
            sums_mismatches.append({"path": "SHA256SUMS.txt", "reason": "missing"})
        sums_ok = bool(sums_entries > 0 and not sums_mismatches)

        expected_archive_files = {str(row.get("path") or "") for row in rows}
        expected_archive_files.update({"campaign_freeze_manifest.json", "SHA256SUMS.txt"})
        optional_archive_files = {"campaign_freeze_manifest.sig", "campaign_freeze_public_key.pem"}
        actual_archive_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        unexpected_files = sorted(actual_archive_files - expected_archive_files - optional_archive_files)
        missing_declared_files = sorted(expected_archive_files - actual_archive_files)
        archive_membership_ok = not unexpected_files and not missing_declared_files

        signature_status = "not_present"
        sig = root / "campaign_freeze_manifest.sig"
        pub = _resolve_ref(public_key) if public_key else (root / "campaign_freeze_public_key.pem")
        if sig.is_file():
            if not pub.is_file():
                signature_status = "present_public_key_missing"
            else:
                try:
                    cp = subprocess.run(
                        ["openssl", "dgst", "-sha256", "-verify", str(pub), "-signature", str(sig), str(manifest_path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                    )
                    signature_status = "valid" if cp.returncode == 0 else "invalid"
                except OSError:
                    signature_status = "verification_unavailable"
        signature_ok = signature_status in {"not_present", "valid"}
        ok = bool(
            schema_ok
            and manifest_payload_hash_ok
            and file_list_hash_ok
            and profile_hash_ok
            and not mismatches
            and sums_ok
            and archive_membership_ok
            and signature_ok
        )
        return {
            "schema": "onnx-splitpoint/campaign-freeze-verification",
            "schema_version": 1,
            "created_at": now_iso(),
            "archive": str(src),
            "ok": ok,
            "schema_ok": schema_ok,
            "manifest_payload_hash_ok": manifest_payload_hash_ok,
            "file_list_hash_ok": file_list_hash_ok,
            "profile_hash_ok": profile_hash_ok,
            "sha256sums_ok": sums_ok,
            "sha256sums_entry_count": sums_entries,
            "archive_membership_ok": archive_membership_ok,
            "unexpected_files": unexpected_files,
            "missing_declared_files": missing_declared_files,
            "mismatches": mismatches,
            "sha256sums_mismatches": sums_mismatches,
            "signature_status": signature_status,
            "manifest": manifest,
        }


def create_campaign_skeleton(output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    template_root = Path(__file__).resolve().parent / "resources" / "campaign_templates"
    if template_root.is_dir():
        for source in sorted(template_root.rglob("*")):
            if not source.is_file():
                continue
            target = out / source.relative_to(template_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
    contract_spec = out / "pipeline_contract_sources.yaml"
    if not contract_spec.exists():
        write_text(contract_spec, """contracts:\n  - id: classification_preprocessing\n    kind: preprocessing\n    task: classification\n    path: configs/classification_preprocessing.yaml\n  - id: detection_preprocessing\n    kind: preprocessing\n    task: detection\n    path: configs/detection_preprocessing.yaml\n  - id: detection_decoder\n    kind: decoder\n    task: detection\n    path: configs/detection_decoder.yaml\n  - id: detection_nms\n    kind: nms\n    task: detection\n    path: configs/detection_nms.yaml\n  - id: endpoint_adapters\n    kind: adapter\n    task: all\n    path: configs/endpoint_adapters.yaml\n    implementation_sources:\n      - adapters/endpoint_adapter.py\n""")
    readme = out / "README.md"
    if not readme.exists():
        write_text(readme, """# Final campaign inputs\n\nRecommended final dataset split:\n\n- classification calibration: a fixed subset of ImageNet training data;\n- classification validation: ILSVRC2012 validation;\n- detection calibration: a fixed subset of COCO 2017 train;\n- detection validation: COCO 2017 val.\n\nDo not place the same image content in calibration and validation. Generate all four manifests with `onnx-splitpoint-campaign dataset-manifest`, generate the pipeline contract manifest, create/attest the hold-out registry, fit ranking models on development rows, and run `preflight` before the final campaign.\n""")
    return {
        "root": out,
        "contract_spec": contract_spec,
        "energy_calibration_spec": out / "energy_calibration_spec.yaml",
        "protocol_freeze_spec": out / "protocol_freeze_spec.yaml",
        "holdout_selection_template": out / "HOLDOUT_SELECTION_TEMPLATE.md",
        "readme": readme,
    }


def _matrix_pipeline_contract_payloads() -> dict[str, dict[str, Any]]:
    """Return the locked contracts for the evaluated three-model matrix.

    The stable contract stops at prepared RGB uint8 semantics.  Numeric dtype,
    layout, quantisation and normalisation remain backend-attested runtime
    identities, because ORT, TensorRT, Hailo and DeepX do not consume one
    universal numeric tensor representation.
    """

    classification = canonical_image_preprocessing_contract(
        "classification", [224, 224]
    )
    detection = canonical_image_preprocessing_contract(
        "detection", [640, 640]
    )
    common = {
        "schema": "onnx-splitpoint/evaluated-matrix-pipeline-contract",
        "schema_version": 1,
        "locked": True,
    }
    return {
        "classification_preprocessing.yaml": {
            **common,
            "contract_id": "classification_prepared_rgb_uint8_v2",
            "kind": "preprocessing",
            "task": "classification",
            "canonical_prepared_image": classification,
            "canonical_prepared_image_sha256": (
                preprocessing_contract_sha256(classification)
            ),
            "runtime_numeric_identity_policy": "backend_attested",
            "note": (
                "Direct bilinear resize to 224x224; no short-side resize or "
                "centre crop. Backend-specific numeric tensors are attested "
                "at runtime."
            ),
        },
        "detection_preprocessing.yaml": {
            **common,
            "contract_id": "detection_prepared_rgb_uint8_v2",
            "kind": "preprocessing",
            "task": "detection",
            "canonical_prepared_image": detection,
            "canonical_prepared_image_sha256": (
                preprocessing_contract_sha256(detection)
            ),
            "runtime_numeric_identity_policy": "backend_attested",
            "note": (
                "Fixed centred 640x640 letterbox with RGB pad value 114 and "
                "Python rounding; no stride auto-padding."
            ),
        },
        "detection_decoder_matrix.yaml": {
            **common,
            "contract_id": "evaluated_matrix_detection_decoders_v2",
            "kind": "decoder",
            "task": "detection",
            "class_count": 80,
            "box_format": "xyxy",
            "coordinate_space": "original_image",
            "model_contracts": {
                "yolo26s": {
                    "decoder_id": (
                        "yolo26_regcls_ltrb_classaware_nms_v1"
                    ),
                    "raw_output_format": "ultralytics_regcls",
                    "integrated_output_format": "bn6_detections",
                    "endpoint_policy": (
                        "runtime_attested_raw_or_integrated_nms"
                    ),
                },
                "yolov7_paper": {
                    "decoder_id": (
                        "yolov7_paper_standard_anchor_classaware_nms_v2"
                    ),
                    "model_sha256": (
                        "7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d"
                    ),
                    "anchor_table_id": (
                        "yolov7_paper_standard_anchors_640_v1"
                    ),
                    "anchors_by_stride": [
                        {"stride": 8, "anchors_wh": [[12, 16], [19, 36], [40, 28]]},
                        {"stride": 16, "anchors_wh": [[36, 75], [76, 55], [72, 146]]},
                        {"stride": 32, "anchors_wh": [[142, 110], [192, 243], [459, 401]]},
                    ],
                    "raw_output_format": "multiscale_head",
                    "endpoint_policy": "frozen_raw_multiscale_head",
                },
            },
            "note": (
                "One task-level manifest row contains the two exact model "
                "contracts so the native energy planner resolves exactly one "
                "decoder artifact for detection."
            ),
        },
        "detection_nms.yaml": {
            **common,
            "contract_id": "detection_class_aware_nms_v1",
            "kind": "nms",
            "task": "detection",
            "implementation": "numpy_class_aware_nms_xyxy_v1",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 300,
            "class_aware": True,
            "multi_label": False,
            "coordinate_space": "original_image",
        },
    }


def _pipeline_contract_semantics(
    rows: Sequence[Mapping[str, Any]],
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the exact v2.75.28 evaluated-matrix contract semantics."""

    structured_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        path = (
            _resolve_ref(row.get("path"), base_dir)
            if row.get("path")
            else None
        )
        structured_rows.append((dict(row), _load_structured(path) if path else {}))

    preprocessing_ok = True
    preprocessing_tasks: set[str] = set()
    decoder_ok = False
    nms_ok = False
    for row, structured in structured_rows:
        kind = str(row.get("kind") or "").strip().lower()
        task = str(row.get("task") or structured.get("task") or "").strip().lower()
        if kind == "preprocessing":
            preprocessing_tasks.add(task)
            target = [224, 224] if task == "classification" else [640, 640]
            if task not in {"classification", "detection"}:
                preprocessing_ok = False
                continue
            expected = canonical_image_preprocessing_contract(task, target)
            preprocessing_ok = bool(
                preprocessing_ok
                and structured.get("canonical_prepared_image") == expected
                and structured.get("canonical_prepared_image_sha256")
                == preprocessing_contract_sha256(expected)
                and structured.get("runtime_numeric_identity_policy")
                == "backend_attested"
                and "input_layout" not in structured
                and "input_dtype" not in structured
            )
        elif kind == "decoder" and task == "detection":
            contracts = (
                structured.get("model_contracts")
                if isinstance(structured.get("model_contracts"), Mapping)
                else {}
            )
            yolo26 = (
                contracts.get("yolo26s")
                if isinstance(contracts.get("yolo26s"), Mapping)
                else {}
            )
            yolov7 = (
                contracts.get("yolov7_paper")
                if isinstance(contracts.get("yolov7_paper"), Mapping)
                else {}
            )
            decoder_ok = bool(
                set(contracts) == {"yolo26s", "yolov7_paper"}
                and _number(structured.get("class_count")) == 80.0
                and structured.get("box_format") == "xyxy"
                and structured.get("coordinate_space") == "original_image"
                and yolo26.get("decoder_id")
                == "yolo26_regcls_ltrb_classaware_nms_v1"
                and yolo26.get("raw_output_format")
                == "ultralytics_regcls"
                and yolo26.get("integrated_output_format")
                == "bn6_detections"
                and yolo26.get("endpoint_policy")
                == "runtime_attested_raw_or_integrated_nms"
                and yolov7.get("decoder_id")
                == "yolov7_paper_standard_anchor_classaware_nms_v2"
                and yolov7.get("model_sha256")
                == "7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d"
                and yolov7.get("anchor_table_id")
                == "yolov7_paper_standard_anchors_640_v1"
                and yolov7.get("anchors_by_stride") == [
                    {"stride": 8, "anchors_wh": [[12, 16], [19, 36], [40, 28]]},
                    {"stride": 16, "anchors_wh": [[36, 75], [76, 55], [72, 146]]},
                    {"stride": 32, "anchors_wh": [[142, 110], [192, 243], [459, 401]]},
                ]
                and yolov7.get("raw_output_format") == "multiscale_head"
                and yolov7.get("endpoint_policy")
                == "frozen_raw_multiscale_head"
            )
        elif kind == "nms" and task == "detection":
            nms_ok = bool(
                structured.get("implementation")
                == "numpy_class_aware_nms_xyxy_v1"
                and _number(structured.get("confidence_threshold")) == 0.25
                and _number(structured.get("iou_threshold")) == 0.45
                and _number(structured.get("max_detections")) == 300.0
                and structured.get("class_aware") is True
                and structured.get("multi_label") is False
                and structured.get("coordinate_space") == "original_image"
            )
    return {
        "ok": bool(
            preprocessing_ok
            and preprocessing_tasks == {"classification", "detection"}
            and decoder_ok
            and nms_ok
        ),
        "preprocessing_ok": preprocessing_ok,
        "preprocessing_tasks": sorted(preprocessing_tasks),
        "decoder_matrix_ok": decoder_ok,
        "nms_ok": nms_ok,
    }


def _resolve_matrix_model(
    *, models_root: Path, model_id: str, explicit_path: str = ""
) -> Path:
    if explicit_path:
        candidate = _resolve_ref(explicit_path, models_root)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{model_id} model not found: {candidate}"
            )
        return candidate

    normalized_names = {
        "resnet50": {"resnet50"},
        "yolo26s": {"yolo26s"},
        # The exact runtime identity is yolov7_paper.  An official export may
        # still use the conventional filename yolov7.onnx.
        "yolov7_paper": {"yolov7paper", "yolov7"},
    }[model_id]
    matches = sorted(
        path.resolve()
        for path in models_root.rglob("*.onnx")
        if re.sub(r"[^a-z0-9]", "", path.stem.lower()) in normalized_names
    )
    if len(matches) != 1:
        detail = ", ".join(str(path) for path in matches) or "none"
        raise ValueError(
            f"expected exactly one {model_id} ONNX below {models_root}; "
            f"found {len(matches)} ({detail}). Pass its explicit --{model_id.replace('_', '-')}-model path."
        )
    return matches[0]


def _binary_artifact(binary: str, *, artifact_id: str) -> dict[str, Any]:
    value = str(binary or "").strip()
    if not value:
        raise ValueError(f"{artifact_id} binary must not be empty")
    candidate = Path(value).expanduser()
    if candidate.is_file() or candidate.is_absolute() or "/" in value:
        return {
            "id": artifact_id,
            "kind": "measurement_implementation",
            "path": str(candidate.resolve()),
        }
    resolved = shutil.which(value)
    if not resolved:
        for fallback in (
            Path.home() / ".cargo" / "bin" / value,
            Path("/usr/local/bin") / value,
            Path("/usr/bin") / value,
        ):
            if fallback.is_file():
                resolved = str(fallback)
                break
    if resolved:
        return {
            "id": artifact_id,
            "kind": "measurement_implementation",
            "path": str(Path(resolved).resolve()),
        }
    return {
        "id": artifact_id,
        "kind": "measurement_implementation",
        "executable": value,
    }


def _preparation_artifact_conflicts(
    payload: Mapping[str, Any], *, campaign_dir: Path
) -> list[str]:
    """Return mutations/missing files relative to a prior preparation seal."""

    data = dict(payload or {})
    conflicts: list[str] = []
    expected_payload_hash = str(data.get("payload_sha256") or "")
    actual_payload_hash = sha256_json(
        {
            key: value
            for key, value in data.items()
            if key != "payload_sha256"
        }
    )
    if not expected_payload_hash or expected_payload_hash != actual_payload_hash:
        conflicts.append("preparation_manifest_payload_hash_mismatch")
    for row in list(data.get("artifacts") or []):
        if not isinstance(row, Mapping):
            conflicts.append("preparation_artifact_row_invalid")
            continue
        path = _resolve_ref(row.get("path"), campaign_dir)
        expected = str(row.get("sha256") or "")
        actual = sha256_file(path) or ""
        if not path.is_file():
            conflicts.append(f"missing:{path}")
        elif not _sha256_equal(expected, actual):
            conflicts.append(f"sha256_mismatch:{path}")
    return conflicts


def prepare_evaluated_matrix_campaign(
    *,
    campaign_dir: str | Path,
    models_root: str | Path,
    energy_method_attested_by: str,
    accept_validated_energy_method_reuse: bool,
    template: str | Path | None = None,
    resnet50_model: str = "",
    yolo26s_model: str = "",
    yolov7_paper_model: str = "",
    collector_binary: str = "urecs-data-collector",
    power_calculations_binary: str = "power_calculations",
) -> dict[str, Any]:
    """Materialise the exact evaluated-matrix profile and locked manifests.

    This command intentionally does not create or overwrite a sealed profile.
    It preserves any older working profile once, then writes a new working
    profile whose generated inputs contain no editable placeholders.
    """

    signer = str(energy_method_attested_by or "").strip()
    if not accept_validated_energy_method_reuse:
        raise ValueError(
            "explicit acceptance of the validated Wachsmuth energy method is required"
        )
    if not signer:
        raise ValueError("energy method attestation requires a non-empty signer")

    root = Path(campaign_dir).expanduser().resolve()
    model_root = Path(models_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"campaign directory not found: {root}")
    if not model_root.is_dir():
        raise FileNotFoundError(f"models root not found: {model_root}")
    template_path = (
        _resolve_ref(template)
        if template
        else Path(__file__).resolve().parent
        / "resources"
        / "evaluation_profiles"
        / "thesis_final_evaluated_matrix_v1.yaml"
    )
    profile = copy.deepcopy(_load_structured(template_path))
    if not profile:
        raise ValueError(f"could not load evaluated-matrix template: {template_path}")

    campaign_inputs = root / "campaign_inputs"
    required_inputs = (
        "imagenet_train_calibration_manifest.json",
        "imagenet_val_manifest.json",
        "coco2017_train_calibration_manifest.json",
        "coco2017_val_manifest.json",
        "instances_val2017.json",
    )
    missing_inputs = [
        str(campaign_inputs / name)
        for name in required_inputs
        if not (campaign_inputs / name).is_file()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "required existing campaign inputs are missing: "
            + ", ".join(missing_inputs)
        )

    generated_target = campaign_inputs / "generated_v27528"
    preparation_manifest_target = generated_target / "preparation_manifest.json"
    working_profile = root / "final_profile_working.yaml"
    backup_profile = root / "final_profile_working.pre_v27528.yaml"
    sealed_profile = root / "sealed_final_profile.yaml"
    rollback_target = campaign_inputs / ".generated_v27528.rollback"
    if rollback_target.exists():
        raise RuntimeError(
            "an interrupted preparation rollback directory exists; review "
            f"it before retrying: {rollback_target}"
        )

    sealed_payload = _load_structured(sealed_profile)
    sealed_campaign = (
        sealed_payload.get("campaign")
        if isinstance(sealed_payload.get("campaign"), Mapping)
        else {}
    )
    sealed_generated_targets = {
        (generated_target / "pipeline_contract_manifest.json").resolve(),
        (
            generated_target
            / "urecs_fs_input_validated_method_manifest.json"
        ).resolve(),
    }
    sealed_references = [
        _resolve_ref(value, sealed_profile.parent)
        for value in (
            sealed_campaign.get("pipeline_contract_manifest"),
            sealed_campaign.get("energy_calibration_manifest"),
        )
        if str(value or "").strip()
    ]
    if any(path in sealed_generated_targets for path in sealed_references):
        raise RuntimeError(
            "sealed_final_profile.yaml already references generated_v27528; "
            "refusing to mutate sealed campaign evidence"
        )

    existing_working = _load_structured(working_profile)
    existing_campaign = (
        existing_working.get("campaign")
        if isinstance(existing_working.get("campaign"), Mapping)
        else {}
    )
    existing_is_generated = bool(
        existing_campaign.get("pipeline_contract_manifest")
        == "campaign_inputs/generated_v27528/pipeline_contract_manifest.json"
        and existing_campaign.get("energy_calibration_manifest")
        == "campaign_inputs/generated_v27528/"
        "urecs_fs_input_validated_method_manifest.json"
    )
    if preparation_manifest_target.is_file():
        preparation_conflicts = _preparation_artifact_conflicts(
            _load_structured(preparation_manifest_target),
            campaign_dir=root,
        )
        if preparation_conflicts:
            raise RuntimeError(
                "prepared campaign files were changed after generation; "
                "refusing to overwrite reviewed work: "
                + ", ".join(preparation_conflicts[:20])
            )
    elif existing_is_generated or generated_target.exists():
        raise RuntimeError(
            "generated_v27528 exists without a valid preparation manifest; "
            "refusing a non-atomic overwrite"
        )

    model_paths = {
        "resnet50": _resolve_matrix_model(
            models_root=model_root,
            model_id="resnet50",
            explicit_path=resnet50_model,
        ),
        "yolo26s": _resolve_matrix_model(
            models_root=model_root,
            model_id="yolo26s",
            explicit_path=yolo26s_model,
        ),
        "yolov7_paper": _resolve_matrix_model(
            models_root=model_root,
            model_id="yolov7_paper",
            explicit_path=yolov7_paper_model,
        ),
    }

    model_rows = list(
        ((profile.get("model_suite") or {}).get("primary") or [])
    )
    by_id = {
        str(row.get("id") or ""): row
        for row in model_rows
        if isinstance(row, Mapping)
    }
    # A v2.75.27 template may still carry the wrong alias.  Repair it before
    # enforcing the exact matrix identity.
    if "yolov7" in by_id and "yolov7_paper" not in by_id:
        by_id["yolov7"]["id"] = "yolov7_paper"
        by_id["yolov7_paper"] = by_id.pop("yolov7")
    if set(by_id) != {"resnet50", "yolo26s", "yolov7_paper"}:
        raise ValueError(
            "evaluated-matrix template must contain exactly resnet50, "
            "yolo26s and yolov7_paper"
        )
    for model_id, path in model_paths.items():
        by_id[model_id]["path"] = str(path)
        by_id[model_id]["model_sha256"] = sha256_file(path)

    generated = Path(
        tempfile.mkdtemp(prefix=".generated_v27528.staging.", dir=campaign_inputs)
    )
    configs = generated / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    payloads = _matrix_pipeline_contract_payloads()
    for name, payload in payloads.items():
        _write_yaml(configs / name, payload)

    package_root = Path(__file__).resolve().parent
    preprocessing_sources = [
        package_root / "preprocessing_contract.py",
        package_root / "resources" / "templates" / "run_split_onnxruntime.py.txt",
    ]
    detection_sources = [
        package_root / "runners" / "harness" / "yolo.py",
        package_root / "native_detection_postprocess.py",
        package_root / "resources" / "templates" / "run_split_onnxruntime.py.txt",
    ]
    contract_spec = {
        "schema": "onnx-splitpoint/pipeline-contract-source-spec",
        "schema_version": 2,
        "portable_paths": True,
        "claim_scope": EVALUATED_MATRIX_CLAIM_SCOPE,
        "model_ids": ["resnet50", "yolo26s", "yolov7_paper"],
        "contracts": [
            {
                "id": "classification_preprocessing",
                "kind": "preprocessing",
                "task": "classification",
                "path": "configs/classification_preprocessing.yaml",
                "implementation_sources": [
                    str(path) for path in preprocessing_sources
                ],
            },
            {
                "id": "detection_preprocessing",
                "kind": "preprocessing",
                "task": "detection",
                "path": "configs/detection_preprocessing.yaml",
                "implementation_sources": [
                    str(path) for path in preprocessing_sources
                ],
            },
            {
                "id": "detection_decoder_matrix",
                "kind": "decoder",
                "task": "detection",
                "path": "configs/detection_decoder_matrix.yaml",
                "implementation_sources": [
                    str(path) for path in detection_sources
                ],
            },
            {
                "id": "detection_nms",
                "kind": "nms",
                "task": "detection",
                "path": "configs/detection_nms.yaml",
                "implementation_sources": [
                    str(path) for path in detection_sources
                ],
            },
        ],
    }
    spec_path = _write_yaml(generated / "pipeline_contract_sources.yaml", contract_spec)
    pipeline_manifest_path = create_pipeline_contract_manifest(
        spec=spec_path,
        output=generated / "pipeline_contract_manifest.json",
    )

    energy_sources = [
        package_root / "energy" / "collector.py",
        package_root / "energy" / "config.py",
        package_root / "energy" / "metrics.py",
        package_root / "resources" / "remote_scripts" / "energy_measurement_cli.py",
        package_root / "resources" / "remote_scripts" / "native_producer_energy_plan.py",
        package_root
        / "resources"
        / "remote_scripts"
        / "run_native_producer_energy_from_summary.py",
    ]
    energy_spec = {
        "schema": "onnx-splitpoint/energy-method-source-spec",
        "schema_version": 2,
        "portable_paths": True,
        "evidence_mode": INHERITED_VALIDATED_ENERGY_METHOD_MODE,
        "channel_id": "urecs_fs_input_channel_0",
        "scope": "FS",
        "measurement_point": "complete_system_input",
        "sample_rate_hz": 2000,
        "locked": True,
        "method": {
            "collector": "urecs-data-collector",
            "collector_mode": "fast_firmware",
            "postprocessor": "power_calculations",
            "sample_rate_hz": 2000,
            "output_semantics": "calibrated_input_energy_unsubtracted",
            "implementation_policy": "exact_validated_implementation_reuse",
        },
        "validation_reference": dict(WACHSMUTH_ENERGY_METHOD_REFERENCE),
        "reuse_attestation": {
            "validated_method_accepted": True,
            "exact_implementation_reused": True,
            "new_calibration_required": False,
            "attested_by": signer,
            "attested_at": now_iso(),
            "statement": (
                "Channel 0 measures complete-system input power on all three "
                "setups. The measurement implementation validated by Joris "
                "Wachsmuth is reused unchanged; no new calibration or "
                "reference measurement is introduced by this campaign."
            ),
        },
        "channel_bindings": [dict(row) for row in FINAL_MATRIX_SETUP_BINDINGS],
        "artifacts": [
            *[
                {
                    "id": f"tool_source_{path.stem}",
                    "kind": "measurement_implementation",
                    "path": str(path),
                }
                for path in energy_sources
            ],
            _binary_artifact(
                collector_binary, artifact_id="urecs_data_collector_binary"
            ),
            _binary_artifact(
                power_calculations_binary,
                artifact_id="power_calculations_binary",
            ),
        ],
    }
    energy_spec_path = _write_yaml(
        generated / "urecs_fs_input_validated_method_spec.yaml", energy_spec
    )
    energy_manifest_path = create_energy_calibration_manifest(
        spec=energy_spec_path,
        output=generated / "urecs_fs_input_validated_method_manifest.json",
    )

    pipeline_verification = verify_pipeline_contract_manifest(
        _load_structured(pipeline_manifest_path),
        require_locked=True,
        manifest_path=pipeline_manifest_path,
    )
    energy_verification = verify_energy_calibration_manifest(
        _load_structured(energy_manifest_path),
        require_final=True,
        expected_channel_bindings=FINAL_MATRIX_SETUP_BINDINGS,
    )
    if not pipeline_verification.get("ok"):
        raise RuntimeError(
            f"generated pipeline contracts failed verification: {pipeline_verification}"
        )
    if not energy_verification.get("ok"):
        raise RuntimeError(
            f"generated energy method manifest failed verification: {energy_verification}"
        )

    campaign = (
        dict(profile.get("campaign") or {})
        if isinstance(profile.get("campaign"), Mapping)
        else {}
    )
    campaign.update({
        "claim_scope": EVALUATED_MATRIX_CLAIM_SCOPE,
        "pipeline_contract_manifest": (
            "campaign_inputs/generated_v27528/pipeline_contract_manifest.json"
        ),
        "energy_calibration_manifest": (
            "campaign_inputs/generated_v27528/"
            "urecs_fs_input_validated_method_manifest.json"
        ),
    })
    profile["campaign"] = campaign
    measurement = (
        dict(profile.get("measurement_campaign") or {})
        if isinstance(profile.get("measurement_campaign"), Mapping)
        else {}
    )
    system_power = (
        dict(measurement.get("system_power") or {})
        if isinstance(measurement.get("system_power"), Mapping)
        else {}
    )
    system_power.update({
        "scope": "FS",
        "source": "urecs_input_channel",
        "channel": 0,
        "sample_rate_hz": 2000,
        "measurement_point": "complete_system_input",
        "required_setup_ids": [
            str(row["setup_id"]) for row in FINAL_MATRIX_SETUP_BINDINGS
        ],
        "energy_evidence_mode": INHERITED_VALIDATED_ENERGY_METHOD_MODE,
    })
    measurement["system_power"] = system_power
    profile["measurement_campaign"] = measurement

    # Use the same strict schema gate as the actual evaluation loader before
    # replacing any working profile.  Preparation provenance belongs in the
    # separate preparation manifest, not in ad-hoc profile fields.
    from .benchmark.evaluation_profiles import (
        validate_evaluation_profile_payload,
    )
    validate_evaluation_profile_payload(
        profile, source="prepare-evaluated-matrix generated profile"
    )

    staged_working_profile = _write_yaml(
        generated / ".final_profile_working.staged.yaml", profile
    )
    created_staged = [
        *[configs / name for name in payloads],
        spec_path,
        pipeline_manifest_path,
        energy_spec_path,
        energy_manifest_path,
        staged_working_profile,
    ]

    def _committed_artifact_path(staged: Path) -> Path:
        if staged == staged_working_profile:
            return working_profile
        return generated_target / staged.relative_to(generated)

    def _portable_campaign_path(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return str(path)

    backup_expected = bool(
        backup_profile.is_file()
        or (working_profile.is_file() and not existing_is_generated)
    )
    preparation_manifest = {
        "schema": "onnx-splitpoint/evaluated-matrix-preparation-manifest",
        "schema_version": 1,
        "created_at": now_iso(),
        "tool_version": TOOL_PACKAGE_VERSION,
        "generator": "onnx-splitpoint-campaign/prepare-evaluated-matrix",
        "template": str(template_path),
        "template_sha256": sha256_file(template_path),
        "campaign_dir": str(root),
        "working_profile": "final_profile_working.yaml",
        "working_profile_sha256": sha256_file(staged_working_profile),
        "sealed_profile_written": False,
        "previous_working_profile_backup": (
            "final_profile_working.pre_v27528.yaml"
            if backup_expected
            else ""
        ),
        "models": [
            {
                "id": model_id,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for model_id, path in model_paths.items()
        ],
        "energy_channel_bindings": [
            dict(row) for row in FINAL_MATRIX_SETUP_BINDINGS
        ],
        "artifacts": [
            {
                "path": _portable_campaign_path(
                    _committed_artifact_path(path)
                ),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in created_staged
        ],
        "pipeline_verification": pipeline_verification,
        "energy_method_verification": energy_verification,
    }
    preparation_manifest["payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in preparation_manifest.items()
            if key != "payload_sha256"
        }
    )
    write_json(
        generated / "preparation_manifest.json", preparation_manifest
    )

    if working_profile.is_file() and not existing_is_generated:
        if backup_profile.is_file():
            if not _sha256_equal(
                sha256_file(working_profile), sha256_file(backup_profile)
            ):
                raise RuntimeError(
                    "the pre-v2.75.28 backup already exists with different "
                    "content; refusing to overwrite the current working profile"
                )
        else:
            shutil.copy2(working_profile, backup_profile)

    previous_generated_existed = generated_target.exists()
    new_directory_committed = False
    try:
        if previous_generated_existed:
            generated_target.replace(rollback_target)
        generated.replace(generated_target)
        new_directory_committed = True
        committed_profile = (
            generated_target / ".final_profile_working.staged.yaml"
        )
        committed_profile.replace(working_profile)
    except Exception:
        if new_directory_committed and generated_target.exists():
            shutil.rmtree(generated_target)
        if rollback_target.exists():
            rollback_target.replace(generated_target)
        raise
    else:
        if rollback_target.exists():
            shutil.rmtree(rollback_target)

    pipeline_manifest_path = generated_target / "pipeline_contract_manifest.json"
    energy_manifest_path = (
        generated_target / "urecs_fs_input_validated_method_manifest.json"
    )
    preparation_manifest_path = generated_target / "preparation_manifest.json"
    return {
        "ok": True,
        "working_profile": str(working_profile),
        "sealed_profile_written": False,
        "previous_working_profile_backup": (
            str(backup_profile) if backup_profile.is_file() else ""
        ),
        "pipeline_contract_manifest": str(pipeline_manifest_path),
        "energy_method_manifest": str(energy_manifest_path),
        "preparation_manifest": str(preparation_manifest_path),
        "model_ids": list(model_paths),
        "energy_setup_ids": [
            str(row["setup_id"]) for row in FINAL_MATRIX_SETUP_BINDINGS
        ],
        "next_step": (
            "Run onnx-splitpoint-campaign preflight --profile "
            f"{working_profile} --out-dir {root / 'preflight'}"
        ),
    }


def _main_dataset(ns: argparse.Namespace) -> int:
    path = create_dataset_manifest(
        task=ns.task,
        role=ns.role,
        dataset_id=ns.dataset_id,
        split=ns.split,
        root=ns.root,
        output=ns.out,
        annotations=ns.annotations or None,
        labels=ns.labels or None,
        hash_mode=ns.hash_mode,
        max_items=ns.max_items,
        selection_strategy=ns.selection_strategy,
        selection_seed=ns.selection_seed,
    )
    print(path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Final-campaign manifest and freeze utilities.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create a final-campaign input skeleton.")
    p_init.add_argument("--out", required=True)

    p_prepare = sub.add_parser(
        "prepare-evaluated-matrix",
        help=(
            "Create the exact locked three-model matrix inputs and a working "
            "profile from existing dataset manifests."
        ),
    )
    p_prepare.add_argument("--campaign-dir", required=True)
    p_prepare.add_argument("--models-root", required=True)
    p_prepare.add_argument("--template", default="")
    p_prepare.add_argument("--resnet50-model", default="")
    p_prepare.add_argument("--yolo26s-model", default="")
    p_prepare.add_argument("--yolov7-paper-model", default="")
    p_prepare.add_argument("--collector-binary", default="urecs-data-collector")
    p_prepare.add_argument(
        "--power-calculations-binary", default="power_calculations"
    )
    p_prepare.add_argument("--energy-method-attested-by", required=True)
    p_prepare.add_argument(
        "--accept-validated-energy-method-reuse",
        action="store_true",
        help=(
            "Attest that channel 0 is complete-system input power and the "
            "unchanged measurement implementation validated by Joris "
            "Wachsmuth is reused without a new calibration."
        ),
    )

    p_data = sub.add_parser("dataset-manifest", help="Create a content-addressed dataset manifest.")
    p_data.add_argument("--task", required=True, choices=["classification", "detection"])
    p_data.add_argument("--role", required=True, choices=["calibration", "validation", "screening"])
    p_data.add_argument("--dataset-id", required=True)
    p_data.add_argument("--split", required=True)
    p_data.add_argument("--root", required=True)
    p_data.add_argument("--annotations", default="")
    p_data.add_argument("--labels", default="")
    p_data.add_argument("--hash-mode", default="content", choices=["content", "paths"])
    p_data.add_argument("--max-items", type=int, default=0)
    p_data.add_argument("--selection-strategy", default="deterministic_hash", choices=["deterministic_hash", "class_stratified", "sorted_first"])
    p_data.add_argument("--selection-seed", type=int, default=20260710)
    p_data.add_argument("--out", required=True)

    p_contract = sub.add_parser("pipeline-contract", help="Hash preprocessing, decoder, and NMS configuration files.")
    p_contract.add_argument("--spec", required=True)
    p_contract.add_argument("--out", required=True)

    p_energy_cal = sub.add_parser(
        "energy-calibration",
        help=(
            "Create locked, hash-verifiable full-system energy evidence "
            "(direct calibration or inherited validated method)."
        ),
    )
    p_energy_cal.add_argument("--spec", required=True)
    p_energy_cal.add_argument("--out", required=True)

    p_holdout = sub.add_parser("holdout-registry", help="Create a model-role and hold-out-attestation registry from a profile.")
    p_holdout.add_argument("--profile", required=True)
    p_holdout.add_argument("--out", required=True)

    p_pre = sub.add_parser("preflight", help="Validate all software-side prerequisites of a campaign profile.")
    p_pre.add_argument("--profile", required=True)
    p_pre.add_argument("--out-dir", "--out", dest="out_dir", default="")

    p_freeze = sub.add_parser("freeze", help="Create a checksum-sealed campaign archive, optionally including prediction freezes.")
    p_freeze.add_argument("--profile", required=True)
    p_freeze.add_argument("--run-dir", default="")
    p_freeze.add_argument("--out", required=True)
    p_freeze.add_argument("--signer", default="")
    p_freeze.add_argument("--private-key", default="")
    p_freeze.add_argument("--public-key", default="")

    p_verify = sub.add_parser("verify", help="Verify a campaign freeze archive and optional detached signature.")
    p_verify.add_argument("archive", nargs="?", default="")
    p_verify.add_argument("--archive", dest="archive_option", default="")
    p_verify.add_argument("--public-key", default="")

    p_approve = sub.add_parser("approve-predictions", help="Approve a prospective hold-out prediction freeze before benchmark execution.")
    p_approve.add_argument("--run-dir", required=True)
    p_approve.add_argument("--model-id", required=True)
    p_approve.add_argument("--signer", required=True)
    p_approve.add_argument("--out", default="")
    p_approve.add_argument("--private-key", default="")
    p_approve.add_argument("--public-key", default="")

    p_verify_approval = sub.add_parser("verify-approval", help="Verify a model-level prediction-freeze approval.")
    p_verify_approval.add_argument("approval", nargs="?", default="")
    p_verify_approval.add_argument("--approval", dest="approval_option", default="")
    p_verify_approval.add_argument("--public-key", default="")
    p_verify_approval.add_argument("--require-cryptographic-signature", action="store_true")

    p_protocol = sub.add_parser("protocol-freeze", help="Freeze prospective candidate, DAG, prediction, policy, energy, and model-role inputs.")
    p_protocol.add_argument("--profile", required=True)
    p_protocol.add_argument("--out", required=True)
    p_protocol.add_argument("--version", default="")
    p_protocol.add_argument("--amendment", default="")
    p_protocol.add_argument("--signer", default="")

    p_protocol_verify = sub.add_parser("verify-protocol", help="Fail if a protocol freeze or any referenced prospective input changed.")
    p_protocol_verify.add_argument("--freeze", required=True)
    p_protocol_verify.add_argument("--profile", default="")

    ns = ap.parse_args(list(argv) if argv is not None else None)
    if ns.cmd == "init":
        print(json.dumps({k: str(v) for k, v in create_campaign_skeleton(ns.out).items()}, indent=2))
        return 0
    if ns.cmd == "prepare-evaluated-matrix":
        if not ns.accept_validated_energy_method_reuse:
            ap.error(
                "prepare-evaluated-matrix requires "
                "--accept-validated-energy-method-reuse"
            )
        result = prepare_evaluated_matrix_campaign(
            campaign_dir=ns.campaign_dir,
            models_root=ns.models_root,
            energy_method_attested_by=ns.energy_method_attested_by,
            accept_validated_energy_method_reuse=True,
            template=(ns.template or None),
            resnet50_model=ns.resnet50_model,
            yolo26s_model=ns.yolo26s_model,
            yolov7_paper_model=ns.yolov7_paper_model,
            collector_binary=ns.collector_binary,
            power_calculations_binary=ns.power_calculations_binary,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if ns.cmd == "dataset-manifest":
        return _main_dataset(ns)
    if ns.cmd == "pipeline-contract":
        print(create_pipeline_contract_manifest(spec=ns.spec, output=ns.out))
        return 0
    if ns.cmd == "energy-calibration":
        print(create_energy_calibration_manifest(spec=ns.spec, output=ns.out))
        return 0
    if ns.cmd == "holdout-registry":
        print(create_holdout_registry(profile=ns.profile, output=ns.out))
        return 0
    if ns.cmd == "preflight":
        p = _resolve_ref(ns.profile)
        report = build_campaign_readiness(_load_structured(p), profile_path=p)
        if ns.out_dir:
            out = _resolve_ref(ns.out_dir)
            write_json(out / "campaign_readiness.json", report)
            write_text(out / "campaign_readiness.md", readiness_markdown(report))
            write_csv(out / "campaign_readiness.csv", [dict(row) for row in list(report.get("checks") or []) if isinstance(row, Mapping)])
        print(json.dumps(report, indent=2, ensure_ascii=False))
        successful_statuses = {
            "ready",
            "final_ready",
            "development_ready",
            "incomplete",
        }
        return 0 if report.get("status") in successful_statuses else 2
    if ns.cmd == "freeze":
        print(create_campaign_freeze(profile=ns.profile, output=ns.out, run_dir=(ns.run_dir or None), signer=ns.signer, private_key=(ns.private_key or None), public_key=(ns.public_key or None)))
        return 0
    if ns.cmd == "verify":
        archive = ns.archive_option or ns.archive
        if not archive:
            ap.error("verify requires ARCHIVE or --archive")
        report = verify_campaign_freeze(archive, public_key=(ns.public_key or None))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok") else 2
    if ns.cmd == "approve-predictions":
        print(create_prediction_freeze_approval(
            run_dir=ns.run_dir,
            model_id=ns.model_id,
            signer=ns.signer,
            output=(ns.out or None),
            private_key=(ns.private_key or None),
            public_key=(ns.public_key or None),
        ))
        return 0
    if ns.cmd == "verify-approval":
        approval = ns.approval_option or ns.approval
        if not approval:
            ap.error("verify-approval requires APPROVAL or --approval")
        report = verify_prediction_freeze_approval(
            approval,
            public_key=(ns.public_key or None),
            require_cryptographic_signature=bool(ns.require_cryptographic_signature),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok") else 2
    if ns.cmd == "protocol-freeze":
        print(create_protocol_freeze(
            profile=ns.profile,
            output=ns.out,
            version=(ns.version or None),
            amendment=(ns.amendment or None),
            signer=ns.signer,
        ))
        return 0
    if ns.cmd == "verify-protocol":
        report = verify_protocol_freeze(ns.freeze, profile=(ns.profile or None))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok") else 2
    return 2


__all__ = [
    "CAMPAIGN_CLAIM_SCOPES",
    "DIRECT_ENERGY_CALIBRATION_MODE",
    "EVALUATED_MATRIX_CLAIM_SCOPE",
    "FINAL_MATRIX_SETUP_BINDINGS",
    "INHERITED_VALIDATED_ENERGY_METHOD_MODE",
    "RANKING_GENERALIZATION_CLAIM_SCOPE",
    "WACHSMUTH_ENERGY_METHOD_REFERENCE",
    "build_campaign_readiness",
    "apply_ranking_model_bundle",
    "campaign_references",
    "create_energy_calibration_manifest",
    "create_campaign_freeze",
    "create_campaign_skeleton",
    "create_candidate_universe_manifest",
    "create_dataset_manifest",
    "create_holdout_registry",
    "create_pipeline_contract_manifest",
    "prepare_evaluated_matrix_campaign",
    "create_prediction_freeze_approval",
    "create_protocol_freeze",
    "deterministic_audit_cases",
    "materialize_campaign_inputs",
    "readiness_markdown",
    "resolve_campaign_claim_scope",
    "stable_candidate_identity",
    "validate_calibration_validation_separation",
    "validate_ranking_model_bundle",
    "verify_dataset_manifest",
    "verify_energy_calibration_manifest",
    "verify_campaign_freeze",
    "verify_holdout_registry",
    "verify_pipeline_contract_manifest",
    "verify_prediction_freeze_approval",
    "verify_protocol_freeze",
]


if __name__ == "__main__":
    raise SystemExit(main())
