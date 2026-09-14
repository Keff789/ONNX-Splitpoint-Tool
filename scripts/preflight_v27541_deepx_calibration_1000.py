#!/usr/bin/env python3
"""Read-only, fail-closed dataset preflight for the v2.75.41 DeepX canary.

The canary must not replace the v2.75.40 500-item calibration manifest.  Its
1000-item calibration data therefore lives in a count/seed-specific namespace.
This script only reads manifests and dataset files; it never creates, downloads,
copies, registers, repairs, or overwrites data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.campaign import verify_dataset_manifest
from onnx_splitpoint_tool.benchmark.classification_validation_presets import (
    project_classification_validation_subset,
)
from onnx_splitpoint_tool.deepx import calibration_size_canary as canary_authority
from onnx_splitpoint_tool.quality_cache import (
    image_ids_fingerprint,
    json_fingerprint,
)
from scripts.pin_v27541_deepx_calibration_baseline import (
    PinError,
    validate_frozen_b500_authority,
)


EXPECTED_ITEMS = 1000
BASELINE_ITEMS = 500
EXPECTED_CLASSES = 1000
EXPECTED_SEED = 20260710
EXPECTED_PROFILE = "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std"
EXPECTED_PREPROCESSING = "imagenet_mean_std"
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    canary_authority.EXPECTED_VALIDATION_MANIFEST_SHA256
)
EXPECTED_VALIDATION_IMAGE_IDS_SHA256 = (
    canary_authority.EXPECTED_VALIDATION_IMAGE_IDS_SHA256
)
EXPECTED_VALIDATION_GROUND_TRUTH_SHA256 = (
    canary_authority.EXPECTED_VALIDATION_GROUND_TRUTH_SHA256
)
DEFAULT_DATASET_ROOT = (
    Path.home()
    / ".onnx_splitpoint_tool"
    / "final_datasets"
)
DEDICATED_ROOT = (
    DEFAULT_DATASET_ROOT
    / "v27541_imagenet_n1000_s20260710"
)
DEFAULT_CALIBRATION_MANIFEST = (
    DEDICATED_ROOT
    / "manifests"
    / "imagenet_train_calibration_manifest.json"
)
DEFAULT_VALIDATION_MANIFEST = (
    DEFAULT_DATASET_ROOT
    / "manifests"
    / "imagenet_val_manifest.json"
)
LEGACY_CALIBRATION_POINTER = (
    DEFAULT_DATASET_ROOT
    / "manifests"
    / "imagenet_train_calibration_manifest.json"
)
DEFAULT_BASELINE_CALIBRATION_MANIFEST = (
    DEFAULT_DATASET_ROOT
    / "v27541_imagenet_n500_s20260710"
    / "manifests"
    / "imagenet_train_calibration_manifest.json"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CANDIDATE_PROFILE_SOURCE = (
    ROOT
    / "profiles"
    / "resnet50_v27541_deepx_calibration_1000_imagenet_mean_std.yaml"
)


class PreflightError(RuntimeError):
    """One required item could not be proven from read-only evidence."""


def _bare_sha256(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return raw.split(":", 1)[1] if raw.startswith("sha256:") else raw


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable_dataset_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Mirror the host-independent dataset identity used by both runners."""

    identity_rows: list[dict[str, Any]] = []
    legacy_identity_rows: list[dict[str, Any]] = []
    legacy_writer_shape = True
    for raw in list(payload.get("items") or []):
        if not isinstance(raw, Mapping):
            raise PreflightError("portable validation identity contains a non-object item")
        raw_digest = raw.get("sha256")
        raw_sample_id = raw.get("sample_id")
        raw_relative_path = raw.get("relative_path")
        raw_class_name = raw.get("class_name", "")
        digest_is_bare = bool(
            isinstance(raw_digest, str)
            and len(raw_digest) == 64
            and all(ch in "0123456789abcdef" for ch in raw_digest)
        )
        digest_is_legacy = bool(
            isinstance(raw_digest, str)
            and len(raw_digest) == 71
            and raw_digest.startswith("sha256:")
            and all(ch in "0123456789abcdef" for ch in raw_digest[7:])
        )
        sample_id_ok = bool(
            isinstance(raw_sample_id, str)
            and raw_sample_id
            and raw_sample_id == raw_sample_id.strip()
        )
        relative_path_ok = bool(
            isinstance(raw_relative_path, str)
            and raw_relative_path
            and raw_relative_path == raw_relative_path.strip()
            and not raw_relative_path.startswith("/")
            and "\\" not in raw_relative_path
            and all(
                part not in {"", ".", ".."}
                for part in raw_relative_path.split("/")
            )
        )
        if (
            not (digest_is_bare or digest_is_legacy)
            or not sample_id_ok
            or not relative_path_ok
            or not isinstance(raw_class_name, str)
        ):
            raise PreflightError("portable validation item identity is incomplete")
        digest = raw_digest[7:] if digest_is_legacy else raw_digest
        sample_id = raw_sample_id
        relative_path = raw_relative_path
        identity_rows.append({
            "sample_id": sample_id,
            "relative_path": relative_path,
            "sha256": digest,
            "class_name": raw_class_name,
        })
        # v2.75.40 manifests were written before item digests were normalized
        # to bare hex for this aggregate identity.  Mirror the deliberately
        # narrow compatibility form accepted by verify_dataset_manifest:
        # preserve the four original serializer values exactly.  The portable
        # identity emitted below always remains the normalized form.
        legacy_identity_rows.append({
            "sample_id": raw.get("sample_id"),
            "relative_path": raw.get("relative_path"),
            "sha256": raw.get("sha256", ""),
            "class_name": raw.get("class_name", ""),
        })
        legacy_writer_shape = legacy_writer_shape and digest_is_legacy
    if not identity_rows:
        raise PreflightError("portable validation identity is empty")
    items_sha = _canonical_sha256(identity_rows)
    legacy_items_sha = _canonical_sha256(legacy_identity_rows)
    raw_declared_items_sha = payload.get("items_identity_sha256")
    declared_items_sha = (
        raw_declared_items_sha[7:]
        if isinstance(raw_declared_items_sha, str)
        and raw_declared_items_sha.startswith("sha256:")
        else raw_declared_items_sha
    )
    declared_ok = bool(
        isinstance(declared_items_sha, str)
        and len(declared_items_sha) == 64
        and all(ch in "0123456789abcdef" for ch in declared_items_sha)
        and (
            declared_items_sha == items_sha
            or (
                legacy_writer_shape
                and declared_items_sha == legacy_items_sha
            )
        )
    )
    if not declared_ok:
        raise PreflightError("portable validation item identity SHA-256 mismatch")

    def component_sha(name: str) -> str:
        component = payload.get(name)
        return _bare_sha256(component.get("sha256")) if isinstance(
            component, Mapping
        ) else ""

    portable = {
        "schema": "onnx-splitpoint/portable-dataset-identity",
        "schema_version": 1,
        "dataset_id": str(payload.get("dataset_id") or ""),
        "task": str(payload.get("task") or "").strip().lower(),
        "role": str(payload.get("role") or "").strip().lower(),
        "split": str(payload.get("split") or ""),
        "hash_mode": str(payload.get("hash_mode") or "").strip().lower(),
        "item_count": len(identity_rows),
        "items_identity_sha256": items_sha,
        "annotations_sha256": component_sha("annotations"),
        "labels_sha256": component_sha("labels"),
    }
    return _canonical_sha256(portable)


def _runtime_validation_cohort_audit(
    manifest_path: Path,
    payload: Mapping[str, Any],
    *,
    selection_seed: int = EXPECTED_SEED,
) -> dict[str, Any]:
    """Project and bind the exact 500-image production validation cohort.

    ``manifest_path`` describes the fully verified 50,000-image ImageNet source
    pool.  The benchmark suite does not execute that complete pool in Standard
    mode: it uses ``project_classification_validation_subset`` to construct a
    deterministic, suite-local 500-image manifest.  The frozen v2.75.40
    authorities therefore apply to that projected manifest and its records,
    not to the portable identity of the source pool.

    This function is deliberately read-only.  It invokes the same pure
    selection/manifest projection used by production materialisation and hashes
    the bytes production would write, but creates no directories or files.
    """

    errors: list[str] = []
    root_raw = str(payload.get("root") or "").strip()
    if not root_raw:
        raise PreflightError("validation source manifest has no dataset root")
    source_root = Path(root_raw).expanduser()
    if not source_root.is_absolute():
        source_root = manifest_path.parent / source_root
    try:
        source_root = source_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreflightError(
            f"validation source dataset root is unavailable: {source_root}"
        ) from exc
    if not source_root.is_dir():
        raise PreflightError(
            f"validation source dataset root is not a directory: {source_root}"
        )

    projection = project_classification_validation_subset(
        resolved=source_root,
        requested=str(source_root),
        max_images=BASELINE_ITEMS,
        selection_seed=int(selection_seed),
        explicit_manifest=manifest_path,
        base_dir=manifest_path.parent,
    )
    if not isinstance(projection, Mapping):
        raise PreflightError(
            "production validation cohort projection returned no samples"
        )
    projected_manifest = projection.get("manifest")
    if not isinstance(projected_manifest, Mapping):
        raise PreflightError(
            "production validation cohort projection has no manifest"
        )
    projected_manifest = dict(projected_manifest)
    selection = projected_manifest.get("selection")
    samples = projected_manifest.get("samples")
    if not isinstance(selection, Mapping) or not isinstance(samples, list):
        raise PreflightError(
            "production validation cohort projection is structurally incomplete"
        )

    projected_bytes = (
        json.dumps(projected_manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    projected_manifest_sha = hashlib.sha256(projected_bytes).hexdigest()
    manifest_expected = _bare_sha256(EXPECTED_VALIDATION_MANIFEST_SHA256)
    if projected_manifest_sha != manifest_expected:
        errors.append(
            "validation_runtime_cohort_manifest_differs_from_frozen_v27540_b_authority"
        )

    records: list[dict[str, Any]] = []
    for index, raw in enumerate(samples):
        if not isinstance(raw, Mapping):
            errors.append("validation_runtime_cohort_contains_non_object_sample")
            continue
        image_id = Path(str(raw.get("image") or "")).name
        label_id = raw.get("label_id")
        if not image_id:
            errors.append("validation_runtime_cohort_image_id_missing")
            continue
        if isinstance(label_id, bool):
            errors.append("validation_runtime_cohort_label_id_invalid")
            continue
        try:
            label = int(label_id)
        except (TypeError, ValueError):
            errors.append("validation_runtime_cohort_label_id_missing")
            continue
        records.append({
            "image_id": image_id,
            "label_id": label,
            "projection_index": index,
        })
    records.sort(key=lambda row: str(row["image_id"]))
    image_ids = [str(row["image_id"]) for row in records]
    if len(image_ids) != len(set(image_ids)):
        errors.append("validation_runtime_cohort_image_ids_not_unique")

    image_ids_sha = ""
    ground_truth_sha = ""
    if image_ids and len(image_ids) == len(set(image_ids)):
        image_ids_sha = image_ids_fingerprint(image_ids)
        ground_truth_sha = json_fingerprint([{
            "image_id": row["image_id"],
            "label_id": row["label_id"],
        } for row in records])
    if image_ids_sha != _bare_sha256(EXPECTED_VALIDATION_IMAGE_IDS_SHA256):
        errors.append(
            "validation_runtime_cohort_image_ids_differ_from_frozen_v27540_b_authority"
        )
    if ground_truth_sha != _bare_sha256(
        EXPECTED_VALIDATION_GROUND_TRUTH_SHA256
    ):
        errors.append(
            "validation_runtime_cohort_ground_truth_differs_from_frozen_v27540_b_authority"
        )

    source_items = [
        raw for raw in list(payload.get("items") or [])
        if isinstance(raw, Mapping)
    ]
    cardinality_ok = bool(
        len(samples) == BASELINE_ITEMS
        and len(records) == BASELINE_ITEMS
        and int(selection_seed) == EXPECTED_SEED
        and _integer(selection.get("seed")) == EXPECTED_SEED
        and _integer(selection.get("requested_images")) == BASELINE_ITEMS
        and _integer(selection.get("selected_images")) == BASELINE_ITEMS
        and _integer(selection.get("source_population")) == len(source_items)
    )
    if not cardinality_ok:
        errors.append(
            "validation_runtime_cohort_cardinality_seed_or_population_invalid"
        )

    source_manifest = projection.get("source_manifest")
    source_binding_ok = bool(
        isinstance(source_manifest, Path)
        and source_manifest.resolve() == manifest_path.resolve()
        and str(projected_manifest.get("source") or "") == str(source_root)
        and str(projected_manifest.get("source_manifest") or "")
        == str(manifest_path.resolve())
    )
    if not source_binding_ok:
        errors.append("validation_runtime_cohort_source_binding_invalid")

    errors = list(dict.fromkeys(errors))
    return {
        "ok": not errors,
        "errors": errors,
        "projection": (
            "benchmark.classification_validation_presets."
            "project_classification_validation_subset"
        ),
        "read_only": True,
        "source_manifest": str(manifest_path),
        "source_root": str(source_root),
        "source_item_count": len(source_items),
        "runtime_item_count": len(records),
        "source_and_runtime_cardinality_are_distinct": (
            len(source_items) != len(records)
        ),
        "selection": dict(selection),
        "destination_relative": str(
            projection.get("destination_relative") or ""
        ),
        "projected_manifest_bytes": len(projected_bytes),
        "projected_manifest_sha256": projected_manifest_sha,
        "expected_projected_manifest_sha256": manifest_expected,
        "image_ids_sha256": image_ids_sha,
        "expected_image_ids_sha256": _bare_sha256(
            EXPECTED_VALIDATION_IMAGE_IDS_SHA256
        ),
        "ground_truth_sha256": ground_truth_sha,
        "expected_ground_truth_sha256": _bare_sha256(
            EXPECTED_VALIDATION_GROUND_TRUTH_SHA256
        ),
        "image_ids_preview": image_ids[:10],
        "source_binding_ok": source_binding_ok,
        "cardinality_ok": cardinality_ok,
    }


def _frozen_candidate_manifest_paths() -> dict[str, Path]:
    profile = _load_yaml_object(
        CANDIDATE_PROFILE_SOURCE,
        label="frozen v2.75.41 candidate source profile",
    )
    campaign = profile.get("campaign")
    manifests = campaign.get("dataset_manifests") if isinstance(
        campaign, Mapping
    ) else None
    classification = manifests.get("classification") if isinstance(
        manifests, Mapping
    ) else None
    if (
        profile.get("name")
        != "resnet50_v27541_deepx_calibration_1000_imagenet_mean_std"
        or not isinstance(classification, Mapping)
    ):
        raise PreflightError("frozen candidate profile manifest binding is invalid")
    result: dict[str, Path] = {}
    for role in ("calibration", "validation"):
        raw = Path(str(classification.get(role) or "")).expanduser()
        if not raw.is_absolute():
            raise PreflightError(
                f"frozen candidate profile {role} manifest is not absolute"
            )
        result[role] = _absolute_without_following_leaf(raw)
    return result


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreflightError(f"{label} is missing or is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(
            f"{label} is not readable JSON: {path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PreflightError(f"{label} must contain one JSON object: {path}")
    return dict(payload)


def _absolute_without_following_leaf(path: str | Path) -> Path:
    """Return an absolute path while preserving a leaf symlink for checks."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _integer(value: Any, *, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _manifest_seed_evidence(payload: Mapping[str, Any]) -> dict[str, int]:
    evidence: dict[str, int] = {}
    selection = payload.get("selection")
    if isinstance(selection, Mapping) and "seed" in selection:
        evidence["selection.seed"] = _integer(selection.get("seed"))
    provisioning = payload.get("provisioning_selection")
    if isinstance(provisioning, Mapping) and "seed" in provisioning:
        evidence["provisioning_selection.seed"] = _integer(
            provisioning.get("seed")
        )
    return evidence


def _explicit_class_evidence(payload: Mapping[str, Any]) -> dict[str, int]:
    evidence: dict[str, int] = {}
    for prefix, block in (
        ("selection", payload.get("selection")),
        ("provisioning_selection", payload.get("provisioning_selection")),
    ):
        if not isinstance(block, Mapping):
            continue
        for key in ("selected_class_count", "class_count"):
            if key in block:
                evidence[f"{prefix}.{key}"] = _integer(block.get(key))
    for key in ("selected_class_count", "class_count"):
        if key in payload:
            evidence[key] = _integer(payload.get(key))
    return evidence


def _item_identity(item: Mapping[str, Any]) -> tuple[str, str, str]:
    sample_id = str(
        item.get("sample_id") or item.get("relative_path") or ""
    ).strip().replace("\\", "/")
    content_sha256 = _bare_sha256(item.get("sha256"))
    class_name = str(item.get("class_name") or "").strip()
    return sample_id, content_sha256, class_name


def _audit_manifest(
    path: Path,
    *,
    label: str,
    role: str,
    expected_items: int | None,
    expected_classes: int | None,
    full_file_verification: bool,
    require_seed: bool = True,
    require_exact_inventory: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json_object(path, label=label)
    items = [
        dict(item)
        for item in list(payload.get("items") or [])
        if isinstance(item, Mapping)
    ]
    declared_count = _integer(payload.get("item_count"))
    verification = verify_dataset_manifest(
        payload,
        verify_files=full_file_verification,
        verification_mode=("full" if full_file_verification else "manifest_only"),
    )
    seed_evidence = _manifest_seed_evidence(payload)
    identities = [_item_identity(item) for item in items]
    identity_complete = bool(items) and all(
        sample_id
        and len(content_sha256) == 64
        and all(ch in "0123456789abcdef" for ch in content_sha256)
        for sample_id, content_sha256, _class_name in identities
    )
    classes = [class_name for _sample_id, _sha, class_name in identities]
    item_class_count = len(set(classes)) if classes and all(classes) else 0
    explicit_classes = _explicit_class_evidence(payload)
    explicit_classes_ok = all(
        count == expected_classes
        for count in explicit_classes.values()
    ) if expected_classes is not None else True
    classes_ok = (
        item_class_count == expected_classes and explicit_classes_ok
        if expected_classes is not None
        else True
    )
    count_ok = (
        declared_count == expected_items and len(items) == expected_items
        if expected_items is not None
        else declared_count == len(items) and bool(items)
    )
    seed_ok = (
        bool(seed_evidence)
        and seed_evidence.get("selection.seed") == EXPECTED_SEED
        and all(value == EXPECTED_SEED for value in seed_evidence.values())
        if require_seed
        else True
    )
    task_role_ok = (
        str(payload.get("task") or "").strip().lower() == "classification"
        and str(payload.get("role") or "").strip().lower() == role
    )
    inventory: dict[str, Any] = {
        "required": bool(require_exact_inventory),
        "ok": True,
        "root": str(payload.get("root") or ""),
        "manifest_image_count": len(items),
        "root_image_count": None,
        "missing_from_root_preview": [],
        "extra_in_root_preview": [],
    }
    if require_exact_inventory:
        root = Path(str(payload.get("root") or "")).expanduser()
        manifest_paths = sorted(
            str(item.get("relative_path") or "").strip().replace("\\", "/")
            for item in items
        )
        root_paths: list[str] = []
        symlinks: list[str] = []
        if root.is_dir():
            for candidate in root.rglob("*"):
                if candidate.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative = candidate.relative_to(root).as_posix()
                if candidate.is_symlink() or not candidate.is_file():
                    symlinks.append(relative)
                else:
                    root_paths.append(relative)
        missing = sorted(set(manifest_paths) - set(root_paths))
        extra = sorted(set(root_paths) - set(manifest_paths))
        inventory.update({
            "ok": bool(
                root.is_dir()
                and not symlinks
                and len(manifest_paths) == len(set(manifest_paths))
                and sorted(root_paths) == manifest_paths
            ),
            "root_image_count": len(root_paths),
            "symlink_or_non_regular_count": len(symlinks),
            "symlink_or_non_regular_preview": symlinks[:10],
            "missing_from_root_preview": missing[:10],
            "extra_in_root_preview": extra[:10],
        })
    errors: list[str] = []
    if not bool(verification.get("ok")):
        errors.append("manifest_integrity_or_dataset_file_verification_failed")
    if not task_role_ok:
        errors.append(f"expected_task_classification_and_role_{role}")
    if not count_ok:
        errors.append(
            f"expected_exactly_{expected_items}_items"
            if expected_items is not None else "manifest_item_count_invalid"
        )
    if not seed_ok:
        errors.append(f"selection_seed_must_be_{EXPECTED_SEED}")
    if not identity_complete:
        errors.append("item_identity_evidence_incomplete")
    if not classes_ok:
        errors.append(
            f"expected_item_proven_class_coverage_{expected_classes}"
        )
    if not bool(inventory.get("ok")):
        errors.append("dataset_root_inventory_must_exactly_match_manifest")
    audit = {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "ok": not errors,
        "errors": errors,
        "schema": payload.get("schema"),
        "task": payload.get("task"),
        "role": payload.get("role"),
        "declared_item_count": declared_count,
        "parsed_item_count": len(items),
        "seed_evidence": seed_evidence,
        "item_identity_complete": identity_complete,
        "item_class_count": item_class_count,
        "explicit_class_count_evidence": explicit_classes,
        "manifest_verification": verification,
        "inventory": inventory,
        "items_identity_sha256": str(payload.get("items_identity_sha256") or ""),
        "manifest_payload_sha256": str(payload.get("manifest_payload_sha256") or ""),
    }
    return payload, audit


def _strict_calibration_manifest_audit(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_count: int,
    label: str,
) -> dict[str, Any]:
    """Apply the final Canary's exact ImageNet/selection/inventory validator.

    Before the B1000 build there is no DeepX cache contract yet.  Construct a
    validation-only exact-v2 envelope from the already content-verified
    manifest, then reuse the frozen final verifier's manifest semantics.
    """

    raw_verification = verify_dataset_manifest(
        payload, verify_files=True, verification_mode="full",
    )
    verification = {
        key: raw_verification.get(key)
        for key in (
            "ok", "schema_ok", "payload_hash_ok", "identity_hash_ok",
            "item_count_ok", "verification_mode", "manifest_item_count",
            "checked_item_count", "missing_count", "mismatch_count",
        )
    }
    items: list[dict[str, Any]] = []
    for raw in list(payload.get("items") or []):
        if not isinstance(raw, Mapping):
            raise PreflightError(f"{label} contains a non-object item")
        items.append(canary_authority._manifest_item_identity(raw, label=label))
    inventory = sorted(({
        "relative_path": item["relative_path"],
        "size_bytes": item["size_bytes"],
        "sha256": item["sha256"],
    } for item in items), key=lambda row: row["relative_path"])
    root = Path(str(payload.get("root") or "")).expanduser().resolve()
    contract: dict[str, Any] = {
        "schema": "onnx-splitpoint/deepx-calibration-manifest-contract",
        "schema_version": 1,
        "task": "classification",
        "effective_count": expected_count,
        "status": "resolved",
        "dataset_id": str(payload.get("dataset_id") or ""),
        "split": str(payload.get("split") or ""),
        "role": str(payload.get("role") or ""),
        "hash_mode": str(payload.get("hash_mode") or ""),
        "item_count": expected_count,
        "items_identity_sha256": str(
            payload.get("items_identity_sha256") or ""
        ),
        "manifest_payload_sha256": str(
            payload.get("manifest_payload_sha256") or ""
        ),
        "manifest_file_sha256": _file_sha256(path),
        "manifest_file_name": path.name,
        "dataset_root_name": root.name,
        "manifest_verification": verification,
        "root_inventory_count": expected_count,
        "root_inventory_sha256": _canonical_sha256(inventory),
        # This derived envelope is used only to invoke the pure verifier.  The
        # real registry binding is produced and sealed by the later build.
        "dataset_registry_binding_sha256": _canonical_sha256({
            "scope": "v27541-read-only-preflight-validation-envelope",
            "manifest_file_sha256": _file_sha256(path),
        }),
    }
    contract["identity_sha256"] = _canonical_sha256(contract)
    try:
        _strict_payload, strict_items = canary_authority._load_manifest(
            path,
            contract=contract,
            expected_count=expected_count,
            label=label,
        )
    except Exception as exc:
        raise PreflightError(
            f"{label} failed frozen ImageNet calibration semantics: {exc}"
        ) from exc
    return {
        "ok": True,
        "validator": "deepx.calibration_size_canary._load_manifest",
        "item_count": len(strict_items),
        "dataset_id": str(payload.get("dataset_id") or ""),
        "split": str(payload.get("split") or ""),
        "hash_mode": str(payload.get("hash_mode") or ""),
        "selection_receipt_verified": bool(payload.get("provisioning_selection")),
    }


def _classification_disjointness(
    calibration: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    calibration_identities = [
        _item_identity(dict(item))
        for item in list(calibration.get("items") or [])
        if isinstance(item, Mapping)
    ]
    validation_identities = [
        _item_identity(dict(item))
        for item in list(validation.get("items") or [])
        if isinstance(item, Mapping)
    ]
    identities_complete = bool(calibration_identities and validation_identities) and all(
        sample_id and content_sha256
        for sample_id, content_sha256, _class_name
        in [*calibration_identities, *validation_identities]
    )
    calibration_ids = {sample_id for sample_id, _sha, _class in calibration_identities}
    validation_ids = {sample_id for sample_id, _sha, _class in validation_identities}
    calibration_hashes = {sha for _sample_id, sha, _class in calibration_identities}
    validation_hashes = {sha for _sample_id, sha, _class in validation_identities}
    id_overlap = sorted(calibration_ids & validation_ids)
    content_overlap = sorted(calibration_hashes & validation_hashes)
    return {
        "ok": bool(identities_complete and not id_overlap and not content_overlap),
        "identities_complete": identities_complete,
        "calibration_item_count": len(calibration_identities),
        "validation_item_count": len(validation_identities),
        "sample_id_overlap_count": len(id_overlap),
        "content_sha256_overlap_count": len(content_overlap),
        "sample_id_overlap_preview": id_overlap[:10],
        "content_sha256_overlap_preview": content_overlap[:10],
    }


def _load_yaml_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreflightError(f"{label} is missing or not a regular file: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(
            f"{label} is unreadable YAML: {path} ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PreflightError(f"{label} must contain one YAML object: {path}")
    return dict(payload)


def _find_baseline_manifest(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(run_dir.rglob("*.json")):
        if "manifest" not in path.name.lower() or path.is_symlink() or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, Mapping):
            continue
        if (
            str(payload.get("schema") or "") == "onnx-splitpoint/dataset-manifest"
            and str(payload.get("task") or "").lower() == "classification"
            and str(payload.get("role") or "").lower() == "calibration"
            and _integer(payload.get("item_count")) == BASELINE_ITEMS
        ):
            candidates.append((path, dict(payload)))
    if not candidates:
        raise PreflightError(
            "baseline run contains no run-local 500-item classification calibration "
            "dataset manifest; a profile pointer to the mutable final-dataset registry "
            "is deliberately not accepted"
        )
    identity_groups = {
        str(payload.get("items_identity_sha256") or "").strip().lower()
        for _path, payload in candidates
    }
    if "" in identity_groups or len(identity_groups) != 1:
        raise PreflightError(
            "baseline run contains ambiguous or identity-less 500-item calibration manifests: "
            + ", ".join(str(path) for path, _payload in candidates[:8])
        )
    return candidates[0]


def _baseline_profile_audit(run_dir: Path) -> dict[str, Any]:
    profile_path = run_dir / "profile.yaml"
    profile = _load_yaml_object(profile_path, label="baseline resolved profile")
    deepx = profile.get("deepx_build")
    if not isinstance(deepx, Mapping):
        deepx = {}
    observed = {
        "profile_name": str(profile.get("name") or ""),
        "classification_preprocessing": str(
            deepx.get("classification_preprocessing") or ""
        ),
        "calibration_items": _integer(
            deepx.get("calib_count", deepx.get("calibration_items"))
        ),
        "calibration_method": str(deepx.get("calibration_method") or ""),
        "optimization_level": _integer(
            deepx.get("opt_level", deepx.get("optimization_level"))
        ),
    }
    expected = {
        "profile_name": EXPECTED_PROFILE,
        "classification_preprocessing": EXPECTED_PREPROCESSING,
        "calibration_items": BASELINE_ITEMS,
        "calibration_method": "ema",
        "optimization_level": 0,
    }
    return {
        "path": str(profile_path),
        "ok": observed == expected,
        "observed": observed,
        "expected": expected,
    }


def _baseline_cache_binding_audit(
    run_dir: Path,
    baseline_path: Path,
    baseline_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        arm = canary_authority.load_calibration_arm(
            run_dir,
            label="B500",
            expected_calibration_count=BASELINE_ITEMS,
            calibration_manifest_path=baseline_path,
        )
    except Exception as exc:
        raise PreflightError(
            "baseline B500 EvaluationRun failed exact cache/receipt/DXNN/quality "
            f"validation: {exc}"
        ) from exc
    evidence = arm.base
    try:
        frozen_authority = validate_frozen_b500_authority(arm)
    except PinError as exc:
        raise PreflightError(
            "baseline B500 EvaluationRun is internally valid but is not the "
            f"frozen v2.75.40 Arm-B authority: {exc}"
        ) from exc
    status_path = (
        run_dir
        / "models"
        / "resnet50"
        / "benchmark_set"
        / "deepx"
        / "deepx_artifact_status.json"
    )
    status = _read_json_object(status_path, label="baseline DeepX artifact status")
    contract = status.get("cache_contract")
    if not isinstance(contract, Mapping):
        contract = {}
    calibration = contract.get("calibration_manifest_contract")
    if not isinstance(calibration, Mapping):
        calibration = {}
    build = contract.get("build_options")
    if not isinstance(build, Mapping):
        build = {}
    strict_baseline_items = arm.calibration_items
    expected_manifest_file_sha = _bare_sha256(calibration.get("manifest_file_sha256"))
    observed = {
        "status": str(status.get("status") or ""),
        "cache_schema": str(contract.get("schema") or ""),
        "cache_schema_version": _integer(contract.get("schema_version")),
        "task": str(contract.get("task") or ""),
        "target": str(contract.get("target") or ""),
        "classification_preprocessing": str(
            contract.get("classification_preprocessing") or ""
        ),
        "calibration_contract_status": str(calibration.get("status") or ""),
        "calibration_effective_count": _integer(calibration.get("effective_count")),
        "calibration_item_count": _integer(calibration.get("item_count")),
        "calibration_items_identity_sha256": _bare_sha256(
            calibration.get("items_identity_sha256")
        ),
        "manifest_items_identity_sha256": _bare_sha256(
            baseline_manifest.get("items_identity_sha256")
        ),
        "calibration_manifest_payload_sha256": _bare_sha256(
            calibration.get("manifest_payload_sha256")
        ),
        "manifest_payload_sha256": _bare_sha256(
            baseline_manifest.get("manifest_payload_sha256")
        ),
        "calibration_manifest_file_sha256": expected_manifest_file_sha,
        "run_local_manifest_file_sha256": _file_sha256(baseline_path),
        "build_calibration_count": _integer(build.get("calibration_count")),
        "build_calibration_method": str(build.get("calibration_method") or ""),
        "build_opt_level": _integer(build.get("opt_level")),
        "validated_cache_contract_sha256": _bare_sha256(
            evidence.cache_contract_sha256
        ),
        "declared_cache_contract_sha256": _bare_sha256(
            contract.get("contract_sha256")
        ),
        "validated_calibration_contract_sha256": _bare_sha256(
            evidence.calibration_identity_sha256
        ),
        "declared_calibration_contract_sha256": _bare_sha256(
            calibration.get("identity_sha256")
        ),
        "validated_calibration_items_identity_sha256": _bare_sha256(
            evidence.calibration_items_identity_sha256
        ),
        "validated_dxnn_sha256": _bare_sha256(evidence.dxnn_sha256),
        "full_cache_receipt_dxnn_and_quality_evidence_validated": True,
        "frozen_calibration_manifest_item_count": len(strict_baseline_items),
    }
    required_pairs = (
        (
            observed["calibration_items_identity_sha256"],
            observed["manifest_items_identity_sha256"],
        ),
        (
            observed["calibration_manifest_payload_sha256"],
            observed["manifest_payload_sha256"],
        ),
        (
            observed["calibration_manifest_file_sha256"],
            observed["run_local_manifest_file_sha256"],
        ),
        (
            observed["validated_cache_contract_sha256"],
            observed["declared_cache_contract_sha256"],
        ),
        (
            observed["validated_calibration_contract_sha256"],
            observed["declared_calibration_contract_sha256"],
        ),
        (
            observed["validated_calibration_items_identity_sha256"],
            observed["calibration_items_identity_sha256"],
        ),
    )
    ok = bool(
        observed["status"] == "ok"
        and observed["cache_schema"]
        == "onnx-splitpoint/deepx-full-cache-contract"
        and observed["cache_schema_version"] == 2
        and observed["task"] == "classification"
        and observed["target"] == "deepx_m1"
        and observed["classification_preprocessing"] == EXPECTED_PREPROCESSING
        and observed["calibration_contract_status"] == "resolved"
        and observed["calibration_effective_count"] == BASELINE_ITEMS
        and observed["calibration_item_count"] == BASELINE_ITEMS
        and all(left and left == right for left, right in required_pairs)
        and observed["build_calibration_count"] == BASELINE_ITEMS
        and observed["build_calibration_method"] == "ema"
        and observed["build_opt_level"] == 0
        and len(observed["validated_dxnn_sha256"]) == 64
        and observed["full_cache_receipt_dxnn_and_quality_evidence_validated"]
        and observed["frozen_calibration_manifest_item_count"] == BASELINE_ITEMS
    )
    return {
        "path": str(status_path),
        "ok": ok,
        "observed": observed,
        "frozen_b500_authority": frozen_authority,
    }


def _subset_audit(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    def frozen_key(item: Mapping[str, Any], *, label: str) -> tuple[Any, ...]:
        identity = canary_authority._manifest_item_identity(item, label=label)
        return (
            identity["sample_id"], identity["relative_path"],
            identity["size_bytes"], identity["sha256"],
            identity["class_name"],
        )

    baseline_keys = {
        frozen_key(item, label="B500")
        for item in list(baseline.get("items") or [])
        if isinstance(item, Mapping)
    }
    current_keys = {
        frozen_key(item, label="B1000")
        for item in list(current.get("items") or [])
        if isinstance(item, Mapping)
    }
    missing = sorted(baseline_keys - current_keys)
    return {
        "ok": bool(
            len(baseline_keys) == BASELINE_ITEMS
            and len(current_keys) == EXPECTED_ITEMS
            and baseline_keys < current_keys
        ),
        "relation": "B500_proper_subset_of_B1000",
        "baseline_unique_item_count": len(baseline_keys),
        "current_unique_item_count": len(current_keys),
        "intersection_count": len(baseline_keys & current_keys),
        "missing_from_b1000_count": len(missing),
        "missing_from_b1000_preview": [
            {
                "sample_id": sample_id,
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "sha256": sha,
                "class_name": class_name,
            }
            for sample_id, relative_path, size_bytes, sha, class_name
            in missing[:10]
        ],
    }


def _remediation() -> dict[str, Any]:
    registry = DEDICATED_ROOT / "dataset_registry.json"
    register_command = " \\\n  ".join([
        "./.venv/bin/python scripts/provision_final_datasets.py",
        f"--registry {registry}",
        "register-imagenet",
        "--train-root /ABSOLUTE/PATH/TO/CLASS_ORGANIZED/train_calibration_n1000_s20260710",
        "--val-root /ABSOLUTE/PATH/TO/CLASS_ORGANIZED/imagenet_val",
        f"--calibration-items {EXPECTED_ITEMS}",
        f"--seed {EXPECTED_SEED}",
    ])
    kaggle_command = " \\\n  ".join([
        "./.venv/bin/python scripts/provision_final_datasets.py",
        f"--registry {registry}",
        "provision-imagenet-kaggle",
        f"--root {DEDICATED_ROOT}",
        "--accept-terms",
        f"--calibration-items {EXPECTED_ITEMS}",
        f"--seed {EXPECTED_SEED}",
    ])
    pin_command = " \\\n  ".join([
        "./.venv/bin/python scripts/pin_v27541_deepx_calibration_baseline.py",
        "--baseline-run /ABSOLUTE/PATH/TO/resnet50_v27540_deepx_preprocess_b_imagenet_mean_std_RUN",
        f"--source {LEGACY_CALIBRATION_POINTER}",
        f"--out {DEFAULT_BASELINE_CALIBRATION_MANIFEST}",
    ])
    return {
        "automatic_action_taken": False,
        "safety_note": (
            "Both commands use the isolated v2.75.41 registry namespace. They do "
            "not target or overwrite the existing 500-item final-dataset manifest/root."
        ),
        "launch_gate_note": (
            "Pin B500 before provisioning B1000, then rerun this preflight with "
            "--baseline-run; dataset-only readiness is not launch readiness."
        ),
        "register_existing_exact_1000_item_root": register_command,
        "provision_isolated_kaggle_copy": kaggle_command,
        "pin_proven_b500_before_provisioning": pin_command,
        "register_precondition": (
            "--train-root must already be a dedicated class-organized directory "
            "containing exactly 1000 ordinary image files and 1000 classes; do not "
            "point it at the complete ImageNet train tree."
        ),
        "expected_manifest": str(DEFAULT_CALIBRATION_MANIFEST),
        "expected_pinned_baseline_manifest": str(
            DEFAULT_BASELINE_CALIBRATION_MANIFEST
        ),
    }


def build_preflight_report(
    *,
    calibration_manifest: str | Path = DEFAULT_CALIBRATION_MANIFEST,
    validation_manifest: str | Path = DEFAULT_VALIDATION_MANIFEST,
    baseline_run: str | Path | None = None,
    baseline_calibration_manifest: str | Path = (
        DEFAULT_BASELINE_CALIBRATION_MANIFEST
    ),
) -> dict[str, Any]:
    calibration_path = _absolute_without_following_leaf(calibration_manifest)
    validation_path = _absolute_without_following_leaf(validation_manifest)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    current: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None

    try:
        frozen_paths = _frozen_candidate_manifest_paths()
        profile_paths_ok = bool(
            calibration_path == frozen_paths["calibration"]
            and validation_path == frozen_paths["validation"]
        )
        checks.append({
            "id": "candidate_profile_manifest_paths",
            "status": "pass" if profile_paths_ok else "fail",
            "profile_source": str(CANDIDATE_PROFILE_SOURCE),
            "observed": {
                "calibration": str(calibration_path),
                "validation": str(validation_path),
            },
            "expected": {
                role: str(path) for role, path in frozen_paths.items()
            },
        })
        if not profile_paths_ok:
            errors.append(
                "preflight manifests differ from the frozen v2.75.41 candidate profile"
            )
    except Exception as exc:
        checks.append({
            "id": "candidate_profile_manifest_paths",
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        })
        errors.append(str(exc))

    dedicated_ok = calibration_path != LEGACY_CALIBRATION_POINTER.resolve()
    checks.append({
        "id": "dedicated_calibration_manifest_namespace",
        "status": "pass" if dedicated_ok else "fail",
        "path": str(calibration_path),
        "forbidden_legacy_pointer": str(LEGACY_CALIBRATION_POINTER),
    })
    if not dedicated_ok:
        errors.append("v2.75.41 must not use the mutable legacy 500-item manifest pointer")

    try:
        current, current_audit = _audit_manifest(
            calibration_path,
            label="v2.75.41 calibration manifest",
            role="calibration",
            expected_items=EXPECTED_ITEMS,
            expected_classes=EXPECTED_CLASSES,
            full_file_verification=True,
            require_exact_inventory=True,
        )
        if current_audit["ok"]:
            current_audit["frozen_calibration_semantics"] = (
                _strict_calibration_manifest_audit(
                    calibration_path,
                    current,
                    expected_count=EXPECTED_ITEMS,
                    label="B1000",
                )
            )
        checks.append({
            "id": "calibration_manifest_exact_1000",
            "status": "pass" if current_audit["ok"] else "fail",
            "evidence": current_audit,
        })
        errors.extend(current_audit["errors"])
    except Exception as exc:
        checks.append({
            "id": "calibration_manifest_exact_1000",
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        })
        errors.append(str(exc))

    try:
        validation, validation_audit = _audit_manifest(
            validation_path,
            label="classification validation manifest",
            role="validation",
            expected_items=None,
            expected_classes=None,
            full_file_verification=True,
            require_seed=False,
        )
        # The portable identity belongs to the complete registered source
        # population (50,000 files in the production ImageNet manifest).  Keep
        # it as independently verified evidence, but do not compare it with the
        # v2.75.40 B authority: that authority is the byte identity of the
        # suite-local 500-image runtime manifest.
        portable_validation_sha = _portable_dataset_manifest_sha256(validation)
        validation_audit["source_portable_manifest_sha256"] = (
            portable_validation_sha
        )
        validation_audit["source_manifest_authority_scope"] = (
            "fully_verified_source_population"
        )
        runtime_cohort = _runtime_validation_cohort_audit(
            validation_path,
            validation,
        )
        validation_audit["runtime_cohort"] = runtime_cohort
        validation_audit["frozen_v27540_b_authority_scope"] = (
            "suite_local_500_image_runtime_manifest_and_records"
        )
        if not runtime_cohort["ok"]:
            validation_audit["ok"] = False
            validation_audit["errors"].extend(runtime_cohort["errors"])
        validation_audit["errors"] = list(dict.fromkeys(
            validation_audit["errors"]
        ))
        checks.append({
            "id": "validation_manifest_integrity",
            "status": "pass" if validation_audit["ok"] else "fail",
            "evidence": validation_audit,
        })
        errors.extend(validation_audit["errors"])
    except Exception as exc:
        checks.append({
            "id": "validation_manifest_integrity",
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        })
        errors.append(str(exc))

    if current is not None and validation is not None:
        disjointness = _classification_disjointness(current, validation)
        checks.append({
            "id": "calibration_validation_disjoint",
            "status": "pass" if disjointness["ok"] else "fail",
            "evidence": disjointness,
        })
        if not disjointness["ok"]:
            errors.append("classification calibration and validation disjointness is not proven")
    else:
        checks.append({
            "id": "calibration_validation_disjoint",
            "status": "fail",
            "error": "not evaluated because one manifest is unavailable or invalid",
        })

    baseline_report: dict[str, Any] = {
        "requested": bool(baseline_run),
        "status": "missing_required",
    }
    if baseline_run:
        run_dir = Path(baseline_run).expanduser().resolve()
        requested_baseline_path = _absolute_without_following_leaf(
            baseline_calibration_manifest
        )
        try:
            if not run_dir.is_dir():
                raise PreflightError(f"baseline EvaluationRun is missing: {run_dir}")
            profile_audit = _baseline_profile_audit(run_dir)
            baseline_manifest_source = "pinned_or_explicit"
            if requested_baseline_path.is_file():
                baseline_path = requested_baseline_path
            else:
                # Compatibility fallback only. A separately pinned/explicit
                # manifest is authoritative whenever it exists.
                baseline_path, _candidate = _find_baseline_manifest(run_dir)
                baseline_manifest_source = "run_local_fallback"
            baseline, baseline_audit = _audit_manifest(
                baseline_path,
                label="pinned/explicit v2.75.40 B500 calibration manifest",
                role="calibration",
                expected_items=BASELINE_ITEMS,
                expected_classes=BASELINE_ITEMS,
                full_file_verification=True,
                require_exact_inventory=True,
            )
            cache_binding = _baseline_cache_binding_audit(
                run_dir, baseline_path, baseline
            )
            subset = (
                _subset_audit(baseline, current)
                if current is not None
                else {"ok": False, "error": "B1000 manifest unavailable"}
            )
            baseline_ok = bool(
                profile_audit["ok"]
                and baseline_audit["ok"]
                and cache_binding["ok"]
                and subset["ok"]
            )
            baseline_report = {
                "requested": True,
                "status": "pass" if baseline_ok else "fail",
                "run_dir": str(run_dir),
                "requested_baseline_manifest": str(requested_baseline_path),
                "resolved_baseline_manifest": str(baseline_path),
                "baseline_manifest_source": baseline_manifest_source,
                "profile": profile_audit,
                "manifest": baseline_audit,
                "deepx_cache_binding": cache_binding,
                "subset": subset,
            }
            if not baseline_ok:
                errors.append("the actual v2.75.40 B500 proper-subset relation is not proven")
        except Exception as exc:
            baseline_report = {
                "requested": True,
                "status": "fail",
                "run_dir": str(run_dir),
                "requested_baseline_manifest": str(requested_baseline_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(str(exc))
        checks.append({
            "id": "baseline_b500_proper_subset_of_b1000",
            "status": baseline_report["status"],
            "evidence": baseline_report,
        })
    else:
        errors.append(
            "--baseline-run is required for launch-ready status so the bound "
            "B500 proper-subset relation can be proven"
        )
        checks.append({
            "id": "baseline_b500_proper_subset_of_b1000",
            "status": "fail",
            "error": "required --baseline-run was not supplied",
            "evidence": baseline_report,
        })

    legacy_observation: dict[str, Any] = {
        "path": str(LEGACY_CALIBRATION_POINTER),
        "exists": LEGACY_CALIBRATION_POINTER.is_file(),
    }
    if LEGACY_CALIBRATION_POINTER.is_file():
        try:
            legacy = _read_json_object(
                LEGACY_CALIBRATION_POINTER,
                label="legacy final-dataset calibration pointer",
            )
            legacy_observation.update({
                "item_count": _integer(legacy.get("item_count")),
                "seed_evidence": _manifest_seed_evidence(legacy),
                "preserved_500_item_pointer": _integer(legacy.get("item_count")) == 500,
            })
        except Exception as exc:
            legacy_observation["error"] = f"{type(exc).__name__}: {exc}"

    dataset_check_ids = {
        "candidate_profile_manifest_paths",
        "dedicated_calibration_manifest_namespace",
        "calibration_manifest_exact_1000",
        "validation_manifest_integrity",
        "calibration_validation_disjoint",
    }
    dataset_checks = [
        row for row in checks if row.get("id") in dataset_check_ids
    ]
    dataset_only_ready = bool(dataset_checks) and all(
        row.get("status") == "pass" for row in dataset_checks
    )
    ready = bool(checks) and all(row.get("status") == "pass" for row in checks)
    report = {
        "schema": "onnx-splitpoint/v27541-deepx-calibration-preflight",
        "schema_version": 1,
        "status": "ready" if ready else "blocked",
        "ready": ready,
        "dataset_only_ready": dataset_only_ready,
        "read_only": True,
        "network_used": False,
        "expected": {
            "task": "classification",
            "role": "calibration",
            "calibration_items": EXPECTED_ITEMS,
            "calibration_classes": EXPECTED_CLASSES,
            "selection_seed": EXPECTED_SEED,
            "validation_items_at_runtime": 500,
            "bootstrap_repetitions": 500,
        },
        "inputs": {
            "calibration_manifest": str(calibration_path),
            "validation_manifest": str(validation_path),
            "baseline_run": str(Path(baseline_run).expanduser()) if baseline_run else "",
            "baseline_calibration_manifest": str(
                Path(baseline_calibration_manifest).expanduser()
            ) if baseline_run else "",
        },
        "checks": checks,
        "baseline": baseline_report,
        "legacy_500_pointer_observation": legacy_observation,
        "errors": list(dict.fromkeys(errors)),
        "remediation": _remediation() if not ready else None,
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only/fail-closed v2.75.41 DeepX 1000-item calibration "
            "dataset preflight"
        )
    )
    parser.add_argument(
        "--calibration-manifest",
        default=str(DEFAULT_CALIBRATION_MANIFEST),
        help="Dedicated 1000-item ImageNet train-calibration manifest",
    )
    parser.add_argument(
        "--validation-manifest",
        default=str(DEFAULT_VALIDATION_MANIFEST),
        help="ImageNet validation manifest used for the disjointness proof",
    )
    parser.add_argument(
        "--baseline-run",
        required=True,
        help=(
            "Required v2.75.40 ImageNet-Mean/Std EvaluationRun. The pinned/explicit "
            "B500 manifest and its DeepX cache binding must "
            "prove B500 as a proper subset of B1000."
        ),
    )
    parser.add_argument(
        "--baseline-calibration-manifest",
        default=str(DEFAULT_BASELINE_CALIBRATION_MANIFEST),
        help=(
            "Pinned/explicit 500-item v2.75.40 calibration manifest. It is "
            "fully verified and must match the baseline run's exact DeepX "
            "cache contract. A run-local manifest is only a missing-file fallback."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_preflight_report(
        calibration_manifest=args.calibration_manifest,
        validation_manifest=args.validation_manifest,
        baseline_run=args.baseline_run or None,
        baseline_calibration_manifest=args.baseline_calibration_manifest,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ready"]:
        for error in report.get("errors") or []:
            print(f"BLOCKED: {error}", file=sys.stderr)
        remediation = report.get("remediation") or {}
        if remediation:
            print(
                "No data was changed. Safe isolated preparation commands are "
                "included in the JSON remediation block.",
                file=sys.stderr,
            )
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
