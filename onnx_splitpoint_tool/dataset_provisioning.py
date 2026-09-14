"""Provision and register final ImageNet/COCO datasets for thesis campaigns.

The module keeps the dataset installation outside the source tree and writes a
content-addressed registry under ``~/.onnx_splitpoint_tool/final_datasets``.
COCO 2017 can be downloaded from the public official archives. ImageNet is
license gated; the tool therefore supports an authenticated Kaggle CLI route
(after the user has accepted the competition rules) and importing an existing
local copy. v60k provisions both validation and disjoint train-derived
calibration roles in one action, reports readiness per task, and can repair or
reuse an already materialised COCO calibration subset without downloading the
large train archive again. It never embeds credentials or bypasses
dataset terms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Optional, Sequence

from .campaign import (
    create_dataset_manifest,
    validate_calibration_validation_separation,
    verify_dataset_manifest,
)
from .workflow.artifacts import now_iso, sha256_file, sha256_json, write_json

REGISTRY_SCHEMA = "onnx-splitpoint/final-dataset-registry"
REGISTRY_VERSION = 1
DEFAULT_SEED = 20260710
# COCO publishes the 2017 archives through the images.cocodataset.org S3 bucket.
# Some environments currently see a hostname-mismatch certificate on the custom
# HTTPS endpoint.  We never disable certificate validation.  Instead, ``auto``
# first tries HTTPS and then the official HTTP endpoint *only* with the pinned
# archive MD5 below.  The downloaded archive is accepted only when its digest
# matches the published COCO archive digest.
COCO_ARCHIVES: dict[str, dict[str, Any]] = {
    "val2017": {
        "filename": "val2017.zip",
        "relative_url": "zips/val2017.zip",
        "urls": [
            "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
            "https://images.cocodataset.org/zips/val2017.zip",
            "http://images.cocodataset.org/zips/val2017.zip",
        ],
        "md5": "442b8da7639aecaf257c1dceb8ba8c80",
        "estimated_mib": 778,
    },
    "train2017": {
        "filename": "train2017.zip",
        "relative_url": "zips/train2017.zip",
        "urls": [
            "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip",
            "https://images.cocodataset.org/zips/train2017.zip",
            "http://images.cocodataset.org/zips/train2017.zip",
        ],
        "md5": "cced6f7f71b7629ddf16f17bbcfab6b2",
        "estimated_mib": 18440,
    },
    "annotations": {
        "filename": "annotations_trainval2017.zip",
        "relative_url": "annotations/annotations_trainval2017.zip",
        "urls": [
            "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ],
        "md5": "f4bbac642086de4f52a3fdda2de5fa2c",
        "estimated_mib": 241,
    },
}
COCO_DOWNLOAD_POLICIES = ("auto", "https_only", "pinned_http")
IMAGENET_KAGGLE_COMPETITION = "imagenet-object-localization-challenge"
IMAGENET_KAGGLE_ARCHIVE = "imagenet_object_localization_patched2019.tar.gz"
IMAGENET_KAGGLE_METADATA = ("LOC_val_solution.csv", "LOC_synset_mapping.txt")
IMAGENET_KAGGLE_DOWNLOAD_MODES = (
    "validation_only",
    "validation_via_private_kernel",
    "direct_validation_files",
    "full_competition",
)
IMAGENET_VALIDATION_EXPECTED_IMAGES = 50000
IMAGENET_EXPORT_CHUNK_MIB = 256
IMAGENET_EXPORT_KERNEL_SLUG = "onnx-splitpoint-imagenet-val-export-v60h"
IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG = "onnx-splitpoint-imagenet-calibration-export-v60j"
IMAGENET_EXPORT_SCHEMA = "onnx-splitpoint/imagenet-validation-export"
IMAGENET_CALIBRATION_EXPORT_SCHEMA = "onnx-splitpoint/imagenet-train-calibration-export"
IMAGENET_KAGGLE_RULES_URL = (
    "https://www.kaggle.com/competitions/"
    "imagenet-object-localization-challenge/rules"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LogFn = Optional[Callable[[str], None]]


def _log(log: LogFn, message: str) -> None:
    if callable(log):
        log(str(message))
    else:
        print(message, flush=True)


def default_dataset_root() -> Path:
    override = str(os.environ.get("ONNX_SPLITPOINT_FINAL_DATASETS") or "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".onnx_splitpoint_tool" / "final_datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def default_registry_path(root: str | Path | None = None) -> Path:
    return (Path(root).expanduser() if root else default_dataset_root()) / "dataset_registry.json"


def _canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("registry_payload_sha256", None)
    return out


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path).expanduser() if path else default_registry_path()
    if p.is_file():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, Mapping):
                out = dict(obj)
                out.setdefault("datasets", {})
                out.setdefault("manifests", {})
                out.setdefault("settings", {})
                return out
        except Exception:
            pass
    return {
        "schema": REGISTRY_SCHEMA,
        "schema_version": REGISTRY_VERSION,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "root": str(p.parent.resolve()),
        "datasets": {},
        "manifests": {},
        "settings": {
            "classification_calibration_items": 1000,
            "detection_calibration_items": 1000,
            "selection_seed": DEFAULT_SEED,
        },
    }


def save_registry(payload: Mapping[str, Any], path: str | Path | None = None) -> Path:
    p = Path(path).expanduser() if path else default_registry_path(payload.get("root") if isinstance(payload, Mapping) else None)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["schema"] = REGISTRY_SCHEMA
    out["schema_version"] = REGISTRY_VERSION
    out["updated_at"] = now_iso()
    out["root"] = str(p.parent.resolve())
    out["registry_payload_sha256"] = sha256_json(_canonical_payload(out))
    return write_json(p, out)


def _safe_member_path(name: str) -> PurePosixPath:
    member = PurePosixPath(str(name).replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return member


def _safe_extract_zip(archive: Path, destination: Path, *, log: LogFn = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            rel = _safe_member_path(info.filename)
            target = destination.joinpath(*rel.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    _log(log, f"[datasets] extracted {archive.name} -> {destination}")


def _safe_extract_tar(archive: Path, destination: Path, *, log: LogFn = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    mode = "r:*"
    with tarfile.open(archive, mode) as tf:
        for member in tf.getmembers():
            rel = _safe_member_path(member.name)
            # Links can escape the extraction root; do not accept them.
            if member.issym() or member.islnk():
                raise ValueError(f"links are not permitted in dataset archive: {member.name}")
            target = destination.joinpath(*rel.parts).resolve()
            target.relative_to(destination.resolve())
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"special archive members are not permitted in dataset archives: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is None:
                raise ValueError(f"could not read archive member: {member.name}")
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)
    _log(log, f"[datasets] extracted {archive.name} -> {destination}")



class DatasetDownloadError(RuntimeError):
    """Raised when all configured dataset download sources fail."""


class DatasetIntegrityError(RuntimeError):
    """Raised when a downloaded archive does not match its pinned digest."""


def _file_digest(path: Path, algorithm: str = "md5") -> str:
    hasher = hashlib.new(str(algorithm))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def _verify_pinned_archive(path: Path, *, expected_md5: str, log: LogFn = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise DatasetIntegrityError(f"archive is missing or empty: {path}")
    actual = _file_digest(path, "md5")
    expected = str(expected_md5 or "").strip().lower()
    if not expected:
        raise DatasetIntegrityError(f"no pinned digest configured for {path.name}")
    if actual != expected:
        raise DatasetIntegrityError(
            f"digest mismatch for {path.name}: expected md5 {expected}, got {actual}"
        )
    if not zipfile.is_zipfile(path):
        raise DatasetIntegrityError(f"downloaded file is not a ZIP archive: {path}")
    row = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "md5": actual,
        "verified": True,
    }
    _log(log, f"[datasets] verified {path.name}: md5={actual}")
    return row


def _is_certificate_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ssl.SSLCertVerificationError, ssl.CertificateError)):
            return True
        text = str(current).lower()
        if "certificate verify failed" in text or "hostname mismatch" in text:
            return True
        nxt = getattr(current, "reason", None) or getattr(current, "__cause__", None)
        current = nxt if isinstance(nxt, BaseException) else None
    return False


def _download_one(
    url: str,
    destination: Path,
    *,
    overwrite: bool = False,
    timeout: float = 120.0,
    log: LogFn = None,
) -> Path:
    """Download one URL with resumable ``.part`` support.

    TLS verification is always left enabled.  Partial files are intentionally
    retained on transient failures so a second attempt can resume them.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    if overwrite:
        destination.unlink(missing_ok=True)
        part.unlink(missing_ok=True)

    resume_from = int(part.stat().st_size) if part.is_file() else 0
    headers = {"User-Agent": "ONNX-Splitpoint-Tool/0.14.7 dataset provisioner"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        _log(log, f"[datasets] resume {destination.name} at {resume_from / 1024**2:.1f} MiB")
    req = urllib.request.Request(url, headers=headers)
    _log(log, f"[datasets] download {url}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = int(getattr(response, "status", 200) or 200)
            append = resume_from > 0 and status == 206
            if resume_from > 0 and not append:
                _log(log, "[datasets] source did not accept Range; restarting archive")
                resume_from = 0
            mode = "ab" if append else "wb"
            total_header = int(response.headers.get("Content-Length") or 0)
            total = total_header + resume_from if append and total_header else total_header
            copied = resume_from
            last = time.monotonic()
            with part.open(mode) as out:
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    copied += len(chunk)
                    if time.monotonic() - last > 2.0:
                        if total:
                            _log(
                                log,
                                f"[datasets] {destination.name}: "
                                f"{copied / 1024**2:.1f}/{total / 1024**2:.1f} MiB",
                            )
                        else:
                            _log(log, f"[datasets] {destination.name}: {copied / 1024**2:.1f} MiB")
                        last = time.monotonic()
        if not part.is_file() or part.stat().st_size <= 0:
            raise RuntimeError(f"empty download: {url}")
        part.replace(destination)
    except Exception:
        # Keep a non-empty partial download for a later resume.  Empty partials
        # are removed so they cannot be mistaken for progress.
        try:
            if part.exists() and part.stat().st_size <= 0:
                part.unlink()
        except Exception:
            pass
        raise

    _log(log, f"[datasets] downloaded {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
    return destination


def _source_urls(
    spec: Mapping[str, Any],
    policy: str,
    *,
    mirror_base: str | None = None,
) -> list[str]:
    if policy not in COCO_DOWNLOAD_POLICIES:
        raise ValueError(f"unsupported COCO download policy: {policy}")
    urls: list[str] = []
    custom_base = str(
        mirror_base
        or os.environ.get("ONNX_SPLITPOINT_COCO_MIRROR")
        or ""
    ).strip()
    relative = str(spec.get("relative_url") or "").strip().lstrip("/")
    if custom_base and relative:
        parsed = urllib.parse.urlparse(custom_base)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("COCO mirror base must use http:// or https://")
        urls.append(custom_base.rstrip("/") + "/" + relative)
    urls.extend(str(u) for u in list(spec.get("urls") or []) if str(u).strip())
    # Preserve source preference while removing duplicates.
    urls = list(dict.fromkeys(urls))
    if policy == "https_only":
        return [u for u in urls if u.lower().startswith("https://")]
    if policy == "pinned_http":
        return [u for u in urls if u.lower().startswith("http://")]
    return urls


def _download_verified_archive(
    spec: Mapping[str, Any],
    destination: Path,
    *,
    overwrite: bool = False,
    timeout: float = 120.0,
    policy: str = "auto",
    mirror_base: str | None = None,
    log: LogFn = None,
) -> Path:
    expected_md5 = str(spec.get("md5") or "").strip().lower()
    if destination.is_file() and not overwrite:
        try:
            _verify_pinned_archive(destination, expected_md5=expected_md5, log=log)
            _log(log, f"[datasets] reuse verified {destination}")
            return destination
        except DatasetIntegrityError as exc:
            quarantine = destination.with_name(
                destination.name + f".invalid-{int(time.time())}"
            )
            destination.replace(quarantine)
            _log(log, f"[datasets] quarantined invalid archive: {quarantine} ({exc})")

    errors: list[str] = []
    urls = _source_urls(spec, policy, mirror_base=mirror_base)
    if not urls:
        raise DatasetDownloadError(f"no sources available under policy={policy}")

    for index, url in enumerate(urls, start=1):
        is_plain_http = url.lower().startswith("http://")
        if is_plain_http and not expected_md5:
            errors.append(f"{url}: rejected because no pinned digest is configured")
            continue
        if is_plain_http:
            _log(
                log,
                "[datasets] using the official HTTP archive endpoint only with "
                f"pinned md5={expected_md5}; TLS verification is not disabled",
            )
        try:
            _download_one(url, destination, overwrite=overwrite and index == 1, timeout=timeout, log=log)
            _verify_pinned_archive(destination, expected_md5=expected_md5, log=log)
            return destination
        except Exception as exc:
            if _is_certificate_error(exc):
                _log(
                    log,
                    "[datasets] HTTPS certificate validation failed for the custom COCO host; "
                    "the certificate is not bypassed. Trying the next pinned source.",
                )
            else:
                _log(log, f"[datasets] source failed: {type(exc).__name__}: {exc}")
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            # A completed but invalid archive must not be reused for the next source.
            if destination.is_file():
                try:
                    destination.unlink()
                except Exception:
                    pass
            # A partial file may be resumed from the next equivalent official endpoint.
            continue

    joined = "\n  - ".join(errors)
    raise DatasetDownloadError(
        f"all download sources failed for {destination.name}:\n  - {joined}"
    )




def _find_named(root: Path, names: Sequence[str], *, want_dir: bool | None = None) -> Optional[Path]:
    lowered = {str(x).lower() for x in names}
    candidates: list[Path] = []
    if root.name.lower() in lowered:
        candidates.append(root)
    for p in root.rglob("*"):
        if p.name.lower() in lowered:
            candidates.append(p)
    for p in sorted(candidates, key=lambda x: (len(x.parts), str(x))):
        if want_dir is True and p.is_dir():
            return p
        if want_dir is False and p.is_file():
            return p
        if want_dir is None and p.exists():
            return p
    return None


def _register_dataset(registry: dict[str, Any], key: str, *, root: Path, annotations: Path | None = None, labels: Path | None = None, source: str = "local", extra: Mapping[str, Any] | None = None) -> None:
    record: dict[str, Any] = {
        "root": str(root.resolve()),
        "source": source,
        "registered_at": now_iso(),
        "exists": root.is_dir(),
    }
    if annotations:
        record["annotations"] = str(annotations.resolve())
        record["annotations_sha256"] = sha256_file(annotations) if annotations.is_file() else ""
    if labels:
        record["labels"] = str(labels.resolve())
        record["labels_sha256"] = sha256_file(labels) if labels.is_file() else ""
    if extra:
        record.update(dict(extra))
    registry.setdefault("datasets", {})[key] = record



def _task_separation_pass(report: Mapping[str, Any], task: str) -> bool:
    task_l = str(task).strip().lower()
    for row in list(report.get("checks") or []):
        if isinstance(row, Mapping) and str(row.get("task") or "").lower() == task_l:
            return str(row.get("status") or "").lower() == "pass"
    return False


def _task_manifest_keys(task: str) -> tuple[str, str]:
    task_l = str(task).strip().lower()
    if task_l == "classification":
        return "classification_calibration", "classification_validation"
    if task_l == "detection":
        return "detection_calibration", "detection_validation"
    raise ValueError(f"unsupported task: {task}")


def _task_separation_summary(
    manifests: Sequence[Mapping[str, Any]],
    task: str,
) -> dict[str, Any]:
    """Return a task-scoped separation result.

    ``validate_calibration_validation_separation`` intentionally evaluates both
    tasks and therefore reports ``ok=false`` while the other task has not yet
    been provisioned.  That is correct for *global* readiness but misleading
    after a successful COCO-only or ImageNet-only action.  This helper retains
    the same overlap check while making the requested task authoritative.
    """

    task_l = str(task).strip().lower()
    full = validate_calibration_validation_separation(manifests)
    check = next(
        (
            dict(row)
            for row in list(full.get("checks") or [])
            if isinstance(row, Mapping)
            and str(row.get("task") or "").strip().lower() == task_l
        ),
        {"task": task_l, "status": "missing"},
    )
    status = str(check.get("status") or "missing").lower()
    return {
        "schema": "onnx-splitpoint/task-dataset-separation-report",
        "schema_version": 1,
        "created_at": now_iso(),
        "task": task_l,
        "requested_task": task_l,
        "ok": status == "pass",
        "requested_task_ok": status == "pass",
        "status": status,
        "check": check,
        "checks": [check],
    }


def _manifest_seed(payload: Mapping[str, Any]) -> int | None:
    provenance = dict(payload.get("provisioning_selection") or {})
    selection = dict(payload.get("selection") or {})
    value = provenance.get("seed", selection.get("seed"))
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _coco_existing_calibration_subset(
    *,
    base: Path,
    coco_root: Path,
    registry: Mapping[str, Any],
    train_annotations: Path,
    calibration_items: int,
    seed: int,
    allow_rebuild_manifest: bool = True,
    log: LogFn = None,
) -> dict[str, Any] | None:
    """Locate and validate a previously materialised COCO calibration subset.

    The large ``train2017.zip`` is only a transport artefact.  Once the selected
    images, filtered annotations, selection provenance and content-addressed
    manifest are present, the archive is no longer required.  This function is
    deliberately network-free and can also reconstruct a missing registry entry
    or manifest from those materialised assets.
    """

    requested = max(1, int(calibration_items))
    seed_i = int(seed)
    manifests = dict(registry.get("manifests") or {})
    candidates: list[Path] = []
    registered = str(manifests.get("detection_calibration") or "").strip()
    if registered:
        candidates.append(Path(registered).expanduser())
    candidates.append(base / "manifests" / "coco2017_train_calibration_manifest.json")

    seen: set[str] = set()
    for manifest_path in candidates:
        key = str(manifest_path)
        if key in seen or not manifest_path.is_file():
            continue
        seen.add(key)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            verification = verify_dataset_manifest(payload, verify_files=True)
            if not bool(verification.get("ok")):
                continue
            if str(payload.get("task") or "") != "detection":
                continue
            if str(payload.get("role") or "") != "calibration":
                continue
            if int(payload.get("item_count") or 0) != requested:
                continue
            if _manifest_seed(payload) != seed_i:
                continue
            root_path = Path(str(payload.get("root") or "")).expanduser()
            annotations_ref = dict(payload.get("annotations") or {})
            annotations_path = Path(str(annotations_ref.get("path") or "")).expanduser()
            provenance = dict(payload.get("provisioning_selection") or {})
            selection_path = Path(
                str(
                    provenance.get("selection_manifest")
                    or (root_path.parent / "selection.json")
                )
            ).expanduser()
            if not root_path.is_dir() or not annotations_path.is_file() or not selection_path.is_file():
                continue
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if int(selection.get("selected_count") or -1) != requested:
                continue
            if int(selection.get("selection_seed") or -1) != seed_i:
                continue
            source_ann_hash = str(selection.get("source_annotations_sha256") or "")
            if train_annotations.is_file() and source_ann_hash:
                if sha256_file(train_annotations) != source_ann_hash:
                    continue
            selection_expected = str(provenance.get("selection_manifest_sha256") or "")
            if selection_expected and sha256_file(selection_path) != selection_expected:
                continue
            _log(
                log,
                "[datasets] reuse verified COCO calibration subset and manifest; "
                "train2017.zip is not required: " + str(root_path),
            )
            return {
                "status": "reused",
                "root": root_path,
                "annotations": annotations_path,
                "selection": selection_path,
                "manifest": manifest_path,
                "selected_count": requested,
                "population_count": int(selection.get("source_population_count") or 0),
                "source_annotations_sha256": source_ann_hash,
                "selection_seed": seed_i,
                "verification": verification,
            }
        except Exception:
            continue

    if not allow_rebuild_manifest:
        return None

    subset_parent = coco_root / "prepared" / f"train_calibration_n{requested}_s{seed_i}"
    subset_root = subset_parent / "images"
    subset_annotations = subset_parent / "instances_train2017_calibration.json"
    selection_path = subset_parent / "selection.json"
    if not (subset_root.is_dir() and subset_annotations.is_file() and selection_path.is_file()):
        return None
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if int(selection.get("selected_count") or -1) != requested:
            return None
        if int(selection.get("selection_seed") or -1) != seed_i:
            return None
        source_ann_hash = str(selection.get("source_annotations_sha256") or "")
        if train_annotations.is_file() and source_ann_hash:
            if sha256_file(train_annotations) != source_ann_hash:
                return None
        selected_names = {
            str(row.get("file_name") or "")
            for row in list(selection.get("selected_images") or [])
            if isinstance(row, Mapping) and str(row.get("file_name") or "")
        }
        actual_names = {
            str(path.relative_to(subset_root)).replace("\\", "/")
            for path in subset_root.rglob("*")
            if path.is_file()
        }
        if selected_names and actual_names != selected_names:
            return None
        if len(actual_names) != requested:
            return None

        manifest_path = base / "manifests" / "coco2017_train_calibration_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        create_dataset_manifest(
            task="detection",
            role="calibration",
            dataset_id="coco2017-train-calibration",
            split="train2017",
            root=subset_root,
            annotations=subset_annotations,
            output=manifest_path,
            max_items=0,
            selection_strategy="sorted",
            selection_seed=seed_i,
        )
        _annotate_dataset_manifest(
            manifest_path,
            provisioning_selection={
                "source_split": "train2017",
                "source_population_count": int(selection.get("source_population_count") or 0),
                "strategy": "deterministic_hash",
                "seed": seed_i,
                "requested_items": requested,
                "selected_items": requested,
                "selection_manifest": str(selection_path),
                "selection_manifest_sha256": sha256_file(selection_path),
                "source_annotations_sha256": source_ann_hash,
                "selection_uses_model_predictions": False,
                "recovered_without_train_archive": True,
            },
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        verification = verify_dataset_manifest(payload, verify_files=True)
        if not bool(verification.get("ok")):
            return None
        _log(
            log,
            "[datasets] rebuilt COCO calibration manifest from the existing materialised "
            "subset; no network download was used: " + str(manifest_path),
        )
        return {
            "status": "recovered",
            "root": subset_root,
            "annotations": subset_annotations,
            "selection": selection_path,
            "manifest": manifest_path,
            "selected_count": requested,
            "population_count": int(selection.get("source_population_count") or 0),
            "source_annotations_sha256": source_ann_hash,
            "selection_seed": seed_i,
            "verification": verification,
        }
    except Exception:
        return None



def _stable_selection_key(seed: int, *parts: Any) -> str:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _annotate_dataset_manifest(
    manifest_path: Path,
    *,
    provisioning_selection: Mapping[str, Any],
) -> Path:
    """Attach deterministic provisioning provenance to a generated manifest.

    ``create_dataset_manifest`` records the selected files and their content hashes.
    Materialised subsets additionally need to retain how those files were selected
    from the upstream population.  This helper adds that provenance and refreshes
    the payload hash without changing the manifest schema.
    """

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["provisioning_selection"] = dict(provisioning_selection)
    payload["manifest_payload_sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key != "manifest_payload_sha256"}
    )
    return write_json(manifest_path, payload)


def _create_coco_calibration_subset_from_archive(
    *,
    archive: Path,
    annotations: Path,
    coco_root: Path,
    calibration_items: int,
    seed: int,
    log: LogFn = None,
) -> dict[str, Any]:
    """Materialise a deterministic COCO train-derived calibration subset.

    The full ``train2017`` archive is never extracted.  Only the selected images are
    copied from the ZIP, which keeps the final benchmark host compact while still
    deriving calibration data from the official train split.
    """

    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise FileNotFoundError(f"COCO train archive missing or invalid: {archive}")
    if not annotations.is_file():
        raise FileNotFoundError(f"COCO train annotations missing: {annotations}")
    requested = max(1, int(calibration_items))
    annotation_payload = json.loads(annotations.read_text(encoding="utf-8"))
    images = [dict(row) for row in list(annotation_payload.get("images") or []) if isinstance(row, Mapping)]
    images = [row for row in images if str(row.get("file_name") or "").strip()]
    if not images:
        raise ValueError(f"COCO train annotations contain no images: {annotations}")
    selected = sorted(
        images,
        key=lambda row: _stable_selection_key(
            seed,
            row.get("id"),
            row.get("file_name"),
        ),
    )[: min(requested, len(images))]
    selected_ids = {int(row.get("id")) for row in selected if row.get("id") is not None}
    subset_name = f"train_calibration_n{len(selected)}_s{int(seed)}"
    subset_root = coco_root / "prepared" / subset_name / "images"
    subset_ann = coco_root / "prepared" / subset_name / "instances_train2017_calibration.json"
    selection_path = coco_root / "prepared" / subset_name / "selection.json"
    source_ann_sha = sha256_file(annotations)

    reuse = False
    if selection_path.is_file() and subset_ann.is_file() and subset_root.is_dir():
        try:
            existing = json.loads(selection_path.read_text(encoding="utf-8"))
            expected_names = {str(row.get("file_name") or "") for row in selected}
            actual_names = {
                str(path.relative_to(subset_root)).replace("\\", "/")
                for path in subset_root.rglob("*")
                if path.is_file()
            }
            reuse = bool(
                existing.get("source_annotations_sha256") == source_ann_sha
                and int(existing.get("selection_seed") or -1) == int(seed)
                and int(existing.get("selected_count") or -1) == len(selected)
                and actual_names == expected_names
            )
        except Exception:
            reuse = False
    if reuse:
        _log(log, f"[datasets] reuse materialised COCO calibration subset: {subset_root}")
    else:
        if subset_root.parent.exists():
            shutil.rmtree(subset_root.parent)
        subset_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            member_by_basename: dict[str, str] = {}
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member_by_basename.setdefault(Path(info.filename).name, info.filename)
            for index, row in enumerate(selected, start=1):
                file_name = str(row.get("file_name") or "").strip()
                member_name = f"train2017/{file_name}"
                try:
                    info = zf.getinfo(member_name)
                except KeyError:
                    fallback = member_by_basename.get(Path(file_name).name)
                    if not fallback:
                        raise FileNotFoundError(
                            f"COCO calibration image {file_name} is absent from {archive}"
                        )
                    info = zf.getinfo(fallback)
                _safe_member_path(info.filename)
                destination = subset_root / file_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if index % 100 == 0 or index == len(selected):
                    _log(log, f"[datasets] COCO calibration subset: {index}/{len(selected)}")

        filtered = {
            key: value
            for key, value in annotation_payload.items()
            if key not in {"images", "annotations"}
        }
        filtered["images"] = selected
        filtered["annotations"] = [
            row
            for row in list(annotation_payload.get("annotations") or [])
            if isinstance(row, Mapping)
            and row.get("image_id") is not None
            and int(row.get("image_id")) in selected_ids
        ]
        subset_ann.parent.mkdir(parents=True, exist_ok=True)
        subset_ann.write_text(json.dumps(filtered, ensure_ascii=False), encoding="utf-8")
        selection = {
            "schema": "onnx-splitpoint/coco-calibration-selection",
            "schema_version": 1,
            "created_at": now_iso(),
            "source_split": "train2017",
            "source_population_count": len(images),
            "source_annotations": str(annotations.resolve()),
            "source_annotations_sha256": source_ann_sha,
            "source_archive": str(archive.resolve()),
            "source_archive_md5": str(COCO_ARCHIVES["train2017"]["md5"]),
            "selection_strategy": "deterministic_hash",
            "selection_seed": int(seed),
            "requested_count": requested,
            "selected_count": len(selected),
            "selected_images": [
                {
                    "image_id": row.get("id"),
                    "file_name": str(row.get("file_name") or ""),
                }
                for row in selected
            ],
        }
        selection["selection_payload_sha256"] = sha256_json(
            {key: value for key, value in selection.items() if key != "selection_payload_sha256"}
        )
        write_json(selection_path, selection)

    return {
        "root": subset_root,
        "annotations": subset_ann,
        "selection": selection_path,
        "selected_count": len(selected),
        "population_count": len(images),
        "source_annotations_sha256": source_ann_sha,
        "selection_seed": int(seed),
    }

def _make_manifests_for_coco(registry: dict[str, Any], root: Path, *, calibration_items: int, seed: int, log: LogFn = None) -> dict[str, str]:
    val_root = root / "val2017"
    train_root = root / "train2017"
    ann_root = root / "annotations"
    val_ann = ann_root / "instances_val2017.json"
    train_ann = ann_root / "instances_train2017.json"
    manifests_dir = root.parent / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    if val_root.is_dir() and val_ann.is_file():
        p = create_dataset_manifest(
            task="detection", role="validation", dataset_id="coco2017-val", split="val2017",
            root=val_root, annotations=val_ann, output=manifests_dir / "coco2017_val_manifest.json",
            max_items=0, selection_strategy="sorted", selection_seed=seed,
        )
        out["detection_validation"] = str(p)
        _log(log, f"[datasets] COCO validation manifest: {p}")
    if train_root.is_dir() and train_ann.is_file():
        p = create_dataset_manifest(
            task="detection", role="calibration", dataset_id="coco2017-train-calibration", split="train2017",
            root=train_root, annotations=train_ann, output=manifests_dir / "coco2017_train_calibration_manifest.json",
            max_items=max(1, int(calibration_items)), selection_strategy="deterministic_hash", selection_seed=seed,
        )
        out["detection_calibration"] = str(p)
        _log(log, f"[datasets] COCO calibration manifest: {p}")
    registry.setdefault("manifests", {}).update(out)
    return out


def provision_coco2017(
    *,
    root: str | Path | None = None,
    include_train: bool = False,
    overwrite: bool = False,
    calibration_items: int = 1000,
    seed: int = DEFAULT_SEED,
    registry_path: str | Path | None = None,
    download_policy: str = "auto",
    mirror_base: str | None = None,
    materialize_calibration_subset: bool = True,
    retain_train_archive: bool = False,
    log: LogFn = None,
) -> dict[str, Any]:
    """Provision selected COCO assets and leave the registry ready for their roles.

    v60k treats the materialised calibration subset as the durable asset.  A valid
    subset and manifest are reused even after the large train archive has been
    removed.  The archive is downloaded only when the requested item count/seed
    cannot be satisfied from existing verified assets.
    """

    base = Path(root).expanduser().resolve() if root else default_dataset_root()
    coco_root = base / "coco2017"
    downloads = base / "downloads"
    coco_root.mkdir(parents=True, exist_ok=True)
    if download_policy not in COCO_DOWNLOAD_POLICIES:
        raise ValueError(
            f"download_policy must be one of {', '.join(COCO_DOWNLOAD_POLICIES)}"
        )

    registry_file = (
        Path(registry_path).expanduser().resolve()
        if registry_path
        else default_registry_path(base)
    )
    registry = load_registry(registry_file)
    val_ann = coco_root / "annotations" / "instances_val2017.json"
    train_ann = coco_root / "annotations" / "instances_train2017.json"
    manifests_dir = base / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    archives: dict[str, Path] = {}
    validation_manifest_path = manifests_dir / "coco2017_val_manifest.json"
    validation_reusable = False
    if (
        not overwrite
        and validation_manifest_path.is_file()
        and (coco_root / "val2017").is_dir()
        and val_ann.is_file()
    ):
        try:
            validation_payload = json.loads(
                validation_manifest_path.read_text(encoding="utf-8")
            )
            validation_reusable = bool(
                verify_dataset_manifest(validation_payload, verify_files=True).get("ok")
            )
        except Exception:
            validation_reusable = False

    if validation_reusable:
        _log(
            log,
            "[datasets] reuse verified COCO validation dataset and manifest: "
            + str(coco_root / "val2017"),
        )
        validation_manifest = validation_manifest_path
    else:
        archives["val2017"] = _download_verified_archive(
            COCO_ARCHIVES["val2017"],
            downloads / str(COCO_ARCHIVES["val2017"]["filename"]),
            overwrite=overwrite,
            policy=download_policy,
            mirror_base=mirror_base,
            log=log,
        )
        archives["annotations"] = _download_verified_archive(
            COCO_ARCHIVES["annotations"],
            downloads / str(COCO_ARCHIVES["annotations"]["filename"]),
            overwrite=overwrite,
            policy=download_policy,
            mirror_base=mirror_base,
            log=log,
        )
        if overwrite or not (coco_root / "val2017").is_dir():
            _safe_extract_zip(archives["val2017"], coco_root, log=log)
        if overwrite or not val_ann.is_file() or not train_ann.is_file():
            _safe_extract_zip(archives["annotations"], coco_root, log=log)
        validation_manifest = create_dataset_manifest(
            task="detection",
            role="validation",
            dataset_id="coco2017-val",
            split="val2017",
            root=coco_root / "val2017",
            annotations=val_ann,
            output=validation_manifest_path,
            max_items=0,
            selection_strategy="sorted",
            selection_seed=seed,
        )
        _log(log, f"[datasets] COCO validation manifest: {validation_manifest}")

    _register_dataset(
        registry,
        "coco2017_validation",
        root=coco_root / "val2017",
        annotations=val_ann,
        source="official_coco_archive",
    )
    manifests: dict[str, str] = {
        "detection_validation": str(validation_manifest)
    }

    subset_result: dict[str, Any] | None = None
    train_archive_removed = False
    train_archive_downloaded = False
    if include_train:
        if materialize_calibration_subset and not overwrite:
            subset_result = _coco_existing_calibration_subset(
                base=base,
                coco_root=coco_root,
                registry=registry,
                train_annotations=train_ann,
                calibration_items=calibration_items,
                seed=seed,
                allow_rebuild_manifest=True,
                log=log,
            )

        if subset_result is None:
            archives["train2017"] = _download_verified_archive(
                COCO_ARCHIVES["train2017"],
                downloads / str(COCO_ARCHIVES["train2017"]["filename"]),
                overwrite=overwrite,
                timeout=300.0,
                policy=download_policy,
                mirror_base=mirror_base,
                log=log,
            )
            train_archive_downloaded = True
            if materialize_calibration_subset:
                subset_result = _create_coco_calibration_subset_from_archive(
                    archive=archives["train2017"],
                    annotations=train_ann,
                    coco_root=coco_root,
                    calibration_items=calibration_items,
                    seed=seed,
                    log=log,
                )
                calibration_root = Path(subset_result["root"])
                calibration_annotations = Path(subset_result["annotations"])
                calibration_manifest = create_dataset_manifest(
                    task="detection",
                    role="calibration",
                    dataset_id="coco2017-train-calibration",
                    split="train2017",
                    root=calibration_root,
                    annotations=calibration_annotations,
                    output=manifests_dir / "coco2017_train_calibration_manifest.json",
                    max_items=0,
                    selection_strategy="sorted",
                    selection_seed=seed,
                )
                _annotate_dataset_manifest(
                    calibration_manifest,
                    provisioning_selection={
                        "source_split": "train2017",
                        "source_population_count": int(subset_result["population_count"]),
                        "strategy": "deterministic_hash",
                        "seed": int(seed),
                        "requested_items": int(calibration_items),
                        "selected_items": int(subset_result["selected_count"]),
                        "selection_manifest": str(subset_result["selection"]),
                        "selection_manifest_sha256": sha256_file(Path(subset_result["selection"])),
                        "source_annotations_sha256": str(subset_result["source_annotations_sha256"]),
                        "selection_uses_model_predictions": False,
                    },
                )
                subset_result["manifest"] = calibration_manifest
                subset_result["status"] = "created"
                _log(log, f"[datasets] COCO calibration manifest: {calibration_manifest}")
            else:
                if overwrite or not (coco_root / "train2017").is_dir():
                    _safe_extract_zip(archives["train2017"], coco_root, log=log)
                calibration_root = coco_root / "train2017"
                calibration_annotations = train_ann
                calibration_manifest = create_dataset_manifest(
                    task="detection",
                    role="calibration",
                    dataset_id="coco2017-train-calibration",
                    split="train2017",
                    root=calibration_root,
                    annotations=calibration_annotations,
                    output=manifests_dir / "coco2017_train_calibration_manifest.json",
                    max_items=max(1, int(calibration_items)),
                    selection_strategy="deterministic_hash",
                    selection_seed=seed,
                )
                subset_result = {
                    "status": "created_full_train",
                    "root": calibration_root,
                    "annotations": calibration_annotations,
                    "selection": Path(),
                    "manifest": calibration_manifest,
                    "selected_count": int(calibration_items),
                    "selection_seed": int(seed),
                }

        assert subset_result is not None
        calibration_root = Path(subset_result["root"])
        calibration_annotations = Path(subset_result["annotations"])
        calibration_manifest = Path(subset_result["manifest"])
        _register_dataset(
            registry,
            "coco2017_calibration",
            root=calibration_root,
            annotations=calibration_annotations,
            source=(
                "official_coco_train_materialized_subset"
                if materialize_calibration_subset
                else "official_coco_archive"
            ),
            extra={
                "selection_manifest": str(subset_result.get("selection") or ""),
                "selection_seed": int(seed),
                "requested_items": int(calibration_items),
                "selected_items": int(subset_result.get("selected_count") or calibration_items),
                "source_population_count": int(subset_result.get("population_count") or 0),
                "full_train_extracted": not materialize_calibration_subset,
                "reused_without_train_archive": not train_archive_downloaded,
            },
        )
        manifests["detection_calibration"] = str(calibration_manifest)

    registry.setdefault("manifests", {}).update(manifests)
    settings = registry.setdefault("settings", {})
    settings["detection_calibration_items"] = int(calibration_items)
    settings["selection_seed"] = int(seed)
    settings["coco_download_policy"] = str(download_policy)
    settings["coco_include_train"] = bool(include_train)
    settings["coco_materialize_calibration_subset"] = bool(materialize_calibration_subset)
    settings["coco_retain_train_archive"] = bool(retain_train_archive)
    settings["coco_mirror_base"] = str(mirror_base or "")
    saved = save_registry(registry, registry_file)

    disjointness: dict[str, Any] | None = None
    if manifests.get("detection_calibration") and manifests.get("detection_validation"):
        calibration_payload = json.loads(
            Path(manifests["detection_calibration"]).read_text(encoding="utf-8")
        )
        validation_payload = json.loads(
            Path(manifests["detection_validation"]).read_text(encoding="utf-8")
        )
        disjointness = _task_separation_summary(
            [calibration_payload, validation_payload], "detection"
        )
        if not bool(disjointness["ok"]):
            raise RuntimeError(
                "COCO calibration and validation manifests are not disjoint: "
                + json.dumps(disjointness, ensure_ascii=False)
            )

    if (
        include_train
        and materialize_calibration_subset
        and not retain_train_archive
        and archives.get("train2017")
        and Path(archives["train2017"]).is_file()
    ):
        Path(archives["train2017"]).unlink()
        train_archive_removed = True
        _log(
            log,
            "[datasets] removed the verified COCO train archive after materialising "
            "the calibration subset",
        )

    readiness = registry_status(saved, verify_manifests=False)
    return {
        "status": "ok",
        "dataset": "coco2017",
        "provisioning_scope": "validation_and_calibration" if include_train else "validation_only",
        "root": str(coco_root),
        "registry": str(saved),
        "manifests": manifests,
        "download_policy": download_policy,
        "mirror_base": str(mirror_base or ""),
        "archives": {key: str(value) for key, value in archives.items()},
        "train_archive_downloaded": train_archive_downloaded,
        "train_archive_removed": train_archive_removed,
        "calibration_subset": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in dict(subset_result or {}).items()
        },
        "calibration_validation_disjointness": disjointness,
        "task_readiness": dict((readiness.get("task_readiness") or {}).get("detection") or {}),
        "global_readiness": {
            "ready_for_final_profile": bool(readiness.get("ready_for_final_profile")),
            "missing_required_manifests": list(readiness.get("missing_required_manifests") or []),
        },
    }

def register_coco2017(
    *, val_root: str | Path, annotations_root: str | Path, train_root: str | Path | None = None,
    calibration_items: int = 1000, seed: int = DEFAULT_SEED, registry_path: str | Path | None = None,
    log: LogFn = None,
) -> dict[str, Any]:
    val = Path(val_root).expanduser().resolve()
    anns = Path(annotations_root).expanduser().resolve()
    if anns.is_file():
        ann_dir = anns.parent
        val_ann = anns
    else:
        ann_dir = anns
        val_ann = ann_dir / "instances_val2017.json"
    if not val.is_dir() or not val_ann.is_file():
        raise FileNotFoundError("COCO val2017 root or instances_val2017.json is missing")
    registry_file = Path(registry_path).expanduser().resolve() if registry_path else default_registry_path()
    base = registry_file.parent
    registry = load_registry(registry_file)
    _register_dataset(registry, "coco2017_validation", root=val, annotations=val_ann, source="registered_local")
    train = Path(train_root).expanduser().resolve() if train_root else None
    if train and train.is_dir():
        train_ann = ann_dir / "instances_train2017.json"
        if not train_ann.is_file():
            raise FileNotFoundError(f"missing {train_ann}")
        _register_dataset(registry, "coco2017_calibration", root=train, annotations=train_ann, source="registered_local")
    # create_dataset_manifest does not require a common physical parent. Build directly.
    mdir = base / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, str] = {}
    pv = create_dataset_manifest(task="detection", role="validation", dataset_id="coco2017-val", split="val2017", root=val, annotations=val_ann, output=mdir / "coco2017_val_manifest.json", max_items=0, selection_strategy="sorted", selection_seed=seed)
    manifests["detection_validation"] = str(pv)
    if train and train.is_dir():
        pc = create_dataset_manifest(task="detection", role="calibration", dataset_id="coco2017-train-calibration", split="train2017", root=train, annotations=ann_dir / "instances_train2017.json", output=mdir / "coco2017_train_calibration_manifest.json", max_items=max(1, calibration_items), selection_strategy="deterministic_hash", selection_seed=seed)
        manifests["detection_calibration"] = str(pc)
    registry.setdefault("manifests", {}).update(manifests)
    saved = save_registry(registry, registry_file)
    _log(log, f"[datasets] registered COCO: {saved}")
    return {"status": "ok", "dataset": "coco2017", "registry": str(saved), "manifests": manifests}


def _hardlink_or_copy(src: Path, dst: Path, *, mode: str = "hardlink") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _classification_class_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    class_dirs: list[Path] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            has_image = any(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in directory.iterdir()
            )
        except OSError:
            has_image = False
        if has_image:
            class_dirs.append(directory)
    return sorted(class_dirs, key=lambda path: path.name)


def _organize_imagenet_val_from_solution(val_root: Path, solution_csv: Path, destination: Path, *, link_mode: str = "hardlink", log: LogFn = None) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    with solution_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = str(row.get("ImageId") or row.get("image_id") or "").strip()
            prediction = str(row.get("PredictionString") or row.get("prediction") or "").strip()
            if not image_id or not prediction:
                continue
            wnid = prediction.split()[0]
            if wnid.startswith("n") and wnid[1:].isdigit():
                mapping[image_id] = wnid
    if not mapping:
        raise ValueError(f"could not parse ImageNet validation solution: {solution_csv}")
    linked = 0
    for image_id, wnid in mapping.items():
        candidates = [val_root / f"{image_id}.JPEG", val_root / f"{image_id}.jpeg", val_root / f"{image_id}.jpg"]
        src = next((p for p in candidates if p.is_file()), None)
        if src is None:
            continue
        _hardlink_or_copy(src, destination / wnid / src.name, mode=link_mode)
        linked += 1
    if linked == 0:
        raise FileNotFoundError(f"no ImageNet validation images from {solution_csv} were found under {val_root}")
    _log(log, f"[datasets] organized ImageNet validation: {linked} images -> {destination}")
    return destination


def _find_imagenet_layout(root: Path, *, log: LogFn = None, link_mode: str = "hardlink") -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    train = _find_named(root, ["train", "train2012"], want_dir=True)
    val = _find_named(root, ["val", "val2012", "validation"], want_dir=True)
    labels = _find_named(root, ["LOC_synset_mapping.txt", "synset_words.txt"], want_dir=False)
    solution = _find_named(root, ["LOC_val_solution.csv"], want_dir=False)
    # Some official copies contain the raw tar files. Extract only when the target is absent.
    raw_train_tar = _find_named(root, ["ILSVRC2012_img_train.tar"], want_dir=False)
    raw_val_tar = _find_named(root, ["ILSVRC2012_img_val.tar"], want_dir=False)
    canonical = root / "prepared"
    if val is None and raw_val_tar is not None:
        val_flat = canonical / "val_flat"
        if not val_flat.is_dir() or not any(val_flat.iterdir()):
            _safe_extract_tar(raw_val_tar, val_flat, log=log)
        val = val_flat
    if train is None and raw_train_tar is not None:
        # The train archive contains one tar per class. Extracting the entire set is large,
        # but this is an explicitly requested provisioning operation.
        train_raw = canonical / "train_archives"
        if not train_raw.is_dir() or not any(train_raw.iterdir()):
            _safe_extract_tar(raw_train_tar, train_raw, log=log)
        train_dir = canonical / "train"
        train_dir.mkdir(parents=True, exist_ok=True)
        for class_tar in sorted(train_raw.glob("n*.tar")):
            class_dir = train_dir / class_tar.stem
            if not class_dir.is_dir() or not any(class_dir.iterdir()):
                _safe_extract_tar(class_tar, class_dir, log=log)
        train = train_dir
    if val is not None:
        has_class_dirs = any(p.is_dir() and p.name.startswith("n") for p in val.iterdir()) if val.is_dir() else False
        if not has_class_dirs and solution is not None:
            val = _organize_imagenet_val_from_solution(val, solution, canonical / "val_by_wnid", link_mode=link_mode, log=log)
    return train, val, labels


def register_imagenet(
    *, train_root: str | Path | None, val_root: str | Path, labels: str | Path | None = None,
    calibration_items: int = 1000, seed: int = DEFAULT_SEED, registry_path: str | Path | None = None,
    source: str = "registered_local", log: LogFn = None,
) -> dict[str, Any]:
    train = Path(train_root).expanduser().resolve() if train_root else None
    val = Path(val_root).expanduser().resolve()
    labels_path = Path(labels).expanduser().resolve() if labels else None
    if not val.is_dir():
        raise FileNotFoundError(f"ImageNet validation root not found: {val}")
    if train is not None and not train.is_dir():
        raise FileNotFoundError(f"ImageNet training root not found: {train}")
    val_classes = _classification_class_dirs(val)
    if not val_classes:
        raise ValueError(
            "ImageNet validation must be class-organised before registration. "
            "For a flat ILSVRC2012_img_val.tar, use import-imagenet together with "
            "LOC_val_solution.csv so each image receives its WNID class."
        )
    if train is not None and not _classification_class_dirs(train):
        raise ValueError(
            "ImageNet train/calibration root must contain class subdirectories with images"
        )
    registry_file = Path(registry_path).expanduser().resolve() if registry_path else default_registry_path()
    base = registry_file.parent
    registry = load_registry(registry_file)
    _register_dataset(
        registry,
        "imagenet_validation",
        root=val,
        labels=labels_path,
        source=source,
        extra={"class_directory_count": len(val_classes)},
    )
    if train:
        _register_dataset(registry, "imagenet_calibration", root=train, labels=labels_path, source=source)
    mdir = base / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, str] = {}
    pv = create_dataset_manifest(task="classification", role="validation", dataset_id="ilsvrc2012-val", split="val", root=val, labels=labels_path, output=mdir / "imagenet_val_manifest.json", max_items=0, selection_strategy="sorted", selection_seed=seed)
    manifests["classification_validation"] = str(pv)
    if train:
        pc = create_dataset_manifest(task="classification", role="calibration", dataset_id="ilsvrc2012-train-calibration", split="train", root=train, labels=labels_path, output=mdir / "imagenet_train_calibration_manifest.json", max_items=max(1, calibration_items), selection_strategy="class_stratified", selection_seed=seed)
        manifests["classification_calibration"] = str(pc)
    registry.setdefault("manifests", {}).update(manifests)
    registry.setdefault("settings", {})["classification_calibration_items"] = int(calibration_items)
    registry["settings"]["selection_seed"] = int(seed)
    saved = save_registry(registry, registry_file)
    _log(log, f"[datasets] registered ImageNet: {saved}")
    return {"status": "ok", "dataset": "imagenet", "registry": str(saved), "manifests": manifests}


def import_imagenet(
    *, source: str | Path, root: str | Path | None = None, calibration_items: int = 1000,
    seed: int = DEFAULT_SEED, link_mode: str = "hardlink", registry_path: str | Path | None = None,
    validation_solution: str | Path | None = None, labels: str | Path | None = None,
    source_name: str = "imported_local",
    log: LogFn = None,
) -> dict[str, Any]:
    src = Path(source).expanduser().resolve()
    base = Path(root).expanduser().resolve() if root else default_dataset_root()
    target = base / "imagenet2012"
    supplied_solution = Path(validation_solution).expanduser().resolve() if validation_solution else None
    supplied_labels = Path(labels).expanduser().resolve() if labels else None
    if supplied_solution and not supplied_solution.is_file():
        raise FileNotFoundError(f"ImageNet validation solution not found: {supplied_solution}")
    if supplied_labels and not supplied_labels.is_file():
        raise FileNotFoundError(f"ImageNet label map not found: {supplied_labels}")

    if src.is_file():
        target.mkdir(parents=True, exist_ok=True)
        raw_val_names = {"ilsvrc2012_img_val.tar", "ilsvrc2012_img_val.tar.gz", "ilsvrc2012_img_val.tgz"}
        if src.name.lower() in raw_val_names and tarfile.is_tarfile(src):
            val_flat = target / "prepared" / "val_flat"
            if not val_flat.is_dir() or not any(val_flat.iterdir()):
                _safe_extract_tar(src, val_flat, log=log)
            layout_root = target
        elif zipfile.is_zipfile(src):
            _safe_extract_zip(src, target, log=log)
            layout_root = target
        elif tarfile.is_tarfile(src):
            _safe_extract_tar(src, target, log=log)
            layout_root = target
        else:
            raise ValueError(f"unsupported ImageNet archive: {src}")
    elif src.is_dir():
        layout_root = src
    else:
        raise FileNotFoundError(src)
    train, val, discovered_labels = _find_imagenet_layout(layout_root, log=log, link_mode=link_mode)
    labels_path = supplied_labels or discovered_labels
    if val is None and supplied_solution:
        flat_candidates = [
            layout_root / "prepared" / "val_flat",
            layout_root / "val",
            layout_root,
        ]
        flat = next(
            (
                p for p in flat_candidates
                if p.is_dir() and any(x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS for x in p.iterdir())
            ),
            None,
        )
        if flat:
            val = _organize_imagenet_val_from_solution(
                flat,
                supplied_solution,
                layout_root / "prepared" / "val_by_wnid",
                link_mode=link_mode,
                log=log,
            )
    if val is None:
        raise FileNotFoundError(
            "ImageNet validation directory could not be located. Supply a class-organized val directory, "
            "or import ILSVRC2012_img_val.tar together with LOC_val_solution.csv."
        )
    return register_imagenet(
        train_root=train,
        val_root=val,
        labels=labels_path,
        calibration_items=calibration_items,
        seed=seed,
        registry_path=registry_path,
        source=str(source_name or "imported_local"),
        log=log,
    )



def _candidate_kaggle_commands() -> list[list[str]]:
    candidates: list[list[str]] = []
    found = shutil.which("kaggle")
    if found:
        candidates.append([found])
    exe_dir = Path(sys.executable).resolve().parent
    for name in ("kaggle", "kaggle.exe"):
        candidate = exe_dir / name
        if candidate.is_file():
            candidates.append([str(candidate)])
    try:
        if importlib.util.find_spec("kaggle") is not None:
            # Depending on the installed Kaggle client version, either the
            # package or the explicit CLI module provides ``__main__``.
            candidates.extend(
                [
                    [sys.executable, "-m", "kaggle"],
                    [sys.executable, "-m", "kaggle.cli"],
                ]
            )
    except Exception:
        pass
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for command in candidates:
        key = tuple(command)
        if key not in seen:
            seen.add(key)
            deduped.append(command)
    return deduped


def resolve_kaggle_cli(*, timeout: float = 20.0) -> tuple[list[str] | None, str]:
    errors: list[str] = []
    for command in _candidate_kaggle_commands():
        try:
            proc = subprocess.run(
                command + ["--version"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            output = str(proc.stdout or "").strip()
            if proc.returncode == 0:
                return command, output
            errors.append(f"{' '.join(command)}: rc={proc.returncode}: {output}")
        except Exception as exc:
            errors.append(f"{' '.join(command)}: {type(exc).__name__}: {exc}")
    return None, "; ".join(errors)


def _kaggle_output_has_error(output: str) -> bool:
    """Return ``True`` for CLI text that represents a failed operation.

    Kaggle CLI 2.x occasionally prints a structured error but still exits with
    status code zero.  Provisioning must therefore inspect both return code and
    stdout before continuing with a stateful operation such as a kernel push.
    """

    text = str(output or "").strip().lower()
    fatal_markers = (
        "kernel push error:",
        "authentication required to call the kaggle api",
        "you must accept this competition's rules",
        "permission 'competitions.participate' was denied",
        "permission 'kernels.get' was denied",
        "cannot access kernel",
        "403 - forbidden",
        "401 - unauthorized",
    )
    return any(marker in text for marker in fatal_markers)


def _first_kaggle_file_from_listing(output: str) -> str:
    """Extract the first competition file name from CSV or table output."""

    for raw_line in str(output or "").splitlines():
        line = raw_line.strip().strip('"')
        if not line or line.lower().startswith("next page token"):
            continue
        first = line.split(",", 1)[0].strip().strip('"')
        if first.lower() in {"name", "filename", "file", "ref"}:
            continue
        if "/" in first or "." in Path(first).name:
            return first
    return ""


def _probe_kaggle_competition_download_access(
    command: Sequence[str],
    competition: str,
    filename: str,
    *,
    log: LogFn = None,
) -> dict[str, Any]:
    """Probe whether competition rules permit downloading one tiny file.

    Listing competition files does not prove that an account has joined the
    competition or accepted its rules.  The first ImageNet entry is normally a
    small XML annotation, so downloading it into a temporary directory provides
    a cheap, non-destructive participation check before a private kernel is
    created.
    """

    if not filename:
        return {
            "checked": False,
            "download_access": None,
            "rules_accepted": None,
            "probe_file": "",
            "output": "No competition file name was available for a rules probe.",
        }
    with tempfile.TemporaryDirectory(prefix="onnx-splitpoint-kaggle-rules-") as temp:
        destination = Path(temp)
        variants = [
            [
                "competitions",
                "download",
                competition,
                "-f",
                filename,
                "-p",
                str(destination),
                "-o",
                "-q",
            ],
            [
                "competitions",
                "download",
                "-c",
                competition,
                "-f",
                filename,
                "-p",
                str(destination),
                "-o",
                "-q",
            ],
        ]
        outputs: list[str] = []
        for args in variants:
            proc = _run_kaggle(command, args, log=log, timeout=120.0)
            output = str(proc.stdout or "")
            outputs.append(output)
            ok = proc.returncode == 0 and not _kaggle_output_has_error(output)
            if ok and any(path.is_file() for path in destination.rglob("*")):
                return {
                    "checked": True,
                    "download_access": True,
                    "rules_accepted": True,
                    "probe_file": filename,
                    "output": output.strip()[-4000:],
                }
        merged = "\n".join(outputs).strip()
        lower = merged.lower()
        rules_missing = (
            "accept this competition's rules" in lower
            or "competitions.participate" in lower
            or "403 - forbidden" in lower
        )
        return {
            "checked": True,
            "download_access": False,
            "rules_accepted": False if rules_missing else None,
            "probe_file": filename,
            "output": merged[-4000:],
        }


def _ensure_kaggle_competition_rules_accepted(
    command: Sequence[str],
    competition: str,
    filenames: Sequence[str],
    *,
    log: LogFn = None,
) -> dict[str, Any]:
    """Require effective competition participation before kernel creation."""

    probe_file = next((str(name) for name in filenames if str(name).strip()), "")
    probe = _probe_kaggle_competition_download_access(
        command,
        competition,
        probe_file,
        log=log,
    )
    if probe.get("download_access") is True:
        _log(
            log,
            f"[kaggle] competition rules/download preflight passed using {probe_file}",
        )
        return probe
    output = str(probe.get("output") or "")
    if probe.get("rules_accepted") is False:
        raise PermissionError(
            "Kaggle API authentication and file listing are valid, but the account has not "
            "accepted the ImageNet competition rules required for downloads and notebook "
            f"datasources. Open {IMAGENET_KAGGLE_RULES_URL}, click Join Competition / "
            "I Understand and Accept, then rerun the access check. Probe output: "
            + output[-2000:]
        )
    raise RuntimeError(
        "Kaggle could list ImageNet files but could not download the small participation "
        "probe. The tool will not create a private export kernel until this is resolved. "
        f"Probe file={probe_file!r}. Output: {output[-3000:]}"
    )


def kaggle_cli_status(
    *,
    competition: str = IMAGENET_KAGGLE_COMPETITION,
    check_access: bool = False,
    log: LogFn = None,
) -> dict[str, Any]:
    command, version = resolve_kaggle_cli()
    credential_candidates: list[Path] = []
    config_dir = str(os.environ.get("KAGGLE_CONFIG_DIR") or "").strip()
    if config_dir:
        configured = Path(config_dir).expanduser()
        credential_candidates.extend(
            [configured / "kaggle.json", configured / "access_token"]
        )
    credential_candidates.extend(
        [
            Path.home() / ".kaggle" / "kaggle.json",
            Path.home() / ".kaggle" / "access_token",
            Path.home() / ".config" / "kaggle" / "kaggle.json",
            Path.home() / ".config" / "kaggle" / "access_token",
        ]
    )
    credential_files = [str(path) for path in credential_candidates if path.is_file()]
    environment_credentials = bool(
        str(os.environ.get("KAGGLE_API_TOKEN") or "").strip()
        or (
            str(os.environ.get("KAGGLE_USERNAME") or "").strip()
            and str(os.environ.get("KAGGLE_KEY") or "").strip()
        )
    )
    payload: dict[str, Any] = {
        "available": command is not None,
        "command": command or [],
        "version": version,
        "credential_files": credential_files,
        "credential_file_detected": bool(credential_files),
        "environment_credentials_present": environment_credentials,
        "authentication_configured_hint": bool(credential_files or environment_credentials),
        "python": sys.executable,
        "access_checked": False,
        "competition_access": None,
        "competition_listing_access": None,
        "competition_download_access": None,
        "competition_rules_accepted": None,
        "competition_rules_url": IMAGENET_KAGGLE_RULES_URL,
    }
    if command and check_access:
        try:
            proc = _run_kaggle(
                command,
                [
                    "competitions",
                    "files",
                    competition,
                    "--page-size",
                    "1",
                    "-v",
                    "-q",
                ],
                log=log,
                timeout=60.0,
            )
            if proc.returncode != 0 or _kaggle_output_has_error(str(proc.stdout or "")):
                proc = _run_kaggle(
                    command,
                    [
                        "competitions",
                        "files",
                        "-c",
                        competition,
                        "--page-size",
                        "1",
                        "-v",
                        "-q",
                    ],
                    log=log,
                    timeout=60.0,
                )
            listing_output = str(proc.stdout or "")
            listing_ok = proc.returncode == 0 and not _kaggle_output_has_error(
                listing_output
            )
            payload["access_checked"] = True
            payload["competition_access"] = listing_ok
            payload["competition_listing_access"] = listing_ok
            payload["access_output"] = listing_output.strip()[-4000:]
            if listing_ok:
                probe_file = _first_kaggle_file_from_listing(listing_output)
                probe = _probe_kaggle_competition_download_access(
                    command,
                    competition,
                    probe_file,
                    log=log,
                )
                payload["competition_download_access"] = probe.get("download_access")
                payload["competition_rules_accepted"] = probe.get("rules_accepted")
                payload["competition_rules_probe_file"] = probe.get("probe_file")
                payload["competition_rules_probe_output"] = probe.get("output")
                if probe.get("rules_accepted") is False:
                    payload["action_required"] = (
                        "Join the ImageNet competition and accept its rules on Kaggle before "
                        "using it as a notebook datasource: " + IMAGENET_KAGGLE_RULES_URL
                    )
        except Exception as exc:
            payload["access_checked"] = True
            payload["competition_access"] = False
            payload["competition_listing_access"] = False
            payload["access_output"] = f"{type(exc).__name__}: {exc}"
    return payload


def install_optional_dataset_dependencies(
    *,
    install_coco: bool = True,
    install_kaggle: bool = True,
    upgrade: bool = True,
    log: LogFn = None,
) -> dict[str, Any]:
    packages: list[str] = []
    if install_coco:
        packages.append("pycocotools>=2.0.7")
    if install_kaggle:
        packages.append("kaggle>=1.6")
    if not packages:
        return {"status": "ok", "installed": [], "python": sys.executable}
    pip_probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if pip_probe.returncode != 0:
        raise RuntimeError(
            "pip is not available in the Python environment that runs the GUI. "
            f"Repair that environment first: {sys.executable}. Details: "
            f"{str(pip_probe.stdout or '').strip()}"
        )
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(packages)
    _log(log, f"[datasets] installing into active GUI environment: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    output_lines: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        output_lines.append(line)
        _log(log, f"[pip] {line}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"optional dataset dependency installation failed with exit code {rc}. "
            f"Run with the same interpreter shown in the log: {sys.executable}"
        )
    importlib.invalidate_caches()
    status = kaggle_cli_status()
    return {
        "status": "ok",
        "installed": packages,
        "python": sys.executable,
        "kaggle_cli": status,
        "output_tail": output_lines[-20:],
    }


def _run_kaggle(
    command: Sequence[str],
    args: Sequence[str],
    *,
    log: LogFn = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = list(command) + [str(x) for x in args]
    _log(log, f"[kaggle] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if proc.stdout:
        for line in proc.stdout.splitlines():
            _log(log, f"[kaggle] {line}")
    return proc


def _kaggle_competition_files(
    command: Sequence[str],
    competition: str,
    *,
    log: LogFn = None,
) -> tuple[list[str], str]:
    variants = [
        ["competitions", "files", competition, "--page-size", "20", "-v", "-q"],
        ["competitions", "files", "-c", competition, "--page-size", "20", "-v", "-q"],
    ]
    outputs: list[str] = []
    for args in variants:
        proc = _run_kaggle(command, args, log=log, timeout=120.0)
        output = str(proc.stdout or "")
        outputs.append(output)
        if proc.returncode != 0:
            continue
        names: list[str] = []
        for line in output.splitlines():
            stripped = line.strip().strip('"')
            if not stripped:
                continue
            # CSV output starts with a file-name column; tolerate both comma and
            # whitespace tables from older Kaggle clients.
            first = stripped.split(",", 1)[0].strip().strip('"')
            if first.lower() in {"name", "filename", "file", "ref"}:
                continue
            if any(token in first.lower() for token in ("imagenet", "ilsvrc", "loc_")):
                names.append(first)
        if names:
            return sorted(set(names)), output
        # A successful empty listing is still useful for a precise error below.
        return [], output
    raise RuntimeError(
        "Kaggle CLI could not list competition files. Check authentication and "
        "accept the competition rules in your own Kaggle account.\n" + "\n".join(outputs[-2:])
    )


def _kaggle_download_file(
    command: Sequence[str],
    competition: str,
    filename: str,
    destination: Path,
    *,
    log: LogFn = None,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    exact = destination / Path(filename).name
    if exact.is_file() and exact.stat().st_size > 0:
        _log(log, f"[kaggle] reuse {exact}")
        return exact
    variants = [
        ["competitions", "download", competition, "-f", filename, "-p", str(destination), "-o"],
        ["competitions", "download", "-c", competition, "-f", filename, "-p", str(destination), "-o"],
    ]
    for args in variants:
        proc = _run_kaggle(command, args, log=log, timeout=None)
        if proc.returncode == 0:
            if exact.is_file() and exact.stat().st_size > 0:
                return exact
            # Some Kaggle versions wrap a requested file in an additional ZIP.
            wrappers = sorted(
                [
                    p
                    for p in destination.glob("*.zip")
                    if p.is_file() and p.name.startswith(Path(filename).name)
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for wrapper in wrappers:
                if zipfile.is_zipfile(wrapper):
                    _safe_extract_zip(wrapper, destination, log=log)
                    if exact.is_file() and exact.stat().st_size > 0:
                        return exact
            candidates = sorted(
                [
                    p
                    for p in destination.iterdir()
                    if p.is_file() and p.name.startswith(Path(filename).name)
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
    raise RuntimeError(f"Kaggle CLI failed to download competition file: {filename}")


def _kaggle_download_all(
    command: Sequence[str],
    competition: str,
    destination: Path,
    *,
    log: LogFn = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    variants = [
        ["competitions", "download", competition, "-p", str(destination), "-o"],
        ["competitions", "download", "-c", competition, "-p", str(destination), "-o"],
    ]
    for args in variants:
        proc = _run_kaggle(command, args, log=log, timeout=None)
        if proc.returncode == 0:
            return
    raise RuntimeError("Kaggle CLI failed to download the ImageNet competition archive")




def _normalise_kaggle_username(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urllib.parse.urlparse(raw)
        parts = [part for part in parsed.path.split("/") if part]
        raw = parts[0] if parts else ""
    raw = raw.lstrip("@").strip()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{1,49}", raw):
        raise ValueError(
            "Kaggle username must contain only letters, digits, '_' or '-' and must "
            "be 2-50 characters long"
        )
    return raw


def resolve_kaggle_username(
    explicit: str | None = None,
    *,
    command: Sequence[str] | None = None,
    log: LogFn = None,
) -> tuple[str, str]:
    """Resolve a public Kaggle username without reading or logging secret tokens.

    Access-token files intentionally contain only the secret token, so they do not
    expose the account name.  The resolver first uses an explicit value, then
    environment/config metadata, and finally a best-effort ``kernels list --mine``
    probe.  It never prints credential contents.
    """

    if str(explicit or "").strip():
        return _normalise_kaggle_username(explicit), "explicit"

    env_name = str(os.environ.get("KAGGLE_USERNAME") or "").strip()
    if env_name:
        return _normalise_kaggle_username(env_name), "environment"

    candidates: list[Path] = []
    config_dir = str(os.environ.get("KAGGLE_CONFIG_DIR") or "").strip()
    if config_dir:
        candidates.extend(
            [
                Path(config_dir).expanduser() / "credentials.json",
                Path(config_dir).expanduser() / "kaggle.json",
            ]
        )
    candidates.extend(
        [
            Path.home() / ".kaggle" / "credentials.json",
            Path.home() / ".kaggle" / "kaggle.json",
            Path.home() / ".config" / "kaggle" / "credentials.json",
            Path.home() / ".config" / "kaggle" / "kaggle.json",
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        username = str((payload or {}).get("username") or "").strip()
        if username:
            return _normalise_kaggle_username(username), f"config:{path}"

    resolved_command = list(command) if command else None
    if resolved_command is None:
        resolved_command, _ = resolve_kaggle_cli()
    if resolved_command:
        # Legacy/OAuth configurations may expose a username through config view.
        proc = _run_kaggle(
            resolved_command,
            ["config", "view"],
            log=None,
            timeout=30.0,
        )
        if proc.returncode == 0:
            for line in str(proc.stdout or "").splitlines():
                match = re.search(r"\busername\b\s*[:=]\s*([^\s]+)", line, re.I)
                if match:
                    try:
                        return _normalise_kaggle_username(match.group(1)), "kaggle-config"
                    except ValueError:
                        pass

        # A user with at least one kernel can be resolved from its ref.  A user
        # without kernels must enter the public username explicitly in the GUI.
        proc = _run_kaggle(
            resolved_command,
            ["kernels", "list", "--mine", "--page-size", "1", "-v", "-q"],
            log=None,
            timeout=60.0,
        )
        if proc.returncode == 0:
            for line in str(proc.stdout or "").splitlines():
                first = line.strip().strip('"').split(",", 1)[0].strip().strip('"')
                if "/" not in first or first.lower() in {"ref", "id"}:
                    continue
                owner = first.split("/", 1)[0]
                try:
                    return _normalise_kaggle_username(owner), "kernels-list"
                except ValueError:
                    continue

    _log(
        log,
        "[kaggle] username could not be inferred from the access-token file; "
        "enter the public Kaggle username in the ImageNet provisioning tab",
    )
    return "", "unresolved"


def _imagenet_export_template_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "resources"
        / "templates"
        / "kaggle_imagenet_val_export.py.txt"
    )


def _imagenet_calibration_export_template_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "resources"
        / "templates"
        / "kaggle_imagenet_train_calibration_export.py.txt"
    )


def _write_imagenet_export_kernel(
    destination: Path,
    *,
    username: str,
    competition: str,
    kernel_slug: str = IMAGENET_EXPORT_KERNEL_SLUG,
    expected_image_count: int = IMAGENET_VALIDATION_EXPECTED_IMAGES,
    chunk_mib: int = IMAGENET_EXPORT_CHUNK_MIB,
) -> dict[str, Any]:
    username = _normalise_kaggle_username(username)
    slug = str(kernel_slug or IMAGENET_EXPORT_KERNEL_SLUG).strip().lower()
    slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")
    if not slug or len(slug) > 80:
        raise ValueError("invalid Kaggle kernel slug")
    destination.mkdir(parents=True, exist_ok=True)
    template_path = _imagenet_export_template_path()
    if not template_path.is_file():
        raise FileNotFoundError(f"ImageNet Kaggle export template is missing: {template_path}")
    source = template_path.read_text(encoding="utf-8")
    source = source.replace("__COMPETITION_SLUG_JSON__", json.dumps(str(competition)))
    source = source.replace("__EXPECTED_IMAGE_COUNT__", str(int(expected_image_count)))
    source = source.replace("__CHUNK_BYTES__", str(int(chunk_mib) * 1024 * 1024))
    code_file = destination / "export_imagenet_validation.py"
    code_file.write_text(source, encoding="utf-8")

    kernel_ref = f"{username}/{slug}"
    metadata = {
        "id": kernel_ref,
        # Keep the title-derived slug identical to the explicit id.  Kaggle warns
        # (and can behave surprisingly) when these diverge.
        "title": slug.replace("-", " "),
        "code_file": code_file.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "competition_sources": [str(competition)],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path = destination / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    provenance = {
        "schema": "onnx-splitpoint/kaggle-imagenet-export-job",
        "schema_version": 1,
        "created_at": now_iso(),
        "kernel_ref": kernel_ref,
        "competition": str(competition),
        "expected_image_count": int(expected_image_count),
        "chunk_mib": int(chunk_mib),
        "code_sha256": sha256_file(code_file),
        "metadata_sha256": sha256_file(metadata_path),
    }
    write_json(destination / "export_job_manifest.json", provenance)
    return {
        "kernel_ref": kernel_ref,
        "kernel_slug": slug,
        "work_dir": str(destination),
        "code_file": str(code_file),
        "metadata": str(metadata_path),
        "provenance": provenance,
    }


def _write_imagenet_calibration_export_kernel(
    destination: Path,
    *,
    username: str,
    competition: str,
    calibration_items: int,
    seed: int,
    kernel_slug: str = IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG,
    chunk_mib: int = IMAGENET_EXPORT_CHUNK_MIB,
) -> dict[str, Any]:
    """Generate the private Kaggle job that exports only train calibration images."""

    username = _normalise_kaggle_username(username)
    slug = str(kernel_slug or IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG).strip().lower()
    slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")
    if not slug or len(slug) > 80:
        raise ValueError("invalid Kaggle calibration kernel slug")
    destination.mkdir(parents=True, exist_ok=True)
    template_path = _imagenet_calibration_export_template_path()
    if not template_path.is_file():
        raise FileNotFoundError(
            f"ImageNet Kaggle calibration export template is missing: {template_path}"
        )
    source = template_path.read_text(encoding="utf-8")
    source = source.replace("__COMPETITION_SLUG_JSON__", json.dumps(str(competition)))
    source = source.replace("__CALIBRATION_ITEMS__", str(max(1, int(calibration_items))))
    source = source.replace("__SELECTION_SEED__", str(int(seed)))
    source = source.replace("__CHUNK_BYTES__", str(int(chunk_mib) * 1024 * 1024))
    code_file = destination / "export_imagenet_train_calibration.py"
    code_file.write_text(source, encoding="utf-8")
    kernel_ref = f"{username}/{slug}"
    metadata = {
        "id": kernel_ref,
        "title": slug.replace("-", " "),
        "code_file": code_file.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "competition_sources": [str(competition)],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path = destination / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    provenance = {
        "schema": "onnx-splitpoint/kaggle-imagenet-calibration-export-job",
        "schema_version": 1,
        "created_at": now_iso(),
        "kernel_ref": kernel_ref,
        "competition": str(competition),
        "calibration_items": max(1, int(calibration_items)),
        "selection_seed": int(seed),
        "chunk_mib": int(chunk_mib),
        "code_sha256": sha256_file(code_file),
        "metadata_sha256": sha256_file(metadata_path),
    }
    write_json(destination / "export_job_manifest.json", provenance)
    return {
        "kernel_ref": kernel_ref,
        "kernel_slug": slug,
        "work_dir": str(destination),
        "code_file": str(code_file),
        "metadata": str(metadata_path),
        "provenance": provenance,
    }


def _parse_kaggle_kernel_status(output: str) -> tuple[str, str]:
    """Normalise Kaggle CLI status strings across CLI generations.

    Kaggle CLI 2.2.x prints values such as ``KernelWorkerStatus.RUNNING`` and
    ``KernelWorkerStatus.ERROR``.  Older versions print plain ``running`` or
    ``complete``.  Returning the enum-qualified token caused v60g to continue
    polling an already failed kernel; v60h reduces all forms to one canonical
    terminal/running vocabulary.
    """

    text = str(output or "").strip()
    status = ""
    match = re.search(r"status\s+[\"']([^\"']+)[\"']", text, re.I)
    if match:
        status = match.group(1).strip().lower()
    if not status:
        match = re.search(
            r"\b(?:kernelworkerstatus\.)?"
            r"(queued|pending|running|complete|completed|success|error|failed|cancelled|canceled)\b",
            text,
            re.I,
        )
        if match:
            status = match.group(1).lower()
    if "." in status:
        status = status.rsplit(".", 1)[-1]
    aliases = {
        "completed": "complete",
        "success": "complete",
        "failed": "error",
        "canceled": "cancelled",
    }
    return aliases.get(status, status), text


def _assert_kaggle_kernel_push_succeeded(
    proc: subprocess.CompletedProcess[str],
    *,
    competition: str,
    kernel_ref: str,
) -> None:
    """Validate a Kaggle kernel push using return code *and* stdout.

    Kaggle CLI 2.2.x may emit ``Kernel push error: ...`` while returning zero.
    Continuing to poll in that case produces repeated, misleading
    ``kernels.get`` errors.  This guard converts the push response into a single
    actionable failure before any status polling begins.
    """

    output = str(proc.stdout or "").strip()
    lower = output.lower()
    failed = proc.returncode != 0 or _kaggle_output_has_error(output)
    if not failed:
        return
    if (
        "accept this competition's rules" in lower
        or "competitions.participate" in lower
    ):
        raise PermissionError(
            "Kaggle authentication and file listing work, but this account has not accepted "
            f"the rules required to attach '{competition}' as a notebook datasource. Open "
            f"{IMAGENET_KAGGLE_RULES_URL}, click Join Competition / I Understand and Accept, "
            "then run 'Check Kaggle setup/access' again. The private export kernel was not "
            f"created, so '{kernel_ref}' must not be polled. Kaggle output: {output[-2000:]}"
        )
    raise RuntimeError(
        "Could not create or update the private Kaggle validation-export kernel "
        f"'{kernel_ref}'. Kaggle output: {output[-4000:]}"
    )


def _decode_kaggle_log_text(path: Path) -> str:
    """Decode the JSON-stream log format returned by ``kaggle kernels output``."""

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    try:
        payload = json.loads(raw)
    except Exception:
        return raw
    if isinstance(payload, list):
        chunks: list[str] = []
        for row in payload:
            if isinstance(row, Mapping):
                chunks.append(str(row.get("data") or ""))
        return "".join(chunks)
    return raw


def _collect_kaggle_kernel_failure_diagnostics(
    command: Sequence[str],
    kernel_ref: str,
    destination: Path,
    *,
    status_output: str = "",
    log: LogFn = None,
) -> dict[str, Any]:
    """Download and summarise error output from a failed private Kaggle kernel.

    Kaggle does not place this log in the local application log directory.  It
    is a kernel output artefact and must be requested explicitly.  v60h performs
    that request automatically and leaves a stable local diagnostic directory.
    """

    destination = Path(destination).expanduser().resolve()
    result: dict[str, Any] = {
        "kernel_ref": str(kernel_ref),
        "destination": str(destination),
        "download_ok": False,
        "files": [],
        "traceback": "",
        "status_output": str(status_output or ""),
    }
    try:
        files = _download_kaggle_kernel_output(
            command,
            kernel_ref,
            destination,
            log=log,
        )
        result["download_ok"] = True
        result["files"] = [str(path) for path in files]
    except Exception as exc:
        result["download_error"] = repr(exc)
        destination.mkdir(parents=True, exist_ok=True)

    traceback_text = ""
    error_txt = list(destination.rglob("imagenet_export_error.txt"))
    if error_txt:
        traceback_text = error_txt[0].read_text(encoding="utf-8", errors="replace")
    if not traceback_text:
        error_json = list(destination.rglob("imagenet_export_error.json"))
        if error_json:
            try:
                payload = json.loads(error_json[0].read_text(encoding="utf-8"))
                traceback_text = str(payload.get("traceback") or payload.get("message") or "")
            except Exception:
                traceback_text = error_json[0].read_text(encoding="utf-8", errors="replace")
    if not traceback_text:
        logs = sorted(destination.rglob("*.log"))
        if logs:
            traceback_text = _decode_kaggle_log_text(logs[0])
    result["traceback"] = traceback_text

    summary_lines = [
        "ONNX Split-Point Tool — Kaggle ImageNet export failure",
        f"kernel: {kernel_ref}",
        f"diagnostic directory: {destination}",
        "",
        "Kaggle status output:",
        str(status_output or "(none)"),
        "",
        "Kernel traceback/log:",
        traceback_text or "No traceback artefact was returned by Kaggle.",
    ]
    summary = destination / "kaggle_imagenet_export_failure_summary.txt"
    summary.write_text("\n".join(summary_lines), encoding="utf-8")
    result["summary"] = str(summary)
    write_json(destination / "kaggle_imagenet_export_failure.json", result)
    _log(log, f"[kaggle] failure diagnostics saved to {summary}")
    return result


def _wait_for_kaggle_kernel(
    command: Sequence[str],
    kernel_ref: str,
    *,
    timeout_s: int = 10800,
    poll_interval_s: float = 20.0,
    failure_output_dir: Path | None = None,
    log: LogFn = None,
) -> dict[str, Any]:
    started = time.monotonic()
    last_output = ""
    consecutive_status_failures = 0
    while True:
        proc = _run_kaggle(
            command,
            ["kernels", "status", kernel_ref],
            log=log,
            timeout=120.0,
        )
        status, output = _parse_kaggle_kernel_status(str(proc.stdout or ""))
        last_output = output
        elapsed = time.monotonic() - started
        _log(log, f"[kaggle] export kernel status={status or 'unknown'} elapsed={elapsed:.0f}s")
        lower = output.lower()
        if (
            "cannot access kernel" in lower
            or "permission 'kernels.get' was denied" in lower
            or "wrong kernel slug" in lower
        ):
            raise RuntimeError(
                "Kaggle could not access the private export kernel after push. This normally "
                "means the push failed or the configured owner/slug is wrong. Status polling "
                f"has been stopped immediately. Kernel={kernel_ref}. Output: {output[-3000:]}"
            )

        if status in {"error", "cancelled"}:
            diagnostics: dict[str, Any] = {}
            if failure_output_dir is not None:
                diagnostics = _collect_kaggle_kernel_failure_diagnostics(
                    command,
                    kernel_ref,
                    Path(failure_output_dir),
                    status_output=output,
                    log=log,
                )
            trace = str(diagnostics.get("traceback") or "").strip()
            summary_path = str(diagnostics.get("summary") or "")
            details = trace[-6000:] if trace else output[-4000:]
            suffix = f" Local diagnostics: {summary_path}." if summary_path else ""
            raise RuntimeError(
                f"Kaggle validation export kernel ended with status {status}."
                f"{suffix} Kernel details:\n{details}"
            )

        if proc.returncode != 0 or _kaggle_output_has_error(output):
            consecutive_status_failures += 1
            if consecutive_status_failures >= 3:
                raise RuntimeError(
                    "Kaggle kernel status failed three consecutive times; polling was stopped. "
                    f"Kernel={kernel_ref}. Last output: {output[-3000:]}"
                )
        else:
            consecutive_status_failures = 0
        if proc.returncode == 0 and status == "complete":
            return {"status": status, "elapsed_s": elapsed, "output": output}
        if elapsed >= float(timeout_s):
            raise TimeoutError(
                f"Kaggle validation export kernel did not complete within {timeout_s}s. "
                f"Last status output: {last_output[-2000:]}"
            )
        time.sleep(max(0.05, float(poll_interval_s)))


def _download_kaggle_kernel_output(
    command: Sequence[str],
    kernel_ref: str,
    destination: Path,
    *,
    log: LogFn = None,
) -> list[Path]:
    # This directory is dedicated to one generated export. Remove stale files
    # recursively so a previous interrupted run cannot be mistaken for the latest
    # kernel output (Kaggle clients may preserve an additional output directory).
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    args = [
        "kernels",
        "output",
        kernel_ref,
        "-p",
        str(destination),
        "-o",
        "--page-size",
        "200",
    ]
    proc = _run_kaggle(command, args, log=log, timeout=None)
    if proc.returncode != 0:
        raise RuntimeError(
            "Kaggle CLI failed to download the private validation-export output: "
            + str(proc.stdout or "")[-4000:]
        )
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    return files


def collect_imagenet_kernel_diagnostics(
    *,
    root: str | Path | None = None,
    kaggle_username: str | None = None,
    kernel_slug: str = IMAGENET_EXPORT_KERNEL_SLUG,
    output_dir: str | Path | None = None,
    log: LogFn = None,
) -> dict[str, Any]:
    """Download the latest private ImageNet export log/error artefacts.

    This is the programmatic counterpart of ``kaggle kernels output`` and is
    exposed in the GUI so users do not need to know where Kaggle stores remote
    logs. No credential material is copied into the diagnostic directory.
    """

    command, version = resolve_kaggle_cli()
    if not command:
        raise RuntimeError(
            "Kaggle CLI is not installed in the Python environment that runs the GUI."
        )
    username, source = resolve_kaggle_username(
        kaggle_username, command=command, log=log
    )
    if not username:
        raise ValueError(
            "Kaggle username is required to locate the private export kernel."
        )
    slug = str(kernel_slug or IMAGENET_EXPORT_KERNEL_SLUG).strip().lower()
    slug = re.sub(r"[^a-z0-9-]+", "-", slug).strip("-")
    if not slug:
        raise ValueError("invalid Kaggle kernel slug")
    kernel_ref = f"{username}/{slug}"
    base = Path(root).expanduser().resolve() if root else default_dataset_root()
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else base / "downloads" / "imagenet_kaggle" / "private_validation_export_failure"
    )
    status_proc = _run_kaggle(
        command, ["kernels", "status", kernel_ref], log=log, timeout=120.0
    )
    status, status_output = _parse_kaggle_kernel_status(
        str(status_proc.stdout or "")
    )
    result = _collect_kaggle_kernel_failure_diagnostics(
        command,
        kernel_ref,
        destination,
        status_output=status_output,
        log=log,
    )
    result.update(
        {
            "status": status or "unknown",
            "kaggle_cli_version": version,
            "username_source": source,
        }
    )
    write_json(destination / "kaggle_imagenet_export_failure.json", result)
    return result


def _verify_and_reassemble_imagenet_export(
    output_dir: Path,
    *,
    log: LogFn = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "imagenet_val_export_manifest.json"
    if not manifest_path.is_file():
        # Some Kaggle clients preserve a single output subdirectory.
        candidates = list(output_dir.rglob("imagenet_val_export_manifest.json"))
        if len(candidates) == 1:
            manifest_path = candidates[0]
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Kaggle export manifest is missing under {output_dir}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(payload.get("schema") or "") != IMAGENET_EXPORT_SCHEMA:
        raise ValueError(f"unexpected ImageNet export schema in {manifest_path}")
    parts = list(payload.get("parts") or [])
    if not parts:
        raise ValueError("ImageNet export manifest contains no archive parts")
    parts = sorted(parts, key=lambda row: int((row or {}).get("index", 0)))
    base = manifest_path.parent
    verified_parts: list[Path] = []
    for expected_index, row in enumerate(parts):
        if int(row.get("index", -1)) != expected_index:
            raise ValueError("ImageNet export part indices are not contiguous")
        name = str(row.get("name") or "")
        part_path = base / name
        if not part_path.is_file():
            found = list(output_dir.rglob(name))
            if len(found) == 1:
                part_path = found[0]
        if not part_path.is_file():
            raise FileNotFoundError(f"missing ImageNet export part: {name}")
        expected_bytes = int(row.get("bytes") or 0)
        if expected_bytes and part_path.stat().st_size != expected_bytes:
            raise DatasetIntegrityError(
                f"size mismatch for {name}: expected {expected_bytes}, got {part_path.stat().st_size}"
            )
        expected_sha = str(row.get("sha256") or "").lower().removeprefix("sha256:")
        actual_sha = sha256_file(part_path).lower().removeprefix("sha256:")
        if not expected_sha or actual_sha != expected_sha:
            raise DatasetIntegrityError(
                f"SHA-256 mismatch for {name}: expected {expected_sha}, got {actual_sha}"
            )
        verified_parts.append(part_path)
        _log(log, f"[datasets] verified ImageNet export part {expected_index + 1}/{len(parts)}")

    archive_path = output_dir / str(payload.get("archive_name") or "ILSVRC2012_img_val.tar")
    archive_tmp = archive_path.with_suffix(archive_path.suffix + ".part")
    digest = hashlib.sha256()
    total = 0
    with archive_tmp.open("wb") as output:
        for part_path in verified_parts:
            with part_path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
    expected_total = int(payload.get("archive_bytes") or 0)
    expected_sha = str(payload.get("archive_sha256") or "").lower().removeprefix("sha256:")
    if expected_total and total != expected_total:
        archive_tmp.unlink(missing_ok=True)
        raise DatasetIntegrityError(
            f"reassembled ImageNet archive size mismatch: expected {expected_total}, got {total}"
        )
    actual_sha = digest.hexdigest().lower()
    if not expected_sha or actual_sha != expected_sha:
        archive_tmp.unlink(missing_ok=True)
        raise DatasetIntegrityError(
            f"reassembled ImageNet archive SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    archive_tmp.replace(archive_path)
    if not tarfile.is_tarfile(archive_path):
        raise DatasetIntegrityError(f"reassembled ImageNet archive is not a valid tar: {archive_path}")

    def verify_small_asset(key: str, fallback_name: str) -> Path:
        row = dict(payload.get(key) or {})
        name = str(row.get("name") or fallback_name)
        path = base / name
        if not path.is_file():
            found = list(output_dir.rglob(name))
            if len(found) == 1:
                path = found[0]
        if not path.is_file():
            raise FileNotFoundError(f"missing ImageNet export asset: {name}")
        expected = str(row.get("sha256") or "").lower().removeprefix("sha256:")
        actual = sha256_file(path).lower().removeprefix("sha256:")
        if expected and expected != actual:
            raise DatasetIntegrityError(
                f"SHA-256 mismatch for {name}: expected {expected}, got {actual}"
            )
        return path

    solution = verify_small_asset("solution", "LOC_val_solution.csv")
    mapping = verify_small_asset("synset_mapping", "LOC_synset_mapping.txt")
    expected_count = int(payload.get("image_count") or 0)
    if expected_count <= 0:
        raise ValueError("ImageNet export manifest has no positive image_count")
    _log(
        log,
        f"[datasets] reassembled verified ImageNet validation archive: "
        f"{archive_path.stat().st_size / 1024**3:.2f} GiB, images={expected_count}",
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "archive": str(archive_path),
        "archive_sha256": actual_sha,
        "solution": str(solution),
        "labels": str(mapping),
        "image_count": expected_count,
        "part_count": len(verified_parts),
        "part_paths": [str(path) for path in verified_parts],
    }


def _verify_and_reassemble_imagenet_calibration_export(
    output_dir: Path,
    *,
    log: LogFn = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "imagenet_train_calibration_export_manifest.json"
    if not manifest_path.is_file():
        candidates = list(output_dir.rglob("imagenet_train_calibration_export_manifest.json"))
        if len(candidates) == 1:
            manifest_path = candidates[0]
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"ImageNet calibration export manifest is missing under {output_dir}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(payload.get("schema") or "") != IMAGENET_CALIBRATION_EXPORT_SCHEMA:
        raise ValueError(f"unexpected ImageNet calibration export schema in {manifest_path}")
    parts = sorted(
        [dict(row) for row in list(payload.get("parts") or []) if isinstance(row, Mapping)],
        key=lambda row: int(row.get("index", 0)),
    )
    if not parts:
        raise ValueError("ImageNet calibration export manifest contains no archive parts")
    base = manifest_path.parent
    verified_parts: list[Path] = []
    for expected_index, row in enumerate(parts):
        if int(row.get("index", -1)) != expected_index:
            raise ValueError("ImageNet calibration export part indices are not contiguous")
        name = str(row.get("name") or "")
        part_path = base / name
        if not part_path.is_file():
            found = list(output_dir.rglob(name))
            if len(found) == 1:
                part_path = found[0]
        if not part_path.is_file():
            raise FileNotFoundError(f"missing ImageNet calibration export part: {name}")
        expected_bytes = int(row.get("bytes") or 0)
        if expected_bytes and part_path.stat().st_size != expected_bytes:
            raise DatasetIntegrityError(
                f"size mismatch for {name}: expected {expected_bytes}, got {part_path.stat().st_size}"
            )
        expected_sha = str(row.get("sha256") or "").lower().removeprefix("sha256:")
        actual_sha = sha256_file(part_path).lower().removeprefix("sha256:")
        if not expected_sha or expected_sha != actual_sha:
            raise DatasetIntegrityError(
                f"SHA-256 mismatch for {name}: expected {expected_sha}, got {actual_sha}"
            )
        verified_parts.append(part_path)
        _log(
            log,
            f"[datasets] verified ImageNet calibration export part "
            f"{expected_index + 1}/{len(parts)}",
        )

    archive_name = str(
        payload.get("archive_name") or "ILSVRC2012_img_train_calibration.tar"
    )
    archive_path = output_dir / archive_name
    archive_tmp = archive_path.with_suffix(archive_path.suffix + ".part")
    digest = hashlib.sha256()
    total = 0
    with archive_tmp.open("wb") as output:
        for part_path in verified_parts:
            with part_path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
    expected_total = int(payload.get("archive_bytes") or 0)
    expected_sha = str(payload.get("archive_sha256") or "").lower().removeprefix("sha256:")
    if expected_total and total != expected_total:
        archive_tmp.unlink(missing_ok=True)
        raise DatasetIntegrityError(
            f"reassembled ImageNet calibration archive size mismatch: expected {expected_total}, got {total}"
        )
    actual_sha = digest.hexdigest().lower()
    if not expected_sha or expected_sha != actual_sha:
        archive_tmp.unlink(missing_ok=True)
        raise DatasetIntegrityError(
            f"reassembled ImageNet calibration archive SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    archive_tmp.replace(archive_path)
    if not tarfile.is_tarfile(archive_path):
        raise DatasetIntegrityError(
            f"reassembled ImageNet calibration archive is not a valid tar: {archive_path}"
        )

    selection_row = dict(payload.get("selection") or {})
    selection_name = str(
        selection_row.get("name") or "imagenet_train_calibration_selection.json"
    )
    selection_path = base / selection_name
    if not selection_path.is_file():
        found = list(output_dir.rglob(selection_name))
        if len(found) == 1:
            selection_path = found[0]
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"missing ImageNet calibration selection manifest: {selection_name}"
        )
    expected_selection_sha = str(selection_row.get("sha256") or "").lower().removeprefix("sha256:")
    actual_selection_sha = sha256_file(selection_path).lower().removeprefix("sha256:")
    if expected_selection_sha and expected_selection_sha != actual_selection_sha:
        raise DatasetIntegrityError(
            f"ImageNet calibration selection SHA-256 mismatch: expected {expected_selection_sha}, got {actual_selection_sha}"
        )
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_count = int(payload.get("image_count") or 0)
    selected_count = int(selection_payload.get("selected_count") or len(selection_payload.get("items") or []))
    if expected_count <= 0 or selected_count != expected_count:
        raise ValueError(
            f"ImageNet calibration export count mismatch: manifest={expected_count}, selection={selected_count}"
        )
    _log(
        log,
        f"[datasets] reassembled verified ImageNet calibration archive: "
        f"{archive_path.stat().st_size / 1024**2:.1f} MiB, images={expected_count}",
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "archive": str(archive_path),
        "archive_sha256": actual_sha,
        "selection": str(selection_path),
        "selection_sha256": actual_selection_sha,
        "selection_payload": selection_payload,
        "image_count": expected_count,
        "class_count": int(selection_payload.get("selected_class_count") or 0),
        "part_count": len(verified_parts),
        "part_paths": [str(path) for path in verified_parts],
    }


def _remove_imagenet_export_parts(verified: Mapping[str, Any], *, log: LogFn = None) -> int:
    """Remove temporary chunks after verified reassembly and successful import.

    Chunk hashes remain in the export manifest and the reassembled tar remains
    available for reproduction. This avoids an unnecessary second ~6 GiB local
    copy of the validation archive.
    """

    removed = 0
    for raw in list(verified.get("part_paths") or []):
        path = Path(str(raw)).expanduser()
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        _log(log, f"[datasets] removed {removed} verified temporary ImageNet export chunks")
    return removed


def _provision_imagenet_via_private_kernel(
    *,
    command: Sequence[str],
    competition: str,
    username: str,
    kernel_slug: str,
    base: Path,
    download_dir: Path,
    calibration_items: int,
    seed: int,
    link_mode: str,
    registry_path: str | Path | None,
    kernel_timeout_s: int,
    kernel_poll_interval_s: float,
    log: LogFn,
) -> dict[str, Any]:
    username = _normalise_kaggle_username(username)
    kernel_dir = download_dir / "private_validation_export_kernel"
    job = _write_imagenet_export_kernel(
        kernel_dir,
        username=username,
        competition=competition,
        kernel_slug=kernel_slug,
    )
    kernel_ref = str(job["kernel_ref"])
    _log(
        log,
        "[datasets] Kaggle exposes ImageNet as an individual-file tree. "
        "Launching a private CPU kernel that packages only the 50,000 validation images; "
        "the full training/test bundle is not downloaded to this machine.",
    )
    push = _run_kaggle(
        command,
        ["kernels", "push", "-p", str(kernel_dir), "-t", str(int(kernel_timeout_s))],
        log=log,
        timeout=300.0,
    )
    _assert_kaggle_kernel_push_succeeded(
        push,
        competition=competition,
        kernel_ref=kernel_ref,
    )
    failure_output_dir = download_dir / "private_validation_export_failure"
    status = _wait_for_kaggle_kernel(
        command,
        kernel_ref,
        timeout_s=int(kernel_timeout_s),
        poll_interval_s=float(kernel_poll_interval_s),
        failure_output_dir=failure_output_dir,
        log=log,
    )
    output_dir = download_dir / "private_validation_export_output"
    downloaded = _download_kaggle_kernel_output(
        command,
        kernel_ref,
        output_dir,
        log=log,
    )
    verified = _verify_and_reassemble_imagenet_export(output_dir, log=log)
    result = import_imagenet(
        source=verified["archive"],
        root=base,
        calibration_items=calibration_items,
        seed=seed,
        link_mode=link_mode,
        registry_path=registry_path,
        validation_solution=verified["solution"],
        labels=verified["labels"],
        source_name="kaggle_private_validation_export",
        log=log,
    )
    removed_parts = _remove_imagenet_export_parts(verified, log=log)
    retained_output_files = [str(path) for path in downloaded if Path(path).is_file()]
    verified["transport_parts_removed"] = bool(removed_parts)
    result.update(
        {
            "terms_attested": True,
            "download_mode": "validation_via_private_kernel",
            "kaggle_username": username,
            "kernel_ref": kernel_ref,
            "kernel_status": status,
            "kernel_job": job,
            "export": verified,
            "downloaded_output_file_count": len(downloaded),
            "retained_output_files": retained_output_files,
            "temporary_export_parts_removed": removed_parts,
            "calibration_manifest_created": False,
            "calibration_note": (
                "A disjoint train-derived calibration subset must be imported separately; "
                "the private export contains only the official validation split."
            ),
        }
    )
    return result


def _provision_imagenet_calibration_via_private_kernel(
    *,
    command: Sequence[str],
    competition: str,
    username: str,
    kernel_slug: str,
    base: Path,
    download_dir: Path,
    calibration_items: int,
    seed: int,
    registry_path: str | Path | None,
    kernel_timeout_s: int,
    kernel_poll_interval_s: float,
    log: LogFn,
) -> dict[str, Any]:
    username = _normalise_kaggle_username(username)
    kernel_dir = download_dir / "private_calibration_export_kernel"
    job = _write_imagenet_calibration_export_kernel(
        kernel_dir,
        username=username,
        competition=competition,
        calibration_items=calibration_items,
        seed=seed,
        kernel_slug=kernel_slug,
    )
    kernel_ref = str(job["kernel_ref"])
    _log(
        log,
        "[datasets] launching a private CPU kernel that exports only the deterministic "
        f"ImageNet train calibration subset ({max(1, int(calibration_items))} images); "
        "the complete training corpus is not downloaded to this machine.",
    )
    push = _run_kaggle(
        command,
        ["kernels", "push", "-p", str(kernel_dir), "-t", str(int(kernel_timeout_s))],
        log=log,
        timeout=300.0,
    )
    _assert_kaggle_kernel_push_succeeded(
        push,
        competition=competition,
        kernel_ref=kernel_ref,
    )
    status = _wait_for_kaggle_kernel(
        command,
        kernel_ref,
        timeout_s=int(kernel_timeout_s),
        poll_interval_s=float(kernel_poll_interval_s),
        failure_output_dir=download_dir / "private_calibration_export_failure",
        log=log,
    )
    output_dir = download_dir / "private_calibration_export_output"
    downloaded = _download_kaggle_kernel_output(
        command,
        kernel_ref,
        output_dir,
        log=log,
    )
    verified = _verify_and_reassemble_imagenet_calibration_export(
        output_dir,
        log=log,
    )
    selected_count = int(verified["image_count"])
    target = (
        base
        / "imagenet2012"
        / "prepared"
        / f"train_calibration_n{selected_count}_s{int(seed)}"
    )
    if target.exists():
        shutil.rmtree(target)
    _safe_extract_tar(Path(verified["archive"]), target, log=log)
    class_dirs = _classification_class_dirs(target)
    extracted_count = sum(
        1
        for path in target.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if extracted_count != selected_count:
        raise DatasetIntegrityError(
            f"ImageNet calibration extraction count mismatch: expected {selected_count}, got {extracted_count}"
        )
    if not class_dirs:
        raise DatasetIntegrityError(
            f"ImageNet calibration export is not class-organised: {target}"
        )

    registry_file = (
        Path(registry_path).expanduser().resolve()
        if registry_path
        else default_registry_path(base)
    )
    registry = load_registry(registry_file)
    validation_record = dict(
        (registry.get("datasets") or {}).get("imagenet_validation") or {}
    )
    validation_root = Path(str(validation_record.get("root") or "")).expanduser()
    if not validation_root.is_dir():
        raise RuntimeError(
            "ImageNet validation must be provisioned before the train calibration subset"
        )
    labels_value = str(validation_record.get("labels") or "").strip()
    labels_path = Path(labels_value).expanduser() if labels_value else None
    _register_dataset(
        registry,
        "imagenet_calibration",
        root=target,
        labels=labels_path,
        source="kaggle_private_train_calibration_export",
        extra={
            "selection_manifest": str(verified["selection"]),
            "selection_manifest_sha256": str(verified["selection_sha256"]),
            "selection_seed": int(seed),
            "requested_items": int(calibration_items),
            "selected_items": selected_count,
            "selected_class_count": int(verified.get("class_count") or 0),
            "kernel_ref": kernel_ref,
            "source_archive_sha256": str(verified["archive_sha256"]),
        },
    )
    manifests_dir = base / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    calibration_manifest = create_dataset_manifest(
        task="classification",
        role="calibration",
        dataset_id="ilsvrc2012-train-calibration",
        split="train",
        root=target,
        labels=labels_path,
        output=manifests_dir / "imagenet_train_calibration_manifest.json",
        max_items=0,
        selection_strategy="sorted",
        selection_seed=seed,
    )
    selection_payload = dict(verified.get("selection_payload") or {})
    _annotate_dataset_manifest(
        calibration_manifest,
        provisioning_selection={
            "source_split": "train",
            "source_population": "ILSVRC2012 train",
            "strategy": str(selection_payload.get("strategy") or "class_stratified_deterministic_hash"),
            "seed": int(seed),
            "requested_items": int(calibration_items),
            "selected_items": selected_count,
            "selected_class_count": int(verified.get("class_count") or 0),
            "selection_manifest": str(verified["selection"]),
            "selection_manifest_sha256": str(verified["selection_sha256"]),
            "kernel_ref": kernel_ref,
            "selection_uses_model_predictions": False,
        },
    )
    registry.setdefault("manifests", {})["classification_calibration"] = str(
        calibration_manifest
    )
    settings = registry.setdefault("settings", {})
    settings["classification_calibration_items"] = int(calibration_items)
    settings["selection_seed"] = int(seed)
    settings["imagenet_include_calibration"] = True
    settings["imagenet_calibration_export_kernel_ref"] = kernel_ref
    saved = save_registry(registry, registry_file)

    validation_manifest_path = Path(
        str((registry.get("manifests") or {}).get("classification_validation") or "")
    ).expanduser()
    disjointness: dict[str, Any] | None = None
    if validation_manifest_path.is_file():
        calibration_payload = json.loads(
            calibration_manifest.read_text(encoding="utf-8")
        )
        validation_payload = json.loads(
            validation_manifest_path.read_text(encoding="utf-8")
        )
        disjointness = validate_calibration_validation_separation(
            [calibration_payload, validation_payload]
        )
        disjointness["requested_task"] = "classification"
        disjointness["requested_task_ok"] = _task_separation_pass(
            disjointness, "classification"
        )
        if not bool(disjointness["requested_task_ok"]):
            raise RuntimeError(
                "ImageNet calibration and validation manifests are not disjoint: "
                + json.dumps(disjointness, ensure_ascii=False)
            )

    removed_parts = _remove_imagenet_export_parts(verified, log=log)
    return {
        "status": "ok",
        "dataset": "imagenet",
        "role": "calibration",
        "root": str(target),
        "registry": str(saved),
        "manifest": str(calibration_manifest),
        "kernel_ref": kernel_ref,
        "kernel_status": status,
        "kernel_job": job,
        "export": verified,
        "downloaded_output_file_count": len(downloaded),
        "temporary_export_parts_removed": removed_parts,
        "calibration_validation_disjointness": disjointness,
    }


def _manifest_ready_for_requested_subset(
    manifest_path: Path,
    *,
    expected_items: int | None = None,
    expected_seed: int | None = None,
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        verification = verify_dataset_manifest(payload, verify_files=True)
        if not bool(verification.get("ok")):
            return False
        if expected_items is not None and int(payload.get("item_count") or 0) != int(expected_items):
            return False
        if expected_seed is not None:
            provenance = dict(payload.get("provisioning_selection") or {})
            selection = dict(payload.get("selection") or {})
            seed = provenance.get("seed", selection.get("seed"))
            if seed is None or int(seed) != int(expected_seed):
                return False
        return True
    except Exception:
        return False


def provision_imagenet_complete(
    *,
    root: str | Path | None = None,
    accept_terms: bool = False,
    include_calibration: bool = True,
    calibration_items: int = 1000,
    seed: int = DEFAULT_SEED,
    link_mode: str = "hardlink",
    registry_path: str | Path | None = None,
    competition: str = IMAGENET_KAGGLE_COMPETITION,
    download_mode: str = "validation_only",
    allow_large_download: bool = False,
    kaggle_username: str | None = None,
    validation_kernel_slug: str = IMAGENET_EXPORT_KERNEL_SLUG,
    calibration_kernel_slug: str = IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG,
    kernel_timeout_s: int = 10800,
    kernel_poll_interval_s: float = 20.0,
    log: LogFn = None,
) -> dict[str, Any]:
    """One-click ImageNet provisioning for validation and train calibration roles."""

    if not accept_terms:
        raise PermissionError(
            "ImageNet provisioning requires an explicit terms attestation after accepting "
            "the competition rules in the Kaggle account."
        )
    base = Path(root).expanduser().resolve() if root else default_dataset_root()
    registry_file = (
        Path(registry_path).expanduser().resolve()
        if registry_path
        else default_registry_path(base)
    )
    registry = load_registry(registry_file)
    validation_manifest = Path(
        str((registry.get("manifests") or {}).get("classification_validation") or "")
    ).expanduser()
    validation_record = dict(
        (registry.get("datasets") or {}).get("imagenet_validation") or {}
    )
    validation_root = Path(str(validation_record.get("root") or "")).expanduser()
    validation_ready = bool(
        validation_root.is_dir()
        and _manifest_ready_for_requested_subset(validation_manifest)
    )
    if validation_ready:
        _log(
            log,
            f"[datasets] reuse verified ImageNet validation set and manifest: {validation_root}",
        )
        validation_result: dict[str, Any] = {
            "status": "reused",
            "root": str(validation_root),
            "manifest": str(validation_manifest),
            "registry": str(registry_file),
        }
    else:
        validation_result = provision_imagenet_kaggle(
            root=base,
            accept_terms=True,
            calibration_items=calibration_items,
            seed=seed,
            link_mode=link_mode,
            registry_path=registry_file,
            competition=competition,
            download_mode=download_mode,
            allow_large_download=allow_large_download,
            kaggle_username=kaggle_username,
            kernel_slug=validation_kernel_slug,
            kernel_timeout_s=kernel_timeout_s,
            kernel_poll_interval_s=kernel_poll_interval_s,
            log=log,
        )

    calibration_result: dict[str, Any] = {"status": "not_requested"}
    if include_calibration:
        registry = load_registry(registry_file)
        calibration_manifest = Path(
            str((registry.get("manifests") or {}).get("classification_calibration") or "")
        ).expanduser()
        calibration_record = dict(
            (registry.get("datasets") or {}).get("imagenet_calibration") or {}
        )
        calibration_root = Path(
            str(calibration_record.get("root") or "")
        ).expanduser()
        calibration_ready = bool(
            calibration_root.is_dir()
            and _manifest_ready_for_requested_subset(
                calibration_manifest,
                expected_items=max(1, int(calibration_items)),
                expected_seed=int(seed),
            )
        )
        if calibration_ready:
            _log(
                log,
                f"[datasets] reuse verified ImageNet calibration subset and manifest: {calibration_root}",
            )
            calibration_result = {
                "status": "reused",
                "root": str(calibration_root),
                "manifest": str(calibration_manifest),
                "registry": str(registry_file),
            }
        else:
            command, version = resolve_kaggle_cli()
            if not command:
                raise RuntimeError(
                    "Kaggle CLI is required to export the ImageNet train calibration subset"
                )
            names, _ = _kaggle_competition_files(command, competition, log=log)
            _ensure_kaggle_competition_rules_accepted(
                command,
                competition,
                names,
                log=log,
            )
            username, source = resolve_kaggle_username(
                kaggle_username,
                command=command,
                log=log,
            )
            if not username:
                raise RuntimeError(
                    "Enter the public Kaggle username to own the private ImageNet calibration export kernel"
                )
            _log(
                log,
                f"[kaggle] calibration export owner={username} (source={source}, cli={version or 'unknown'})",
            )
            calibration_result = _provision_imagenet_calibration_via_private_kernel(
                command=command,
                competition=competition,
                username=username,
                kernel_slug=calibration_kernel_slug,
                base=base,
                download_dir=base / "downloads" / "imagenet_kaggle",
                calibration_items=calibration_items,
                seed=seed,
                registry_path=registry_file,
                kernel_timeout_s=int(kernel_timeout_s),
                kernel_poll_interval_s=float(kernel_poll_interval_s),
                log=log,
            )

    registry = load_registry(registry_file)
    settings = registry.setdefault("settings", {})
    settings["imagenet_include_calibration"] = bool(include_calibration)
    settings["classification_calibration_items"] = int(calibration_items)
    settings["selection_seed"] = int(seed)
    settings["imagenet_calibration_export_kernel_ref"] = (
        str(calibration_result.get("kernel_ref") or settings.get("imagenet_calibration_export_kernel_ref") or "")
    )
    saved = save_registry(registry, registry_file)
    status = registry_status(saved, verify_manifests=True)
    return {
        "status": "ok",
        "dataset": "imagenet",
        "provisioning_scope": (
            "validation_and_calibration" if include_calibration else "validation_only"
        ),
        "registry": str(saved),
        "validation": validation_result,
        "calibration": calibration_result,
        "manifests": dict(load_registry(saved).get("manifests") or {}),
        "readiness": status,
    }

def provision_imagenet_kaggle(
    *,
    root: str | Path | None = None,
    accept_terms: bool = False,
    calibration_items: int = 1000,
    seed: int = DEFAULT_SEED,
    link_mode: str = "hardlink",
    registry_path: str | Path | None = None,
    competition: str = IMAGENET_KAGGLE_COMPETITION,
    download_mode: str = "validation_only",
    allow_large_download: bool = False,
    kaggle_username: str | None = None,
    kernel_slug: str = IMAGENET_EXPORT_KERNEL_SLUG,
    kernel_timeout_s: int = 10800,
    kernel_poll_interval_s: float = 20.0,
    log: LogFn = None,
) -> dict[str, Any]:
    """Provision ImageNet through the authenticated Kaggle account.

    ``validation_only`` is the safe smart mode.  It first reuses separately
    downloadable validation artefacts when Kaggle exposes them.  When the
    competition is exposed as the current million-file tree, it falls back to a
    private Kaggle CPU kernel that packages only the 50,000 validation images and
    their labels.  The full competition bundle is downloaded only in the explicit
    ``full_competition`` mode.
    """

    if not accept_terms:
        raise PermissionError(
            "ImageNet provisioning requires an explicit terms attestation after accepting "
            "the dataset/competition terms in your own account."
        )
    mode = str(download_mode or "validation_only").strip()
    if mode not in IMAGENET_KAGGLE_DOWNLOAD_MODES:
        raise ValueError(
            "download_mode must be one of: " + ", ".join(IMAGENET_KAGGLE_DOWNLOAD_MODES)
        )
    command, version = resolve_kaggle_cli()
    if not command:
        raise RuntimeError(
            "Kaggle CLI is not installed in the Python environment that runs the GUI. "
            f"Use the GUI button 'Install/repair dataset support' or run: "
            f"\"{sys.executable}\" -m pip install --upgrade \"kaggle>=1.6\""
        )
    base = Path(root).expanduser().resolve() if root else default_dataset_root()
    download_dir = base / "downloads" / "imagenet_kaggle"
    download_dir.mkdir(parents=True, exist_ok=True)
    cli_state = kaggle_cli_status()
    _log(
        log,
        f"[datasets] Kaggle CLI detected ({version or 'version unknown'}); "
        "credentials are read by Kaggle and are not stored by this tool",
    )
    if not bool(cli_state.get("authentication_configured_hint")):
        _log(
            log,
            "[datasets] no standard Kaggle credential file or environment token was "
            "detected. The API request will still be attempted, but you may need to "
            "configure Kaggle authentication and accept the competition rules first.",
        )

    if mode == "full_competition":
        if not allow_large_download:
            raise PermissionError(
                "The full ImageNet competition download is very large. Set "
                "allow_large_download=True only after explicitly confirming the storage cost."
            )
        _kaggle_download_all(command, competition, download_dir, log=log)
        archives = sorted(
            [
                path
                for path in download_dir.iterdir()
                if path.is_file() and (zipfile.is_zipfile(path) or tarfile.is_tarfile(path))
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not archives:
            raise FileNotFoundError(
                f"Kaggle CLI completed but no archive was found in {download_dir}"
            )
        result = import_imagenet(
            source=archives[0],
            root=base,
            calibration_items=calibration_items,
            seed=seed,
            link_mode=link_mode,
            registry_path=registry_path,
            source_name="kaggle_full_competition",
            log=log,
        )
        result.update(
            {
                "source_archive": str(archives[0]),
                "terms_attested": True,
                "download_mode": mode,
            }
        )
        _record_imagenet_provisioning_settings(
            result.get("registry"),
            download_mode=mode,
            calibration_items=calibration_items,
            seed=seed,
        )
        return result

    if mode == "validation_via_private_kernel":
        names, _ = _kaggle_competition_files(command, competition, log=log)
        _ensure_kaggle_competition_rules_accepted(
            command,
            competition,
            names,
            log=log,
        )
        username, username_source = resolve_kaggle_username(
            kaggle_username, command=command, log=log
        )
        if not username:
            raise RuntimeError(
                "Kaggle authentication and ImageNet competition access work, but the public "
                "Kaggle username cannot be inferred from an access_token file. Enter the "
                "username shown in your Kaggle profile URL in the ImageNet provisioning tab "
                "or pass --kaggle-username. No token is stored in the tool configuration."
            )
        _log(log, f"[kaggle] kernel owner={username} (source={username_source})")
        result = _provision_imagenet_via_private_kernel(
            command=command,
            competition=competition,
            username=username,
            kernel_slug=kernel_slug,
            base=base,
            download_dir=download_dir,
            calibration_items=calibration_items,
            seed=seed,
            link_mode=link_mode,
            registry_path=registry_path,
            kernel_timeout_s=int(kernel_timeout_s),
            kernel_poll_interval_s=float(kernel_poll_interval_s),
            log=log,
        )
        _record_imagenet_provisioning_settings(
            result.get("registry"),
            download_mode="validation_via_private_kernel",
            calibration_items=calibration_items,
            seed=seed,
            kaggle_username=username,
            kernel_ref=str(result.get("kernel_ref") or ""),
        )
        return result

    names, raw_listing = _kaggle_competition_files(command, competition, log=log)
    lower_map = {name.lower(): name for name in names}

    def find_suffix(*suffixes: str) -> str | None:
        for lower, original in lower_map.items():
            if any(lower.endswith(suffix.lower()) for suffix in suffixes):
                return original
        return None

    selected = [
        find_suffix("ILSVRC2012_img_val.tar"),
        find_suffix("LOC_val_solution.csv"),
        find_suffix("LOC_synset_mapping.txt", "synset_words.txt"),
    ]
    direct_available = bool(selected[0] and selected[1])
    individual_tree = any(
        "/annotations/cls-loc/" in name.lower()
        or "/data/cls-loc/" in name.lower()
        for name in names
    )

    if not direct_available:
        if mode == "direct_validation_files":
            visible = ", ".join(names[:12]) if names else "<no file names returned>"
            raise RuntimeError(
                "Kaggle currently exposes this competition as individual files rather than "
                "as ILSVRC2012_img_val.tar plus LOC_val_solution.csv. Direct validation-file "
                "mode therefore cannot be used. Select validation_only or "
                "validation_via_private_kernel, which packages only the validation split in a "
                "private Kaggle kernel. Visible first-page entries: " + visible
            )

        username, username_source = resolve_kaggle_username(
            kaggle_username, command=command, log=log
        )
        if username:
            _ensure_kaggle_competition_rules_accepted(
                command,
                competition,
                names,
                log=log,
            )
            _log(
                log,
                f"[kaggle] no standalone validation tar on the first competition page; "
                f"using private validation export owned by {username} "
                f"(username source={username_source}, individual_tree={individual_tree})",
            )
            result = _provision_imagenet_via_private_kernel(
                command=command,
                competition=competition,
                username=username,
                kernel_slug=kernel_slug,
                base=base,
                download_dir=download_dir,
                calibration_items=calibration_items,
                seed=seed,
                link_mode=link_mode,
                registry_path=registry_path,
                kernel_timeout_s=int(kernel_timeout_s),
                kernel_poll_interval_s=float(kernel_poll_interval_s),
                log=log,
            )
            _record_imagenet_provisioning_settings(
                result.get("registry"),
                download_mode="validation_via_private_kernel",
                calibration_items=calibration_items,
                seed=seed,
                kaggle_username=username,
                kernel_ref=str(result.get("kernel_ref") or ""),
            )
            return result

        visible = ", ".join(names[:12]) if names else "<no file names returned>"
        next_page_note = ""
        if "next page token" in raw_listing.lower():
            next_page_note = (
                " The listing has more pages, but walking and downloading 50,000 images as "
                "individual API calls would be slow and rate-limit prone."
            )
        raise RuntimeError(
            "Kaggle authentication and competition access are valid. The competition is "
            "exposed as an individual-file tree, not as a separately downloadable validation "
            "archive. Enter your public Kaggle username in the GUI; validation_only will then "
            "create a private Kaggle CPU kernel that exports only the 50,000 validation images "
            "in chunked, hash-verified form. The full 160+ GB bundle is not downloaded."
            + next_page_note
            + " Visible first-page entries: "
            + visible
        )

    selected_dir = download_dir / "direct_validation_files"
    downloaded: dict[str, Path] = {}
    for filename in [item for item in selected if item]:
        downloaded[str(filename)] = _kaggle_download_file(
            command, competition, str(filename), selected_dir, log=log
        )

    val_name = str(selected[0])
    solution_name = str(selected[1])
    labels_name = str(selected[2]) if selected[2] else ""
    val_archive = downloaded.get(val_name)
    solution_path = downloaded.get(solution_name)
    labels_path = downloaded.get(labels_name) if labels_name else None
    if (
        not val_archive
        or not val_archive.is_file()
        or not solution_path
        or not solution_path.is_file()
    ):
        raise FileNotFoundError(
            "Kaggle reported success, but the ImageNet validation archive or validation "
            "solution is missing from the download directory."
        )
    result = import_imagenet(
        source=val_archive,
        root=base,
        calibration_items=calibration_items,
        seed=seed,
        link_mode=link_mode,
        registry_path=registry_path,
        validation_solution=solution_path,
        labels=labels_path,
        source_name="kaggle_direct_validation_files",
        log=log,
    )
    result.update(
        {
            "terms_attested": True,
            "download_mode": "direct_validation_files",
            "downloaded_files": {name: str(path) for name, path in downloaded.items()},
            "calibration_manifest_created": False,
            "calibration_note": (
                "A disjoint train-derived calibration subset must be imported separately; "
                "the full training archive was intentionally not downloaded."
            ),
        }
    )
    _record_imagenet_provisioning_settings(
        result.get("registry"),
        download_mode="direct_validation_files",
        calibration_items=calibration_items,
        seed=seed,
    )
    return result


def _record_imagenet_provisioning_settings(
    registry_path: Any,
    *,
    download_mode: str,
    calibration_items: int,
    seed: int,
    kaggle_username: str = "",
    kernel_ref: str = "",
) -> None:
    if not registry_path:
        return
    path = Path(str(registry_path)).expanduser()
    registry = load_registry(path)
    settings = registry.setdefault("settings", {})
    settings["imagenet_download_mode"] = str(download_mode)
    settings["classification_calibration_items"] = int(calibration_items)
    settings["selection_seed"] = int(seed)
    if str(kaggle_username or "").strip():
        settings["kaggle_username"] = str(kaggle_username).strip()
    if str(kernel_ref or "").strip():
        settings["imagenet_validation_export_kernel_ref"] = str(kernel_ref).strip()
    save_registry(registry, path)

def registry_status(path: str | Path | None = None, *, verify_manifests: bool = False) -> dict[str, Any]:
    registry = load_registry(path)
    datasets: dict[str, Any] = {}
    for key, raw in dict(registry.get("datasets") or {}).items():
        row = dict(raw or {}) if isinstance(raw, Mapping) else {}
        root = Path(str(row.get("root") or "")).expanduser()
        ann = Path(str(row.get("annotations") or "")).expanduser() if row.get("annotations") else None
        labels = Path(str(row.get("labels") or "")).expanduser() if row.get("labels") else None
        datasets[str(key)] = {
            "root": str(root),
            "root_exists": root.is_dir(),
            "annotations": str(ann) if ann else "",
            "annotations_exists": bool(ann and ann.is_file()),
            "labels": str(labels) if labels else "",
            "labels_exists": bool(labels and labels.is_file()),
        }

    manifests: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for key, value in dict(registry.get("manifests") or {}).items():
        p = Path(str(value or "")).expanduser()
        row: dict[str, Any] = {"path": str(p), "exists": p.is_file()}
        if p.is_file():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                payloads[str(key)] = dict(payload)
                if verify_manifests:
                    row["verification"] = verify_dataset_manifest(payload, verify_files=True)
            except Exception as exc:
                row["load_error"] = f"{type(exc).__name__}: {exc}"
                if verify_manifests:
                    row["verification"] = {
                        "status": "error",
                        "ok": False,
                        "error": row["load_error"],
                    }
        manifests[str(key)] = row

    task_readiness: dict[str, Any] = {}
    for task in ("classification", "detection"):
        calibration_key, validation_key = _task_manifest_keys(task)
        required_keys = [calibration_key, validation_key]
        missing = [
            key for key in required_keys
            if not bool((manifests.get(key) or {}).get("exists"))
        ]
        task_payloads = [payloads[key] for key in required_keys if key in payloads]
        separation = _task_separation_summary(task_payloads, task)
        verification_ok = True
        if verify_manifests:
            verification_ok = (
                not missing
                and all(
                    bool(((manifests.get(key) or {}).get("verification") or {}).get("ok"))
                    for key in required_keys
                )
            )
        dataset_key_prefix = "imagenet" if task == "classification" else "coco2017"
        dataset_keys = [
            f"{dataset_key_prefix}_calibration",
            f"{dataset_key_prefix}_validation",
        ]

        # The content-addressed manifests are the authoritative campaign
        # evidence.  Older registries and hand-created test registries may
        # contain valid manifests without duplicate ``datasets`` records.
        # Accept an existing manifest root as a recoverable dataset binding;
        # ``repair-registry`` can materialise the convenience records later
        # without network access.
        dataset_binding_status: dict[str, Any] = {}
        datasets_ok = True
        for role_index, (dataset_key, manifest_key) in enumerate(
            zip(dataset_keys, required_keys)
        ):
            registry_root_ok = bool((datasets.get(dataset_key) or {}).get("root_exists"))
            payload = payloads.get(manifest_key) or {}
            manifest_root = Path(str(payload.get("root") or "")).expanduser()
            manifest_root_ok = manifest_root.is_dir()
            binding_ok = bool(registry_root_ok or manifest_root_ok)
            datasets_ok = bool(datasets_ok and binding_ok)
            dataset_binding_status[dataset_key] = {
                "ok": binding_ok,
                "registry_root_ok": registry_root_ok,
                "manifest_root": str(manifest_root) if str(payload.get("root") or "") else "",
                "manifest_root_ok": manifest_root_ok,
                "source": (
                    "registry" if registry_root_ok
                    else "manifest" if manifest_root_ok
                    else "missing"
                ),
            }
        ready = bool(
            not missing
            and datasets_ok
            and separation.get("ok")
            and verification_ok
        )
        task_readiness[task] = {
            "task": task,
            "ready": ready,
            "required_manifests": required_keys,
            "missing_manifests": missing,
            "datasets_ok": datasets_ok,
            "dataset_bindings": dataset_binding_status,
            "verification_requested": bool(verify_manifests),
            "verification_ok": bool(verification_ok),
            "calibration_validation_disjointness": separation,
        }

    required = [
        "classification_calibration",
        "classification_validation",
        "detection_calibration",
        "detection_validation",
    ]
    all_payloads = [payloads[key] for key in required if key in payloads]
    global_separation = validate_calibration_validation_separation(all_payloads)
    global_separation["status"] = (
        "ok" if bool(global_separation.get("ok")) else "incomplete_or_overlap"
    )
    ready = all(bool(task_readiness[task]["ready"]) for task in task_readiness)
    missing_required = [
        key for key in required
        if not bool((manifests.get(key) or {}).get("exists"))
    ]
    return {
        "schema": "onnx-splitpoint/final-dataset-status",
        "schema_version": 2,
        "registry": str(Path(path).expanduser() if path else default_registry_path()),
        "ready_for_final_profile": ready,
        "ready_for_classification_profile": bool(task_readiness["classification"]["ready"]),
        "ready_for_detection_profile": bool(task_readiness["detection"]["ready"]),
        "task_readiness": task_readiness,
        "datasets": datasets,
        "manifests": manifests,
        "calibration_validation_disjointness": global_separation,
        "missing_required_manifests": missing_required,
    }


def repair_dataset_registry(
    *,
    root: str | Path | None = None,
    registry_path: str | Path | None = None,
    verify_manifests: bool = True,
    log: LogFn = None,
) -> dict[str, Any]:
    """Reindex already provisioned assets without downloading any dataset.

    This is intentionally safe after ``train2017.zip`` has been deleted.  The
    content-addressed manifests and materialised subset are the durable evidence;
    the transport archive is not part of registry readiness.
    """

    base = Path(root).expanduser().resolve() if root else default_dataset_root()
    registry_file = (
        Path(registry_path).expanduser().resolve()
        if registry_path
        else default_registry_path(base)
    )
    registry = load_registry(registry_file)
    manifest_dir = base / "manifests"
    known = {
        "classification_calibration": manifest_dir / "imagenet_train_calibration_manifest.json",
        "classification_validation": manifest_dir / "imagenet_val_manifest.json",
        "detection_calibration": manifest_dir / "coco2017_train_calibration_manifest.json",
        "detection_validation": manifest_dir / "coco2017_val_manifest.json",
    }
    recovered: dict[str, Any] = {}
    for key, manifest_path in known.items():
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            verification = verify_dataset_manifest(payload, verify_files=verify_manifests)
            if not bool(verification.get("ok")):
                recovered[key] = {
                    "status": "invalid",
                    "manifest": str(manifest_path),
                    "verification": verification,
                }
                continue
            task = str(payload.get("task") or "")
            role = str(payload.get("role") or "")
            dataset_key = {
                ("classification", "calibration"): "imagenet_calibration",
                ("classification", "validation"): "imagenet_validation",
                ("detection", "calibration"): "coco2017_calibration",
                ("detection", "validation"): "coco2017_validation",
            }.get((task, role))
            if not dataset_key:
                continue
            root_path = Path(str(payload.get("root") or "")).expanduser()
            annotations_ref = dict(payload.get("annotations") or {})
            labels_ref = dict(payload.get("labels") or {})
            annotations = (
                Path(str(annotations_ref.get("path") or "")).expanduser()
                if annotations_ref.get("path") else None
            )
            labels = (
                Path(str(labels_ref.get("path") or "")).expanduser()
                if labels_ref.get("path") else None
            )
            _register_dataset(
                registry,
                dataset_key,
                root=root_path,
                annotations=annotations,
                labels=labels,
                source="recovered_from_verified_manifest",
                extra={"recovered_without_download": True},
            )
            registry.setdefault("manifests", {})[key] = str(manifest_path)
            recovered[key] = {
                "status": "recovered",
                "manifest": str(manifest_path),
                "dataset": dataset_key,
                "root": str(root_path),
                "verification": verification,
            }
            _log(log, f"[datasets] recovered {key} from {manifest_path}")
        except Exception as exc:
            recovered[key] = {
                "status": "error",
                "manifest": str(manifest_path),
                "error": f"{type(exc).__name__}: {exc}",
            }

    saved = save_registry(registry, registry_file)
    status = registry_status(saved, verify_manifests=verify_manifests)
    return {
        "status": "ok" if recovered else "no_assets_found",
        "network_used": False,
        "registry": str(saved),
        "recovered": recovered,
        "readiness": status,
    }


def _print_result(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Provision/register final ImageNet and COCO datasets for ONNX Splitpoint campaigns")
    ap.add_argument("--registry", default="", help="Optional registry JSON path")
    sub = ap.add_subparsers(dest="command", required=True)

    st = sub.add_parser("status", help="Show registry and manifest readiness")
    st.add_argument("--verify", action="store_true", help="Re-hash manifest contents")

    repair = sub.add_parser(
        "repair-registry",
        help="Reindex already materialised datasets/manifests without downloading archives",
    )
    repair.add_argument("--root", default="")
    repair.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip re-hashing dataset files while reindexing",
    )

    coco = sub.add_parser("provision-coco", help="Download official COCO 2017 archives and create manifests")
    coco.add_argument("--root", default="")
    coco.add_argument("--include-train", action="store_true", help="Also provision a disjoint train-derived calibration subset")
    coco.add_argument("--overwrite", action="store_true")
    coco.add_argument("--calibration-items", type=int, default=1000)
    coco.add_argument(
        "--keep-train-archive",
        action="store_true",
        help="Keep the verified ~18 GB train2017 ZIP after the selected subset is materialised",
    )
    coco.add_argument(
        "--extract-full-train",
        action="store_true",
        help="Extract all train2017 images instead of materialising only the selected subset",
    )
    coco.add_argument("--seed", type=int, default=DEFAULT_SEED)
    coco.add_argument(
        "--download-policy", choices=COCO_DOWNLOAD_POLICIES, default="auto",
        help="auto=HTTPS then pinned official HTTP fallback; https_only; pinned_http",
    )
    coco.add_argument(
        "--mirror-base", default="",
        help="Optional trusted mirror base, e.g. https://mirror.example/coco",
    )

    rc = sub.add_parser("register-coco", help="Register an existing COCO 2017 installation")
    rc.add_argument("--val-root", required=True)
    rc.add_argument("--train-root", default="")
    rc.add_argument("--annotations-root", required=True)
    rc.add_argument("--calibration-items", type=int, default=500)
    rc.add_argument("--seed", type=int, default=DEFAULT_SEED)

    imp = sub.add_parser("import-imagenet", help="Import/register an existing ImageNet copy or archive")
    imp.add_argument("--source", required=True)
    imp.add_argument("--root", default="")
    imp.add_argument("--calibration-items", type=int, default=1000)
    imp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    imp.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    imp.add_argument(
        "--validation-solution", default="",
        help="LOC_val_solution.csv for a raw ILSVRC2012_img_val.tar",
    )
    imp.add_argument(
        "--labels", default="",
        help="Optional LOC_synset_mapping.txt / synset label file",
    )

    ri = sub.add_parser("register-imagenet", help="Register class-organized ImageNet train/val directories")
    ri.add_argument("--train-root", default="")
    ri.add_argument("--val-root", required=True)
    ri.add_argument("--labels", default="")
    ri.add_argument("--calibration-items", type=int, default=1000)
    ri.add_argument("--seed", type=int, default=DEFAULT_SEED)

    kag = sub.add_parser("provision-imagenet-kaggle", help="Use authenticated Kaggle CLI after accepting ImageNet competition terms")
    kag.add_argument("--root", default="")
    kag.add_argument("--accept-terms", action="store_true", help="Attest that you accepted the dataset/competition terms")
    kag.add_argument("--calibration-items", type=int, default=1000)
    kag.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Provision only validation; by default the tool also exports a disjoint train calibration subset",
    )
    kag.add_argument("--seed", type=int, default=DEFAULT_SEED)
    kag.add_argument("--link-mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    kag.add_argument("--competition", default="imagenet-object-localization-challenge")
    kag.add_argument(
        "--download-mode", choices=list(IMAGENET_KAGGLE_DOWNLOAD_MODES),
        default="validation_only",
        help=(
            "validation_only=smart direct/private-kernel mode; "
            "validation_via_private_kernel=force private validation export; "
            "direct_validation_files=only standalone tar/CSV; full_competition=large bundle"
        ),
    )
    kag.add_argument(
        "--kaggle-username", default="",
        help="Public Kaggle username used as owner of the private validation export kernel",
    )
    kag.add_argument(
        "--kernel-slug", default=IMAGENET_EXPORT_KERNEL_SLUG,
        help="Private Kaggle kernel slug used for validation-only packaging",
    )
    kag.add_argument(
        "--calibration-kernel-slug",
        default=IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG,
        help="Private Kaggle kernel slug used for the train calibration subset export",
    )
    kag.add_argument(
        "--kernel-timeout-s", type=int, default=10800,
        help="Maximum wait time for the private Kaggle export kernel",
    )
    kag.add_argument(
        "--allow-large-download", action="store_true",
        help="Required for full_competition; prevents accidental ~160+ GB downloads",
    )

    kd = sub.add_parser(
        "collect-imagenet-kernel-diagnostics",
        help="Download log/error artefacts from the latest private ImageNet export kernel",
    )
    kd.add_argument("--root", default="")
    kd.add_argument("--kaggle-username", default="")
    kd.add_argument("--kernel-slug", default=IMAGENET_EXPORT_KERNEL_SLUG)
    kd.add_argument("--output-dir", default="")

    ks = sub.add_parser("kaggle-status", help="Check Kaggle CLI, credentials and optional ImageNet competition access")
    ks.add_argument("--check-access", action="store_true")
    ks.add_argument("--competition", default=IMAGENET_KAGGLE_COMPETITION)

    deps = sub.add_parser("install-deps", help="Install/repair optional dataset packages in this Python environment")
    deps.add_argument("--no-coco", action="store_true", help="Do not install pycocotools")
    deps.add_argument("--no-kaggle", action="store_true", help="Do not install Kaggle CLI")
    deps.add_argument("--no-upgrade", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    registry = str(ns.registry or "").strip() or None
    try:
        if ns.command == "status":
            result = registry_status(registry, verify_manifests=bool(ns.verify))
        elif ns.command == "repair-registry":
            result = repair_dataset_registry(
                root=(ns.root or None),
                registry_path=registry,
                verify_manifests=not bool(ns.no_verify),
            )
            result["kaggle_cli"] = kaggle_cli_status()
        elif ns.command == "provision-coco":
            result = provision_coco2017(
                root=ns.root or None,
                include_train=bool(ns.include_train),
                overwrite=bool(ns.overwrite),
                calibration_items=ns.calibration_items,
                seed=ns.seed,
                registry_path=registry,
                download_policy=ns.download_policy,
                mirror_base=ns.mirror_base or None,
                materialize_calibration_subset=not bool(ns.extract_full_train),
                retain_train_archive=bool(ns.keep_train_archive),
            )
        elif ns.command == "register-coco":
            result = register_coco2017(val_root=ns.val_root, train_root=ns.train_root or None, annotations_root=ns.annotations_root, calibration_items=ns.calibration_items, seed=ns.seed, registry_path=registry)
        elif ns.command == "import-imagenet":
            result = import_imagenet(
                source=ns.source,
                root=ns.root or None,
                calibration_items=ns.calibration_items,
                seed=ns.seed,
                link_mode=ns.link_mode,
                registry_path=registry,
                validation_solution=ns.validation_solution or None,
                labels=ns.labels or None,
            )
        elif ns.command == "register-imagenet":
            result = register_imagenet(train_root=ns.train_root or None, val_root=ns.val_root, labels=ns.labels or None, calibration_items=ns.calibration_items, seed=ns.seed, registry_path=registry)
        elif ns.command == "provision-imagenet-kaggle":
            result = provision_imagenet_complete(
                root=ns.root or None,
                accept_terms=bool(ns.accept_terms),
                include_calibration=not bool(ns.skip_calibration),
                calibration_items=ns.calibration_items,
                seed=ns.seed,
                link_mode=ns.link_mode,
                registry_path=registry,
                competition=ns.competition,
                download_mode=ns.download_mode,
                allow_large_download=bool(ns.allow_large_download),
                kaggle_username=ns.kaggle_username or None,
                validation_kernel_slug=ns.kernel_slug,
                calibration_kernel_slug=ns.calibration_kernel_slug,
                kernel_timeout_s=int(ns.kernel_timeout_s),
            )
        elif ns.command == "collect-imagenet-kernel-diagnostics":
            result = collect_imagenet_kernel_diagnostics(
                root=ns.root or None,
                kaggle_username=ns.kaggle_username or None,
                kernel_slug=ns.kernel_slug,
                output_dir=ns.output_dir or None,
            )
        elif ns.command == "kaggle-status":
            result = kaggle_cli_status(
                competition=ns.competition, check_access=bool(ns.check_access)
            )
        elif ns.command == "install-deps":
            result = install_optional_dataset_dependencies(
                install_coco=not bool(ns.no_coco),
                install_kaggle=not bool(ns.no_kaggle),
                upgrade=not bool(ns.no_upgrade),
            )
        else:
            raise RuntimeError(f"unsupported command: {ns.command}")
        _print_result(result)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# v60m: development profiles keep screening validation unless final validation
# was explicitly selected; calibration manifests may still auto-bind.
from onnx_splitpoint_tool.v60m_policy import install_dataset_binding_guards as _v60m_install_dataset_binding_guards
_v60m_install_dataset_binding_guards(globals())
