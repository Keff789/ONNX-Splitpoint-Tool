"""Bind a resolved Evaluation Profile to the content-addressed dataset registry.

The GUI-wide dataset registry is deliberately external to a profile so datasets
survive tool upgrades.  A run, however, must snapshot concrete manifest paths.
This module fills only missing profile fields, verifies the selected manifests,
and records exactly what was bound.  Explicit profile values always win.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..campaign import verify_dataset_manifest
from ..dataset_provisioning import default_registry_path, load_registry
from .artifacts import now_iso, sha256_file, sha256_json

_MANIFEST_BINDINGS = {
    ("classification", "calibration"): "classification_calibration",
    ("classification", "validation"): "classification_validation",
    ("detection", "calibration"): "detection_calibration",
    ("detection", "validation"): "detection_validation",
}


def _resolve_ref(value: str, *, profile_path: str | Path | None = None) -> Path:
    p = Path(str(value or "")).expanduser()
    if not p.is_absolute() and profile_path:
        p = Path(profile_path).expanduser().resolve().parent / p
    return p.resolve()


def dataset_registry_path_for_profile(
    profile: Mapping[str, Any], *, profile_path: str | Path | None = None
) -> Path:
    campaign = profile.get("campaign") if isinstance(profile.get("campaign"), Mapping) else {}
    raw = str(
        (campaign or {}).get("dataset_registry")
        or os.environ.get("ONNX_SPLITPOINT_DATASET_REGISTRY")
        or default_registry_path()
    ).strip()
    return _resolve_ref(raw, profile_path=profile_path)


def manifest_dataset_root(manifest_ref: str | Path, *, profile_path: str | Path | None = None) -> str:
    """Return the verified dataset root recorded by a dataset manifest."""
    p = _resolve_ref(str(manifest_ref), profile_path=profile_path)
    if not p.is_file():
        return ""
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    root = Path(str(payload.get("root") or "")).expanduser()
    if not root.is_absolute():
        root = p.parent / root
    return str(root.resolve()) if root.is_dir() else ""


def bind_profile_dataset_registry(
    profile: Mapping[str, Any],
    *,
    profile_path: str | Path | None = None,
    verify_manifests: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill missing dataset-manifest fields from the external registry.

    The returned profile is a deep copy.  This function never overwrites an
    explicit manifest path and never edits the source YAML.  It is therefore
    safe for development profiles while still making the resolved run profile
    self-contained and reproducible.
    """
    out = copy.deepcopy(dict(profile or {}))
    campaign = dict(out.get("campaign") or {}) if isinstance(out.get("campaign"), Mapping) else {}
    auto_bind = bool(campaign.get("auto_bind_dataset_registry", True))
    registry_path = dataset_registry_path_for_profile(out, profile_path=profile_path)
    report: dict[str, Any] = {
        "schema": "onnx-splitpoint/dataset-registry-binding",
        "schema_version": 1,
        "created_at": now_iso(),
        "enabled": auto_bind,
        "registry_path": str(registry_path),
        "registry_exists": registry_path.is_file(),
        "registry_sha256": str(sha256_file(registry_path) or ""),
        "bound": [],
        "preserved_explicit": [],
        "missing": [],
        "invalid": [],
        "status": "disabled" if not auto_bind else "missing_registry",
    }
    campaign["dataset_registry"] = str(registry_path)
    campaign["auto_bind_dataset_registry"] = auto_bind
    out["campaign"] = campaign
    if not auto_bind:
        return out, report
    if not registry_path.is_file():
        return out, report

    registry = load_registry(registry_path)
    manifests = dict(registry.get("manifests") or {})
    profile_manifests = dict(campaign.get("dataset_manifests") or {}) if isinstance(campaign.get("dataset_manifests"), Mapping) else {}
    for task in ("classification", "detection"):
        block = dict(profile_manifests.get(task) or {}) if isinstance(profile_manifests.get(task), Mapping) else {}
        for role in ("calibration", "validation"):
            current = str(block.get(role) or "").strip()
            key = _MANIFEST_BINDINGS[(task, role)]
            selected = current or str(manifests.get(key) or "").strip()
            if current:
                report["preserved_explicit"].append({"task": task, "role": role, "path": current})
            elif selected:
                report["bound"].append({"task": task, "role": role, "registry_key": key, "path": selected})
            else:
                report["missing"].append({"task": task, "role": role, "registry_key": key})
                continue
            p = _resolve_ref(selected, profile_path=profile_path)
            # profile.yaml inside an EvalRun is a resolved snapshot.  Store an
            # absolute path there even when the source profile used a relative
            # reference, otherwise the copied profile would resolve relative to
            # the run directory instead of the source YAML directory.
            block[role] = str(p)
            if not p.is_file():
                report["invalid"].append({"task": task, "role": role, "path": str(p), "reason": "manifest_missing"})
                continue
            if verify_manifests:
                try:
                    payload = json.loads(p.read_text(encoding="utf-8"))
                    verification = verify_dataset_manifest(payload, verify_files=True)
                    if not bool(verification.get("ok")):
                        report["invalid"].append({
                            "task": task,
                            "role": role,
                            "path": str(p),
                            "reason": "manifest_verification_failed",
                            "errors": list(verification.get("errors") or [])[:20],
                        })
                except Exception as exc:
                    report["invalid"].append({"task": task, "role": role, "path": str(p), "reason": f"{type(exc).__name__}: {exc}"})
        profile_manifests[task] = block
    campaign["dataset_manifests"] = profile_manifests

    # Bind the official COCO annotation file when the registry already records
    # it and the profile has not supplied a local path.
    official = dict(out.get("official_coco_evaluation") or {}) if isinstance(out.get("official_coco_evaluation"), Mapping) else {}
    coco_record = dict((registry.get("datasets") or {}).get("coco2017_validation") or {})
    coco_annotations = str(coco_record.get("annotations") or "").strip()
    if not str(official.get("annotations") or "").strip() and coco_annotations:
        official["annotations"] = coco_annotations
        report["bound"].append({"field": "official_coco_evaluation.annotations", "path": coco_annotations})
    if official:
        out["official_coco_evaluation"] = official

    campaign["dataset_registry_sha256"] = str(sha256_file(registry_path) or "")
    campaign["dataset_registry_binding_sha256"] = sha256_json({
        "registry": str(registry_path),
        "manifests": profile_manifests,
        "official_coco_annotations": str(official.get("annotations") or ""),
    })
    out["campaign"] = campaign
    report["bound_count"] = len(report["bound"])
    report["missing_count"] = len(report["missing"])
    report["invalid_count"] = len(report["invalid"])
    report["status"] = "ok" if not report["invalid"] and not report["missing"] else ("partial" if report["bound"] else "incomplete")
    return out, report


def calibration_manifest_for_task(profile: Mapping[str, Any], task: str) -> str:
    campaign = profile.get("campaign") if isinstance(profile.get("campaign"), Mapping) else {}
    manifests = (campaign or {}).get("dataset_manifests") if isinstance((campaign or {}).get("dataset_manifests"), Mapping) else {}
    block = (manifests or {}).get(str(task).strip().lower()) if isinstance((manifests or {}).get(str(task).strip().lower()), Mapping) else {}
    return str((block or {}).get("calibration") or "").strip()


def validation_manifest_for_task(profile: Mapping[str, Any], task: str) -> str:
    campaign = profile.get("campaign") if isinstance(profile.get("campaign"), Mapping) else {}
    manifests = (campaign or {}).get("dataset_manifests") if isinstance((campaign or {}).get("dataset_manifests"), Mapping) else {}
    block = (manifests or {}).get(str(task).strip().lower()) if isinstance((manifests or {}).get(str(task).strip().lower()), Mapping) else {}
    return str((block or {}).get("validation") or "").strip()

# v60m: development profiles keep screening validation unless final validation
# was explicitly selected; calibration manifests may still auto-bind.
from onnx_splitpoint_tool.v60m_policy import install_dataset_binding_guards as _v60m_install_dataset_binding_guards
_v60m_install_dataset_binding_guards(globals())
