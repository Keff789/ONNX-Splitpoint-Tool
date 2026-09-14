from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..artifact_store import ArtifactStore, artifact_store_enabled, canonical_hash, default_store, sha256_file

_VOLATILE_KEYS = {"created_at", "updated_at", "timestamp", "started_at", "ended_at", "elapsed_s", "duration_s", "pid", "hostname", "source_run"}


def _stable(value: Any) -> Any:
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            k = str(key)
            if k.lower() in _VOLATILE_KEYS:
                continue
            if k.lower().endswith("_path") or k.lower() in {"path", "output_dir", "outdir", "workdir", "log_path"}:
                p = Path(str(item)).expanduser()
                if p.is_file():
                    out[k + "_sha256"] = sha256_file(p)
                    out[k + "_name"] = p.name
                else:
                    out[k + "_name"] = p.name or str(item)
                continue
            out[k] = _stable(item)
        return out
    if isinstance(value, list):
        return [_stable(x) for x in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _nearest_manifest(artifact: Path) -> tuple[Path | None, dict[str, Any]]:
    names = ("build_manifest.json", "build_status.json", "manifest.json", "status.json", "artifact_manifest.json")
    for parent in (artifact.parent, *artifact.parents[:5]):
        for name in names:
            candidate = parent / name
            if candidate.is_file():
                return candidate, _read_json(candidate)
    return None, {}


def _source_onnx_hash(artifact: Path) -> str:
    candidates: list[Path] = []
    for parent in (artifact.parent, *artifact.parents[:5]):
        candidates.extend(sorted(parent.glob("*.onnx")))
        if candidates:
            break
    for candidate in candidates:
        with contextlib.suppress(Exception):
            return sha256_file(candidate)
    return ""


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".hef":
        return "hailo_hef"
    if suffix == ".dxnn":
        return "deepx_dxnn"
    if suffix in {".engine", ".plan"}:
        return "tensorrt_engine"
    return ""


def _profile_contract(profile: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind == "hailo_hef":
        return _stable(dict(profile.get("hailo_build") or (profile.get("build") or {}).get("hailo") or {}))
    if kind == "deepx_dxnn":
        return _stable(dict(profile.get("deepx_build") or (profile.get("build") or {}).get("deepx") or {}))
    return {}


def artifact_contract_for_path(*, artifact: Path, suite_dir: Path, model_id: str,
                               task: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    kind = _artifact_kind(artifact)
    rel = str(artifact.relative_to(suite_dir)) if artifact.is_relative_to(suite_dir) else artifact.name
    manifest_path, manifest = _nearest_manifest(artifact)
    parts = [x.lower() for x in Path(rel).parts]
    target = ""
    for token in ("hailo10h", "hailo10", "hailo8l", "hailo8", "deepx_m1", "deepx"):
        if token in parts:
            target = token
            break
    role = "full" if "full" in parts else ("part2" if "part2" in parts else ("part1" if "part1" in parts else "artifact"))
    case_id = next((x for x in Path(rel).parts if str(x).lower().startswith("b") and str(x)[1:].isdigit()), "")
    return {
        "schema": "onnx-splitpoint/compiler-artifact-contract/v1",
        "kind": kind,
        "model_id": str(model_id),
        "task": str(task or "auto"),
        "target": target,
        "role": role,
        "case_id": case_id,
        "relative_path": rel,
        "source_onnx_sha256": _source_onnx_hash(artifact),
        "build_profile": _profile_contract(profile, kind),
        "compiler_manifest": _stable(manifest),
        "manifest_name": manifest_path.name if manifest_path else "",
    }


def register_benchmark_set_artifacts(*, suite_dir: str | Path, model_id: str,
                                     task: str, profile: Mapping[str, Any],
                                     source_run: str = "") -> dict[str, Any]:
    if not artifact_store_enabled():
        return {"enabled": False, "registered": []}
    suite = Path(suite_dir).expanduser().resolve()
    if not suite.is_dir():
        return {"enabled": True, "registered": [], "warning": f"suite missing: {suite}"}
    cfg = dict(profile.get("artifact_store") or (profile.get("build") or {}).get("artifact_store") or {})
    allowed = {
        "hailo_hef": bool(cfg.get("register_hailo", True)),
        "deepx_dxnn": bool(cfg.get("register_deepx", True)),
        "tensorrt_engine": bool(cfg.get("register_tensorrt", False)),
    }
    store = ArtifactStore(cfg.get("root") or None)
    pin = bool(cfg.get("pin_for_campaign", False))
    registered: list[dict[str, Any]] = []
    for artifact in sorted(p for p in suite.rglob("*") if p.is_file() and _artifact_kind(p)):
        if ".hailo-generations" in artifact.relative_to(suite).parts:
            continue
        kind = _artifact_kind(artifact)
        if not allowed.get(kind, False):
            continue
        try:
            contract = artifact_contract_for_path(artifact=artifact, suite_dir=suite, model_id=model_id, task=task, profile=profile)
            metadata = {
                    "suite_dir": str(suite),
                    "relative_path": str(artifact.relative_to(suite)),
                    "task": task,
                    "legacy_cache_key": (_nearest_manifest(artifact)[1].get("cache_key") or _nearest_manifest(artifact)[1].get("cacheKey") or ""),
                }
            registration = dict(
                contract=contract, metadata=metadata,
                source_run=source_run or os.environ.get("ONNX_SPLITPOINT_RUN_ID", ""),
                pin=pin, pin_label="final-campaign" if pin else "",
            )
            if kind == "hailo_hef":
                from ..hailo_backend import (
                    _hailo_receipt_path, _load_valid_hailo_receipt,
                    _publish_hailo_bundle,
                )
                snapshot = artifact.resolve(strict=True)
                receipt = _load_valid_hailo_receipt(snapshot)
                if receipt is None:
                    raise ValueError("legacy_unsealed" if not _hailo_receipt_path(snapshot).is_file()
                                     else "hailo_receipt_invalid")
                if not (snapshot.parent / "cache_meta.json").is_file():
                    snapshot = _publish_hailo_bundle(
                        source_hef=snapshot, destination=artifact, receipt=receipt,
                        source="benchmark_set_registration",
                    )
                metadata.update(legacy_cache_key=receipt["cache_key"], build_receipt=receipt)
                record = store.register_hailo_bundle(
                    source_path=snapshot, receipt_path=_hailo_receipt_path(snapshot),
                    cache_meta_path=snapshot.parent / "cache_meta.json", **registration,
                )
            else:
                record = store.register(source_path=artifact, kind=kind, **registration)
            registered.append({"artifact_id": record.artifact_id, "kind": kind, "path": str(artifact),
                               "contract_hash": record.contract_hash, "artifact_hash": record.artifact_hash})
        except Exception as exc:
            registered.append({"kind": kind, "path": str(artifact), "error": str(exc)})
    report = {"enabled": True, "store": str(store.root), "registered": registered,
              "successful": sum(1 for x in registered if "artifact_id" in x),
              "failed": sum(1 for x in registered if "error" in x)}
    out = suite / "artifact_registry_registration.json"
    with contextlib.suppress(Exception):
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def restore_exact_artifact(*, artifact_path: str | Path, suite_dir: str | Path,
                           model_id: str, task: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    """Restore a missing artifact when the exact suite contract is known.

    The function is deliberately conservative. It is intended for call sites
    where the output path and source subgraph already exist.
    """
    artifact = Path(artifact_path).expanduser().absolute(); suite = Path(suite_dir).expanduser().resolve()
    if artifact.is_file():
        if _artifact_kind(artifact) == "hailo_hef":
            from ..hailo_backend import _load_valid_hailo_receipt
            if _load_valid_hailo_receipt(artifact) is None:
                return {"hit": False, "reason": "hailo_receipt_invalid_or_missing"}
        return {"hit": True, "existing": True, "path": str(artifact)}
    kind = _artifact_kind(artifact)
    if not kind or not artifact_store_enabled():
        return {"hit": False, "reason": "unsupported_or_disabled"}
    cfg = dict(profile.get("artifact_store") or (profile.get("build") or {}).get("artifact_store") or {})
    store = ArtifactStore(cfg.get("root") or None)
    contract = artifact_contract_for_path(artifact=artifact, suite_dir=suite, model_id=model_id, task=task, profile=profile)
    verify = str(cfg.get("verify_on_reuse") or os.environ.get("ONNX_SPLITPOINT_ARTIFACT_VERIFY") or "metadata")
    record = store.lookup(kind=kind, contract=contract, verify=verify)
    if record is None:
        return {"hit": False, "reason": "contract_miss", "contract_hash": canonical_hash(contract)}
    if kind == "hailo_hef":
        from ..hailo_backend import _load_valid_hailo_receipt, _publish_hailo_bundle
        try:
            valid, reason = store.validate_record(record, verify="strict")
            if not valid:
                return {"hit": False, "reason": reason}
            paths = store.bundle_paths(record)
            receipt = _load_valid_hailo_receipt(paths["hef"])
            if receipt is None:
                return {"hit": False, "reason": "hailo_receipt_invalid"}
            _publish_hailo_bundle(source_hef=paths["hef"], destination=artifact,
                                  receipt=receipt, source="benchmark_set_restore")
            return {"hit": True, "existing": False, "path": str(artifact),
                    "artifact_id": record.artifact_id, "contract_hash": record.contract_hash,
                    "method": "atomic_hailo_bundle"}
        except (ValueError, OSError) as exc:
            return {"hit": False, "reason": str(exc)}
    method = store.materialize(record, artifact, reference=f"{model_id}:{artifact.relative_to(suite)}")
    return {"hit": True, "existing": False, "path": str(artifact), "artifact_id": record.artifact_id,
            "contract_hash": record.contract_hash, "method": method}
