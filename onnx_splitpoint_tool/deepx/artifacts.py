from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deepx_cache_key(
    *,
    onnx_path: str | Path,
    config_path: str | Path,
    target: str = "deepx_m1",
    variant: str = "full",
    cache_contract: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a DeepX cache key.

    Legacy callers retain the historical key shape.  v60r callers can provide a
    ``cache_contract`` that binds the artefact to the task, calibration manifest,
    effective calibration count, preprocessing contract and compiler identity.
    This prevents a detection Part1 DXNN from reusing an older classification
    calibration artefact merely because the ONNX and generated DX-COM config
    filenames happen to match.
    """
    onnx_hash = sha256_file(onnx_path)
    cfg_hash = sha256_file(config_path)
    if not cache_contract:
        return f"{target}_{variant}_{onnx_hash[:16]}_{cfg_hash[:16]}"
    payload = {
        "schema": "onnx-splitpoint/deepx-cache-key",
        "schema_version": 2,
        "target": str(target),
        "variant": str(variant),
        "onnx_sha256": onnx_hash,
        "config_sha256": cfg_hash,
        "cache_contract": dict(cache_contract),
    }
    digest = _canonical_json_hash(payload)
    return f"{target}_{variant}_v2_{digest[:32]}"


def deepx_cached_artifact_compatible(
    *,
    cache_dir: str | Path,
    expected_contract: Mapping[str, Any],
    require_artifact_identity: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    """Validate a cached DXNN against the exact v60r build contract.

    A legacy cache entry without a contract is deliberately incompatible for
    task-sensitive Part1 reuse.  The old files are left untouched, but a new
    task-safe key is built instead.
    """
    root = Path(cache_dir).expanduser()
    model = root / "model.dxnn"
    manifest_path = root / "build_manifest.json"
    if not model.is_file():
        return False, "model_missing", {}
    if not manifest_path.is_file():
        return False, "legacy_manifest_missing", {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"manifest_unreadable:{type(exc).__name__}", {}
    actual = payload.get("cache_contract") if isinstance(payload, Mapping) else None
    if not isinstance(actual, Mapping):
        return False, "legacy_contract_missing", dict(payload or {})
    expected = json.loads(json.dumps(dict(expected_contract), sort_keys=True, default=str))
    observed = json.loads(json.dumps(dict(actual), sort_keys=True, default=str))
    if observed != expected:
        return False, "cache_contract_mismatch", dict(payload or {})
    if require_artifact_identity:
        artifact = payload.get("artifact") if isinstance(payload, Mapping) else None
        if not isinstance(artifact, Mapping):
            return False, "artifact_identity_missing", dict(payload or {})
        expected_sha256 = str(artifact.get("sha256") or "").strip().lower()
        try:
            expected_bytes = int(artifact.get("bytes"))
        except Exception:
            expected_bytes = -1
        if (
            len(expected_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha256)
            or expected_bytes <= 0
        ):
            return False, "artifact_identity_invalid", dict(payload or {})
        try:
            matches = (
                model.stat().st_size == expected_bytes
                and strict_artifact_sha256(model) == expected_sha256
            )
        except Exception as exc:
            return (
                False,
                f"artifact_identity_unreadable:{type(exc).__name__}",
                dict(payload or {}),
            )
        if not matches:
            return False, "artifact_identity_mismatch", dict(payload or {})
    return True, "compatible", dict(payload or {})


def deepx_cached_artifact_identity_compatible(
    *, cache_dir: str | Path,
) -> tuple[bool, str, dict[str, Any]]:
    """Verify only the immutable DXNN bytes recorded by a cache manifest.

    Legacy Full cache keys do not carry the newer task/cache contract.  They
    may still be reused by the historical workflow, but only when their
    build-manifest artifact identity matches a strict full-file SHA-256 read.
    """

    root = Path(cache_dir).expanduser()
    model = root / "model.dxnn"
    manifest_path = root / "build_manifest.json"
    if not model.is_file():
        return False, "model_missing", {}
    if not manifest_path.is_file():
        return False, "manifest_missing", {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"manifest_unreadable:{type(exc).__name__}", {}
    if not isinstance(payload, Mapping):
        return False, "manifest_not_object", {}
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping):
        return False, "artifact_identity_missing", dict(payload)
    expected_sha256 = str(artifact.get("sha256") or "").strip().lower()
    try:
        expected_bytes = int(artifact.get("bytes"))
    except Exception:
        expected_bytes = -1
    if (
        len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256)
        or expected_bytes <= 0
    ):
        return False, "artifact_identity_invalid", dict(payload)
    try:
        matches = (
            model.stat().st_size == expected_bytes
            and strict_artifact_sha256(model) == expected_sha256
        )
    except Exception as exc:
        return False, f"artifact_identity_unreadable:{type(exc).__name__}", dict(payload)
    if not matches:
        return False, "artifact_identity_mismatch", dict(payload)
    return True, "artifact_identity_verified", dict(payload)


def cache_dxnn_artifact(
    *,
    dxnn_path: str | Path,
    manifest: Mapping[str, Any] | None = None,
    cache_root: str | Path = "~/Models/BackendArtifacts/deepx",
    cache_key: str,
    allow_overwrite: bool = True,
) -> Path:
    source = Path(dxnn_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"DeepX DXNN source missing: {source}")
    dst_dir = Path(cache_root).expanduser() / cache_key
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.dxnn"
    source_sha256 = strict_artifact_sha256(source)
    if dst.is_file() and not allow_overwrite:
        if (
            dst.stat().st_size != source.stat().st_size
            or strict_artifact_sha256(dst) != source_sha256
        ):
            raise FileExistsError(
                "refusing to overwrite a different content-addressed DeepX cache artifact"
            )
    elif not dst.is_file() or source.resolve() != dst.resolve():
        shutil.copy2(source, dst)
    payload = dict(manifest or {})
    payload["artifact"] = {
        "path": "model.dxnn",
        "sha256": source_sha256,
        "bytes": int(source.stat().st_size),
    }
    if payload:
        (dst_dir / "build_manifest.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return dst

# v60m: unchanged files use a stat-keyed SHA-256 cache in development;
# final campaigns still re-hash strictly.
from onnx_splitpoint_tool.v60m_policy import install_hash_wrappers as _v60m_install_hash_wrappers
_v60m_install_hash_wrappers(globals())


def strict_artifact_sha256(
    path: str | Path, chunk_size: int = 1024 * 1024,
) -> str:
    """Hash every artifact byte, bypassing the development stat/probe cache."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()
