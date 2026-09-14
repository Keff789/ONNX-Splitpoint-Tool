from __future__ import annotations

"""Resolve exact legacy artifacts for a later, separately controlled resume.

This module is deliberately local-only.  It does not execute SSH, copy files,
modify a command contract, or invoke a runner/preflight.  Callers provide the
small explicit set of artifacts that may be rehydrated and the remote roots
within which their original target paths are allowed.  The result is a
deterministic, hash-bound stage map for a separate staging layer.

Sources are accepted only when their bytes match the expected SHA-256 and,
when supplied by the old contract/manifest, the expected size.  An old run
mirror is searched first; the unified artifact store is a content-addressed
fallback.  Symlinks and path rebasing outside the caller's allow-list are
rejected.
"""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


STAGE_MAP_SCHEMA = "onnx-splitpoint/resume-artifact-stage-map"
STAGE_MAP_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DEFAULT_PAYLOAD_MANIFEST_ROLES = frozenset(
    {"semantic_output_manifest", "semantic_boundary_manifest"}
)
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class ResumeArtifactResolutionError(ValueError):
    """A stable, fail-closed artifact-resolution failure."""

    def __init__(
        self,
        code: str,
        *,
        role: str = "",
        detail: str = "",
    ) -> None:
        self.code = str(code)
        self.role = str(role)
        self.detail = str(detail)
        message = self.code
        if self.role:
            message += f":role={self.role}"
        if self.detail:
            message += f":{self.detail}"
        super().__init__(message)


@dataclass(frozen=True)
class ArtifactRequirement:
    """One immutable byte identity and its exact remote destination."""

    role: str
    remote_path: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class _NormalizedRequirement:
    role: str
    remote_path: str
    sha256: str
    size_bytes: int | None


@dataclass(frozen=True)
class _RequirementGroup:
    remote_path: str
    sha256: str
    size_bytes: int | None
    roles: tuple[str, ...]


@dataclass(frozen=True)
class _Candidate:
    path: Path
    root: Path
    source_kind: str
    resolution: str
    priority: int
    suffix_score: int


@dataclass(frozen=True)
class _VerifiedSource:
    path: Path
    root: Path
    source_kind: str
    resolution: str
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _normalise_role(value: Any) -> str:
    role = str(value or "").strip()
    if (
        not role
        or len(role) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in role)
    ):
        raise ResumeArtifactResolutionError("invalid_artifact_role")
    return role


def _normalise_sha256(value: Any, *, role: str) -> str:
    digest = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ResumeArtifactResolutionError(
            "invalid_expected_sha256", role=role,
        )
    return digest


def _normalise_size(value: Any, *, role: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResumeArtifactResolutionError(
            "invalid_expected_size", role=role,
        )
    return int(value)


def _normalise_remote_path(value: Any, *, role: str) -> str:
    raw = str(value or "")
    if (
        not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
        or not raw.startswith("/")
        or raw.startswith("//")
    ):
        raise ResumeArtifactResolutionError(
            "invalid_remote_target", role=role,
        )
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or "." in path.parts
        or ".." in path.parts
        or str(path) != raw
        or not path.name
    ):
        raise ResumeArtifactResolutionError(
            "invalid_remote_target", role=role,
        )
    return str(path)


def _normalise_requirement(
    value: ArtifactRequirement | Mapping[str, Any],
) -> _NormalizedRequirement:
    if isinstance(value, ArtifactRequirement):
        raw_role = value.role
        raw_path = value.remote_path
        raw_sha = value.sha256
        raw_size = value.size_bytes
    elif isinstance(value, Mapping):
        raw_role = value.get("role")
        raw_path = value.get("remote_path", value.get("path"))
        raw_sha = value.get("sha256")
        raw_size = value.get("size_bytes")
        if raw_size in (None, ""):
            raw_size = value.get("bytes")
    else:
        raise ResumeArtifactResolutionError("invalid_artifact_requirement")
    role = _normalise_role(raw_role)
    return _NormalizedRequirement(
        role=role,
        remote_path=_normalise_remote_path(raw_path, role=role),
        sha256=_normalise_sha256(raw_sha, role=role),
        size_bytes=_normalise_size(raw_size, role=role),
    )


def _normalise_remote_root(value: Any) -> str:
    root = _normalise_remote_path(value, role="allowed_remote_root")
    # A stage target must be a child, not the allow-list root itself.  Requiring
    # at least two components also prevents accidental broad roots such as
    # ``/home`` from silently authorising unrelated destinations.
    if len(PurePosixPath(root).parts) < 3:
        raise ResumeArtifactResolutionError(
            "unsafe_allowed_remote_root", detail=root,
        )
    return root


def _is_child_remote_path(target: str, root: str) -> bool:
    try:
        relative = PurePosixPath(target).relative_to(PurePosixPath(root))
    except ValueError:
        return False
    return bool(relative.parts)


def _require_allowed_target(
    requirement: _NormalizedRequirement,
    allowed_roots: Sequence[str],
) -> None:
    if not any(
        _is_child_remote_path(requirement.remote_path, root)
        for root in allowed_roots
    ):
        raise ResumeArtifactResolutionError(
            "remote_target_outside_allowed_roots",
            role=requirement.role,
            detail=requirement.remote_path,
        )


def _merge_requirements(
    requirements: Iterable[ArtifactRequirement | Mapping[str, Any]],
    *,
    allowed_roots: Sequence[str],
) -> list[_RequirementGroup]:
    by_target: dict[str, dict[str, Any]] = {}
    for raw in requirements:
        requirement = _normalise_requirement(raw)
        _require_allowed_target(requirement, allowed_roots)
        existing = by_target.get(requirement.remote_path)
        if existing is None:
            by_target[requirement.remote_path] = {
                "sha256": requirement.sha256,
                "size_bytes": requirement.size_bytes,
                "roles": {requirement.role},
            }
            continue
        existing_size = existing["size_bytes"]
        if (
            existing["sha256"] != requirement.sha256
            or (
                existing_size is not None
                and requirement.size_bytes is not None
                and existing_size != requirement.size_bytes
            )
        ):
            raise ResumeArtifactResolutionError(
                "conflicting_remote_target",
                role=requirement.role,
                detail=requirement.remote_path,
            )
        if existing_size is None and requirement.size_bytes is not None:
            existing["size_bytes"] = requirement.size_bytes
        existing["roles"].add(requirement.role)
    return [
        _RequirementGroup(
            remote_path=remote_path,
            sha256=str(value["sha256"]),
            size_bytes=value["size_bytes"],
            roles=tuple(sorted(value["roles"])),
        )
        for remote_path, value in sorted(by_target.items())
    ]


def _resolved_local_roots(values: Iterable[str | Path]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in values:
        root = Path(value).expanduser()
        try:
            resolved = root.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if not resolved.is_dir() or resolved in roots:
            continue
        roots.append(resolved)
    return tuple(roots)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_regular_path(path: Path, root: Path) -> Path | None:
    """Return a resolved regular file only when no in-root component is a link."""
    try:
        lexical = path if path.is_absolute() else root / path
        relative = lexical.relative_to(root)
    except ValueError:
        return None
    cursor = root
    try:
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None
        resolved = lexical.resolve(strict=True)
        if not _path_is_within(resolved, root):
            return None
        mode = os.lstat(resolved).st_mode
        if not stat.S_ISREG(mode):
            return None
    except (FileNotFoundError, OSError, RuntimeError):
        return None
    return resolved


def _common_suffix_score(left: Sequence[str], right: Sequence[str]) -> int:
    score = 0
    for first, second in zip(reversed(left), reversed(right)):
        if first != second:
            break
        score += 1
    return score


def _iter_named_files(root: Path, filename: str) -> Iterable[Path]:
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        names[:] = sorted(
            name for name in names
            if not (base / name).is_symlink()
        )
        if filename in files:
            yield base / filename


def _run_mirror_candidates(
    requirement: _RequirementGroup,
    roots: Sequence[Path],
) -> list[_Candidate]:
    remote = PurePosixPath(requirement.remote_path)
    candidates: dict[Path, _Candidate] = {}
    for root in roots:
        exact = root.joinpath(*remote.parts[1:])
        safe_exact = _safe_regular_path(exact, root)
        if safe_exact is not None:
            candidates[safe_exact] = _Candidate(
                path=safe_exact,
                root=root,
                source_kind="run_mirror",
                resolution="exact_remote_relative_path",
                priority=0,
                suffix_score=len(remote.parts),
            )
        for lexical in _iter_named_files(root, remote.name):
            safe = _safe_regular_path(lexical, root)
            if safe is None or safe in candidates:
                continue
            relative_parts = safe.relative_to(root).parts
            candidates[safe] = _Candidate(
                path=safe,
                root=root,
                source_kind="run_mirror",
                resolution="verified_longest_suffix",
                priority=1,
                suffix_score=_common_suffix_score(
                    relative_parts, remote.parts,
                ),
            )
    return list(candidates.values())


def _store_object_directories(root: Path, digest: str) -> tuple[Path, ...]:
    candidates = (
        root / "objects" / "sha256" / digest[:2] / digest,
        root / "objects" / digest,
        root / "sha256" / digest[:2] / digest,
        root / digest[:2] / digest,
        root / digest,
    )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _artifact_store_candidates(
    requirement: _RequirementGroup,
    roots: Sequence[Path],
) -> list[_Candidate]:
    candidates: dict[Path, _Candidate] = {}
    for root in roots:
        for directory in _store_object_directories(root, requirement.sha256):
            try:
                entries = sorted(directory.iterdir())
            except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                continue
            for lexical in entries:
                safe = _safe_regular_path(lexical, root)
                if safe is None or safe in candidates:
                    continue
                candidates[safe] = _Candidate(
                    path=safe,
                    root=root,
                    source_kind="artifact_store",
                    resolution="content_addressed_sha256_object",
                    priority=2,
                    suffix_score=int(
                        lexical.name == PurePosixPath(
                            requirement.remote_path
                        ).name
                    ),
                )
    return list(candidates.values())


def _verify_candidate(
    candidate: _Candidate,
    requirement: _RequirementGroup,
) -> _VerifiedSource | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate.path, flags)
    except OSError:
        return None
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or (
                requirement.size_bytes is not None
                and before.st_size != requirement.size_bytes
            )
        ):
            return None
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(descriptor)
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        actual_sha = digest.hexdigest()
        if not stable_identity or actual_sha != requirement.sha256:
            return None
        return _VerifiedSource(
            path=candidate.path,
            root=candidate.root,
            source_kind=candidate.source_kind,
            resolution=candidate.resolution,
            sha256=actual_sha,
            size_bytes=int(after.st_size),
            device=int(after.st_dev),
            inode=int(after.st_ino),
            mtime_ns=int(after.st_mtime_ns),
        )
    finally:
        os.close(descriptor)


def _resolve_requirement(
    requirement: _RequirementGroup,
    *,
    run_mirror_roots: Sequence[Path],
    artifact_store_roots: Sequence[Path],
) -> _VerifiedSource:
    candidates = _run_mirror_candidates(requirement, run_mirror_roots)
    candidates.extend(
        _artifact_store_candidates(requirement, artifact_store_roots)
    )
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.priority,
            -item.suffix_score,
            str(item.path),
        ),
    )
    for candidate in ordered:
        verified = _verify_candidate(candidate, requirement)
        if verified is not None:
            return verified
    raise ResumeArtifactResolutionError(
        "exact_source_not_found",
        role=",".join(requirement.roles),
        detail=(
            f"remote_path={requirement.remote_path};"
            f"sha256={requirement.sha256};"
            f"size_bytes={requirement.size_bytes};"
            f"candidate_count={len(ordered)}"
        ),
    )


def _strict_json_object(data: bytes, *, role: str) -> dict[str, Any]:
    duplicate = False

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeArtifactResolutionError(
            "payload_manifest_invalid",
            role=role,
            detail=type(exc).__name__,
        ) from exc
    if duplicate or not isinstance(value, Mapping):
        raise ResumeArtifactResolutionError(
            "payload_manifest_invalid",
            role=role,
            detail="duplicate_keys_or_non_object",
        )
    return dict(value)


def _read_verified_manifest(
    requirement: _RequirementGroup,
    source: _VerifiedSource,
) -> dict[str, Any]:
    if source.size_bytes > _MAX_MANIFEST_BYTES:
        raise ResumeArtifactResolutionError(
            "payload_manifest_too_large",
            role=",".join(requirement.roles),
        )
    try:
        data = source.path.read_bytes()
    except OSError as exc:
        raise ResumeArtifactResolutionError(
            "payload_manifest_unreadable",
            role=",".join(requirement.roles),
        ) from exc
    if (
        len(data) != source.size_bytes
        or hashlib.sha256(data).hexdigest() != requirement.sha256
    ):
        raise ResumeArtifactResolutionError(
            "payload_manifest_changed_after_resolution",
            role=",".join(requirement.roles),
        )
    return _strict_json_object(
        data, role=",".join(requirement.roles),
    )


def _is_payload_manifest_role(
    roles: Sequence[str],
    payload_manifest_roles: frozenset[str],
) -> bool:
    return any(
        role in payload_manifest_roles
        or any(role.endswith(f".{name}") for name in payload_manifest_roles)
        for role in roles
    )


def _payload_requirements(
    requirement: _RequirementGroup,
    source: _VerifiedSource,
) -> list[ArtifactRequirement]:
    manifest = _read_verified_manifest(requirement, source)
    rows = manifest.get("payload_artifacts")
    if not isinstance(rows, list) or not rows:
        raise ResumeArtifactResolutionError(
            "payload_artifacts_missing",
            role=",".join(requirement.roles),
        )
    set_sha = str(manifest.get("payload_artifacts_sha256") or "").strip().lower()
    if (
        _SHA256_RE.fullmatch(set_sha) is None
        or _canonical_sha256(rows) != set_sha
    ):
        raise ResumeArtifactResolutionError(
            "payload_artifacts_sha256_mismatch",
            role=",".join(requirement.roles),
        )
    base_role = requirement.roles[0]
    base_remote = PurePosixPath(requirement.remote_path).parent
    result: list[ArtifactRequirement] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ResumeArtifactResolutionError(
                "payload_artifact_invalid",
                role=base_role,
                detail=f"index={index}",
            )
        payload_role = _normalise_role(raw.get("role") or f"payload[{index}]")
        raw_path = str(raw.get("path") or "")
        if not raw_path:
            raise ResumeArtifactResolutionError(
                "payload_artifact_invalid",
                role=base_role,
                detail=f"index={index};missing_path",
            )
        path = PurePosixPath(raw_path)
        remote_path = str(path if path.is_absolute() else base_remote / path)
        result.append(
            ArtifactRequirement(
                role=f"{base_role}.payload[{index}]:{payload_role}",
                remote_path=remote_path,
                sha256=str(raw.get("sha256") or ""),
                size_bytes=(
                    raw.get("size_bytes")
                    if raw.get("size_bytes") not in (None, "")
                    else raw.get("bytes")
                ),
            )
        )
    return result


def requirements_from_contract_artifacts(
    contract: Mapping[str, Any],
    *,
    roles: Iterable[str],
) -> list[ArtifactRequirement]:
    """Extract only explicitly authorised artifact roles from a contract.

    No implicit "stage everything in the contract" operation is provided:
    interpreter binaries, runners and system tools must never become resume
    staging targets merely because they appear in an old contract.
    """
    if not isinstance(contract, Mapping):
        raise ResumeArtifactResolutionError("invalid_command_contract")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ResumeArtifactResolutionError("contract_artifacts_missing")
    requested_roles = tuple(_normalise_role(role) for role in roles)
    if not requested_roles:
        raise ResumeArtifactResolutionError("artifact_roles_empty")
    result: list[ArtifactRequirement] = []
    for role in requested_roles:
        raw = artifacts.get(role)
        if not isinstance(raw, Mapping):
            raise ResumeArtifactResolutionError(
                "contract_artifact_missing", role=role,
            )
        size_bytes = raw.get("size_bytes")
        if size_bytes in (None, ""):
            size_bytes = raw.get("bytes")
        result.append(
            ArtifactRequirement(
                role=role,
                remote_path=str(raw.get("path") or ""),
                sha256=str(raw.get("sha256") or ""),
                size_bytes=size_bytes,
            )
        )
    return result


def build_resume_artifact_stage_map(
    requirements: Iterable[ArtifactRequirement | Mapping[str, Any]],
    *,
    run_mirror_roots: Iterable[str | Path],
    artifact_store_roots: Iterable[str | Path],
    allowed_remote_roots: Iterable[str],
    expand_payload_manifests: bool = True,
    payload_manifest_roles: Iterable[str] = _DEFAULT_PAYLOAD_MANIFEST_ROLES,
) -> dict[str, Any]:
    """Resolve exact bytes and return a deterministic, non-executing stage map.

    The caller must separately transport the listed local files and must
    re-check their source stat identity, SHA-256 and size immediately before
    doing so.  This function intentionally does not expose a copy/SSH helper.
    """
    allowed = tuple(
        sorted({_normalise_remote_root(value) for value in allowed_remote_roots})
    )
    if not allowed:
        raise ResumeArtifactResolutionError("allowed_remote_roots_empty")
    mirror_roots = _resolved_local_roots(run_mirror_roots)
    store_roots = _resolved_local_roots(artifact_store_roots)
    groups = _merge_requirements(requirements, allowed_roots=allowed)
    if not groups:
        raise ResumeArtifactResolutionError("artifact_requirements_empty")

    resolved: dict[str, tuple[_RequirementGroup, _VerifiedSource]] = {}
    for requirement in groups:
        resolved[requirement.remote_path] = (
            requirement,
            _resolve_requirement(
                requirement,
                run_mirror_roots=mirror_roots,
                artifact_store_roots=store_roots,
            ),
        )

    if expand_payload_manifests:
        payload_roles = frozenset(
            _normalise_role(role) for role in payload_manifest_roles
        )
        expanded: list[ArtifactRequirement] = []
        for requirement, source in resolved.values():
            if _is_payload_manifest_role(requirement.roles, payload_roles):
                expanded.extend(_payload_requirements(requirement, source))
        if expanded:
            combined_inputs: list[ArtifactRequirement] = [
                ArtifactRequirement(
                    role=role,
                    remote_path=requirement.remote_path,
                    sha256=requirement.sha256,
                    size_bytes=requirement.size_bytes,
                )
                for requirement, _source in resolved.values()
                for role in requirement.roles
            ]
            combined_inputs.extend(expanded)
            combined = _merge_requirements(
                combined_inputs, allowed_roots=allowed,
            )
            for requirement in combined:
                existing = resolved.get(requirement.remote_path)
                if existing is not None:
                    old_requirement, source = existing
                    resolved[requirement.remote_path] = (requirement, source)
                    continue
                resolved[requirement.remote_path] = (
                    requirement,
                    _resolve_requirement(
                        requirement,
                        run_mirror_roots=mirror_roots,
                        artifact_store_roots=store_roots,
                    ),
                )

    entries: list[dict[str, Any]] = []
    for remote_path in sorted(resolved):
        requirement, source = resolved[remote_path]
        entries.append(
            {
                "roles": list(requirement.roles),
                "remote_path": requirement.remote_path,
                "source_path": str(source.path),
                "source_kind": source.source_kind,
                "source_root": str(source.root),
                "resolution": source.resolution,
                "sha256": source.sha256,
                "size_bytes": source.size_bytes,
                "expected_size_bytes": requirement.size_bytes,
                "source_stat": {
                    "device": source.device,
                    "inode": source.inode,
                    "mtime_ns": source.mtime_ns,
                    "size_bytes": source.size_bytes,
                },
            }
        )
    payload: dict[str, Any] = {
        "schema": STAGE_MAP_SCHEMA,
        "schema_version": STAGE_MAP_SCHEMA_VERSION,
        "status": "ready",
        "local_only": True,
        "transport_performed": False,
        "source_revalidation_required_before_transport": True,
        "allowed_remote_roots": list(allowed),
        "artifact_count": len(entries),
        "total_bytes": sum(int(row["size_bytes"]) for row in entries),
        "entries": entries,
    }
    payload["stage_map_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "ArtifactRequirement",
    "ResumeArtifactResolutionError",
    "STAGE_MAP_SCHEMA",
    "STAGE_MAP_SCHEMA_VERSION",
    "build_resume_artifact_stage_map",
    "requirements_from_contract_artifacts",
]
