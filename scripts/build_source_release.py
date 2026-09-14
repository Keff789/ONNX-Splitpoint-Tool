#!/usr/bin/env python3
"""Build or verify the deterministic, manifest-bound Tool source ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_source_manifest as source_manifest  # noqa: E402


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RELEASE_INDEX_FILES = ("SOURCE_MANIFEST.json", "SHA256SUMS.txt")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_manifest(root: Path) -> dict[str, Any]:
    verification = source_manifest.verify(root)
    if verification.get("ok") is not True:
        raise ValueError(
            "source manifest is stale or invalid; use an untouched clean "
            "source tree. Regenerating SOURCE_MANIFEST.json is a release-"
            "maintainer action and must follow an explicit source audit: "
            f"{verification.get('checks')!r}"
        )
    payload = json.loads(
        (root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"),
    )
    if not isinstance(payload, dict):
        raise ValueError("SOURCE_MANIFEST.json must contain an object")
    return payload


def _safe_source(root: Path, relative: str) -> Path:
    logical = PurePosixPath(str(relative))
    if (
        logical.is_absolute()
        or not logical.parts
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise ValueError(f"unsafe source-release path: {relative!r}")
    source = root.joinpath(*logical.parts)
    cursor = root
    for part in logical.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(
                f"source-release members may not be symlinks: {relative}"
            )
    if not source.is_file():
        raise ValueError(f"source-release member is not a file: {relative}")
    try:
        source.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"source-release member escapes the source root: {relative}"
        ) from exc
    return source


def _release_members(
    root: Path,
    manifest: Mapping[str, Any],
) -> list[tuple[str, Path]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("SOURCE_MANIFEST.json has no file list")
    names = [
        str(row.get("path") or "")
        for row in rows
        if isinstance(row, Mapping)
    ]
    if len(names) != len(rows) or len(set(names)) != len(names):
        raise ValueError("source manifest contains invalid or duplicate paths")
    names.extend(RELEASE_INDEX_FILES)
    if len(set(names)) != len(names):
        raise ValueError("source manifest unexpectedly contains index files")
    return [
        (name, _safe_source(root, name))
        for name in sorted(names)
    ]


def _archive_mode(path: Path) -> int:
    permissions = 0o755 if path.stat().st_mode & 0o111 else 0o644
    return stat.S_IFREG | permissions


def _require_external_output(root: Path, output: Path) -> None:
    try:
        output.relative_to(root)
    except ValueError:
        return
    raise ValueError(
        "source-release output must be outside the source root: "
        f"{output}"
    )


def _member_name(prefix: str, relative: str) -> str:
    clean_prefix = str(prefix or "").strip().strip("/")
    logical_prefix = PurePosixPath(clean_prefix)
    if (
        not clean_prefix
        or logical_prefix.is_absolute()
        or any(part in {"", ".", ".."} for part in logical_prefix.parts)
    ):
        raise ValueError(f"unsafe source-release prefix: {prefix!r}")
    return f"{logical_prefix.as_posix()}/{relative}"


def verify_archive(
    root: Path,
    archive_path: Path,
    *,
    prefix: str | None = None,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    archive_path = archive_path.expanduser().resolve()
    manifest = _load_verified_manifest(root)
    resolved_prefix = prefix or (
        f"ONNX-Splitpoint-Tool_v{manifest['package_version']}"
    )
    members = _release_members(root, manifest)
    expected = [
        _member_name(resolved_prefix, relative)
        for relative, _source in members
    ]
    checks: dict[str, bool] = {
        "archive_exists": archive_path.is_file(),
    }
    if not archive_path.is_file():
        return {
            "schema": "onnx-splitpoint/source-release-verification",
            "schema_version": 1,
            "ok": False,
            "checks": checks,
            "archive": str(archive_path),
        }

    metadata_ok = True
    content_ok = True
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        checks.update(
            {
                "member_order_and_set": names == expected,
                "member_names_unique": len(names) == len(set(names)),
                "crc_clean": archive.testzip() is None,
                "archive_comment_empty": archive.comment == b"",
            }
        )
        if names == expected:
            for info, (_relative, source) in zip(infos, members):
                expected_mode = _archive_mode(source)
                metadata_ok = metadata_ok and all(
                    (
                        info.date_time == FIXED_ZIP_TIMESTAMP,
                        info.create_system == 3,
                        info.compress_type == zipfile.ZIP_DEFLATED,
                        info.external_attr >> 16 == expected_mode,
                        info.extra == b"",
                        info.comment == b"",
                    )
                )
                content_ok = (
                    content_ok
                    and archive.read(info.filename) == source.read_bytes()
                )
        else:
            metadata_ok = False
            content_ok = False
    checks["deterministic_metadata"] = metadata_ok
    checks["content_matches_source"] = content_ok
    return {
        "schema": "onnx-splitpoint/source-release-verification",
        "schema_version": 1,
        "ok": all(checks.values()),
        "checks": checks,
        "archive": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": _sha256_file(archive_path),
        "member_count": len(expected),
        "manifest_file_count": int(manifest.get("file_count") or 0),
        "package_version": manifest.get("package_version"),
        "workflow_version": manifest.get("workflow_version"),
        "prefix": resolved_prefix,
    }


def build_release(
    root: Path,
    output: Path,
    *,
    prefix: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()
    _require_external_output(root, output)
    manifest = _load_verified_manifest(root)
    resolved_prefix = prefix or (
        f"ONNX-Splitpoint-Tool_v{manifest['package_version']}"
    )
    members = _release_members(root, manifest)
    if output.exists() and not force:
        raise FileExistsError(
            f"source-release output already exists: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"temporary source-release output already exists: {temporary}"
        )
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for relative, source in members:
                info = zipfile.ZipInfo(
                    _member_name(resolved_prefix, relative),
                    date_time=FIXED_ZIP_TIMESTAMP,
                )
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = _archive_mode(source) << 16
                archive.writestr(
                    info,
                    source.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        verification = verify_archive(
            root,
            temporary,
            prefix=resolved_prefix,
        )
        if verification.get("ok") is not True:
            raise ValueError(
                "new source-release archive failed verification: "
                f"{verification.get('checks')!r}"
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return verify_archive(root, output, prefix=resolved_prefix)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-archive", type=Path)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.expanduser().resolve()
    if args.out is not None and args.verify_archive is not None:
        raise SystemExit("--out and --verify-archive are mutually exclusive")
    manifest = _load_verified_manifest(root)
    prefix = args.prefix or (
        f"ONNX-Splitpoint-Tool_v{manifest['package_version']}"
    )
    if args.verify_archive is not None:
        result = verify_archive(
            root,
            args.verify_archive,
            prefix=prefix,
        )
    else:
        output = args.out or (
            root.parent / f"{prefix}_source.zip"
        )
        result = build_release(
            root,
            output,
            prefix=prefix,
            force=bool(args.force),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
