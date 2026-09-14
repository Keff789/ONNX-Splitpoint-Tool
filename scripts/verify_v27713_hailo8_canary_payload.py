#!/usr/bin/env python3
"""Verify and safely extract one Hailo-8 structural canary payload.

This helper deliberately uses only the Python standard library.  The caller
must bind the archive bytes and the embedded manifest bytes independently.
No payload member is executed before this verifier has accepted the complete,
exact file set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "onnx-splitpoint/v27713-hailo8-structural-canary-payload"
SCHEMA_VERSION = 1
MANIFEST_MEMBER = "payload/canary_payload_manifest.json"
MAX_FILE_COUNT = 20_000
MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024


class PayloadVerificationError(RuntimeError):
    """The uploaded payload is not exactly the locally sealed payload."""


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _load_json_bytes(raw: bytes) -> dict[str, Any]:
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
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except Exception as exc:
        raise PayloadVerificationError(
            f"payload_manifest_json_invalid:{type(exc).__name__}"
        ) from exc
    if duplicate or not isinstance(value, dict):
        raise PayloadVerificationError("payload_manifest_invalid_or_duplicate_json")
    return value


def _safe_member_name(value: str) -> str:
    name = str(value or "")
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or path.is_absolute()
        or ".." in path.parts
        or path.parts[:1] != ("payload",)
        or len(path.parts) < 2
        or any(part in {"", "."} for part in path.parts)
        or path.as_posix() != name
    ):
        raise PayloadVerificationError(f"unsafe_payload_member:{name}")
    return path.as_posix()


def _manifest_file_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("files"), list)
        or not isinstance(payload.get("models"), list)
    ):
        raise PayloadVerificationError("payload_manifest_contract_invalid")
    rows: dict[str, dict[str, Any]] = {}
    for raw in payload["files"]:
        if not isinstance(raw, dict):
            raise PayloadVerificationError("payload_manifest_file_row_invalid")
        declared_path = str(raw.get("path") or "")
        declared = PurePosixPath(declared_path)
        if (
            not declared_path
            or declared.is_absolute()
            or declared_path.startswith("/")
            or ".." in declared.parts
            or declared.as_posix() != declared_path
        ):
            raise PayloadVerificationError(
                f"payload_manifest_path_invalid:{declared_path}"
            )
        relative = _safe_member_name(f"payload/{declared_path}")
        if relative in rows:
            raise PayloadVerificationError(
                f"payload_manifest_duplicate_path:{relative}"
            )
        digest = str(raw.get("sha256") or "").strip().lower()
        size = raw.get("size_bytes")
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or type(size) is not int
            or size < 0
            or size > MAX_MEMBER_BYTES
        ):
            raise PayloadVerificationError(
                f"payload_manifest_file_identity_invalid:{relative}"
            )
        rows[relative] = dict(raw)
    if not rows or len(rows) > MAX_FILE_COUNT:
        raise PayloadVerificationError("payload_manifest_file_count_invalid")
    if sum(int(row["size_bytes"]) for row in rows.values()) > MAX_TOTAL_BYTES:
        raise PayloadVerificationError("payload_manifest_total_bytes_exceeded")
    return rows


def _validate_model_bindings(
    payload: dict[str, Any], rows: dict[str, dict[str, Any]],
) -> None:
    if (
        payload.get("artifact_policy") != "cache_verify_only"
        or payload.get("runtime_entry_point")
        != "scripts/smoke_hailo10_full_from_benchmarkset.py"
        or payload.get("compile_or_build_entry_points_allowed") != []
    ):
        raise PayloadVerificationError("payload_execution_contract_invalid")
    required_scripts = {
        "payload/scripts/smoke_hailo10_hef_runner.py",
        "payload/scripts/smoke_hailo10_full_from_benchmarkset.py",
        "payload/scripts/verify_v27713_hailo8_canary_payload.py",
    }
    if not required_scripts.issubset(rows):
        raise PayloadVerificationError("payload_runtime_scripts_missing")

    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise PayloadVerificationError("payload_model_inventory_invalid")
    identities: set[tuple[int, str]] = set()

    def bind(
        model: dict[str, Any], path_key: str, sha_key: str,
        size_key: str = "",
    ) -> None:
        relative = str(model.get(path_key) or "")
        digest = str(model.get(sha_key) or "").strip().lower()
        row = rows.get(f"payload/{relative}")
        if (
            not relative
            or row is None
            or row.get("sha256") != digest
            or (
                size_key
                and int(row.get("size_bytes") or -1)
                != int(model.get(size_key) or -2)
            )
        ):
            raise PayloadVerificationError(
                f"payload_model_file_binding_invalid:{path_key}"
            )

    for raw in models:
        if not isinstance(raw, dict):
            raise PayloadVerificationError("payload_model_inventory_row_invalid")
        model_id = str(raw.get("model_id") or "")
        index = raw.get("index")
        task = str(raw.get("task") or "")
        if (
            type(index) is not int
            or index < 0
            or not model_id
            or task not in {"classification", "detection"}
            or (index, model_id) in identities
        ):
            raise PayloadVerificationError("payload_model_identity_invalid")
        identities.add((index, model_id))
        bind(raw, "benchmark_set_json_path", "benchmark_set_json_sha256")
        bind(raw, "output_contracts_path", "output_contracts_sha256")
        bind(raw, "receipt_path", "receipt_sha256")
        bind(raw, "hef_path", "hef_sha256", "hef_size_bytes")
        onnx_path = str(raw.get("onnx_path") or "")
        onnx_sha = str(raw.get("onnx_sha256") or "")
        if bool(onnx_path) != bool(onnx_sha):
            raise PayloadVerificationError("payload_onnx_binding_partial")
        if onnx_path:
            bind(raw, "onnx_path", "onnx_sha256")
            receipt_source_sha = str(
                raw.get("receipt_source_onnx_sha256") or ""
            ).strip().lower()
            receipt_compiler_sha = str(
                raw.get("receipt_compiler_onnx_sha256") or ""
            ).strip().lower()
            if (
                receipt_source_sha != onnx_sha
                or len(receipt_compiler_sha) != 64
                or any(
                    char not in "0123456789abcdef"
                    for char in receipt_compiler_sha
                )
            ):
                raise PayloadVerificationError(
                    "payload_onnx_receipt_binding_invalid"
                )
        for digest_key in (
            "preprocessing_contract_sha256",
            "output_contract_resolution_sha256",
        ):
            digest = str(raw.get(digest_key) or "").strip().lower()
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise PayloadVerificationError(
                    f"payload_model_digest_invalid:{digest_key}"
                )
        images = raw.get("images")
        if not isinstance(images, list) or not images:
            raise PayloadVerificationError("payload_model_images_invalid")
        image_paths: set[str] = set()
        for image in images:
            if not isinstance(image, dict):
                raise PayloadVerificationError("payload_image_row_invalid")
            path = str(image.get("payload_path") or "")
            if path in image_paths:
                raise PayloadVerificationError("payload_image_path_duplicate")
            image_paths.add(path)
            bind(image, "payload_path", "sha256", "size_bytes")


def verify_and_extract(
    archive_path: Path,
    destination: Path,
    *,
    expected_archive_sha256: str,
    expected_archive_size: int,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    archive_path = Path(os.path.abspath(os.fspath(archive_path)))
    destination = Path(os.path.abspath(os.fspath(destination)))
    archive_stat = archive_path.lstat()
    if stat.S_ISLNK(archive_stat.st_mode) or not stat.S_ISREG(archive_stat.st_mode):
        raise PayloadVerificationError("payload_archive_not_regular")
    if (
        type(expected_archive_size) is not int
        or archive_stat.st_size != expected_archive_size
        or _sha256_file(archive_path) != expected_archive_sha256
    ):
        raise PayloadVerificationError("payload_archive_identity_mismatch")
    if os.path.lexists(destination):
        raise PayloadVerificationError("payload_destination_must_not_exist")

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not members or len(members) > MAX_FILE_COUNT + 1:
            raise PayloadVerificationError("payload_archive_member_count_invalid")
        by_name: dict[str, tarfile.TarInfo] = {}
        total = 0
        for member in members:
            name = _safe_member_name(member.name)
            if name in by_name:
                raise PayloadVerificationError(
                    f"payload_archive_duplicate_member:{name}"
                )
            if not member.isfile() or member.size < 0 or member.size > MAX_MEMBER_BYTES:
                raise PayloadVerificationError(
                    f"payload_archive_member_type_or_size_invalid:{name}"
                )
            total += int(member.size)
            if total > MAX_TOTAL_BYTES:
                raise PayloadVerificationError("payload_archive_total_bytes_exceeded")
            by_name[name] = member

        manifest_member = by_name.get(MANIFEST_MEMBER)
        if manifest_member is None:
            raise PayloadVerificationError("payload_manifest_member_missing")
        manifest_handle = archive.extractfile(manifest_member)
        if manifest_handle is None:
            raise PayloadVerificationError("payload_manifest_member_unreadable")
        manifest_raw = manifest_handle.read()
        manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
        if manifest_sha256 != expected_manifest_sha256:
            raise PayloadVerificationError("payload_manifest_sha256_mismatch")
        manifest = _load_json_bytes(manifest_raw)
        file_rows = _manifest_file_rows(manifest)
        expected_names = set(file_rows) | {MANIFEST_MEMBER}
        if set(by_name) != expected_names:
            missing = sorted(expected_names - set(by_name))
            extra = sorted(set(by_name) - expected_names)
            raise PayloadVerificationError(
                f"payload_archive_exact_set_mismatch:missing={missing}:extra={extra}"
            )

        for name, row in file_rows.items():
            member = by_name[name]
            if int(member.size) != int(row["size_bytes"]):
                raise PayloadVerificationError(
                    f"payload_member_size_mismatch:{name}"
                )
            handle = archive.extractfile(member)
            if handle is None or _sha256_stream(handle) != row["sha256"]:
                raise PayloadVerificationError(
                    f"payload_member_sha256_mismatch:{name}"
                )
        _validate_model_bindings(manifest, file_rows)

        destination.mkdir(mode=0o700, parents=False)
        destination_root = destination.resolve(strict=True)
        for name in sorted(expected_names):
            member = by_name[name]
            target = destination / PurePosixPath(name).relative_to("payload")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if target.parent.resolve(strict=True) != (
                destination_root / target.parent.relative_to(destination)
            ):
                raise PayloadVerificationError(
                    f"payload_extract_parent_identity_invalid:{name}"
                )
            handle = archive.extractfile(member)
            if handle is None:
                raise PayloadVerificationError(f"payload_member_unreadable:{name}")
            with target.open("xb") as output:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    output.write(block)

    observed: set[str] = set()
    for current, directories, filenames in os.walk(destination, followlinks=False):
        current_path = Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise PayloadVerificationError("payload_extract_contains_symlink")
        for name in filenames:
            path = current_path / name
            relative = path.relative_to(destination).as_posix()
            member_name = f"payload/{relative}"
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise PayloadVerificationError(
                    f"payload_extract_file_unsafe:{relative}"
                )
            observed.add(member_name)
            if member_name == MANIFEST_MEMBER:
                if _sha256_file(path) != expected_manifest_sha256:
                    raise PayloadVerificationError("extracted_manifest_sha256_mismatch")
                continue
            row = file_rows.get(member_name)
            if (
                row is None
                or info.st_size != int(row["size_bytes"])
                or _sha256_file(path) != row["sha256"]
            ):
                raise PayloadVerificationError(
                    f"payload_extract_identity_mismatch:{relative}"
                )
    if observed != expected_names:
        raise PayloadVerificationError("payload_extract_exact_set_mismatch")
    return {
        "schema": "onnx-splitpoint/v27713-hailo8-structural-canary-payload-attestation",
        "schema_version": 1,
        "status": "PASS",
        "archive_sha256": expected_archive_sha256,
        "archive_size_bytes": expected_archive_size,
        "manifest_sha256": expected_manifest_sha256,
        "file_count": len(file_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-archive-size", type=int, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    ns = parser.parse_args()
    try:
        result = verify_and_extract(
            Path(ns.archive),
            Path(ns.destination),
            expected_archive_sha256=str(ns.expected_archive_sha256),
            expected_archive_size=int(ns.expected_archive_size),
            expected_manifest_sha256=str(ns.expected_manifest_sha256),
        )
    except Exception as exc:
        result = {
            "schema": "onnx-splitpoint/v27713-hailo8-structural-canary-payload-attestation",
            "schema_version": 1,
            "status": "FAIL",
            "reason": f"{type(exc).__name__}:{exc}",
        }
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
