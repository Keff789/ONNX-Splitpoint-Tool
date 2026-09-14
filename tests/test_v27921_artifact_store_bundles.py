from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import pytest

import onnx_splitpoint_tool.artifact_store as artifact_store
from onnx_splitpoint_tool.artifact_store import ArtifactStore


def _bundle(root: Path, content: bytes = b"first-compiled-hef") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    hef = root / "compiled.hef"
    hef.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    receipt = {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "hef_sha256": digest, "hef_size_bytes": len(content),
        "cache_key": "exact-build-key", "cache_payload": {"model": "yolo26m", "boundary": "b398"},
        "preprocessing_contract_sha256": "preprocessing-token",
        "net_name": "yolo26m_part1_b398", "hw_arch": "hailo10h",
    }
    meta = {
        "schema": "onnx-splitpoint/hailo-hef-cache-meta-v2",
        "hef_sha256": digest, "hef_size": len(content),
        "cache_key": receipt["cache_key"], "payload": receipt["cache_payload"],
        "preprocessing_contract_sha256": receipt["preprocessing_contract_sha256"],
        "net_name": receipt["net_name"], "hw_arch": receipt["hw_arch"],
    }
    receipt_path = root / "hailo_hef_build_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    meta_path = root / "cache_meta.json"
    meta_path.write_text(json.dumps(meta))
    return dict(source_path=hef, receipt_path=receipt_path, cache_meta_path=meta_path,
                contract={"model": "yolo26m", "boundary": "b398", "backend": "hailo10h"})


def test_bundle_keeps_all_three_files_and_reuses_sealed_registration(tmp_path: Path):
    store = ArtifactStore(tmp_path / "store")
    fixture = _bundle(tmp_path / "source")
    first = store.register_hailo_bundle(**fixture)
    paths = store.bundle_paths(first)
    assert all(path.is_file() for path in paths.values())
    assert store.validate_record(first) == (True, "verified")
    second = store.register_hailo_bundle(**fixture, pin=True, pin_label="campaign")
    assert second.artifact_id == first.artifact_id
    assert second.object_path == first.object_path
    assert second.pinned and second.pin_label == "campaign"
    fixture["source_path"].write_bytes(b"mutated source")
    fixture["receipt_path"].write_text("{}")
    assert store.validate_record(first) == (True, "verified")


@pytest.mark.parametrize("failure", ["copy_receipt", "rename_generation", "commit_database"])
def test_failed_publication_preserves_previous_bundle_and_references(tmp_path: Path, monkeypatch, failure: str):
    store = ArtifactStore(tmp_path / "store")
    first = store.register_hailo_bundle(**_bundle(tmp_path / "first"), pin=True, pin_label="campaign")
    store.materialize(first, tmp_path / "run" / "compiled.hef", reference="evidence-run")
    original_files = {key: path.read_bytes() for key, path in store.bundle_paths(first).items()}
    updated = _bundle(tmp_path / "second", b"updated-compiled-hef")
    if failure == "copy_receipt":
        monkeypatch.setattr(artifact_store, "_owned_reflink_copy", lambda *args: 1)
        real_copy = artifact_store.shutil.copy2
        def fail_copy(src, dst, *args, **kwargs):
            if Path(src).name == "hailo_hef_build_receipt.json":
                raise OSError("injected receipt copy failure")
            return real_copy(src, dst, *args, **kwargs)
        monkeypatch.setattr(artifact_store.shutil, "copy2", fail_copy)
    elif failure == "rename_generation":
        real_replace = artifact_store.os.replace
        def fail_rename(src, dst):
            if Path(dst).name.startswith("generation-"):
                raise OSError("injected generation rename failure")
            return real_replace(src, dst)
        monkeypatch.setattr(artifact_store.os, "replace", fail_rename)
    else:
        def fail_commit(**kwargs):
            assert kwargs["object_path"].is_file()
            assert (kwargs["object_path"].parent / "cache_meta.json").is_file()
            assert (kwargs["object_path"].parent / "hailo_hef_build_receipt.json").is_file()
            raise sqlite3.OperationalError("injected database failure")
        monkeypatch.setattr(store, "_register_published_record", fail_commit)
    with pytest.raises((OSError, sqlite3.OperationalError)):
        store.register_hailo_bundle(**updated)
    surviving = store.lookup(kind="hailo_hef", contract=first.contract, verify="strict")
    assert surviving is not None and surviving.object_path == first.object_path
    assert surviving.pinned and surviving.pin_label == "campaign"
    assert {key: path.read_bytes() for key, path in store.bundle_paths(surviving).items()} == original_files
    with store._connect() as con:
        assert con.execute("SELECT ref_value FROM artifact_refs").fetchone()[0] == "evidence-run"
    assert not list(store.root.rglob(".staging-*"))


def test_successful_replacement_preserves_old_bytes_pin_and_reference(tmp_path: Path):
    store = ArtifactStore(tmp_path / "store")
    first = store.register_hailo_bundle(**_bundle(tmp_path / "first"), pin=True, pin_label="campaign")
    store.materialize(first, tmp_path / "run.hef", reference="evidence")
    replacement = store.register_hailo_bundle(**_bundle(tmp_path / "second", b"replacement-hef"))
    assert replacement.artifact_id == first.artifact_id
    assert replacement.pinned and replacement.pin_label == "campaign"
    assert Path(first.object_path).read_bytes() == b"first-compiled-hef"
    assert Path(replacement.object_path).read_bytes() == b"replacement-hef"
    with store._connect() as con:
        assert con.execute("SELECT COUNT(*) FROM artifact_refs").fetchone()[0] == 1


def test_duplicate_selection_ignores_recency_and_skips_same_size_corruption(tmp_path: Path):
    store = ArtifactStore(tmp_path / "store")
    for number in range(3):
        source = tmp_path / f"candidate-{number}.dxnn"
        source.write_bytes(f"valid-{number}".encode())
        store.register(source_path=source, kind="deepx_dxnn", contract={"number": number},
                       metadata={"legacy_cache_key": "same-alias"})
    ordered = store.candidates_by_metadata(kind="deepx_dxnn", key="legacy_cache_key", value="same-alias")
    for row in reversed(ordered):
        store.lookup(kind=row.kind, contract=row.contract)
    selected = store.find_by_metadata(kind="deepx_dxnn", key="legacy_cache_key", value="same-alias", verify="strict")
    assert selected.artifact_id == ordered[0].artifact_id
    Path(ordered[0].object_path).write_bytes(b"x" * ordered[0].size_bytes)
    selected = store.find_by_metadata(kind="deepx_dxnn", key="legacy_cache_key", value="same-alias", verify="strict")
    assert selected.artifact_id == ordered[1].artifact_id


@pytest.mark.parametrize("sidecar", ["receipt", "cache_meta"])
def test_corrupt_sidecar_prevents_lookup_and_export(tmp_path: Path, sidecar: str):
    store = ArtifactStore(tmp_path / "store")
    record = store.register_hailo_bundle(**_bundle(tmp_path / "source"), pin=True)
    store.bundle_paths(record)[sidecar].write_text("{}")
    assert not store.validate_record(record)[0]
    assert store.lookup(kind=record.kind, contract=record.contract, verify="strict") is None
    assert not store.verify(strict=True)["ok"]
    pack = tmp_path / "existing.zip"
    pack.write_bytes(b"previous-backup")
    with pytest.raises(ValueError):
        store.export_pack(pack)
    assert pack.read_bytes() == b"previous-backup"


def test_pack_roundtrip_keeps_complete_seal_and_old_packs_still_import(tmp_path: Path):
    store = ArtifactStore(tmp_path / "store")
    record = store.register_hailo_bundle(**_bundle(tmp_path / "source"), pin=True)
    legacy = tmp_path / "legacy.dxnn"
    legacy.write_bytes(b"deepx-artifact")
    store.register(source_path=legacy, kind="deepx_dxnn", contract={"kind": "deepx"}, pin=True)
    pack = store.export_pack(tmp_path / "backup.zip")
    with zipfile.ZipFile(pack) as archive:
        assert any(name.endswith("/hailo_hef_build_receipt.json") for name in archive.namelist())
        assert any(name.endswith("/cache_meta.json") for name in archive.namelist())
    imported = ArtifactStore(tmp_path / "imported")
    assert imported.import_pack(pack)["imported_count"] == 2
    restored = imported.lookup(kind="hailo_hef", contract=record.contract, verify="strict")
    assert restored is not None and imported.validate_record(restored)[0]
    assert {key: p.read_bytes() for key, p in imported.bundle_paths(restored).items()} == {
        key: p.read_bytes() for key, p in store.bundle_paths(record).items()
    }


def test_read_only_probes_do_not_create_store_or_change_access_state(tmp_path: Path):
    missing = ArtifactStore(tmp_path / "absent", read_only=True)
    assert not missing.root.exists()
    with pytest.raises(sqlite3.OperationalError):
        missing.list()
    assert not missing.root.exists()
    store = ArtifactStore(tmp_path / "store")
    original = store.register_hailo_bundle(**_bundle(tmp_path / "source"))
    readonly = ArtifactStore(store.root, read_only=True)
    for _ in range(2):
        assert readonly.validate_record(original)[0]
        assert readonly.lookup(kind=original.kind, contract=original.contract, verify="strict") is not None
        assert readonly.find_by_metadata(kind="hailo_hef", key="legacy_cache_key", value="exact-build-key")
    after = readonly.list()[0]
    assert after.last_accessed == original.last_accessed
    assert after.verification_status == original.verification_status


def test_index_bare_hef_labels_legacy_unsealed(tmp_path: Path):
    cache = tmp_path / "cache" / "old-generation"
    cache.mkdir(parents=True)
    (cache / "compiled.hef").write_bytes(b"legacy-hef")
    store = ArtifactStore(tmp_path / "store")
    assert store.index_existing([cache.parent])["indexed_count"] == 1
    assert store.list()[0].metadata["bundle_status"] == "legacy_unsealed"


def test_register_repairs_same_size_corrupt_existing_content(tmp_path: Path):
    store = ArtifactStore(tmp_path / "store")
    source = tmp_path / "compiled.dxnn"
    source.write_bytes(b"intact-bytes")
    first = store.register(source_path=source, kind="deepx_dxnn", contract={"model": "first"})
    Path(first.object_path).write_bytes(b"x" * first.size_bytes)
    repaired = store.register(source_path=source, kind="deepx_dxnn", contract={"model": "second"})
    assert store.validate_record(repaired)[0]


def test_materialize_publishes_triplet_without_mutating_previous_snapshot(tmp_path: Path):
    store = ArtifactStore(tmp_path / "store")
    first = store.register_hailo_bundle(**_bundle(tmp_path / "first"))
    target = tmp_path / "destination" / "compiled.hef"
    assert store.materialize(first, target) == "atomic_hailo_bundle"
    old_snapshot = target.resolve()
    old_receipt = (old_snapshot.parent / "hailo_hef_build_receipt.json").read_bytes()
    second = store.register_hailo_bundle(**_bundle(tmp_path / "second", b"replacement-hef"))
    assert store.materialize(second, target) == "atomic_hailo_bundle"
    new_snapshot = target.resolve()
    assert new_snapshot != old_snapshot
    assert old_snapshot.read_bytes() == b"first-compiled-hef"
    assert (old_snapshot.parent / "hailo_hef_build_receipt.json").read_bytes() == old_receipt
    assert new_snapshot.read_bytes() == b"replacement-hef"
    assert (new_snapshot.parent / "cache_meta.json").is_file()
    with pytest.raises(ValueError, match="immutable"):
        store.materialize(second, old_snapshot)
    legacy_path = tmp_path / "legacy.hef"
    legacy_path.write_bytes(b"bare-hef")
    legacy = store.register(source_path=legacy_path, kind="hailo_hef", contract={"legacy": True})
    with pytest.raises(ValueError, match="HEF-only"):
        store.materialize(legacy, target)
    with pytest.raises(ValueError, match="legacy_unsealed"):
        store.materialize(legacy, tmp_path / "empty-destination" / "compiled.hef")
    assert target.resolve() == new_snapshot
