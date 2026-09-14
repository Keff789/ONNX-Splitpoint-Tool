from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool.artifact_store import ArtifactStore
from onnx_splitpoint_tool.remote.bundle import build_suite_bundle, remote_minimal_bundle_patterns
from onnx_splitpoint_tool.remote import bundle as bundle_module
from onnx_splitpoint_tool.workflow.artifact_registry_binding import register_benchmark_set_artifacts
from onnx_splitpoint_tool.workflow.hailo_remote_binding import _discover_hefs
from test_v2751_hailo_exact_artifact_store_restore import _register_v275_record


def _published_suite(tmp_path, monkeypatch):
    fixture = _register_v275_record(tmp_path, monkeypatch)
    suite = tmp_path / "suite"
    destination = suite / "b398/hailo/hailo8/part1/compiled.hef"
    receipt = json.loads(hailo_backend._hailo_receipt_path(fixture["old_hef"]).read_text())
    hailo_backend._publish_hailo_bundle(
        source_hef=fixture["old_hef"], destination=destination, receipt=receipt,
    )
    fresh = tmp_path / "new.hef"
    fresh.write_bytes(b"current-generation-hef")
    receipt.update(hef_sha256=hashlib.sha256(fresh.read_bytes()).hexdigest(),
                   hef_size_bytes=fresh.stat().st_size)
    hailo_backend._publish_hailo_bundle(source_hef=fresh, destination=destination, receipt=receipt)
    return suite, destination, receipt


def test_runtime_discovery_uses_only_current_public_hef(tmp_path, monkeypatch):
    suite, destination, _ = _published_suite(tmp_path, monkeypatch)
    found = _discover_hefs(suite, tmp_path, [], model_id="model", task="classification")
    assert len(found) == 1
    assert found[0]["path"] == str(destination)
    assert Path(found[0]["path"]).read_bytes() == b"current-generation-hef"


def test_remote_archive_carries_one_regular_complete_generation(tmp_path, monkeypatch):
    suite, destination, receipt = _published_suite(tmp_path, monkeypatch)
    includes, excludes = remote_minimal_bundle_patterns()
    archive = tmp_path / "suite.tar.gz"
    build_suite_bundle(suite, archive, includes=includes, excludes=excludes)
    prefix = destination.parent.relative_to(suite).as_posix()
    with tarfile.open(archive) as tf:
        names = tf.getnames()
        generation = destination.resolve().parent.relative_to(suite).as_posix()
        assert {name for name in names if ".hailo-" in name} == {
            f"{generation}/{name}" for name in
            ("compiled.hef", "hailo_hef_build_receipt.json", "cache_meta.json")
        }
        for name in ("compiled.hef", "hailo_hef_build_receipt.json", "cache_meta.json"):
            link = tf.getmember(f"{generation}/{name}")
            assert link.islnk() and link.linkname == f"{prefix}/{name}"
        for name in ("compiled.hef", "hailo_hef_build_receipt.json", "cache_meta.json"):
            assert tf.getmember(f"{prefix}/{name}").isfile()
        assert tf.extractfile(f"{prefix}/compiled.hef").read() == b"current-generation-hef"
        stored_receipt = json.load(tf.extractfile(f"{prefix}/hailo_hef_build_receipt.json"))
        stored_meta = json.load(tf.extractfile(f"{prefix}/cache_meta.json"))
        assert stored_receipt["hef_sha256"] == stored_meta["hef_sha256"] == receipt["hef_sha256"]


def test_suite_registration_stores_triplet_without_backup_duplicates(tmp_path, monkeypatch):
    suite, destination, _ = _published_suite(tmp_path, monkeypatch)
    store_root = tmp_path / "suite-store"
    report = register_benchmark_set_artifacts(
        suite_dir=suite, model_id="model", task="classification",
        profile={"artifact_store": {"root": str(store_root)}},
    )
    assert report["successful"] == 1, report
    assert report["failed"] == 0
    store = ArtifactStore(store_root)
    records = store.list(kind="hailo_hef")
    assert len(records) == 1
    record = records[0]
    assert store.validate_record(record, verify="strict")[0]
    assert record.metadata["relative_path"] == destination.relative_to(suite).as_posix()
    assert store.bundle_paths(record)["hef"].read_bytes() == b"current-generation-hef"


def test_archive_snapshot_survives_publication_between_sibling_scans(tmp_path, monkeypatch):
    suite, destination, receipt = _published_suite(tmp_path, monkeypatch)
    def interleaved_candidates(*_args):
        yield destination
        changed = tmp_path / "concurrent.hef"
        changed.write_bytes(b"published-during-bundle-scan")
        newer = dict(receipt, hef_sha256=hashlib.sha256(changed.read_bytes()).hexdigest(),
                     hef_size_bytes=changed.stat().st_size)
        hailo_backend._publish_hailo_bundle(source_hef=changed, destination=destination, receipt=newer)
        yield destination.parent / "hailo_hef_build_receipt.json"
        yield destination.parent / "cache_meta.json"
    monkeypatch.setattr(bundle_module, "_iter_selected_candidates", interleaved_candidates)
    archive = tmp_path / "concurrent.tar.gz"
    build_suite_bundle(suite, archive)
    with tarfile.open(archive) as tf:
        prefix = destination.parent.relative_to(suite).as_posix()
        data = tf.extractfile(f"{prefix}/compiled.hef").read()
        actual_receipt = json.load(tf.extractfile(f"{prefix}/hailo_hef_build_receipt.json"))
        meta = json.load(tf.extractfile(f"{prefix}/cache_meta.json"))
        assert data == b"current-generation-hef"
        assert actual_receipt["hef_sha256"] == meta["hef_sha256"] == hashlib.sha256(data).hexdigest()
    assert destination.read_bytes() == b"published-during-bundle-scan"


def test_suite_registration_does_not_save_a_new_hef_only_record(tmp_path, monkeypatch):
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "1")
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "compiled.hef").write_bytes(b"legacy")
    report = register_benchmark_set_artifacts(
        suite_dir=suite, model_id="model", task="classification",
        profile={"artifact_store": {"root": str(tmp_path / "store")}},
    )
    assert report["successful"] == 0
    assert report["registered"][0]["error"] == "legacy_unsealed"
