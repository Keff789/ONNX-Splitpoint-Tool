from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_cache_bundle as bundle
from onnx_splitpoint_tool.artifact_store import ArtifactStore
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "_hailo_sdk_version_token", lambda: "hailo-dataflow-compiler:3.31.0")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"offline-source-onnx-identity")
    contract = canonical_image_preprocessing_contract("classification", (224, 224))
    key, payload = backend._hailo_cache_key(
        model_path=model, activation_part1=None, hw_arch="hailo10h", opt_level=1,
        calib_dir=None, calib_count=1, calib_batch_size=1, extra_model_script="",
        start_nodes=None, end_nodes=None, preprocessing_contract=contract,
        effective_calib_count=1, calibration_storage="memory",
        calibration_memory_cap_bytes=64 * 1024 * 1024,
        net_name="yolo26m_part1_b398", net_input_shapes={"images": [1, 3, 224, 224]},
        disable_rt_metadata_extraction=True,
    )
    records = []
    for number, content in enumerate((b"first-compiled-hef", b"other-valid-compiled-hef")):
        directory = tmp_path / f"source-{number}"
        directory.mkdir()
        hef = directory / "compiled.hef"
        hef.write_bytes(content)
        receipt = backend._write_hailo_receipt(
            hef_path=hef, source_onnx=model, compiler_onnx=model,
            hw_arch="hailo10h", net_name="yolo26m_part1_b398",
            preprocessing_contract=contract,
            preprocessing_sha256=preprocessing_contract_sha256(contract),
            cache_key=key, cache_payload=payload,
            calibration_identity=payload["calibration_identity"], calibration_count=1,
        )
        records.append((hef, receipt))
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED", "1")
    return records


def _publish(record, destination):
    hef, receipt = record
    return backend._publish_hailo_bundle(source_hef=hef, destination=destination, receipt=receipt)


def _assert_tuple(path, receipt):
    snapshot = bundle.snapshot_hef(path)
    loaded = backend._load_valid_hailo_receipt(snapshot)
    assert loaded == receipt
    metadata = json.loads((snapshot.parent / bundle.META_NAME).read_text())
    assert backend._hailo_cache_meta_matches_receipt(metadata, receipt)
    assert backend._hailo_cache_bundle_status(path)["status"] == "sealed"
    return snapshot


def test_pointer_commit_publishes_all_three_and_keeps_previous_snapshot(tmp_path, artifacts):
    target = tmp_path / "output" / "compiled.hef"
    previous = _publish(artifacts[0], target)
    _assert_tuple(target, artifacts[0][1])
    current = _publish(artifacts[1], target)
    assert current != previous
    _assert_tuple(target, artifacts[1][1])
    _assert_tuple(previous, artifacts[0][1])
    for name in (target.name, bundle.RECEIPT_NAME, bundle.META_NAME):
        assert (target.parent / name).is_symlink()
        assert (target.parent / name).resolve().parent == current.parent


def test_repeated_identical_backup_does_not_duplicate_generation(tmp_path, artifacts):
    target = tmp_path / "output" / "compiled.hef"
    first = _publish(artifacts[0], target)
    before = sorted((target.parent / bundle.GENERATIONS_NAME).iterdir())
    for _ in range(5):
        assert _publish(artifacts[0], target) == first
    assert sorted((target.parent / bundle.GENERATIONS_NAME).iterdir()) == before


def test_failed_pointer_commit_keeps_prior_complete_bundle(tmp_path, artifacts, monkeypatch):
    target = tmp_path / "output" / "compiled.hef"
    first = _publish(artifacts[0], target)
    replace = bundle._replace_link

    def fail_commit(path, link_target):
        if path.name == bundle.POINTER_NAME:
            raise OSError("simulated interrupted commit")
        return replace(path, link_target)

    monkeypatch.setattr(bundle, "_replace_link", fail_commit)
    with pytest.raises(OSError, match="interrupted commit"):
        _publish(artifacts[1], target)
    assert bundle.snapshot_hef(target) == first
    _assert_tuple(target, artifacts[0][1])
    # The newly compiled, validated tuple also remains recoverable locally.
    retained = list((target.parent / bundle.GENERATIONS_NAME).glob("*/compiled.hef"))
    assert any(p.read_bytes() == artifacts[1][0].read_bytes() for p in retained)


def test_invalid_candidate_never_replaces_existing_generation(tmp_path, artifacts):
    target = tmp_path / "output" / "compiled.hef"
    first = _publish(artifacts[0], target)
    with pytest.raises(ValueError, match="bundle_validation_failed"):
        _publish((artifacts[1][0], artifacts[0][1]), target)
    assert bundle.snapshot_hef(target) == first
    _assert_tuple(target, artifacts[0][1])


def test_unsupported_symlinks_preserve_original_files(tmp_path, artifacts, monkeypatch):
    target = tmp_path / "legacy" / "compiled.hef"
    target.parent.mkdir()
    for name, data in ((target.name, b"old-hef"), (bundle.RECEIPT_NAME, b"old-receipt"), (bundle.META_NAME, b"old-meta")):
        (target.parent / name).write_bytes(data)
    before = {p.name: p.read_bytes() for p in target.parent.iterdir()}

    def unsupported(*args, **kwargs):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(bundle.os, "symlink", unsupported)
    with pytest.raises(OSError, match="symlinks unavailable"):
        _publish(artifacts[0], target)
    for name, data in before.items():
        assert (target.parent / name).read_bytes() == data
        assert not (target.parent / name).is_symlink()


def test_hef_only_legacy_is_explicit_non_reusable_and_preserved(tmp_path, artifacts):
    target = tmp_path / "legacy" / "compiled.hef"
    target.parent.mkdir()
    target.write_bytes(b"unsealed-historical-hef")
    before = sorted(target.parent.iterdir())
    status = backend._hailo_cache_bundle_status(target)
    assert status["status"] == "legacy_unsealed"
    assert status["reusable"] is False
    assert backend._load_valid_hailo_receipt(target) is None
    assert sorted(target.parent.iterdir()) == before
    _publish(artifacts[0], target)
    previous = list((target.parent / bundle.GENERATIONS_NAME).glob("previous-*/compiled.hef"))
    assert len(previous) == 1
    assert previous[0].read_bytes() == b"unsealed-historical-hef"
    _assert_tuple(target, artifacts[0][1])


def test_concurrent_publishers_and_readers_use_consistent_snapshots(tmp_path, artifacts):
    target = tmp_path / "output" / "compiled.hef"
    _publish(artifacts[0], target)

    def publish_many(record):
        for _ in range(12):
            _publish(record, target)

    def read_many():
        for _ in range(120):
            snapshot = bundle.snapshot_hef(target)
            receipt = backend._load_valid_hailo_receipt(snapshot)
            assert receipt is not None
            metadata = json.loads((snapshot.parent / bundle.META_NAME).read_text())
            assert backend._hailo_cache_meta_matches_receipt(metadata, receipt)

    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(publish_many, row) for row in artifacts]
        tasks += [executor.submit(read_many) for _ in range(2)]
        for task in tasks:
            task.result(timeout=30)


def _register(store, record, sequence, receipt=None):
    hef, original = record
    receipt = original if receipt is None else receipt
    return store.register(
        source_path=hef, kind="hailo_hef", contract={"fixture": sequence},
        metadata={"legacy_cache_key": original["cache_key"], "build_receipt": receipt},
    )


def _restore(tmp_path, receipt, diagnostics):
    return backend._restore_hailo_v2_artifact_store_exact(
        destination=tmp_path / "restore" / "compiled.hef", cache_dir=tmp_path / "cache",
        cache_key=receipt["cache_key"], cache_payload=receipt["cache_payload"],
        preprocessing_sha256=receipt["preprocessing_contract_sha256"],
        source_onnx_sha256=receipt["source_onnx_sha256"],
        net_name=receipt["net_name"], hw_arch=receipt["hw_arch"],
        diagnostics=diagnostics, read_only=True,
    )


def test_invalid_low_id_duplicate_cannot_mask_valid_candidate(tmp_path, artifacts):
    store = ArtifactStore(tmp_path / "store")
    invalid = copy.deepcopy(artifacts[0][1])
    invalid["prepared_calibration_identity_sha256"] = "0" * 64
    _register(store, artifacts[0], "invalid-first", receipt=invalid)
    good = _register(store, artifacts[0], "valid-second")
    audit = {}
    restored = _restore(tmp_path, artifacts[0][1], audit)
    assert restored is not None
    assert restored["artifact_id"] == good.artifact_id
    assert audit["matched_candidates"] == 2
    assert audit["valid_candidates"] == 1
    assert audit["rejected_candidates"] == 1
    assert not (tmp_path / "cache").exists()
    _assert_tuple(tmp_path / "restore" / "compiled.hef", artifacts[0][1])


def test_equivalent_duplicates_choose_lowest_id_independent_of_list_order(tmp_path, artifacts, monkeypatch):
    store = ArtifactStore(tmp_path / "store")
    first = _register(store, artifacts[0], "one")
    _register(store, artifacts[0], "two")
    original_list = ArtifactStore.list
    monkeypatch.setattr(ArtifactStore, "list", lambda self, **kwargs: list(reversed(original_list(self, **kwargs))))
    audit = {}
    result = _restore(tmp_path, artifacts[0][1], audit)
    assert result["artifact_id"] == first.artifact_id
    assert audit["valid_candidates"] == 2


def test_conflicting_valid_duplicates_fail_closed_without_touching_existing_output(tmp_path, artifacts):
    store = ArtifactStore(tmp_path / "store")
    _register(store, artifacts[0], "one")
    _register(store, artifacts[1], "two")
    target = tmp_path / "restore" / "compiled.hef"
    before = _publish(artifacts[0], target)
    audit = {}
    assert _restore(tmp_path, artifacts[0][1], audit) is None
    assert audit["status"] == "conflicting_valid_duplicates"
    assert audit["valid_candidates"] == 2
    assert bundle.snapshot_hef(target) == before
    _assert_tuple(target, artifacts[0][1])


def test_metadata_corruption_invalidates_complete_generation(tmp_path, artifacts):
    target = tmp_path / "output" / "compiled.hef"
    published = _publish(artifacts[0], target)
    meta = published.parent / bundle.META_NAME
    content = json.loads(meta.read_text())
    content["hef_sha256"] = "0" * 64
    meta.write_text(json.dumps(content))
    assert backend._hailo_cache_bundle_status(target)["status"] == "metadata_invalid"
    assert backend._load_valid_hailo_receipt(target) is None


def test_cache_only_wrapper_never_registers_or_creates_store(tmp_path, artifacts):
    target = tmp_path / "output" / "compiled.hef"
    _publish(artifacts[0], target)
    result = {"ok": True, "hef_path": str(target)}
    backend._v60s_hailo_register(result, {"cache_only": True}, {})
    assert not (tmp_path / "store").exists()


def test_forced_compiler_replacement_keeps_old_generation_and_backups(tmp_path, artifacts, monkeypatch):
    import sys
    from types import SimpleNamespace

    target = tmp_path / "build" / "compiled.hef"
    previous = _publish(artifacts[0], target)
    receipt = artifacts[0][1]
    calls = []

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def translate_onnx_model(self, **kwargs):
            calls.append("translate")

        def get_hn_dict(self):
            return {"layers": {"images": {"type": "input_layer", "output_shape": [1, 224, 224, 3]}}}

        def load_model_script(self, script):
            pass

        def optimize(self, data):
            calls.append("optimize")

        def compile(self):
            calls.append("compile")
            return b"new-forced-build-hef"

    monkeypatch.setitem(sys.modules, "hailo_sdk_client", SimpleNamespace(ClientRunner=FakeRunner))
    monkeypatch.setattr(backend, "_infer_hailo_image_preprocess", lambda **kwargs: receipt["preprocessing_contract"]["image_scale"])
    monkeypatch.setattr(backend, "hailo_dfc_workspace_preflight", lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ROOT", str(tmp_path / "local-cache"))
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CACHE_ENABLED", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE", "memory")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB", "64")
    result = backend.hailo_build_hef(
        tmp_path / "model.onnx", outdir=target.parent,
        net_name=receipt["net_name"], hw_arch=receipt["hw_arch"],
        net_input_shapes=receipt["cache_payload"]["net_input_shapes"],
        task="classification", preprocessing_contract=receipt["preprocessing_contract"],
        force=True, fixup=False, opt_level=1, calib_count=1, calib_batch_size=1,
    )
    assert result.ok, result.error
    assert calls == ["translate", "optimize", "compile"]
    assert target.read_bytes() == b"new-forced-build-hef"
    assert previous.read_bytes() == artifacts[0][0].read_bytes()
    _assert_tuple(previous, receipt)
    new_receipt = backend._load_valid_hailo_receipt(target)
    _assert_tuple(target, new_receipt)
    _assert_tuple(tmp_path / "local-cache" / new_receipt["cache_key"] / "compiled.hef", new_receipt)
    records = ArtifactStore(tmp_path / "store").list(kind="hailo_hef")
    assert len(records) == 1
    assert records[0].metadata["bundle_status"] == "sealed"
    assert result.details["artifact_store_bundle_backup"] == "sealed"


def test_store_backup_failure_is_visible_without_losing_valid_output(tmp_path, artifacts, monkeypatch, caplog):
    target = tmp_path / "output" / "compiled.hef"
    _publish(artifacts[0], target)

    def fail_backup(self, **kwargs):
        raise OSError("simulated store unavailable")

    monkeypatch.setattr(ArtifactStore, "register_hailo_bundle", fail_backup)
    result = {"ok": True, "hef_path": str(target)}
    backend._v60s_hailo_register(result, {}, {})
    assert result["details"]["artifact_store_bundle_backup"] == "failed"
    assert "store unavailable" in result["details"]["artifact_store_backup_error"]
    assert "atomic bundle backup failed" in caplog.text
    _assert_tuple(target, artifacts[0][1])
