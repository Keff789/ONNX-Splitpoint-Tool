from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from onnx_splitpoint_tool import build_evidence_store as store_module
from onnx_splitpoint_tool.build_evidence import (
    ABORTED_UNKNOWN,
    COMPILE_INFEASIBLE,
    PARSER_UNSUPPORTED,
    TRANSIENT_INFRASTRUCTURE,
    BuildEvidenceError,
    build_evidence_index,
    canonical_build_key,
    load_build_evidence_index,
    make_build_evidence_record,
    write_build_evidence_index,
)
from onnx_splitpoint_tool.build_evidence_store import BuildEvidenceStore, STORE_ROOT_ENV


def _key(index: int = 1) -> dict:
    return canonical_build_key(
        full_source_onnx_sha256=f"{index:064x}",
        builder_source_onnx_sha256="2" * 64,
        compiler_onnx_sha256="3" * 64,
        boundary_endpoint_contract_sha256="4" * 64,
        backend="hailo_dfc",
        hw_arch="hailo8",
        compiler_version="hailo-dataflow-compiler:3.33.1",
        recipe={
            "optimization_level": 1,
            "model_script_sha256": "5" * 64,
            "start_nodes": [],
            "end_nodes": ["head"],
        },
        calibration={
            "identity": "manifest:" + "6" * 64,
            "effective_count": 5,
            "requested_count": 5,
            "batch_size": 1,
        },
        preprocessing_contract_sha256="7" * 64,
    )


def test_default_root_override_and_missing_lookup_are_read_only(tmp_path, monkeypatch):
    root = tmp_path / "never-created" / "evidence"
    monkeypatch.setenv(STORE_ROOT_ENV, str(root))
    store = BuildEvidenceStore()
    assert store.root == root
    assert store.lookup(_key()).status == "MISS"
    assert not root.parent.exists()


@pytest.mark.parametrize("state,reusable", [
    (PARSER_UNSUPPORTED, True), (COMPILE_INFEASIBLE, True),
    (TRANSIENT_INFRASTRUCTURE, False), (ABORTED_UNKNOWN, False),
])
def test_terminal_record_survives_new_run_without_original_files(tmp_path, state, reusable):
    root = tmp_path / "evidence"
    record = BuildEvidenceStore(root).record(
        _key(), state,
        evidence_origin={"kind": "terminal_attempt", "run": "deleted-old-run", "model": "yolo26s", "boundary": "b364"},
        reason_code="original_terminal_outcome",
    )
    assert record["reusable"] is reusable
    another_run = BuildEvidenceStore(root)
    decision = another_run.lookup(_key())
    assert decision.reusable is reusable
    assert decision.status == ("HIT" if reusable else "MISS")
    loaded = load_build_evidence_index(another_run.index_path)
    assert loaded["records"] == [record]


def test_historical_snapshot_discovered_immediately_and_never_modified(tmp_path):
    store = BuildEvidenceStore(tmp_path)
    assert store.lookup(_key()).status == "MISS"
    legacy = tmp_path / "v2783_yolo11_gate_a.json"
    record = make_build_evidence_record(
        _key(), COMPILE_INFEASIBLE, evidence_origin={"run": "already-deleted"},
    )
    write_build_evidence_index(legacy, build_evidence_index([record]))
    before = legacy.stat()
    legacy_bytes = legacy.read_bytes()
    assert store.lookup(_key()).state == COMPILE_INFEASIBLE
    store.record(_key(2), PARSER_UNSUPPORTED)
    after = legacy.stat()
    assert legacy.read_bytes() == legacy_bytes
    assert (after.st_mtime_ns, after.st_ino) == (before.st_mtime_ns, before.st_ino)
    assert store.lookup(_key()).reusable
    assert store.lookup(_key(2)).reusable


def test_exact_identity_changes_do_not_reuse_prior_failure(tmp_path):
    store = BuildEvidenceStore(tmp_path)
    store.record(_key(), COMPILE_INFEASIBLE)
    changed = copy.deepcopy(_key())
    changed["compiler_version"] = "hailo-dataflow-compiler:3.34.0"
    assert store.lookup(changed).status == "MISS"


def test_conflicting_legacy_and_live_negatives_are_visible_and_retained(tmp_path):
    legacy = tmp_path / "v2783_yolo11_gate_a.json"
    old_record = make_build_evidence_record(_key(), COMPILE_INFEASIBLE, evidence_origin={"run": "old"})
    write_build_evidence_index(legacy, build_evidence_index([old_record]))
    store = BuildEvidenceStore(tmp_path)
    store.record(_key(), PARSER_UNSUPPORTED)
    decision = store.lookup(_key())
    assert decision.status == "CONFLICT"
    assert not decision.reusable
    assert decision.reason == "conflicting_deterministic_negative_outcomes"
    assert load_build_evidence_index(legacy)["records"] == [old_record]
    assert load_build_evidence_index(store.index_path)["record_count"] == 1


def test_identical_records_are_idempotent_without_replacing_index(tmp_path):
    store = BuildEvidenceStore(tmp_path)
    first = store.record(_key(), COMPILE_INFEASIBLE)
    before = store.index_path.stat()
    second = store.record(_key(), COMPILE_INFEASIBLE)
    assert first == second
    assert store.index_path.stat().st_ino == before.st_ino
    assert load_build_evidence_index(store.index_path)["record_count"] == 1


def test_distinct_attempts_are_preserved(tmp_path):
    store = BuildEvidenceStore(tmp_path)
    store.record(_key(), TRANSIENT_INFRASTRUCTURE, evidence_origin={"attempt": "one"})
    store.record(_key(), COMPILE_INFEASIBLE, evidence_origin={"attempt": "two"})
    index = load_build_evidence_index(store.index_path)
    assert index["record_count"] == 2
    assert store.lookup(_key()).state == COMPILE_INFEASIBLE


def test_invalid_evidence_is_visible_but_unrelated_schema_is_ignored(tmp_path):
    store = BuildEvidenceStore(tmp_path)
    store.record(_key(), COMPILE_INFEASIBLE)
    unrelated = tmp_path / "recovery_result.json"
    unrelated.write_text(json.dumps({"schema": "recovery-result/v1", "ok": True}))
    assert store.lookup(_key()).reusable
    legacy = tmp_path / "v2783_yolo11_gate_a.json"
    payload = build_evidence_index()
    payload["record_count"] = 999
    legacy.write_text(json.dumps(payload))
    decision = store.lookup(_key())
    assert decision.status == "ERROR"
    assert decision.evidence_origin["index_errors"][0]["path"] == str(legacy)
    assert decision.evidence_origin["index_errors"][0]["reason"] == "noncanonical_build_index"


def test_corrupt_live_index_is_not_overwritten(tmp_path):
    store = BuildEvidenceStore(tmp_path)
    store.index_path.write_bytes(b'{"schema":')
    before = store.index_path.read_bytes()
    assert store.lookup(_key()).status == "ERROR"
    with pytest.raises(BuildEvidenceError, match="invalid_json"):
        store.record(_key(), COMPILE_INFEASIBLE)
    assert store.index_path.read_bytes() == before


def test_failed_atomic_publication_preserves_old_complete_snapshot(tmp_path, monkeypatch):
    store = BuildEvidenceStore(tmp_path)
    store.record(_key(), COMPILE_INFEASIBLE)
    old = store.index_path.read_bytes()

    def failed_replace(*args, **kwargs):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(store_module.os, "replace", failed_replace)
    with pytest.raises(OSError, match="injected replacement"):
        store.record(_key(2), PARSER_UNSUPPORTED)
    assert store.index_path.read_bytes() == old
    assert store.lookup(_key()).reusable
    assert store.lookup(_key(2)).status == "MISS"
    assert not list(tmp_path.glob("*.tmp"))


def test_index_symlink_is_rejected_without_touching_target(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("sensitive sentinel")
    root = tmp_path / "evidence"
    root.mkdir()
    store = BuildEvidenceStore(root)
    store.index_path.symlink_to(outside)
    assert store.lookup(_key()).status == "ERROR"
    with pytest.raises(OSError):
        store.record(_key(), COMPILE_INFEASIBLE)
    assert outside.read_text() == "sensitive sentinel"


def test_concurrent_threads_do_not_lose_terminal_records(tmp_path):
    barrier = threading.Barrier(12)

    def write(index):
        barrier.wait()
        BuildEvidenceStore(tmp_path).record(_key(index), COMPILE_INFEASIBLE)

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(write, range(1, 13)))
    store = BuildEvidenceStore(tmp_path)
    assert load_build_evidence_index(store.index_path)["record_count"] == 12
    assert all(store.lookup(_key(index)).reusable for index in range(1, 13))


def test_concurrent_processes_do_not_lose_terminal_records(tmp_path):
    source_root = str(Path(__file__).resolve().parents[1])
    code = (
        "import json,sys; "
        "sys.path.insert(0,sys.argv[1]); "
        "from onnx_splitpoint_tool.build_evidence_store import BuildEvidenceStore; "
        "BuildEvidenceStore(sys.argv[2]).record(json.loads(sys.argv[3]),'COMPILE_INFEASIBLE')"
    )

    def write(index):
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code, source_root, str(tmp_path), json.dumps(_key(index))],
            capture_output=True, text=True, timeout=30,
        )
        assert completed.returncode == 0, completed.stderr

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(1, 17)))
    store = BuildEvidenceStore(tmp_path)
    assert load_build_evidence_index(store.index_path)["record_count"] == 16
    assert all(store.lookup(_key(index)).reusable for index in range(1, 17))


def test_readers_see_complete_generations_during_atomic_publication(tmp_path):
    store = BuildEvidenceStore(tmp_path)
    store.record(_key(), COMPILE_INFEASIBLE)
    started = threading.Event()

    def write():
        started.set()
        for index in range(2, 25):
            store.record(_key(index), PARSER_UNSUPPORTED)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(write)
        started.wait()
        for _ in range(60):
            decision = BuildEvidenceStore(tmp_path).lookup(_key())
            assert decision.status == "HIT", decision.as_dict()
        future.result()
    assert load_build_evidence_index(store.index_path)["record_count"] == 24
