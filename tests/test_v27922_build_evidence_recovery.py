from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from onnx_splitpoint_tool import build_evidence as evidence
from onnx_splitpoint_tool.hailo_attempt_receipts import begin_hailo_attempt, finalize_hailo_attempt
from onnx_splitpoint_tool.hailo_cache_bundle import publish_bundle
from test_v2783_build_evidence import _create_positive_b5_fixture, _write_json


def _meta(receipt):
    return {
        "schema": "onnx-splitpoint/hailo-hef-cache-meta-v2",
        "cache_key": receipt["cache_key"], "payload": receipt["cache_payload"],
        "hef_size": receipt["hef_size_bytes"], "hef_sha256": receipt["hef_sha256"],
        "preprocessing_contract_sha256": receipt["preprocessing_contract_sha256"],
        "net_name": receipt["net_name"], "hw_arch": receipt["hw_arch"],
        "created_at": 1.0, "source": "offline-test",
    }


def _publish(fixture, *, data=None):
    receipt = json.loads(fixture["receipt"].read_text())
    source = fixture["hef"]
    if data is not None:
        source = fixture["run"].parent / "new.hef"
        source.write_bytes(data)
        receipt.update(hef_sha256=hashlib.sha256(data).hexdigest(), hef_size_bytes=len(data))
    return publish_bundle(source_hef=source, destination=fixture["hef"], receipt=receipt,
                          cache_meta=_meta(receipt), validator=lambda staged: (
                              hashlib.sha256(staged.read_bytes()).hexdigest() == receipt["hef_sha256"]))


def test_real_publication_harvest_lookup_and_atomic_materialization(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    _publish(fixture)
    current = _publish(fixture, data=b"second complete generation")
    before = {str(path): os.lstat(path).st_mtime_ns for path in fixture["run"].rglob("*")}
    index = evidence.harvest_b5_run(fixture["run"])
    after = {str(path): os.lstat(path).st_mtime_ns for path in fixture["run"].rglob("*")}
    assert before == after
    assert index["state_counts"][evidence.ARTIFACT_PASS] == 1
    assert index["unresolved_observations"] == []
    record = index["records"][0]
    assert record["artifact"]["relative_path"] == current.relative_to(fixture["run"]).as_posix()
    assert record["artifact"]["cache_meta_sha256"]
    decision = evidence.lookup_build_evidence(index, fixture["key"], artifact_root=fixture["run"])
    assert decision.status == "HIT"
    destination = tmp_path / "recovered"
    result = evidence.materialize_verified_artifact(decision, destination, fixture["key"],
                                                  artifact_root=fixture["run"])
    assert result["ok"]
    members = [destination / name for name in evidence._HAILO_BUNDLE_NAMES]
    assert all(path.is_symlink() for path in members)
    assert len({path.resolve().parent for path in members}) == 1
    assert evidence.verify_hailo_artifact(members[0]).cache_meta_path is not None


def test_harvest_pins_one_generation_if_pointer_changes_after_discovery(tmp_path, monkeypatch):
    fixture = _create_positive_b5_fixture(tmp_path)
    first = _publish(fixture)
    real_snapshot = evidence._snapshot_published_hailo_bundle
    def replace_after_snapshot(directory):
        snapshot = real_snapshot(directory)
        _publish(fixture, data=b"concurrent newer generation")
        return snapshot
    monkeypatch.setattr(evidence, "_snapshot_published_hailo_bundle", replace_after_snapshot)
    index = evidence.harvest_b5_run(fixture["run"])
    assert index["state_counts"][evidence.ARTIFACT_PASS] == 1
    assert index["unresolved_observations"] == []
    assert index["records"][0]["artifact"]["relative_path"] == first.relative_to(fixture["run"]).as_posix()
    assert fixture["hef"].read_bytes() == b"concurrent newer generation"
    assert evidence.lookup_build_evidence(index, fixture["key"], artifact_root=fixture["run"]).status == "HIT"


@pytest.mark.parametrize("kind", ["arbitrary_alias", "external_pointer", "generation_link", "meta_link"])
def test_bundle_support_does_not_follow_arbitrary_or_external_symlinks(tmp_path, kind):
    fixture = _create_positive_b5_fixture(tmp_path)
    generation = _publish(fixture).parent
    directory = fixture["hef"].parent
    if kind == "arbitrary_alias":
        fixture["receipt"].unlink()
        fixture["receipt"].symlink_to(generation / "hailo_hef_build_receipt.json")
    elif kind == "external_pointer":
        pointer = directory / ".hailo-current"
        pointer.unlink()
        pointer.symlink_to(generation.absolute(), target_is_directory=True)
    elif kind == "generation_link":
        moved = tmp_path / "outside_generation"
        generation.rename(moved)
        generation.symlink_to(moved, target_is_directory=True)
    else:
        meta = generation / "cache_meta.json"
        moved = tmp_path / "outside_meta.json"
        meta.rename(moved)
        meta.symlink_to(moved)
    with pytest.raises(evidence.BuildEvidenceError, match="unsafe_.*symlink"):
        evidence.harvest_b5_run(fixture["run"])


def test_cache_meta_corruption_invalidates_positive_instead_of_partial_recovery(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    generation = _publish(fixture).parent
    meta = json.loads((generation / "cache_meta.json").read_text())
    meta["hef_sha256"] = "0" * 64
    _write_json(generation / "cache_meta.json", meta)
    index = evidence.harvest_b5_run(fixture["run"])
    assert index["state_counts"][evidence.ARTIFACT_PASS] == 0
    assert any("hailo_cache_meta_mismatch" in row["reason_code"] for row in index["unresolved_observations"])


@pytest.mark.parametrize("error, expected", [
    ("CUDA memory allocation failed: out of memory", evidence.TRANSIENT_INFRASTRUCTURE),
    ("Mapping Failed: CUDA_ERROR_OUT_OF_MEMORY", evidence.TRANSIENT_INFRASTRUCTURE),
    ("Mapping failed: ResourceExhaustedError: OOM when allocating tensor", evidence.TRANSIENT_INFRASTRUCTURE),
    ("Allocation failed", evidence.TRANSIENT_INFRASTRUCTURE),
    ("Allocator agent infeasible: concat22 / Agent infeasible", evidence.COMPILE_INFEASIBLE),
    ("Mapping Failed: No successful assignments", evidence.COMPILE_INFEASIBLE),
    ("UnsupportedShuffleLayerError: unsupported operation", evidence.PARSER_UNSUPPORTED),
    ("Unsupported operation; no space left on device", evidence.TRANSIENT_INFRASTRUCTURE),
    ("Mapping failed; cancelled by user", evidence.ABORTED_UNKNOWN),
])
def test_resource_and_abort_failures_never_become_deterministic_negatives(error, expected):
    assert evidence.classify_build_outcome({"ok": False, "error": error}) == expected


def _modern_attempt(fixture, *, include_key, error="Mapping Failed: Agent infeasible"):
    directory = fixture["hef"].parent
    attempt = begin_hailo_attempt(outdir=directory, bound={
        "onnx_path": str(directory.parent.parent.parent / "part1.onnx"),
        "compiler_onnx_path": str(fixture["compiler"]), "hw_arch": "hailo8",
    })
    details = {"build_evidence": {
        "key": fixture["key"], "cache_key_v3": evidence.canonical_sha256(fixture["payload"]),
        "cache_payload_v3": fixture["payload"],
    }} if include_key else {}
    return finalize_hailo_attempt(attempt=attempt, result={
        "ok": False, "error": error, "fixed_onnx_path": str(fixture["compiler"]), "details": details,
    })


def test_modern_terminal_attempts_preserve_exact_negatives_and_deduplicate_pointer(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    immutable = _modern_attempt(fixture, include_key=True)
    fixture["hef"].unlink()
    fixture["receipt"].unlink()
    index = evidence.harvest_b5_run(fixture["run"])
    assert index["state_counts"][evidence.COMPILE_INFEASIBLE] == 1
    assert index["unresolved_observations"] == []
    assert index["records"][0]["evidence_origin"]["attempt"].endswith(immutable.name)
    assert evidence.lookup_build_evidence(index, fixture["key"]).status == "HIT"


def test_recovered_terminal_pointer_without_original_or_onnx_remains_usable(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    immutable = _modern_attempt(fixture, include_key=True)
    for path in fixture["hef"].parent.rglob("*"):
        if path.is_file() and path.name != "terminal_attempt.json":
            path.unlink()
    assert not immutable.exists()
    index = evidence.harvest_b5_run(fixture["run"])
    assert index["state_counts"][evidence.COMPILE_INFEASIBLE] == 1
    assert evidence.lookup_build_evidence(index, fixture["key"]).status == "HIT"


def test_old_terminal_receipt_without_exact_identity_is_visible_unresolved(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    _modern_attempt(fixture, include_key=False)
    fixture["hef"].unlink()
    fixture["receipt"].unlink()
    index = evidence.harvest_b5_run(fixture["run"])
    assert index["records"] == []
    assert len(index["unresolved_observations"]) == 1
    row = index["unresolved_observations"][0]
    assert row["state"] == evidence.COMPILE_INFEASIBLE
    assert row["reusable"] is False
    assert row["reason_code"] == "exact_key_unavailable:terminal_attempt_exact_identity_missing"


def test_modern_oom_attempt_is_recorded_but_never_reused(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    _modern_attempt(fixture, include_key=True, error="CUDA memory allocation failed: out of memory")
    fixture["hef"].unlink()
    fixture["receipt"].unlink()
    index = evidence.harvest_b5_run(fixture["run"])
    assert index["state_counts"][evidence.TRANSIENT_INFRASTRUCTURE] == 1
    assert not evidence.lookup_build_evidence(index, fixture["key"]).reusable


@pytest.mark.parametrize("result, terminal", [
    ({"ok": True, "returncode": "143"}, True),
    ({"ok": False, "error": "Mapping failed", "status": "cancelled"}, True),
    ({"ok": True}, False),
])
def test_cancellation_and_nonterminal_take_precedence_over_success(result, terminal):
    assert evidence.classify_build_outcome(result, terminal=terminal) == evidence.ABORTED_UNKNOWN


def test_modern_terminal_receipt_uses_matching_existing_context(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    _modern_attempt(fixture, include_key=False)
    context = {"schema": evidence.BUILD_CONTEXT_SCHEMA, "schema_version": 1, "key": fixture["key"]}
    context["context_sha256"] = evidence.canonical_sha256(context)
    _write_json(fixture["hef"].parent / "build_evidence_context.json", context)
    fixture["hef"].unlink()
    fixture["receipt"].unlink()
    index = evidence.harvest_b5_run(fixture["run"])
    assert len(index["records"]) == 1
    assert evidence.lookup_build_evidence(index, fixture["key"]).status == "HIT"


def test_terminal_receipt_rejects_disagreeing_compiler_digest(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    path = _modern_attempt(fixture, include_key=True)
    value = json.loads(path.read_text())
    value["compiler_onnx_sha256"] = "a" * 64
    _write_json(path, value)
    (path.parent / "terminal_attempt.json").unlink()
    fixture["hef"].unlink()
    fixture["receipt"].unlink()
    index = evidence.harvest_b5_run(fixture["run"])
    assert index["records"] == []
    assert index["unresolved_observations"][0]["reason_code"] == "exact_key_unavailable:compiler_onnx_sha256_mismatch"


def test_known_negative_lookup_receipt_is_not_learned_as_new_compiler_failure(tmp_path):
    fixture = _create_positive_b5_fixture(tmp_path)
    path = _modern_attempt(fixture, include_key=True)
    value = json.loads(path.read_text())
    value["result_summary"].update(skipped=True, failure_kind="known_negative_build_evidence")
    _write_json(path, value)
    (path.parent / "terminal_attempt.json").unlink()
    fixture["hef"].unlink()
    fixture["receipt"].unlink()
    index = evidence.harvest_b5_run(fixture["run"])
    assert index["records"] == []
    assert index["unresolved_observations"][0]["reason_code"] == "lookup_or_probe_receipt_not_compiler_evidence"
