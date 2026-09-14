from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_cache_bundle as bundle
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _runner(run_dir: Path) -> EvaluationWorkflowRunner:
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = "terminal-hailo-fixture"
    runner.session_id = "4" * 32
    runner.artifact_index_path = run_dir / "artifact_index.json"
    runner.artifact_index = {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 1,
        "run_id": runner.run_id,
        "artifacts": [],
    }
    runner.outputs = {}
    runner.report_paths = []
    return runner


@pytest.fixture
def published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend, "_hailo_sdk_version_token", lambda: "hailo-dataflow-compiler:3.31.0")
    model = tmp_path / "source.onnx"
    model.write_bytes(b"offline-source-onnx")
    contract = canonical_image_preprocessing_contract("classification", (224, 224))
    key, payload = backend._hailo_cache_key(
        model_path=model, activation_part1=None, hw_arch="hailo8", opt_level=1,
        calib_dir=None, calib_count=1, calib_batch_size=1, extra_model_script="",
        start_nodes=None, end_nodes=None, preprocessing_contract=contract,
        effective_calib_count=1, calibration_storage="memory",
        calibration_memory_cap_bytes=64 * 1024 * 1024,
        net_name="mobilenet_v3_large_full", net_input_shapes={"images": [1, 3, 224, 224]},
        disable_rt_metadata_extraction=True,
    )
    source = tmp_path / "compiled.hef"
    source.write_bytes(b"offline-hef-content")
    receipt = backend._write_hailo_receipt(
        hef_path=source, source_onnx=model, compiler_onnx=model,
        hw_arch="hailo8", net_name="mobilenet_v3_large_full",
        preprocessing_contract=contract,
        preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=key, cache_payload=payload,
        calibration_identity=payload["calibration_identity"], calibration_count=1,
    )
    runner = _runner(tmp_path / "run")
    alias = runner.run_dir / "models/mobilenet_v3_large/benchmark_set/legacy_suite/hailo/hailo8/full/compiled.hef"
    generation_hef = backend._publish_hailo_bundle(source_hef=source, destination=alias, receipt=receipt)
    return runner, alias, generation_hef


def test_original_publisher_finalizes_and_verifies_canonical_generation(published):
    runner, alias, hef = published
    aliases = [alias, alias.parent / bundle.RECEIPT_NAME, alias.parent / bundle.META_NAME]
    runner._register_artifacts(aliases, kind="hailo_artifact", producer_stage="build")
    runner.outputs = {"hef": str(alias)}
    runner.report_paths = [alias.parent / bundle.RECEIPT_NAME]

    runner._finalize_artifact_index(status="ok")

    rows = runner.artifact_index["artifacts"]
    indexed = {row["path"] for row in rows}
    for name in (alias.name, bundle.RECEIPT_NAME, bundle.META_NAME):
        canonical = (hef.parent / name).relative_to(runner.run_dir).as_posix()
        assert canonical in indexed
        assert (alias.parent / name).relative_to(runner.run_dir).as_posix() not in indexed
    assert len(rows) == len(indexed)
    assert all(not (runner.run_dir / row["path"]).is_symlink() for row in rows)
    assert runner._verify_terminal_artifact_index(
        required_paths=runner._terminal_evidence_candidates() + aliases,
    )["ok"] is True


def test_old_mutable_alias_record_is_canonicalized_on_finalization(published):
    runner, alias, hef = published
    runner.artifact_index["artifacts"] = [{"path": alias.relative_to(runner.run_dir).as_posix()}]
    runner._finalize_artifact_index(status="ok")
    assert hef.relative_to(runner.run_dir).as_posix() in {
        row["path"] for row in runner.artifact_index["artifacts"]
    }


@pytest.mark.parametrize("damage", [
    "external_alias", "direct_generation_alias", "dotdot_alias", "broken_alias",
    "missing_alias", "absolute_pointer", "dotdot_pointer", "broken_pointer",
    "linked_generations_root", "linked_generation", "linked_member",
    "missing_receipt", "invalid_receipt", "invalid_meta", "mutated_hef",
])
def test_unapproved_or_incomplete_bundle_aliases_still_fail(published, tmp_path, damage):
    runner, alias, hef = published
    pointer = alias.parent / bundle.POINTER_NAME
    receipt = hef.parent / bundle.RECEIPT_NAME
    metadata = hef.parent / bundle.META_NAME
    if damage in {"external_alias", "direct_generation_alias", "dotdot_alias", "broken_alias", "missing_alias"}:
        alias.unlink()
        if damage == "external_alias":
            external = tmp_path / "external.hef"
            external.write_bytes(hef.read_bytes())
            alias.symlink_to(external)
        elif damage == "direct_generation_alias":
            alias.symlink_to(hef.relative_to(alias.parent))
        elif damage == "dotdot_alias":
            alias.symlink_to(f"{bundle.POINTER_NAME}/../{hef.parent.name}/{hef.name}")
        elif damage == "broken_alias":
            alias.symlink_to(f"{bundle.POINTER_NAME}/missing.hef")
    elif damage in {"absolute_pointer", "dotdot_pointer", "broken_pointer"}:
        pointer.unlink()
        if damage == "absolute_pointer":
            pointer.symlink_to(hef.parent)
        elif damage == "dotdot_pointer":
            pointer.symlink_to(f"{bundle.GENERATIONS_NAME}/../{bundle.GENERATIONS_NAME}/{hef.parent.name}")
        else:
            pointer.symlink_to(f"{bundle.GENERATIONS_NAME}/" + "f" * 32)
    elif damage in {"linked_generation", "linked_generations_root"}:
        target = hef.parent if damage == "linked_generation" else hef.parent.parent
        moved = tmp_path / "relocated-generation"
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=True)
    elif damage == "linked_member":
        external = tmp_path / "external.hef"
        external.write_bytes(hef.read_bytes())
        hef.unlink()
        hef.symlink_to(external)
    elif damage == "missing_receipt":
        receipt.unlink()
    elif damage == "invalid_receipt":
        receipt.write_text("{}")
    elif damage == "invalid_meta":
        metadata.write_text("{}")
    else:
        hef.write_bytes(b"changed-hef-bytes")

    with pytest.raises(RuntimeError, match="terminal_(hailo_bundle_alias_invalid|evidence_path_is_symlink)"):
        runner._finalize_artifact_index(status="ok")


def test_registration_rejects_redirected_hailo_alias_before_resolving(published, tmp_path):
    runner, alias, hef = published
    alias.unlink()
    alias.symlink_to(hef)
    with pytest.raises(RuntimeError, match="artifact_index_path_is_symlink"):
        runner._register_artifacts([alias], kind="hef", producer_stage="build")


@pytest.mark.parametrize("external", [False, True])
def test_unrelated_symlink_is_not_admitted(published, tmp_path, external):
    runner, alias, hef = published
    other = alias.parent / "arbitrary.json"
    target = tmp_path / "outside.json" if external else alias.parent / "real.json"
    target.write_text("{}")
    other.symlink_to(target)
    with pytest.raises(RuntimeError, match="terminal_evidence_path_is_symlink"):
        runner._finalize_artifact_index(status="ok")
    with pytest.raises(RuntimeError, match="artifact_index_path_is_symlink"):
        runner._register_artifacts([other], kind="report", producer_stage="fixture")


@pytest.mark.parametrize("name", ["compiled.hef", bundle.RECEIPT_NAME, bundle.META_NAME])
def test_mutated_generation_evidence_is_detected_after_closure(published, name):
    runner, alias, hef = published
    runner._finalize_artifact_index(status="ok")
    required = runner._terminal_evidence_candidates()
    (hef.parent / name).write_bytes(b"mutated")
    verification = runner._verify_terminal_artifact_index(required_paths=required)
    assert verification["ok"] is False
    assert "sha256_mismatch" in {row["error"] for row in verification["errors"]}


def test_legacy_regular_hef_evidence_is_preserved_without_relabeling(published):
    runner, alias, hef = published
    legacy = runner.run_dir / "models/old/compiled.hef"
    legacy.parent.mkdir()
    legacy.write_bytes(b"old-unsealed-hef")
    runner._finalize_artifact_index(status="ok")
    assert backend._hailo_cache_bundle_status(legacy)["status"] == "legacy_unsealed"
    assert legacy.relative_to(runner.run_dir).as_posix() in {
        row["path"] for row in runner.artifact_index["artifacts"]
    }


def test_shared_resolver_admits_only_publisher_aliases(published):
    runner, alias, hef = published
    assert bundle.resolve_published_bundle_member(alias, root=runner.run_dir) == hef
    assert bundle.resolve_published_bundle_member(
        alias.parent / bundle.POINTER_NAME, root=runner.run_dir,
    ) == hef.parent
    assert bundle.resolve_published_bundle_member(hef, root=runner.run_dir) is None
    assert bundle.resolve_published_bundle_member(
        alias, root=runner.run_dir / "reports",
    ) is None


def test_new_published_generation_is_not_covered_by_previous_terminal_index(published, tmp_path):
    import hashlib

    runner, alias, hef = published
    runner._finalize_artifact_index(status="ok")
    receipt = json.loads((hef.parent / bundle.RECEIPT_NAME).read_text())
    next_source = tmp_path / "next.hef"
    next_source.write_bytes(b"another-valid-compile")
    receipt["hef_sha256"] = hashlib.sha256(next_source.read_bytes()).hexdigest()
    receipt["hef_size_bytes"] = next_source.stat().st_size
    new_hef = backend._publish_hailo_bundle(
        source_hef=next_source, destination=alias, receipt=receipt,
    )
    assert new_hef != hef
    assert hef.is_file()
    verification = runner._verify_terminal_artifact_index(
        required_paths=runner._terminal_evidence_candidates(),
    )
    assert verification["ok"] is False
    assert "required_path_unindexed" in {row["error"] for row in verification["errors"]}


def test_registration_still_rejects_directory_alias_even_for_local_artifact(published):
    runner, alias, hef = published
    linked = runner.run_dir / "linked-directory"
    linked.symlink_to(alias.parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="artifact_index_path_is_symlink"):
        runner._register_artifacts(
            [linked / alias.name], kind="hef", producer_stage="fixture",
        )
