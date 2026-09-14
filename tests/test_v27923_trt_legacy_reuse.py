"""Execute the real embedded verifiers with local sealed fixture artifacts.

The trtexec stand-in only records deserialize requests; it cannot compile.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.remote_run import (
    _remote_trt_cache_probe_command,
    _remote_trt_cache_retention_command,
)
from tests.test_v27920_trt_namespace_retention import (
    _canonical_sha, _migration_receipt, _owner, _run_retention,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    base = tmp_path / "remote"
    managed = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
    old = managed / "model-old"
    current = managed / "model-current"
    _owner(old, last_used=1, builder_abi="a" * 64, stable_key=old.name)
    _owner(current, last_used=2, builder_abi="a" * 64, stable_key=current.name)
    calls = tmp_path / "deserialize.calls"
    builder = tmp_path / "trtexec"
    builder.write_text(
        '#!/bin/sh\ncase "$1" in --loadEngine=*) ;; *) exit 64 ;; esac\n'
        'test "$2" = --skipInference || exit 64\n'
        f"printf '%s\\n' \"$1\" >> '{calls}'\n",
    )
    builder.chmod(0o755)
    digest = _migration_receipt(old, builder, b"model-full")
    receipt = next(old.rglob("engine_build_receipt.json"))
    contract = {"complete": True, "allowed_shapes": [""]}
    requirement = {
        "role": "trt_full", "item_id": "setup/full", "source_sha256": digest,
        "shape_contract": contract,
        "expected_engine_filename": "full_fp16.engine",
        "expected_relative_leaf": f"full/{digest}/fp16",
    }
    payload = {
        "remote_base": str(base), "stable_key": current.name,
        "legacy_suite_key": old.name, "canonical_full_sha256": [digest],
        "builder_abi_sha256": "a" * 64, "trtexec_sha256": _sha(builder),
        "precision": "fp16", "workspace_mb": 4096,
        "requirements": [requirement], "model_id": "model", "setup_id": "setup",
    }
    kwargs = {
        "current_key": current.name, "stable_engine_key": current.name,
        "legacy_suite_key": old.name, "builder_abi_sha256": "a" * 64,
        "current_trtexec_sha256": _sha(builder), "native_trt_precision": "fp16",
        "native_trt_workspace_mb": 4096,
        "canonical_full_onnx_sha256": [digest],
        "source_shape_contracts": {digest: contract},
    }
    return base, old, current, receipt, builder, calls, payload, kwargs


def _probe(payload):
    completed = subprocess.run(
        ["bash", "-c", _remote_trt_cache_probe_command(payload)],
        text=True, capture_output=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    marker = "SPLITPOINT_REMOTE_TRT_CACHE_PREFLIGHT="
    return json.loads(next(
        line[len(marker):] for line in completed.stdout.splitlines()
        if line.startswith(marker)
    ))["observations"][0]


def _reseal(receipt, mutate):
    raw = json.loads(receipt.read_text())
    raw.pop("receipt_sha256")
    mutate(raw)
    raw["receipt_sha256"] = _canonical_sha(raw)
    receipt.write_text(json.dumps(raw))


def test_changed_owner_key_reuses_exact_receipt_and_preserves_source(tmp_path):
    base, old, current, receipt, _, calls, payload, kwargs = _fixture(tmp_path)
    original = {p: p.read_bytes() for p in old.rglob("*") if p.is_file()}
    hit = _probe(payload)
    assert hit["status"] == "HIT"
    assert hit["evidence"]["namespace_owner_status"] == "legacy_owner_engine_key_changed"
    assert hit["evidence"]["previous_engine_cache_key"] == old.name
    assert hit["evidence"]["current_engine_cache_key"] == current.name
    assert not calls.exists()  # Read-only preflight runs no executable.
    result = _run_retention(base, **kwargs)
    migration = result["legacy_receipt_migration"]
    assert migration["migrated_receipts"] == 1
    assert migration["owner_key_changes"][0]["decision"] == "verify_individual_receipts"
    for path, contents in original.items():
        assert path.read_bytes() == contents
    assert len(calls.read_text().splitlines()) == 1
    migrated = _probe(payload)
    assert migrated["status"] == "HIT"
    assert migrated["source_namespace"] == "current"
    assert migrated["evidence"]["engine_sha256"] == hit["evidence"]["engine_sha256"]
    assert _run_retention(base, **kwargs)["legacy_receipt_migration"]["migrated_receipts"] == 0
    assert len(calls.read_text().splitlines()) == 1


@pytest.mark.parametrize("failure,probe_reason", [
    ("abi", "legacy_owner_builder_abi_mismatch"),
    ("engine", "engine_sha256_mismatch"),
    ("source", "source_onnx_mismatch"),
    ("workspace", "receipt_workspace_mismatch"),
    ("shape", "receipt_shape_mismatch"),
    ("shape_unavailable", "shape_contract_unavailable"),
    ("precision", "build_contract_mismatch"),
    ("unknown_flag", "receipt_unknown_build_args"),
    ("receipt_digest", "receipt_integrity_mismatch"),
])
def test_owner_key_change_cannot_bypass_individual_gates(tmp_path, failure, probe_reason):
    base, old, _, receipt, _, calls, payload, kwargs = _fixture(tmp_path)
    if failure == "abi":
        _owner(old, last_used=1, builder_abi="b" * 64, stable_key=old.name)
    elif failure == "engine":
        (receipt.parent / "full_fp16.engine").write_bytes(b"corrupted")
    elif failure == "source":
        payload["requirements"][0]["source_sha256"] = "b" * 64
        payload["canonical_full_sha256"] = ["b" * 64]
        kwargs["canonical_full_onnx_sha256"] = ["b" * 64]
        kwargs["source_shape_contracts"] = {"b" * 64: {"complete": True, "allowed_shapes": [""]}}
    elif failure == "shape_unavailable":
        payload["requirements"][0]["shape_contract"]["complete"] = False
    elif failure == "receipt_digest":
        raw = json.loads(receipt.read_text())
        raw["dry_run"] = True
        receipt.write_text(json.dumps(raw))
    else:
        flag = {"workspace": "--workspace=64", "shape": "--shapes=input:2x3x4x4",
                "precision": "--int8", "unknown_flag": "--useDLACore=0"}[failure]
        _reseal(receipt, lambda raw: raw["command"].append(flag))
    hit = _probe(payload)
    assert hit["status"] != "HIT"
    assert hit["reason"] == probe_reason
    result = _run_retention(base, **kwargs)
    assert result["legacy_receipt_migration"]["migrated_receipts"] == 0
    assert not calls.exists()


def test_duplicate_selection_is_deterministic_and_skips_invalid_first(tmp_path):
    base, old, current, first_receipt, builder, calls, payload, kwargs = _fixture(tmp_path)
    # Exact prior suite wins over lexical order, but only if it is valid.
    second = old.parent / "model-aaa"
    _owner(second, last_used=1, builder_abi="a" * 64, stable_key=second.name)
    _migration_receipt(second, builder, b"model-full")
    second_receipt = next(second.rglob("engine_build_receipt.json"))
    assert _probe(payload)["receipt_path"] == str(first_receipt)
    (first_receipt.parent / "full_fp16.engine").write_bytes(b"bad-first")
    assert _probe(payload)["receipt_path"] == str(second_receipt)
    result = _run_retention(base, **kwargs)
    assert result["legacy_receipt_migration"]["migrated_receipts"] == 1
    assert "model-aaa" in calls.read_text()
    assert _probe(payload)["source_namespace"] == "current"
    assert old.is_dir() and second.is_dir()


def test_migration_does_not_evict_original_to_meet_namespace_limit(tmp_path):
    base, old, _, receipt, _, _, _, kwargs = _fixture(tmp_path)
    before = receipt.read_bytes()
    command = _remote_trt_cache_retention_command(
        remote_base=str(base), max_namespaces=1, max_bytes=1024**3, **kwargs,
    )
    result = subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert '"reason":"legacy_migration_source"' in result.stdout
    assert old.is_dir() and receipt.read_bytes() == before


def test_partial_target_is_reported_without_overwriting_either_generation(tmp_path):
    base, old, current, receipt, _, calls, payload, kwargs = _fixture(tmp_path)
    target = current / payload["requirements"][0]["expected_relative_leaf"]
    target.mkdir(parents=True)
    partial = target / "full_fp16.engine"
    partial.write_bytes(b"unsealed-keep-for-recovery")
    original_receipt = receipt.read_bytes()
    hit = _probe(payload)
    assert hit["status"] == "UNKNOWN"
    assert hit["reason"] == "current_cache_leaf_blocks_migration"
    migration = _run_retention(base, **kwargs)["legacy_receipt_migration"]
    assert migration["migrated_receipts"] == 0
    assert migration["partial_target_leaves"] == 1
    assert partial.read_bytes() == b"unsealed-keep-for-recovery"
    assert receipt.read_bytes() == original_receipt
    assert not calls.exists()


def test_uint8_bridge_migration_keeps_original_metadata_bytes(tmp_path):
    from tests.test_v27920_remote_trt_cache_preflight import _receipt

    base, old, current, _, builder, calls, payload, kwargs = _fixture(tmp_path)
    leaf = old / "b024/part2/uint8_cast_fp16"
    _receipt(
        leaf=leaf, source_bytes=b"cast-bridge", builder=builder,
        role="part2", engine_precision="uint8_cast_fp16",
    )
    original = leaf / "original.onnx"
    original.write_bytes(b"original-part2")
    bridge = leaf / "source.onnx"
    digest = _sha(original)
    metadata = leaf / "uint8_cast_bridge_meta.json"
    metadata.write_text(json.dumps({
        "schema": "onnx-splitpoint/uint8-cast-bridge", "schema_version": 1,
        "source": str(original), "source_sha256": digest,
        "source_size_bytes": original.stat().st_size,
        "bridge": str(bridge), "bridge_sha256": _sha(bridge),
        "bridge_size_bytes": bridge.stat().st_size,
        "input_dtype": "UINT8", "cast_to": "FLOAT", "replaced_uses": 1,
        "input_name": "input", "cast_output": "cast_output",
    }))
    original_metadata = metadata.read_bytes()
    requirement = payload["requirements"][0]
    requirement.update({
        "role": "trt_p2", "case_id": "b024", "source_sha256": digest,
        "source_size_bytes": original.stat().st_size,
        "expected_engine_filename": "part2_uint8_cast_fp16.engine",
        "expected_relative_leaf": f"splits/b024/part2/{digest}/uint8_cast_fp16",
    })
    payload["precision"] = "uint8_cast_fp16"
    kwargs["native_trt_precision"] = "uint8_cast_fp16"
    kwargs["source_shape_contracts"] = {digest: requirement["shape_contract"]}
    assert _probe(payload)["status"] == "HIT"
    migration = _run_retention(base, **kwargs)["legacy_receipt_migration"]
    assert migration["migrated_receipts"] == 1
    copied_meta = current / requirement["expected_relative_leaf"] / metadata.name
    assert metadata.read_bytes() == original_metadata
    assert str(current) in json.loads(copied_meta.read_text())["bridge"]
    assert copied_meta.stat().st_ino != metadata.stat().st_ino
    assert len(calls.read_text().splitlines()) == 1
    assert _probe(payload)["status"] == "HIT"
