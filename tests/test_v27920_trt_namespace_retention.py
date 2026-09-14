from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from onnx_splitpoint_tool.benchmark.remote_run import (
    _remote_trt_cache_retention_command,
    _stable_trt_engine_cache_key,
    _trt_persistent_cache_layout,
    _trt_persistent_engine_relative_dir,
)


def _canonical_sha(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _owner(
    namespace: Path,
    *,
    last_used: float,
    builder_abi: str = "",
    stable_key: str = "",
) -> None:
    namespace.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "onnx-splitpoint/managed-trt-cache-owner",
        "schema_version": 1,
        "owner": "onnx-splitpoint-tool",
        "cache_key": namespace.name,
        "created_at_unix": last_used,
        "last_used_at_unix": last_used,
    }
    if builder_abi:
        payload["trt_builder_abi_sha256"] = builder_abi
    if stable_key:
        payload["trt_engine_cache_key"] = stable_key
    (namespace / ".splitpoint_trt_cache_owner.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _simple_retention_receipt(namespace: Path, payload: bytes) -> None:
    leaf = namespace / "test-leaf"
    leaf.mkdir(parents=True, exist_ok=True)
    engine = leaf / "engine.plan"
    engine.write_bytes(payload)
    receipt = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "engine": str(engine.resolve()),
        "engine_sha256": hashlib.sha256(payload).hexdigest(),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    (leaf / "engine_build_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8",
    )


def _migration_receipt(namespace: Path, trtexec: Path, source_payload: bytes) -> str:
    source_sha = hashlib.sha256(source_payload).hexdigest()
    # Deliberately use the pre-v2.79.20, case-scoped Full layout.
    leaf = (
        namespace / "b132" / "native" / source_sha[:2] / source_sha
        / "full" / "fp16"
    )
    leaf.mkdir(parents=True, exist_ok=True)
    source = leaf / "source.onnx"
    engine = leaf / "full_fp16.engine"
    source.write_bytes(source_payload)
    engine.write_bytes(b"serialized-engine")
    receipt = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "command": [
            str(trtexec.resolve()),
            f"--onnx={source.resolve()}",
            f"--saveEngine={engine.resolve()}",
            "--fp16",
            "--memPoolSize=workspace:4096",
        ],
        "source_onnx": str(source.resolve()),
        "source_onnx_sha256": source_sha,
        "engine": str(engine.resolve()),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "trtexec": str(trtexec.resolve()),
        "trtexec_sha256": hashlib.sha256(trtexec.read_bytes()).hexdigest(),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    (leaf / "engine_build_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8",
    )
    return source_sha


def _run_retention(base: Path, **kwargs) -> dict:
    command = _remote_trt_cache_retention_command(
        remote_base=str(base),
        max_namespaces=int(kwargs.pop("max_namespaces", 6)),
        max_bytes=int(kwargs.pop("max_bytes", 1024**3)),
        **kwargs,
    )
    completed = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    marker = "SPLITPOINT_TRT_RETENTION_JSON="
    rows = [
        json.loads(line[len(marker):])
        for line in completed.stdout.splitlines() if line.startswith(marker)
    ]
    assert len(rows) == 1, completed.stdout
    return rows[0]


def _suite(root: Path, case_id: str) -> Path:
    suite = root / case_id / "benchmark_set"
    (suite / "models").mkdir(parents=True)
    (suite / case_id).mkdir()
    (suite / "models" / "regnet.onnx").write_bytes(b"same-regnet-full")
    (suite / case_id / "source_part2.onnx").write_bytes(
        ("split-" + case_id).encode("ascii")
    )
    (suite / "benchmark_plan.json").write_text(json.dumps({
        "model_suite": {"primary": [{"id": "regnet_x_1_6gf"}]},
        "runs": [{
            "id": "ort_tensorrt", "provider": "tensorrt",
            "native_trt_precision": "fp16",
            "native_trt_workspace_mb": 4096,
        }],
    }), encoding="utf-8")
    (suite / "benchmark_set.json").write_text(json.dumps({
        "model": "models/regnet.onnx",
        "model_name": "regnet_x_1_6gf",
        "cases": [{"case_id": case_id}],
    }), encoding="utf-8")
    return suite


def test_full_namespace_and_leaf_are_split_independent(tmp_path: Path) -> None:
    suite_b132 = _suite(tmp_path, "b132")
    suite_b045 = _suite(tmp_path, "b045")
    builder = {
        "trtexec_sha256": "a" * 64,
        "trtexec_size_bytes": 123,
        "selected_gpu_target": {
            "visible_device_index": "0", "name": "Orin NX",
            "compute_capability": "8.7", "driver_version": "x",
        },
        "linked_runtime_libraries": [],
    }
    assert _stable_trt_engine_cache_key(
        suite_b132, builder_abi=builder,
    ) == _stable_trt_engine_cache_key(suite_b045, builder_abi=builder)

    full_sha = hashlib.sha256(b"same-regnet-full").hexdigest()
    assert _trt_persistent_engine_relative_dir(
        role="full", source_onnx_sha256=full_sha, precision="fp16",
        case_id="b132",
    ) == Path("full") / full_sha / "fp16"
    assert _trt_persistent_engine_relative_dir(
        role="full", source_onnx_sha256=full_sha, precision="fp16",
        case_id="b045",
    ) == Path("full") / full_sha / "fp16"
    assert _trt_persistent_engine_relative_dir(
        role="part2", source_onnx_sha256="b" * 64,
        precision="fp16", case_id="b132",
    ) != _trt_persistent_engine_relative_dir(
        role="part2", source_onnx_sha256="b" * 64,
        precision="fp16", case_id="b045",
    )
    layout = _trt_persistent_cache_layout(
        namespace_root="/cache/regnet-key",
        canonical_full_onnx=[{"sha256": full_sha}],
    )
    assert layout["full_root"] == "/cache/regnet-key/full"
    assert layout["splits_root"] == "/cache/regnet-key/splits"


def test_partial_stable_namespace_imports_only_missing_legacy_leaf(
    tmp_path: Path,
) -> None:
    base = tmp_path / "splitpoint_runs"
    managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
    legacy_key = "regnet_x_1_6gf-legacy"
    current_key = "regnet_x_1_6gf-current"
    legacy = managed / legacy_key
    current = managed / current_key
    _owner(legacy, last_used=1.0, builder_abi="c" * 64)
    _owner(current, last_used=2.0)
    (current / "already-present-unrelated.txt").write_text(
        "partial namespace", encoding="utf-8",
    )
    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    trtexec.chmod(0o755)
    source_sha = _migration_receipt(legacy, trtexec, b"regnet-full")
    common = {
        "current_key": current_key,
        "legacy_suite_key": legacy_key,
        "stable_engine_key": current_key,
        "builder_abi_sha256": "c" * 64,
        "current_trtexec_sha256": hashlib.sha256(trtexec.read_bytes()).hexdigest(),
        "native_trt_precision": "fp16",
        "native_trt_workspace_mb": 4096,
        "canonical_full_onnx_sha256": (source_sha,),
    }

    first = _run_retention(base, **common)
    migration = first["legacy_receipt_migration"]
    assert migration["target_preexisting"] is True
    assert migration["trigger"] == "partial_namespace_missing_artifact_scan"
    assert migration["migrated_receipts"] == 1
    canonical_leaf = current / "full" / source_sha / "fp16"
    assert (canonical_leaf / "full_fp16.engine").read_bytes() == b"serialized-engine"
    assert not (current / "b132").exists()

    second = _run_retention(base, **common)
    second_migration = second["legacy_receipt_migration"]
    assert second_migration["migrated_receipts"] == 0
    assert second_migration["existing_receipts"] == 1


def test_migration_rejects_wrong_dynamic_shape_contract(tmp_path: Path) -> None:
    base = tmp_path / "splitpoint_runs"
    managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
    legacy_key = "dynamic-legacy"
    current_key = "dynamic-current"
    legacy = managed / legacy_key
    current = managed / current_key
    _owner(legacy, last_used=1.0, builder_abi="a" * 64)
    _owner(current, last_used=2.0)
    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    trtexec.chmod(0o755)
    source_sha = _migration_receipt(legacy, trtexec, b"dynamic-model")
    result = _run_retention(
        base,
        current_key=current_key,
        legacy_suite_key=legacy_key,
        stable_engine_key=current_key,
        builder_abi_sha256="a" * 64,
        current_trtexec_sha256=hashlib.sha256(
            trtexec.read_bytes()
        ).hexdigest(),
        native_trt_precision="fp16",
        native_trt_workspace_mb=4096,
        canonical_full_onnx_sha256=(source_sha,),
        source_shape_contracts={
            source_sha: {
                "complete": True,
                "allowed_shapes": ["input:1x3x640x640"],
            },
        },
    )
    migration = result["legacy_receipt_migration"]
    assert migration["migrated_receipts"] == 0
    assert migration["rejected_receipts"] >= 1


def test_uint8_cast_migration_uses_original_part2_digest(tmp_path: Path) -> None:
    base = tmp_path / "splitpoint_runs"
    managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
    legacy_key = "cast-legacy"
    current_key = "cast-current"
    legacy = managed / legacy_key
    current = managed / current_key
    _owner(legacy, last_used=1.0, builder_abi="b" * 64)
    _owner(current, last_used=2.0)
    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    trtexec.chmod(0o755)
    leaf = legacy / "b024/part2/uint8_cast_fp16"
    leaf.mkdir(parents=True)
    original = leaf / "original_part2.onnx"
    bridge = leaf / "source.onnx"
    engine = leaf / "part2_uint8_cast_fp16.engine"
    original.write_bytes(b"original-part2")
    bridge.write_bytes(b"generated-cast-bridge")
    engine.write_bytes(b"serialized-engine")
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    bridge_sha = hashlib.sha256(bridge.read_bytes()).hexdigest()
    (leaf / "uint8_cast_bridge_meta.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/uint8-cast-bridge",
        "schema_version": 1,
        "source": str(original.resolve()),
        "source_sha256": original_sha,
        "source_size_bytes": original.stat().st_size,
        "bridge": str(bridge.resolve()),
        "bridge_sha256": bridge_sha,
        "bridge_size_bytes": bridge.stat().st_size,
        "input_name": "boundary",
        "input_dtype": "UINT8",
        "cast_output": "boundary__uint8_cast_to_float",
        "cast_to": "FLOAT",
        "replaced_uses": 1,
        "precision_tag": "uint8_cast_fp16",
    }))
    receipt = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "command": [
            str(trtexec.resolve()), f"--onnx={bridge.resolve()}",
            f"--saveEngine={engine.resolve()}", "--fp16",
            "--memPoolSize=workspace:4096",
        ],
        "source_onnx": str(bridge.resolve()),
        "source_onnx_sha256": bridge_sha,
        "engine": str(engine.resolve()),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "trtexec": str(trtexec.resolve()),
        "trtexec_sha256": hashlib.sha256(trtexec.read_bytes()).hexdigest(),
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    (leaf / "engine_build_receipt.json").write_text(json.dumps(receipt))

    result = _run_retention(
        base,
        current_key=current_key,
        legacy_suite_key=legacy_key,
        stable_engine_key=current_key,
        builder_abi_sha256="b" * 64,
        current_trtexec_sha256=hashlib.sha256(
            trtexec.read_bytes()
        ).hexdigest(),
        native_trt_precision="uint8_cast_fp16",
        native_trt_workspace_mb=4096,
        source_shape_contracts={
            original_sha: {"complete": True, "allowed_shapes": [""]},
        },
    )
    migration = result["legacy_receipt_migration"]
    assert migration["migrated_receipts"] == 1
    destination = (
        current / "splits/b024/part2" / original_sha / "uint8_cast_fp16"
    )
    assert (destination / "part2_uint8_cast_fp16.engine").is_file()
    migrated_meta = json.loads(
        (destination / "uint8_cast_bridge_meta.json").read_text()
    )
    assert Path(migrated_meta["bridge"]).parent == destination


def test_existing_v27919_case_scoped_full_leaf_is_adopted_in_place(
    tmp_path: Path,
) -> None:
    base = tmp_path / "splitpoint_runs"
    managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
    current_key = "regnet_x_1_6gf-current"
    current = managed / current_key
    _owner(current, last_used=2.0)
    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    trtexec.chmod(0o755)
    source_sha = _migration_receipt(current, trtexec, b"regnet-full")
    legacy_engine = next((current / "b132").rglob("full_fp16.engine"))

    result = _run_retention(
        base,
        current_key=current_key,
        stable_engine_key=current_key,
        builder_abi_sha256="c" * 64,
        current_trtexec_sha256=hashlib.sha256(trtexec.read_bytes()).hexdigest(),
        native_trt_precision="fp16",
        native_trt_workspace_mb=4096,
        canonical_full_onnx_sha256=(source_sha,),
    )

    migration = result["legacy_receipt_migration"]
    assert migration["source_keys"] == [current_key]
    assert migration["migrated_receipts"] == 1
    canonical_engine = (
        current / "full" / source_sha / "fp16" / "full_fp16.engine"
    )
    assert canonical_engine.read_bytes() == legacy_engine.read_bytes()
    # Same-filesystem adoption does not duplicate the multi-GiB engine data.
    assert canonical_engine.stat().st_ino == legacy_engine.stat().st_ino


def test_retention_prefers_current_abi_and_records_eviction_reason(
    tmp_path: Path,
) -> None:
    base = tmp_path / "splitpoint_runs"
    managed = base / "_onnx_splitpoint_cache" / "tensorrt_managed_v27516"
    compatible = managed / "regnet_x_1_6gf-compatible"
    incompatible = managed / "other-incompatible"
    _owner(
        compatible, last_used=1.0, builder_abi="b" * 64,
        stable_key="regnet_x_1_6gf-compatible",
    )
    _owner(
        incompatible, last_used=99.0, builder_abi="d" * 64,
        stable_key="other-incompatible",
    )
    _simple_retention_receipt(compatible, b"compatible")
    _simple_retention_receipt(incompatible, b"incompatible")

    result = _run_retention(
        base,
        current_key="regnet_x_1_6gf-current",
        stable_engine_key="regnet_x_1_6gf-current",
        builder_abi_sha256="b" * 64,
        max_namespaces=2,
    )
    assert compatible.is_dir()
    assert not incompatible.exists()
    assert result["removed"][0]["name"] == "other-incompatible"
    assert result["removed"][0]["reason"] == "retained_cache_limit_lru"
    history = managed / ".eviction_history.jsonl"
    event = json.loads(history.read_text(encoding="utf-8").splitlines()[-1])
    assert event["namespace_key"] == "other-incompatible"
    assert event["reason"] == "retained_cache_limit_lru"
    assert event["builder_abi_match"] is False

    replay = _run_retention(
        base,
        current_key="other-incompatible",
        stable_engine_key="other-incompatible",
        builder_abi_sha256="d" * 64,
        max_namespaces=4,
    )
    assert replay["prior_eviction"]["namespace_key"] == "other-incompatible"
    assert replay["prior_eviction"]["reason"] == "retained_cache_limit_lru"
