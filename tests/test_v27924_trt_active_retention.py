"""A finished compiler command must not make later Native inputs disposable."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.remote_run import (
    _remote_storage_preflight,
    _remote_trt_cache_retention_command,
)


def _run(base: Path, key: str, *, preserve: bool, limit: int = 6):
    base.mkdir(parents=True, exist_ok=True)
    command = _remote_trt_cache_retention_command(
        remote_base=str(base), current_key=key,
        max_namespaces=limit, max_bytes=1024**3,
        preserve_existing_namespaces=preserve,
    )
    proc = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, timeout=30,
    )
    marker = "SPLITPOINT_TRT_RETENTION_JSON="
    results = [json.loads(x[len(marker):]) for x in proc.stdout.splitlines()
               if x.startswith(marker)]
    assert len(results) == 1, proc.stdout + proc.stderr
    return proc.returncode, results[0]


def _bound_artifacts(namespace: Path) -> dict[str, str]:
    """Write real bound files plus a valid receipt, making them LRU eligible."""
    artifacts = {}
    for rel, value in {
        "full/model/source.onnx": b"full source bytes",
        "full/model/full_fp16.engine": b"serialized engine bytes",
        "native_split_quality/b060/part1.hef": b"quality-bound part1 bytes",
    }.items():
        path = namespace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        artifacts[str(path)] = hashlib.sha256(value).hexdigest()
    engine = namespace / "full/model/full_fp16.engine"
    receipt = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1, "build_returncode": 0, "dry_run": False,
        "engine": str(engine.resolve()), "engine_sha256": artifacts[str(engine)],
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    receipt_path = engine.parent / "engine_build_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    artifacts[str(receipt_path)] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return artifacts


def test_seven_models_preserve_all_earlier_bound_sources(tmp_path: Path):
    base = tmp_path / "remote"
    managed = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
    bound = {}
    models = ["resnet50", "mobilenet", "regnet", "yolo11l", "yolo26m", "yolo26s", "model7"]
    for index, model in enumerate(models, 1):
        key = model + "-cache"
        rc, result = _run(base, key, preserve=True)
        assert rc == 0
        assert result["removed"] == []
        assert result["reason"] == "retention_deferred_active_workflow"
        assert result["managed_count"] == index
        bound.update(_bound_artifacts(managed / key))
        # Simulate later consumers opening their retained exact bound paths.
        for path, digest in bound.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert result["retention_limits_exceeded"] is True
    # A later warm dispatch must use the same preservation branch as bootstrap.
    rc, result = _run(base, "model7-cache", preserve=True)
    assert rc == 0 and result["removed"] == []
    assert not (managed / ".eviction_history.jsonl").exists()


def test_retained_byte_target_is_deferred_without_deleting_files(tmp_path: Path):
    base = tmp_path / "remote"
    rc, _ = _run(base, "old-cache", preserve=True)
    assert rc == 0
    old = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516/old-cache"
    sparse = old / "large-existing-artifact"
    with sparse.open("wb") as handle:
        handle.truncate(1024**3 + 100)
    rc, result = _run(base, "next-cache", preserve=True)
    assert rc == 0 and result["retention_limits_exceeded"] is True
    assert result["removed"] == []
    assert sparse.stat().st_size == 1024**3 + 100


def test_standalone_retention_still_evicts_valid_inactive_lru(tmp_path: Path):
    base = tmp_path / "remote"
    managed = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516"
    rc, _ = _run(base, "old-cache", preserve=True)
    assert rc == 0
    _bound_artifacts(managed / "old-cache")
    rc, result = _run(base, "next-cache", preserve=False, limit=1)
    assert rc == 0
    assert [x["name"] for x in result["removed"]] == ["old-cache"]
    assert not (managed / "old-cache").exists()


def test_active_workflow_does_not_admit_unsafe_current_namespace(tmp_path: Path):
    base = tmp_path / "remote"
    rc, _ = _run(base, "current-cache", preserve=True)
    assert rc == 0
    current = base / "_onnx_splitpoint_cache/tensorrt_managed_v27516/current-cache"
    target = tmp_path / "unowned-original"
    target.write_bytes(b"must remain")
    (current / "unsafe").symlink_to(target)
    rc, result = _run(base, "current-cache", preserve=True)
    assert rc == 75
    assert result["admission_ok"] is False
    assert result["reason"] == "current_managed_trt_cache_contains_symlink"
    assert result["removed"] == [] and target.read_bytes() == b"must remain"


@pytest.mark.parametrize("kind", ["bytes", "inodes"])
def test_physical_storage_admission_still_rejects_without_mutation(tmp_path: Path, kind: str):
    class LocalReadOnlyTransport:
        def run_read_only(self, cmd, *, timeout):
            proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
            return proc.returncode, proc.stdout + proc.stderr

    preserved = tmp_path / "original.hef"
    preserved.write_bytes(b"valuable compiled bytes")
    with pytest.raises(RuntimeError, match="refusing remote mutation"):
        _remote_storage_preflight(
            LocalReadOnlyTransport(), str(tmp_path),
            required_free_bytes=2**100 if kind == "bytes" else 0,
            required_free_inodes=2**100 if kind == "inodes" else 0,
            stage="active_workflow_physical_admission",
        )
    assert preserved.read_bytes() == b"valuable compiled bytes"
    assert list(tmp_path.iterdir()) == [preserved]
