"""Small real validation scopes across normal parallel preparation/execution.

The dispatch manager, public execution wrapper, digests and final rehash are
real. Existing software-only batch fixtures control SDK/transport/quality leaves.
"""
import hashlib
import json
from pathlib import Path
import threading

import pytest

from onnx_splitpoint_tool import generic_only_plan as admission
from onnx_splitpoint_tool.workflow import generic_only_parallel as parallel
from test_generic_only_parallel import batch, write


_NORMAL_EXECUTE = parallel.entry.execute_plan


def install_validation(batch, monkeypatch):
    sources, calls, hashes, scopes = {}, [], [], {}
    for group in batch.groups():
        path = batch.tmp / f"quality-{group['index']}.json"
        path.write_bytes(b'{"records":[1]}')
        sources[group["index"]] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    original_digest = admission._file_sha256
    def counted(path):
        hashes.append(Path(path))
        return original_digest(path)
    monkeypatch.setattr(admission, "_file_sha256", counted)

    def proof(row):
        index = row["identity"]["index"]
        calls.append(index)
        admission.read_bound(sources[index])
        return {"status": "reference_close", "quality_eligible": True}
    monkeypatch.setattr(admission, "_validate_quality_reference", proof)

    def validate(plan, *_):
        index = int(plan["execution_id"].split("-")[-1])
        scope = admission._VALIDATION_SCOPE.get()
        scopes.setdefault(index, []).append(scope)
        for _ in range(3):
            admission.validate_quality_reference({"task": "detection", "identity": {"index": index}})
        return plan["rows"]
    monkeypatch.setattr(parallel.entry, "validate_plan", validate)
    controlled_execute = parallel.entry.execute_plan
    def execution(path, plan, **kwargs):
        validate(plan)
        return controlled_execute(path, **kwargs)
    monkeypatch.setattr(parallel.entry, "_execute_plan", execution)
    monkeypatch.setattr(parallel.entry, "execute_plan", _NORMAL_EXECUTE)
    return sources, calls, hashes, scopes


def test_first_and_final_hash_with_same_scope_in_real_execution_wrapper(batch, monkeypatch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m1", "b", "192.0.2.2")])
    sources, calls, hashes, scopes = install_validation(batch, monkeypatch)
    barrier = threading.Barrier(2)
    def overlap(group, cancel, source_guard):
        barrier.wait(timeout=5)
        assert admission._VALIDATION_SCOPE.get() is scopes[group["index"]][0]
        source_guard()
        write(group["output_root"] / "complete.json", {})
        return 0
    batch.callbacks.update({0: overlap, 1: overlap})
    assert parallel.execute_batch(path) == 0
    assert sorted(calls) == [0, 1]
    assert all(len(values) == 2 and values[0] is values[1] for values in scopes.values())
    assert scopes[0][0] is not scopes[1][0]
    assert all(hashes.count(Path(source["path"])) == 2 for source in sources.values())
    assert admission._VALIDATION_SCOPE.get() is None
    admission.read_bound(sources[0])
    assert hashes.count(Path(sources[0]["path"])) == 3


@pytest.mark.parametrize("hide_file_version_change", [False, True])
def test_final_full_rehash_blocks_drift_after_last_executor_read(
        batch, monkeypatch, hide_file_version_change):
    path = batch.prepare([("m1", "a", "192.0.2.1")])
    sources, calls, hashes, scopes = install_validation(batch, monkeypatch)
    source = Path(sources[0]["path"])
    original_version = admission._file_version
    before = original_version(source)
    if hide_file_version_change:
        monkeypatch.setattr(admission, "_file_version", lambda value: (
            before if Path(value) == source else original_version(value)))
    def change(group, *_):
        write(group["output_root"] / "complete.json", {})
        source.write_bytes(b'{"records":[2]}')
        return 0
    batch.callbacks[0] = change
    assert parallel.execute_batch(path) == 1
    status = json.loads((batch.tmp / "batch/batch-status.json").read_text())
    assert status["groups"][0]["status"] == "blocked"
    assert "source changed" in status["groups"][0]["error"]
    assert batch.calls == [0]
    assert admission._VALIDATION_SCOPE.get() is None


def test_prepare_failure_closes_every_open_group_scope_before_any_batch_start(batch, monkeypatch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m1", "b", "192.0.2.2")])
    sources, calls, hashes, scopes = install_validation(batch, monkeypatch)
    validate = parallel.entry.validate_plan
    def reject(plan, *args):
        rows = validate(plan, *args)
        if plan["execution_id"] == "execution-1":
            raise ValueError("fixture invalid immutable plan")
        return rows
    monkeypatch.setattr(parallel.entry, "validate_plan", reject)
    with pytest.raises(ValueError, match="invalid immutable plan"):
        parallel.execute_batch(path)
    assert not (batch.tmp / "batch/batch-start.json").exists()
    assert batch.calls == []
    assert all(hashes.count(Path(source["path"])) == 2 for source in sources.values())
    assert admission._VALIDATION_SCOPE.get() is None


def test_never_dispatched_cancelled_group_still_gets_final_full_hash(batch, monkeypatch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m1", "alias-a", "192.0.2.1")])
    sources, calls, hashes, scopes = install_validation(batch, monkeypatch)
    def cancel(group, event, _guard):
        event.set()
        write(group["output_root"] / "complete.json", {})
        return 0
    batch.callbacks[0] = cancel
    assert parallel.execute_batch(path) == 1
    assert batch.calls == [0]
    assert all(hashes.count(Path(source["path"])) == 2 for source in sources.values())
    assert admission._VALIDATION_SCOPE.get() is None


def test_batch_does_not_share_an_existing_outer_mutable_scope(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1")])
    with admission.validation_scope():
        with pytest.raises(ValueError, match="independent validation scopes"):
            parallel.execute_batch(path)
    assert batch.calls == []
    assert not (batch.tmp / "batch/batch-start.json").exists()
