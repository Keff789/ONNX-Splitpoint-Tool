"""Real outer dispatch manager; controlled normal-executor/Quality admission leaves.

No transport, SDK, model inference, released source claim or hardware is used.
The prior dispatcher tests separately exercise the genuine normal executor.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import generic_only_parallel as parallel
from onnx_splitpoint_tool.workflow import generic_only_cell_policy as policy
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def cleanup(group, *, unsafe=False):
    plan = parallel.entry.read_json(group["path"])
    setup = group["setup_id"]
    root = Path(plan["output_root"])
    scope = RemoteProcessLeaseScope(Path(plan["original_run"]).name, plan["execution_id"] + ":" + setup)
    directory = root / "remote-process-leases" / plan["execution_id"] / setup
    directory.mkdir(parents=True, exist_ok=True)
    write(root / "cleanup-status.json", {"active_owned_remote_operations": int(unsafe),
        "registry_cancelled_or_poisoned": unsafe,
        "per_backend_journals": {setup: {"active_owned_remote_operations": int(unsafe),
            "registry_cancelled_or_poisoned": unsafe, "journal": str(directory),
            "journal_pending_count": 0, "journal_cancelled": False, "journal_empty": True,
            "ownership_scope": {"run_sha256": scope.run_sha256, "session_sha256": scope.session_sha256}}}})


@pytest.fixture
def batch(tmp_path, monkeypatch):
    """Only workload/Quality/source-release leaves are controlled; batch is real."""
    epoch = tmp_path / "epoch.json"
    write(epoch, {"software_test_source_epoch": 1})
    expected_epoch = parallel.descriptor(epoch)
    bundle = tmp_path / "frozen-bundle.tar.gz"
    bundle.write_bytes(b"software fixture immutable runner bundle")

    class ControlledFreeze:
        def __init__(self, descriptor):
            assert descriptor == expected_epoch
            self.descriptor = descriptor
            self.payload = {"runner_bundle": parallel.descriptor(bundle)}
            self.failed = False
            self.lock = threading.Lock()
            self.verify()

        def verify(self, *, full=False):
            with self.lock:
                try:
                    if self.failed:
                        raise ValueError("Shared source already failed")
                    parallel.checked(self.descriptor)
                except Exception:
                    self.failed = True
                    raise

    monkeypatch.setattr(parallel, "SourceFreeze", ControlledFreeze)
    monkeypatch.setattr(parallel.entry, "validate_plan", lambda plan, *_: plan["rows"])
    monkeypatch.setattr(parallel.entry, "bound_targets", lambda primary, rows: {
        rows[0]["setup_id"]: {"host": rows[0]["test_host"]}})
    groups, calls, active, intervals = [], [], {}, []
    lock = threading.Lock()
    callbacks = {}

    def prepare(specs):
        nonlocal groups
        plans = []
        for index, (model, setup, host) in enumerate(specs):
            root = tmp_path / f"group-{index}"
            root.mkdir()
            (root / "runner-bundle.tar.gz").write_bytes(bundle.read_bytes())
            primary = tmp_path / "producer" / f"run-{index}"
            primary.mkdir(parents=True)
            plan = {"scope_mode": parallel.MODE, "admission_mode": "generic_only",
                "product_freeze": deepcopy(expected_epoch),
                "cell_execution_control": deepcopy(policy.CONTROL), "execution_id": f"execution-{index}",
                "original_run": str(primary), "output_root": str(root), "scope_source": str(tmp_path / "scope.csv"),
                "rows": [{"case_key": f"case-{index}", "setup_id": setup, "model_id": model,
                    "test_host": host, "result_path": str(root / "results" / "result.json")}]}
            path = root / "plan" / "execution-plan.json"
            write(path, plan)
            plans.append({"plan": parallel.descriptor(path)})
        manifest = {"schema": parallel.SCHEMA, "parallel_remote_setups": True, "max_parallel_setups": 3,
            "setup_queue_mode": "per_setup", "output_root": str(tmp_path / "batch"),
            "source_freeze": expected_epoch, "model_ids": list(dict.fromkeys(value[0] for value in specs)),
            "plans": plans}
        path = tmp_path / "batch.json"
        write(path, manifest)
        groups = parallel._prepare_batch(path, resume=False)[4]
        return path

    def complete(group):
        if not (group["output_root"] / "complete.json").is_file():
            return None
        # Genuine normal first-start and snapshot remain required in this leaf.
        plan = parallel.entry.read_json(group["path"])
        start = parallel.entry.read_json(group["output_root"] / "production-start.json")
        assert start["initial_plan_sha256"] == parallel.entry.file_sha256(
            group["output_root"] / "plan/initial-execution-plan.json")
        assert start["execution_id"] == plan["execution_id"]
        parallel._clean_completed_plan(plan, {group["setup_id"]})
        return {"valid_repetitions": 3, "processes": []}

    monkeypatch.setattr(parallel, "_completed", complete)

    def execute(path, *, resume, _cancel_event, _source_guard):
        group = next(value for value in groups if value["path"] == path)
        plan = parallel.entry.read_json(path)
        policy.admit_controller_attempt(plan, path, group["output_root"],
            group["output_root"] / "runner-bundle.tar.gz", resume=resume, started_at=parallel.entry.now())
        with lock:
            calls.append(group["index"])
            key = group["physical_dut_key"]
            active[key] = active.get(key, 0) + 1
            assert active[key] == 1
        entered = time.monotonic()
        try:
            callback = callbacks.get(group["index"])
            if callback:
                return callback(group, _cancel_event, _source_guard)
            _source_guard()
            process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(.12)"])
            assert process.wait(timeout=5) == 0
            _source_guard()
            write(group["output_root"] / "complete.json", {})
            return 0
        finally:
            cleanup(group, unsafe=group.get("unsafe", False))
            with lock:
                active[key] -= 1
                intervals.append((group["index"], entered, time.monotonic()))

    monkeypatch.setattr(parallel.entry, "execute_plan", execute)
    return SimpleNamespace(prepare=prepare, calls=calls, callbacks=callbacks, epoch=epoch,
        groups=lambda: groups, intervals=intervals, tmp=tmp_path)


def test_three_devices_overlap_aliases_serialize_and_status_is_main_thread_only(batch, monkeypatch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m1", "alias-a", "192.0.2.1"),
                          ("m1", "b", "192.0.2.2"), ("m1", "c", "192.0.2.3")])
    original = parallel.entry.write_json
    writers = []
    def record(path, value):
        if Path(path).name == "batch-status.json":
            writers.append(threading.current_thread())
        return original(path, value)
    monkeypatch.setattr(parallel.entry, "write_json", record)
    assert parallel.execute_batch(path) == 0
    intervals = {index: (start, stop) for index, start, stop in batch.intervals}
    assert max(intervals[index][0] for index in (0, 2, 3)) < min(intervals[index][1] for index in (0, 2, 3))
    assert intervals[1][0] >= intervals[0][1]
    assert writers and set(writers) == {threading.main_thread()}
    state = parallel.entry.read_json(batch.tmp / "batch/batch-status.json")
    assert all(row["status"] == "completed" for row in state["groups"])
    assert all(row["controller_group_started_at"] < row["controller_group_finished_at"] for row in state["groups"])


def test_clean_cell_failure_allows_next_same_device_and_other_branch(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m2", "alias", "192.0.2.1"), ("m2", "b", "192.0.2.2")])
    batch.callbacks[0] = lambda *_: 1
    assert parallel.execute_batch(path) == 1
    assert set(batch.calls) == {0, 1, 2}
    state = parallel.entry.read_json(batch.tmp / "batch/batch-status.json")
    assert state["groups"][0]["status"] == "failed"
    assert state["physical_resource_stops"] == {}


def test_unresolved_cleanup_fences_alias_before_worker_return_and_on_resume(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m2", "alias", "192.0.2.1"), ("m2", "b", "192.0.2.2")])
    def unsafe(group, *_):
        group["unsafe"] = True
        return 1
    batch.callbacks[0] = unsafe
    assert parallel.execute_batch(path) == 1
    assert set(batch.calls) == {0, 2}
    state = parallel.entry.read_json(batch.tmp / "batch/batch-status.json")
    assert state["groups"][1]["status"] == "blocked_physical_resource"
    receipt = next((batch.tmp / "batch/physical-stops").glob("*.json"))
    retained = receipt.read_bytes()
    assert parallel.execute_batch(path, resume=True) == 1
    assert set(batch.calls) == {0, 2}
    assert receipt.read_bytes() == retained
    assert len(list((batch.tmp / "batch/batch-attempts").glob("attempt_????.json"))) == 2


def test_pre_registry_resource_failure_without_cleanup_fences_later_alias(batch, monkeypatch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m2", "alias", "192.0.2.1"), ("m1", "b", "192.0.2.2")])
    normal_leaf = parallel.entry.execute_plan
    def before_registry(path, **kwargs):
        if path == batch.groups()[0]["path"]:
            batch.calls.append(0)
            return 1  # Normal resource lock failed before registry/start creation.
        return normal_leaf(path, **kwargs)
    monkeypatch.setattr(parallel.entry, "execute_plan", before_registry)
    assert parallel.execute_batch(path) == 1
    assert set(batch.calls) == {0, 2}
    state = parallel.entry.read_json(batch.tmp / "batch/batch-status.json")
    assert state["groups"][1]["status"] == "blocked_physical_resource"
    assert state["physical_resource_stops"]["dut:192.0.2.1"]["reason"] == "normal_physical_resource_release_unproven"


def test_success_exit_with_unresolved_cleanup_cannot_complete_batch(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1")])
    def unsafe_success(group, *_):
        group["unsafe"] = True
        write(group["output_root"] / "complete.json", {})
        return 0
    batch.callbacks[0] = unsafe_success
    assert parallel.execute_batch(path) == 1
    assert parallel.entry.read_json(batch.tmp / "batch/batch-status.json")["physical_resource_stops"]


def test_common_source_drift_cancels_active_group_at_next_repeat_boundary(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m1", "b", "192.0.2.2"),
                          ("m2", "c", "192.0.2.1")])
    both = threading.Barrier(2)
    active_stopped = threading.Event()
    def mutate(group, cancel, guard):
        both.wait(timeout=5)
        batch.epoch.write_text("changed actual software source epoch")
        guard()
    def active(group, cancel, guard):
        both.wait(timeout=5)
        assert cancel.wait(5)
        active_stopped.set()
        guard()
    batch.callbacks.update({0: mutate, 1: active})
    assert parallel.execute_batch(path) == 1
    assert active_stopped.is_set() and set(batch.calls) == {0, 1}
    state = parallel.entry.read_json(batch.tmp / "batch/batch-status.json")
    assert state["shared_source_failed"] is True and state["cancel_requested"] is True


def test_explicit_shared_cancel_stops_before_next_boundary_and_queued_alias(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m2", "alias", "192.0.2.1")])
    def cancel(group, event, guard):
        event.set()
        with pytest.raises(RuntimeError, match="cancelled before the next"):
            guard()
        return 1
    batch.callbacks[0] = cancel
    assert parallel.execute_batch(path) == 1
    assert batch.calls == [0]
    state = parallel.entry.read_json(batch.tmp / "batch/batch-status.json")
    assert state["cancel_requested"] is True and state["shared_source_failed"] is False


def test_original_batch_manifest_mutation_cancels_all_queued_groups(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m2", "alias", "192.0.2.1")])
    def mutate(group, event, guard):
        manifest = parallel.entry.read_json(path)
        manifest["unexpected_scope_change"] = True
        write(path, manifest)
        guard()
    batch.callbacks[0] = mutate
    assert parallel.execute_batch(path) == 1
    assert batch.calls == [0]
    state = parallel.entry.read_json(batch.tmp / "batch/batch-status.json")
    assert state["cancel_requested"] is True


def test_queued_plan_mutation_does_not_enter_normal_executor(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m2", "alias", "192.0.2.1"), ("m1", "b", "192.0.2.2")])
    def edit_queued(group, *_):
        later = batch.groups()[1]
        plan = parallel.entry.read_json(later["path"])
        plan["rows"][0]["runtime_config"] = {"input": "substitution"}
        write(later["path"], plan)
        write(group["output_root"] / "complete.json", {})
        return 0
    batch.callbacks[0] = edit_queued
    assert parallel.execute_batch(path) == 1
    assert set(batch.calls) == {0, 2}
    state = parallel.entry.read_json(batch.tmp / "batch/batch-status.json")
    assert "Queued Generic plan changed" in state["groups"][1]["error"]


def test_explicit_resume_preserves_completed_plan_and_original_batch_markers(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1"), ("m1", "b", "192.0.2.2")])
    assert parallel.execute_batch(path) == 0
    start = (batch.tmp / "batch/batch-start.json").read_bytes()
    original_starts = [(group["output_root"] / "production-start.json").read_bytes() for group in batch.groups()]
    with pytest.raises(FileExistsError, match="explicit --resume"):
        parallel.execute_batch(path)
    assert parallel.execute_batch(path, resume=True) == 0
    assert sorted(batch.calls) == [0, 1]
    assert (batch.tmp / "batch/batch-start.json").read_bytes() == start
    assert [(group["output_root"] / "production-start.json").read_bytes() for group in batch.groups()] == original_starts
    assert all(row["status"] == "completed_reused" for row in
        parallel.entry.read_json(batch.tmp / "batch/batch-status.json")["groups"])


def test_resume_rejects_immutable_plan_mutation_before_dispatch(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1")])
    assert parallel.execute_batch(path) == 0
    group = batch.groups()[0]
    plan = parallel.entry.read_json(group["path"])
    plan["rows"][0]["result_path"] += ".substituted"
    write(group["path"], plan)
    with pytest.raises(ValueError, match="binding changed"):
        parallel.execute_batch(path, resume=True)
    assert batch.calls == [0]


def test_parallel_policy_is_explicit_and_cannot_import_paired_plans(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1")])
    manifest = parallel.entry.read_json(path)
    manifest["max_parallel_setups"] = 4
    write(path, manifest)
    with pytest.raises(ValueError, match="maximum three"):
        parallel.execute_batch(path)
    manifest["max_parallel_setups"] = 3
    group = batch.groups()[0]
    plan = parallel.entry.read_json(group["path"])
    plan["scope_mode"] = "generic_completed_task"
    write(group["path"], plan)
    manifest["plans"][0]["plan"] = parallel.descriptor(group["path"])
    write(path, manifest)
    with pytest.raises(ValueError, match="cannot import Native/paired"):
        parallel.execute_batch(path)
    assert batch.calls == []


def test_child_controller_epoch_cannot_change_even_when_runtime_bundle_is_identical(batch):
    path = batch.prepare([("m1", "a", "192.0.2.1")])
    group = batch.groups()[0]
    plan = parallel.entry.read_json(group["path"])
    archive_sha = parallel.entry.file_sha256(group["output_root"] / "runner-bundle.tar.gz")
    old_epoch = batch.tmp / "older-controller-epoch.json"
    write(old_epoch, {"software_test_source_epoch": "older controller, identical runtime archive"})
    plan["product_freeze"] = parallel.descriptor(old_epoch)
    write(group["path"], plan)
    manifest = parallel.entry.read_json(path)
    manifest["plans"][0]["plan"] = parallel.descriptor(group["path"])
    write(path, manifest)
    with pytest.raises(ValueError, match="exact shared controller/source freeze"):
        parallel.execute_batch(path)
    assert parallel.entry.file_sha256(group["output_root"] / "runner-bundle.tar.gz") == archive_sha
    assert batch.calls == [] and not (batch.tmp / "batch/batch-start.json").exists()


def test_completed_raw_fast_path_requires_genuine_original_start(batch, monkeypatch):
    path = batch.prepare([("m1", "a", "192.0.2.1")])
    group = batch.groups()[0]
    from onnx_splitpoint_tool import generic_completion_reporting
    monkeypatch.setattr(generic_completion_reporting, "load_valid_repetitions", lambda *_: [{}, {}, {}])
    # Recover the real function rather than this fixture's controlled admission.
    original = parallel.__loader__.get_source(parallel.__name__)
    import ast
    node = next(node for node in ast.parse(original).body if isinstance(node, ast.FunctionDef) and node.name == "_completed")
    namespace = dict(parallel.__dict__)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(Path(parallel.__file__)), "exec"), namespace)
    with pytest.raises(FileNotFoundError, match="production-start"):
        namespace["_completed"](group)


def test_real_main_manifest_cannot_admit_unindexed_worktree_readers(tmp_path):
    """Exact old reproduced bug: genuine main manifest + real changed readers."""
    main = Path("/home/kmika/ONNX-Splitpoint-Tool")
    if not main.is_dir() or main.resolve() == Path(parallel.__file__).resolve().parents[2]:
        pytest.skip("Requires the preserved published-main and isolated-worktree test setup")
    freeze = {"schema": "generic420-published-product-freeze-v1", "controller_commit": "main-real-epoch",
        "published_origin_main": "main-real-epoch", "source_manifest": parallel.descriptor(main / "SOURCE_MANIFEST.json"),
        "source_checksums": parallel.descriptor(main / "SHA256SUMS.txt"),
        "imports": {name: parallel.descriptor(__import__(name, fromlist=["x"]).__file__) for name in parallel._READERS}}
    path = tmp_path / "mixed-freeze.json"
    write(path, freeze)
    with pytest.raises(ValueError, match="outside the verified manifest source root"):
        parallel.SourceFreeze(parallel.descriptor(path))


@pytest.fixture
def frozen_sources(tmp_path, monkeypatch):
    """Small source-tree fixture; normal release verifier is a declared test leaf."""
    root = tmp_path / "source"
    root.mkdir()
    modules = {}
    names = (*parallel._READERS, "onnx_splitpoint_tool.source_integrity", "onnx_splitpoint_tool.gui.controller",
             "onnx_splitpoint_tool.split_export_runners")
    for name in names:
        path = root / (name.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# controlled source-tree member\n")
        modules[name] = SimpleNamespace(__file__=str(path))
    records = [{"path": path.relative_to(root).as_posix(), "sha256": parallel.entry.file_sha256(path),
                "size": path.stat().st_size} for path in root.rglob("*.py")]
    write(root / "SOURCE_MANIFEST.json", {"files": records})
    (root / "SHA256SUMS.txt").write_text("controlled unit source inventory")
    tree = tmp_path / "freeze/runner-bundle"
    (tree / "runner").mkdir(parents=True)
    (tree / "runner/run_split_onnxruntime.py").write_text("# actual unit runner bytes\n")
    bundle = tree.with_suffix(".tar.gz")
    def make_bundle(destination):
        destination.mkdir(parents=True)
        output = destination.with_suffix(".tar.gz")
        with tarfile.open(output, "w:gz") as archive:
            archive.add(tree / "runner/run_split_onnxruntime.py", arcname="runner/run_split_onnxruntime.py")
        return output
    bundle.write_bytes(make_bundle(tmp_path / "unit-writer").read_bytes())
    freeze = {"schema": "generic420-published-product-freeze-v1", "controller_commit": "fixture",
        "published_origin_main": "fixture", "source_manifest": parallel.descriptor(root / "SOURCE_MANIFEST.json"),
        "source_checksums": parallel.descriptor(root / "SHA256SUMS.txt"),
        "runner_bundle": parallel.descriptor(bundle), "runtime_sources": [parallel.descriptor(path) for path in tree.rglob("*.py")],
        "imports": {name: parallel.descriptor(modules[name].__file__) for name in parallel._READERS}}
    monkeypatch.setattr(parallel.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(parallel, "verify_installed_source_integrity", lambda *_: {"ok": True})
    monkeypatch.setattr(parallel.entry, "prepare_runner_bundle", make_bundle)
    path = tmp_path / "freeze/FREEZE.json"
    write(path, freeze)
    return SimpleNamespace(path=path, freeze=freeze, root=root, tree=tree)


def test_source_tree_manifest_and_normal_runtime_projection_join(frozen_sources):
    frozen = parallel.SourceFreeze(parallel.descriptor(frozen_sources.path))
    assert frozen.failed is False
    frozen_sources.tree.joinpath("runner/run_split_onnxruntime.py").write_text("changed source")
    with pytest.raises(ValueError, match="changed or missing"):
        frozen.verify()
    assert frozen.failed is True


def test_product_reader_must_be_indexed_even_inside_same_tree(frozen_sources):
    path = frozen_sources.root / "SOURCE_MANIFEST.json"
    manifest = parallel.entry.read_json(path)
    manifest["files"] = manifest["files"][1:]
    write(path, manifest)
    frozen_sources.freeze["source_manifest"] = parallel.descriptor(path)
    write(frozen_sources.path, frozen_sources.freeze)
    with pytest.raises(ValueError, match="manifest membership"):
        parallel.SourceFreeze(parallel.descriptor(frozen_sources.path))


def test_runtime_source_list_cannot_omit_actual_executed_member(frozen_sources):
    frozen_sources.freeze["runtime_sources"] = []
    write(frozen_sources.path, frozen_sources.freeze)
    with pytest.raises(ValueError, match="source list differs"):
        parallel.SourceFreeze(parallel.descriptor(frozen_sources.path))


def test_foreign_runtime_bundle_cannot_borrow_genuine_manifest(frozen_sources, monkeypatch, tmp_path):
    def foreign_writer(destination):
        destination.mkdir(parents=True)
        source = destination / "different.py"
        source.write_text("# different current product bytes\n")
        output = destination.with_suffix(".tar.gz")
        with tarfile.open(output, "w:gz") as archive:
            archive.add(source, arcname="runner/run_split_onnxruntime.py")
        return output
    monkeypatch.setattr(parallel.entry, "prepare_runner_bundle", foreign_writer)
    with pytest.raises(ValueError, match="normal product writer"):
        parallel.SourceFreeze(parallel.descriptor(frozen_sources.path))
