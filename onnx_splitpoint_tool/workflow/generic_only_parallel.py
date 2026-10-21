"""Normal Generic-only plans on the existing physical setup dispatch manager.

Each group keeps its prepared execution ID, normal dispatcher, locks, leases,
repeat budget and output tree. Only independent physical setup queues overlap.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
import hashlib
import importlib
import json
from pathlib import Path
import queue
import signal
import tarfile
import tempfile
import threading

from ..generic_only_plan import MODE
from ..source_integrity import verify_installed_source_integrity
from ..remote.process_lease import RemoteProcessLeaseJournal, RemoteProcessLeaseScope
from . import generic_completed_task as entry
from .execution_binding import PhysicalSetupDispatchManager, physical_dut_key
from .generic_only_cell_policy import plan_binding, validate_dispatch_history, read_dispatch_attempts
from .run_control import EvaluationRunLock


SCHEMA = "onnx-splitpoint/generic-only-parallel-execution/v1"
_READERS = ("onnx_splitpoint_tool.workflow.generic_completed_task",
    "onnx_splitpoint_tool.workflow.generic_only_parallel",
    "onnx_splitpoint_tool.workflow.generic_only_cell_policy",
    "onnx_splitpoint_tool.workflow.execution_binding",
    "onnx_splitpoint_tool.generic_only_plan", "onnx_splitpoint_tool.generic_only_reporting")


def descriptor(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": entry.file_sha256(path), "size_bytes": path.stat().st_size}


def checked(value):
    if not isinstance(value, dict):
        raise ValueError("Parallel Generic source descriptor missing")
    path = Path(str(value.get("path") or ""))
    size = value.get("size_bytes", value.get("size"))
    if (not path.is_absolute() or path.is_symlink() or not path.is_file()
            or entry.file_sha256(path) != value.get("sha256")
            or size is not None and (type(size) is not int or path.stat().st_size != size)):
        raise ValueError("Parallel Generic source changed or missing: " + str(path))
    return path


class SourceFreeze:
    """One shared source epoch, checked again before every queued group."""
    def __init__(self, value):
        self.descriptor = dict(value)
        self.path = checked(value)
        self.payload = entry.read_json(self.path)
        self.lock = threading.RLock()
        self.failed = False
        freeze = self.payload
        if (freeze.get("schema") not in {"detectors20-published-product-freeze-v1", "generic420-published-product-freeze-v1"}
                or not freeze.get("controller_commit")
                or freeze.get("published_origin_main") != freeze["controller_commit"]):
            raise ValueError("Parallel Generic requires one published source freeze")
        imports = freeze.get("imports") or {}
        if any(module not in imports for module in _READERS):
            raise ValueError("Parallel Generic freeze omits a controller/manager reader")
        self.source_root = checked(freeze["source_manifest"]).resolve().parent
        if checked(freeze["source_manifest"]).name != "SOURCE_MANIFEST.json":
            raise ValueError("Parallel source manifest must be the normal installed inventory")
        if checked(freeze["source_checksums"]).resolve() != self.source_root / "SHA256SUMS.txt":
            raise ValueError("Parallel source checksums are outside the manifest source root")
        manifest = entry.read_json(self.source_root / "SOURCE_MANIFEST.json")
        self.inventory = {record["path"]: record for record in manifest["files"]}
        if len(self.inventory) != len(manifest["files"]):
            raise ValueError("Parallel source manifest contains duplicate logical paths")
        for value in imports.values():
            self._product_source(checked(value))
        for module in _READERS:
            actual = Path(importlib.import_module(module).__file__).resolve()
            expected = imports[module]
            if actual != checked(expected).resolve():
                raise ValueError("Parallel Generic imported reader is outside the frozen source")
            self._product_source(actual)
        # These ordinary writer/verifier implementations supply the runtime
        # source projection below, and must belong to that same installed tree.
        for module in ("onnx_splitpoint_tool.source_integrity", "onnx_splitpoint_tool.gui.controller",
                       "onnx_splitpoint_tool.split_export_runners"):
            self._product_source(Path(importlib.import_module(module).__file__).resolve())
        self.sources = [freeze["source_manifest"], freeze["source_checksums"], freeze["runner_bundle"],
                        *imports.values(), *freeze["runtime_sources"]]
        self.verify(full=True)
        self._verify_runtime_projection()

    def _product_source(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.source_root):
            raise ValueError("Frozen product reader is outside the verified manifest source root")
        record = self.inventory.get(path.relative_to(self.source_root).as_posix())
        if (record is None or record.get("sha256") != entry.file_sha256(path)
                or record.get("size") != path.stat().st_size):
            raise ValueError("Frozen product reader has no exact source-manifest membership")

    @staticmethod
    def _bundle_files(path):
        records = {}
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                name = Path(member.name)
                if (not member.isfile() or name.is_absolute() or ".." in name.parts
                        or member.name in records):
                    raise ValueError("Frozen normal runner bundle has an unsafe/duplicate member")
                stream = archive.extractfile(member)
                records[member.name] = {"sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                                        "size": member.size}
        return records

    def _verify_runtime_projection(self):
        frozen = self._bundle_files(checked(self.payload["runner_bundle"]))
        tree = checked(self.payload["runner_bundle"]).parent / "runner-bundle"
        runtime = {}
        for value in self.payload["runtime_sources"]:
            path = checked(value).resolve()
            if not path.is_relative_to(tree) or path.suffix != ".py":
                raise ValueError("Frozen runtime source is outside its normal bundle tree")
            name = path.relative_to(tree).as_posix()
            if name in runtime:
                raise ValueError("Duplicate frozen runtime source")
            runtime[name] = {"sha256": value["sha256"], "size": path.stat().st_size}
        if runtime != {name: value for name, value in frozen.items() if name.endswith(".py")}:
            raise ValueError("Frozen runtime Python source list differs from the actual runner bundle")
        # The regular product writer copies the measured modules and renders
        # the actual runner. Comparing all members also covers non-Python assets
        # and generated compatibility files without inventing an origin alias.
        with tempfile.TemporaryDirectory(prefix="generic-parallel-source-") as directory:
            actual = entry.prepare_runner_bundle(Path(directory) / "runner-bundle")
            if self._bundle_files(actual) != frozen:
                raise ValueError("Frozen runtime bundle differs from its verified normal product writer")

    def verify(self, *, full=False):
        with self.lock:
            if self.failed:
                raise ValueError("Shared Generic source epoch already failed")
            try:
                checked(self.descriptor)
                for value in self.sources:
                    checked(value)
                if full:
                    report = verify_installed_source_integrity(self.source_root)
                    if report.get("ok") is not True:
                        raise ValueError("Normal installed source integrity rejected the parallel epoch")
            except BaseException:
                self.failed = True
                raise


def _write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def _prepare_batch(manifest_path, *, resume):
    manifest_path = Path(manifest_path).resolve()
    source = descriptor(manifest_path)
    manifest = entry.read_json(manifest_path)
    checked(source)
    if (manifest.get("schema") != SCHEMA or manifest.get("parallel_remote_setups") is not True
            or type(manifest.get("max_parallel_setups")) is not int or manifest["max_parallel_setups"] != 3
            or manifest.get("setup_queue_mode") != "per_setup"
            or not isinstance(manifest.get("plans"), list) or not manifest["plans"]):
        raise ValueError("Explicit Generic-only per-setup parallel batch with maximum three setups required")
    freeze = SourceFreeze(manifest["source_freeze"])
    output = Path(manifest["output_root"]).resolve()
    start = output / "batch-start.json"
    prior = entry.read_json(start) if resume and start.is_file() else None
    if resume and (prior is None or prior.get("manifest") != source):
        raise ValueError("Batch resume requires its unchanged original manifest/start")
    if prior is not None and (not isinstance(prior.get("plans"), list)
                              or len(prior["plans"]) != len(manifest["plans"])):
        raise ValueError("Parallel batch resume scope differs")
    if not resume and start.exists():
        raise FileExistsError("Batch start retained; use explicit --resume")
    groups, roots, execution_ids = [], [], set()
    for index, value in enumerate(manifest["plans"]):
        declared = value["plan"]
        path = Path(declared["path"]).resolve()
        if prior is not None:
            snapshot = prior["plans"][index]
            if snapshot["plan"] != declared:
                raise ValueError("Batch resume plan selection changed")
            initial = entry.read_json(checked(snapshot["initial_plan"]))
            current = entry.read_json(path)
            if plan_binding(current) != plan_binding(initial):
                raise ValueError("Batch plan execution/config/source/input binding changed")
        else:
            checked(declared)
            current = entry.read_json(path)
        if current.get("scope_mode") != MODE or current.get("admission_mode") != "generic_only":
            raise ValueError("Parallel batch cannot import Native/paired plans")
        if current.get("product_freeze") != freeze.descriptor:
            raise ValueError("Parallel child plan differs from the exact shared controller/source freeze")
        primary = Path(current["original_run"]).resolve()
        rows = entry.validate_plan(current, Path(current["scope_source"]), primary)
        targets = entry.bound_targets(primary, rows)
        models = {row["model_id"] for row in rows}
        if len(targets) != 1 or len(models) != 1:
            raise ValueError("Each parallel group must be one prepared model/setup plan")
        setup, target = next(iter(targets.items()))
        root = Path(current["output_root"]).resolve()
        if (root in roots or any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots)
                or output == root or output.is_relative_to(root) or root.is_relative_to(output)
                or output.is_relative_to(primary)):
            raise ValueError("Parallel plan output trees overlap or include their source/batch state")
        if current["execution_id"] in execution_ids:
            raise ValueError("Parallel plans share a measurement execution identity")
        if entry.file_sha256(root / "runner-bundle.tar.gz") != freeze.payload["runner_bundle"]["sha256"]:
            raise ValueError("Parallel prepared runner differs from the shared source epoch")
        roots.append(root); execution_ids.add(current["execution_id"])
        groups.append({"index": index, "plan": declared, "path": path, "payload": current,
            "model_id": next(iter(models)), "setup_id": setup, "physical_dut_key": physical_dut_key(target),
            "output_root": root})
    models = manifest.get("model_ids")
    if (not isinstance(models, list) or not models or len(models) != len(set(models))
            or set(models) != {group["model_id"] for group in groups}):
        raise ValueError("Parallel batch requires its complete frozen model order")
    if prior is not None and len(prior["plans"]) != len(groups):
        raise ValueError("Parallel batch resume scope differs")
    return manifest, source, output, freeze, groups


def _clean_completed_plan(plan, targets):
    root = Path(plan["output_root"])
    cleanup = entry.read_json(root / "cleanup-status.json")
    if (cleanup.get("active_owned_remote_operations") != 0
            or cleanup.get("registry_cancelled_or_poisoned") is not False):
        raise ValueError("Completed plan has no clean normal ownership proof")
    for setup in targets:
        independent = plan.get("cell_execution_control") is not None
        scope = RemoteProcessLeaseScope(Path(plan["original_run"]).name,
            plan["execution_id"] + (":" + setup if independent else ""))
        directory = root / "remote-process-leases" / plan["execution_id"]
        if independent:
            directory = directory / setup
            bound = cleanup["per_backend_journals"][setup]
            if (Path(bound["journal"]).resolve() != directory.resolve()
                    or bound["ownership_scope"] != {"run_sha256": scope.run_sha256, "session_sha256": scope.session_sha256}
                    or bound.get("active_owned_remote_operations") != 0
                    or bound.get("registry_cancelled_or_poisoned") is not False
                    or bound.get("journal_pending_count") != 0 or bound.get("journal_cancelled") is not False
                    or bound.get("journal_empty") is not True):
                raise ValueError("Completed plan ownership journal scope differs")
        if not directory.is_dir() or directory.is_symlink() or any(directory.iterdir()):
            raise ValueError("Completed plan matching ownership journal is not actually empty")
        journal = RemoteProcessLeaseJournal(scope=scope, directory=directory)
        if journal.pending_count() or journal.is_cancelled():
            raise ValueError("Completed plan ownership cleanup remains unresolved")


def _require_group_binding(group):
    current = entry.read_json(group["path"])
    if plan_binding(current) != plan_binding(group["payload"]):
        raise ValueError("Queued Generic plan changed its bound execution/config/source/input projection")
    return current


def _completed(group):
    from ..generic_completion_reporting import load_valid_repetitions, validate_completion_case
    from ..generic_only_reporting import generic_only_admission
    plan = entry.read_json(group["path"])
    rows, proofs = plan["rows"], []
    if not all(len(load_valid_repetitions(row["result_path"], row)) == 3 for row in rows):
        return None
    root = Path(plan["output_root"])
    start = entry.read_json(root / "production-start.json")
    if (start.get("execution_id") != plan["execution_id"]
            or Path(start.get("plan", "")).resolve() != group["path"].resolve()
            or start.get("runner_bundle_sha256") != entry.file_sha256(root / "runner-bundle.tar.gz")):
        raise ValueError("Complete raw work has no genuine original normal production-start binding")
    if plan.get("cell_execution_control") is not None:
        initial = root / "plan/initial-execution-plan.json"
        if (entry.file_sha256(initial) != start.get("initial_plan_sha256")
                or plan_binding(entry.read_json(initial)) != plan_binding(plan)):
            raise ValueError("Complete raw work differs from its original independent-plan snapshot")
    for row in rows:
        path = Path(row["result_path"])
        before = entry.file_sha256(path)
        observation = validate_completion_case(entry.read_json(path), row, result_root=path.parent)
        admission = generic_only_admission(observation, row, cell_control=plan.get("cell_execution_control"))
        validate_dispatch_history(row, observation["repetitions"], read_dispatch_attempts(path))
        if (admission["eligible_for_generic_only_execution"] is not True or entry.file_sha256(path) != before):
            raise ValueError("Parsed complete plan has no unchanged normal functional/dispatch admission")
        proofs.extend({key: repetition.get(key) for key in (
            "repetition_index", "execution_id", "boot_id", "pid", "process_start_ticks", "started_at", "completed_at")}
            for repetition in observation["repetitions"])
    _clean_completed_plan(plan, {group["setup_id"]})
    return {"valid_repetitions": len(proofs), "processes": proofs,
        "original_terminal_status": entry.read_json(Path(plan["output_root"]) / "terminal-status.json")}


def execute_batch(manifest_path, *, resume=False):
    """Run normal single-setup plans, preserving every plan's own durable state."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("Parallel batch controller must own the main-thread signal handlers")
    manifest, source, output, freeze, groups = _prepare_batch(manifest_path, resume=resume)
    output.mkdir(parents=True, exist_ok=True)
    cancel, events = threading.Event(), queue.SimpleQueue()
    manager = None
    old_handlers = {}
    records = [{"index": group["index"], "plan": group["plan"], "source_freeze": freeze.descriptor,
                "model_id": group["model_id"], "setup_id": group["setup_id"],
                "physical_dut_key": group["physical_dut_key"], "status": "queued"} for group in groups]
    attempt = None
    quarantine, quarantine_lock = {}, threading.RLock()
    def source_guard(*, full=False):
        if cancel.is_set():
            raise RuntimeError("Parallel Generic cancelled before the next group/case/repetition boundary")
        try:
            checked(source)
            freeze.verify(full=full)
        except BaseException:
            cancel.set()
            raise
    def publish():
        with quarantine_lock:
            physical_stops = dict(quarantine)
        entry.write_json(output / "batch-status.json", {"schema": SCHEMA, "updated_at": entry.now(),
            "manifest": source, "source_freeze": freeze.descriptor, "groups": records,
            "cancel_requested": cancel.is_set(), "shared_source_failed": freeze.failed,
            "physical_resource_stops": physical_stops,
            "timestamp_scope": "controller group/child-process windows, not NPU activity"})
    def quarantine_group(group, error):
        key = group["physical_dut_key"]
        receipt = output / "physical-stops" / (attempt.stem + f"_group_{group['index']:04d}.json")
        record = {"schema": SCHEMA, "at": entry.now(), "manifest": source,
            "source_freeze": freeze.descriptor, "original_batch_start": descriptor(output / "batch-start.json"),
            "physical_dut_key": key, "group_index": group["index"], "plan": group["plan"],
            "reason": "normal_physical_resource_release_unproven", "error": str(error),
            "automatic_clear": False, "receipt": str(receipt)}
        # This immutable per-group event is written before the dispatch manager
        # releases its physical queue head. Shared status remains main-thread-only.
        with quarantine_lock:
            quarantine[key] = record
            _write_exclusive(receipt, record)
    def work(group):
        index = group["index"]
        events.put((index, {"status": "running", "controller_group_started_at": entry.now()}))
        result = {}
        ownership_check_required = False
        def group_guard():
            source_guard()
            _require_group_binding(group)
        try:
            if cancel.is_set():
                return {"status": "cancelled_before_dispatch"}
            with quarantine_lock:
                if group["physical_dut_key"] in quarantine:
                    return {"status": "blocked_physical_resource",
                            "error": "Earlier normal resource release remains unproven for this physical DUT"}
            source_guard(full=True)
            _require_group_binding(group)
            ownership_check_required = (group["output_root"] / "production-start.json").exists()
            complete = _completed(group)
            if complete is not None:
                group_guard()
                return {"status": "completed_reused", "evidence": complete, "normal_execute_called": False}
            plan = _require_group_binding(group)
            started = (group["output_root"] / "production-start.json").exists()
            if started and (not resume or plan.get("cell_execution_control") is None):
                return {"status": "blocked", "error": "Retained incomplete plan requires its explicit bounded resume policy"}
            ownership_check_required = True
            rc = entry.execute_plan(group["path"], resume=started, _cancel_event=cancel, _source_guard=group_guard)
            group_guard()
            complete = _completed(group)
            group_guard()
            if rc == 0 and complete is not None:
                result = {"status": "completed", "evidence": complete, "normal_execute_called": True}
            else:
                result = {"status": "failed", "normal_exitcode": rc, "normal_execute_called": True}
            return result
        except Exception as error:
            return {"status": "blocked", "error": f"{type(error).__name__}: {error}", "shared_source_failed": freeze.failed}
        finally:
            if ownership_check_required:
                try:
                    _clean_completed_plan(entry.read_json(group["path"]), {group["setup_id"]})
                except Exception as error:
                    quarantine_group(group, error)
            events.put((index, {"controller_group_finished_at": entry.now()}))
    with EvaluationRunLock(out_root=output.parent, run_dir=output,
            owner={"run_id": output.name, "purpose": "generic_only_parallel_batch"}):
        start = output / "batch-start.json"
        if not resume:
            snapshots = []
            for group in groups:
                path = output / "plan-snapshots" / f"plan_{group['index']:04d}.json"
                _write_exclusive(path, group["payload"])
                snapshots.append({"plan": group["plan"], "initial_plan": descriptor(path)})
            _write_exclusive(start, {"manifest": source, "source_freeze": freeze.descriptor,
                "started_at": entry.now(), "plans": snapshots})
        attempts = output / "batch-attempts"
        existing = sorted(attempts.glob("attempt_*.json")) if attempts.exists() else []
        # Only immutable start markers participate in the sequence.
        existing = [path for path in existing if path.stem.removeprefix("attempt_").isdigit()]
        if [path.name for path in existing] != [f"attempt_{i:04d}.json" for i in range(1, len(existing) + 1)]:
            raise ValueError("Parallel batch attempt history has missing markers")
        for path in existing:
            marker = entry.read_json(path)
            if marker.get("manifest") != source or marker.get("original_batch_start") != descriptor(start):
                raise ValueError("Parallel batch attempt belongs to a different original start")
        attempt = attempts / f"attempt_{len(existing) + 1:04d}.json"
        _write_exclusive(attempt, {"started_at": entry.now(), "explicit_resume": resume,
            "manifest": source, "original_batch_start": descriptor(start)})
        for path in sorted((output / "physical-stops").glob("*.json")):
            stopped = entry.read_json(path)
            if (stopped.get("manifest") != source or stopped.get("source_freeze") != freeze.descriptor
                    or stopped.get("original_batch_start") != descriptor(start)
                    or stopped.get("physical_dut_key") not in {group["physical_dut_key"] for group in groups}
                    or Path(stopped.get("receipt", "")).resolve() != path.resolve()):
                raise ValueError("Retained physical-resource stop differs from the original batch binding")
            quarantine[stopped["physical_dut_key"]] = stopped
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, lambda *_: cancel.set())
        try:
            manager = PhysicalSetupDispatchManager(max_workers=3, model_ids=manifest["model_ids"])
            futures = {manager.submit_group(group["model_id"], group["physical_dut_key"], work, group): group["index"]
                       for group in groups}
            for model in manifest["model_ids"]:
                manager.finish_registration(model)
            publish()
            while futures:
                if cancel.is_set():
                    for future in futures:
                        future.cancel()
                done, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                while not events.empty():
                    index, delta = events.get()
                    records[index].update(delta)
                for future in done:
                    index = futures.pop(future)
                    records[index].update({"status": "cancelled_before_dispatch"} if future.cancelled() else future.result())
                    with quarantine_lock:
                        if (records[index]["physical_dut_key"] in quarantine
                                and records[index]["status"] in {"completed", "completed_reused"}):
                            records[index]["status"] = "blocked_physical_resource_cleanup"
                publish()
        finally:
            if manager is not None:
                manager.shutdown(wait=True, cancel_futures=cancel.is_set())
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            while not events.empty():
                index, delta = events.get()
                records[index].update(delta)
            publish()
        rc = 0 if (not cancel.is_set() and not freeze.failed and not quarantine
                   and all(row["status"] in {"completed", "completed_reused"} for row in records)) else 1
        entry.write_json(attempt.with_suffix(".terminal.json"), {"finished_at": entry.now(), "exitcode": rc,
            "groups": records, "cancel_requested": cancel.is_set(), "shared_source_failed": freeze.failed})
        return rc
