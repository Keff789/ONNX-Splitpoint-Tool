"""Bounded functional control using the existing Generic completion pipeline.

The caller verifies prepared-file provenance, the unchanged config/source closure,
and process ownership before calling this helper. There is no file loading,
preprocessing, build, fallback, transport, or hardware cancellation here. In
particular, an SDK call which does not return requires the caller's existing
owned-process deadline/lease; coordinator timeouts cannot prove SDK completion.

Each task is ``{task_id, prepared_input, feed}``: prepared_input is an already
verified descriptor (path/sha256/input_name/shape/dtype), and feed maps that same
input_name to its already prepared ndarray. File SHA and raw tensor SHA are
different observations for formats such as NPY; provenance remains caller-owned.
The return value contains owned numpy output/result snapshots for the caller to
persist. Its functional pipeline timing is deliberately not returned as FPS.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import math
import re

import numpy as np

from .task_completion import measure_split_completion, normalize_stage1_execution


class MixedControlError(RuntimeError):
    """A failed control with partial observations, never a throughput result."""

    def __init__(self, message, report):
        super().__init__(message)
        self.partial_report = report


def _same(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right))
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (isinstance(left, Mapping) and isinstance(right, Mapping)
                and left.keys() == right.keys()
                and all(_same(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def _raw_snapshot(outputs):
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError("mixed control requires materialized named output arrays")
    result = {}
    for name, value in outputs.items():
        if (not isinstance(name, str) or not name or not isinstance(value, np.ndarray)
                or value.dtype.kind not in "biuf" or not value.size
                or not np.isfinite(value).all()):
            raise ValueError("mixed control output tensor is invalid or unfinished")
        result[name] = np.array(value, copy=True)
        result[name].flags.writeable = False
    return result


def _fresh_processor(factory):
    processor = factory()
    if (not callable(getattr(processor, "process", None))
            or type(getattr(processor, "completed_count", None)) is not int
            or processor.completed_count != 0):
        raise ValueError("mixed control requires a fresh normal completion processor")
    return processor


def _complete(processor, outputs):
    raw = _raw_snapshot(outputs)  # Own raw tensors before processing/SDK reuse.
    before = processor.completed_count
    processor.process(outputs)
    if type(processor.completed_count) is not int or processor.completed_count != before + 1:
        raise RuntimeError("mixed control completion counter mismatch")
    completed = getattr(processor, "last_result", None)
    if not isinstance(completed, Mapping) or not completed:
        raise RuntimeError("mixed control normal processor has no completed result")
    return {"outputs": raw, "completed_result": deepcopy(completed)}


def _compare(reference, actual, comparator):
    left, right = reference["outputs"], actual["outputs"]
    if (left.keys() != right.keys() or any(left[name].shape != right[name].shape
            or left[name].dtype != right[name].dtype for name in left)):
        return {"output_contract_equal": False}, False
    if comparator is None:
        passed = _same(left, right) and _same(reference["completed_result"], actual["completed_result"])
        return {"raw_outputs_and_completed_result_exact": bool(passed)}, bool(passed)
    # Existing product comparators can be wrapped by the caller for either
    # logits/Top1/Top5 or the existing detection completion contract. No local
    # epsilon, score threshold, or detection tolerance is introduced.
    metrics, passed = comparator(reference, actual)
    if not isinstance(metrics, Mapping) or type(passed) is not bool:
        raise ValueError("mixed control comparator must return (metrics mapping, bool)")
    return deepcopy(dict(metrics)), passed


def _tasks(tasks, config):
    if (not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes))
            or len(tasks) != 32):
        raise ValueError("mixed control requires exactly 32 explicit ordered tasks")
    spec = config.get("input_binding", {}).get("prepared_input") or {}
    name, shape, dtype = spec.get("input_name"), spec.get("shape"), spec.get("dtype")
    if (not isinstance(name, str) or not name or not isinstance(shape, list) or not shape
            or any(type(dim) is not int or dim <= 0 for dim in shape) or not dtype):
        raise ValueError("mixed control requires the unchanged prepared input shape/dtype contract")
    expected_dtype = np.dtype(dtype)
    if expected_dtype.kind not in "biuf":
        raise ValueError("mixed control prepared dtype unsupported")
    identifiers, by_source, frozen = set(), {}, []
    for item in tasks:
        if not isinstance(item, Mapping):
            raise ValueError("mixed control task descriptor malformed")
        task_id, descriptor, feed = item.get("task_id"), item.get("prepared_input"), item.get("feed")
        if not isinstance(task_id, str) or not task_id or task_id in identifiers:
            raise ValueError("mixed control task IDs must be nonempty and unique")
        identifiers.add(task_id)
        if (not isinstance(descriptor, Mapping) or not descriptor.get("path")
                or not isinstance(descriptor.get("sha256"), str)
                or re.fullmatch(r"[a-f0-9]{64}", descriptor["sha256"]) is None
                or any(descriptor.get(key) != spec.get(key) for key in ("input_name", "shape", "dtype"))
                or not isinstance(feed, Mapping) or set(feed) != {name}):
            raise ValueError("mixed control task differs from its verified input descriptor/contract")
        value = feed[name]
        if (not isinstance(value, np.ndarray) or list(value.shape) != shape
                or value.dtype != expected_dtype or not value.flags.c_contiguous
                or not np.isfinite(value).all()):
            raise ValueError("mixed control prepared array shape/dtype/contiguity differs")
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        source_sha = descriptor["sha256"]
        if source_sha in by_source and by_source[source_sha][0] != digest:
            raise ValueError("mixed control one source hash names different tensor payloads")
        if source_sha not in by_source:
            owned = np.array(value, copy=True)
            by_source[source_sha] = digest, owned
        frozen.append({"task_id": task_id, "prepared_input": deepcopy(dict(descriptor)),
            "input_sha256": source_sha, "prepared_tensor_sha256": digest,
            "feed": {name: by_source[source_sha][1]}})
    if len({item[0] for item in by_source.values()}) < 2:
        raise ValueError("mixed control requires at least two distinct actual prepared tensor payloads")
    return frozen


def _slot_recipe(tasks, inflight):
    """The FIFO slot pool must reuse each binding with changed input bytes."""
    groups = [list(range(slot, len(tasks), inflight)) for slot in range(inflight)]
    for slot, indices in enumerate(groups):
        if len({tasks[index]["prepared_tensor_sha256"] for index in indices}) < 2:
            raise ValueError("32-task mixed control requires distinct prepared inputs on every "
                             f"reused stage1 slot; slot={slot}, inflight={inflight}")
    return groups


def _discriminating_slot_references(tasks, references, groups, comparator):
    for slot, indices in enumerate(groups):
        distinct = {}
        for index in indices:
            distinct.setdefault(tasks[index]["prepared_tensor_sha256"], references[index])
        values = list(distinct.values())
        if not any(not _same(first["outputs"], other["outputs"])
                   and not _compare(first, other, comparator)[1]
                   for ordinal, first in enumerate(values) for other in values[ordinal + 1:]):
            raise RuntimeError("mixed control serial references cannot distinguish the prepared "
                               f"inputs on reused stage1 slot {slot}")


def normalize_boundary_observation(value):
    """Validate the explicit, source-bound opt-in; never infer it from a plan."""
    if not isinstance(value, Mapping) or set(value) != {
            "schema_version", "minimum_boundary_tensors", "part1_outputs", "part2_inputs",
            "part2_to_part1", "passthrough_inputs", "transfer_contract"}:
        raise ValueError("mixed boundary observation contract malformed")
    if (type(value["schema_version"]) is not int or value["schema_version"] != 1
            or type(value["minimum_boundary_tensors"]) is not int
            or value["minimum_boundary_tensors"] < 2):
        raise ValueError("mixed boundary observation requires at least two boundary tensors")
    result = deepcopy(dict(value))
    for role in ("part1_outputs", "part2_inputs"):
        specs = value[role]
        if not isinstance(specs, Mapping) or not specs:
            raise ValueError("mixed boundary observation requires complete named tensor contracts")
        for name, spec in specs.items():
            if (not isinstance(name, str) or not name or not isinstance(spec, Mapping)
                    or set(spec) != {"shape", "dtype"} or not isinstance(spec["shape"], list)
                    or not spec["shape"] or any(type(dim) is not int or dim <= 0 for dim in spec["shape"])
                    or not isinstance(spec["dtype"], str)):
                raise ValueError("mixed boundary observation tensor contract malformed")
            try:
                dtype = np.dtype(spec["dtype"])
            except TypeError as error:
                raise ValueError("mixed boundary observation tensor dtype malformed") from error
            if dtype.kind not in "biuf" or not dtype.isnative or str(dtype) != spec["dtype"]:
                raise ValueError("mixed boundary observation tensor dtype unsupported")
    mapping, passthrough = value["part2_to_part1"], value["passthrough_inputs"]
    if (not isinstance(mapping, Mapping) or not isinstance(passthrough, Mapping)
            or set(mapping) & set(passthrough)
            or set(mapping) | set(passthrough) != set(value["part2_inputs"])
            or any(not isinstance(source, str) or source not in value["part1_outputs"] for source in mapping.values())
            or set(mapping.values()) != set(value["part1_outputs"])
            or len(set(mapping.values())) < value["minimum_boundary_tensors"]
            or any(not isinstance(name, str) or not name for name in passthrough.values())
            or not isinstance(value["transfer_contract"], Mapping) or not value["transfer_contract"]):
        raise ValueError("mixed boundary observation mapping/transfer contract malformed")
    return result


def _boundary_snapshot(tensors, specs):
    if not isinstance(tensors, Mapping) or set(tensors) != set(specs):
        raise RuntimeError("mixed boundary actual tensor names differ from the complete contract")
    result = {}
    for name, spec in specs.items():
        value = tensors[name]
        if (not isinstance(value, np.ndarray) or list(value.shape) != spec["shape"]
                or str(value.dtype) != spec["dtype"] or not np.isfinite(value).all()):
            raise RuntimeError("mixed boundary actual tensor shape/dtype/content invalid: " + name)
        result[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
            "tensor_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            "strides": list(value.strides), "c_contiguous": bool(value.flags.c_contiguous),
            "f_contiguous": bool(value.flags.f_contiguous), "owns_data": bool(value.flags.owndata),
            "buffer_address": int(value.ctypes.data), "nbytes": int(value.nbytes)}
    return result


def _boundary_content(snapshot):
    return {name: {key: value[key] for key in ("shape", "dtype", "tensor_sha256")}
            for name, value in snapshot.items()}


def _buffer_arrays(value):
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, Mapping):
        return [array for item in value.values() for array in _buffer_arrays(item)]
    if isinstance(value, (tuple, list)):
        return [array for item in value for array in _buffer_arrays(item)]
    return []


def _buffer_descriptor(value):
    address = int(value.ctypes.data)
    offsets = [(int(dim) - 1) * int(stride) for dim, stride in zip(value.shape, value.strides)]
    return {"shape": list(value.shape), "dtype": str(value.dtype), "strides": list(value.strides),
        "buffer_address": address, "nbytes": int(value.nbytes),
        "byte_span": [address + sum(min(0, offset) for offset in offsets),
                      address + sum(max(0, offset) for offset in offsets) + value.dtype.itemsize]}


class _BoundaryObservation:
    """Hash actual maps only inside the bounded pre-FPS functional control."""

    def __init__(self, contract, runtime, tasks):
        self.contract = normalize_boundary_observation(contract)
        if not callable(runtime):
            raise ValueError("mixed boundary observation requires the live runtime mapping/IO callback")
        self.runtime, self.tasks = runtime, tasks
        self.records = {"serial": {}, "mixed": {}}
        self.sdk_buffers = []
        self.slot_ownership = None
        self.first_runtime = None

    def bind_adapter(self, adapter):
        slots = getattr(adapter, "slots", None)
        if not isinstance(slots, list) or len(slots) != adapter.execution_policy["inflight"]:
            raise ValueError("mixed boundary observation cannot inspect the real adapter slots")
        groups = []
        for slot in slots:
            # Inspect production-owned arrays; SDK binding objects are opaque.
            arrays = _buffer_arrays({key: slot[key] for key in (
                "buffer", "outputs", "inputs", "output_buffers", "input_buffers", "binding") if key in slot})
            unique = {id(value): value for value in arrays}
            if not unique:
                raise ValueError("mixed boundary observation adapter slot has no inspectable buffers")
            groups.append(list(unique.values()))
        for ordinal, arrays in enumerate(groups):
            if any(np.shares_memory(first, second) for first in arrays
                   for other in groups[ordinal + 1:] for second in other):
                raise RuntimeError("mixed boundary observation accelerator slots share buffers")
        self.sdk_buffers = [value for group in groups for value in group]
        self.slot_ownership = {"slot_count": len(groups), "inspected_buffer_count": len(self.sdk_buffers),
            "slots_disjoint": True, "slots": [{"slot_index": index,
                "buffers": [_buffer_descriptor(array) for array in arrays]}
                for index, arrays in enumerate(groups)]}

    def collected(self, phase, index, outputs):
        if index in self.records[phase]:
            raise RuntimeError("mixed boundary observation duplicate task collection")
        snapshot = _boundary_snapshot(outputs, self.contract["part1_outputs"])
        if any(np.shares_memory(value, owned)
                for value in outputs.values() for owned in self.sdk_buffers):
            raise RuntimeError("mixed boundary output still aliases a reusable SDK slot")
        if any(np.shares_memory(value, prepared) for value in outputs.values()
                for prepared in self.tasks[index]["feed"].values()):
            raise RuntimeError("mixed boundary output aliases its prepared image")
        self.records[phase][index] = {"part1_at_collection": snapshot,
            "no_reusable_sdk_buffer_alias": True, "no_prepared_input_alias": True}

    def transferred(self, phase, index, outputs, transfer, feed):
        record = self.records[phase][index]
        before = _boundary_snapshot(outputs, self.contract["part1_outputs"])
        record["part1_before_transfer"] = before
        if _boundary_content(before) != _boundary_content(record["part1_at_collection"]):
            raise RuntimeError("mixed boundary P1 data/buffer changed while queued")
        inputs = transfer(outputs, feed)
        record["part1_after_transfer"] = _boundary_snapshot(outputs, self.contract["part1_outputs"])
        if record["part1_after_transfer"] != before:
            raise RuntimeError("mixed boundary transfer mutated its P1 input")
        record["part2_after_transfer"] = _boundary_snapshot(inputs, self.contract["part2_inputs"])
        if any(np.shares_memory(value, owned) for value in inputs.values() for owned in self.sdk_buffers):
            raise RuntimeError("mixed boundary P2 input aliases a reusable SDK slot")
        runtime = self.runtime()
        if (not isinstance(runtime, Mapping) or set(runtime) != {
                "part1_io", "part2_io", "part2_to_part1", "passthrough_inputs", "transfer_contract"}
                or any(runtime.get(key) != self.contract[key] for key in (
                    "part2_to_part1", "passthrough_inputs", "transfer_contract"))
                or not isinstance(runtime["part1_io"], Mapping) or not runtime["part1_io"]
                or not isinstance(runtime["part2_io"], Mapping)
                or runtime["part2_io"].get("inputs") != self.contract["part2_inputs"]
                or set(runtime["part1_io"].get("canonical_outputs", [])) != set(outputs)):
            raise RuntimeError("mixed boundary live mapping/IO/layout/quantization differs from contract")
        if any(source not in feed for source in runtime["passthrough_inputs"].values()):
            raise RuntimeError("mixed boundary passthrough is not an actual original input")
        if self.first_runtime is None:
            self.first_runtime = deepcopy(dict(runtime))
        elif runtime != self.first_runtime:
            raise RuntimeError("mixed boundary live runtime metadata changed during the control")
        record["runtime"] = deepcopy(dict(runtime))
        record["part2_to_part1_aliases"] = {name: [source for source, value in outputs.items()
            if np.shares_memory(inputs[name], value)] for name in inputs}
        return inputs

    def inferred(self, phase, index, inputs, stage2, feed):
        record = self.records[phase][index]
        before = _boundary_snapshot(inputs, self.contract["part2_inputs"])
        record["part2_before_inference"] = before
        if before != record["part2_after_transfer"]:
            raise RuntimeError("mixed boundary P2 input changed before inference")
        outputs = stage2(inputs, feed)
        record["part2_after_inference"] = _boundary_snapshot(inputs, self.contract["part2_inputs"])
        if record["part2_after_inference"] != before:
            raise RuntimeError("mixed boundary P2 mutated its input")
        return outputs

    def compared(self, index):
        serial, mixed = self.records["serial"][index], self.records["mixed"][index]
        for name in ("part1_at_collection", "part2_before_inference"):
            if _boundary_content(serial[name]) != _boundary_content(mixed[name]):
                raise RuntimeError("mixed boundary serial/async tensor content differs for task "
                                   + self.tasks[index]["task_id"])
        if serial["runtime"] != mixed["runtime"]:
            raise RuntimeError("mixed boundary runtime mapping/IO changed between serial and async")
        return {"serial": serial, "mixed": mixed, "passed": True}

    def summary(self):
        return {"schema_version": 1, "contract": deepcopy(self.contract),
            "observed_task_count": len(self.records["mixed"]),
            "observed_boundary_tensor_count": len(set(self.first_runtime["part2_to_part1"].values())),
            "all_boundary_tensors_equal": True, "unchanged_until_inference": True,
            "slot_buffer_ownership": deepcopy(self.slot_ownership)}


class _AssociatedAsync:
    """Attach the declared task to existing SDK handles without scheduling jobs."""

    def __init__(self, adapter, tasks, prepared, boundary=None):
        self.adapter, self.tasks, self.prepared = adapter, tasks, prepared
        self.execution_policy = adapter.execution_policy
        self.cleanup_timeout_s = adapter.cleanup_timeout_s
        self.pending = {}
        self.submitted = []
        self.collected = []
        self.slot_tasks = {}
        self.drain_started = self.drain_completed = False
        self.boundary = boundary

    def submit(self):
        index = self.prepared["next"]
        handle = self.adapter.submit()
        if self.prepared["next"] != index + 1:
            raise RuntimeError("mixed control adapter must prepare exactly one input per submission")
        if id(handle) in self.pending:
            raise RuntimeError("mixed control adapter returned a duplicate active handle")
        token = {"handle": handle, "task_index": index}
        self.pending[id(handle)] = token
        self.slot_tasks.setdefault(id(handle), []).append(index)
        self.submitted.append(index)
        return token

    def collect(self, token):
        handle, index = token["handle"], token["task_index"]
        if self.pending.get(id(handle)) is not token or index != len(self.collected):
            raise RuntimeError("mixed control collected a lost, duplicate, or reordered task")
        outputs = self.adapter.collect(handle)
        if self.boundary is not None:
            self.boundary.collected("mixed", index, outputs)
        self.pending.pop(id(handle))
        self.collected.append(index)
        # Preserve the production adapter's payload ownership behavior. We do
        # not clone P1 here: returning a reused P1 buffer must remain detectable
        # after it passes through the real FIFO/consumer.
        return {"task_index": index, "outputs": outputs}

    def drain(self):
        self.drain_started = True
        self.adapter.drain()
        if self.adapter.pending:
            raise RuntimeError("mixed control adapter still owns undrained jobs")
        self.pending.clear()
        self.drain_completed = True

    def retain_unresolved(self):
        retain = getattr(self.adapter, "retain_unresolved", None)
        if callable(retain):
            retain()


def run_mixed_control(tasks, *, config, original_sources, serial_stage1, transfer,
                      stage2, processor_factory, stage1_async_factory=None,
                      compare=None, clone_payload=None, boundary_observation=None,
                      boundary_runtime=None):
    """Compare a finite mixed sequence against serial compiled-subnet outputs.

    Stage callbacks are the initialized Generic runtime functions:
    serial_stage1(feed), transfer(boundary, feed), stage2(inputs, feed).
    An async factory receives only prepare_inputs and must return the exact
    existing policy-bound adapter; it does not replace the P2/processor path.
    ``compare`` optionally wraps an existing product comparison and returns
    ``(metrics, passed)`` from two records containing outputs/completed_result.
    Missing comparison means exact array and normal completion-result equality.
    clone_payload, when supplied, is the unchanged normal P1 tensor-map copy
    callback. Input snapshots are owned and writable for SDK compatibility;
    their raw hashes are rechecked after each phase to reject mutation.
    Every FIFO binding slot must receive distinguishable input payloads on
    reuse. The fixed sequence i%2 for tasks 0..15, then inverted i%2 for 16..31,
    covers the prescribed 4/8-slot policies; larger pools may not fit a valid
    per-slot reuse control in the authorized 32-task budget.
    An explicit boundary_observation plus boundary_runtime callback additionally
    records every actual P1/P2 tensor, live mapping and SDK buffer ownership.
    It compares transfer bytes with the compiled serial path; it does not
    reimplement layout or dequantization arithmetic. Absence preserves the
    original final-output-only control and makes no multi-tensor proof claim.
    """
    if not isinstance(config, Mapping) or not isinstance(original_sources, Mapping) or not original_sources:
        raise ValueError("mixed control requires the bound config and original source descriptors")
    execution = normalize_stage1_execution(config.get("stage1_execution"))
    depth, timeout = config.get("queue_depth"), config.get("timeout_s")
    if (type(depth) is not int or depth < 1 or type(timeout) not in (int, float)
            or not math.isfinite(timeout) or timeout <= 0 or config.get("energy_capture")):
        raise ValueError("mixed control requires the unchanged finite throughput config")
    if any(not callable(fn) for fn in (serial_stage1, transfer, stage2, processor_factory)):
        raise ValueError("mixed control requires existing compiled-runtime/completion callbacks")
    asynchronous = execution["policy"] != "synchronous"
    if asynchronous != callable(stage1_async_factory):
        raise ValueError("mixed control async factory must match the prepared policy; no fallback")
    if compare is not None and not callable(compare):
        raise ValueError("mixed control comparator is not callable")
    if clone_payload is not None and not callable(clone_payload):
        raise ValueError("mixed control payload copy callback is not callable")
    frozen = _tasks(tasks, config)
    if (boundary_observation is None) != (boundary_runtime is None):
        raise ValueError("mixed boundary observation contract/runtime callback must be explicit together")
    boundary = (_BoundaryObservation(boundary_observation, boundary_runtime, frozen)
                if boundary_observation is not None else None)
    slot_groups = _slot_recipe(frozen, execution["inflight"])
    reference_processor, candidate_processor = _fresh_processor(processor_factory), _fresh_processor(processor_factory)
    if reference_processor is candidate_processor:
        raise ValueError("mixed control processor factory reused one mutable processor")
    report = {"status": "started", "purpose": "generic_mixed_input_functional_control",
        "checked_outside_throughput_timer": True, "scientific_statistics_published": False,
        "stage1_execution": execution, "queue_depth": depth, "timeout_s": timeout,
        "handoff_payload_copy": clone_payload is not None,
        "requested_task_count": len(frozen), "distinct_prepared_tensor_count": len({
            item["prepared_tensor_sha256"] for item in frozen}),
        "original_sources": deepcopy(dict(original_sources)),
        "runtime_binding": {key: deepcopy(config[key]) for key in (
            "identity", "execution_id", "original_run", "expected_runtime_artifacts",
            "runtime_source_files", "execution_environment") if key in config},
        "serial_reference_count": 0, "mixed_completed_count": 0,
        "drain_completed": False, "observations": []}
    references, candidates = [], []
    prepared = {"next": 0}
    associated = None

    def prepare_inputs():
        index = prepared["next"]
        if index >= len(frozen):
            raise RuntimeError("mixed control submitted more tasks than declared")
        prepared["next"] += 1
        return frozen[index]["feed"]

    def serial_producer():
        index = prepared["next"]
        feed = prepare_inputs()
        outputs = serial_stage1(feed)
        if boundary is not None:
            boundary.collected("mixed", index, outputs)
        return {"task_index": index, "outputs": outputs}

    def pipeline_transfer(envelope):
        if (not isinstance(envelope, Mapping) or set(envelope) != {"task_index", "outputs"}
                or type(envelope["task_index"]) is not int
                or envelope["task_index"] != len(candidates)):
            raise RuntimeError("mixed control FIFO task association mismatch")
        index = envelope["task_index"]
        inputs = (boundary.transferred("mixed", index, envelope["outputs"], transfer, frozen[index]["feed"])
                  if boundary is not None else transfer(envelope["outputs"], frozen[index]["feed"]))
        return {"task_index": index, "inputs": inputs}

    current = {"task_index": None}

    def pipeline_stage2(envelope):
        index = envelope["task_index"]
        current["task_index"] = index
        return (boundary.inferred("mixed", index, envelope["inputs"], stage2, frozen[index]["feed"])
                if boundary is not None else stage2(envelope["inputs"], frozen[index]["feed"]))

    def copy_envelope(envelope):
        return {"task_index": envelope["task_index"], "outputs": clone_payload(envelope["outputs"])}

    def check_input_bytes():
        for item in frozen:
            value = next(iter(item["feed"].values()))
            if (list(value.shape) != item["prepared_input"]["shape"]
                    or str(value.dtype) != item["prepared_input"]["dtype"]
                    or hashlib.sha256(value.tobytes()).hexdigest() != item["prepared_tensor_sha256"]):
                raise RuntimeError("mixed control runtime mutated the prepared input")

    class Recorder:
        completed_count = 0

        def process(self, outputs):
            index = current["task_index"]
            if index != self.completed_count or index != len(candidates):
                raise RuntimeError("mixed control completion task order mismatch")
            candidates.append(_complete(candidate_processor, outputs))
            self.completed_count += 1
            report["mixed_completed_count"] = self.completed_count

    try:
        if asynchronous:
            adapter = stage1_async_factory(prepare_inputs)
            if (normalize_stage1_execution(adapter.execution_policy) != execution
                    or not isinstance(getattr(adapter, "pending", None), Mapping)
                    or adapter.pending or prepared["next"] != 0):
                raise ValueError("mixed control adapter differs from prepared policy or already owns work")
            if boundary is not None:
                boundary.bind_adapter(adapter)
            associated = _AssociatedAsync(adapter, frozen, prepared, boundary)
        for index, item in enumerate(frozen):
            feed = item["feed"]
            if boundary is None:
                outputs = stage2(transfer(serial_stage1(feed), feed), feed)
            else:
                p1 = serial_stage1(feed)
                boundary.collected("serial", index, p1)
                p2 = boundary.transferred("serial", index, p1, transfer, feed)
                outputs = boundary.inferred("serial", index, p2, stage2, feed)
            references.append(_complete(reference_processor, outputs))
            report["serial_reference_count"] = len(references)
        check_input_bytes()
        # Every reused slot must see an observable cross-input change, not
        # merely a different host task ID or detection invocation counter.
        _discriminating_slot_references(frozen, references, slot_groups, compare)
        pipeline = measure_split_completion(serial_producer, pipeline_transfer, pipeline_stage2,
            Recorder(), warmup=0, tasks=len(frozen), queue_depth=depth,
            timeout_s=float(timeout), stage1_async=associated,
            clone_payload=copy_envelope if clone_payload is not None else None)
        check_input_bytes()
        report["drain_completed"] = associated.drain_completed if associated is not None else True
        if (prepared["next"] != len(frozen) or len(candidates) != len(frozen)
                or pipeline["completed_task_count"] != len(frozen)
                or pipeline["processor_count_delta"] != len(frozen)
                or pipeline["stage1_execution"] != execution
                or not report["drain_completed"]
                or (associated is not None and (associated.pending or associated.adapter.pending
                    or associated.submitted != list(range(len(frozen)))
                    or associated.collected != associated.submitted))):
            raise RuntimeError("mixed control task counts, policy, or full drain differ")
        report["stage1_peak_uncollected_jobs"] = pipeline.get("stage1_peak_uncollected_jobs", 1)
        observed_groups = list(associated.slot_tasks.values()) if associated is not None else slot_groups
        if observed_groups != slot_groups:
            raise RuntimeError("mixed control observed adapter slot reuse differs from its FIFO recipe")
        report["stage1_slot_input_coverage"] = [{"slot_index": slot,
            "task_ids": [frozen[index]["task_id"] for index in indices],
            "prepared_tensor_sha256": sorted({frozen[index]["prepared_tensor_sha256"] for index in indices}),
            "distinct_prepared_tensor_count": len({frozen[index]["prepared_tensor_sha256"] for index in indices}),
            "serial_outputs_discriminating": True}
            for slot, indices in enumerate(observed_groups)]
        for index, (item, reference, actual) in enumerate(zip(frozen, references, candidates)):
            metrics, passed = _compare(reference, actual, compare)
            record = {key: deepcopy(item[key]) for key in (
                "task_id", "prepared_input", "input_sha256", "prepared_tensor_sha256")}
            record.update(serial=reference, mixed=actual, metrics=metrics, passed=passed)
            report["observations"].append(record)
            if boundary is not None:
                record["boundary_observation"] = boundary.compared(index)
            if not passed:
                raise RuntimeError("mixed control output/result mismatch for task " + item["task_id"])
        report.update(status="passed", exact_task_order=True, all_declared_tasks_completed=True,
                      comparison="exact_outputs_and_completed_results" if compare is None else "existing_product_comparator")
        if boundary is not None:
            report["boundary_observation"] = boundary.summary()
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        if associated is not None:
            report["drain_completed"] = associated.drain_completed
        # Preserve all observed completed snapshots, including after a compare
        # failure. Never expose the diagnostic pipeline's timer/rate as a result.
        report["serial_references"] = references
        report["mixed_results"] = candidates
        if boundary is not None:
            report["boundary_observation_partial"] = boundary.records
        raise MixedControlError(report["error"], report) from error
