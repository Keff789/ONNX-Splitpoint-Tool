"""Optional bounded accelerator admission for Generic completed-task runs.

This adapter changes accelerator admission only. The Generic FIFO, transfer,
TensorRT consumer and completion processor remain in their usual code paths.
"""
from __future__ import annotations

from collections import deque
import math
from collections.abc import Mapping

import numpy as np

from .task_completion import normalize_stage1_execution


class HailoAsyncStage1:
    def __init__(self, session, prepare_inputs, *, inflight):
        self.execution_policy = normalize_stage1_execution(
            {"policy": "hailo_async_slots_v1", "inflight": inflight})
        self.session = session
        self.prepare_inputs = prepare_inputs
        for name in ("_create_reusable_binding_slot", "_prepare_infer_inputs",
                     "_submit_reusable_slot", "_wait_reusable_slot", "_materialize_binding_outputs"):
            if not callable(getattr(session, name, None)):
                raise ValueError("Generic async P1 requires the Hailo InferModel slot API")
        if type(session.timeout_ms) is not int or session.timeout_ms <= 0:
            raise ValueError("Generic async P1 requires a bounded driver timeout")
        # A failing collection may use a job wait and callback wait, followed
        # by the same bounded waits for every pending slot during cleanup.
        self.cleanup_timeout_s = 2.0 * (inflight + 1) * session.timeout_ms / 1000.0 + 2.0
        self.slots = [session._create_reusable_binding_slot() for _ in range(inflight)]
        self.available = deque(self.slots)
        self.pending = {}

    def submit(self):
        if not self.available:
            raise RuntimeError("Generic async P1 exceeded its slot budget")
        slot = self.available.popleft()
        # Track ownership before submission so an exception after admission
        # still leaves the binding reachable by drain(). No frozen input is
        # substituted for the prepared input adapter inside the timed phase.
        self.pending[id(slot)] = slot
        inputs = self.session._prepare_infer_inputs(self.prepare_inputs())
        self.session._submit_reusable_slot(slot, inputs, copy_inputs=True)
        return slot

    def _wait(self, slot):
        self.session._wait_reusable_slot(slot)
        # The shared helper clears its job on a callback timeout. Checking the
        # event again prevents a subsequent cleanup wait from silently treating
        # that still-unfinished binding as reusable.
        done = slot.get("done")
        if done is None or not done.is_set():
            raise TimeoutError("Generic async P1 callback remains unfinished")

    def collect(self, slot):
        if self.pending.get(id(slot)) is not slot:
            raise RuntimeError("Generic async P1 received an unowned slot")
        self._wait(slot)
        outputs = self.session._materialize_binding_outputs(slot["binding"], copy_outputs=True)
        self.pending.pop(id(slot))
        self.available.append(slot)
        return outputs

    def drain(self):
        """Wait every submitted binding, also after another slot has failed."""
        errors = []
        for key, slot in list(self.pending.items()):
            try:
                self._wait(slot)
                self.pending.pop(key)
                self.available.append(slot)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError("Generic async P1 failed to drain all submitted slots") from errors[0]


# A failed DX-RT submission may have exposed buffers without returning a job ID.
# A failed wait also does not prove completion. Never release those owners by GC.
_UNRESOLVED_DEEPX_OWNERS = []


def _deepx_contracts(engine, role):
    getter = getattr(engine, f"get_{role}_tensors_info", None)
    if not callable(getter):
        raise ValueError(f"Generic DeepX async requires {role} SDK tensor metadata")
    info = getter()
    if not isinstance(info, (list, tuple)) or not info:
        raise ValueError(f"Generic DeepX async {role} tensor metadata missing")
    result = []
    for item in info:
        if not isinstance(item, Mapping):
            raise ValueError("Generic DeepX async tensor metadata malformed")
        name, shape = item.get("name"), item.get("shape")
        if (not isinstance(name, str) or not name or not isinstance(shape, (tuple, list)) or not shape
                or any(isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer))
                       or dim <= 0 for dim in shape) or item.get("dtype") is None):
            raise ValueError("Generic DeepX async tensor name/shape/dtype invalid")
        dtype = np.dtype(item["dtype"])
        if dtype.kind not in "biuf" or not dtype.isnative or item.get("elem_size", dtype.itemsize) != dtype.itemsize:
            raise ValueError("Generic DeepX async tensor dtype unsupported")
        shape = tuple(int(dim) for dim in shape)
        result.append({"name": name, "shape": shape, "dtype": dtype,
                       "nbytes": math.prod(shape) * dtype.itemsize})
    names = [item["name"] for item in result]
    if len(set(names)) != len(names):
        raise ValueError("Generic DeepX async tensor names duplicate")
    for suffix, expected in (("tensor_names", names), ("tensor_sizes", [item["nbytes"] for item in result]),
                             ("tensor_count", len(result)), ("size", sum(item["nbytes"] for item in result))):
        method = getattr(engine, f"get_{role}_{suffix}", None)
        if callable(method):
            actual = method()
            if (list(actual) if isinstance(expected, list) else actual) != expected:
                raise ValueError("Generic DeepX async tensor metadata inconsistent")
    return result


class DeepXAsyncStage1:
    """Use DX-RT's run_async/wait API with independent, owned output slabs.

    The observed SDK API has no wait timeout or cancellation argument. The
    completion coordinator can fail a global deadline, but an unfinished SDK
    call remains owned until the existing process lease terminates that worker.
    No background pool of synchronous engine.run calls or fallback is used.
    """
    cleanup_timeout_s = 1.0  # Coordinator join grace, not a claimed SDK timeout.

    def __init__(self, session, prepare_inputs, *, inflight):
        self.execution_policy = normalize_stage1_execution(
            {"policy": "deepx_async_slots_v1", "inflight": inflight})
        self.session, self.engine, self.prepare_inputs = session, session._engine, prepare_inputs
        if any(not callable(getattr(self.engine, name, None)) for name in ("run_async", "wait")):
            raise ValueError("Generic DeepX async requires run_async and wait; no synchronous fallback")
        self.input_contracts = _deepx_contracts(self.engine, "input")
        self.output_contracts = _deepx_contracts(self.engine, "output")
        if any(len(item["shape"]) >= 4 and item["shape"][0] != 1 for item in self.input_contracts):
            raise ValueError("Generic DeepX async requires batch one")
        for name in ("_prepare_async_inputs", "_materialize_outputs"):
            if not callable(getattr(session, name, None)):
                raise ValueError("Generic DeepX async requires the exact Generic input/output adapter")
        self.slots = []
        for _ in range(inflight):
            slab = np.empty(sum(item["nbytes"] for item in self.output_contracts), dtype=np.uint8)
            offset, views = 0, []
            for item in self.output_contracts:
                views.append(np.ndarray(item["shape"], dtype=item["dtype"], buffer=slab, offset=offset))
                offset += item["nbytes"]
            self.slots.append({"buffer": slab, "outputs": views,
                "inputs": [np.empty(item["shape"], dtype=item["dtype"]) for item in self.input_contracts],
                "state": "free", "job_id": None})
        self.available = deque(self.slots)
        self.pending = {}
        self.request_id = 0
        self.retained = False

    def _retain(self):
        if not self.retained:
            _UNRESOLVED_DEEPX_OWNERS.append(self)
            self.retained = True

    def retain_unresolved(self):
        """Keep the engine and buffers alive after a coordinator deadline."""
        self._retain()

    def submit(self):
        if self.retained:
            raise RuntimeError("Generic DeepX async completion remains unknown")
        if not self.available:
            raise RuntimeError("Generic DeepX async exceeded its slot budget")
        arrays = self.session._prepare_async_inputs(self.prepare_inputs())
        if not isinstance(arrays, (list, tuple)) or len(arrays) != len(self.input_contracts):
            raise ValueError("Generic DeepX async exact input count mismatch")
        checked = []
        for value, contract in zip(arrays, self.input_contracts):
            if isinstance(value, np.ndarray) and contract["shape"] == (1,) + value.shape:
                value = value[None]  # Only an explicit missing leading batch-one axis.
            if (not isinstance(value, np.ndarray) or value.shape != contract["shape"]
                    or value.dtype != contract["dtype"] or not value.flags.c_contiguous):
                raise ValueError("Generic DeepX async exact input shape/dtype/contiguity mismatch")
            checked.append(value)
        slot = self.available.popleft()
        for source, owned in zip(checked, slot["inputs"]):
            np.copyto(owned, source, casting="no")
        self.pending[id(slot)] = slot
        slot["state"] = "submitting"
        try:
            job_id = self.engine.run_async(slot["inputs"], user_arg=self.request_id,
                                           output_buffer=slot["buffer"])
            self.request_id += 1
            if type(job_id) is not int or job_id < 0 or any(
                    other is not slot and other["job_id"] == job_id for other in self.pending.values()):
                raise ValueError("Generic DeepX async job ID invalid or duplicate")
        except BaseException:
            slot["state"] = "submit_unresolved"
            self._retain()
            raise
        slot["job_id"], slot["state"] = job_id, "submitted"
        return slot

    def _wait(self, slot):
        if slot["state"] != "submitted":
            raise RuntimeError("Generic DeepX async cannot wait an unresolved submission twice")
        slot["state"] = "waiting"
        try:
            outputs = self.engine.wait(slot["job_id"])
        except BaseException:
            slot["state"] = "wait_unresolved"
            self._retain()
            raise
        slot["state"] = "ready"
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(self.output_contracts):
            raise RuntimeError("Generic DeepX async output count mismatch")
        for value, owned, contract in zip(outputs, slot["outputs"], self.output_contracts):
            if (not isinstance(value, np.ndarray) or value.shape != contract["shape"]
                    or value.dtype != contract["dtype"] or not value.flags.c_contiguous
                    or value.ctypes.data != owned.ctypes.data or value.nbytes != owned.nbytes):
                raise RuntimeError("Generic DeepX async output does not alias its owned slot")
        return outputs

    def _release(self, slot):
        if slot["state"] != "ready":
            raise RuntimeError("Generic DeepX async release before confirmed completion")
        self.pending.pop(id(slot))
        slot["job_id"], slot["state"] = None, "free"
        self.available.append(slot)

    def collect(self, slot):
        if self.pending.get(id(slot)) is not slot:
            raise RuntimeError("Generic DeepX async received an unowned slot")
        outputs = self._wait(slot)
        materialized = self.session._materialize_outputs([np.array(value, copy=True) for value in outputs])
        self._release(slot)
        return materialized

    def drain(self):
        errors = []
        for slot in list(self.pending.values()):
            try:
                if slot["state"] == "submitted":
                    self._wait(slot)
                if slot["state"] != "ready":
                    self._retain()
                    raise RuntimeError("Generic DeepX async completion remains unresolved")
            except BaseException as error:
                errors.append(error)
            finally:
                if slot["state"] == "ready":
                    self._release(slot)
        if errors:
            raise RuntimeError("Generic DeepX async failed to drain all submitted slots") from errors[0]
