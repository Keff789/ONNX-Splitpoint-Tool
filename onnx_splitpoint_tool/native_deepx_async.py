"""The fixed Native DeepX A4 submit/wait and output ownership policy.

This module imports no SDK and creates no engine.  The caller supplies its one
resident engine, prepares inputs before timing, and supplies the existing task
completion functions.  Generic and Hailo runtimes do not use this module.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

if __package__ == "splitpoint_runners":
    from .request_latency import RequestLatency
else:
    from .runners.request_latency import RequestLatency


INFLIGHT_LIMIT = 4
FIFO_DEPTH = 3


class NativeExecutionLifecycle:
    """Small Native worker evidence, recorded at actual runtime call entry.

    A returned asynchronous job is unfinished until its wait returns.  Failed
    constructors/calls therefore leave explicit unknown completion evidence.
    This is observational data, not an SDK cancellation or release operation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = {
            "phase": "pre_runtime_guard", "completion": "not_started",
            "runtime_construction_started": False,
            "runtime_construction_completed": False, "inference_started": False,
            "runtime_constructors_entered": 0, "runtime_constructors_returned": 0,
            "runtime_calls_entered": 0, "runtime_calls_completed": 0,
        }

    def construction_enter(self):
        with self._lock:
            self._state.update(phase="runtime_construction", completion="unknown",
                               runtime_construction_started=True,
                               runtime_construction_completed=False)
            self._state["runtime_constructors_entered"] += 1

    def construction_return(self):
        with self._lock:
            self._state["runtime_constructors_returned"] += 1
            self._state["runtime_construction_completed"] = (
                self._state["runtime_constructors_entered"] == self._state["runtime_constructors_returned"])

    def inference_enter(self):
        with self._lock:
            self._state.update(phase="inference", completion="unknown", inference_started=True)
            self._state["runtime_calls_entered"] += 1

    def inference_return(self):
        with self._lock:
            self._state["runtime_calls_completed"] += 1
            if (self._state["runtime_construction_completed"] is True
                    and self._state["runtime_calls_completed"] == self._state["runtime_calls_entered"]):
                self._state.update(phase="drained", completion="drained")

    def inference_not_entered(self):
        """Undo admission bookkeeping when the final deadline denies a call."""
        with self._lock:
            self._state["runtime_calls_entered"] -= 1
            if self._state["runtime_calls_entered"] == 0:
                self._state.update(inference_started=False, phase="runtime_construction", completion="unknown")
            elif self._state["runtime_calls_entered"] == self._state["runtime_calls_completed"]:
                self._state.update(phase="drained", completion="drained")

    def report(self):
        with self._lock:
            return dict(self._state)


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


def read_tensor_contracts(engine: Any, role: str) -> tuple[TensorContract, ...]:
    """Bind SDK tensor order, names, exact shapes, types and byte sizes once."""
    if role not in {"input", "output"}:
        raise ValueError("deepx_tensor_role_invalid")
    getter = getattr(engine, f"get_{role}_tensors_info", None)
    if not callable(getter):
        raise ValueError(f"deepx_{role}_sdk_tensor_metadata_missing")
    info = getter()
    if not isinstance(info, (list, tuple)) or not info:
        raise ValueError(f"deepx_{role}_sdk_tensor_metadata_invalid")
    contracts: list[TensorContract] = []
    for item in info:
        if not isinstance(item, Mapping):
            raise ValueError(f"deepx_{role}_sdk_tensor_metadata_invalid")
        name, shape = item.get("name"), item.get("shape")
        if (not isinstance(name, str) or not name
                or not isinstance(shape, (list, tuple)) or not shape
                or any(isinstance(d, (bool, np.bool_))
                       or not isinstance(d, (int, np.integer)) or d <= 0 for d in shape)):
            raise ValueError(f"deepx_{role}_sdk_tensor_name_shape_invalid")
        if item.get("dtype") is None:
            raise ValueError(f"deepx_{role}_sdk_tensor_dtype_invalid")
        try:
            dtype = np.dtype(item["dtype"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"deepx_{role}_sdk_tensor_dtype_invalid") from exc
        if dtype.kind not in "biuf" or not dtype.isnative:
            raise ValueError(f"deepx_{role}_sdk_tensor_dtype_unsupported")
        if "elem_size" in item and item["elem_size"] != dtype.itemsize:
            raise ValueError(f"deepx_{role}_sdk_tensor_element_size_mismatch")
        contracts.append(TensorContract(name, tuple(int(d) for d in shape), dtype))
    names = [item.name for item in contracts]
    if len(set(names)) != len(names):
        raise ValueError(f"deepx_{role}_sdk_tensor_names_duplicate")
    for suffix, expected in (
        ("tensor_names", names),
        ("tensor_sizes", [item.nbytes for item in contracts]),
    ):
        method = getattr(engine, f"get_{role}_{suffix}", None)
        if callable(method) and list(method()) != expected:
            raise ValueError(f"deepx_{role}_sdk_{suffix}_mismatch")
    for suffix, expected in (
        ("tensor_count", len(contracts)),
        ("size", sum(item.nbytes for item in contracts)),
    ):
        method = getattr(engine, f"get_{role}_{suffix}", None)
        if callable(method) and method() != expected:
            raise ValueError(f"deepx_{role}_sdk_{suffix}_mismatch")
    return tuple(contracts)


@dataclass
class _Slot:
    buffer: np.ndarray
    outputs: list[np.ndarray]
    state: str = "free"
    request_id: int | None = None
    job_id: int | None = None
    inputs: list[np.ndarray] | None = None


# A failed SDK call can have admitted work without returning a usable job ID.
# Retain those buffers until process exit.  There is deliberately no "force
# release" API that could declare an unknown native completion safe.
_UNRESOLVED_OWNERS: list[Any] = []


class NativeDeepXAsync:
    """One supplied engine, four independent multioutput slots, ordered waits.

    SDK defaults (including buffer_count, use_ort and bound options) are never
    readjusted.  ``prepare_input`` is a setup operation; ``submit`` accepts only
    the resulting exact SDK tensor contract, with no cast or layout adaptation.
    A successful wait holds its slot until the caller has copied and released it.
    """

    def __init__(self, engine: Any, *, input_names: Any = None,
                 output_names: Any = None, input_templates: Any = None,
                 output_templates: Any = None, execution_lifecycle: Any = None):
        if not callable(getattr(engine, "run_async", None)) or not callable(getattr(engine, "wait", None)):
            raise ValueError("deepx_async_api_missing")
        self.engine = engine
        self.execution_lifecycle = execution_lifecycle
        self.input_contracts = read_tensor_contracts(engine, "input")
        self.output_contracts = read_tensor_contracts(engine, "output")
        if any(len(item.shape) >= 4 and item.shape[0] != 1 for item in self.input_contracts):
            raise ValueError("deepx_native_batch_one_required")
        self.input_names = tuple(item.name for item in self.input_contracts)
        self.output_names = tuple(item.name for item in self.output_contracts)
        for role, names, templates, contracts in (
            ("input", input_names, input_templates, self.input_contracts),
            ("output", output_names, output_templates, self.output_contracts),
        ):
            if names is not None and list(names) != [item.name for item in contracts]:
                raise ValueError(f"deepx_{role}_expected_name_order_mismatch")
            if templates is not None:
                if len(templates) != len(contracts):
                    raise ValueError(f"deepx_{role}_expected_count_mismatch")
                for value, contract in zip(templates, contracts):
                    self._validate_array(value, contract, role)
        offsets = []
        total_bytes = 0
        for item in self.output_contracts:
            offsets.append(total_bytes)
            total_bytes += item.nbytes
        self.output_offsets = tuple(offsets)
        self.slots = []
        for _ in range(INFLIGHT_LIMIT):
            # DX-RT takes one output base pointer even when the Python wrapper
            # accepts a list.  Native tensor offsets are cumulative SDK tensor
            # byte sizes in SDK output order, without padding.  Independent
            # per-head allocations would undersize that first base pointer.
            slab = np.empty(total_bytes, dtype=np.uint8)
            views = [np.ndarray(item.shape, dtype=item.dtype, buffer=slab, offset=offset)
                     for item, offset in zip(self.output_contracts, self.output_offsets)]
            self.slots.append(_Slot(slab, views))
        self.pending: deque[_Slot] = deque()
        self._quarantined = False
        self._unresolved_serial_inputs: list[Any] = []
        self.reset()

    @staticmethod
    def _validate_array(value: Any, contract: TensorContract, role: str) -> None:
        if (not isinstance(value, np.ndarray) or value.shape != contract.shape
                or value.dtype != contract.dtype or not value.flags.c_contiguous):
            raise ValueError(f"deepx_{role}_shape_dtype_contiguity_mismatch:{contract.name}")

    def _inputs(self, value: Any, *, prepare: bool = False) -> list[np.ndarray]:
        if isinstance(value, Mapping):
            if set(value) != set(self.input_names):
                raise ValueError("deepx_input_names_mismatch")
            arrays = [value[name] for name in self.input_names]
        elif isinstance(value, np.ndarray):
            arrays = [value]
        elif isinstance(value, (list, tuple)):
            arrays = list(value)
        else:
            raise ValueError("deepx_input_container_invalid")
        if len(arrays) != len(self.input_contracts):
            raise ValueError("deepx_input_count_mismatch")
        result = []
        for array, contract in zip(arrays, self.input_contracts):
            # The sole supported correction makes the explicit compiled batch
            # dimension visible.  It never infers layouts or reshapes by size.
            if (prepare and isinstance(array, np.ndarray)
                    and contract.shape == (1,) + array.shape):
                array = np.expand_dims(array, axis=0)
            self._validate_array(array, contract, "input")
            result.append(array)
        return result

    def prepare_input(self, value: Any) -> Any:
        """Validate before timing; add only an exact missing leading batch one."""
        arrays = self._inputs(value, prepare=True)
        return arrays[0] if len(arrays) == 1 else arrays

    def reset(self) -> None:
        if (self.pending or self._unresolved_serial_inputs
                or any(slot.state != "free" for slot in self.slots)):
            raise RuntimeError("deepx_async_reset_before_full_drain")
        self.request_ids: set[int] = set()
        self.submitted = self.wait_completed = self.peak = 0
        self.submit_attempts = self.submit_failures = self.wait_failures = 0
        self.submit_times_ms: list[float] = []
        self.submit_admission_times_s: list[float] = []
        self.wait_times_ms: list[float] = []
        self.failure_notes: list[str] = []

    def _retain_unresolved(self) -> None:
        if not self._quarantined:
            _UNRESOLVED_OWNERS.append(self)
            self._quarantined = True

    def retain_unresolved_serial(self, inputs: Any, reason: str) -> None:
        """Retain an entered serial SDK call whose successful return is absent.

        The caller marks actual SDK entry and clears it only after the SDK
        returns.  Retention never validates, transforms or copies inputs: even
        a malformed input may already have reached the native runtime.
        """
        self._unresolved_serial_inputs.append(inputs)
        self.failure_notes.append("serial_completion_unresolved:" + str(reason))
        self._retain_unresolved()

    def submit(self, request_id: int, value: Any, *, deadline_s: float | None = None) -> int | None:
        if type(request_id) is not int or request_id < 0 or request_id in self.request_ids:
            raise ValueError("deepx_request_id_invalid_or_duplicate")
        if (self._unresolved_serial_inputs
                or any(slot.state in {"submit_unresolved", "wait_unresolved"} for slot in self.slots)):
            raise RuntimeError("deepx_async_runtime_completion_unresolved")
        if len(self.pending) >= INFLIGHT_LIMIT:
            raise RuntimeError("deepx_async_pending_limit_exceeded")
        slot = next((slot for slot in self.slots if slot.state == "free"), None)
        if slot is None:
            raise RuntimeError("deepx_output_slot_reused_before_release")
        arrays = self._inputs(value)
        # Resolve the bound SDK method before the final admission check too;
        # even attribute access may execute wrapper code.
        run_async = self.engine.run_async
        slot.state, slot.request_id, slot.inputs = "submitting", request_id, arrays
        self.request_ids.add(request_id)
        self.submit_attempts += 1
        if self.execution_lifecycle is not None:
            self.execution_lifecycle.inference_enter()
        begin = time.perf_counter()
        # This checked host timestamp is the admission linearization point,
        # immediately before the SDK invocation.  SDK/device entry is not
        # observable atomically with the clock check and may follow a host
        # scheduling pause.  Denied attempts have never exposed memory to SDK.
        if deadline_s is not None and begin >= deadline_s:
            slot.state = "free"
            slot.request_id = slot.inputs = None
            self.request_ids.remove(request_id)
            self.submit_attempts -= 1
            if self.execution_lifecycle is not None:
                self.execution_lifecycle.inference_not_entered()
            return None
        try:
            job_id = run_async(arrays, user_arg=request_id, output_buffer=slot.buffer)
            if type(job_id) is not int or job_id < 0:
                raise ValueError("deepx_async_job_id_invalid")
            if any(other is not slot and other.state != "free" and other.job_id == job_id
                   for other in self.slots):
                raise ValueError("deepx_async_job_id_duplicate_active")
        except BaseException as exc:
            slot.state = "submit_unresolved"
            self.submit_failures += 1
            self.failure_notes.append("submit_unresolved:" + repr(exc))
            self._retain_unresolved()
            raise
        finally:
            self.submit_admission_times_s.append(begin)
            self.submit_times_ms.append((time.perf_counter() - begin) * 1000.0)
        slot.job_id, slot.state = job_id, "submitted"
        self.pending.append(slot)
        self.submitted += 1
        self.peak = max(self.peak, len(self.pending))
        return job_id

    def wait_oldest(self) -> tuple[int, list[np.ndarray]]:
        if not self.pending:
            raise RuntimeError("deepx_async_wait_without_pending_job")
        slot = self.pending.popleft()
        begin = time.perf_counter()
        try:
            outputs = self.engine.wait(slot.job_id)
        except BaseException as exc:
            slot.state = "wait_unresolved"
            self.wait_failures += 1
            self.failure_notes.append("wait_unresolved:" + repr(exc))
            self._retain_unresolved()
            raise
        finally:
            self.wait_times_ms.append((time.perf_counter() - begin) * 1000.0)
        # SDK completion is known even if its returned tensor metadata is bad.
        if self.execution_lifecycle is not None:
            self.execution_lifecycle.inference_return()
        slot.state = "ready"
        self.wait_completed += 1
        try:
            if not isinstance(outputs, (list, tuple)) or len(outputs) != len(self.output_contracts):
                raise RuntimeError("deepx_async_output_count_mismatch")
            for output, owned, contract in zip(outputs, slot.outputs, self.output_contracts):
                self._validate_array(output, contract, "output")
                if output.ctypes.data != owned.ctypes.data or output.nbytes != owned.nbytes:
                    raise RuntimeError("deepx_async_output_does_not_alias_its_owned_slot")
        except BaseException as exc:
            self.wait_failures += 1
            self.failure_notes.append("completed_output_validation_failed:" + repr(exc))
            raise
        return int(slot.request_id), list(outputs)

    def release(self, request_id: int) -> None:
        slots = [slot for slot in self.slots if slot.request_id == request_id and slot.state != "free"]
        if len(slots) != 1 or slots[0].state != "ready":
            raise RuntimeError("deepx_output_slot_release_before_completion_or_duplicate")
        slot = slots[0]
        slot.state = "free"
        slot.request_id = slot.job_id = slot.inputs = None

    def drain(self) -> list[str]:
        """Await known jobs; unknown completions remain retained and unsafe."""
        errors: list[str] = []
        while self.pending:
            try:
                request_id, _ = self.wait_oldest()
                self.release(request_id)
            except BaseException as exc:
                errors.append("drain:" + repr(exc))
        for slot in self.slots:
            if slot.state == "ready":
                self.release(int(slot.request_id))
            elif slot.state != "free":
                errors.append(f"drain_unresolved:{slot.state}:request={slot.request_id}")
        if self._unresolved_serial_inputs:
            errors.append("drain_unresolved:serial_sdk_completion_unproven")
        return errors

    def report(self) -> dict[str, Any]:
        unresolved = bool(self._unresolved_serial_inputs) or any(
            slot.state in {"submit_unresolved", "wait_unresolved"} for slot in self.slots)
        return {
            "submitted": self.submitted, "submit_attempts": self.submit_attempts,
            "wait_completed": self.wait_completed,
            "pending": sum(slot.state in {"submitting", "submitted", "submit_unresolved", "wait_unresolved"}
                           for slot in self.slots),
            "unreleased": sum(slot.state != "free" for slot in self.slots),
            "observed_max_inflight": self.peak, "configured_inflight_limit": INFLIGHT_LIMIT,
            "slot_count": len(self.slots), "submit_failures": self.submit_failures,
            "wait_failures": self.wait_failures, "failure_notes": list(self.failure_notes),
            "unsafe_runtime_state": unresolved,
            "serial_completion_unresolved": bool(self._unresolved_serial_inputs),
            "scope": "successful host submits through ordered successful wait observations; no NPU occupancy claim",
            "output_slot_policy": "four_independent_caller_owned_contiguous_multioutput_slabs",
            "sdk_output_copy_policy": "SDK may copy internal outputs into caller slots; physical copy count unobserved",
            "admission_timestamp_scope": "checked host timestamp immediately before SDK invocation; SDK/device entry time unobserved",
        }


class NativeDeepXAsyncError(RuntimeError):
    def __init__(self, result: dict[str, Any], adapter: NativeDeepXAsync):
        super().__init__("; ".join(result["errors"]))
        self.result = result
        self.adapter = adapter
        self.unsafe_runtime_state = result["unsafe_runtime_state"]


def _arrays(value: Any):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _arrays(item)


def _snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    if isinstance(value, Mapping):
        return {name: _snapshot(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


def run_a4(adapter: NativeDeepXAsync, input_for_request: Callable[[int], Any], *,
           count: int | None = None, duration_s: float = 0.0,
           map_outputs: Callable[[int, list[np.ndarray]], Any] | None = None,
           consume: Callable[[int, Any], Any] | None = None,
           complete: Callable[[int, Any], Any] | None = None,
           on_submit: Callable[[int], Any] | None = None,
           capture: bool = False, cancel_event: threading.Event | None = None,
           timeout_s: float | None = None) -> dict[str, Any]:
    """Run one fully drained phase, continuously refilling its fixed four slots.

    A positive duration selects deadline admission (``count`` is then metadata).
    Every accepted request finishes during the final drain.  ``map_outputs``
    must own its payload before slot reuse.  Full runs completion inline.  A
    ``consume`` callback selects the identical producer/FIFO3/serial consumer
    arrangement for every split, without a separate postprocessing worker.

    Cancellation/timeouts are cooperative between SDK/callback calls.  DX-RT's
    blocking wait has no timeout argument; a permanently blocked native call
    still requires the existing owning-process watchdog, never a detached thread
    or a fabricated successful drain.
    """
    if count is not None and (type(count) is not int or count < 0):
        raise ValueError("deepx_a4_count_invalid")
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s < 0 or (not duration_s and count is None):
        raise ValueError("deepx_a4_duration_or_count_invalid")
    if timeout_s is not None and (not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0):
        raise ValueError("deepx_a4_timeout_invalid")
    if not callable(input_for_request):
        raise ValueError("deepx_a4_input_factory_missing")
    adapter.reset()
    latency = RequestLatency(
        int(count or 0), task_complete=True,
        start_anchor="native_deepx_a4:before_sdk_submit",
        end_anchor="native_deepx_a4:after_required_task_completion",
        enabled=not bool(duration_s),
    )
    cancelled = cancel_event if cancel_event is not None else threading.Event()
    producer_done = threading.Event()
    fifo: queue.Queue[Any] = queue.Queue(maxsize=FIFO_DEPTH)
    errors: list[str] = []
    unsafe_reasons: list[str] = []
    completed_ids: list[int] = []
    starts: dict[int, float] = {}
    completion_times: list[float] = []
    latencies: list[float] = []
    captured: dict[int, Any] = {}
    last_outputs: Any = None
    last_result: Any = None
    last_pending_outputs: Any = None
    next_id = 0
    observed_queue = 0
    start_event = threading.Event()
    ready = threading.Barrier(3) if consume is not None else None
    started = 0.0
    deadline = math.inf

    def check_stop() -> bool:
        if cancelled.is_set():
            if not errors:
                errors.append("deepx_a4_cancelled")
            return True
        if timeout_s is not None and time.perf_counter() >= started + float(timeout_s):
            errors.append("deepx_a4_timeout")
            cancelled.set()
            return True
        return False

    def submit_next() -> bool:
        nonlocal next_id
        if (check_stop() or (duration_s and time.perf_counter() >= deadline)
                or (not duration_s and next_id >= int(count))):
            return False
        value = input_for_request(next_id)
        # Recheck after input acquisition and instrumentation: no SDK admission
        # starts on or after the duration deadline.
        if duration_s and time.perf_counter() >= deadline:
            return False
        if check_stop():
            return False
        if on_submit is not None:
            on_submit(next_id)
        stamp = time.perf_counter()
        if duration_s and stamp >= deadline:
            return False
        starts[next_id] = stamp
        latency.start(next_id)
        job_id = adapter.submit(next_id, value, deadline_s=deadline if duration_s else None)
        if job_id is None:
            starts.pop(next_id)
            return False
        next_id += 1
        return True

    def finished(request_id: int, outputs: Any) -> None:
        nonlocal last_result, last_pending_outputs, last_outputs
        result = complete(request_id, outputs) if complete is not None else None
        latency.complete(request_id)
        timestamp = time.perf_counter()
        completed_ids.append(request_id)
        completion_times.append(timestamp)
        latencies.append((timestamp - starts[request_id]) * 1000.0)
        last_result = result
        if capture:
            last_outputs = _snapshot(outputs)
            captured[request_id] = last_outputs
        else:
            # Only the last values need an additional report snapshot.  Earlier
            # consumer staging may be overwritten after its completed task.
            # No further inference runs between worker join and this snapshot.
            last_pending_outputs = outputs

    def put(payload: Any) -> bool:
        nonlocal observed_queue
        while not check_stop():
            try:
                fifo.put(payload, timeout=0.05)
                observed_queue = max(observed_queue, fifo.qsize())
                return True
            except queue.Full:
                pass
        return False

    def produce() -> None:
        if ready is not None:
            ready.wait()
            start_event.wait()
        try:
            for _ in range(INFLIGHT_LIMIT):
                if not submit_next():
                    break
            while adapter.pending and not check_stop():
                request_id, raw = adapter.wait_oldest()
                payload = (map_outputs(request_id, raw) if map_outputs is not None
                           else [np.array(value, copy=True, order="C") for value in raw])
                if any(np.shares_memory(value, source) for value in _arrays(payload) for source in raw):
                    raise RuntimeError("deepx_a4_mapped_payload_aliases_runtime_output")
                adapter.release(request_id)
                # Rolling refill precedes consumer work/FIFO backpressure.
                submit_next()
                if consume is None:
                    finished(request_id, payload)
                elif not put((request_id, payload)):
                    break
        except BaseException as exc:
            errors.append("producer:" + repr(exc))
            cancelled.set()
        finally:
            errors.extend(adapter.drain())
            if errors:
                cancelled.set()
            producer_done.set()

    def consume_serial() -> None:
        ready.wait()
        start_event.wait()
        try:
            while not check_stop():
                try:
                    request_id, payload = fifo.get(timeout=0.05)
                except queue.Empty:
                    if producer_done.is_set():
                        break
                    continue
                try:
                    outputs = consume(request_id, payload)
                except BaseException as exc:
                    unsafe_reasons.append("consumer_synchronized_return_missing:" + repr(exc))
                    adapter.retain_unresolved_serial([consume, payload], str(exc))
                    raise
                finished(request_id, outputs)
        except BaseException as exc:
            errors.append("consumer:" + repr(exc))
            cancelled.set()

    workers: list[threading.Thread] = []
    if consume is not None:
        workers = [threading.Thread(target=produce, name="native-deepx-a4-producer"),
                   threading.Thread(target=consume_serial, name="native-deepx-trt-consumer")]
        for worker in workers:
            worker.start()
        ready.wait()
    started = time.perf_counter()
    deadline = started + duration_s if duration_s else math.inf
    if workers:
        start_event.set()
        for worker in workers:
            worker.join()
    else:
        produce()
    drained_at = time.perf_counter()
    if last_pending_outputs is not None:
        last_outputs = _snapshot(last_pending_outputs) if consume is not None else last_pending_outputs
    observation = adapter.report()
    observation.update(queue_depth=FIFO_DEPTH if consume is not None else 0,
                       observed_max_queue=observed_queue)
    if cancelled.is_set() and not errors:
        errors.append("deepx_a4_cancelled")
    if completed_ids != list(range(adapter.submitted)):
        errors.append("deepx_a4_completion_count_order_invalid")
    if not duration_s and len(completed_ids) != count:
        errors.append("deepx_a4_requested_count_not_completed")
    if observation["pending"] or observation["unreleased"]:
        errors.append("deepx_a4_runtime_drain_unresolved")
    if not errors and not fifo.empty():
        errors.append("deepx_a4_fifo_not_drained")
    if duration_s and any(stamp >= deadline for stamp in adapter.submit_admission_times_s):
        errors.append("deepx_a4_host_admission_after_deadline")
    ended = completion_times[-1] if completion_times else started
    makespan = max(0.0, ended - started)
    result = {
        "runtime_mode": "A4", "deepx_runtime_mode": "A4", "deepx_max_inflight": INFLIGHT_LIMIT,
        "requested_count": count, "duration_s": duration_s,
        "admission_deadline_s": deadline if duration_s else None,
        "submitted_count": adapter.submitted, "completed_count": len(completed_ids),
        "request_ids": completed_ids, "request_latencies_ms": latencies,
        "request_latency": latency.report(),
        "request_latency_scope": "API submit start to completed task including FIFO and required completion; no external arrival latency",
        "completion_times_s": completion_times,
        "timer_start_s": started, "timer_end_s": ended, "drain_end_s": drained_at,
        "makespan_s": makespan,
        "fps_makespan": len(completed_ids) / makespan if makespan > 0 else None,
        "actual_pending_max": observation["observed_max_inflight"], "observation": observation,
        "drained": not errors and not observation["unreleased"] and fifo.empty(),
        "last_outputs": last_outputs, "last_result": last_result, "captured": captured,
        "errors": errors,
        "unsafe_runtime_state": bool(observation["unsafe_runtime_state"] or unsafe_reasons),
        "unsafe_runtime_reasons": unsafe_reasons + list(observation["failure_notes"]),
        "submit_times_ms": list(adapter.submit_times_ms), "wait_times_ms": list(adapter.wait_times_ms),
        "submit_admission_times_s": list(adapter.submit_admission_times_s),
    }
    if errors:
        raise NativeDeepXAsyncError(result, adapter)
    return result
