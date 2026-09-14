from __future__ import annotations

import ctypes
import threading
from pathlib import Path

import numpy as np

from scripts import native_deepx_trt_e2e_from_benchmarkset as deepx
from scripts import native_hailo10_trt_e2e_from_benchmarkset as hailo10


ROOT = Path(__file__).resolve().parents[1]


def test_workflow_does_not_cap_known_deepx_effort() -> None:
    source = (ROOT / "onnx_splitpoint_tool/workflow/runner.py").read_text(encoding="utf-8")
    assert "deepx_smoke_frames" not in source
    assert "deepx_smoke_warmup" not in source
    assert '"--frames", str(frames)' in source
    assert '"--warmup", str(warmup)' in source


def test_hailo8_dump_and_measurement_contract_is_outside_hotloop() -> None:
    source = (ROOT / "scripts/native_hailo_trt_fifo_from_benchmarkset.py").read_text(encoding="utf-8")
    cpp = source.split("CPP_SOURCE = r'''", 1)[1].split("'''", 1)[0]
    assert "fully drain every warm-up item" in cpp
    assert "workers_ready == 2" in cpp
    assert "separate_bound_inference_after_measurement" in cpp
    assert "slot.input_rgb.assign" not in cpp
    assert cpp.index("for (int seq = 0; seq < opt.warmup; ++seq)") < cpp.index("std::thread a(p1_thread)")
    assert cpp.index("double makespan_ms") < cpp.index("hailo.infer(dump_rgb, last_boundary_dump)")


class _Buffer:
    def __init__(self, value: np.ndarray):
        self.value = value

    def get_buffer(self) -> np.ndarray:
        return self.value


class _FakeHailo10Session:
    _hef_output_names = ["boundary"]
    _output_name_hef_to_canonical = {"boundary": "trt_input"}
    output_shapes = {"trt_input": (1, 4)}

    def _prepare_infer_inputs(self, inputs):
        return inputs

    def _create_reusable_binding_slot(self):
        return {"binding": object(), "done": threading.Event()}

    def _fill_reusable_slot_inputs(self, *_args, **_kwargs):
        return None

    def _submit_reusable_slot(self, slot, *_args, **_kwargs):
        slot["done"].set()

    def _wait_reusable_slot(self, _slot):
        return None

    def _binding_output(self, _binding, _name):
        return _Buffer(np.arange(4, dtype=np.float32).reshape(1, 4))


class _FakeTRT:
    inputs = ["trt_input"]
    shapes = {"trt_input": (1, 4)}
    dtypes = {"trt_input": np.dtype(np.float32)}
    host_memory_policy = "cuda_pinned_all_bindings"
    output_materialization_policy = "synchronized_d2h_pinned_view_no_post_copy"

    def prepare_inputs(self, feeds):
        self._feeds = feeds

    def run_prepared(self):
        return {"out": np.zeros((1, 2), dtype=np.float32)}

    def run(self, _feeds):
        return {"out": np.zeros((1, 2), dtype=np.float32)}


def test_hailo10_and_deepx_use_drained_warmup_and_ready_boundaries() -> None:
    trt = _FakeTRT()
    h10 = hailo10._hailo10_async_fifo_run(
        _FakeHailo10Session(), {"images": np.zeros(1)}, trt,
        frames=7, warmup=2, inflight=2, queue_depth=2,
    )
    assert h10["consumed_frames"] == 7
    assert h10["produced_frames"] == 7
    assert h10["payload_copy_ms"] == 0.0
    assert h10["boundary_copy_count"] == 1
    assert h10["trt_input_copy_ms"] >= 0.0
    assert h10["warmup_contract"] == "fully_drained_before_worker_start"
    assert h10["measurement_boundary"] == "workers_ready_to_last_completed_trt_frame"

    class _FakeDeepX:
        def run(self, _inputs):
            return [np.arange(4, dtype=np.float32).reshape(1, 4)]

    dx = deepx._deepx_fifo_run(
        _FakeDeepX(), np.zeros((1,), dtype=np.float32), trt,
        frames=5, warmup=2, queue_depth=2,
    )
    assert dx["completed_frames"] == 5
    assert dx["trt_input_copy_ms"] >= 0.0
    assert dx["warmup_contract"] == "fully_drained_before_worker_start"
    assert dx["measurement_boundary"] == "workers_ready_to_last_completed_trt_frame"


def test_repetition_aggregation_is_median_never_best_of() -> None:
    base = {
        "frames": 10,
        "completed_frames": 10,
        "makespan_ms": 1000.0,
        "fps_makespan": 10.0,
    }
    rows = [
        dict(base, fps_makespan=10.0),
        dict(base, fps_makespan=100.0),
        dict(base, fps_makespan=11.0),
    ]
    result = hailo10._aggregate_repetition_metrics(rows)
    assert result["fps_makespan"] == 11.0
    assert result["fps_makespan"] != max(row["fps_makespan"] for row in rows)
    assert result["repetition_aggregation"] == "median_never_best_of"
    assert len(result["repetition_evidence"]) == 3
    assert result["repetition_records"] == result["repetition_evidence"]
    assert result["repetition_count_requested"] == 3
    assert result["repetition_count_attempted"] == 3
    assert result["repetition_count_valid"] == 3
    assert result["repetition_status"] == "complete"
    assert result["fps_median"] == 11.0
    assert result["fps_makespan_ci95_low"] <= 11.0 <= result["fps_makespan_ci95_high"]


class _FakeCuda:
    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    def __init__(self):
        self.buffers = {}
        self.freed_host = []

    def _alloc(self, size):
        buf = ctypes.create_string_buffer(int(size))
        ptr = ctypes.addressof(buf)
        self.buffers[ptr] = buf
        return 0, ptr

    cudaMalloc = _alloc
    cudaMallocHost = _alloc

    def cudaFree(self, ptr):
        self.buffers.pop(int(ptr), None)
        return 0

    def cudaFreeHost(self, ptr):
        self.freed_host.append(int(ptr))
        self.buffers.pop(int(ptr), None)
        return 0

    def cudaMemcpyAsync(self, dst, src, size, _kind, _stream):
        ctypes.memmove(int(dst), int(src), int(size))
        return 0

    def cudaStreamSynchronize(self, _stream):
        return 0

    def cudaStreamDestroy(self, _stream):
        return 0


class _FakeContext:
    def __init__(self):
        self.addresses = {}

    def set_tensor_address(self, name, ptr):
        self.addresses[name] = ptr
        return True

    def execute_async_v3(self, _stream):
        return True


def test_shared_python_trt_consumer_uses_pinned_views_without_post_copy() -> None:
    consumer = hailo10.NativeTRT.__new__(hailo10.NativeTRT)
    consumer.cudart = _FakeCuda()
    consumer.ctx = _FakeContext()
    consumer.stream = 1
    consumer.inputs = []
    consumer.outputs = []
    consumer.shapes = {}
    consumer.dtypes = {}
    consumer.dev = {}
    consumer.host_in = {}
    consumer.host_out = {}
    consumer.host_ptr = {}
    consumer._host_backing = {}
    consumer.host_memory_policy = "cuda_pinned_all_bindings"
    consumer.output_materialization_policy = "synchronized_d2h_pinned_view_no_post_copy"

    consumer._reg("input", (1, 4), np.dtype(np.float32), True)
    consumer._reg("output", (1, 2), np.dtype(np.float32), False)
    outputs = consumer.run({"input": np.arange(4, dtype=np.float32).reshape(1, 4)})

    assert outputs["output"] is consumer.host_out["output"]
    assert consumer.host_in["input"].ctypes.data == consumer.host_ptr["input"]
    assert consumer.host_out["output"].ctypes.data == consumer.host_ptr["output"]
    host_ptr_count = len(consumer.host_ptr)
    consumer.close()
    assert len(consumer.cudart.freed_host) == host_ptr_count
