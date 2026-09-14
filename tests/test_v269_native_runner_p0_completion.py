from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.runners.backends.hailo_backend import _HailoInferModelSession
from scripts import native_deepx_trt_e2e_from_benchmarkset as deepx


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _TRTShape:
    inputs = ["boundary"]
    shapes = {"boundary": (1, 4)}
    dtypes = {"boundary": np.dtype(np.float32)}


def test_deepx_fifo_payload_is_one_owned_copy_and_survives_runtime_buffer_reuse() -> None:
    runtime_buffer = np.arange(4, dtype=np.float32).reshape(1, 4)
    name, first, meta = deepx._map_deepx_output_to_trt([runtime_buffer], _TRTShape())
    assert name == "boundary"
    assert first.flags.owndata
    assert first.flags.c_contiguous
    assert not np.shares_memory(first, runtime_buffer)
    assert meta["boundary_copy_count"] == 1
    assert meta["fifo_payload_owns_memory"] is True
    assert meta["fifo_payload_shares_deepx_output"] is False

    runtime_buffer[...] = 99.0
    _name, second, second_meta = deepx._map_deepx_output_to_trt([runtime_buffer], _TRTShape())
    assert first.tolist() == [[0.0, 1.0, 2.0, 3.0]]
    assert second.tolist() == [[99.0, 99.0, 99.0, 99.0]]
    assert not np.shares_memory(first, second)
    assert second_meta["boundary_copy_count"] == 1


class _Job:
    def wait(self, _timeout_ms: int) -> None:
        return None


class _ConfiguredModel:
    def __init__(self, *, callbacks_per_submission: int = 1):
        self.callbacks_per_submission = callbacks_per_submission

    def wait_for_async_ready(self, *, timeout_ms: int) -> None:
        assert timeout_ms > 0

    def run_async(self, _bindings, callback):
        for _ in range(self.callbacks_per_submission):
            callback(SimpleNamespace(exception=None))
        return _Job()


def _completion_session(*, callbacks_per_submission: int = 1):
    session = _HailoInferModelSession.__new__(_HailoInferModelSession)
    session._configured_model = _ConfiguredModel(
        callbacks_per_submission=callbacks_per_submission,
    )
    session.timeout_ms = 10
    session.copy_outputs = False
    session._prepare_infer_inputs = lambda inputs: inputs
    session._hef_output_names = ["output_layer1"]
    session._output_name_hef_to_canonical = {"output_layer1": "output0"}
    session.output_shapes = {"output0": (3,)}
    created: list[dict[str, object]] = []

    def create_slot():
        done = threading.Event()
        done.set()
        slot: dict[str, object] = {
            "index": len(created), "binding": object(), "job": None,
            "done": done, "exception": None, "prefilled": True,
            "submitted_count": 0, "completed_count": 0,
            "postprocess_completed_count": 0,
            "output_buffers": {
                "output_layer1": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            },
            "counter_lock": threading.Lock(),
        }
        created.append(slot)
        return slot

    session._create_reusable_binding_slot = create_slot
    session._fill_reusable_slot_inputs = lambda *_args, **_kwargs: None
    waited_indices: list[int] = []
    original_wait = _HailoInferModelSession._wait_reusable_slot.__get__(session)

    def observed_wait(slot):
        waited_indices.append(int(slot["index"]))
        return original_wait(slot)

    session._wait_reusable_slot = observed_wait
    session._test_created_slots = created
    return session, waited_indices


def test_hailo_throughput_counts_observed_callbacks_and_drains_only_used_slots() -> None:
    session, waited = _completion_session()
    result = session.benchmark_throughput(
        {"input": np.zeros((1,), dtype=np.float32)},
        frames=2, inflight=4, warmup_frames=1,
    )
    assert result["requested_frames"] == 2
    assert result["completed_frames"] == 2
    assert result["completed_work_units_status"] == "exact_runtime_counter"
    assert result["warmup_completed_frames"] == 1
    assert result["completion_interval_semantics"].startswith("reciprocal_steady_state")
    assert "frame_ms" not in result
    assert set(waited).issubset({0, 1})


def test_hailo_throughput_fails_on_callback_count_mismatch() -> None:
    session, _waited = _completion_session(callbacks_per_submission=2)
    with pytest.raises(RuntimeError, match="completion count mismatch"):
        session.benchmark_throughput(
            {"input": np.zeros((1,), dtype=np.float32)},
            frames=2, inflight=2, warmup_frames=0,
        )


def test_hailo_throughput_counts_only_after_canonical_host_postprocess() -> None:
    session, _waited = _completion_session()
    callback_observations: list[tuple[list[str], list[float], int]] = []

    def postprocess(outputs: dict[str, np.ndarray]) -> None:
        completed_before_callback_returns = sum(
            int(slot.get("completed_count") or 0)
            for slot in session._test_created_slots
        )
        callback_observations.append((
            sorted(outputs), outputs["output0"].tolist(),
            completed_before_callback_returns,
        ))

    result = session.benchmark_throughput(
        {"input": np.zeros((1,), dtype=np.float32)},
        frames=2,
        inflight=2,
        warmup_frames=0,
        postprocess_callback=postprocess,
    )

    assert callback_observations == [
        (["output0"], [1.0, 2.0, 3.0], 0),
        (["output0"], [1.0, 2.0, 3.0], 1),
    ]
    assert result["completed_frames"] == 2
    assert result["postprocess_completed_frames"] == 2
    assert result["postprocess_completion_status"] == "exact_runtime_counter"
    assert result["completed_work_units_source"] == (
        "hailo_infermodel_frozen_postprocess_success_callback_counter"
    )


def test_full_runner_never_turns_latency_into_fps() -> None:
    full = _load_script("v269_full_no_reciprocal", "scripts/native_full_baseline_eval_runner.py")
    parsed = full._parse_trtexec_text(
        "Latency: min = 2 ms, max = 8 ms, mean = 4 ms\n"
    )
    assert parsed["latency_mean_ms"] == 4.0
    assert "fps_makespan" not in parsed
    assert "derived_from_latency" not in str(parsed)


def test_repetition_aggregator_requires_unique_runtime_instances_for_independence() -> None:
    h10 = _load_script(
        "v269_h10_repetition_scope",
        "scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
    )
    rows = [
        {"fps_makespan": float(index + 1), "frames": 2, "runtime_instance_id": runtime_id}
        for index, runtime_id in enumerate(("runtime-a", "runtime-b", "runtime-c"))
    ]
    fresh = h10._aggregate_repetition_metrics(
        rows,
        repetition_runtime_scope="fresh_runtime_per_repetition",
        repetition_independence_verified=True,
    )
    assert fresh["repetition_independence_verified"] is True
    assert [row["runtime_instance_id"] for row in fresh["repetition_records"]] == [
        "runtime-a", "runtime-b", "runtime-c",
    ]

    duplicated = h10._aggregate_repetition_metrics(
        [dict(row, runtime_instance_id="same-runtime") for row in rows],
        repetition_runtime_scope="fresh_runtime_per_repetition",
        repetition_independence_verified=True,
    )
    assert duplicated["repetition_independence_verified"] is False
