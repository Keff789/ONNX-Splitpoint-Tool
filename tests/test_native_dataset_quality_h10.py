from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool import native_dataset_quality_h10 as adapter
from scripts import native_hailo10_trt_e2e_from_benchmarkset as native


_NATIVE_FIFO = native._hailo10_async_fifo_run
_NATIVE_LOAD_BINDING = native._load_split_quality_binding


class _Completion:
    def __init__(self) -> None:
        self.completed_count = 0
        self.outputs: dict[str, np.ndarray] = {}
        self.execution_contract = {"fixture": "existing-completion-contract"}
        self.attestation_calls = 0

    def process(self, outputs: dict[str, np.ndarray]) -> None:
        self.completed_count += 1
        self.outputs = {name: value.copy() for name, value in outputs.items()}

    def attestation(self, *, completed_work_units: int) -> dict[str, Any]:
        self.attestation_calls += 1
        assert self.completed_count == completed_work_units == 1
        return {"attested": True, "status": "passed", "observation_relation": "same_hotloop_sentinel",
                "exact_result_claim_bound": True, "completed_work_units": 1, "completion_count": 1}


@pytest.fixture
def h10(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    bs = tmp_path / "benchmark_set"
    case = bs / "b043"
    case.mkdir(parents=True)
    (case / "model_part1_043.onnx").write_bytes(b"part1-onnx")
    hef = tmp_path / "part1.hef"
    hef.write_bytes(b"existing-hef")
    engine = tmp_path / "part2.engine"
    engine.write_bytes(b"existing-engine")
    metadata = {
        "boundary_tensor": {"name": "cut", "runtime_name": "raw/cut", "shape": [2, 2, 2], "dtype": "uint8"},
        "metadata_sha256": "sealed-boundary-digest",
    }
    boundary = tmp_path / "boundary.json"
    boundary.write_text(json.dumps(metadata))
    selection = {
        "model_id": "detector", "setup_id": "hailo10_setup", "precision": "uint8_dequant_fp16",
        "boundary_layout": "memory_nhwc_to_nchw", "preprocess_mode": "letterbox",
        "letterbox_pad_value": 114, "hailo_format": "uint8", "dequant_scale": 0.03125,
        "dequant_zero_point": 17.0,
    }
    binding = {
        "preselection": selection, "eval_run_id": "original-native-binding-run",
        "source_run_id": "hailo10h_to_trt", "artifacts": {
            "part1_runtime": {"path": str(hef)}, "engine": {"path": str(engine)},
            "boundary_metadata": {"path": str(boundary)},
        },
    }
    binding_path = tmp_path / "native_binding.json"
    binding_path.write_text(json.dumps(binding))
    (tmp_path / "native_trt_meta.json").write_text(json.dumps({
        "uint8_cast_bridge": {"schema": "onnx-splitpoint/uint8-dequant-bridge",
            "boundary_layout": {"requested": "memory_nhwc_to_nchw", "effective": "memory_nhwc_to_nchw"},
            "scale": 0.03125, "zero_point": 17.0},
    }))
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 2), (153, 51, 102)).save(image)
    events: list[tuple[Any, ...]] = []
    quantized_sources: list[np.ndarray] = []

    class Session:
        def describe_io(self) -> dict[str, Any]:
            events.append(("describe_io",))
            return {"runtime_output_format": "uint8", "runtime_output_formats": {"raw/cut": "UINT8"},
                    "runtime_output_quantization": {"raw/cut": {"scale": 0.03125, "zero_point": 17.0}}}

        def quantize_input(self, name: str, source: np.ndarray) -> np.ndarray:
            assert name == "images"
            quantized_sources.append(source.copy())
            return np.clip(np.rint(source / 0.25 + 2), 0, 255).astype(np.uint8)

    session = Session()
    prepared = SimpleNamespace(
        input_names=["images"], output_names=["cut"],
        handle=SimpleNamespace(session=session, runtime_input_shapes={"images": (6, 8, 3)}, input_shapes={}),
    )

    class Backend:
        def __init__(self, **options: Any):
            events.append(("backend", options))

        def prepare(self, cfg: Any, out_dir: Path) -> Any:
            events.append(("prepare", cfg, out_dir))
            return prepared

        def cleanup(self, value: Any) -> None:
            assert value is prepared
            events.append(("cleanup",))

    class TRT:
        def __init__(self, path: Path):
            assert path == engine
            events.append(("trt", path))
            self.inputs = ["cut"]
            self.outputs = ["scores"]
            self.shapes = {"cut": (1, 2, 2, 2), "scores": (1, 6, 1)}
            self.dtypes = {"cut": np.dtype("uint8"), "scores": np.dtype("float32")}

        def close(self) -> None:
            events.append(("trt_close",))

    def load_binding(ns: Any, *, case: str, task: str) -> dict[str, Any]:
        events.append(("binding", ns, case, task))
        assert ns.native_split_quality_binding == str(binding_path)
        assert ns.model_id == "detector"
        assert ns.setup_id == "hailo10_setup"
        assert ns.eval_run_id == "original-native-binding-run"
        assert ns.source_run_id == "hailo10h_to_trt"
        assert case == "b043" and task == "detection"
        return binding

    def fifo(sess: Any, inputs: dict[str, np.ndarray], trt: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(("fifo", sess, inputs, trt, kwargs))
        assert sess is session
        assert kwargs == {
            "frames": 1, "warmup": 0, "duration_s": 0.0, "inflight": 8, "queue_depth": 3,
            "task": "detection", "completion_runtime": kwargs["completion_runtime"], "warmup_completion_runtime": None,
        }
        native._pick_hailo_output({"cut": np.zeros((2, 2, 2), np.uint8)}, trt)
        kwargs["completion_runtime"].process({"scores": np.arange(6, dtype=np.float32).reshape(1, 6, 1)})
        return {"frames": 1, "produced_frames": 1, "consumed_frames": 1, "completed_frames": 1,
                "completed_work_units": 1, "fps_makespan": 12345, "p1_ms": 77,
                "completion_execution_attestation": kwargs["completion_runtime"].attestation(completed_work_units=1),
                "completion_execution_contract": kwargs["completion_runtime"].execution_contract}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Native dataset Quality must never build or enter FPS main")

    monkeypatch.setattr(native, "HailoBackend", Backend)
    monkeypatch.setattr(native, "NativeTRT", TRT)
    monkeypatch.setattr(native, "_load_split_quality_binding", load_binding)
    monkeypatch.setattr(native, "_hailo10_async_fifo_run", fifo)
    monkeypatch.setattr(native, "_build_engine", forbidden)
    monkeypatch.setattr(native, "main", forbidden)
    config = {
        "benchmark_set": str(bs), "case_id": "b043", "hef": str(hef), "engine": str(engine),
        "native_split_quality_binding": str(binding_path), "precision": "uint8_dequant_fp16",
        "boundary_layout": "memory_nhwc_to_nchw", "dequant_scale": 0.03125,
        "dequant_zero_point": 17.0, "preprocess_mode": "letterbox", "letterbox_pad_value": 114,
        "engine_build_forbidden": True, "eval_run_id": "new-dataset-run",
    }
    return SimpleNamespace(config=config, events=events, image=image, prepared=prepared,
                           session=session, quantized_sources=quantized_sources, fifo=fifo, tmp_path=tmp_path)


def test_reuses_actual_native_preprocessing_and_fifo_with_one_persistent_runtime(h10: SimpleNamespace) -> None:
    original_boundary = native._STRICT_SPLIT_BOUNDARY
    original_evidence = native._STRICT_SPLIT_BOUNDARY_EVIDENCE
    before = {name: Path(h10.config[name]).read_bytes() for name in ("hef", "engine")}
    with adapter.H10NativeQualityRuntime(h10.config, h10.tmp_path / "work") as runtime:
        assert runtime.output_signature == {"scores": {"shape": [1, 6, 1], "dtype": "float32"}}
        assert runtime.source_paths[0] == str(Path(native.__file__).resolve())
        for _ in range(2):
            completion = _Completion()
            evidence = runtime.run_image(h10.image, completion)
            assert completion.completed_count == 1
            assert completion.attestation_calls == 1
            assert evidence["completion_execution_attestation"]["completion_count"] == 1
            assert evidence["completion_execution_contract"] is completion.execution_contract
            assert evidence["native_completed_count"] == 1
            assert evidence["native_runtime_source_file_sha256"] == hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
            assert evidence["runtime_boundary_evidence"]["status"] == "exact_runtime_boundary_verified"
            assert evidence["runtime_boundary_evidence"]["shape"] == [2, 2, 2]
            assert set(evidence) == {"timing_input_binding", "prepared_input_snapshot", "native_completed_count",
                                     "native_runtime_source_file_sha256", "runtime_boundary_evidence",
                                     "completion_execution_attestation", "completion_execution_contract"}
            assert evidence["timing_input_binding"]["input_image_sha256"] == hashlib.sha256(h10.image.read_bytes()).hexdigest()
            fifo_call = [event for event in h10.events if event[0] == "fifo"][-1]
            image_array = fifo_call[2]["images"]
            assert evidence["prepared_input_snapshot"] == [{"name": "images", "shape": [6, 8, 3], "dtype": "uint8",
                "sha256": hashlib.sha256(image_array.tobytes()).hexdigest()}]
            assert image_array.flags.c_contiguous
            source = h10.quantized_sources[-1]
            assert source.dtype == np.float32
            np.testing.assert_allclose(source[0], np.full((8, 3), 114 / 255, dtype=np.float32))
            np.testing.assert_allclose(source[2, 2], np.asarray([153, 51, 102], np.float32) / 255)
            assert native._STRICT_SPLIT_BOUNDARY is original_boundary
            assert native._STRICT_SPLIT_BOUNDARY_EVIDENCE is original_evidence
    assert [event[0] for event in h10.events].count("backend") == 1
    assert [event[0] for event in h10.events].count("prepare") == 1
    assert [event[0] for event in h10.events].count("trt") == 1
    assert [event[0] for event in h10.events][-2:] == ["trt_close", "cleanup"]
    options = next(event[1] for event in h10.events if event[0] == "backend")
    assert options == {"strict": True, "hef_path": h10.config["hef"], "hw_arch": "hailo10h", "runtime_api": "infer_model",
        "quantized_inputs": True, "quantized_outputs": True, "hotloop": True, "copy_outputs": True,
        "onnx_model_path": str(Path(h10.config["benchmark_set"]) / "b043/model_part1_043.onnx")}
    assert before == {name: Path(h10.config[name]).read_bytes() for name in ("hef", "engine")}


@pytest.mark.parametrize("flag", ["build", "build_missing_engine", "force_rebuild_engine", "force_build", "force_rebuild"])
def test_rejects_build_requests_before_opening_hardware(h10: SimpleNamespace, flag: str) -> None:
    with pytest.raises(RuntimeError, match="engine_build_forbidden"):
        with adapter.H10NativeQualityRuntime({**h10.config, flag: True}, h10.tmp_path / "work"):
            pass
    assert not h10.events


@pytest.mark.parametrize("option", ["quantized_inputs", "quantized_outputs", "copy_outputs"])
def test_requires_native_uint8_and_output_copy_options(h10: SimpleNamespace, option: str) -> None:
    with pytest.raises(RuntimeError, match="required_native_option"):
        with adapter.H10NativeQualityRuntime({**h10.config, option: False}, h10.tmp_path / "work"):
            pass
    assert not h10.events


def test_rejects_missing_engine_without_building(h10: SimpleNamespace) -> None:
    Path(h10.config["engine"]).unlink()
    with pytest.raises(RuntimeError, match="artifact_missing:engine"):
        with adapter.H10NativeQualityRuntime(h10.config, h10.tmp_path / "work"):
            pass
    assert not h10.events


def test_rejects_sealed_preselection_drift_before_runtime_open(h10: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="preselection_mismatch:boundary_layout"):
        with adapter.H10NativeQualityRuntime({**h10.config, "boundary_layout": "as_input"}, h10.tmp_path / "work"):
            pass
    assert [event[0] for event in h10.events] == ["binding"]


def test_engine_quantization_is_checked_by_actual_native_guard(h10: SimpleNamespace) -> None:
    metadata = h10.tmp_path / "native_trt_meta.json"
    payload = json.loads(metadata.read_text())
    payload["uint8_cast_bridge"]["scale"] = 0.25
    metadata.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="Part2 engine QuantInfo mismatch"):
        with adapter.H10NativeQualityRuntime(h10.config, h10.tmp_path / "work"):
            pass
    assert "trt" not in [event[0] for event in h10.events]
    assert h10.events[-1] == ("cleanup",)


def test_existing_native_binding_validator_rejects_invalid_seal_before_hardware(h10: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "_load_split_quality_binding", _NATIVE_LOAD_BINDING)
    with pytest.raises(RuntimeError, match="native_split_quality_local_validation_failed"):
        with adapter.H10NativeQualityRuntime(h10.config, h10.tmp_path / "work"):
            pass
    assert not h10.events


def test_actual_native_fifo_calls_injected_completion_exactly_once(h10: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native, "_hailo10_async_fifo_run", _NATIVE_FIFO)
    session = h10.session
    session._hef_output_names = ["raw/cut"]
    session._output_name_hef_to_canonical = {"raw/cut": "cut"}
    session.output_shapes = {"cut": (1, 2, 2, 2)}
    session._prepare_infer_inputs = lambda inputs: inputs
    session._create_reusable_binding_slot = lambda: {"binding": object(), "done": threading.Event()}
    session._fill_reusable_slot_inputs = lambda *args: None

    def submit(slot: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        h10.events.append(("submit",))
        slot["done"].set()

    session._submit_reusable_slot = submit
    session._wait_reusable_slot = lambda slot: None
    session._binding_output = lambda binding, name: SimpleNamespace(
        get_buffer=lambda: np.zeros((2, 2, 2), np.uint8),
    )

    class Completion(_Completion):
        def attestation(self, *, completed_work_units: int) -> dict[str, Any]:
            assert completed_work_units == self.completed_count == 1
            return {
                "attested": True, "status": "passed", "observation_relation": "same_hotloop_sentinel",
                "exact_result_claim_bound": True, "completed_work_units": 1, "completion_count": 1,
            }

    with adapter.H10NativeQualityRuntime(h10.config, h10.tmp_path / "work") as runtime:
        def prepare_inputs(feeds: dict[str, np.ndarray]) -> None:
            assert list(feeds) == ["cut"]
            assert feeds["cut"].shape == (2, 2, 2)
            assert feeds["cut"].dtype == np.uint8
            h10.events.append(("trt_prepare_inputs",))

        runtime._trt.prepare_inputs = prepare_inputs
        runtime._trt.run_prepared = lambda: {"scores": np.zeros((1, 6, 1), np.float32)}
        completion = Completion()
        result = runtime.run_image(h10.image, completion)
        assert result["native_completed_count"] == completion.completed_count == 1
        assert result["runtime_boundary_evidence"]["trt_input_shape"] == [1, 2, 2, 2]
    assert [event[0] for event in h10.events].count("submit") == 1
    assert [event[0] for event in h10.events].count("trt_prepare_inputs") == 1


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_any_other_native_completion_count(h10: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    def fifo(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = h10.fifo(*args, **kwargs)
        result["completed_work_units"] = count
        return result

    monkeypatch.setattr(native, "_hailo10_async_fifo_run", fifo)
    with pytest.raises(RuntimeError, match="exactly_one_completion_required"):
        with adapter.H10NativeQualityRuntime(h10.config, h10.tmp_path / "work") as runtime:
            runtime.run_image(h10.image, _Completion())
    assert [event[0] for event in h10.events][-2:] == ["trt_close", "cleanup"]


def test_fifo_error_restores_existing_native_globals_and_cleans_up(h10: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    boundary = {"existing": "boundary"}
    evidence = {"existing": "evidence"}
    monkeypatch.setattr(native, "_STRICT_SPLIT_BOUNDARY", boundary)
    monkeypatch.setattr(native, "_STRICT_SPLIT_BOUNDARY_EVIDENCE", evidence)

    def fifo(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("injected native FIFO failure")

    monkeypatch.setattr(native, "_hailo10_async_fifo_run", fifo)
    with pytest.raises(RuntimeError, match="injected native FIFO failure"):
        with adapter.H10NativeQualityRuntime(h10.config, h10.tmp_path / "work") as runtime:
            runtime.run_image(h10.image, _Completion())
    assert native._STRICT_SPLIT_BOUNDARY is boundary
    assert native._STRICT_SPLIT_BOUNDARY_EVIDENCE is evidence
    assert [event[0] for event in h10.events][-2:] == ["trt_close", "cleanup"]
