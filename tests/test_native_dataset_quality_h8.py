from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool import native_dataset_quality_h8 as adapter


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


@pytest.fixture(params=["uint8", "float32"])
def h8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> SimpleNamespace:
    native = adapter._load_native_runtime()
    actual_binding_loader = native._load_native_split_quality_binding
    actual_fifo = native._hailo8_python_fifo_run
    hailo_format = str(request.param)
    bs = tmp_path / "benchmark_set"
    bs.mkdir()
    hef = tmp_path / "part1.hef"
    hef.write_bytes(b"existing-hailo8-hef")
    engine = tmp_path / "part2.engine"
    engine.write_bytes(b"existing-tensorrt-engine")
    boundary = tmp_path / "boundary.json"
    boundary.write_text(json.dumps({"boundary_tensor": {
        "name": "cut", "runtime_name": "raw/cut", "shape": [2, 2, 2], "dtype": hailo_format,
    }}))
    selection = {
        "model_id": "detector", "setup_id": "hailo8_setup", "precision": "fp16" if hailo_format == "float32" else "uint8_dequant_fp16",
        "hailo_format": hailo_format, "boundary_layout": "as_input", "preprocess_mode": "letterbox", "letterbox_pad_value": 114,
    }
    binding = {"preselection": selection, "eval_run_id": "original-native-binding", "source_run_id": "hailo8_to_trt", "artifacts": {
        "part1_runtime": {"path": str(hef)}, "engine": {"path": str(engine)}, "boundary_metadata": {"path": str(boundary)},
    }}
    binding_path = tmp_path / "native_binding.json"
    binding_path.write_text(json.dumps(binding))
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 2), (153, 51, 102)).save(image)
    events: list[tuple[Any, ...]] = []
    prepared = SimpleNamespace(input_names=["images"], handle=SimpleNamespace(runtime_input_shapes={"images": (6, 8, 3)}))
    state = SimpleNamespace(boundary_name="raw/cut", boundary_shape=(2, 2, 2), boundary_dtype=hailo_format)

    class Backend:
        def __init__(self, **options: Any):
            events.append(("backend", options))

        def prepare(self, cfg: Any, work: Path) -> Any:
            events.append(("prepare", cfg, work))
            return prepared

        def run(self, value: Any, inputs: dict[str, np.ndarray]) -> Any:
            assert value is prepared
            events.append(("hailo_run", inputs))
            return SimpleNamespace(outputs={state.boundary_name: np.zeros(state.boundary_shape, dtype=state.boundary_dtype)})

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
            self.dtypes = {"cut": np.dtype(hailo_format), "scores": np.dtype("float32")}

        def run(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            assert list(feeds) == ["cut"]
            assert feeds["cut"].shape == self.shapes["cut"]
            assert feeds["cut"].dtype == self.dtypes["cut"]
            events.append(("trt_run",))
            return {"scores": np.arange(6, dtype=np.float32).reshape(1, 6, 1)}

        def close(self) -> None:
            events.append(("trt_close",))

    def load_binding(path: str, *, args: Any, case: str, task: str) -> dict[str, Any]:
        events.append(("binding", args))
        assert path == str(binding_path)
        assert args.model_id == "detector" and args.setup_id == "hailo8_setup"
        assert args.eval_run_id == "original-native-binding" and args.source_run_id == "hailo8_to_trt"
        assert args.build is False
        assert case == "b043" and task == "detection"
        return binding

    def fifo(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(("fifo", kwargs))
        assert kwargs == {
            "frames": 1, "warmup": 0, "duration_s": 0.0, "queue_depth": 3,
            "completion_runtime": kwargs["completion_runtime"], "warmup_completion_runtime": None,
            "expected_boundary_name": "raw/cut", "expected_boundary_shape": [2, 2, 2], "expected_boundary_dtype": hailo_format,
        }
        return actual_fifo(*args, **kwargs)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Dataset Quality must never invoke Native FPS main or a build")

    monkeypatch.setattr(native, "HailoBackend", Backend)
    monkeypatch.setattr(native, "NativeTRT", TRT)
    monkeypatch.setattr(native, "_load_native_split_quality_binding", load_binding)
    monkeypatch.setattr(native, "_hailo8_python_fifo_run", fifo)
    monkeypatch.setattr(native, "main", forbidden)
    monkeypatch.setattr(native, "_prepare_cpp_wrapper", forbidden)
    config = {"benchmark_set": str(bs), "case_id": "b043", "hef": str(hef), "engine": str(engine),
              "native_split_quality_binding": str(binding_path), **{key: value for key, value in selection.items() if key not in {"model_id", "setup_id"}},
              "device_id": "0000:01:00.0", "engine_build_forbidden": True, "eval_run_id": "new-dataset-evaluation"}
    return SimpleNamespace(native=native, config=config, events=events, image=image, prepared=prepared, state=state,
                           hailo_format=hailo_format, fifo=fifo, actual_binding_loader=actual_binding_loader, tmp_path=tmp_path)


def test_existing_native_runtime_preprocess_and_real_fifo_complete_once_per_image(h8: SimpleNamespace) -> None:
    before = {name: Path(h8.config[name]).read_bytes() for name in ("hef", "engine")}
    with adapter.H8NativeQualityRuntime(h8.config, h8.tmp_path / "work") as runtime:
        assert runtime.output_signature == {"scores": {"shape": [1, 6, 1], "dtype": "float32"}}
        assert runtime.source_paths[:2] == (str(adapter._H8_SOURCE.resolve()), str(Path(h8.native._native_trt_module.__file__).resolve()))
        for _ in range(2):
            completion = _Completion()
            evidence = runtime.run_image(h8.image, completion)
            assert evidence["native_completed_count"] == completion.completed_count == 1
            assert completion.attestation_calls == 1
            assert evidence["completion_execution_attestation"]["completion_count"] == 1
            assert evidence["completion_execution_contract"] == completion.execution_contract
            assert set(evidence) == {"timing_input_binding", "prepared_input_snapshot", "native_completed_count",
                                     "native_runtime_source_file_sha256", "runtime_boundary_evidence",
                                     "completion_execution_attestation", "completion_execution_contract"}
            assert evidence["native_runtime_source_file_sha256"] == hashlib.sha256(adapter._H8_SOURCE.read_bytes()).hexdigest()
            assert evidence["runtime_boundary_evidence"]["status"] == "exact_runtime_boundary_verified"
            assert evidence["runtime_boundary_evidence"]["runtime_name"] == "raw/cut"
            assert evidence["runtime_boundary_evidence"]["shape"] == [2, 2, 2]
            assert evidence["timing_input_binding"]["executed_rgb_matches_common_preprocessing"] is True
            assert evidence["timing_input_binding"]["input_image_sha256"] == hashlib.sha256(h8.image.read_bytes()).hexdigest()
            inputs = [event[1] for event in h8.events if event[0] == "hailo_run"][-1]
            array = inputs["images"]
            assert array.dtype == np.dtype(h8.hailo_format) and array.flags.c_contiguous
            assert evidence["prepared_input_snapshot"] == [{"name": "images", "shape": [6, 8, 3], "dtype": h8.hailo_format,
                                                           "sha256": hashlib.sha256(array.tobytes()).hexdigest()}]
            factor = 1.0 if h8.hailo_format == "uint8" else 1 / 255
            np.testing.assert_allclose(array[0], np.full((8, 3), 114 * factor, dtype=np.float32))
            np.testing.assert_allclose(array[2, 2], np.asarray([153, 51, 102], np.float32) * factor)
    names = [event[0] for event in h8.events]
    assert names.count("backend") == names.count("prepare") == names.count("trt") == 1
    assert names.count("hailo_run") == names.count("trt_run") == 2
    assert names[-2:] == ["cleanup", "trt_close"]
    options = next(event[1] for event in h8.events if event[0] == "backend")
    assert options == {"strict": True, "hw_arch": "hailo8", "hef_path": h8.config["hef"], "compile_backend": "local",
        "runtime_api": "vstreams", "quantized_inputs": h8.hailo_format == "uint8", "quantized_outputs": h8.hailo_format == "uint8",
        "persistent_activation": True, "copy_outputs": True, "device_id": "0000:01:00.0"}
    assert before == {name: Path(h8.config[name]).read_bytes() for name in ("hef", "engine")}


@pytest.mark.parametrize("flag", ["build", "build_missing_engine", "force_rebuild_engine", "force_build", "force_rebuild"])
def test_rejects_build_requests_before_opening_hardware(h8: SimpleNamespace, flag: str) -> None:
    with pytest.raises(RuntimeError, match="engine_build_forbidden"):
        with adapter.H8NativeQualityRuntime({**h8.config, flag: True}, h8.tmp_path / "work"):
            pass
    assert not h8.events


def test_missing_engine_never_triggers_build(h8: SimpleNamespace) -> None:
    Path(h8.config["engine"]).unlink()
    with pytest.raises(RuntimeError, match="artifact_missing:engine"):
        with adapter.H8NativeQualityRuntime(h8.config, h8.tmp_path / "work"):
            pass
    assert not h8.events


def test_existing_eight_artifact_validator_rejects_invalid_binding(h8: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(h8.native, "_load_native_split_quality_binding", h8.actual_binding_loader)
    with pytest.raises(RuntimeError, match="native_split_quality_local_validation_failed"):
        with adapter.H8NativeQualityRuntime(h8.config, h8.tmp_path / "work"):
            pass
    assert not h8.events


def test_sealed_format_cannot_drift(h8: SimpleNamespace) -> None:
    other_format = "float32" if h8.hailo_format == "uint8" else "uint8"
    with pytest.raises(RuntimeError, match="preselection_mismatch:hailo_format"):
        with adapter.H8NativeQualityRuntime({**h8.config, "hailo_format": other_format}, h8.tmp_path / "work"):
            pass
    assert [event[0] for event in h8.events] == ["binding"]


@pytest.mark.parametrize("field,value,error", [("boundary_name", "wrong/name", "name_mismatch"),
    ("boundary_shape", (1, 2, 2, 2), "shape_mismatch"), ("boundary_dtype", "float16", "dtype_mismatch")])
def test_actual_fifo_rejects_runtime_boundary_drift_and_cleans_up(h8: SimpleNamespace, field: str, value: Any, error: str) -> None:
    setattr(h8.state, field, value)
    with pytest.raises(RuntimeError, match=f"native_split_quality_runtime_boundary_{error}"):
        with adapter.H8NativeQualityRuntime(h8.config, h8.tmp_path / "work") as runtime:
            runtime.run_image(h8.image, _Completion())
    assert [event[0] for event in h8.events][-2:] == ["cleanup", "trt_close"]


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_other_native_completion_counts(h8: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    def fifo(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = h8.fifo(*args, **kwargs)
        result["completed_work_units"] = count
        return result

    monkeypatch.setattr(h8.native, "_hailo8_python_fifo_run", fifo)
    with pytest.raises(RuntimeError, match="exactly_one_completion_required"):
        with adapter.H8NativeQualityRuntime(h8.config, h8.tmp_path / "work") as runtime:
            runtime.run_image(h8.image, _Completion())
    assert [event[0] for event in h8.events][-2:] == ["cleanup", "trt_close"]


def test_pins_short_native_trt_import_to_existing_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("# incompatible consumer source\n")
    monkeypatch.setitem(sys.modules, "native_hailo10_trt_e2e_from_benchmarkset", SimpleNamespace(__file__=str(unrelated)))
    with pytest.raises(RuntimeError, match="native_trt_source_mismatch"):
        adapter._load_native_runtime()
