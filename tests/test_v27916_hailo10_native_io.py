from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.native_split_quality import (
    known_native_split_policy,
    materialize_native_split_preselection,
)
from onnx_splitpoint_tool.runners.backends.hailo_backend import (
    _HailoInferModelSession,
    _hailo_native_vstream_format_type_name,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"


class _QuantInfo:
    def __init__(self, scale: float, zero_point: float) -> None:
        self.qp_scale = float(scale)
        self.qp_zp = float(zero_point)


class _Transform:
    calls: list[str] = []

    @classmethod
    def quantize_input_buffer(cls, source, destination, count, quant) -> None:
        cls.calls.append("quantize")
        limits = np.iinfo(destination.dtype)
        values = np.rint(
            source.reshape(-1)[:count] / quant.qp_scale + quant.qp_zp
        )
        destination.reshape(-1)[:count] = np.clip(
            values, limits.min, limits.max,
        ).astype(destination.dtype)

    @classmethod
    def dequantize_output_buffer(cls, source, destination, count, quant) -> None:
        cls.calls.append("dequantize")
        destination.reshape(-1)[:count] = (
            source.reshape(-1)[:count].astype(np.float32) - quant.qp_zp
        ) * quant.qp_scale


def _bare_session() -> _HailoInferModelSession:
    session = object.__new__(_HailoInferModelSession)
    session._hpf = SimpleNamespace(HailoRTTransformUtils=_Transform)
    session._input_name_canonical_to_hef = {"input": "hef/input"}
    session._output_name_hef_to_canonical = {"hef/output": "cut"}
    session._input_dtypes = {"hef/input": np.uint8}
    session._output_dtypes = {"hef/output": np.uint8}
    session._input_quant_infos = {}
    session._output_quant_infos = {}
    return session


@pytest.mark.parametrize(
    ("source", "scale", "zero_point", "expected"),
    [
        ([0.0, 0.5, 1.0], 1.0 / 255.0, 0.0, [0, 127, 255]),
        ([-2.0, 0.0, 2.0], 0.02, 100.0, [0, 100, 200]),
    ],
)
def test_hailo10_exact_hef_input_quantization_handles_yolo_and_imagenet(
    source: list[float], scale: float, zero_point: float, expected: list[int],
) -> None:
    session = _bare_session()
    session._input_quant_infos["hef/input"] = _QuantInfo(scale, zero_point)
    _Transform.calls.clear()

    result = session.quantize_input(
        "input", np.asarray(source, dtype=np.float32),
    )

    assert result.dtype == np.uint8
    # NumPy/HailoRT use nearest rounding; 0.5/(1/255) is represented just below
    # 127.5 in float32 and therefore resolves to 127 on the tested contract.
    assert result.tolist() == expected
    assert _Transform.calls == ["quantize"]


def test_hailo10_native_output_dequantization_uses_per_output_quant_info() -> None:
    session = _bare_session()
    session._output_quant_infos["hef/output"] = _QuantInfo(0.25, 7.0)
    _Transform.calls.clear()

    result = session.dequantize_output(
        "cut", np.asarray([7, 9, 3], dtype=np.uint8),
    )

    np.testing.assert_allclose(result, [0.0, 0.5, -1.0])
    assert result.dtype == np.float32
    assert _Transform.calls == ["dequantize"]


def test_hailo10_prequantized_energy_feed_is_not_quantized_twice() -> None:
    session = _bare_session()
    session._input_quant_infos["hef/input"] = _QuantInfo(0.02, 100.0)
    _Transform.calls.clear()
    source = np.arange(12, dtype=np.uint8).reshape(3, 4).T

    result = session.quantize_input("input", source)

    np.testing.assert_array_equal(result, source)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert _Transform.calls == []


def test_hailo10_output_host_buffer_uses_the_hef_native_uint8_type() -> None:
    format_name = "UINT8"
    numpy_dtype = np.uint8
    quant = _QuantInfo(0.125, 11.0)
    stream = SimpleNamespace(
        quant_infos=[quant],
        shape=(2, 3),
        set_format_type=lambda value: setattr(stream, "selected", value),
    )
    info = SimpleNamespace(
        name="hef/output", shape=(2, 3), quant_info=quant,
        # InferModel may expose a host-transformed FLOAT32 vstream even though
        # the underlying HEF stream is native UINT8.
        format=SimpleNamespace(type="FLOAT32"),
    )
    low_level_info = SimpleNamespace(
        name="raw/output", format=SimpleNamespace(type=format_name),
    )
    hef = SimpleNamespace(
        get_network_group_names=lambda: ["network"],
        get_output_stream_infos=lambda _group: [low_level_info],
        get_stream_names_from_vstream_name=lambda _name, _group: [
            "raw/output"
        ],
    )
    session = object.__new__(_HailoInferModelSession)
    session.quantized_outputs = True
    session._hef = hef
    session._hpf = SimpleNamespace(
        FormatType=SimpleNamespace(UINT8="u8", UINT16="u16", FLOAT32="f32")
    )
    session._hef_output_names = ["hef/output"]
    session._infer_model = SimpleNamespace(output=lambda _name=None: stream)
    session._output_format_names = {}
    session._output_dtypes = {}
    session._output_quant_infos = {}

    session._set_output_formats([info])

    assert stream.selected == "u8"
    assert np.dtype(session._output_dtypes["hef/output"]) == np.dtype(numpy_dtype)
    assert session._output_quant_infos["hef/output"] is quant
    assert session._alloc_output_buffers()["hef/output"].dtype == numpy_dtype


def test_hailo10_uint16_output_fails_closed_before_trt_uint8_bridge() -> None:
    quant = _QuantInfo(0.125, 11.0)
    stream = SimpleNamespace(
        quant_infos=[quant],
        set_format_type=lambda value: setattr(stream, "selected", value),
    )
    info = SimpleNamespace(
        name="hef/output", shape=(2, 3), quant_info=quant,
        format=SimpleNamespace(type="FLOAT32"),
    )
    hef = SimpleNamespace(
        get_network_group_names=lambda: ["network"],
        get_output_stream_infos=lambda _group: [SimpleNamespace(
            name="raw/output", format=SimpleNamespace(type="UINT16"),
        )],
        get_stream_names_from_vstream_name=lambda _name, _group: [
            "raw/output"
        ],
    )
    session = object.__new__(_HailoInferModelSession)
    session.quantized_outputs = True
    session._hef = hef
    session._hpf = SimpleNamespace(
        FormatType=SimpleNamespace(UINT8="u8", UINT16="u16", FLOAT32="f32")
    )
    session._hef_output_names = ["hef/output"]
    session._infer_model = SimpleNamespace(output=lambda _name=None: stream)
    session._output_format_names = {}
    session._output_dtypes = {}
    session._output_quant_infos = {}

    with pytest.raises(RuntimeError, match="required.*UINT8|resolve to UINT8"):
        session._set_output_formats([info])


def _load_generated_runner_template():
    loader = importlib.machinery.SourceFileLoader(
        "_v27916_generated_runner_template", str(TEMPLATE),
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = module
    loader.exec_module(module)
    return module


def _load_python_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_hailo10_native_input_selection_uses_underlying_hef_and_exact_quant_info() -> None:
    template = _load_generated_runner_template()
    quant = _QuantInfo(0.02, 100.0)
    info = SimpleNamespace(
        name="hef/input", shape=(2, 3, 1), quant_info=quant,
        format=SimpleNamespace(type="FLOAT32"),
    )
    low_level_info = SimpleNamespace(
        name="raw/input", format=SimpleNamespace(type="UINT8"),
    )
    hef = SimpleNamespace(
        get_network_group_names=lambda: ["network"],
        get_input_stream_infos=lambda _group: [low_level_info],
        get_stream_names_from_vstream_name=lambda _name, _group: ["raw/input"],
    )

    for session_type in (_HailoInferModelSession, template.HailoInferModelSession):
        stream = SimpleNamespace(
            quant_infos=[quant],
            set_format_type=lambda value: setattr(stream, "selected", value),
        )
        session = object.__new__(session_type)
        session.quantized_inputs = True
        session._hef = hef
        session._hpf = SimpleNamespace(
            FormatType=SimpleNamespace(UINT8="u8", UINT16="u16", FLOAT32="f32")
        )
        session._hef_input_names = ["hef/input"]
        session._infer_model = SimpleNamespace(input=lambda _name=None: stream)
        session._input_format_names = {}
        session._input_dtypes = {}
        session._input_quant_infos = {}

        session._set_input_formats([info])

        assert stream.selected == "u8"
        assert session._input_quant_infos["hef/input"] is quant
        assert np.dtype(session._input_dtypes["hef/input"]) == np.dtype(np.uint8)


def test_hailo10_native_format_resolution_fails_closed_when_low_level_mapping_is_ambiguous() -> None:
    info = SimpleNamespace(
        name="hef/output", format=SimpleNamespace(type="UINT8"),
    )
    hef = SimpleNamespace(
        get_network_group_names=lambda: ["network"],
        get_output_stream_infos=lambda _group: [
            SimpleNamespace(
                name="raw/a", format=SimpleNamespace(type="UINT8"),
            ),
            SimpleNamespace(
                name="raw/b", format=SimpleNamespace(type="UINT8"),
            ),
        ],
        get_stream_names_from_vstream_name=lambda _name, _group: [
            "raw/a", "raw/b"
        ],
    )

    assert _hailo_native_vstream_format_type_name(
        hef, info, direction="output", fallback="",
    ) == ""


def test_generated_runner_prepares_hailo10_uint8_before_infermodel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_generated_runner_template()
    monkeypatch.delenv("ONNX_SPLITPOINT_HAILO_QUANTIZED_IMAGE_INPUTS", raising=False)
    monkeypatch.delenv("SPLITPOINT_HAILO_QUANTIZED_IMAGE_INPUTS", raising=False)
    assert module._hailo_image_quantized_inputs_default("hailo10h") is True
    assert module._hailo_image_quantized_inputs_default("hailo8") is False

    seen: dict[str, np.ndarray] = {}

    class _Session:
        quantized_inputs = True
        runtime_input_shapes = {"input": (1, 2, 3)}
        input_shapes = {}

        @staticmethod
        def quantize_input(name: str, values: np.ndarray) -> np.ndarray:
            seen[name] = np.array(values, copy=True)
            return np.clip(np.rint(values / 0.02 + 100.0), 0, 255).astype(
                np.uint8
            )

    source = np.asarray(
        [[[[ -2.0, -1.0]], [[0.0, 1.0]], [[2.0, 3.0]]]],
        dtype=np.float32,
    )
    prepared = module._prepare_hailo_image_input(_Session(), source, "input")

    assert seen["input"].dtype == np.float32
    assert float(seen["input"].min()) < 0.0
    assert prepared.dtype == np.uint8
    assert prepared.flags.c_contiguous


def test_generated_runner_selects_native_hailo10_input_for_split_only() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    assert "_hailo10_native_part1_io" not in source
    assert (
        "quantized_inputs=_hailo_image_quantized_inputs_default(stage1_tok)"
        in source
    )
    assert (
        "quantized_inputs=_hailo_image_quantized_inputs_default()"
        in source
    )
    native_source = (
        ROOT / "scripts/native_hailo10_trt_e2e_from_benchmarkset.py"
    ).read_text(encoding="utf-8")
    assert "--precision', default='uint8_dequant_fp16'" in native_source


def test_hailo10_fallback_uses_dequant_bridge_without_changing_hailo8() -> None:
    from onnx_splitpoint_tool.workflow.runner import (
        _native_precision_for_backend,
    )

    assert _native_precision_for_backend(
        "hailo10h", "uint8_cast_fp16",
    ) == "uint8_dequant_fp16"
    assert _native_precision_for_backend(
        "hailo8", "uint8_cast_fp16",
    ) == "uint8_cast_fp16"
    assert _native_precision_for_backend(
        "hailo10h", "float32_layout_fp16",
    ) == "float32_layout_fp16"
    remote_runner = (
        ROOT / "scripts/native_producer_e2e_eval_runner.py"
    ).read_text(encoding="utf-8")
    assert "runtime_precision_default='uint8_dequant_fp16'" in remote_runner


def test_hailo10_vstreams_override_is_rejected_and_quant_info_is_reported() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "Hailo-10/15 requires the InferModel runtime" in template
    assert '"runtime_input_quantization"' in template
    assert '"runtime_output_quantization"' in template


def test_hailo10_engine_bridge_must_match_exact_hef_quant_info(
    tmp_path: Path,
) -> None:
    runner = _load_python_script(
        ROOT / "scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
        "_v27916_hailo10_exact_engine_quant",
    )
    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    engine = engine_dir / "part2_uint8_dequant_fp16.engine"
    engine.write_bytes(b"engine")
    (engine_dir / "native_trt_meta.json").write_text(
        '{"uint8_cast_bridge":{"schema":"onnx-splitpoint/uint8-dequant-bridge",'
        '"scale":0.125,"zero_point":11,"boundary_layout":{"effective":"as_input"}}}',
        encoding="utf-8",
    )
    contract = runner._require_exact_hailo10_engine_quantization(
        engine, {"scale": 0.125, "zero_point": 11.0},
        boundary_layout="as_input",
    )
    assert contract["dequant_scale"] == pytest.approx(0.125)
    with pytest.raises(RuntimeError, match="QuantInfo mismatch"):
        runner._require_exact_hailo10_engine_quantization(
            engine, {"scale": 0.25, "zero_point": 11.0},
            boundary_layout="as_input",
        )
    with pytest.raises(RuntimeError, match="boundary layout mismatch"):
        runner._require_exact_hailo10_engine_quantization(
            engine, {"scale": 0.125, "zero_point": 11.0},
            boundary_layout="memory_nhwc_to_nchw",
        )
    (engine_dir / "native_trt_meta.json").write_text(
        '{"uint8_cast_bridge":{"schema":"wrong","scale":0.125,'
        '"zero_point":11,"boundary_layout":{"effective":"as_input"}}}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not declare"):
        runner._require_exact_hailo10_engine_quantization(
            engine, {"scale": 0.125, "zero_point": 11.0},
            boundary_layout="as_input",
        )


def test_central_hailo_backend_forbids_hailo10_vstreams_only() -> None:
    source = (
        ROOT / "onnx_splitpoint_tool/runners/backends/hailo_backend.py"
    ).read_text(encoding="utf-8")
    assert 'if runtime_api == "vstreams" and (' in source
    assert '"hailo10" in hw_arch_l or "hailo15" in hw_arch_l' in source
    assert "Hailo-10/15 requires runtime_api='infer_model'" in source


@pytest.mark.parametrize("task", ["classification", "detection"])
def test_native_hailo10_runner_quantizes_the_correct_onnx_input_domain(
    tmp_path: Path, task: str,
) -> None:
    pillow = pytest.importorskip("PIL.Image")
    runner = _load_python_script(
        ROOT / "scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
        f"_v27916_hailo10_native_runner_{task}",
    )
    image_path = tmp_path / f"{task}.png"
    pillow.fromarray(
        np.asarray([[[0, 128, 255]]], dtype=np.uint8), mode="RGB",
    ).save(image_path)
    seen: dict[str, np.ndarray] = {}

    class _Session:
        @staticmethod
        def quantize_input(name: str, values: np.ndarray) -> np.ndarray:
            seen[name] = np.array(values, copy=True)
            return np.full(values.shape, 7, dtype=np.uint8)

    prepared = SimpleNamespace(
        input_names=["images"],
        handle=SimpleNamespace(
            runtime_input_shapes={"images": (1, 1, 1, 3)},
            input_shapes={},
            session=_Session(),
        ),
    )

    result = runner._make_input(
        prepared, True, image=str(image_path), task=task,
    )["images"]

    source = seen["images"]
    assert source.dtype == np.float32
    assert source.flags.c_contiguous
    if task == "classification":
        assert float(source.min()) < 0.0
        assert float(source.max()) > 1.0
    else:
        np.testing.assert_allclose(
            source.reshape(-1), [0.0, 128.0 / 255.0, 1.0], rtol=0, atol=1e-7,
        )
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous


def test_hailo10_quality_input_dump_uses_rgb_not_hef_codes(
    tmp_path: Path,
) -> None:
    pillow = pytest.importorskip("PIL.Image")
    runner = _load_python_script(
        ROOT / "scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
        "_v27916_hailo10_quality_rgb_reference",
    )
    image_path = tmp_path / "classification.png"
    pillow.fromarray(
        np.asarray([[[0, 128, 255]]], dtype=np.uint8), mode="RGB",
    ).save(image_path)
    hef_codes = np.full((1, 1, 1, 3), 7, dtype=np.uint8)

    rgb, shape = runner._reference_input_hwc_uint8(
        input_image=str(image_path),
        inputs={"input": hef_codes},
        preprocess_mode="resize",
        letterbox_pad_value=114,
    )

    assert shape == [1, 1, 3]
    assert rgb is not None
    np.testing.assert_array_equal(rgb.reshape(-1), [0, 128, 255])


def test_native_hailo10_runner_rejects_uint16_for_the_uint8_bridge() -> None:
    runner = _load_python_script(
        ROOT / "scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
        "_v27916_hailo10_native_runner_output_format",
    )
    prepared = SimpleNamespace(
        handle=SimpleNamespace(
            session=SimpleNamespace(
                describe_io=lambda: {"runtime_output_format": "uint16"},
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="required UINT8"):
        runner._selected_hailo_output_format(prepared, True)

    prepared.handle.session.describe_io = lambda: {
        "runtime_output_format": "float32",
    }
    with pytest.raises(RuntimeError, match="required UINT8"):
        runner._selected_hailo_output_format(prepared, True)


@pytest.mark.parametrize(
    ("model_id", "case_id", "expected_task"),
    [("resnet50", "b052", "classification"), ("yolo26s", "b024", "detection")],
)
def test_hailo10_quality_policy_selects_native_uint8_dequant_bridge(
    model_id: str, case_id: str, expected_task: str,
) -> None:
    policy = known_native_split_policy(
        model_id=model_id,
        case_id=case_id,
        setup_id="orin_nx_hailo10_01",
        backend="hailo10h_to_trt",
    )
    assert policy is not None
    assert policy["task"] == expected_task
    assert policy["precision"] == "uint8_dequant_fp16"
    assert policy["hailo_format"] == "uint8"
    assert policy["boundary_dtype"] == "uint8"
    assert policy["quantization_policy"] == "from_exact_part1_boundary_metadata"


def test_hailo10_materialized_boundary_binds_exact_quantization() -> None:
    policy = known_native_split_policy(
        model_id="yolo26s", case_id="b024",
        setup_id="orin_nx_hailo10_01", backend="hailo10h_to_trt",
    )
    assert policy is not None
    part1_bytes = b"test-hef"
    part1 = {
        "path": "/test/part1.hef",
        "sha256": hashlib.sha256(part1_bytes).hexdigest(),
        "size_bytes": len(part1_bytes),
    }
    metadata = {
        "schema": "onnx-splitpoint/native-part1-boundary-metadata",
        "schema_version": 1,
        "model_id": "yolo26s",
        "case_id": "b024",
        "setup_id": "orin_nx_hailo10_01",
        "backend": "hailo10h_to_trt",
        "part1_artifact_sha256": part1["sha256"],
        "part1_artifact_size_bytes": part1["size_bytes"],
        "boundary_tensor_count": 1,
        "boundary_tensor": {
            "name": "cut",
            "shape": [80, 80, 128],
            "canonical_part2_shape": [1, 128, 80, 80],
            "dtype": "uint8",
            "quantization": {
                "source": "hailort_hef_output_vstream_info",
                "scale": 0.09418510645627975,
                "zero_point": 180.0,
            },
        },
        "boundary_layout": "memory_nhwc_to_nchw",
        "boundary_transform": "uint8_dequant_then_layout",
    }
    metadata["metadata_sha256"] = canonical_json_sha256(metadata)
    metadata_artifact = {
        "path": "/test/boundary.json",
        "sha256": "a" * 64,
        "size_bytes": 123,
    }

    selection = materialize_native_split_preselection(
        policy=policy,
        part1_artifact=part1,
        boundary_metadata=metadata,
        boundary_metadata_artifact=metadata_artifact,
    )

    assert selection["boundary_transform"] == "uint8_dequant_then_layout"
    assert selection["dequant_scale"] == pytest.approx(0.09418510645627975)
    assert selection["dequant_zero_point"] == pytest.approx(180.0)
