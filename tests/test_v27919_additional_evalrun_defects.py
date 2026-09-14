from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from onnx_splitpoint_tool.native_split_quality import (
    known_native_split_policy,
)
from onnx_splitpoint_tool.runners import native_split_quality_runtime
from onnx_splitpoint_tool.workflow.runner import (
    _native_model_family_task_v27919,
    _native_selection_contract_runs_v270e,
)


@pytest.mark.parametrize(
    ("model_id", "family"),
    [
        ("mobilenet_v3_large", "mobilenet"),
        ("regnet_x_1_6gf", "regnet"),
    ],
)
@pytest.mark.parametrize(
    ("backend", "precision", "boundary_dtype"),
    [
        ("hailo8_to_trt", "float32_layout_fp16", "float32"),
        ("hailo10h_to_trt", "uint8_dequant_fp16", "uint8"),
        ("deepx_to_trt", "float32_layout_fp16", "float32"),
    ],
)
def test_known_classification_models_have_native_split_policy(
    model_id: str,
    family: str,
    backend: str,
    precision: str,
    boundary_dtype: str,
) -> None:
    policy = known_native_split_policy(
        model_id=model_id,
        case_id="b027",
        setup_id="classification_setup",
        backend=backend,
    )

    assert policy is not None
    assert policy["model_family"] == family
    assert policy["task"] == "classification"
    assert policy["precision"] == precision
    assert policy["boundary_dtype"] == boundary_dtype


def test_unknown_model_still_has_no_inferred_native_split_policy() -> None:
    assert known_native_split_policy(
        model_id="unregistered_model",
        case_id="b001",
        setup_id="classification_setup",
        backend="hailo8_to_trt",
    ) is None


@pytest.mark.parametrize("backend", ["hailo8", "hailo10h", "deepx"])
def test_mixed_classification_cohort_reaches_native_contract_denominator(
    backend: str,
) -> None:
    selected = {
        "resnet50": ["b002"],
        "mobilenet_v3_large": ["b027", "b056"],
        "regnet_x_1_6gf": ["b023"],
    }

    contracts = _native_selection_contract_runs_v270e(backend, selected)
    by_model = {
        str(contract["models"][0]): contract for contract in contracts
    }
    assert set(by_model) == set(selected)
    expected_rows = {
        (model, case)
        for model, contract in by_model.items()
        for case in contract["case_map"][model]
    }
    assert expected_rows == {
        (model, case)
        for model, cases in selected.items()
        for case in cases
    }
    assert all(contract["task"] == "classification" for contract in contracts)
    assert all(
        _native_model_family_task_v27919(model)[1] == "classification"
        for model in selected
    )
    if backend == "hailo8":
        assert all(
            contract["precision"] == "float32_layout_fp16"
            and contract["hailo_format"] == "float32"
            for contract in contracts
        )


def test_unknown_native_family_is_not_inferred_as_detection() -> None:
    assert _native_model_family_task_v27919("unregistered_model") == ("", "")
    assert _native_selection_contract_runs_v270e(
        "hailo8", {"unregistered_model": ["b001"]},
    ) == []


def test_hailo8_quantized_hef_storage_can_expose_float32_vstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHEF:
        def __init__(self, _path: str) -> None:
            pass

        def get_output_vstream_infos(self) -> list[object]:
            return [
                SimpleNamespace(
                    name="resnet50_part1/conv24",
                    shape=(28, 28, 512),
                    format=SimpleNamespace(type="UINT8"),
                    quant_info=SimpleNamespace(qp_scale=0.03125, qp_zp=7.0),
                )
            ]

    monkeypatch.delenv("HAILO_PY", raising=False)
    monkeypatch.setitem(sys.modules, "hailo_platform", SimpleNamespace(HEF=FakeHEF))
    part1 = tmp_path / "part1.hef"
    part1.write_bytes(b"quantized-hef")

    metadata = native_split_quality_runtime._hailo_metadata(
        part1=part1,
        part2_input={
            "name": "add_6",
            "shape": [1, 512, 28, 28],
        },
        policy={
            "boundary_dtype": "float32",
            "quantization_policy": "none",
        },
    )

    assert metadata["dtype"] == "float32"
    assert metadata["shape"] == [28, 28, 512]
    assert "quantization" not in metadata


def test_quantized_runtime_still_rejects_wrong_hef_storage_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHEF:
        def __init__(self, _path: str) -> None:
            pass

        def get_output_vstream_infos(self) -> list[object]:
            return [
                SimpleNamespace(
                    name="part1/conv8",
                    shape=(80, 80, 128),
                    format=SimpleNamespace(type="FLOAT32"),
                    quant_info=SimpleNamespace(qp_scale=0.03125, qp_zp=7.0),
                )
            ]

        def get_network_group_names(self) -> list[str]:
            return ["network"]

        def get_output_stream_infos(self, _group: str) -> list[object]:
            return [SimpleNamespace(
                name="raw/conv8",
                format=SimpleNamespace(type="UINT16"),
            )]

        def get_stream_names_from_vstream_name(
            self, _name: str, _group: str,
        ) -> list[str]:
            return ["raw/conv8"]

    monkeypatch.delenv("HAILO_PY", raising=False)
    monkeypatch.setitem(sys.modules, "hailo_platform", SimpleNamespace(HEF=FakeHEF))
    part1 = tmp_path / "part1.hef"
    part1.write_bytes(b"quantized-hef")

    with pytest.raises(
        RuntimeError,
        match=(
            "native_split_quality_hef_boundary_dtype_mismatch:"
            "expected=uint8:observed=uint16"
        ),
    ):
        native_split_quality_runtime._hailo_metadata(
            part1=part1,
            part2_input={
                "name": "conv8",
                "shape": [1, 128, 80, 80],
            },
            policy={
                "boundary_dtype": "uint8",
                "quantization_policy": (
                    "from_exact_part1_boundary_metadata"
                ),
            },
        )


def test_quantized_runtime_rejects_unresolved_native_storage_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHEF:
        def __init__(self, _path: str) -> None:
            pass

        def get_output_vstream_infos(self) -> list[object]:
            return [SimpleNamespace(
                name="part1/conv8",
                shape=(80, 80, 128),
                format=SimpleNamespace(type="FLOAT32"),
                quant_info=SimpleNamespace(qp_scale=0.03125, qp_zp=7.0),
            )]

    monkeypatch.delenv("HAILO_PY", raising=False)
    monkeypatch.setitem(sys.modules, "hailo_platform", SimpleNamespace(HEF=FakeHEF))
    part1 = tmp_path / "part1.hef"
    part1.write_bytes(b"quantized-hef")

    with pytest.raises(
        RuntimeError,
        match=(
            "native_split_quality_hef_boundary_dtype_mismatch:"
            "expected=uint8:observed=unresolved"
        ),
    ):
        native_split_quality_runtime._hailo_metadata(
            part1=part1,
            part2_input={
                "name": "conv8",
                "shape": [1, 128, 80, 80],
            },
            policy={
                "boundary_dtype": "uint8",
                "quantization_policy": (
                    "from_exact_part1_boundary_metadata"
                ),
            },
        )


def test_quantized_runtime_accepts_legacy_uint8_vstream_without_low_level_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHEF:
        def __init__(self, _path: str) -> None:
            pass

        def get_output_vstream_infos(self) -> list[object]:
            return [SimpleNamespace(
                name="part1/conv8",
                shape=(80, 80, 128),
                format=SimpleNamespace(type="UINT8"),
                quant_info=SimpleNamespace(qp_scale=0.03125, qp_zp=7.0),
            )]

    monkeypatch.delenv("HAILO_PY", raising=False)
    monkeypatch.setitem(sys.modules, "hailo_platform", SimpleNamespace(HEF=FakeHEF))
    part1 = tmp_path / "part1.hef"
    part1.write_bytes(b"quantized-hef")

    metadata = native_split_quality_runtime._hailo_metadata(
        part1=part1,
        part2_input={"name": "conv8", "shape": [1, 128, 80, 80]},
        policy={
            "boundary_dtype": "uint8",
            "quantization_policy": "from_exact_part1_boundary_metadata",
        },
    )

    assert metadata["hef_vstream_format_type"] == "uint8"
    assert metadata["hef_native_storage_dtype"] == "uint8"


def test_quantized_runtime_accepts_legacy_uint8_vstream_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = tmp_path / "hailo_platform.py"
    stub.write_text(
        """
class Value:
    def __init__(self, **values):
        self.__dict__.update(values)

class HEF:
    def __init__(self, _path):
        pass
    def get_output_vstream_infos(self):
        return [Value(
            name='part1/conv8', shape=(80, 80, 128),
            format=Value(type='UINT8'),
            quant_info=Value(qp_scale=0.03125, qp_zp=7.0),
        )]
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HAILO_PY", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    part1 = tmp_path / "part1.hef"
    part1.write_bytes(b"quantized-hef")

    metadata = native_split_quality_runtime._hailo_metadata(
        part1=part1,
        part2_input={"name": "conv8", "shape": [1, 128, 80, 80]},
        policy={
            "boundary_dtype": "uint8",
            "quantization_policy": "from_exact_part1_boundary_metadata",
        },
    )

    assert metadata["hef_vstream_format_type"] == "uint8"
    assert metadata["hef_native_storage_dtype"] == "uint8"


def test_quantized_runtime_accepts_native_uint8_behind_float32_vstream_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = tmp_path / "hailo_platform.py"
    stub.write_text(
        """
class Value:
    def __init__(self, **values):
        self.__dict__.update(values)

class HEF:
    def __init__(self, _path):
        pass
    def get_output_vstream_infos(self):
        return [Value(
            name='part1/conv8', shape=(80, 80, 128),
            format=Value(type='FLOAT32'),
            quant_info=Value(qp_scale=0.03125, qp_zp=7.0),
        )]
    def get_network_group_names(self):
        return ['network']
    def get_output_stream_infos(self, _group):
        return [Value(name='raw/conv8', format=Value(type='UINT8'))]
    def get_stream_names_from_vstream_name(self, _name, _group):
        return ['raw/conv8']
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HAILO_PY", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    part1 = tmp_path / "part1.hef"
    part1.write_bytes(b"quantized-hef")

    metadata = native_split_quality_runtime._hailo_metadata(
        part1=part1,
        part2_input={
            "name": "conv8",
            "shape": [1, 128, 80, 80],
        },
        policy={
            "boundary_dtype": "uint8",
            "quantization_policy": "from_exact_part1_boundary_metadata",
        },
    )

    assert metadata["dtype"] == "uint8"
    assert metadata["hef_vstream_format_type"] == "float32"
    assert metadata["hef_native_storage_dtype"] == "uint8"
    assert metadata["quantization"] == {
        "source": "hailort_hef_output_vstream_info",
        "scale": 0.03125,
        "zero_point": 7.0,
    }
