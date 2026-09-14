from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenPostprocessError,
    build_frozen_postprocess_contract,
    tensor_signature,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from scripts import native_full_baseline_eval_runner as full_runner
from scripts import smoke_hailo10_hef_runner as hailo_runner


ROOT = Path(__file__).resolve().parents[1]
DEEPX_HOTLOOP = ROOT / "scripts" / "native_deepx_full_energy_hotloop.py"
REMOTE_SCRIPTS = (
    ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
)


def _rank5_outputs() -> dict[str, np.ndarray]:
    shapes = {
        "output": (1, 3, 80, 80, 85),
        "clone_1": (1, 3, 40, 40, 85),
        "clone_2": (1, 3, 20, 20, 85),
    }
    outputs: dict[str, np.ndarray] = {}
    for offset, (name, shape) in enumerate(shapes.items()):
        outputs[name] = (
            np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            + np.float32(offset)
        )
    return outputs


def _packed_hwc_outputs(
    logical: dict[str, np.ndarray], *, batched: bool = False,
) -> dict[str, np.ndarray]:
    packed: dict[str, np.ndarray] = {}
    for name, value in logical.items():
        _batch, anchors, height, width, channels = value.shape
        physical = value[0].transpose(1, 2, 0, 3).reshape(
            height, width, anchors * channels,
        )
        packed[name] = physical[None, ...] if batched else physical
    return packed


def _rank5_contract() -> dict[str, object]:
    zeros = {
        name: np.zeros_like(value)
        for name, value in _rank5_outputs().items()
    }
    return build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=zeros,
        input_hw=[640, 640],
        original_wh=[32, 18],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )


def test_deepx_hotloop_bootstraps_repo_root_from_an_isolated_process(
    tmp_path: Path,
) -> None:
    code = (
        "import runpy\n"
        f"runpy.run_path({str(DEEPX_HOTLOOP)!r}, run_name='deepx_import_probe')\n"
        "from onnx_splitpoint_tool.native_detection_postprocess "
        "import FrozenDetectionPostprocessor\n"
        "print(FrozenDetectionPostprocessor.__name__)\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    process = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "FrozenDetectionPostprocessor"


def test_deepx_launcher_prefixes_repo_root_in_child_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")
    child_env = full_runner._deepx_hotloop_env()
    assert child_env["PYTHONPATH"].split(os.pathsep) == [
        str(ROOT),
        str(ROOT / "scripts"),
        "/existing/pythonpath",
    ]
    assert child_env["PYTHONUNBUFFERED"] == "1"
    assert os.environ["PYTHONPATH"] == "/existing/pythonpath"
    assert "env=_deepx_hotloop_env()" in inspect.getsource(
        full_runner._energy_workload_only
    )


def test_deepx_and_hailo_hotloop_remote_mirrors_are_exact() -> None:
    for name in (
        "native_deepx_full_energy_hotloop.py",
        "native_full_baseline_eval_runner.py",
        "smoke_hailo10_hef_runner.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            REMOTE_SCRIPTS / name
        ).read_bytes()


@pytest.mark.parametrize("batched", [False, True])
def test_hailo_packed_heads_canonicalize_to_sealed_rank5(
    batched: bool,
) -> None:
    expected = _rank5_outputs()
    packed = _packed_hwc_outputs(expected, batched=batched)
    contract = _rank5_contract()
    canonical = hailo_runner._canonicalize_frozen_postprocess_outputs(
        packed, contract,
    )

    assert tensor_signature(canonical) == contract[
        "raw_output_tensor_signature"
    ]
    for name, expected_value in expected.items():
        assert canonical[name].flags.c_contiguous
        np.testing.assert_array_equal(canonical[name], expected_value)


def test_hailo_physical_hwc_contract_remains_identity() -> None:
    packed = _packed_hwc_outputs(_rank5_outputs())
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=packed,
        input_hw=[640, 640],
        original_wh=[32, 18],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    canonical = hailo_runner._canonicalize_frozen_postprocess_outputs(
        packed, contract,
    )
    assert tensor_signature(canonical) == contract[
        "raw_output_tensor_signature"
    ]
    for name, value in packed.items():
        assert canonical[name] is value


@pytest.mark.parametrize(
    "mutation",
    [
        "channel_count",
        "height",
        "dtype",
        "missing_name",
        "extra_name",
        "name_order",
        "contract_index",
        "contract_duplicate_name",
    ],
)
def test_hailo_packed_head_canonicalization_fails_closed(
    mutation: str,
) -> None:
    packed = _packed_hwc_outputs(_rank5_outputs())
    contract = _rank5_contract()
    if mutation == "channel_count":
        packed["output"] = packed["output"][:, :, :-1]
    elif mutation == "height":
        packed["output"] = packed["output"][:-1, :, :]
    elif mutation == "dtype":
        packed["output"] = packed["output"].astype(np.float16)
    elif mutation == "missing_name":
        packed.pop("clone_2")
    elif mutation == "extra_name":
        packed["extra"] = np.zeros((1,), dtype=np.float32)
    elif mutation == "name_order":
        packed = {
            "clone_1": packed["clone_1"],
            "output": packed["output"],
            "clone_2": packed["clone_2"],
        }
    elif mutation == "contract_index":
        contract["raw_output_tensor_signature"]["tensors"][1]["index"] = 2
    elif mutation == "contract_duplicate_name":
        contract["raw_output_tensor_signature"]["tensors"][1][
            "name"
        ] = "output"

    with pytest.raises(
        FrozenPostprocessError,
        match="frozen_postprocess_tensor_signature_mismatch",
    ):
        hailo_runner._canonicalize_frozen_postprocess_outputs(
            packed, contract,
        )


def test_hailo_timed_postprocess_consumes_canonicalized_outputs() -> None:
    source = inspect.getsource(hailo_runner.main)
    assert "_canonicalize_frozen_postprocess_outputs(" in source
    assert "canonical_outputs, original_wh=original_wh" in source
