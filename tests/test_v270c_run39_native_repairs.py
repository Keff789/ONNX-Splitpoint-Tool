from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from onnx_splitpoint_tool.native_command_contract import _quality_backend
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    FrozenPostprocessError,
    canonical_json_sha256,
    frozen_postprocess_invariant_identity,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.native_split_quality import (
    canonical_native_split_backend,
    resolve_native_boundary_layout,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/v270c/run39_native_repairs.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _load_script(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _reseal_frozen_contract(contract: dict) -> dict:
    value = copy.deepcopy(contract)
    value.pop("contract_sha256", None)
    value["invariant_identity"] = frozen_postprocess_invariant_identity(value)
    value["invariant_contract_sha256"] = canonical_json_sha256(
        value["invariant_identity"]
    )
    value["contract_sha256"] = canonical_json_sha256(value)
    return value


def test_run39_source_aliases_join_but_foreign_backend_does_not() -> None:
    row = _fixture()["deepx_alias_join"]
    setup = row["setup_id"]
    assert canonical_native_split_backend(
        row["binding_source_run_id"], setup,
    ) == row["central_source_run_id"]
    assert _quality_backend(
        row["binding_source_run_id"], setup,
    ) == row["command_backend"]
    assert canonical_native_split_backend(
        "deepx_to_trt", "orin_nx_hailo8_01",
    ) == "deepx_to_trt"
    assert _quality_backend(
        "deepx_to_trt", "orin_nx_hailo8_01",
    ) == "deepx_to_trt"
    assert canonical_native_split_backend(
        "deepx_to_trt", "orin_nx_hailo8_01",
    ) != canonical_native_split_backend(
        "hailo8_to_trt", "orin_nx_hailo8_01",
    )


@pytest.mark.parametrize("boundary", _fixture()["hailo10_boundaries"])
def test_run39_hailo10_shapes_resolve_exact_nhwc_to_nchw(
    boundary: dict,
) -> None:
    assert resolve_native_boundary_layout(
        boundary["runtime_shape"], boundary["canonical_part2_shape"],
    ) == boundary["expected_270c_layout"]
    h, w, c = boundary["runtime_shape"]
    sentinel = np.arange(h * w * c, dtype=np.float32).reshape(h, w, c)
    transformed = np.transpose(sentinel[None, ...], (0, 3, 1, 2))
    assert transformed.shape == tuple(boundary["canonical_part2_shape"])
    assert transformed[0, c - 1, h - 1, w - 1] == sentinel[h - 1, w - 1, c - 1]


def test_hailo10_layout_resolution_fails_closed_when_unknown_or_ambiguous() -> None:
    with pytest.raises(ValueError, match="boundary_layout_unresolved"):
        resolve_native_boundary_layout([28, 512, 28], [1, 512, 28, 28])
    with pytest.raises(ValueError, match="boundary_layout_ambiguous"):
        resolve_native_boundary_layout([32, 32, 32], [1, 32, 32, 32])
    runtime_source = (
        ROOT / "onnx_splitpoint_tool/runners/native_split_quality_runtime.py"
    ).read_text(encoding="utf-8")
    assert '"resolved_boundary_layout": boundary_layout' in runtime_source
    assert '"resolved_boundary_transform": boundary_transform' in runtime_source


def test_hailo10_strict_extractor_preserves_raw_hwc_memory() -> None:
    module = _load_script(
        "_osp_v270c_hailo10_runner",
        "scripts/native_hailo10_trt_e2e_from_benchmarkset.py",
    )
    raw = np.arange(1 * 2 * 3 * 4, dtype=np.float32).reshape(1, 2, 3, 4)

    class Output:
        def get_buffer(self):
            return raw

    class Session:
        _hef_output_names = ["runtime"]
        _output_name_hef_to_canonical = {"runtime": "boundary"}
        output_shapes = {"boundary": (1, 4, 2, 3)}

        @staticmethod
        def _binding_output(_binding, _name):
            return Output()

    module._STRICT_SPLIT_BOUNDARY = {
        "name": "boundary", "runtime_name": "runtime",
        "shape": [2, 3, 4], "dtype": "float32",
    }
    extracted = module._extract_slot_outputs(Session(), {"binding": object()})
    assert extracted["boundary"].shape == (2, 3, 4)
    np.testing.assert_array_equal(extracted["boundary"], raw[0])


def test_run39_vendored_frozen_postprocess_paths_verify_by_exact_sha() -> None:
    contract = _fixture()["deepx_yolov7_prepared_feed"][
        "frozen_host_postprocess_contract"
    ]
    verified = verify_frozen_postprocess_contract(contract)
    assert verified["contract_sha256"] == contract["contract_sha256"]
    assert verified["legacy_activation_strategy_unbound"] is True
    with pytest.raises(
        FrozenPostprocessError,
        match="legacy_unbound_activation_contract_nonclaim_runtime",
    ):
        FrozenDetectionPostprocessor(contract)

    unknown_path = copy.deepcopy(contract)
    unknown_path["implementation_artifacts"]["yolo_harness"][
        "relative_path"
    ] = "unknown/yolo.py"
    unknown_path = _reseal_frozen_contract(unknown_path)
    with pytest.raises(
        FrozenPostprocessError,
        match="frozen_postprocess_implementation_sha256_mismatch",
    ):
        verify_frozen_postprocess_contract(unknown_path)


def test_run39_deepx_nested_projection_promotes_completion_and_fails_on_conflict() -> None:
    module = _load_script(
        "_osp_v270c_native_full_runner",
        "scripts/native_full_baseline_eval_runner.py",
    )
    fixture = _fixture()["deepx_yolov7_prepared_feed"]
    nested = dict(fixture["nested"])
    nested["prepared_feed_contract_version"] = fixture["top_level"][
        "prepared_feed_contract_version"
    ]
    nested["postprocess_completion_verified"] = True
    nested_only = {
        "run_id": fixture["top_level"]["run_id"],
        "variant": fixture["top_level"]["variant"],
        "runtime_ok": True,
        "prepared_feed_contract_version": fixture["top_level"][
            "prepared_feed_contract_version"
        ],
        "deepx_prepared_feed_benchmark": nested,
    }
    projection, conflicts = module._deepx_prepared_feed_projection(nested_only)
    assert conflicts == []
    assert projection["fps_makespan"] == pytest.approx(7.507413545968622)
    assert projection["completed_frames"] == 1000
    assert projection["completed_work_units"] == 1000
    assert projection["postprocess_completed_frames"] == 1000
    assert projection["postprocess_completion_verified"] is True
    assert projection["performance_benchmark_source"] == "dx_engine_prepared_feed"

    conflicting = dict(nested_only)
    conflicting["completed_frames"] = 999
    _, conflicts = module._deepx_prepared_feed_projection(conflicting)
    assert [item["field"] for item in conflicts] == ["completed_frames"]


@pytest.mark.parametrize(
    "name",
    [
        "native_producer_e2e_eval_runner.py",
        "native_deepx_trt_e2e_from_benchmarkset.py",
        "native_hailo10_trt_e2e_from_benchmarkset.py",
        "native_hailo_trt_fifo_from_benchmarkset.py",
        "native_full_baseline_eval_runner.py",
        "native_producer_validate_visualize.py",
        "native_producer_final_report.py",
        "run_evalrun_native_producer_variants.py",
        "update_evalset_native_producers.py",
    ],
)
def test_changed_remote_script_mirrors_are_byte_identical(name: str) -> None:
    assert (ROOT / "scripts" / name).read_bytes() == (
        ROOT / "onnx_splitpoint_tool/resources/remote_scripts" / name
    ).read_bytes()
