from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest


TEMPLATE = Path("onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt")


def _template_namespace(*names: str) -> dict[str, Any]:
    tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
    wanted = set(names)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    namespace: dict[str, Any] = {
        "Any": Any,
        "DEEPX_PREPARED_FEED_CONTRACT_VERSION": (
            "deepx-sealed-runtime-input-v3"
        ),
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Path": Path,
        "json": json,
        "os": __import__("os"),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<benchmark-suite-template>", "exec"), namespace)
    return namespace


def _real_full_contract(
    shape: list[int], layout: str = "HWC", task: str = "classification",
) -> dict[str, Any]:
    mode = "resize" if task == "classification" else "letterbox"
    pad = 0 if mode == "resize" else 114
    return {
        "backend": "deepx_m1",
        "variant": "full",
        "input": {
            "name": "input",
            "shape": shape,
            "dtype": "uint8",
            "layout": layout,
            "normalization": "embedded_dxcom_preprocessing",
            "color_space": "RGB",
            "task": task,
            "preprocess_mode_requested": "auto",
            "preprocess_mode": mode,
            "preprocess_mode_effective": mode,
            "letterbox_pad_value_requested": 114,
            "letterbox_pad_value_effective": pad,
            "letterbox_pad_value": pad,
        },
    }


def _sealed_prepared_input_binding(
    tmp_path: Path,
    *,
    feed: np.ndarray,
    contract: dict[str, Any],
    task: str,
    source_image: Path,
) -> tuple[Path, dict[str, Any]]:
    """Build the complete binding returned by the sealed-input loader."""
    from onnx_splitpoint_tool.preprocessing_contract import (
        canonical_image_preprocessing_contract,
        preprocessing_contract_sha256,
        runtime_numeric_input_identity,
    )

    inp = dict(contract["input"])
    manifest_path = tmp_path / "native_full_input_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    tensor_path = tmp_path / "runtime_input.bin"
    tensor_bytes = np.ascontiguousarray(feed).tobytes()
    tensor_path.write_bytes(tensor_bytes)
    tensor_sha256 = hashlib.sha256(tensor_bytes).hexdigest()
    source_image_sha256 = hashlib.sha256(source_image.read_bytes()).hexdigest()
    target_hw = (
        [int(feed.shape[0]), int(feed.shape[1])]
        if str(inp["layout"]).upper() == "HWC"
        else [int(feed.shape[1]), int(feed.shape[2])]
    )
    preprocessing_identity = canonical_image_preprocessing_contract(
        task, target_hw,
    )
    preprocessing_sha256 = preprocessing_contract_sha256(
        preprocessing_identity,
    )
    numeric_identity, numeric_sha256 = runtime_numeric_input_identity(
        backend="native_full_deepx",
        task=task,
        preprocessing_contract_sha256_value=preprocessing_sha256,
        runtime_input_name=str(inp["name"]),
        runtime_input_shape=[int(value) for value in feed.shape],
        runtime_input_dtype=str(feed.dtype),
        runtime_input_layout=str(inp["layout"]),
        runtime_color_space=str(inp["color_space"]),
        runtime_normalization=str(inp["normalization"]),
    )
    preprocess_mode = str(inp["preprocess_mode"])
    preprocess = {
        "mode": f"{preprocess_mode}_rgb_uint8",
        "pad_value": int(inp["letterbox_pad_value"]),
        "layout": str(inp["layout"]),
        "normalization": str(inp["normalization"]),
        "color_space": str(inp["color_space"]),
    }
    return manifest_path, {
        "prepared_input_manifest": str(manifest_path.resolve()),
        "prepared_input_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "prepared_input_file": str(tensor_path.resolve()),
        "prepared_input_sha256": tensor_sha256,
        "prepared_input_file_sha256": tensor_sha256,
        "prepared_input_bytes": len(tensor_bytes),
        "prepared_input_name": str(inp["name"]),
        "prepared_input_shape": [int(value) for value in feed.shape],
        "prepared_input_dtype": str(feed.dtype),
        "prepared_input_layout": str(inp["layout"]),
        "prepared_input_source_image_id": source_image.name,
        "prepared_input_source_image_sha256": source_image_sha256,
        "runtime_color_space": str(inp["color_space"]),
        "runtime_normalization": str(inp["normalization"]),
        "runtime_preprocess_mode": preprocess_mode,
        "runtime_preprocessing_identity": preprocessing_identity,
        "runtime_preprocessing_sha256": preprocessing_sha256,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": numeric_sha256,
        "prepared_input_binding_verified": True,
        "prepared_input_source": "sealed_semantic_dump_runtime_tensor",
        "preprocess": preprocess,
    }


def _install_shared_input_loader(
    monkeypatch: pytest.MonkeyPatch, feed: np.ndarray,
    *, expected_setup_id: str = "",
) -> None:
    package = types.ModuleType("splitpoint_runners")
    package.__path__ = []  # type: ignore[attr-defined]
    module = types.ModuleType("splitpoint_runners.native_full_input")
    def _load(*_args, **kwargs):
        if expected_setup_id:
            assert kwargs["expected_setup_id"] == expected_setup_id
        return {"runtime_input": feed.copy()}

    module.load_sealed_deepx_native_full_input = _load
    module.prepare_and_seal_deepx_native_full_input = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixture supplies an existing sealed manifest")
        )
    )
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(
        sys.modules, "splitpoint_runners.native_full_input", module,
    )


def test_new_suite_materialization_uses_fixed_template_and_version_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from onnx_splitpoint_tool.gui import controller

    # Keep this unit test focused on the harness; runner-library vendoring has
    # its own coverage and does not affect the generated script bytes.  The
    # current materializer nevertheless verifies that its mandatory
    # Quality-FIRST runtime closure exists, so the stub must preserve that
    # postcondition instead of pretending that no files were vendored.
    def _stub_runner_lib(dst: Path) -> None:
        runner_dir = Path(dst) / "splitpoint_runners"
        runner_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "native_split_quality_runtime.py",
            "native_split_quality.py",
                "native_command_contract.py",
                "native_trt_from_benchmarkset.py",
                "native_detection_postprocess.py",
                "native_full_input.py",
                "preprocessing_contract.py",
            ):
            (runner_dir / name).write_text("# test runtime stub\n", encoding="utf-8")

    monkeypatch.setattr(controller, "_copy_runner_lib", _stub_runner_lib)
    script_path = Path(
        controller.write_benchmark_suite_script(
            tmp_path, bench_json_name="benchmark_set.json",
        )
    )
    generated = script_path.read_text(encoding="utf-8")
    expected = TEMPLATE.read_text(encoding="utf-8").replace(
        "__BENCH_JSON__", "benchmark_set.json",
    )

    assert generated == expected
    assert (
        'DEEPX_PREPARED_FEED_CONTRACT_VERSION = "deepx-sealed-runtime-input-v3"'
        in generated
    )
    assert "def _deepx_input_geometry" in generated
    assert 'expected_rank = {"HWC": 3, "CHW": 3, "NHWC": 4, "NCHW": 4}' in generated

    generator = Path("onnx_splitpoint_tool/workflow/generator_binding.py").read_text(
        encoding="utf-8",
    )
    assert "write_benchmark_suite_script(suite_dir" in generator


def test_remote_full_runner_requires_fixed_suite_attestation() -> None:
    source = Path("scripts/native_full_baseline_eval_runner.py").read_text(
        encoding="utf-8",
    )
    packaged = Path(
        "onnx_splitpoint_tool/resources/remote_scripts/native_full_baseline_eval_runner.py"
    ).read_text(encoding="utf-8")

    assert source == packaged
    assert (
        'DEEPX_PREPARED_FEED_CONTRACT_VERSION = "deepx-sealed-runtime-input-v3"'
        in source
    )
    assert "deepx_prepared_feed_contract_version_mismatch" in source
    assert "prepared_feed_contract_binding_ok" in source


@pytest.mark.parametrize("shape", ([224, 224, 3], [640, 640, 3]))
def test_runtime_hwc_contract_forms_from_v262_are_resolved_exactly(
    tmp_path: Path, shape: list[int],
) -> None:
    namespace = _template_namespace(
        "_deepx_input_geometry",
        "_deepx_shape_from_contract_payload",
        "_deepx_shape_from_dxcom_config",
        "_deepx_input_size_from_shape",
        "_deepx_input_size_from_contract",
    )
    contract = _real_full_contract(shape)
    contract_path = tmp_path / "output_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    namespace["_deepx_contract_candidates"] = lambda _root, _run: [
        (contract_path, "suite.full.output_contract")
    ]
    namespace["_read_json"] = lambda path: json.loads(path.read_text(encoding="utf-8"))

    input_size, resolved = namespace["_deepx_input_size_from_contract"](
        tmp_path, {}, default=999,
    )

    assert input_size == shape[0]
    assert resolved["input"] == contract["input"]
    assert resolved["contract_resolution_status"] == "resolved"
    assert resolved["resolved_input_geometry"] == {
        "shape": shape,
        "layout": "HWC",
        "height": shape[0],
        "width": shape[1],
        "channels": 3,
        "batch": None,
        "batch_shape": [1, *shape],
    }


def test_aggregate_contract_with_conflicting_full_shapes_fails_closed(tmp_path: Path) -> None:
    namespace = _template_namespace(
        "_deepx_input_geometry",
        "_deepx_shape_from_contract_payload",
        "_deepx_shape_from_dxcom_config",
        "_deepx_input_size_from_shape",
        "_deepx_input_size_from_contract",
    )
    aggregate = {
        "contracts": [
            _real_full_contract([224, 224, 3]),
            _real_full_contract([640, 640, 3]),
        ]
    }
    contract_path = tmp_path / "deepx_output_contracts.json"
    contract_path.write_text(json.dumps(aggregate), encoding="utf-8")
    namespace["_deepx_contract_candidates"] = lambda _root, _run: [
        (contract_path, "suite.deepx_output_contracts")
    ]
    namespace["_read_json"] = lambda path: json.loads(path.read_text(encoding="utf-8"))

    input_size, resolved = namespace["_deepx_input_size_from_contract"](
        tmp_path, {}, default=640,
    )

    assert input_size == 640
    assert resolved["contract_resolution_status"] == "ambiguous"
    assert resolved["candidate_signatures"] == [
        {
            "shape": [224, 224, 3], "layout": "HWC", "dtype": "uint8",
            "normalization": "embedded_dxcom_preprocessing",
            "color_space": "RGB", "preprocess_mode": "resize",
            "letterbox_pad_value": "0",
        },
        {
            "shape": [640, 640, 3], "layout": "HWC", "dtype": "uint8",
            "normalization": "embedded_dxcom_preprocessing",
            "color_space": "RGB", "preprocess_mode": "resize",
            "letterbox_pad_value": "0",
        },
    ]


@pytest.mark.parametrize(
    ("shape", "layout"),
    (
        ([224, 224, 3], "NCHW"),
        ([1, 224, 224, 3], "HWC"),
        ([2, 224, 224, 3], "NHWC"),
        ([224, 320, 3], "HWC"),
    ),
)
def test_invalid_or_ambiguous_layout_shape_pairs_are_rejected(
    shape: list[int], layout: str,
) -> None:
    namespace = _template_namespace("_deepx_input_geometry")
    assert namespace["_deepx_input_geometry"](_real_full_contract(shape, layout)) is None


@pytest.mark.parametrize(
    ("shape", "layout", "expected_feed_shape"),
    (
        ([224, 224, 3], "HWC", (224, 224, 3)),
        ([640, 640, 3], "HWC", (640, 640, 3)),
        ([1, 224, 224, 3], "NHWC", (1, 224, 224, 3)),
    ),
)
def test_prepared_feed_honours_unbatched_and_batched_contract_shapes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shape: list[int],
    layout: str,
    expected_feed_shape: tuple[int, ...],
) -> None:
    namespace = _template_namespace("_run_deepx_prepared_feed_benchmark")
    contract = _real_full_contract(shape, layout)
    contract["contract_resolution_status"] = "resolved"
    input_size = 224 if 224 in shape else 640
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"test image placeholder")
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"dxnn")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    sealed_feed = np.zeros(expected_feed_shape, dtype=np.uint8)
    manifest_path, prepared_input_binding = (
        _sealed_prepared_input_binding(
            tmp_path,
            feed=sealed_feed,
            contract=contract,
            task="classification",
            source_image=image_path,
        )
    )

    seen_shapes: list[tuple[int, ...]] = []

    class FakeEngine:
        def __init__(self, model: str) -> None:
            assert model == str(dxnn)

        def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
            assert len(inputs) == 1
            seen_shapes.append(tuple(inputs[0].shape))
            return [np.zeros((1,), dtype=np.float32)]

    cv2 = types.ModuleType("cv2")
    cv2.imread = lambda _path: np.zeros((32, 48, 3), dtype=np.uint8)
    dx_engine = types.ModuleType("dx_engine")
    dx_engine.InferenceEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "dx_engine", dx_engine)
    _install_shared_input_loader(
        monkeypatch, sealed_feed,
        expected_setup_id="setup-from-quality-evidence",
    )

    namespace["_deepx_input_size_from_contract"] = lambda *_args, **_kwargs: (
        input_size, contract,
    )
    namespace["_deepx_input_geometry"] = _template_namespace(
        "_deepx_input_geometry"
    )["_deepx_input_geometry"]
    namespace["_deepx_find_prepared_feed_image"] = lambda *_args, **_kwargs: (
        image_path, "test_contract_image",
    )
    namespace["_deepx_contract_model_id"] = lambda *_args, **_kwargs: "fixture"
    expected_image_path = image_path

    def _load_sealed(
        selected_manifest: Path,
        *,
        input_contract: dict[str, Any],
        image_path: Path,
        task: str,
        np: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert selected_manifest == manifest_path
        assert input_contract == contract
        assert image_path == expected_image_path
        assert task == "classification"
        assert np is sys.modules["numpy"]
        return sealed_feed.copy(), dict(prepared_input_binding)

    namespace["_deepx_load_sealed_prepared_feed"] = _load_sealed

    result = namespace["_run_deepx_prepared_feed_benchmark"](
        tmp_path,
        dxnn,
        {"benchmark_task": "classification"},
        SimpleNamespace(
            runs=2,
            warmup=1,
            energy_measurement_only=False,
            prepared_input_manifest=str(manifest_path),
            quality_evidence_setup_id="setup-from-quality-evidence",
        ),
        results_dir,
    )

    assert result["status"] == "ok"
    assert result["input_contract_mode"] == "explicit"
    assert tuple(result["prepared_feed_shape"]) == expected_feed_shape
    assert result["prepared_feed_contract_version"] == (
        "deepx-sealed-runtime-input-v3"
    )
    assert result["prepared_input_binding_verified"] is True
    assert result["prepared_input_sha256"] == prepared_input_binding[
        "prepared_input_sha256"
    ]
    assert seen_shapes == [expected_feed_shape] * 3


def test_resnet_writer_uses_workflow_model_identity_below_numeric_suite_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    namespace = _template_namespace("_run_deepx_prepared_feed_benchmark")
    root = (
        tmp_path / "resnet50_orin_nx_deepx_m1_01" / "1" / "suite"
    )
    root.mkdir(parents=True)
    assert root.parent.name == "1"

    contract = _real_full_contract([2, 2, 3])
    contract["contract_resolution_status"] = "resolved"
    image_path = root / "resnet-validation.jpg"
    image_path.write_bytes(b"resnet validation image")
    dxnn = root / "resnet50.dxnn"
    dxnn.write_bytes(b"dxnn")
    results_dir = root / "results"
    prepared_dir = results_dir / "prepared_input"
    prepared_dir.mkdir(parents=True)
    sealed_feed = np.zeros((2, 2, 3), dtype=np.uint8)
    manifest_path, prepared_input_binding = _sealed_prepared_input_binding(
        prepared_dir,
        feed=sealed_feed,
        contract=contract,
        task="classification",
        source_image=image_path,
    )

    captured: dict[str, Any] = {}

    def prepare_and_seal_deepx_native_full_input(**kwargs: Any) -> dict[str, str]:
        captured.update(kwargs)
        return {"manifest_path": str(manifest_path)}

    def load_sealed_deepx_native_full_input(
        selected_manifest: Path, **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        assert selected_manifest == manifest_path
        assert kwargs["expected_model"] == "resnet50"
        assert kwargs["expected_setup_id"] == "orin_nx_deepx_m1_01"
        assert kwargs["expected_comparison_backend"] == "deepx"
        return {"runtime_input": sealed_feed.copy()}

    package = types.ModuleType("splitpoint_runners")
    package.__path__ = []  # type: ignore[attr-defined]
    native_full_input = types.ModuleType(
        "splitpoint_runners.native_full_input"
    )
    native_full_input.prepare_and_seal_deepx_native_full_input = (
        prepare_and_seal_deepx_native_full_input
    )
    native_full_input.load_sealed_deepx_native_full_input = (
        load_sealed_deepx_native_full_input
    )
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(
        sys.modules, "splitpoint_runners.native_full_input", native_full_input,
    )

    class FakeEngine:
        def __init__(self, model: str) -> None:
            assert model == str(dxnn)

        def run(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
            assert tuple(inputs[0].shape) == (2, 2, 3)
            return [np.zeros((1, 1000), dtype=np.float32)]

    cv2 = types.ModuleType("cv2")
    cv2.imread = lambda _path: np.zeros((8, 12, 3), dtype=np.uint8)
    dx_engine = types.ModuleType("dx_engine")
    dx_engine.InferenceEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "dx_engine", dx_engine)

    namespace["_deepx_input_size_from_contract"] = (
        lambda *_args, **_kwargs: (2, contract)
    )
    namespace["_deepx_input_geometry"] = _template_namespace(
        "_deepx_input_geometry"
    )["_deepx_input_geometry"]
    namespace["_deepx_find_prepared_feed_image"] = (
        lambda *_args, **_kwargs: (image_path, "workflow_validation_image")
    )
    namespace["_deepx_contract_model_id"] = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("numeric suite directory must not supply model identity")
        )
    )
    namespace["_deepx_load_sealed_prepared_feed"] = (
        lambda *_args, **_kwargs: (
            sealed_feed.copy(), dict(prepared_input_binding)
        )
    )
    args = SimpleNamespace(
        runs=1,
        warmup=0,
        energy_measurement_only=False,
        prepared_input_manifest="",
        quality_evidence_model_id="resnet50",
        quality_evidence_setup_id="orin_nx_deepx_m1_01",
        validation_images="",
    )

    result = namespace["_run_deepx_prepared_feed_benchmark"](
        root,
        dxnn,
        {"benchmark_task": "classification"},
        args,
        results_dir,
    )

    assert result["status"] == "ok"
    assert captured["model"] == "resnet50"
    assert captured["setup_id"] == "orin_nx_deepx_m1_01"
    assert captured["out_dir"] == prepared_dir
    assert args.prepared_input_manifest == str(manifest_path)


@pytest.mark.parametrize(
    ("task", "shape", "expected_mode", "expected_pad"),
    (
        ("classification", [1, 3, 224, 224], "resize", 0),
        ("detection", [1, 3, 640, 640], "letterbox", 114),
    ),
)
def test_workflow_build_contract_is_task_bound(
    task: str, shape: list[int], expected_mode: str, expected_pad: int,
) -> None:
    from onnx_splitpoint_tool.workflow.deepx_build_binding import (
        task_bound_deepx_runtime_input_contract,
    )

    result = task_bound_deepx_runtime_input_contract(
        task=task, input_name="images", source_shape=shape,
    )
    assert result["shape"] == [shape[2], shape[3], 3]
    assert result["layout"] == "HWC"
    assert result["dtype"] == "uint8"
    assert result["color_space"] == "RGB"
    assert result["preprocess_mode_requested"] == "auto"
    assert result["preprocess_mode_effective"] == expected_mode
    assert result["letterbox_pad_value_requested"] == 114
    assert result["letterbox_pad_value_effective"] == expected_pad
    assert result["letterbox_pad_value"] == expected_pad


def test_dxcom_default_loader_geometry_is_task_bound(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.deepx.config import image_model_dxcom_config
    from onnx_splitpoint_tool.workflow.deepx_build_binding import _task_hint

    assert _task_hint({"task": "classification"}, "opaque") == "classification"
    assert _task_hint({"resolved_path": "models/yolo26s.onnx"}, "opaque") == "detection"
    assert _task_hint({}, "opaque") == ""

    classification = image_model_dxcom_config(
        task="classification",
        input_name="images",
        input_shape=[1, 3, 224, 224],
        calibration_dir=tmp_path,
    )
    detection = image_model_dxcom_config(
        task="detection",
        input_name="images",
        input_shape=[1, 3, 640, 640],
        calibration_dir=tmp_path,
    )

    assert classification["default_loader"]["preprocessings"][0] == {
        "resize": {"width": 224, "height": 224}
    }
    assert detection["default_loader"]["preprocessings"][0] == {
        "resize": {
            "mode": "pad",
            "size": 640,
            "pad_location": "edge",
            "pad_value": [114, 114, 114],
        }
    }
    assert classification["default_loader"]["preprocessings"][1:] == (
        detection["default_loader"]["preprocessings"][1:]
    )

    with pytest.raises(ValueError, match="requires task"):
        image_model_dxcom_config(
            task="auto",
            input_shape=[1, 3, 224, 224],
            calibration_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="square model input"):
        image_model_dxcom_config(
            task="classification",
            input_shape=[1, 3, 320, 640],
            calibration_dir=tmp_path,
        )


def test_prepared_feed_reports_exact_shape_mismatch_instead_of_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    namespace = _template_namespace("_run_deepx_prepared_feed_benchmark")
    contract = _real_full_contract([224, 224, 4], "HWC")
    contract["contract_resolution_status"] = "resolved"
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"placeholder")
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"dxnn")
    manifest_path = tmp_path / "native_full_input_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    cv2 = types.ModuleType("cv2")
    cv2.imread = lambda _path: np.zeros((224, 224, 3), dtype=np.uint8)
    dx_engine = types.ModuleType("dx_engine")
    dx_engine.InferenceEngine = lambda _model: (_ for _ in ()).throw(
        AssertionError("engine must not initialize for a mismatched feed")
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "dx_engine", dx_engine)
    _install_shared_input_loader(
        monkeypatch, np.zeros((224, 224, 4), dtype=np.uint8),
        expected_setup_id="setup-from-quality-evidence",
    )

    namespace["_deepx_input_size_from_contract"] = lambda *_args, **_kwargs: (
        224, contract,
    )
    namespace["_deepx_input_geometry"] = _template_namespace(
        "_deepx_input_geometry"
    )["_deepx_input_geometry"]
    namespace["_deepx_find_prepared_feed_image"] = lambda *_args, **_kwargs: (
        image_path, "test_contract_image",
    )
    namespace["_deepx_contract_model_id"] = lambda *_args, **_kwargs: "fixture"
    namespace["_deepx_load_sealed_prepared_feed"] = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("prepared_input_contract_shape_mismatch")
        )
    )

    result = namespace["_run_deepx_prepared_feed_benchmark"](
        tmp_path,
        dxnn,
        {"benchmark_task": "classification"},
        SimpleNamespace(
            runs=1,
            warmup=0,
            energy_measurement_only=False,
            prepared_input_manifest=str(manifest_path),
            quality_evidence_setup_id="setup-from-quality-evidence",
        ),
        tmp_path,
    )

    assert result["status"] == "prepared_input_contract_shape_mismatch"
    assert result["error"] == (
        "ValueError: prepared_input_contract_shape_mismatch"
    )
    assert result["input_contract"] == contract
