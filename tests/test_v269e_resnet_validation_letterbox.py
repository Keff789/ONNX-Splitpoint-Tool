from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image
import pytest

from onnx_splitpoint_tool.resources_utils import copy_resource_file
from onnx_splitpoint_tool.split_export_runners import (
    write_runner_skeleton_onnxruntime,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_TEMPLATE = (
    ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
)


def _save_model(path: Path, *, part: str) -> None:
    image = helper.make_tensor_value_info(
        "image" if part != "part2" else "boundary",
        TensorProto.FLOAT,
        [1, 3, 8, 8],
    )
    if part == "part1":
        nodes = [helper.make_node("Identity", ["image"], ["boundary"])]
        outputs = [
            helper.make_tensor_value_info(
                "boundary", TensorProto.FLOAT, [1, 3, 8, 8],
            )
        ]
    else:
        source = "image" if part == "full" else "boundary"
        nodes = [
            helper.make_node("GlobalAveragePool", [source], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["logits"], axis=1),
        ]
        outputs = [
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, [1, 3],
            )
        ]
    graph = helper.make_graph(nodes, f"tiny_resnet_{part}", [image], outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="v269e-resnet-regression",
    )
    # Keep the fixture readable by both older Jetson ORT and the test runtime.
    model.ir_version = 8
    onnx.save(model, path)


def _make_generated_classification_case(case: Path) -> Path:
    case.mkdir(parents=True)
    vendored_runtime = case / "splitpoint_runners"
    vendored_runtime.mkdir()
    copy_resource_file(
        "native_output_endpoint.py",
        dest=vendored_runtime / "native_output_endpoint.py",
    )
    _save_model(case / "full.onnx", part="full")
    _save_model(case / "part1.onnx", part="part1")
    _save_model(case / "part2.onnx", part="part2")
    (case / "split_manifest.json").write_text(
        json.dumps(
            {
                "full_model": "full.onnx",
                "part1_model": "part1.onnx",
                "part2_model": "part2.onnx",
                "part1_cut_names": ["boundary"],
                "part2_cut_names": ["boundary"],
            }
        ),
        encoding="utf-8",
    )

    validation = case / "validation"
    validation.mkdir()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    Image.fromarray(image, mode="RGB").save(validation / "sample.png")
    (validation / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/classification-validation-manifest",
                "schema_version": 1,
                "samples": [{"image": "sample.png", "label_id": 0}],
            }
        ),
        encoding="utf-8",
    )
    (case / "quality_policy.json").write_text(
        json.dumps(
            {
                "profile_id": "v269e-resnet-regression",
                "statistics": {
                    "execution_location": "central_management",
                    "bootstrap_repetitions": 1,
                    "workers": 1,
                },
                "classification": {
                    "primary_metric": "top1_accuracy",
                    "non_inferiority_margin": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )
    runner = Path(
        write_runner_skeleton_onnxruntime(
            str(case), manifest_filename="split_manifest.json", target="cpu",
        )
    )
    return runner


def _fake_onnxruntime() -> ModuleType:
    """Small ORT-compatible executor for this generated-runner regression.

    ONNX Runtime is an optional package extra and intentionally absent from the
    release-test environment.  The control-flow regression only needs the two
    operations in the tiny fixture, so this adapter keeps the test hermetic
    without replacing any runner-side logic.
    """

    module = ModuleType("onnxruntime")

    class SessionOptions:
        log_severity_level = 2
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    class ValueInfo:
        def __init__(self, value: onnx.ValueInfoProto):
            self.name = str(value.name)
            self.type = "tensor(float)"
            self.shape = [
                int(dim.dim_value) if int(dim.dim_value or 0) > 0 else None
                for dim in value.type.tensor_type.shape.dim
            ]

    class InferenceSession:
        def __init__(self, model_path: str, **_kwargs: object):
            self.model = onnx.load(model_path)
            self._inputs = [ValueInfo(value) for value in self.model.graph.input]
            self._outputs = [ValueInfo(value) for value in self.model.graph.output]

        def get_inputs(self) -> list[ValueInfo]:
            return list(self._inputs)

        def get_outputs(self) -> list[ValueInfo]:
            return list(self._outputs)

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def run(
            self, _output_names: object, feeds: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            value = np.asarray(feeds[self._inputs[0].name], dtype=np.float32)
            if self._outputs[0].name == "boundary":
                return [value.copy()]
            return [value.mean(axis=(2, 3), dtype=np.float32)]

    module.SessionOptions = SessionOptions  # type: ignore[attr-defined]
    module.InferenceSession = InferenceSession  # type: ignore[attr-defined]
    module.get_available_providers = (  # type: ignore[attr-defined]
        lambda: ["CPUExecutionProvider"]
    )
    return module


def test_classification_letterbox_state_dominates_both_central_quality_paths() -> None:
    """The generated main flow must bind preprocessing before either export mode.

    Management CPU reference and quality-first use the same classification
    contract call.  This checks its lexical control-flow dependency rather than
    merely looking for a replacement string.
    """

    tree = ast.parse(
        RUNNER_TEMPLATE.read_text(encoding="utf-8"), filename=str(RUNNER_TEMPLATE),
    )
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    stores = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Name)
        and node.id == "classification_validation_letterbox"
        and isinstance(node.ctx, ast.Store)
    ]
    loads = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Name)
        and node.id == "classification_validation_letterbox"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(stores) == 1
    assert loads
    assert stores[0].lineno < min(node.lineno for node in loads)

    assignment = next(
        node for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(target is stores[0] for target in node.targets)
    )
    # A direct main-body assignment is unconditional after the earlier
    # CPU-reference/quality-evidence argument-mode selection.  This is what
    # makes the value available to both central execution roles.
    assert assignment in main.body
    assert isinstance(assignment.value, ast.Constant)
    assert assignment.value.value is False

    classification_contract_calls = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_classification_quality_contract"
    ]
    assert len(classification_contract_calls) == 1
    letterbox_keyword = next(
        keyword for keyword in classification_contract_calls[0].keywords
        if keyword.arg == "letterbox"
    )
    assert isinstance(letterbox_keyword.value, ast.Name)
    assert letterbox_keyword.value.id == "classification_validation_letterbox"
    assert stores[0].lineno < classification_contract_calls[0].lineno

    # Detector letterboxing remains branch-local, but all of its reads must be
    # dominated by the detection assignment after the classification branch.
    detector_stores = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Name)
        and node.id == "val_letterbox"
        and isinstance(node.ctx, ast.Store)
    ]
    detector_loads = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Name)
        and node.id == "val_letterbox"
        and isinstance(node.ctx, ast.Load)
    ]
    assert len(detector_stores) == 1
    assert detector_loads
    assert detector_stores[0].lineno < min(node.lineno for node in detector_loads)


def test_generated_runner_executes_management_resnet_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    runner = _make_generated_classification_case(tmp_path / "b001")
    generated_tree = ast.parse(
        runner.read_text(encoding="utf-8"), filename=str(runner),
    )
    assert any(
        isinstance(node, ast.Name)
        and node.id == "classification_validation_letterbox"
        for node in ast.walk(generated_tree)
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_onnxruntime())
    monkeypatch.setenv("ONNX_SPLITPOINT_CPU_REFERENCE_ONLY", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_CPU_THREADS", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.chdir(runner.parent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner),
            "--manifest", "split_manifest.json",
            "--provider", "cpu",
            "--variants", "full",
            "--benchmark-task", "classification",
            "--validation-mode", "proxy_classification",
            "--validation-images", "validation",
            "--validation-max-images", "1",
            "--quality-gate-json", "quality_policy.json",
            "--image", "validation/sample.png",
            "--image-scale", "imagenet",
            "--warmup", "0",
            "--runs", "1",
            "--phase-runs", "0",
            "--throughput-frames", "0",
            "--baseline-cache", "off",
            "--viz", "none",
            "--no-report-plots",
            "--out-dir", "results",
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(runner), run_name="__main__")
    assert exit_info.value.code == 0
    combined_output = capsys.readouterr().out
    assert "UnboundLocalError" not in combined_output

    reference_path = (
        runner.parent
        / "results/task_quality_inputs/canonical_classification_reference.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    assert reference["task"] == "classification"
    assert len(reference["records"]) == 1
    preprocessing = reference["quality_contract"]["preprocessing"]["identity"]
    assert preprocessing["image_scale"] == "imagenet"
    assert preprocessing["letterbox"] is False
