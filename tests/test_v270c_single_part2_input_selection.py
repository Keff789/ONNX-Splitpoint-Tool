from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import jsonschema
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    resolve_evaluation_profile,
)
from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationOrchestrationConfig,
    BenchmarkGenerationRuntime,
)
from onnx_splitpoint_tool.gui.controller import write_benchmark_suite_script
from onnx_splitpoint_tool.runners._types import (
    BackendRunOut,
    GraphPlan,
    RunCfg,
    SampleCfg,
    StagePlan,
)
from onnx_splitpoint_tool.runners.backends.base import PreparedHandle
from onnx_splitpoint_tool.runners.graph_runner import GraphRunner
from onnx_splitpoint_tool.split_export_graph import (
    part2_external_inputs_for_boundary,
    part2_input_count_for_boundary,
)
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA = (
    ROOT
    / "onnx_splitpoint_tool"
    / "resources"
    / "schemas"
    / "evaluation_profile.schema.json"
)
SMOKE_PROFILE = (
    ROOT
    / "onnx_splitpoint_tool"
    / "resources"
    / "evaluation_profiles"
    / "smoke_regression_v1.yaml"
)


def _passthrough_model() -> onnx.ModelProto:
    """Three-node graph whose first Part-2 also needs an original input."""

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    skip = helper.make_tensor_value_info("skip", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["x"], ["cut"], name="left"),
            helper.make_node("Add", ["cut", "skip"], ["joined"], name="join"),
            helper.make_node("Relu", ["joined"], ["y"], name="right"),
        ],
        "part2-input-count",
        [x, skip],
        [y],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )


def test_exact_part2_input_count_includes_original_passthrough_input() -> None:
    model = _passthrough_model()
    nodes = list(model.graph.node)
    order = [0, 1, 2]

    # At b0 only one activation crosses the split, but Part-2 also consumes the
    # original ``skip`` input.  Counting cut tensors alone would incorrectly
    # report one input here.
    assert part2_external_inputs_for_boundary(model, order, nodes, 0) == [
        "cut",
        "skip",
    ]
    assert part2_input_count_for_boundary(model, order, nodes, 0) == 2

    # Moving the boundary behind the Add leaves one exact external Part-2 input.
    assert part2_external_inputs_for_boundary(model, order, nodes, 1) == [
        "joined",
    ]
    assert part2_input_count_for_boundary(model, order, nodes, 1) == 1


def test_evaluation_profile_schema_requires_boolean_single_part2_policy() -> None:
    schema = json.loads(PROFILE_SCHEMA.read_text(encoding="utf-8"))
    profile = yaml.safe_load(SMOKE_PROFILE.read_text(encoding="utf-8"))
    profile["selection_policy"]["require_single_part2_input"] = True
    jsonschema.validate(profile, schema)

    invalid = copy.deepcopy(profile)
    invalid["selection_policy"]["require_single_part2_input"] = "true"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)


def test_profile_override_and_start_snapshot_preserve_single_part2_policy(
    tmp_path: Path,
) -> None:
    profile = yaml.safe_load(SMOKE_PROFILE.read_text(encoding="utf-8"))
    profile["selection_policy"]["require_single_part2_input"] = True
    profile_path = tmp_path / "single-part2.yaml"
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    resolution = resolve_evaluation_profile(
        profile_path,
        export_metadata={"model_name": "resnet50", "task": "classification"},
    )
    assert resolution is not None
    assert resolution.overrides["require_single_part2_input"] is True

    snapshot = build_profile_start_snapshot(
        profile_request=str(profile_path),
        source_profile=profile,
        resolved_profile=profile,
        profile_id="single-part2",
        profile_path=str(profile_path),
        profile_source="file",
    )
    assert (
        snapshot["requested_selection"]["selection_policy"][
            "require_single_part2_input"
        ]
        is True
    )
    assert (
        snapshot["resolved_selection"]["selection_policy"][
            "require_single_part2_input"
        ]
        is True
    )
    assert snapshot["effective_execution_plan"]["require_single_part2_input"] is True


def test_generator_skips_multi_input_candidate_and_backfills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool import api as asc

    model = _passthrough_model()
    nodes = list(model.graph.node)
    order = [0, 1, 2]
    out_dir = tmp_path / "suite"
    out_dir.mkdir()

    def _write_runner(case_dir, **_kwargs):
        runner = Path(case_dir) / "run_benchmark.py"
        runner.write_text("# generated test runner\n", encoding="utf-8")
        return str(runner)

    monkeypatch.setattr(asc, "write_runner_skeleton_onnxruntime", _write_runner)

    runtime = BenchmarkGenerationRuntime(
        out_dir=out_dir,
        bench_log_path=out_dir / "benchmark.log",
        state_path=out_dir / "generation_state.json",
        requested_cases=1,
        ranked_candidates=[0, 1],
        candidate_search_pool=[0, 1],
        model_name="passthrough",
        model_source="passthrough.onnx",
        hef_full_policy="end",
    )
    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[0, 1],
        candidate_search_pool=[0, 1],
        out_dir=out_dir,
        base="passthrough",
        pad=3,
        strict_boundary=False,
        model=model,
        nodes=nodes,
        order=order,
        analysis_payload={},
        require_single_part2_input=True,
    )
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda _msg, **_kwargs: None,
        queue_put=lambda _event: None,
        persist_state=lambda **_kwargs: None,
        publish_hailo_diagnostics=lambda *_args, **_kwargs: None,
        predicted_metrics_for_boundary=lambda *_args, **_kwargs: {},
        hailo_parse_entry_for_boundary=lambda *_args, **_kwargs: None,
        hailo_parse_scalar_fields=lambda *_args, **_kwargs: {},
    )

    chosen = BenchmarkGenerationExecutionService().execute_case_build_loop(
        cfg, callbacks
    )

    assert chosen == [1]
    assert [row["boundary"] for row in runtime.cases] == [1]
    assert runtime.discarded_boundaries == {0}
    assert len(runtime.discarded_cases) == 1
    rejected = runtime.discarded_cases[0]
    assert rejected["reason"] == "part2_input_count_not_one"
    assert rejected["part2_input_count"] == 2
    assert rejected["part2_input_names"] == ["cut", "skip"]
    assert not (out_dir / "b000").exists()
    assert (out_dir / "b001" / "split_manifest.json").is_file()
    assert (
        "part2_input_count_not_one"
        in BenchmarkGenerationOrchestrationConfig.__dataclass_fields__[
            "benign_discard_reasons"
        ].default_factory()
    )


def test_generated_suite_records_native_capability_skip_and_returns_to_generic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite_dir = tmp_path / "generated-suite"
    script_path = Path(write_benchmark_suite_script(suite_dir))
    spec = importlib.util.spec_from_file_location(
        "benchmark_suite_v270c_capability_test", script_path
    )
    assert spec is not None and spec.loader is not None
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)

    package = ModuleType("splitpoint_runners")
    package.__path__ = []  # type: ignore[attr-defined]
    runtime_module = ModuleType(
        "splitpoint_runners.native_split_quality_runtime"
    )

    def _unsupported_binding(**_kwargs):
        raise RuntimeError("native_split_quality_part2_input_count:3")

    runtime_module.prepare_native_split_quality_binding = _unsupported_binding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(
        sys.modules,
        "splitpoint_runners.native_split_quality_runtime",
        runtime_module,
    )

    case_dir = suite_dir / "b142"
    case_dir.mkdir()
    args = SimpleNamespace(
        energy_measurement_only=False,
        quality_evidence_eval_id="eval-1",
        quality_evidence_setup_id="hailo8_setup",
        quality_evidence_model_id="yolo26s",
        trt_cache_root=str((tmp_path / "trt-cache").resolve()),
        native_trt_workspace_mb=1024,
        native_trt_build_timeout_s=60,
    )

    result = suite._prepare_native_split_quality_for_case(
        root=suite_dir,
        case_dir=case_dir,
        bench={"model_id": "yolo26s"},
        plan={},
        run={},
        args=args,
        run_id="hailo8_to_trt",
        stage1="hailo8",
        stage2="tensorrt",
        variants=["composed"],
    )

    assert result is None
    skip_path = (
        case_dir
        / "results_hailo8_to_trt"
        / "task_quality_inputs"
        / "native_split_quality_capability_skip.json"
    )
    skip = json.loads(skip_path.read_text(encoding="utf-8"))
    assert skip["reason"] == "part2_input_count_not_one"
    assert skip["error"] == "native_split_quality_part2_input_count:3"
    assert skip["generic_execution_continues"] is True
    assert "Generic execution continues" in capsys.readouterr().out


def test_generic_graph_runner_keeps_multi_input_and_multi_output_support(
    tmp_path: Path,
) -> None:
    class _Stage1Backend:
        name = "stage1"
        capabilities = None

        def prepare(self, _run_cfg, _artifacts_dir):
            return PreparedHandle(
                input_names=["x", "skip"],
                output_names=["cut"],
                handle=None,
            )

        def run(self, _prepared, inputs):
            assert set(inputs) == {"x", "skip"}
            return BackendRunOut(outputs={"cut": np.asarray(inputs["x"]) * 2.0})

        def cleanup(self, _prepared):
            return None

    class _Stage2Backend:
        name = "stage2"
        capabilities = None

        def __init__(self):
            self.seen_inputs = None

        def prepare(self, _run_cfg, _artifacts_dir):
            return PreparedHandle(
                input_names=["cut", "skip"],
                output_names=["sum", "difference"],
                handle=None,
            )

        def run(self, _prepared, inputs):
            self.seen_inputs = {
                name: np.asarray(value).copy() for name, value in inputs.items()
            }
            return BackendRunOut(
                outputs={
                    "sum": inputs["cut"] + inputs["skip"],
                    "difference": inputs["cut"] - inputs["skip"],
                }
            )

        def cleanup(self, _prepared):
            return None

    class _Harness:
        def __init__(self):
            self.outputs = None

        def make_inputs(self, _sample_cfg):
            return {
                "x": np.asarray([[2.0, 3.0]], dtype=np.float32),
                "skip": np.asarray([[5.0, 7.0]], dtype=np.float32),
            }

        def postprocess(self, outputs, *, context):
            assert context["plan"]["label"] == "generic-multi-io"
            self.outputs = outputs
            return {"task": "test", "json": {}}

    stage2 = _Stage2Backend()
    harness = _Harness()
    plan = GraphPlan(
        stages=[
            StagePlan(
                name="part1",
                backend_name="stage1",
                run_cfg=RunCfg(model_path=tmp_path / "part1.onnx"),
            ),
            StagePlan(
                name="part2",
                backend_name="stage2",
                run_cfg=RunCfg(model_path=tmp_path / "part2.onnx"),
            ),
        ],
        sample_cfg=SampleCfg(),
        measured_runs=1,
        measure_interface=True,
        label="generic-multi-io",
    )

    result = GraphRunner(
        {"stage1": _Stage1Backend(), "stage2": stage2}
    ).run_graph(plan, harness, tmp_path / "artifacts")

    assert result.status == "ok"
    assert set(stage2.seen_inputs or {}) == {"cut", "skip"}
    assert set(harness.outputs or {}) == {"sum", "difference"}
    np.testing.assert_allclose(
        harness.outputs["sum"], np.asarray([[9.0, 13.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        harness.outputs["difference"],
        np.asarray([[-1.0, -1.0]], dtype=np.float32),
    )
