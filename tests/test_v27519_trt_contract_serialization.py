from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService
from onnx_splitpoint_tool.management_reference import (
    _cpu_reference_run,
    bind_management_cpu_reference_runs,
)
from onnx_splitpoint_tool.workflow.execution_binding import (
    _performance_run_ids_v263,
)
from onnx_splitpoint_tool.workflow.legacy_benchmarkset_binding import (
    _bind_management_reference_cpu_switch_v27519,
    _bind_management_reference_targets_v27519,
    _infer_run_switches,
    _profile_targets,
)
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _performance_plan_v263,
)


def _constant_model(path: Path, output_shapes: list[list[int]]) -> None:
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 8, 8],
    )
    outputs = []
    nodes = []
    for index, shape in enumerate(output_shapes):
        name = f"output_{index}"
        outputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        )
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                [name],
                value=helper.make_tensor(
                    f"value_{index}",
                    TensorProto.FLOAT,
                    shape,
                    [0.0] * math.prod(shape),
                ),
            )
        )
    graph = helper.make_graph(nodes, "contract_graph", [input_info], outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, path)


def _contract_runner(tmp_path: Path) -> EvaluationWorkflowRunner:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.manifest = {}
    runner.profile_payload = {
        "run_profiles": [{
            "id": "ort_tensorrt",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
        }],
    }
    runner._profile_with_cli_hardware_overrides = (
        lambda: runner.profile_payload
    )
    runner._prepared_full_hailo_info = lambda _path: {}
    return runner


@pytest.mark.parametrize(
    ("model_id", "task", "family", "shapes", "stage", "output_format"),
    (
        (
            "resnet50", "classification", "resnet",
            [[1, 1000]], "classification_logits", None,
        ),
        (
            "yolo26s", "detection", "yolo26",
            [[1, 300, 6]], "decoded_nms", "bn6_detections",
        ),
        (
            "yolo11l", "detection", "yolo11",
            [[1, 84, 8400]], "decoded_pre_nms",
            "ultralytics_decoded",
        ),
        (
            "yolov7_paper", "detection", "yolov7",
            [[1, 3, 4, 5, 6], [1, 3, 2, 3, 6], [1, 3, 1, 2, 6]],
            "raw_head", None,
        ),
    ),
)
def test_real_full_baseline_writer_attests_tensorrt_contract(
    tmp_path: Path,
    model_id: str,
    task: str,
    family: str,
    shapes: list[list[int]],
    stage: str,
    output_format: str | None,
) -> None:
    """The production writer, not a hand-authored row, feeds the strict loader."""

    model_path = tmp_path / f"{model_id}.onnx"
    _constant_model(model_path, shapes)
    runner = _contract_runner(tmp_path)

    artifacts, _details, _message, status = (
        runner._stage_prepare_full_baselines(
            model_id,
            {
                "id": model_id,
                "family": family,
                "task": task,
                "resolved_path": str(model_path),
            },
        )
    )
    assert status == "ok"
    serialized = json.loads(
        artifacts["output_contracts_json"].read_text(encoding="utf-8")
    )
    assert sum(
        row["backend"] == "cuda_ort"
        for row in serialized["contracts"]
    ) == 1

    suite_root = tmp_path / "strict_consumer_suite"
    suite_root.mkdir()
    shutil.copy2(
        artifacts["output_contracts_json"],
        suite_root / "output_contracts.json",
    )
    contract = load_authoritative_output_contract(
        suite_root,
        backend="tensorrt",
        model_id=model_id,
        variant="full",
        task=task,
    )
    assert contract["contract_resolution_status"] == "attested"
    assert contract["authoritative_output_contract"] is True
    assert contract["backend"] == "cuda_ort"
    assert contract["stage"] == stage
    assert contract.get("output_format") == output_format
    needs_postprocess = stage in {"raw_head", "decoded_pre_nms"}
    assert contract["host_tail_required"] is needs_postprocess
    assert contract["postprocessing_required"] is needs_postprocess
    assert contract["requires_external_postprocess"] is needs_postprocess


def test_real_generation_service_materializes_one_semantic_cpu_recipe(
    tmp_path: Path,
) -> None:
    profile = {
        "run_profiles": [
            {
                "id": "ort_tensorrt", "full": "tensorrt",
                "stage1": "tensorrt", "stage2": "tensorrt",
            },
            {
                "id": "hailo8", "full": "hailo8",
                "stage1": "hailo8", "stage2": "hailo8",
            },
            {
                "id": "hailo10", "full": "hailo10",
                "stage1": "hailo10", "stage2": "hailo10",
            },
            {
                "id": "deepx_m1_full", "full": "deepx_m1",
                "stage1": "deepx_m1", "stage2": "deepx_m1",
            },
        ],
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
    }
    targets, explicit_cpu, required = (
        _bind_management_reference_targets_v27519(
            profile,
            _profile_targets(profile, []),
            cache_verify_enabled=False,
        )
    )
    assert required is True
    assert explicit_cpu is False
    assert "cpu_ort" not in targets
    switches = _infer_run_switches(profile, targets)
    assert switches["acc_cpu"] is False
    switches = _bind_management_reference_cpu_switch_v27519(
        switches,
        required=required,
        cache_verify_enabled=False,
    )
    assert switches["acc_cpu"] is True

    validation_root = tmp_path / "validation"
    validation_root.mkdir()
    (validation_root / "sample.jpg").write_bytes(b"fixture")
    run_plan = BenchmarkGenerationService().build_run_plan(
        acc_cpu=bool(switches["acc_cpu"]),
        acc_cuda=bool(switches["acc_cuda"]),
        acc_trt=bool(switches["acc_trt"]),
        acc_h8=bool(switches["acc_h8"]),
        acc_h10=bool(switches["acc_h10"]),
        acc_deepx=bool(switches["acc_deepx"]),
        hailo10_hw="hailo10",
        hailo_custom_full=True,
        hailo_custom_composed=False,
        hailo_custom_part1=False,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=False,
        matrix_deepx_to_trt=False,
        matrix_trt_to_deepx=False,
        validation_images=str(validation_root),
        validation_max_images=12,
    )
    rows = bind_management_cpu_reference_runs(
        run_plan.bench_plan_runs,
        automatic=True,
        require_existing=True,
    )
    assert [row["id"] for row in rows] == [
        "ort_cpu", "ort_tensorrt", "hailo8", "hailo10",
        "deepx_m1_full",
    ]
    cpu_rows = [row for row in rows if row["id"] == "ort_cpu"]
    assert len(cpu_rows) == 1
    assert cpu_rows[0]["validation_images"] == str(validation_root)
    assert cpu_rows[0]["validation_max_images"] == 12

    serialized_plan = {
        "runs": rows,
        "planned_runs": [dict(row) for row in rows],
    }
    consumer = _cpu_reference_run(serialized_plan)
    assert consumer is not None
    assert consumer["provider"] == "cpu"
    assert consumer["semantic_reference_only"] is True
    assert consumer["performance_eligible"] is False

    filtered = _performance_plan_v263(serialized_plan, profile)
    assert filtered["performance_excluded_run_ids"] == ["ort_cpu"]
    assert _performance_run_ids_v263(serialized_plan) == [
        "ort_tensorrt", "hailo8", "hailo10", "deepx_m1_full",
    ]

    cache_targets, _explicit, cache_required = (
        _bind_management_reference_targets_v27519(
            profile,
            _profile_targets(profile, []),
            cache_verify_enabled=True,
        )
    )
    assert cache_required is False
    assert "cpu_ort" not in cache_targets


def test_legacy_cache_canary_forces_cpu_off_despite_cpu_full_text() -> None:
    profile = {
        "run_profiles": [{
            "id": "hailo8_to_trt",
            "type": "mixed_backend",
            "full_reference": "cpu_full",
            "stage1": "hailo8",
            "stage2": "tensorrt",
        }],
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
        "execution_guard": {"mode": "cache_verify_only"},
    }
    targets, _explicit, required = _bind_management_reference_targets_v27519(
        profile,
        _profile_targets(profile, []),
        cache_verify_enabled=True,
    )
    assert "cpu_full" in targets
    switches = _infer_run_switches(profile, targets)
    assert switches["acc_cpu"] is True  # legacy substring inference
    switches = _bind_management_reference_cpu_switch_v27519(
        switches,
        required=required,
        cache_verify_enabled=True,
    )
    assert switches["acc_cpu"] is False

    run_plan = BenchmarkGenerationService().build_run_plan(
        acc_cpu=bool(switches["acc_cpu"]),
        acc_cuda=bool(switches["acc_cuda"]),
        acc_trt=bool(switches["acc_trt"]),
        acc_h8=bool(switches["acc_h8"]),
        acc_h10=bool(switches["acc_h10"]),
        acc_deepx=bool(switches["acc_deepx"]),
        hailo_custom_full=True,
        hailo_custom_composed=False,
        hailo_custom_part1=False,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=True,
        cache_verify_only=True,
        cache_verify_hailo_variants=["full"],
    )
    assert "ort_cpu" not in {
        str(row.get("id") or "") for row in run_plan.bench_plan_runs
    }


def test_legacy_targets_ignore_disabled_profiles_but_not_required_false() -> None:
    profile = {
        "run_profiles": [
            {
                "id": "hailo8_to_trt", "enabled": False,
                "stage1": "hailo8", "stage2": "tensorrt",
            },
            {
                "id": "deepx_m1_to_tensorrt", "enabled": "false",
                "stage1": "deepx_m1", "stage2": "tensorrt",
            },
            {
                "id": "ort_tensorrt", "enabled": True,
                "required": False, "full": "tensorrt",
                "stage1": "tensorrt", "stage2": "tensorrt",
            },
        ],
    }
    targets = _profile_targets(profile, [])
    assert targets == ["tensorrt"]
    switches = _infer_run_switches(profile, targets)
    assert switches["acc_trt"] is True
    assert switches["acc_h8"] is False
    assert switches["acc_deepx"] is False
