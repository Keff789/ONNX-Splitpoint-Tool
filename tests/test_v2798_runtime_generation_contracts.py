from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationService,
    _merge_hailo_case_availability_aliases,
)
from onnx_splitpoint_tool.workflow.required_run_scope import (
    merge_authoritative_run_descriptors,
)
from onnx_splitpoint_tool.workflow.endpoint_lifecycle import (
    _canonical_run_profile,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"


def _suite_functions(names: set[str]) -> Dict[str, Any]:
    tree = ast.parse(SUITE.read_text(encoding="utf-8"), filename=str(SUITE))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Mapping": Mapping,
        "Optional": Optional,
        "Tuple": Tuple,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SUITE), "exec"),
        namespace,
    )
    return namespace


def test_injected_hailo8_full_is_a_materialized_hailo_recipe() -> None:
    plan = merge_authoritative_run_descriptors(
        {"runs": []},
        [{"run_id": "hailo8", "backend": "hailo8", "variant": "full"}],
    )

    assert len(plan["runs"]) == 1
    row = plan["runs"][0]
    assert row["id"] == "hailo8"
    assert row["type"] == "hailo"
    assert row["provider"] == row["backend"] == "hailo8"
    assert row["hw_arch"] == "hailo8"
    assert row["stage1"] == {"type": "hailo", "hw_arch": "hailo8"}
    assert row["stage2"] == {"type": "hailo", "hw_arch": "hailo8"}
    assert row["variants"] == ["full"]
    assert "auto" not in str(row).lower()
    assert "tensorrt" not in str(row).lower()


def test_existing_thin_hailo8_row_is_filled_without_relabeling_provider() -> None:
    plan = merge_authoritative_run_descriptors(
        {"runs": [{"id": "hailo8", "required": True}]},
        [{"run_id": "hailo8", "backend": "hailo8", "variant": "full"}],
    )

    row = plan["runs"][0]
    assert row["type"] == "hailo"
    assert row["provider"] == "hailo8"
    assert row["variants"] == ["full"]


def test_hailo10_alias_availability_has_service_template_parity() -> None:
    availability = {
        "hailo10": {
            "full": True,
            "full_error": "",
            "part1": False,
            "part1_error": "not-built-under-logical-key",
        },
        "hailo10h": {
            "full": False,
            "full_error": "not-built-under-physical-key",
            "part1": True,
            "part1_error": "",
        },
    }
    service_meta = _merge_hailo_case_availability_aliases(availability)[
        "hailo10h"
    ]
    funcs = _suite_functions({
        "_canonical_hailo_arch_key", "_merged_hailo_arch_meta",
    })
    template_meta = funcs["_merged_hailo_arch_meta"](
        availability, "hailo10h",
    )

    assert service_meta["full"] is True
    assert service_meta["part1"] is True
    assert service_meta["full_error"] == ""
    assert service_meta["part1_error"] == ""
    assert template_meta["full"] is True
    assert template_meta["part1"] is True
    assert template_meta["full_error"] == ""
    assert template_meta["part1_error"] == ""


def test_remote_variant_pruning_joins_hailo10_logical_and_physical_keys() -> None:
    funcs = _suite_functions({
        "_resolve_stage_token",
        "_canonical_hailo_arch_key",
        "_merged_hailo_arch_meta",
        "_variant_hailo_requirements_for_run",
        "_prune_case_variants_for_run",
    })
    run = {
        "id": "hailo10_to_tensorrt",
        "type": "matrix",
        "stage1": {"type": "hailo", "hw_arch": "hailo10h"},
        "stage2": {"type": "onnxruntime", "provider": "tensorrt"},
    }
    case = {
        "hailo_case_variant_availability": {
            "hailo10": {"full": True},
            "hailo10h": {"part1": True},
        },
    }

    assert funcs["_prune_case_variants_for_run"](
        run, ["full", "part1", "composed"], case,
        {"hailo10": {"full": "hailo/hailo10h/full/compiled.hef"}},
    ) == ["full", "part1", "composed"]


def test_hailo10_generator_uses_one_canonical_split_run_id() -> None:
    plan = BenchmarkGenerationService().build_run_plan(
        acc_cpu=False,
        acc_cuda=False,
        acc_trt=True,
        acc_h8=False,
        acc_h10=True,
        hailo10_hw="hailo10h",
        hailo_preset="Custom",
        hailo_custom_full=True,
        hailo_custom_composed=True,
        hailo_custom_part1=True,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=True,
    )
    ids = [str(row.get("id") or "") for row in plan.bench_plan_runs]

    assert "hailo10_to_tensorrt" in ids
    assert "hailo10_to_trt" not in ids


def test_hailo10_short_run_id_remains_a_cli_alias() -> None:
    funcs = _suite_functions({"_canonical_plan_run_id"})
    canonical = funcs["_canonical_plan_run_id"]

    assert canonical("hailo10_to_trt") == "hailo10_to_tensorrt"
    assert canonical("hailo10h_to_trt") == "hailo10_to_tensorrt"
    assert canonical("hailo10_to_tensorrt") == "hailo10_to_tensorrt"


def test_endpoint_lifecycle_uses_the_same_canonical_hailo10_run_id() -> None:
    assert _canonical_run_profile("hailo10_to_trt") == (
        "hailo10_to_tensorrt"
    )
    assert _canonical_run_profile("hailo10h_to_trt") == (
        "hailo10_to_tensorrt"
    )
    assert _canonical_run_profile("hailo10_to_tensorrt") == (
        "hailo10_to_tensorrt"
    )


def test_hailo10p_is_not_folded_into_hailo10h_evidence() -> None:
    funcs = _suite_functions({"_canonical_hailo_arch_key"})

    assert funcs["_canonical_hailo_arch_key"]("hailo10p") == "hailo10p"


def test_generic_deepx_dispatch_identity_does_not_activate_trt_companion() -> None:
    funcs = _suite_functions({
        "_native_full_trt_quality_endpoint_id",
        "_bind_native_full_trt_quality_companion",
    })
    args = SimpleNamespace(
        quality_evidence_eval_id="eval-r8b",
        quality_evidence_model_id="yolo11l",
        quality_evidence_setup_id="deepx_setup",
        quality_evidence_endpoint_id="",
    )
    runs = [{
        "id": "ort_tensorrt",
        "_native_full_trt_quality_companion": True,
    }]

    prepared, errors = funcs["_bind_native_full_trt_quality_companion"](
        runs, args=args, quality_only_run_ids=set(),
    )

    assert errors == []
    assert prepared[0]["_native_full_trt_quality_companion"] is False


def test_explicit_plan_endpoint_authority_activates_trt_companion() -> None:
    funcs = _suite_functions({
        "_native_full_trt_quality_endpoint_id",
        "_bind_native_full_trt_quality_companion",
    })
    endpoint_id = "e" * 64
    args = SimpleNamespace(
        quality_evidence_eval_id="eval-r8b",
        quality_evidence_model_id="yolo11l",
        quality_evidence_setup_id="deepx_setup",
        quality_evidence_endpoint_id="",
    )
    runs = [{
        "id": "ort_tensorrt",
        "_native_full_trt_quality_companion": True,
        "quality_canary_endpoint_ids": [endpoint_id],
        "quality_canary_setup_ids": ["deepx_setup"],
    }]

    prepared, errors = funcs["_bind_native_full_trt_quality_companion"](
        runs, args=args, quality_only_run_ids=set(),
    )

    assert errors == []
    assert prepared[0]["_native_full_trt_quality_companion"] is True
    assert prepared[0]["_native_full_trt_quality_companion_id"] == endpoint_id
