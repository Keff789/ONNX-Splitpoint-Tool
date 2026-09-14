from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    runtime_output_contract,
)
from onnx_splitpoint_tool.workflow.runner import duplicate_profile_measurements_v269d


ROOT = Path(__file__).resolve().parents[1]
RUNNER_TEMPLATE = (
    ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
)
SUITE_TEMPLATE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"


def _template_functions(*names: str) -> dict[str, Any]:
    tree = ast.parse(RUNNER_TEMPLATE.read_text(encoding="utf-8"))
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    module = ast.Module(body=selected, type_ignores=[])

    def file_sha256(path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def canonical_sha256(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    namespace: dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Mapping": Mapping,
        "Optional": Optional, "Sequence": Sequence, "Path": Path,
        "np": np, "onnx": onnx,
        "load_authoritative_output_contract": load_authoritative_output_contract,
        "runtime_output_contract": runtime_output_contract,
        "_read_json": lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        "_quality_file_sha256": file_sha256,
        "_quality_contract_sha256": canonical_sha256,
    }
    exec(compile(module, str(RUNNER_TEMPLATE), "exec"), namespace)
    return namespace


def _save_bn6_model(path: Path, *, output_name: str = "output0") -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], [output_name])],
        "bn6",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 300, 6])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 300, 6])],
    )
    onnx.save(helper.make_model(graph), path)


def test_recorded_bn6_contract_requires_exact_graph_and_runtime_attestation(
    tmp_path: Path,
) -> None:
    funcs = _template_functions(
        "_recorded_suite_endpoint_declaration",
        "_central_quality_endpoint_contract",
    )
    suite = tmp_path / "suite"
    case = suite / "b038"
    case.mkdir(parents=True)
    full = suite / "full.onnx"
    part2 = case / "part2.onnx"
    _save_bn6_model(full)
    _save_bn6_model(part2)
    (suite / "benchmark_set.json").write_text(
        json.dumps({"model_name": "yolo26s"}), encoding="utf-8",
    )
    (case / "split_manifest.json").write_text(json.dumps({
        "full_model": "../full.onnx", "part2_model": "part2.onnx",
    }), encoding="utf-8")
    contract = {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "yolo26s",
        "contracts": [{
            "model_id": "yolo26s", "backend": "cuda_ort", "variant": "full",
            "task": "detection",
            "endpoint_mode": "decoded", "contract_status": "recorded",
            "host_tail_required": False, "postprocessing_required": False,
        }],
    }
    (suite / "output_contracts.json").write_text(json.dumps(contract), encoding="utf-8")
    output = np.zeros((1, 300, 6), dtype=np.float32)
    declaration = funcs["_recorded_suite_endpoint_declaration"](
        base_dir=case, full_model=full, terminal_model=part2,
        variant="composed", provider="tensorrt", task="detection",
        output_names=["output0"], outputs=[output],
    )
    assert declaration["stage"] == "decoded_nms"
    endpoint = funcs["_central_quality_endpoint_contract"](
        task="detection", output_names=["output0"], outputs=[output],
        detected_output_format="bn6_detections",
        declared_endpoint_contract=declaration,
    )
    assert endpoint["endpoint_contract_complete"] is True
    assert endpoint["contract_family"] == "decoded_nms"

    # Shape/name alone is still insufficient.
    assert funcs["_central_quality_endpoint_contract"](
        task="detection", output_names=["output0"], outputs=[output],
        detected_output_format="bn6_detections",
    ) == {}

    # A different terminal graph cannot borrow the Full declaration.
    mismatched = case / "mismatched.onnx"
    _save_bn6_model(mismatched, output_name="other")
    assert funcs["_recorded_suite_endpoint_declaration"](
        base_dir=case, full_model=full, terminal_model=mismatched,
        variant="composed", provider="tensorrt", task="detection",
        output_names=["output0"], outputs=[output],
    ) == {}

    wrong_model = dict(contract)
    wrong_model["model_id"] = "unrelated_model"
    wrong_model["contracts"] = [
        {**contract["contracts"][0], "model_id": "unrelated_model"}
    ]
    (suite / "output_contracts.json").write_text(
        json.dumps(wrong_model), encoding="utf-8",
    )
    assert funcs["_recorded_suite_endpoint_declaration"](
        base_dir=case, full_model=full, terminal_model=part2,
        variant="composed", provider="tensorrt", task="detection",
        output_names=["output0"], outputs=[output],
    ) == {}


def test_yolo_auto_preprocessing_is_not_selected_by_visual_sample() -> None:
    source = RUNNER_TEMPLATE.read_text(encoding="utf-8")
    auto_block = source[source.index("# Auto mode must be a model/input contract"):]
    auto_block = auto_block[: auto_block.index("elif _is_detr_like")]
    assert "_yolo_plausibility" not in auto_block
    assert '_run_full_with_scale("raw")' not in auto_block
    assert 'detector_scale = (' in auto_block
    assert "visual sample does not select scientific preprocessing" in auto_block


def test_full_only_suite_run_uses_one_execution_container() -> None:
    tree = ast.parse(SUITE_TEMPLATE.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_normalize_variants", "_case_identifiers", "_cases_for_run"}
    ]
    namespace: dict[str, Any] = {
        "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SUITE_TEMPLATE), "exec"), namespace)
    cases = [{"case_id": "b044"}, {"case_id": "b116"}, {"case_id": "b216"}]
    assert namespace["_cases_for_run"](cases, {"variants": ["full"]}) == [cases[0]]
    assert namespace["_cases_for_run"](
        cases, {"variants": ["full", "composed"]}
    ) == cases


def test_required_profile_duplicate_is_a_blocking_matrix_error() -> None:
    expected = [{"backend": "tensorrt", "variant": "full", "case_id": "full"}]
    measured = [
        {"backend": "tensorrt", "variant": "full", "case_id": case}
        for case in ("b044", "b116", "b216")
    ]
    duplicate = duplicate_profile_measurements_v269d(expected, measured)
    assert duplicate == [{
        "status": "duplicate",
        "error_class": "duplicate_selected_profile_measurement",
        "backend": "tensorrt",
        "variant": "full",
        "case_id": "full",
        "row_count": 3,
        "source_paths": [],
    }]
