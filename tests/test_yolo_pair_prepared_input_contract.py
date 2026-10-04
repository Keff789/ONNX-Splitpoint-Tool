"""Read real saved contracts; new files are limited to external test outputs."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import generic_completion_reporting as reporting
from onnx_splitpoint_tool.native_command_contract import verify_native_command_contract
from onnx_splitpoint_tool.runners.task_completion import load_completion_bound_array
from onnx_splitpoint_tool.workflow.generic_completed_task import prepare_runner_bundle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def originals():
    source = os.environ.get("YOLO_PAIR_INPUT_CASES")
    if not source:
        pytest.skip("requires preserved H8/H10/DeepX original contracts")
    rows = json.loads(Path(source).read_text())["rows"]
    assert len(rows) == 3
    result = []
    for row in rows:
        source = Path(row["native_source"])
        native = json.loads(source.read_text())
        command, status = verify_native_command_contract(native["native_command_contract"])
        assert command is not None, status
        result.append((row, source, command))
    assert "prepared_input" not in result[1][2]["artifacts"]
    assert result[1][2]["prepared_input_contract"]["entries"][0]["artifact_name"] == "prepared_input_00"
    return result


@pytest.mark.parametrize("index", range(3))
def test_original_contract_resolves_exact_bound_array(originals, index):
    case, source, command = originals[index]
    before = copy.deepcopy((case, command))
    path = Path(case["input_binding"]["entries"][0]["local_path"])
    original_bytes = path.read_bytes()
    entry = reporting._verified_pair_prepared_input(command, case, source, Path(case["original_run"]))
    assert entry == case["input_binding"]["entries"][0]
    assert (case, command) == before and path.read_bytes() == original_bytes


@pytest.mark.parametrize("index", [0, 1])
def test_actual_two_augmentation_pairs_complete_without_measurement(originals, index):
    case, source, command = originals[index]
    path = Path(case["result_path"])
    before = {p: p.read_bytes() for p in (source, path)}
    observed = reporting.validate_completion_case(json.loads(path.read_text()), case, result_root=path.parent)
    pair = reporting.completion_pair(observed, case)
    assert observed["valid_repetition_count"] == 3
    assert pair["native_generic_completed_endpoint_evidence"]["status"] == "verified", pair
    assert pair["eligible_for_technical_transfer"] is True
    assert pair["eligible_for_quality_transfer"] is False
    assert pair["eligible_for_claim_transfer"] is False
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("field,value", [
    ("sha256", "0" * 64), ("path", "/foreign/input.npy"),
    ("shape", [640, 640, 4]), ("dtype", "float32"), ("format", "raw"),
])
def test_runtime_alias_cannot_disagree_with_declared_native_input(originals, field, value):
    case, source, command = originals[1]
    case = copy.deepcopy(case)
    case["runtime_config"]["input_binding"]["prepared_input"][field] = value
    with pytest.raises(ValueError, match="prepared_input_projection_mismatch"):
        reporting._verified_pair_prepared_input(command, case, source, Path(case["original_run"]))


@pytest.mark.parametrize("mutation", [
    "entry_missing", "entry_duplicate", "entry_role", "entry_local_path", "entry_sha",
    "runtime_entries", "contract_slot_missing", "contract_slot_wrong", "contract_slot_duplicate",
    "unsupported_format", "contract_entries_empty", "contract_entries_multiple",
])
def test_ambiguous_or_foreign_original_binding_is_rejected(originals, mutation):
    case, source, command = originals[1]
    case, command = copy.deepcopy((case, command))
    binding = case["input_binding"]
    contract = command["prepared_input_contract"]
    if mutation == "entry_missing": binding["entries"] = []
    elif mutation == "entry_duplicate": binding["entries"].append(copy.deepcopy(binding["entries"][0]))
    elif mutation == "entry_role": binding["entries"][0]["artifact_name"] = "prepared_input"
    elif mutation == "entry_local_path": binding["entries"][0]["local_path"] = "/foreign/input.npy"
    elif mutation == "entry_sha": binding["entries"][0]["sha256"] = "0" * 64
    elif mutation == "runtime_entries": case["runtime_config"]["input_binding"]["entries"] = []
    elif mutation.startswith("contract_slot"):
        if mutation == "contract_slot_missing": contract.pop("slot_order")
        elif mutation == "contract_slot_wrong": contract["slot_order"] = ["foreign"]
        else: contract["slot_order"] = ["images", "images"]
    elif mutation == "unsupported_format": contract["format"] = "numpy_npz_v1"
    elif mutation == "contract_entries_empty": contract["entries"] = []
    elif mutation == "contract_entries_multiple": contract["entries"].append(copy.deepcopy(contract["entries"][0]))
    with pytest.raises((ValueError, KeyError)):
        reporting._verified_pair_prepared_input(command, case, source, Path(case["original_run"]))


@pytest.fixture(scope="module")
def generated_loader(tmp_path_factory):
    root = tmp_path_factory.mktemp("normal-generated-runner") / "bundle"
    prepare_runner_bundle(root)
    script = root / "runner/run_split_onnxruntime.py"
    tree = ast.parse(script.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_completion_bound_array")
    namespace = {"_sp_file_sha256": digest}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), namespace)
    return root, namespace["_completion_bound_array"]


@pytest.mark.parametrize("index", range(3))
def test_normal_generated_template_calls_shared_loader_on_real_inputs(originals, generated_loader, monkeypatch, index):
    root, loader = generated_loader
    monkeypatch.syspath_prepend(str(root))
    case, _, _ = originals[index]
    spec = copy.deepcopy(case["runtime_config"]["input_binding"]["prepared_input"])
    spec["path"] = case["input_binding"]["entries"][0]["local_path"]
    value = loader(spec)
    assert value.shape == (640, 640, 3) and value.dtype == np.uint8


@pytest.mark.parametrize("mutation", ["wrong_sha", "shape", "dtype", "npz", "object", "fortran_declared_c"])
def test_shared_loader_checks_real_file_before_normalization(tmp_path, mutation):
    path = tmp_path / "prepared.npy"
    array = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    if mutation == "dtype": array = array.astype(np.float32)
    elif mutation == "shape": array = array.reshape(4, 2, 3)
    elif mutation == "object": array = array.astype(object)
    elif mutation == "fortran_declared_c": array = np.asfortranarray(array)
    if mutation == "npz":
        with path.open("wb") as handle: np.savez(handle, images=array)
    else:
        np.save(path, array)
    spec = {"path": str(path), "sha256": digest(path), "format": "npy",
            "shape": [2, 4, 3], "dtype": "uint8", "c_contiguous": True}
    if mutation == "wrong_sha": spec["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        load_completion_bound_array(spec, file_sha256=digest)


def test_shared_loader_retains_undeclared_order_behavior(tmp_path):
    path = tmp_path / "prepared.npy"
    value = np.asfortranarray(np.arange(24, dtype=np.uint8).reshape(2, 4, 3))
    np.save(path, value)
    loaded = load_completion_bound_array({"path": str(path), "sha256": digest(path),
        "format": "npy", "shape": [2, 4, 3], "dtype": "uint8"}, file_sha256=digest)
    assert loaded.flags.c_contiguous and np.array_equal(loaded, value)
