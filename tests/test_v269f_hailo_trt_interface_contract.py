from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract


ROOT = Path(__file__).resolve().parents[1]


def _load_validator(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator(
    ROOT / "scripts" / "native_producer_validate_visualize.py",
    "v269f_hailo_trt_interface_validator",
)
REMOTE_VALIDATOR = _load_validator(
    ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
    / "native_producer_validate_visualize.py",
    "v269f_hailo_trt_interface_remote_validator",
)
VALIDATOR_COPIES = (VALIDATOR, REMOTE_VALIDATOR)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Any]:
    model = "resnet50"
    case = "b052"
    precision = "float32_layout_fp16"
    setup_id = "orin_nx_hailo8_01"
    local_bs = tmp_path / "native_producers" / "hailo8" / model / "benchmark_set"
    remote_bs = Path("/home/nx/native_fifo_evalsets/run") / model / "benchmark_set"
    report_path = (
        local_bs / "native_pipeline" / case / "hailo_to_trt" / precision
        / "native_fifo_results.json"
    )
    report_path.parent.mkdir(parents=True)
    metadata_path = (
        local_bs / "native_trt" / case / "part2" / precision
        / "native_trt_meta.json"
    )
    remote_engine = (
        remote_bs / "native_trt" / case / "part2" / precision
        / "part2_float32_layout_fp16.engine"
    )
    remote_metadata = remote_engine.parent / "native_trt_meta.json"
    boundary_dir = report_path.parent / "native_fifo_boundary"
    boundary_file = boundary_dir / "boundary.bin"
    boundary_file.parent.mkdir(parents=True)
    boundary_file.write_bytes(b"\x00" * 32)
    remote_boundary_file = (
        remote_bs / "native_pipeline" / case / "hailo_to_trt" / precision
        / "native_fifo_boundary" / boundary_file.name
    )
    boundary_manifest = boundary_dir / "native_fifo_boundary_manifest.json"
    manifest = {
        "schema": "onnx-splitpoint/native-boundary-dump",
        "schema_version": 2,
        "file": str(remote_boundary_file),
        "shape": [1, 2, 2, 2],
        "dtype": "float32",
        "nbytes": 32,
        "trt_input_name": "cut",
        "trt_input_dtype": "float32",
        "trt_input_bytes": 32,
    }
    _write_json(boundary_manifest, manifest)
    metadata = {
        "engine": str(remote_engine),
        "precision": precision,
        "inputs": [{
            "name": "cut", "shape": [1, 2, 2, 2],
            "elem_type": "FLOAT", "has_dynamic": False,
        }],
        "uint8_cast_bridge": {
            "schema": "onnx-splitpoint/float32-layout-bridge",
            "schema_version": 1,
            "input_name": "cut",
            "input_shape": [1, 2, 2, 2],
            "input_dtype": "FLOAT",
            "replaced_uses": 0,
            "boundary_layout": {
                "requested": "as_input", "effective": "as_input",
                "applied": False,
            },
        },
    }
    _write_json(metadata_path, metadata)

    fixture: dict[str, Any] = {
        "model": model,
        "case": case,
        "precision": precision,
        "setup_id": setup_id,
        "local_bs": local_bs,
        "remote_bs": remote_bs,
        "report_path": report_path,
        "metadata_path": metadata_path,
        "remote_engine": remote_engine,
        "remote_metadata": remote_metadata,
        "boundary_file": boundary_file,
        "boundary_manifest": boundary_manifest,
        "metadata": metadata,
        "manifest": manifest,
    }
    _rebind_contract(fixture)
    return fixture


def _rebind_contract(fixture: dict[str, Any]) -> None:
    _write_json(fixture["metadata_path"], fixture["metadata"])
    metadata_sha = _sha256_file(fixture["metadata_path"])
    remote_bs = fixture["remote_bs"]
    contract = seal_native_command_contract({
        "complete": True,
        "backend": "hailo8_to_trt",
        "model": fixture["model"],
        "case": fixture["case"],
        "precision": fixture["precision"],
        "setup_id": fixture["setup_id"],
        "comparison_backend": "hailo8",
        "runner": str(remote_bs / "scripts" / "native_hailo_trt_fifo_from_benchmarkset.py"),
        "runner_sha256": "a" * 64,
        "python_executable": "/home/nx/venv/bin/python",
        "interpreter_identity": {
            "executable": "/home/nx/venv/bin/python",
            "resolved_executable": "/home/nx/venv/bin/python3.10",
            "executable_sha256": "1" * 64,
        },
        "benchmark_set": str(remote_bs),
        "input_image": "/home/nx/dataset/image.jpg",
        "input_image_sha256": "b" * 64,
        "artifacts": {
            "python_executable": {
                "path": "/home/nx/venv/bin/python", "sha256": "1" * 64,
            },
            "hef": {"path": "/home/nx/model.hef", "sha256": "2" * 64},
            "engine": {
                "path": str(fixture["remote_engine"]), "sha256": "3" * 64,
            },
            "native_trt_meta": {
                "path": str(fixture["remote_metadata"]), "sha256": metadata_sha,
            },
            "native_executable": {
                "path": "/home/nx/native_fifo", "sha256": "4" * 64,
            },
            "generated_cpp": {"path": "/home/nx/main.cpp", "sha256": "5" * 64},
            "cmake": {"path": "/home/nx/CMakeLists.txt", "sha256": "6" * 64},
        },
        "runtime_options": {
            "warmup": 10, "queue_depth": 3, "dump_boundary": True,
        },
        "boundary_contract": {
            "metadata_path": str(fixture["remote_metadata"]),
            "metadata_sha256": metadata_sha,
            "boundary_layout_requested": fixture["metadata"]["uint8_cast_bridge"]
                ["boundary_layout"]["requested"],
            "boundary_layout_effective": fixture["metadata"]["uint8_cast_bridge"]
                ["boundary_layout"]["effective"],
            "bridge_schema": fixture["metadata"]["uint8_cast_bridge"]["schema"],
            "dequant_scale": fixture["metadata"]["uint8_cast_bridge"].get("scale"),
            "dequant_zero_point": fixture["metadata"]["uint8_cast_bridge"].get("zero_point"),
        },
    })
    fixture["contract"] = contract
    row = {
        "backend": "hailo8_to_trt",
        "model": fixture["model"], "case": fixture["case"],
        "precision": fixture["precision"], "setup_id": fixture["setup_id"],
        "comparison_backend": "hailo8",
        "native_command_contract": copy.deepcopy(contract),
        "native_command_contract_sha256": contract["contract_sha256"],
        # Producer claims are deliberately untrusted and must be overwritten.
        "interface_contract_pass": False,
    }
    report = {
        **{key: row[key] for key in (
            "backend", "model", "case", "precision", "setup_id",
            "comparison_backend",
        )},
        "native_command_contract": copy.deepcopy(contract),
        "native_command_contract_sha256": contract["contract_sha256"],
        "workload_contract_sha256": contract["contract_sha256"],
        "native_fifo_boundary_manifest": str(
            fixture["remote_bs"] / "native_pipeline" / fixture["case"]
            / "hailo_to_trt" / fixture["precision"] / "native_fifo_boundary"
            / "native_fifo_boundary_manifest.json"
        ),
        "trt_input_dtype": "float32",
        "trt_input_bytes": 32,
        "trt_inputs": ["cut"],
        "boundary_layout": fixture["metadata"]["uint8_cast_bridge"]
            ["boundary_layout"]["effective"],
        "interface_contract_pass": False,
    }
    fixture["row"] = row
    fixture["report"] = report


def _validate(fixture: dict[str, Any]) -> dict[str, Any]:
    return VALIDATOR._validate_hailo_trt_interface_contract(
        fixture["row"], fixture["report"],
        report=fixture["report_path"], roots=[fixture["local_bs"].parents[2]],
    )


def _singleton_identity_evidence(
    *, runtime_shape: list[int], trt_shape: list[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata_sha = "7" * 64
    boundary = {
        "backend": "hailo10h_to_trt",
        "hailo_output_name": "mean",
        "trt_input_name": "mean",
        "boundary_layout": "as_input",
        "boundary_shape_source": "hailo_runtime_output_binding",
        "layout_transform_owner": "tensorrt_part2_input_bridge",
        "runtime_boundary_shape": list(runtime_shape),
        "trt_input_shape": list(trt_shape),
    }
    bridge = {
        "input_name": "mean",
        "input_shape": list(trt_shape),
        "input_dtype": "FLOAT",
        "replaced_uses": 0,
        "boundary_layout": {
            "requested": "as_input",
            "effective": "as_input",
            "applied": False,
        },
    }
    contract = {
        "precision": "float32_layout_fp16",
        "boundary_contract": {
            "boundary_layout_requested": "as_input",
            "boundary_layout_effective": "as_input",
        },
        "quality_boundary_contract": {
            "precision": "float32_layout_fp16",
            "boundary_layout": "as_input",
            "boundary_transform": "identity",
            "boundary_tensor_name": "mean",
            "boundary_tensor_shape": list(runtime_shape),
            "boundary_tensor_dtype": "float32",
            "boundary_metadata_sha256": metadata_sha,
        },
        "quality_preselection": {
            "boundary_layout": "as_input",
            "boundary_transform": "identity",
            "boundary_tensor_name": "mean",
            "boundary_tensor_shape": list(runtime_shape),
            "boundary_tensor_dtype": "float32",
            "canonical_part2_shape": list(trt_shape),
        },
        "runtime_boundary_evidence": {
            "status": "exact_runtime_boundary_verified",
            "runtime_name": "mean",
            "shape": list(runtime_shape),
            "dtype": "float32",
            "element_count": 1,
            "trt_input_name": "mean",
            "trt_input_shape": list(trt_shape),
            "binding_boundary_metadata_sha256": metadata_sha,
        },
    }
    element_count = 1
    for dim in trt_shape:
        element_count *= dim
    contract["runtime_boundary_evidence"]["element_count"] = element_count
    return boundary, bridge, contract


@pytest.mark.parametrize("validator", VALIDATOR_COPIES)
@pytest.mark.parametrize(
    ("runtime_shape", "trt_shape"),
    [
        ([2048], [1, 2048, 1, 1]),
        ([1, 2, 1, 3], [2, 3, 1]),
        ([1], [1, 1, 1, 1]),
    ],
)
def test_as_input_identity_accepts_only_singleton_axis_shape_differences(
    validator: Any, runtime_shape: list[int], trt_shape: list[int],
) -> None:
    boundary, bridge, contract = _singleton_identity_evidence(
        runtime_shape=runtime_shape, trt_shape=trt_shape,
    )

    assert validator._as_input_identity_singleton_contract_verified(
        backend="hailo10h_to_trt",
        input_name="mean",
        input_shape=trt_shape,
        input_dtype="float32",
        manifest_shape=runtime_shape,
        boundary=boundary,
        bridge=bridge,
        contract=contract,
    ) is True


@pytest.mark.parametrize("validator", VALIDATOR_COPIES)
def test_as_input_uint8_dequant_accepts_only_singleton_axis_shape_difference(
    validator: Any,
) -> None:
    runtime_shape = [2048]
    trt_shape = [1, 2048, 1, 1]
    boundary, bridge, contract = _singleton_identity_evidence(
        runtime_shape=runtime_shape, trt_shape=trt_shape,
    )
    bridge.update({
        "schema": "onnx-splitpoint/uint8-dequant-bridge",
        "input_dtype": "UINT8",
        "replaced_uses": 1,
        "scale": 0.03125,
        "zero_point": 17.0,
    })
    contract["precision"] = "uint8_dequant_fp16"
    for field in ("quality_boundary_contract", "quality_preselection"):
        contract[field].update({
            "precision": "uint8_dequant_fp16",
            "boundary_transform": "uint8_dequant",
            "boundary_tensor_dtype": "uint8",
        })
    contract["runtime_boundary_evidence"]["dtype"] = "uint8"

    assert validator._as_input_identity_singleton_contract_verified(
        backend="hailo10h_to_trt",
        input_name="mean",
        input_shape=trt_shape,
        input_dtype="uint8",
        manifest_shape=runtime_shape,
        boundary=boundary,
        bridge=bridge,
        contract=contract,
    ) is True

    bridge["replaced_uses"] = 0
    assert validator._as_input_identity_singleton_contract_verified(
        backend="hailo10h_to_trt",
        input_name="mean",
        input_shape=trt_shape,
        input_dtype="uint8",
        manifest_shape=runtime_shape,
        boundary=boundary,
        bridge=bridge,
        contract=contract,
    ) is False


@pytest.mark.parametrize("validator", VALIDATOR_COPIES)
@pytest.mark.parametrize(
    ("runtime_shape", "trt_shape"),
    [
        ([2, 1024], [1, 2048, 1, 1]),
        ([2, 3], [1, 3, 2, 1]),
        ([2048], [2048]),
    ],
)
def test_as_input_identity_rejects_product_only_order_and_exact_shape_cases(
    validator: Any, runtime_shape: list[int], trt_shape: list[int],
) -> None:
    boundary, bridge, contract = _singleton_identity_evidence(
        runtime_shape=runtime_shape, trt_shape=trt_shape,
    )

    assert validator._as_input_identity_singleton_contract_verified(
        backend="hailo10h_to_trt",
        input_name="mean",
        input_shape=trt_shape,
        input_dtype="float32",
        manifest_shape=runtime_shape,
        boundary=boundary,
        bridge=bridge,
        contract=contract,
    ) is False


@pytest.mark.parametrize("validator", VALIDATOR_COPIES)
@pytest.mark.parametrize(
    "mutation",
    (
        "transform",
        "bridge_applied",
        "bridge_replaced_uses",
        "runtime_manifest_shape",
        "trt_manifest_shape",
        "canonical_shape",
        "metadata_binding",
    ),
)
def test_singleton_shape_exception_requires_every_identity_gate(
    validator: Any, mutation: str,
) -> None:
    runtime_shape = [2048]
    trt_shape = [1, 2048, 1, 1]
    boundary, bridge, contract = _singleton_identity_evidence(
        runtime_shape=runtime_shape, trt_shape=trt_shape,
    )
    if mutation == "transform":
        contract["quality_boundary_contract"]["boundary_transform"] = (
            "layout_only"
        )
    elif mutation == "bridge_applied":
        bridge["boundary_layout"]["applied"] = True
    elif mutation == "bridge_replaced_uses":
        bridge["replaced_uses"] = 1
    elif mutation == "runtime_manifest_shape":
        boundary["runtime_boundary_shape"] = [1, 2048]
    elif mutation == "trt_manifest_shape":
        boundary["trt_input_shape"] = [2048]
    elif mutation == "canonical_shape":
        contract["quality_preselection"]["canonical_part2_shape"] = [2048]
    elif mutation == "metadata_binding":
        contract["runtime_boundary_evidence"][
            "binding_boundary_metadata_sha256"
        ] = "8" * 64
    else:  # pragma: no cover
        raise AssertionError(mutation)

    assert validator._as_input_identity_singleton_contract_verified(
        backend="hailo10h_to_trt",
        input_name="mean",
        input_shape=trt_shape,
        input_dtype="float32",
        manifest_shape=runtime_shape,
        boundary=boundary,
        bridge=bridge,
        contract=contract,
    ) is False


@pytest.mark.parametrize("validator", VALIDATOR_COPIES)
def test_real_b119_singleton_contract_runs_through_remaining_identity_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, validator: Any,
) -> None:
    fixture = _fixture(tmp_path)
    runtime_shape = [8]
    trt_shape = [1, 8, 1, 1]
    boundary, bridge, identity = _singleton_identity_evidence(
        runtime_shape=runtime_shape, trt_shape=trt_shape,
    )
    fixture["manifest"].update(boundary)
    fixture["manifest"]["shape"] = runtime_shape
    _write_json(fixture["boundary_manifest"], fixture["manifest"])
    fixture["metadata"]["inputs"][0].update({
        "name": "mean", "shape": trt_shape,
    })
    fixture["metadata"]["uint8_cast_bridge"].update(bridge)
    _write_json(fixture["metadata_path"], fixture["metadata"])
    metadata_sha = _sha256_file(fixture["metadata_path"])
    contract = copy.deepcopy(fixture["contract"])
    for field in (
        "quality_boundary_contract", "quality_preselection",
        "runtime_boundary_evidence",
    ):
        contract[field] = identity[field]
    contract["backend"] = "hailo10h_to_trt"
    contract["comparison_backend"] = "hailo10h"
    contract["contract_sha256"] = fixture["contract"]["contract_sha256"]
    fixture["row"].update({
        "backend": "hailo10h_to_trt",
        "comparison_backend": "hailo10h",
        "native_command_contract": copy.deepcopy(contract),
    })
    fixture["report"].update({
        "backend": "hailo10h_to_trt",
        "comparison_backend": "hailo10h",
        "trt_inputs": ["mean"],
        "native_command_contract": copy.deepcopy(contract),
    })
    contract["boundary_contract"].update({
        "metadata_sha256": metadata_sha,
        "boundary_layout_requested": "as_input",
        "boundary_layout_effective": "as_input",
    })
    fixture["row"]["native_command_contract"] = copy.deepcopy(contract)
    fixture["report"]["native_command_contract"] = copy.deepcopy(contract)

    monkeypatch.setattr(
        validator,
        "verify_native_command_contract",
        lambda raw, expected_identity: (raw, "verified_test_contract"),
    )

    result = validator._validate_hailo_trt_interface_contract(
        fixture["row"], fixture["report"],
        report=fixture["report_path"],
        roots=[fixture["local_bs"].parents[2]],
    )

    assert result["interface_contract_pass"] is True, result
    assert result["interface_contract_status"] == (
        "verified_native_command_metadata_boundary_and_bridge"
    )
    assert "as_input_identity_singleton_shape_equivalence_verified" in (
        result["interface_contract_verification"]["checks"]
    )
    assert "trt_input_name_shape_dtype_and_exact_bytes_verified" in (
        result["interface_contract_verification"]["checks"]
    )
    assert "bridge_schema_layout_and_parameters_verified" in (
        result["interface_contract_verification"]["checks"]
    )


def test_hailo_trt_interface_is_recomputed_from_bound_artifacts(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    result = _validate(fixture)

    assert result["interface_contract_pass"] is True
    assert result["interface_check_pass"] is True
    assert result["strict_boundary_numeric_pass"] is None
    assert result["interface_contract_status"] == (
        "verified_native_command_metadata_boundary_and_bridge"
    )
    assert result["interface_contract_verification"]["metadata_sha256"] == (
        _sha256_file(fixture["metadata_path"])
    )


def test_hailo_trt_interface_rejects_unsealed_command_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    for target in (fixture["row"], fixture["report"]):
        target["native_command_contract"]["runtime_options"]["queue_depth"] = 99

    result = _validate(fixture)

    assert result["interface_contract_pass"] is False
    assert result["interface_contract_status"] == "native_command_contract_sha256_mismatch"


def test_hailo_trt_interface_rejects_conflicting_contract_copies(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["row"]["native_command_contract"] = seal_native_command_contract({
        **{
            key: value for key, value in fixture["contract"].items()
            if key not in {"contract_sha256", "queue_depth"}
        },
        "runtime_options": {
            **fixture["contract"]["runtime_options"], "queue_depth": 99,
        },
    })

    result = _validate(fixture)

    assert result["interface_contract_pass"] is False
    assert result["interface_contract_status"] == "native_command_contract_copies_conflict"


def test_hailo_trt_interface_rejects_metadata_file_tamper(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["metadata"]["diagnostic_tamper"] = True
    _write_json(fixture["metadata_path"], fixture["metadata"])

    result = _validate(fixture)

    assert result["interface_contract_pass"] is False
    assert result["interface_contract_status"] == "native_trt_metadata_sha256_mismatch"


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ("second_input", "native_trt_exactly_one_input_required"),
        ("input_name", "native_boundary_tensor_identity_mismatch"),
        ("input_shape", "native_boundary_tensor_identity_mismatch"),
        ("input_dtype", "native_boundary_tensor_identity_mismatch"),
        ("bridge_schema", "native_trt_bridge_schema_mismatch"),
        ("bridge_input", "native_trt_bridge_input_contract_mismatch"),
        ("layout", "native_trt_bridge_layout_transform_mismatch"),
    ],
)
def test_hailo_trt_interface_rejects_resealed_metadata_mismatch(
    tmp_path: Path, mutation: str, expected_status: str,
) -> None:
    fixture = _fixture(tmp_path)
    metadata = fixture["metadata"]
    if mutation == "second_input":
        metadata["inputs"].append(copy.deepcopy(metadata["inputs"][0]))
    elif mutation == "input_name":
        metadata["inputs"][0]["name"] = "other"
    elif mutation == "input_shape":
        metadata["inputs"][0]["shape"] = [1, 2, 2, 1]
    elif mutation == "input_dtype":
        metadata["inputs"][0]["elem_type"] = "UINT8"
    elif mutation == "bridge_schema":
        metadata["uint8_cast_bridge"]["schema"] = "untrusted/bridge"
    elif mutation == "bridge_input":
        metadata["uint8_cast_bridge"]["input_name"] = "other"
    elif mutation == "layout":
        metadata["uint8_cast_bridge"]["boundary_layout"] = {
            "requested": "memory_nhwc_to_nchw",
            "effective": "memory_nhwc_to_nchw",
            "applied": False,
        }
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _rebind_contract(fixture)

    result = _validate(fixture)

    assert result["interface_contract_pass"] is False
    assert result["interface_contract_status"] == expected_status


@pytest.mark.parametrize("mutation", ("declared_nbytes", "actual_file_size"))
def test_hailo_trt_interface_rejects_boundary_byte_mismatch(
    tmp_path: Path, mutation: str,
) -> None:
    fixture = _fixture(tmp_path)
    if mutation == "declared_nbytes":
        fixture["manifest"]["nbytes"] = 31
        _write_json(fixture["boundary_manifest"], fixture["manifest"])
    else:
        fixture["boundary_file"].write_bytes(b"\x00" * 31)

    result = _validate(fixture)

    assert result["interface_contract_pass"] is False
    assert result["interface_contract_status"] == "native_boundary_byte_count_mismatch"


def test_hailo_trt_interface_is_unavailable_without_metadata_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = {
        key: value for key, value in fixture["contract"].items()
        if key != "contract_sha256"
    }
    payload["boundary_contract"] = {
        "boundary_layout_requested": "as_input",
        "boundary_layout_effective": "as_input",
    }
    contract = seal_native_command_contract(payload)
    for target in (fixture["row"], fixture["report"]):
        target["native_command_contract"] = copy.deepcopy(contract)
        target["native_command_contract_sha256"] = contract["contract_sha256"]
    fixture["report"]["workload_contract_sha256"] = contract["contract_sha256"]

    result = _validate(fixture)

    assert result["interface_contract_pass"] is None
    assert result["interface_contract_status"] == "native_trt_metadata_binding_unavailable"


def test_remote_validator_and_hailo10_runner_are_exact_mirrors() -> None:
    pairs = [
        (
            ROOT / "scripts" / "native_producer_validate_visualize.py",
            ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
            / "native_producer_validate_visualize.py",
        ),
        (
            ROOT / "scripts" / "native_hailo10_trt_e2e_from_benchmarkset.py",
            ROOT / "onnx_splitpoint_tool" / "resources" / "remote_scripts"
            / "native_hailo10_trt_e2e_from_benchmarkset.py",
        ),
    ]
    for local, remote in pairs:
        assert remote.read_bytes() == local.read_bytes()
