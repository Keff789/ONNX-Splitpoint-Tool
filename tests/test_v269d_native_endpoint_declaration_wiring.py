from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
REMOTE_SCRIPTS = ROOT / "onnx_splitpoint_tool/resources/remote_scripts"


def _load_script(name: str):
    path = SCRIPTS / name
    module_name = "_osp_v269d_" + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_contracts(
    root: Path, *, model_id: str, endpoint_mode: str,
    backend: str = "hailo10",
) -> Path:
    raw = endpoint_mode in {"raw_head", "raw_detection_head"}
    path = root / "output_contracts.json"
    path.write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": model_id,
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "schema_version": 1,
            "model_id": model_id,
            "task": "detection",
            "backend": backend,
            "variant": "full",
            "contract_status": "recorded",
            "endpoint_mode": endpoint_mode,
            "host_tail_required": raw,
            "postprocessing_required": raw,
        }],
    }, sort_keys=True), encoding="utf-8")
    return path


def test_smoke_container_bn6_requires_exact_recorded_declaration(
    tmp_path: Path,
) -> None:
    module = _load_script("smoke_hailo10_hef_runner.py")
    contract_path = _write_contracts(
        tmp_path, model_id="yolo26s", endpoint_mode="decoded",
    )
    outputs = {
        "output0": np.asarray(
            [[[1.0, 2.0, 10.0, 20.0, 0.9, 1.0]]], dtype=np.float32,
        ),
    }
    declaration = module._resolve_declared_output_contract(
        contract_path, hw_arch="hailo10h", model="yolo26s", task="detection",
    )
    endpoint = module._output_contract("detection", outputs, declaration)
    assert declaration["stage"] == "decoded_nms"
    assert endpoint["endpoint_contract_complete"] is True
    assert endpoint["contract_family"] == "decoded_nms"
    assert endpoint["output_endpoint_attestation"]["attested"] is True

    # A readable container without an exact model/backend row remains
    # metadata-only.  Plausible BN6 values cannot supply the missing NMS claim.
    contract_path.write_text(json.dumps({
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "yolo26s",
        "contracts": [],
    }), encoding="utf-8")
    missing = module._resolve_declared_output_contract(
        contract_path, hw_arch="hailo10h", model="yolo26s", task="detection",
    )
    blocked = module._output_contract("detection", outputs, missing)
    assert "stage" not in missing
    assert blocked["endpoint_contract_complete"] is False
    assert blocked["contract_family"] == "unknown"
    assert blocked["output_endpoint_attestation"]["attested"] is False


def test_smoke_container_keeps_yolov7_multiscale_endpoint_raw(
    tmp_path: Path,
) -> None:
    module = _load_script("smoke_hailo10_hef_runner.py")
    contract_path = _write_contracts(
        tmp_path, model_id="yolov7_paper", endpoint_mode="raw_detection_head",
    )
    declaration = module._resolve_declared_output_contract(
        contract_path,
        hw_arch="hailo10h",
        model="yolov7_paper",
        task="detection",
    )
    outputs = {
        "output": np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
        "clone_1": np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
        "clone_2": np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
    }
    endpoint = module._output_contract("detection", outputs, declaration)
    assert declaration["stage"] == "raw_head"
    assert endpoint["endpoint_contract_complete"] is True
    assert endpoint["contract_family"] == "raw_head"
    assert endpoint["stage"] == "raw_head"
    assert endpoint["claim_eligible_e2e"] is False


def test_native_endpoint_runner_mirrors_use_authoritative_loader() -> None:
    names = (
        "native_hailo10_trt_e2e_from_benchmarkset.py",
        "native_deepx_trt_e2e_from_benchmarkset.py",
        "native_hailo_trt_fifo_from_benchmarkset.py",
        "native_full_semantic_dump.py",
        "smoke_hailo10_full_from_benchmarkset.py",
        "smoke_hailo10_hef_runner.py",
    )
    for name in names:
        source = (SCRIPTS / name).read_bytes()
        assert source == (REMOTE_SCRIPTS / name).read_bytes()
        text = source.decode("utf-8")
        if name != "smoke_hailo10_full_from_benchmarkset.py":
            assert "load_authoritative_output_contract" in text
    wrapper = (SCRIPTS / "smoke_hailo10_full_from_benchmarkset.py").read_text(
        encoding="utf-8",
    )
    assert 'benchmark_set / "output_contracts.json"' in wrapper
    assert 'hef.parent / "output_contract.json"' not in wrapper
