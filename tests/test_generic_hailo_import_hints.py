"""Original H10 hints fill an imported H8 suite without a compiler request.

All artifact bytes are owned unit-test fixtures; receipt validation is the
only replaced production operation. No SDK, accelerator, or inference runs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper
import pytest

from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    _hailo_observations,
    build_artifact_cache_preflight,
    resolve_artifact_cache_preflight_policy,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import materialize_backend_artifact_decisions
from onnx_splitpoint_tool.workflow.hailo_remote_binding import materialize_hailo_artifact_service_plan
from test_generic_quality_suite_import import _fixture, _import, _write


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path, *, key="path"):
    return {key: str(path), "sha256": _digest(path), "size_bytes": path.stat().st_size}


def _seal_config(data):
    _write(data["config_path"], data["config"])
    run = data["run"]
    run["generic_quality_completion_sources"]["b001"] = _descriptor(data["config_path"])
    run["generic_quality_completion_config_sha256"]["b001"] = _digest(data["config_path"])


def _prepared(tmp_path):
    data = _fixture(tmp_path)
    run = data["profile"]["run_profiles"][0]
    run.update(id="hailo10_to_trt", stage1={"type": "hailo", "hw_arch": "hailo10"},
               hardware_setup_id="own_hailo10_setup")
    data["profile"]["run_profiles"] = [run]
    data["profile"]["hailo_build"] = {
        "mode": "reuse_only", "force_build": False, "calib_count": 500,
        "calib_batch_size": 8, "optimization_level": 1,
        "calibration_storage": "memmap", "calibration_memory_cap_mb": 256,
        "cache_integrity": "relaxed",
    }
    calibration = tmp_path / "own_calibration_manifest.json"
    _write(calibration, {"OWN_FIXTURE_ONLY": True, "sample_count": 500})
    data["profile"]["campaign"] = {"dataset_manifests": {
        "detection": {"calibration": str(calibration)}}}
    for row in data["catalogue"]:
        hef = data["source"] / row["case_id"] / "hailo/hailo8/part1/compiled.hef"
        hef.parent.mkdir(parents=True, exist_ok=True)
        hef.write_bytes(("OWN historical H8 " + row["case_id"]).encode())
    case = data["source"] / "b001"
    _write(case / "split_manifest.json", {"boundary": 1, "part1": "part1.onnx",
           "part2": "part2.onnx", "cut_tensors": ["own_cut"]})
    graph = helper.make_graph(
        [helper.make_node("Identity", ["own_input"], ["own_cut"])], "OWN_P1_FIXTURE_ONLY",
        [helper.make_tensor_value_info("own_input", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("own_cut", TensorProto.FLOAT, [1, 3, 640, 640])],
    )
    onnx.save(helper.make_model(graph), case / "part1.onnx")
    original = tmp_path / "own_original_regular_generation/b001/hailo/hailo10/part1"
    original.mkdir(parents=True)
    hef, receipt_path = original / "compiled.hef", original / "hailo_hef_build_receipt.json"
    hef.write_bytes(b"OWN original regular H10 HEF fixture\x00")
    preprocessing = {"target_hw": [640, 640], "preprocess_mode": "letterbox",
                     "pad_value": 114, "color_space": "RGB"}
    preprocessing_sha = hashlib.sha256(json.dumps(preprocessing, sort_keys=True).encode()).hexdigest()
    payload = {
        "schema": "onnx-splitpoint/hailo-hef-cache/v3",
        "model_sha256": _digest(case / "part1.onnx"), "hw_arch": "hailo10h",
        "net_name": "model_part1_b1", "net_input_shapes": [1, 3, 640, 640],
        "disable_rt_metadata_extraction": True, "optimization_level": 1,
        "calibration_count": 500, "requested_calibration_count": 500,
        "calibration_batch_size": 8, "calibration_storage": "memmap",
        "calibration_memory_cap_bytes": 256 * 1024**2, "integrity": "relaxed",
        "calibration_identity": "manifest:" + _digest(calibration),
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha,
    }
    receipt = {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v1", "hw_arch": "hailo10h",
        "source_onnx_sha256": _digest(case / "part1.onnx"),
        "compiler_onnx_sha256": _digest(case / "part1.onnx"),
        "hef_sha256": _digest(hef), "hef_size_bytes": hef.stat().st_size,
        "net_name": "model_part1_b1", "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha, "cache_payload": payload,
        "cache_key": hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
    }
    _write(receipt_path, receipt)
    config_path = Path(run["generic_quality_completion_sources"]["b001"]["path"])
    config = json.loads(config_path.read_text())
    artifacts = {
        "part1_runtime": _descriptor(hef, key="local_path"),
        "part1_build_receipt": _descriptor(receipt_path, key="local_path"),
        "source_part1_onnx": {**_descriptor(case / "part1.onnx"), "path": "b001/part1.onnx"},
    }
    config.update(
        identity={"model_id": "model", "case_id": "b001", "setup_id": run["hardware_setup_id"],
                  "comparison_backend": "hailo10h"},
        expected_runtime_artifacts=artifacts,
        generic_artifact_binding={"artifacts": deepcopy(artifacts)},
        completion_contract={"processor_contract": {"input_hw": [640, 640]}},
        preprocess={"mode": "letterbox", "pad_value": 114, "rgb": True},
    )
    data.update(run=run, config=config, config_path=config_path, original_hef=hef,
                original_receipt=receipt_path, receipt=receipt)
    _seal_config(data)
    assert not list(data["source"].rglob("deferred_hailo_build.json"))
    assert not list(data["source"].glob("*/hailo/hailo10/part1/compiled.hef"))
    return data


def _verifier(data, monkeypatch):
    calls = []

    def verify(path, **kwargs):
        path = Path(path)
        assert path.read_bytes() == data["original_hef"].read_bytes()
        assert path.with_name("hailo_hef_build_receipt.json").read_bytes() == data["original_receipt"].read_bytes()
        if kwargs:
            assert path == data["original_hef"]
            assert kwargs == {
                "preprocessing_sha256": data["receipt"]["preprocessing_contract_sha256"],
                "source_onnx_sha256": data["config"]["expected_runtime_artifacts"]["source_part1_onnx"]["sha256"],
                "cache_key": data["receipt"]["cache_key"], "cache_payload": data["receipt"]["cache_payload"],
                "expected_net_name": "model_part1_b1", "expected_net_input_shapes": [1, 3, 640, 640],
                "expected_disable_rt_metadata_extraction": True,
            }
        calls.append((path, kwargs))
        return deepcopy(data["receipt"])

    monkeypatch.setattr(hailo_backend, "_load_valid_hailo_receipt", verify)
    return calls


def _ordinary_observations(data, result):
    contract = json.loads(result.artifacts["benchmark_set_json"].read_text())
    decision_result = materialize_backend_artifact_decisions(
        run_dir=data["run_root"], model_id="model", targets=["hailo10"],
        full_baseline_plan={}, output_contracts={}, benchmark_set_contract=contract)
    decisions = json.loads(decision_result["artifacts"]["backend_artifact_decisions_json"].read_text())
    service_result = materialize_hailo_artifact_service_plan(
        run_dir=data["run_root"], model_id="model", targets=["hailo10"],
        full_baseline_plan={}, output_contracts={}, benchmark_set_contract=contract,
        backend_artifact_decisions=decisions, build_backend_artifacts_request={},
        no_remote=True, execution_mode="generate_and_run")
    service = json.loads(service_result["artifacts"]["hailo_artifact_service_plan_json"].read_text())
    rows, roles = _hailo_observations(run_root=data["run_root"], model_id="model",
        policy=resolve_artifact_cache_preflight_policy(data["profile"]))
    return service, rows, roles


def test_original_h10_hint_is_copied_and_ordinary_preflight_discovers_reuse(tmp_path, monkeypatch):
    data = _prepared(tmp_path)
    originals = {path: path.read_bytes() for root in (data["source"], data["original_hef"].parent)
                 for path in root.rglob("*") if path.is_file()}
    calls = _verifier(data, monkeypatch)
    result = _import(data, targets=["hailo10", "tensorrt"])
    target = result.suite_dir / "b001/hailo/hailo10/part1/compiled.hef"
    assert target.read_bytes() == data["original_hef"].read_bytes()
    assert target.with_name("hailo_hef_build_receipt.json").read_bytes() == data["original_receipt"].read_bytes()
    assert target.stat().st_ino != data["original_hef"].stat().st_ino
    assert target.with_name("hailo_hef_build_receipt.json").stat().st_ino != data["original_receipt"].stat().st_ino
    provenance = json.loads(target.with_name("generic_quality_original_hailo_reuse.json").read_text())
    assert provenance["compiler_dispatch_count"] == 0
    assert provenance["original_receipt_bytes_preserved"] is True
    assert not list(result.suite_dir.rglob("deferred_hailo_build.json"))
    service, rows, roles = _ordinary_observations(data, result)
    assert service["build_attempt_count"] == 0
    assert len(service["case_hef_requests"]) == 1
    discovered = service["case_hef_requests"][0]
    assert discovered["case_id"] == "b001" and discovered["stage"] == "part1"
    assert discovered["status"] == "ready_existing_hef" and discovered["ready"] is True
    assert (data["run_root"] / discovered["hef_path"]).resolve() == target
    assert [(row.item_id, row.status, row.reason) for row in rows] == [
        ("b001:part1", "HIT", "existing_hailo_receipt_valid")]
    assert rows[0].evidence["current_build"] is False
    report = build_artifact_cache_preflight(model_ids=["model"], observations=rows,
        applicable_roles={"model": roles}, block_on_unexpected_cold_builds=True)
    assert report["runtime_dispatch_allowed"] is True
    assert len(calls) == 2 and calls[0][1] and calls[1] == (target, {})
    assert all(path.read_bytes() == body for path, body in originals.items())


@pytest.mark.parametrize("fault,reason", [
    ("part1_graph", "source_onnx_changed"), ("receipt_arch", "receipt_arch_changed"),
    ("nonadjacent_receipt", "original_receipt_not_adjacent"),
])
def test_changed_graph_or_original_receipt_is_rejected_before_materialization(
        tmp_path, monkeypatch, fault, reason):
    data = _prepared(tmp_path)
    if fault == "part1_graph":
        (data["source"] / "b001/part1.onnx").write_bytes(b"OWN mismatching P1 graph")
    else:
        receipt_path = data["original_receipt"]
        if fault == "receipt_arch":
            data["receipt"]["hw_arch"] = "hailo8"
            _write(receipt_path, data["receipt"])
        else:
            receipt_path = tmp_path / "own_nonadjacent_receipt.json"
            _write(receipt_path, {**data["receipt"], "OWN_DIFFERENT_RECEIPT": True})
        descriptor = _descriptor(receipt_path, key="local_path")
        data["config"]["expected_runtime_artifacts"]["part1_build_receipt"] = descriptor
        data["config"]["generic_artifact_binding"]["artifacts"]["part1_build_receipt"] = deepcopy(descriptor)
        _seal_config(data)
    original_hef = data["original_hef"].read_bytes()
    original_receipt = data["original_receipt"].read_bytes()
    calls = _verifier(data, monkeypatch)
    with pytest.raises(ValueError, match="generic_quality_hailo_import_hint_" + reason):
        _import(data, targets=["hailo10", "tensorrt"])
    target = data["run_root"] / "models/model/benchmark_set/legacy_suite/b001/hailo/hailo10/part1"
    assert not (target / "compiled.hef").exists()
    assert not (target / "hailo_hef_build_receipt.json").exists()
    assert calls == []
    assert data["original_hef"].read_bytes() == original_hef
    assert data["original_receipt"].read_bytes() == original_receipt


def test_import_without_original_hint_keeps_h10_absent_and_normal_unknown(tmp_path, monkeypatch):
    data = _prepared(tmp_path)
    for role in ("part1_runtime", "part1_build_receipt"):
        data["config"]["expected_runtime_artifacts"][role].pop("local_path")
        data["config"]["generic_artifact_binding"]["artifacts"][role].pop("local_path")
    _seal_config(data)
    calls = _verifier(data, monkeypatch)
    result = _import(data, targets=["hailo10", "tensorrt"])
    assert not list(result.suite_dir.glob("*/hailo/hailo10/part1/compiled.hef"))
    assert not list(result.suite_dir.rglob("generic_quality_original_hailo_reuse.json"))
    assert not list(result.suite_dir.rglob("deferred_hailo_build.json"))
    service, rows, roles = _ordinary_observations(data, result)
    assert service["build_attempt_count"] == 0
    assert service["case_hef_requests"][0]["ready"] is False
    assert [(row.item_id, row.status) for row in rows] == [("b001:part1", "UNKNOWN")]
    report = build_artifact_cache_preflight(model_ids=["model"], observations=rows,
        applicable_roles={"model": roles}, block_on_unexpected_cold_builds=True)
    assert report["runtime_dispatch_allowed"] is False
    assert calls == []
