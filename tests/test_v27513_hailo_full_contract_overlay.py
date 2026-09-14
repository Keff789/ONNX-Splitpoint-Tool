from __future__ import annotations

import json
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest

from onnx_splitpoint_tool.hailo_backend import (
    _hailo_cache_key,
    _write_hailo_receipt,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    _demote_hailo_full_contract_claim,
    _promote_verified_hailo_full_contracts,
    materialize_backend_artifact_decisions,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)
from scripts import native_full_baseline_eval_runner as full_runner
from scripts import smoke_hailo10_hef_runner as hef_runner


_YOLO26_END_NODES = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv",
]


def _make_yolo26_full_suite(
    tmp_path: Path, *, hw_arch: str,
) -> tuple[Path, Path, Path]:
    """Write one production-shaped, receipt-sealed YOLO26 Full suite."""

    benchmark_set = tmp_path / "benchmark_set"
    backend_key = "hailo8" if hw_arch == "hailo8" else "hailo10"
    source = benchmark_set / "models" / "yolo26s.onnx"
    hef = (
        benchmark_set / "hailo" / backend_key / "full" / "compiled.hef"
    )
    compiler = hef.parent / "yolo26s_hailo_fixed.onnx"
    source.parent.mkdir(parents=True)
    hef.parent.mkdir(parents=True)
    source.write_bytes(b"receipt-bound-yolo26-source-onnx")
    compiler.write_bytes(b"receipt-bound-yolo26-compiler-onnx")
    hef.write_bytes(f"receipt-bound-yolo26-{hw_arch}-hef".encode("utf-8"))

    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch=hw_arch,
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        effective_calib_count=54,
        calibration_storage="memory",
        calibration_memory_cap_bytes=256 * 1024 * 1024,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=_YOLO26_END_NODES,
        preprocessing_contract=preprocessing,
    )
    _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch=hw_arch,
        net_name="yolo26s",
        preprocessing_contract=preprocessing,
        preprocessing_sha256=preprocessing_contract_sha256(preprocessing),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )

    benchmark_payload = {
        "model": source.relative_to(benchmark_set).as_posix(),
        "model_source": str(source),
        "benchmark_task": "detection",
        "hailo": {"hefs": {backend_key: {
            "full": hef.relative_to(benchmark_set).as_posix(),
            "full_build": {
                "ok": True,
                "artifact_hash": full_runner._sha256_file(hef),
                "source_onnx_path": str(source),
                "compiler_onnx_path": str(compiler),
            },
            "full_endpoint_mode": "raw_detection_head",
            "full_end_node_names": list(_YOLO26_END_NODES),
            "full_output_contract": {
                "mode": "yolo26_one2one_raw_head",
                "requires_external_postprocess": True,
                "end_node_names": list(_YOLO26_END_NODES),
            },
        }}},
    }
    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps(benchmark_payload), encoding="utf-8",
    )

    # Mirror the observed stale workflow declaration: Hailo-8 was planned as
    # raw without its compiler nodes, while Hailo-10H was planned decoded.
    if hw_arch == "hailo8":
        provisional_endpoint = {
            "endpoint_mode": "raw_detection_head",
            "host_tail_required": True,
            "postprocessing_required": True,
            "requires_external_postprocess": True,
            "full_end_node_names": [],
        }
    else:
        provisional_endpoint = {
            "endpoint_mode": "decoded",
            "host_tail_required": False,
            "postprocessing_required": False,
            "requires_external_postprocess": False,
            "full_end_node_names": [],
            "output_format": "bn6_detections",
            "output_record_format": "xyxy_score_class",
        }
    output_contracts = {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "yolo26s",
        "task": "detection",
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "schema_version": 1,
            "model_id": "yolo26s",
            "task": "detection",
            "backend": backend_key,
            "variant": "full",
            "contract_status": "pending_build_or_prepare",
            "artifact_binding_status": "pending_receipt_validation",
            **provisional_endpoint,
        }],
    }
    (benchmark_set / "output_contracts.json").write_text(
        json.dumps(output_contracts), encoding="utf-8",
    )
    return benchmark_set, hef, hef.parent / "hailo_hef_build_receipt.json"


def _make_resnet50_classification_full_suite(
    tmp_path: Path, *, hw_arch: str,
) -> tuple[Path, Path, Path]:
    """Write the observed pending ResNet contract plus an exact receipt."""

    benchmark_set = tmp_path / "benchmark_set"
    backend_key = "hailo8" if hw_arch == "hailo8" else "hailo10"
    source = benchmark_set / "models" / "resnet50.onnx"
    hef = (
        benchmark_set / "hailo" / backend_key / "full" / "compiled.hef"
    )
    compiler = hef.parent / "resnet50_hailo_fixed.onnx"
    source.parent.mkdir(parents=True)
    hef.parent.mkdir(parents=True)
    source.write_bytes(b"receipt-bound-resnet50-source-onnx")
    compiler.write_bytes(b"receipt-bound-resnet50-compiler-onnx")
    hef.write_bytes(f"receipt-bound-resnet50-{hw_arch}-hef".encode("utf-8"))

    preprocessing = canonical_image_preprocessing_contract(
        "classification", (224, 224),
    )
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch=hw_arch,
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        effective_calib_count=54,
        calibration_storage="memory",
        calibration_memory_cap_bytes=256 * 1024 * 1024,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=[],
        preprocessing_contract=preprocessing,
    )
    _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch=hw_arch,
        net_name="resnet50",
        preprocessing_contract=preprocessing,
        preprocessing_sha256=preprocessing_contract_sha256(preprocessing),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )

    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({
            "model": source.relative_to(benchmark_set).as_posix(),
            "model_source": str(source),
            "benchmark_task": "classification",
            "hailo": {"hefs": {backend_key: {
                "full": hef.relative_to(benchmark_set).as_posix(),
                "full_build": {
                    "ok": True,
                    "artifact_hash": full_runner._sha256_file(hef),
                    "source_onnx_path": str(source),
                    "compiler_onnx_path": str(compiler),
                },
                "full_endpoint_mode": "decoded",
                "full_end_node_names": [],
                "full_output_contract": {
                    "mode": "classification_logits",
                    "requires_external_postprocess": False,
                    "end_node_names": [],
                },
            }}},
        }),
        encoding="utf-8",
    )
    (benchmark_set / "output_contracts.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/output-contracts",
            "schema_version": 1,
            "model_id": "resnet50",
            "task": "classification",
            "contracts": [{
                "schema": "onnx-splitpoint/output-contract",
                "schema_version": 1,
                "model_id": "resnet50",
                "task": "classification",
                "backend": backend_key,
                "variant": "full",
                "contract_status": "pending_build_or_prepare",
                "artifact_binding_status": "pending_receipt_validation",
                "artifact_binding_error": "hailo_full_receipt_not_verified",
                "endpoint_mode": "decoded",
                "host_tail_required": False,
                "postprocessing_required": False,
                "full_end_node_names": [],
                "source_onnx_multiscale_raw_head": False,
            }],
        }),
        encoding="utf-8",
    )
    return benchmark_set, hef, hef.parent / "hailo_hef_build_receipt.json"


def _native_namespace(tmp_path: Path, *, hw_arch: str) -> Namespace:
    return Namespace(
        frames=10,
        warmup=2,
        inflight=2,
        timeout=60,
        duration_s=0.0,
        setup_id=f"setup_{hw_arch}",
        comparison_backend=hw_arch,
        image_map_data={},
        preprocess_mode="auto",
        letterbox_pad_value=114,
        dump_outputs=False,
        out_dir=str(tmp_path / "native-output"),
    )


def _make_yolov7_source_raw_full_suite(
    tmp_path: Path, *, hw_arch: str,
) -> tuple[Path, Path]:
    """Write the receipt-attested, empty-cut YOLOv7 Full contract."""

    benchmark_set = tmp_path / "benchmark_set"
    backend_key = "hailo8" if hw_arch == "hailo8" else "hailo10"
    source = benchmark_set / "models" / "yolov7_paper.onnx"
    hef = (
        benchmark_set / "hailo" / backend_key / "full" / "compiled.hef"
    )
    compiler = hef.parent / "yolov7_paper_hailo_fixed.onnx"
    source.parent.mkdir(parents=True)
    hef.parent.mkdir(parents=True)
    graph_outputs = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, shape,
        )
        for name, shape in (
            ("head_80", [1, 3, 80, 80, 85]),
            ("head_40", [1, 3, 40, 40, 85]),
            ("head_20", [1, 3, 20, 20, 85]),
        )
    ]
    source.write_bytes(
        onnx.helper.make_model(
            onnx.helper.make_graph(
                [], "yolov7_source_raw_heads", graph_outputs, graph_outputs,
            )
        ).SerializeToString()
    )
    compiler.write_bytes(b"receipt-bound-yolov7-compiler-onnx")
    hef.write_bytes(f"receipt-bound-yolov7-{hw_arch}-hef".encode("utf-8"))

    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler,
        activation_part1=None,
        hw_arch=hw_arch,
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        effective_calib_count=54,
        calibration_storage="memory",
        calibration_memory_cap_bytes=256 * 1024 * 1024,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=[],
        preprocessing_contract=preprocessing,
    )
    _write_hailo_receipt(
        hef_path=hef,
        source_onnx=source,
        compiler_onnx=compiler,
        hw_arch=hw_arch,
        net_name="yolov7_paper",
        preprocessing_contract=preprocessing,
        preprocessing_sha256=preprocessing_contract_sha256(preprocessing),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )

    (benchmark_set / "benchmark_set.json").write_text(
        json.dumps({
            "model": source.relative_to(benchmark_set).as_posix(),
            "model_source": str(source),
            "benchmark_task": "detection",
            "hailo": {"hefs": {backend_key: {
                "full": hef.relative_to(benchmark_set).as_posix(),
                "full_build": {
                    "ok": True,
                    "artifact_hash": full_runner._sha256_file(hef),
                    "source_onnx_path": str(source),
                    "compiler_onnx_path": str(compiler),
                },
                "full_output_contract": {
                    "endpoint_mode": "raw_detection_head",
                    "requires_external_postprocess": True,
                    "postprocessing_required": True,
                    "host_tail_required": True,
                    "full_end_node_names": [],
                    "source_onnx_multiscale_raw_head": True,
                },
            }}},
        }),
        encoding="utf-8",
    )
    (benchmark_set / "output_contracts.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/output-contracts",
            "schema_version": 1,
            "model_id": "yolov7_paper",
            "task": "detection",
            "contracts": [{
                "schema": "onnx-splitpoint/output-contract",
                "schema_version": 1,
                "model_id": "yolov7_paper",
                "task": "detection",
                "backend": backend_key,
                "variant": "full",
                "contract_status": "pending_build_or_prepare",
                "artifact_binding_status": "pending_receipt_validation",
                "endpoint_mode": "raw_detection_head",
                "host_tail_required": True,
                "postprocessing_required": True,
                "requires_external_postprocess": True,
                "full_end_node_names": [],
                "source_onnx_multiscale_raw_head": True,
            }],
        }),
        encoding="utf-8",
    )
    return benchmark_set, hef


def _promote_persisted_contract(
    benchmark_set: Path, *, model: str, task: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    benchmark_path = benchmark_set / "benchmark_set.json"
    contracts_path = benchmark_set / "output_contracts.json"
    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    contracts_payload = json.loads(contracts_path.read_text(encoding="utf-8"))
    contracts = [
        _demote_hailo_full_contract_claim(dict(row))
        for row in contracts_payload["contracts"]
    ]
    promotions = _promote_verified_hailo_full_contracts(
        suite_dir=benchmark_set,
        model_id=model,
        task=task,
        suite_bench=benchmark_payload,
        contracts=contracts,
        copied_verified={},
    )
    assert len(promotions) == 1
    contracts_payload["contracts"] = contracts
    contracts_path.write_text(
        json.dumps(contracts_payload), encoding="utf-8",
    )
    backend_key = next(iter(benchmark_payload["hailo"]["hefs"]))
    benchmark_payload["hailo"]["hefs"][backend_key][
        "full_output_contract"
    ] = dict(contracts[0])
    benchmark_path.write_text(
        json.dumps(benchmark_payload), encoding="utf-8",
    )
    return dict(promotions[0]), dict(contracts[0])


@pytest.mark.parametrize("hw_arch", ["hailo8", "hailo10h"])
@pytest.mark.parametrize(
    "model,task,factory",
    [
        ("yolo26s", "detection", _make_yolo26_full_suite),
        (
            "yolov7_paper", "detection",
            _make_yolov7_source_raw_full_suite,
        ),
    ],
)
def test_hailo_raw_contract_promotion_is_idempotent_after_persistence(
    tmp_path: Path, hw_arch: str, model: str, task: str, factory: Any,
) -> None:
    benchmark_set, _hef, *_rest = factory(tmp_path, hw_arch=hw_arch)
    observed = [
        _promote_persisted_contract(
            benchmark_set, model=model, task=task,
        )
        for _ in range(3)
    ]
    promotions = [row[0] for row in observed]
    contracts = [row[1] for row in observed]
    assert len({row["artifact_binding_sha256"] for row in promotions}) == 1
    assert all(
        row["contract_reconciliation_status"]
        == "verified_suite_artifact_raw_head"
        for row in contracts
    )
    expected_nodes = _YOLO26_END_NODES if model == "yolo26s" else []
    assert all(row["full_end_node_names"] == expected_nodes for row in contracts)
    assert all(row["stage"] == "raw_head" for row in contracts)
    assert all(row["contract_family"] == "raw_head" for row in contracts)


@pytest.mark.parametrize("hw_arch", ["hailo8", "hailo10h"])
def test_sparse_yolov7_materialization_normalizes_and_stays_attested(
    tmp_path: Path, hw_arch: str,
) -> None:
    benchmark_set, hef = _make_yolov7_source_raw_full_suite(
        tmp_path / "source", hw_arch=hw_arch,
    )
    contracts_path = benchmark_set / "output_contracts.json"
    plan_contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
    benchmark_path = benchmark_set / "benchmark_set.json"
    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    backend_key = "hailo8" if hw_arch == "hailo8" else "hailo10"
    full_meta = benchmark_payload["hailo"]["hefs"][backend_key]
    for key in (
        "full_output_contract", "full_endpoint_mode", "full_end_node_names",
    ):
        full_meta.pop(key, None)
    benchmark_path.write_text(json.dumps(benchmark_payload), encoding="utf-8")

    run_dir = tmp_path / "run"
    observed_contracts: list[dict[str, Any]] = []
    observed_promotions: list[dict[str, Any]] = []
    for _pass in range(3):
        result = materialize_backend_artifact_decisions(
            run_dir=run_dir,
            model_id="yolov7_paper",
            targets=[hw_arch],
            full_baseline_plan={"task": "detection", "baselines": []},
            output_contracts=deepcopy(plan_contracts),
            benchmark_set_contract={
                "legacy_suite_dir": str(benchmark_set),
            },
        )
        assert result["metrics"]["recorded_hailo_full_contracts"] == 1
        decision_payload = json.loads(
            (
                run_dir
                / "models/yolov7_paper/benchmark_set/"
                "backend_artifact_decisions.json"
            ).read_text(encoding="utf-8")
        )
        assert decision_payload["hailo_full_requested"] is True
        promotions = decision_payload["recorded_hailo_full_contracts"]
        assert len(promotions) == 1
        observed_promotions.append(dict(promotions[0]))
        persisted = json.loads(
            contracts_path.read_text(encoding="utf-8")
        )["contracts"]
        assert len(persisted) == 1
        observed_contracts.append(dict(persisted[0]))

    assert observed_contracts[0]["requires_external_postprocess"] is True
    for contract in observed_contracts[1:]:
        assert contract["stage"] == "raw_head"
        assert contract["contract_family"] == "raw_head"
        assert contract["raw_endpoint_origin"] == (
            "source_onnx_graph_outputs"
        )
        assert contract["full_end_node_names"] == []
        assert contract["source_onnx_multiscale_raw_head"] is True
        assert contract["endpoint_mode"] == "raw_detection_head"
        assert "mode" not in contract
        assert "end_node_names" not in contract
    assert observed_promotions[1]["artifact_binding_sha256"] == (
        observed_promotions[2]["artifact_binding_sha256"]
    )
    for key in ("source_onnx_sha256", "compiler_onnx_sha256"):
        assert observed_contracts[1][key] == observed_contracts[2][key]
    assert observed_contracts[1]["source_onnx_sha256"] != (
        observed_contracts[1]["compiler_onnx_sha256"]
    )

    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch=hw_arch,
    )
    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    _overlay, evidence = (
        full_runner._prepare_verified_hailo_full_contract_overlay(
            benchmark_set,
            dump_dir=tmp_path / f"{hw_arch}-sparse-overlay",
            model="yolov7_paper",
            task="detection",
            hw_arch=hw_arch,
            hef=hef,
            verified_receipt=verified,
        )
    )
    assert evidence["resolution_status"] == "attested"

    unrequested_run = tmp_path / "unrequested-run"
    unrequested = materialize_backend_artifact_decisions(
        run_dir=unrequested_run,
        model_id="yolov7_paper",
        targets=[hw_arch],
        full_baseline_plan={
            "task": "detection",
            "baselines": [{
                "backend": hw_arch,
                "variant": "full",
                "requested": False,
            }],
        },
        # Deliberately omit the marker from the contract. The baseline's
        # explicit False must remain authoritative through promotion.
        output_contracts=deepcopy(plan_contracts),
        benchmark_set_contract={
            "legacy_suite_dir": str(benchmark_set),
        },
    )
    assert unrequested["metrics"]["recorded_hailo_full_contracts"] == 0
    unrequested_payload = json.loads(
        unrequested["artifacts"]["backend_artifact_decisions_json"].read_text(
            encoding="utf-8",
        )
    )
    assert unrequested_payload["hailo_full_requested"] is False
    assert unrequested_payload["baseline_decisions"][0]["decision"] == (
        "not_requested_by_profile"
    )
    persisted_unrequested = json.loads(
        contracts_path.read_text(encoding="utf-8")
    )["contracts"]
    assert persisted_unrequested[0]["requested"] is False
    assert persisted_unrequested[0]["request_status"] == (
        "not_requested_by_profile"
    )


def test_hailo_promotion_canonicalizes_consistent_legacy_row_aliases(
    tmp_path: Path,
) -> None:
    benchmark_set, _hef, _receipt = _make_yolo26_full_suite(
        tmp_path, hw_arch="hailo8",
    )
    contracts_path = benchmark_set / "output_contracts.json"
    payload = json.loads(contracts_path.read_text(encoding="utf-8"))
    row = payload["contracts"][0]
    row["mode"] = "yolo26_one2one_raw_head"
    row["end_node_names"] = list(_YOLO26_END_NODES)
    row["full_end_node_names"] = list(_YOLO26_END_NODES)
    contracts_path.write_text(json.dumps(payload), encoding="utf-8")

    observed = [
        _promote_persisted_contract(
            benchmark_set, model="yolo26s", task="detection",
        )
        for _ in range(3)
    ]
    assert all("mode" not in contract for _promotion, contract in observed)
    assert all(
        "end_node_names" not in contract
        for _promotion, contract in observed
    )
    assert len({
        promotion["artifact_binding_sha256"]
        for promotion, _contract in observed
    }) == 1


@pytest.mark.parametrize("conflict", ["mode", "nodes"])
def test_hailo_normalized_alias_conflict_stops_before_overlay(
    tmp_path: Path, conflict: str,
) -> None:
    benchmark_set, hef, _receipt = _make_yolo26_full_suite(
        tmp_path, hw_arch="hailo8",
    )
    _promotion, _contract = _promote_persisted_contract(
        benchmark_set, model="yolo26s", task="detection",
    )
    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef, benchmark_set=benchmark_set,
        model="yolo26s", hw_arch="hailo8",
    )
    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    benchmark_path = benchmark_set / "benchmark_set.json"
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    nested = payload["hailo"]["hefs"]["hailo8"]["full_output_contract"]
    if conflict == "mode":
        nested["mode"] = "decoded"
    else:
        nested["end_node_names"] = list(reversed(_YOLO26_END_NODES))
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError, match="hailo_full_overlay_exact_promotion_required",
    ):
        full_runner._prepare_verified_hailo_full_contract_overlay(
            benchmark_set,
            dump_dir=tmp_path / "conflict-overlay",
            model="yolo26s",
            task="detection",
            hw_arch="hailo8",
            hef=hef,
            verified_receipt=verified,
        )


@pytest.mark.parametrize("hw_arch", ["hailo8", "hailo10h"])
def test_receipt_bound_yolo26_overlay_is_attested_and_passed_to_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hw_arch: str,
) -> None:
    benchmark_set, hef, _receipt_path = _make_yolo26_full_suite(
        tmp_path, hw_arch=hw_arch,
    )
    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolo26s",
        hw_arch=hw_arch,
    )
    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    assert verified["compiler_end_nodes"] == _YOLO26_END_NODES

    overlay_path, evidence = (
        full_runner._prepare_verified_hailo_full_contract_overlay(
            benchmark_set,
            dump_dir=tmp_path / "direct-overlay",
            model="yolo26s",
            task="detection",
            hw_arch=hw_arch,
            hef=hef,
            verified_receipt=verified,
        )
    )
    declaration = load_authoritative_output_contract(
        overlay_path,
        backend=hw_arch,
        model_id="yolo26s",
        variant="full",
        task="detection",
    )
    assert evidence["resolution_status"] == "attested"
    assert evidence["full_end_node_names"] == _YOLO26_END_NODES
    assert declaration["contract_resolution_status"] == "attested"
    assert declaration["authoritative_output_contract"] is True
    assert declaration["stage"] == "raw_head"
    assert declaration["contract_family"] == "raw_head"
    assert declaration["full_end_node_names"] == _YOLO26_END_NODES
    assert Path(declaration["recorded_artifact_path"]).resolve() == hef.resolve()

    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        full_runner,
        "_select_hailo_python",
        lambda _arch: ("/test/hailo-python", {"selected": "/test/hailo-python"}),
    )

    def fake_run(command: list[str], **_kwargs: Any) -> dict[str, Any]:
        captured["command"] = [str(value) for value in command]
        return {
            "rc": 1,
            "returncode": 1,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "intentional post-command test stop",
        }

    monkeypatch.setattr(full_runner, "_run", fake_run)
    row = full_runner._native_hailo_full(
        benchmark_set,
        "yolo26s",
        hw_arch,
        _native_namespace(tmp_path, hw_arch=hw_arch),
    )
    assert row["failure_reason"] == "hailo_full_runner_failed"
    command = captured["command"]
    option_index = command.index("--declared-output-contract-json")
    child_overlay = Path(command[option_index + 1])
    assert child_overlay.is_file()
    child_declaration = load_authoritative_output_contract(
        child_overlay,
        backend=hw_arch,
        model_id="yolo26s",
        variant="full",
        task="detection",
    )
    assert child_declaration["contract_resolution_status"] == "attested"
    assert child_declaration["stage"] == "raw_head"
    assert child_declaration["full_end_node_names"] == _YOLO26_END_NODES


def test_diagnostic_fields_preserve_absent_execution_receipt() -> None:
    missing = full_runner._diagnostic_fields({})
    assert missing["returncode"] is None
    assert missing["timed_out"] is None

    complete = full_runner._diagnostic_fields({
        "rc": 0,
        "returncode": 9,
        "timed_out": False,
    })
    assert complete["returncode"] == 0
    assert complete["timed_out"] is False


@pytest.mark.parametrize("hw_arch", ["hailo8", "hailo10h"])
def test_receipt_bound_resnet_overlay_is_attested_and_passed_to_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hw_arch: str,
) -> None:
    benchmark_set, hef, _receipt_path = (
        _make_resnet50_classification_full_suite(
            tmp_path, hw_arch=hw_arch,
        )
    )
    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="resnet50",
        hw_arch=hw_arch,
    )
    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    assert verified["compiler_end_nodes"] == []
    assert verified["raw_endpoint_origin"] == "not_applicable"
    assert verified["source_onnx_raw_head_attestation"] == {}

    overlay_path, evidence = (
        full_runner._prepare_verified_hailo_full_contract_overlay(
            benchmark_set,
            dump_dir=tmp_path / "direct-resnet-overlay",
            model="resnet50",
            task="classification",
            hw_arch=hw_arch,
            hef=hef,
            verified_receipt=verified,
        )
    )
    declaration = load_authoritative_output_contract(
        overlay_path,
        backend=hw_arch,
        model_id="resnet50",
        variant="full",
        task="classification",
    )
    assert evidence["resolution_status"] == "attested"
    assert evidence["task"] == "classification"
    assert evidence["stage"] == "classification_logits"
    assert evidence["raw_endpoint_origin"] == "not_applicable"
    assert declaration["contract_resolution_status"] == "attested"
    assert declaration["authoritative_output_contract"] is True
    assert declaration["stage"] == "classification_logits"
    assert declaration["contract_family"] == "classification_logits"
    assert declaration["host_tail_required"] is False
    assert declaration["postprocessing_required"] is False
    assert Path(declaration["recorded_artifact_path"]).resolve() == hef.resolve()

    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        full_runner,
        "_select_hailo_python",
        lambda _arch: (
            "/test/hailo-python", {"selected": "/test/hailo-python"},
        ),
    )

    def incomplete_run(
        command: list[str], **_kwargs: Any,
    ) -> dict[str, Any]:
        captured["command"] = [str(value) for value in command]
        return {}

    monkeypatch.setattr(full_runner, "_run", incomplete_run)
    row = full_runner._native_hailo_full(
        benchmark_set,
        "resnet50",
        hw_arch,
        _native_namespace(tmp_path, hw_arch=hw_arch),
    )
    assert row["failure_reason"] == "hailo_full_runner_failed"
    assert row["returncode"] is None
    assert row["timed_out"] is None
    command = captured["command"]
    option_index = command.index("--declared-output-contract-json")
    child_declaration = load_authoritative_output_contract(
        Path(command[option_index + 1]),
        backend=hw_arch,
        model_id="resnet50",
        variant="full",
        task="classification",
    )
    assert child_declaration["contract_resolution_status"] == "attested"
    assert child_declaration["stage"] == "classification_logits"
    assert Path(
        child_declaration["recorded_artifact_path"]
    ).resolve() == hef.resolve()


@pytest.mark.parametrize("hw_arch", ["hailo8", "hailo10h"])
@pytest.mark.parametrize("manipulation", ["hef", "receipt", "nodes"])
def test_hailo_full_contract_manipulation_stops_before_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hw_arch: str,
    manipulation: str,
) -> None:
    benchmark_set, hef, receipt_path = _make_yolo26_full_suite(
        tmp_path, hw_arch=hw_arch,
    )
    if manipulation == "hef":
        hef.write_bytes(hef.read_bytes() + b"-tampered")
    elif manipulation == "receipt":
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["cache_key"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    else:
        tampered_nodes = [f"/tampered/head/{index}" for index in range(6)]
        benchmark_path = benchmark_set / "benchmark_set.json"
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        backend_key = "hailo8" if hw_arch == "hailo8" else "hailo10"
        meta = benchmark["hailo"]["hefs"][backend_key]
        meta["full_end_node_names"] = list(tampered_nodes)
        meta["full_output_contract"]["end_node_names"] = list(tampered_nodes)
        benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
        contracts_path = benchmark_set / "output_contracts.json"
        contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
        contracts["contracts"][0]["full_end_node_names"] = list(
            tampered_nodes
        )
        contracts_path.write_text(json.dumps(contracts), encoding="utf-8")

    child_calls: list[list[str]] = []
    monkeypatch.setattr(
        full_runner,
        "_select_hailo_python",
        lambda _arch: ("/test/hailo-python", {"selected": "/test/hailo-python"}),
    )

    def forbidden_child(command: list[str], **_kwargs: Any) -> dict[str, Any]:
        child_calls.append([str(value) for value in command])
        raise AssertionError("tampered Hailo contract reached the child")

    monkeypatch.setattr(full_runner, "_run", forbidden_child)
    row = full_runner._native_hailo_full(
        benchmark_set,
        "yolo26s",
        hw_arch,
        _native_namespace(tmp_path, hw_arch=hw_arch),
    )
    assert row["ok"] is False
    assert row["returncode"] == 4
    assert row["steps"] == []
    assert row["failure_reason"] != "hailo_full_runner_failed"
    assert child_calls == []


@pytest.mark.parametrize("unsafe_kind", ["symlink", "directory"])
def test_existing_unsafe_overlay_output_stops_before_child_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    benchmark_set, _hef, _receipt_path = _make_yolo26_full_suite(
        tmp_path, hw_arch="hailo8",
    )
    ns = _native_namespace(tmp_path, hw_arch="hailo8")
    overlay_path = (
        full_runner._native_full_dump_dir(
            benchmark_set, "yolo26s", "native_full_hailo8", ns,
        )
        / "contract_overlay"
        / "output_contracts.json"
    )
    overlay_path.parent.mkdir(parents=True)
    sentinel = b"must-not-be-overwritten"
    if unsafe_kind == "symlink":
        external_target = tmp_path / "external-overlay-target.json"
        external_target.write_bytes(sentinel)
        overlay_path.symlink_to(external_target)
    else:
        overlay_path.mkdir()
        directory_sentinel = overlay_path / "sentinel.bin"
        directory_sentinel.write_bytes(sentinel)

    child_calls: list[list[str]] = []
    monkeypatch.setattr(
        full_runner,
        "_select_hailo_python",
        lambda _arch: (
            "/test/hailo-python", {"selected": "/test/hailo-python"},
        ),
    )

    def forbidden_child(command: list[str], **_kwargs: Any) -> dict[str, Any]:
        child_calls.append([str(value) for value in command])
        raise AssertionError("unsafe overlay output reached the child")

    monkeypatch.setattr(full_runner, "_run", forbidden_child)
    row = full_runner._native_hailo_full(
        benchmark_set, "yolo26s", "hailo8", ns,
    )

    assert row["ok"] is False
    assert row["returncode"] == 4
    assert row["steps"] == []
    assert row["failure_reason"] == (
        "hailo_full_output_contract_overlay_invalid"
    )
    expected_error = (
        "hailo_full_overlay_output_path_is_symlink"
        if unsafe_kind == "symlink"
        else "hailo_full_overlay_output_path_not_regular"
    )
    assert expected_error in row["status_detail"]
    assert child_calls == []
    if unsafe_kind == "symlink":
        assert overlay_path.is_symlink()
        assert external_target.read_bytes() == sentinel
    else:
        assert overlay_path.is_dir()
        assert directory_sentinel.read_bytes() == sentinel


def test_six_valid_yolo26_hwc_tensors_are_not_endpoint_authority() -> None:
    outputs = {
        "conv61": np.zeros((80, 80, 4), dtype=np.float32),
        "conv64": np.zeros((80, 80, 80), dtype=np.float32),
        "conv77": np.zeros((40, 40, 4), dtype=np.float32),
        "conv80": np.zeros((40, 40, 80), dtype=np.float32),
        "conv91": np.zeros((20, 20, 4), dtype=np.float32),
        "conv94": np.zeros((20, 20, 80), dtype=np.float32),
    }
    contract = hef_runner._output_contract(
        "detection", outputs, declared_contract={},
    )
    assert contract["contract_family"] == "unknown"
    assert contract["stage"] == "unknown"
    assert contract["endpoint_contract_complete"] is False
    assert contract["endpoint_contract_hash"] == ""
    assert contract["claim_eligible_e2e"] is False


@pytest.mark.parametrize("hw_arch", ["hailo8", "hailo10h"])
def test_receipt_bound_yolov7_source_graph_overlay_remains_attested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hw_arch: str,
) -> None:
    benchmark_set, hef = _make_yolov7_source_raw_full_suite(
        tmp_path, hw_arch=hw_arch,
    )
    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch=hw_arch,
    )
    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    assert verified["compiler_end_nodes"] == []
    assert verified["raw_endpoint_origin"] == "source_onnx_graph_outputs"
    source_attestation = verified["source_onnx_raw_head_attestation"]
    source_attestation_sha256 = verified[
        "source_onnx_raw_head_attestation_sha256"
    ]
    assert len(source_attestation_sha256) == 64

    overlay_path, evidence = (
        full_runner._prepare_verified_hailo_full_contract_overlay(
            benchmark_set,
            dump_dir=tmp_path / "direct-yolov7-overlay",
            model="yolov7_paper",
            task="detection",
            hw_arch=hw_arch,
            hef=hef,
            verified_receipt=verified,
        )
    )
    declaration = load_authoritative_output_contract(
        overlay_path,
        backend=hw_arch,
        model_id="yolov7_paper",
        variant="full",
        task="detection",
    )
    assert evidence["resolution_status"] == "attested"
    assert evidence["raw_endpoint_origin"] == "source_onnx_graph_outputs"
    assert evidence["full_end_node_names"] == []
    assert evidence[
        "source_onnx_raw_head_attestation_sha256"
    ] == source_attestation_sha256
    assert declaration["contract_resolution_status"] == "attested"
    assert declaration["stage"] == "raw_head"
    assert declaration["full_end_node_names"] == []
    assert declaration["raw_endpoint_origin"] == "source_onnx_graph_outputs"
    assert declaration["source_onnx_multiscale_raw_head"] is True
    assert declaration[
        "source_onnx_raw_head_attestation"
    ] == source_attestation
    assert declaration[
        "source_onnx_raw_head_attestation_sha256"
    ] == source_attestation_sha256

    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(
        full_runner,
        "_select_hailo_python",
        lambda _arch: ("/test/hailo-python", {"selected": "/test/hailo-python"}),
    )

    def fake_run(command: list[str], **_kwargs: Any) -> dict[str, Any]:
        captured["command"] = [str(value) for value in command]
        return {
            "rc": 1,
            "returncode": 1,
            "timed_out": False,
            "stdout_tail": "",
            "stderr_tail": "intentional post-command test stop",
        }

    monkeypatch.setattr(full_runner, "_run", fake_run)
    row = full_runner._native_hailo_full(
        benchmark_set,
        "yolov7_paper",
        hw_arch,
        _native_namespace(tmp_path, hw_arch=hw_arch),
    )
    assert row["failure_reason"] == "hailo_full_runner_failed"
    command = captured["command"]
    option_index = command.index("--declared-output-contract-json")
    child_declaration = load_authoritative_output_contract(
        Path(command[option_index + 1]),
        backend=hw_arch,
        model_id="yolov7_paper",
        variant="full",
        task="detection",
    )
    assert child_declaration["contract_resolution_status"] == "attested"
    assert child_declaration["raw_endpoint_origin"] == (
        "source_onnx_graph_outputs"
    )


@pytest.mark.parametrize("hw_arch", ["hailo8", "hailo10h"])
def test_yolov7_source_graph_overlay_rejects_attestation_sha_tamper(
    tmp_path: Path, hw_arch: str,
) -> None:
    benchmark_set, hef = _make_yolov7_source_raw_full_suite(
        tmp_path, hw_arch=hw_arch,
    )
    verified, status = full_runner._verified_hailo_full_build_receipt(
        hef=hef,
        benchmark_set=benchmark_set,
        model="yolov7_paper",
        hw_arch=hw_arch,
    )
    assert status == "hailo_hef_build_receipt_verified_exact"
    assert verified is not None
    tampered = deepcopy(verified)
    tampered["source_onnx_raw_head_attestation_sha256"] = "0" * 64

    with pytest.raises(
        ValueError, match="hailo_full_overlay_receipt_binding_mismatch",
    ):
        full_runner._prepare_verified_hailo_full_contract_overlay(
            benchmark_set,
            dump_dir=tmp_path / "tampered-yolov7-overlay",
            model="yolov7_paper",
            task="detection",
            hw_arch=hw_arch,
            hef=hef,
            verified_receipt=tampered,
        )
