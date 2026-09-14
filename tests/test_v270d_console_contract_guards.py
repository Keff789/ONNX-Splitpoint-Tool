from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256,
)
from onnx_splitpoint_tool.hailo_backend import (
    _hailo_cache_key,
    _write_hailo_receipt,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_frozen_postprocess_contract,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v270d_console_guard_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_test_module(name: str):
    path = ROOT / "tests" / name
    spec = importlib.util.spec_from_file_location(
        f"v270d_console_fixture_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _old_decoded_hailo10_suite(
    tmp_path: Path,
) -> tuple[Path, Path, bytes]:
    suite = tmp_path / "suite"
    artifact = suite / "hailo/hailo10/full/compiled.hef"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"verified-yolo26-hailo10-raw-head")
    artifact_sha = __import__("hashlib").sha256(
        artifact.read_bytes(),
    ).hexdigest()
    nodes = [f"/head/{index}" for index in range(6)]
    source_onnx = artifact.parent / "receipt_source.onnx"
    compiler_onnx = artifact.parent / "receipt_compiler.onnx"
    source_onnx.write_bytes(b"verified-yolo26-source-onnx")
    compiler_onnx.write_bytes(b"verified-yolo26-compiler-onnx")
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640),
    )
    cache_key, cache_payload = _hailo_cache_key(
        model_path=compiler_onnx,
        activation_part1=None,
        hw_arch="hailo10h",
        opt_level=1,
        calib_dir=None,
        calib_count=64,
        calib_batch_size=8,
        extra_model_script="",
        start_nodes=None,
        end_nodes=nodes,
        preprocessing_contract=preprocessing,
    )
    _write_hailo_receipt(
        hef_path=artifact,
        source_onnx=source_onnx,
        compiler_onnx=compiler_onnx,
        hw_arch="hailo10h",
        net_name="yolo26s",
        preprocessing_contract=preprocessing,
        preprocessing_sha256=preprocessing_contract_sha256(
            preprocessing
        ),
        cache_key=cache_key,
        cache_payload=cache_payload,
        calibration_identity=str(cache_payload["calibration_identity"]),
        calibration_count=int(cache_payload["calibration_count"]),
    )
    benchmark = {
        "model_id": "yolo26s",
        "benchmark_task": "detection",
        "hailo": {
            "hefs": {
                "hailo10": {
                    "full": "hailo/hailo10/full/compiled.hef",
                    "full_build": {
                        "ok": True,
                        "artifact_hash": artifact_sha,
                        "source_onnx_path": str(source_onnx),
                        "compiler_onnx_path": str(compiler_onnx),
                    },
                    "full_endpoint_mode": "raw_detection_head",
                    "full_end_node_names": nodes,
                    "full_output_contract": {
                        "mode": "yolo26_one2one_raw_head",
                        "requires_external_postprocess": True,
                        "end_node_names": nodes,
                    },
                },
            },
        },
    }
    contracts = {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "yolo26s",
        "task": "detection",
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "model_id": "yolo26s",
            "task": "detection",
            "backend": "hailo10",
            "variant": "full",
            "endpoint_mode": "decoded",
            "host_tail_required": False,
            "postprocessing_required": False,
            "full_end_node_names": [],
        }],
    }
    (suite / "benchmark_set.json").write_text(
        json.dumps(benchmark), encoding="utf-8",
    )
    source_contracts = suite / "output_contracts.json"
    source_contracts.write_text(
        json.dumps(contracts), encoding="utf-8",
    )
    return suite, artifact, source_contracts.read_bytes()


def test_console_inventory_exposes_backend_and_warning_scope() -> None:
    console = _load_script("native_console_smoke.py")
    rows = [
        {
            "backend": "hailo8_to_trt",
            "status": "technical_pass",
            "warnings": [],
        },
        {
            "backend": "native_full_hailo10h",
            "status": "offline_derived_not_claim_evidence",
            "warnings": [
                "declared_endpoint_conflict",
                "offline_derived_not_claim_evidence",
            ],
        },
    ]
    assert console._row_inventory(rows, "backend") == {
        "hailo8_to_trt": 1,
        "native_full_hailo10h": 1,
    }
    assert console._warning_inventory(rows) == {
        "declared_endpoint_conflict": 1,
        "offline_derived_not_claim_evidence": 1,
    }


def test_hailo10_full_overlay_reconciles_old_suite_without_mutation(
    tmp_path: Path,
) -> None:
    console = _load_script("native_console_smoke.py")
    suite, artifact, original_contracts = _old_decoded_hailo10_suite(
        tmp_path,
    )
    overlay, evidence = console._prepare_hailo10_full_contract_overlay(
        suite,
        child_root=tmp_path / "isolated_child",
        model_id="yolo26s",
    )
    assert (suite / "output_contracts.json").read_bytes() == original_contracts
    assert suite.resolve() not in overlay.resolve().parents
    assert evidence["resolution_status"] == "attested"
    assert evidence["stage"] == "raw_head"
    assert len(evidence["full_end_node_names"]) == 6

    declaration = load_authoritative_output_contract(
        overlay,
        backend="hailo10h",
        model_id="yolo26s",
        variant="full",
        task="detection",
    )
    assert declaration["contract_resolution_status"] == "attested"
    assert declaration["contract_reconciliation_status"] == (
        "verified_suite_artifact_raw_head"
    )
    assert Path(declaration["recorded_artifact_path"]) == artifact.resolve()


def test_split_boundary_console_guard_verifies_product_dump_and_hash(
    tmp_path: Path,
) -> None:
    producer = _load_script(
        "native_hailo10_trt_e2e_from_benchmarkset.py",
    )
    console = _load_script("native_console_smoke.py")
    image = tmp_path / "input.jpg"
    image.write_bytes(b"exact-image-identity")

    class TRT:
        inputs = ["cut"]
        shapes = {"cut": (1, 4, 2, 3)}
        dtypes = {"cut": np.dtype(np.float32)}

    manifest = Path(producer._dump_hailo10_boundary(
        tmp_path / "child",
        boundary_name="runtime_cut",
        boundary=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        inputs={"input": np.zeros((8, 8, 3), dtype=np.uint8)},
        trt=TRT(),
        case="b038",
        precision="float32_layout_fp16",
        boundary_layout="memory_nhwc_to_nchw",
        input_image=str(image),
    ))
    producer._seal_manifest_payload_files(manifest)
    _payload, evidence = console._strict_boundary_manifest(
        manifest,
        allowed_root=tmp_path / "child",
        expected_image=image,
    )
    assert evidence["raw_shape"] == [2, 3, 4]
    assert evidence["trt_input_shape"] == [1, 4, 2, 3]
    assert evidence["boundary_layout"] == "memory_nhwc_to_nchw"
    assert evidence["layout_transform_owner"] == (
        "tensorrt_part2_input_bridge"
    )

    boundary_file = Path(
        json.loads(manifest.read_text(encoding="utf-8"))["file"],
    )
    boundary_file.write_bytes(b"x" * boundary_file.stat().st_size)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        console._strict_boundary_manifest(
            manifest,
            allowed_root=tmp_path / "child",
            expected_image=image,
        )


def test_split_console_runs_exact_final_portable_join_and_rejects_resealed_drift(
    tmp_path: Path,
) -> None:
    console = _load_script("native_console_smoke.py")
    fixture = _load_test_module(
        "test_v269f_native_split_final_energy_integrity.py",
    )
    row, output_manifest, _output_payload = fixture._quality_first_row(
        tmp_path,
    )
    evidence = console._verify_final_split_semantic_join(
        row,
        report_path=None,
        output_manifest=output_manifest,
    )
    assert evidence["native_split_semantic_binding_valid"] is True
    assert evidence["native_split_final_portable_binding_valid"] is True

    drifted = copy.deepcopy(row)
    attestation = drifted["native_split_quality_consumer_attestation"]
    attestation.pop("attestation_sha256", None)
    attestation["source_run_id"] = "wrong_backend_after_reseal"
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    with pytest.raises(ValueError, match="final split semantic join rejected"):
        console._verify_final_split_semantic_join(
            drifted,
            report_path=None,
            output_manifest=output_manifest,
        )


def test_isolated_result_root_rejects_eval_tree_before_mkdir(
    tmp_path: Path,
) -> None:
    console = _load_script("native_console_smoke.py")
    eval_root = tmp_path / "archived_eval"
    benchmark_set = eval_root / "yolo26s" / "benchmark_set"
    benchmark_set.mkdir(parents=True)
    for marker in ("models", "native_producers", "reports", "stages"):
        (eval_root / marker).mkdir(exist_ok=True)
    quality_binding = (
        eval_root
        / "quality_first/consumed_bindings/hailo10h/yolo26s/b038.json"
    )
    quality_binding.parent.mkdir(parents=True)
    quality_binding.write_text("{}\n", encoding="utf-8")
    rejected = eval_root / "reports" / "console_attempt"
    assert not rejected.exists()
    with pytest.raises(ValueError, match="must not overlap"):
        console._isolated_result_root(
            rejected,
            "hailo10_split",
            benchmark_set=benchmark_set,
            quality_binding=quality_binding,
        )
    assert not rejected.exists()


def test_authority_result_root_rejects_eval_tree_before_mkdir(
    tmp_path: Path,
) -> None:
    console = _load_script("native_console_smoke.py")
    eval_root = tmp_path / "archived_eval"
    (eval_root / "reports").mkdir(parents=True)
    rejected = eval_root / "reports" / "console_authority"
    assert not rejected.exists()
    with pytest.raises(ValueError, match="must not overlap"):
        console._authority(SimpleNamespace(
            run_dir=eval_root,
            stage=None,
            result_root=rejected,
        ))
    assert not rejected.exists()


def _six_raw_heads() -> dict[str, np.ndarray]:
    return {
        f"head_{index}": np.zeros(
            (1, channels, side, side), dtype=np.float32,
        )
        for index, (channels, side) in enumerate([
            (4, 80),
            (80, 80),
            (4, 40),
            (80, 40),
            (4, 20),
            (80, 20),
        ])
    }


def test_hailo10_full_console_guard_covers_frozen_measured_endpoint(
    tmp_path: Path,
) -> None:
    console = _load_script("native_console_smoke.py")
    runner = _load_script("smoke_hailo10_hef_runner.py")
    suite, artifact, _original_contracts = _old_decoded_hailo10_suite(
        tmp_path,
    )
    child = tmp_path / "isolated_child"
    overlay, _overlay_evidence = (
        console._prepare_hailo10_full_contract_overlay(
            suite,
            child_root=child,
            model_id="yolo26s",
        )
    )
    declaration = load_authoritative_output_contract(
        overlay,
        backend="hailo10h",
        model_id="yolo26s",
        variant="full",
        task="detection",
    )
    outputs = _six_raw_heads()
    frozen = build_frozen_postprocess_contract(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[640, 480],
    )
    frozen = verify_frozen_postprocess_contract(frozen, outputs=outputs)
    measured = FrozenDetectionPostprocessor(frozen).process(
        outputs, original_wh=[640, 480],
    )
    dump_sample = FrozenDetectionPostprocessor(frozen).process(
        outputs, original_wh=[640, 480],
    )
    image = tmp_path / "image.jpg"
    image.write_bytes(b"diagnostic-image-identity")
    output_manifest, input_manifest = runner._write_output_dump(
        outputs,
        child / "native_full_outputs",
        backend="native_full_hailo10h",
        model="yolo26s",
        setup_id="orin_nx_hailo10_01",
        comparison_backend="hailo10h",
        task="detection",
        image=image,
        input_hwc=np.zeros((640, 640, 3), dtype=np.uint8),
        input_tensor=np.zeros((1, 640, 640, 3), dtype=np.uint8),
        input_name="input",
        preprocess={"mode": "letterbox_rgb_uint8"},
        declared_output_contract=declaration,
        frozen_postprocess_contract=frozen,
        frozen_postprocess_result=dump_sample,
        diagnostic_only=True,
    )
    throughput: dict[str, Any] = {
        "frames": 5,
        "requested_frames": 5,
        "minimum_requested_frames": 5,
        "completed_frames": 5,
        "completed_work_units": 5,
        "completed_work_units_status": "exact_runtime_counter",
        "completed_work_units_source": (
            "hailo_infermodel_frozen_postprocess_success_callback_counter"
        ),
        "measurement_control": "exact_frames",
        "warmup_frames": 0,
        "warmup_completed_frames": 0,
        "inflight": 1,
        "postprocess_included": True,
        "postprocess_completed_frames": 5,
        "postprocess_completion_status": "exact_runtime_counter",
        "fps": 12.5,
    }
    report: dict[str, Any] = {
        "ok": True,
        "hef": str(artifact),
        "task": "detection",
        "backend": "native_full_hailo10h",
        "model": "yolo26s",
        "setup_id": "orin_nx_hailo10_01",
        "comparison_backend": "hailo10h",
        "hw_arch": "hailo10h",
        "runtime_api": "infer_model",
        "throughput_mode": True,
        "copy_outputs": True,
        "claim_copy_outputs_verified": True,
        "throughput": throughput,
        "completed_frames": 5,
        "completed_work_units_status": "exact_runtime_counter",
        "output_manifest": str(output_manifest),
        "input_manifest": str(input_manifest),
        "runtime_endpoint_contract_family": "raw_head",
        "host_postprocess_frozen": True,
        "postprocess_included": True,
        "postprocess_completed_frames": 5,
        "frozen_host_postprocess_contract": frozen,
        "frozen_host_postprocess_contract_sha256": frozen[
            "contract_sha256"
        ],
        "frozen_host_postprocess_result": measured,
        "diagnostic_only": True,
        "claim_eligible": False,
        "claim_eligible_e2e": False,
    }
    report_path = child / "hailo10_native_full_results.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    evidence = console._validate_hailo10_full_evidence(
        report,
        report_path=report_path,
        child_root=child,
        expected_image=image,
        expected_frames=5,
        expected_model_id="yolo26s",
        expected_setup_id="orin_nx_hailo10_01",
        expected_declaration=overlay,
    )
    assert evidence["raw_head_count"] == 6
    assert evidence["completed_endpoint_attestation"]["attested"] is True
    assert evidence["completed_frames"] == 5
    assert evidence["postprocess_completed_frames"] == 5
    assert evidence["offline_postprocess"]["technical_pass_count"] == 1
    assert evidence["child_claim_fields_suppressed"] is True
    output_payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert output_payload["diagnostic_only"] is True
    assert output_payload["claim_eligible"] is False
    assert output_payload["claim_eligible_e2e"] is False

    bad = dict(report)
    bad["throughput"] = {
        **throughput,
        "completed_work_units_source": "unverified_counter",
    }
    with pytest.raises(ValueError, match="measurement counters"):
        console._validate_hailo10_full_evidence(
            bad,
            report_path=report_path,
            child_root=child,
            expected_image=image,
            expected_frames=5,
            expected_model_id="yolo26s",
            expected_setup_id="orin_nx_hailo10_01",
            expected_declaration=overlay,
        )
