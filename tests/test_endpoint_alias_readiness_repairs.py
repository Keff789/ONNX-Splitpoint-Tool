from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDecodedNmsPostprocessor,
    build_frozen_decoded_nms_normalization_contract,
    build_normalized_detection_endpoint_attestation,
    canonical_json_sha256,
    tensor_signature,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"endpoint_alias_readiness_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


energy = _load_script("native_producer_energy_plan.py")
final = _load_script("native_producer_final_report.py")
validator = _load_script("native_producer_validate_visualize.py")


def _direct_completion_evidence(
    artifact_dir: Path | None = None,
) -> tuple[
    dict, dict, str, dict[str, np.ndarray],
]:
    outputs = {
        "detections": np.asarray(
            [[
                [8.0, 16.0, 32.0, 48.0, 0.90, 2.0],
                [8.0, 16.0, 32.0, 48.0, 0.80, 2.0],
                [0.0, 0.0, 4.0, 4.0, 0.10, 1.0],
            ]],
            dtype=np.float32,
        ),
    }
    source_hash = "a" * 64
    source_attestation = {
        "schema": "onnx-splitpoint/runtime-output-endpoint-attestation",
        "schema_version": 3,
        "attested": True,
        "status": "passed",
        "endpoint": "decoded_nms",
        "stage": "decoded_nms",
        "values_decoded_xyxy_score_class": True,
        "declaration_attested": True,
        "endpoint_contract_hash": source_hash,
        "tensor_signature": tensor_signature(outputs),
        "declared_contract": {
            "model_id": "yolov7_paper",
            "source_coordinate_space":
                "model_input_letterbox_xyxy_pixels",
        },
    }
    direct = build_frozen_decoded_nms_normalization_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[64, 64],
        original_wh=[80, 60],
        preprocess={
            "mode": "letterbox_rgb_uint8",
            "rgb": True,
            "pad_value": 114,
        },
        source_coordinate_space=(
            "model_input_letterbox_xyxy_pixels"
        ),
        source_endpoint_contract_hash=source_hash,
        source_output_endpoint_attestation=source_attestation,
    )
    result = FrozenDecodedNmsPostprocessor(direct).process(
        outputs, original_wh=[80, 60],
    )
    completion = build_normalized_detection_endpoint_attestation(
        direct,
        result,
        completed_frames=3,
        postprocess_completed_frames=3,
    )
    comparison = completion[
        "completed_task_comparison_endpoint_contract"
    ]
    workload = {
        "frozen_decoded_nms_normalization_contract": direct,
        "frozen_decoded_nms_normalization_contract_sha256":
            direct["contract_sha256"],
        "source_endpoint_contract_hash": source_hash,
        "source_output_endpoint_id": direct["source_output_endpoint_id"],
        "source_output_tensor_signature":
            direct["source_output_tensor_signature"],
        "source_output_endpoint_attestation_sha256":
            direct["source_output_endpoint_attestation_sha256"],
        "completed_task_endpoint_attestation": completion,
        "frozen_decoded_nms_normalization_result": result,
    }
    full_command_contract = {"energy_workload": workload}
    full_command_contract["contract_sha256"] = canonical_json_sha256(
        full_command_contract
    )
    row = {
        "backend": "native_full_deepx",
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": source_hash,
        "output_endpoint_id": direct["source_output_endpoint_id"],
        "output_endpoint_attestation": source_attestation,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation": completion,
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash":
            comparison["endpoint_contract_hash"],
        "completed_task_comparison_output_endpoint_id":
            comparison["output_endpoint_id"],
        "completed_task_completion_mode":
            completion["completed_task_completion_mode"],
        "full_command_contract": full_command_contract,
        "full_command_contract_sha256":
            full_command_contract["contract_sha256"],
    }
    if artifact_dir is not None:
        artifact = result["completed_result_artifact"]
        artifact_path = artifact_dir / "direct_completed_result.json"
        artifact_path.write_text(
            json.dumps(
                artifact,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        artifact_sha256 = canonical_json_sha256(artifact)
        row.update({
            "completed_task_result_artifact": artifact,
            "completed_task_result_artifact_path": str(
                artifact_path.resolve()
            ),
            "completed_task_result_artifact_saved": True,
            "completed_task_result_artifact_sha256": artifact_sha256,
            "completed_task_result_artifact_file_sha256": (
                artifact_sha256
            ),
        })
    return row, workload, comparison["output_endpoint_id"], outputs


def test_direct_bn6_final_report_and_energy_use_strict_comparison_identity(
) -> None:
    row, workload, comparison_id, _outputs = (
        _direct_completion_evidence()
    )

    assert (
        final._explicit_completed_task_comparison_endpoint(row)
        == comparison_id
    )
    identity = energy._energy_endpoint_identity(
        row, row, {"energy_workload": workload},
    )
    assert identity["physical_output_endpoint_id"] == row[
        "output_endpoint_id"
    ]
    assert identity["output_endpoint_id"] == comparison_id
    assert identity["comparison_output_endpoint_id"] == comparison_id
    assert identity["completion_pairing_eligible"] is True
    assert identity["completion_pairing_status"] == (
        "strict_completed_detection_endpoint_verified"
    )


def test_final_report_projects_verified_direct_contract_for_validator(
    tmp_path: Path,
) -> None:
    row, _workload, comparison_id, native_outputs = (
        _direct_completion_evidence(tmp_path)
    )
    row.update({
        "model": "yolov7_paper",
        "case": "full",
        "precision": "fp16",
        "ok": True,
        "fps_makespan": 12.5,
        "outer_makespan_verified": True,
        "completed_frames": 3,
        "postprocess_completed_frames": 3,
    })
    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    (analysis / "native_full_baseline_eval.json").write_text(
        json.dumps({"rows": [row]}),
        encoding="utf-8",
    )

    projected = final._rows_from_native_full(tmp_path)[0]

    assert projected[
        "frozen_decoded_nms_normalization_contract"
    ] == row["full_command_contract"]["energy_workload"][
        "frozen_decoded_nms_normalization_contract"
    ]
    assert projected[
        "frozen_decoded_nms_normalization_contract_sha256"
    ] == projected[
        "frozen_decoded_nms_normalization_contract"
    ]["contract_sha256"]
    assert projected[
        "frozen_decoded_nms_normalization_result"
    ] == row["completed_task_endpoint_attestation"][
        "frozen_decoded_nms_normalization_result"
    ]
    assert projected["normalization_frozen"] is True
    assert projected["direct_bn6_completion_projection_status"] == (
        "strict_direct_bn6_completion_verified"
    )
    assert final._explicit_completed_task_comparison_endpoint(
        projected
    ) == comparison_id
    self_reference = (
        validator._completed_v2_self_reference_detection(
            native_outputs,
            native_outputs,
            projected,
        )
    )
    assert self_reference["available"] is True
    assert self_reference["completed_v2_verified"] is True
    assert self_reference["native_mode"] == (
        "completed_v2:integrated_accelerator_plus_"
        "frozen_normalization"
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "source_attestation",
        "direct_contract",
        "completion",
        "completion_mode",
    ],
)
def test_direct_bn6_completion_tamper_fails_closed(tamper: str) -> None:
    row, _workload, _comparison_id, _outputs = (
        _direct_completion_evidence()
    )
    drifted = copy.deepcopy(row)
    workload = drifted["full_command_contract"]["energy_workload"]

    if tamper == "source_attestation":
        drifted["output_endpoint_attestation"]["status"] = "failed"
    elif tamper == "direct_contract":
        workload[
            "frozen_decoded_nms_normalization_contract"
        ]["confidence_threshold"] = 0.50
    elif tamper == "completion":
        workload["completed_task_endpoint_attestation"][
            "postprocess_completed_frames"
        ] = 2
    else:
        drifted["completed_task_completion_mode"] = (
            "integrated_accelerator"
        )

    assert final._explicit_completed_task_comparison_endpoint(
        drifted
    ) == ""
    identity = energy._energy_endpoint_identity(
        drifted, drifted, drifted["full_command_contract"],
    )
    assert identity["completion_pairing_eligible"] is False
    assert identity["output_endpoint_id"] == identity[
        "physical_output_endpoint_id"
    ]


def test_detection_split_without_measured_completion_remains_physical(
) -> None:
    physical_hash = "b" * 64
    row = {
        "backend": "hailo10h_to_trt",
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": physical_hash,
        "output_endpoint_id": f"detection:raw_head:{physical_hash}",
    }

    identity = energy._energy_endpoint_identity(row, row, {})

    assert identity["output_endpoint_id"] == row["output_endpoint_id"]
    assert identity["endpoint_contract_hash"] == physical_hash
    assert identity["comparison_output_endpoint_id"] == ""
    assert identity["completion_pairing_eligible"] is False
    assert identity["completion_pairing_status"] == (
        "detection_completed_endpoint_attestation_missing"
    )


def test_classification_keeps_physical_endpoint_identity() -> None:
    physical_hash = "c" * 64
    row = {
        "backend": "hailo8_to_trt",
        "task": "classification",
        "stage": "classification_logits",
        "contract_family": "classification_logits",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": physical_hash,
        "output_endpoint_id":
            f"classification:classification_logits:{physical_hash}",
    }

    identity = energy._energy_endpoint_identity(row, row, {})

    assert identity["output_endpoint_id"] == row["output_endpoint_id"]
    assert identity["physical_output_endpoint_id"] == row[
        "output_endpoint_id"
    ]
    assert identity["comparison_output_endpoint_id"] == row[
        "output_endpoint_id"
    ]
    assert identity["completion_pairing_eligible"] is True
    assert identity["completion_pairing_status"] == (
        "classification_physical_endpoint_preserved"
    )


def test_deepx_missing_readiness_is_not_synthesized_false(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    report = tmp_path / "deepx_results.json"
    report.write_text(
        json.dumps({"ok": True, "fps_makespan": 12.5}),
        encoding="utf-8",
    )
    (analysis / "native_deepx_producer_e2e_eval.json").write_text(
        json.dumps({
            "rows": [{
                "model": "resnet50",
                "case": "b001",
                "precision": "uint8_cast_fp16",
                "report": str(report),
                "ok": True,
                "fps_makespan": 12.5,
            }],
        }),
        encoding="utf-8",
    )

    rows = final._rows_from_deepx(tmp_path)

    assert len(rows) == 1
    assert rows[0]["ok"] is True
    assert rows[0]["producer_ready"] is None
    assert rows[0]["consumer_ready"] is None
    assert rows[0]["buildable"] is True
    assert rows[0]["runtime_executable"] is True


def test_deepx_explicit_build_failure_remains_fail_closed(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    report = tmp_path / "deepx_results.json"
    report.write_text(
        json.dumps({
            "ok": True,
            "buildable": False,
            "fps_makespan": 12.5,
        }),
        encoding="utf-8",
    )
    (analysis / "native_deepx_producer_e2e_eval.json").write_text(
        json.dumps({
            "rows": [{
                "model": "resnet50",
                "case": "b001",
                "precision": "uint8_cast_fp16",
                "report": str(report),
                "ok": True,
                "fps_makespan": 12.5,
            }],
        }),
        encoding="utf-8",
    )

    row = final._rows_from_deepx(tmp_path)[0]

    assert row["ok"] is True
    assert row["buildable"] is False
    assert row["runtime_executable"] is True
