from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from onnx_splitpoint_tool.workflow.runner import (
    _native_expected_matrix_status_v60y,
)
from scripts.native_producer_energy_plan import (
    _full_runtime_argv,
    _split_preflight_argv,
)
from scripts.native_producer_final_report import _rows_from_native_full
from scripts.native_producer_validate_visualize import (
    _completed_frozen_nms_attestation_passed,
    _native_full_e2e_contract_gate,
)
from scripts.offline_native_detection_postprocess import audit_pack


ROOT = Path(__file__).resolve().parents[1]


def _yolov7_outputs() -> dict[str, np.ndarray]:
    return {
        "output": np.full((1, 3, 80, 80, 85), -20.0, dtype=np.float32),
        "clone_1": np.full((1, 3, 40, 40, 85), -20.0, dtype=np.float32),
        "clone_2": np.full((1, 3, 20, 20, 85), -20.0, dtype=np.float32),
    }


def _frozen_evidence() -> tuple[dict, dict]:
    outputs = _yolov7_outputs()
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper", outputs=outputs,
        input_hw=[640, 640], original_wh=[1280, 720],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    result = FrozenDetectionPostprocessor(contract).process(
        outputs, original_wh=[1280, 720],
    )
    return contract, result


def test_present_rows_are_not_a_complete_native_matrix_when_one_failed() -> None:
    expected = [
        {"model": "resnet50", "backend": "hailo8_to_trt", "case": "b001"},
        {"model": "yolo26s", "backend": "deepx_to_trt", "case": "b002"},
    ]
    actual = [
        {**expected[0], "ok": True, "status": "ok"},
        {
            **expected[1], "ok": False, "status": "failed",
            "failure_reason": "runtime_boundary_shape_mismatch",
        },
    ]
    status = _native_expected_matrix_status_v60y(expected, actual, [])

    assert status["present_expected_row_count"] == 2
    assert status["successful_expected_row_count"] == 1
    assert status["failed_expected_row_count"] == 1
    assert status["missing_expected_row_count"] == 0
    assert status["row_presence_complete"] is True
    assert status["execution_success_complete"] is False
    assert status["matrix_complete"] is False
    assert status["failed_expected_rows"][0]["failure_reason"] == (
        "runtime_boundary_shape_mismatch"
    )


def test_raw_and_completed_detection_endpoints_remain_separate(
    tmp_path: Path,
) -> None:
    contract, result = _frozen_evidence()
    attestation = build_completed_detection_endpoint_attestation(
        contract, result, completed_frames=1000,
        postprocess_completed_frames=1000,
        source_endpoint_contract_hash="a" * 64,
    )
    row = {
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_hash": "a" * 64,
        "host_postprocess_frozen": True,
        "postprocess_included": True,
        "postprocess_completion_verified": True,
        "frozen_host_postprocess_contract": contract,
        "completed_task_endpoint_attestation": attestation,
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "completed_task_completion_mode": "frozen_host_tail",
        "completed_task_comparison_endpoint_contract": attestation[
            "completed_task_comparison_endpoint_contract"
        ],
        "completed_task_comparison_endpoint_contract_hash": attestation[
            "completed_task_comparison_endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": attestation[
            "completed_task_comparison_output_endpoint_id"
        ],
    }
    assert _completed_frozen_nms_attestation_passed(row) is True

    manifest = tmp_path / "native_full_outputs_manifest.json"
    manifest.write_text(json.dumps({
        "input_contract_mode": "explicit",
        "contract_family": "raw_head",
    }), encoding="utf-8")
    gate = _native_full_e2e_contract_gate(
        manifest, {"backend": "native_full_hailo8", **row}, "detection",
    )
    assert gate["e2e_claim_eligible"] is True
    assert gate["completed_task_endpoint_attested"] is True
    assert row["contract_family"] == "raw_head"


def test_final_report_propagates_frozen_completion_without_relabelling_dump(
    tmp_path: Path,
) -> None:
    contract, result = _frozen_evidence()
    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    (analysis / "native_full_baseline_eval.json").write_text(json.dumps({
        "rows": [{
            "backend": "native_full_hailo8",
            "model": "yolov7_paper",
            "task": "detection",
            "ok": True,
            "fps_makespan": 10.0,
            "completed_frames": 1000,
            "stage": "raw_head",
            "contract_family": "raw_head",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": "b" * 64,
            "host_postprocess_frozen": True,
            "postprocess_included": True,
            "postprocess_completed_frames": 1000,
            "postprocess_completion_verified": True,
            "frozen_host_postprocess_contract": contract,
            "frozen_host_postprocess_contract_sha256": contract[
                "contract_sha256"
            ],
            "frozen_host_postprocess_result": result,
        }],
    }), encoding="utf-8")

    rows = _rows_from_native_full(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["accelerator_output_contract_family"] == "raw_head"
    assert row["completed_task_contract_family"] == "decoded_nms"
    assert row["completed_task_endpoint_attestation_status"] == "passed"
    assert row["completed_task_endpoint_contract_hash"]


def test_energy_commands_reference_remote_contract_file_not_inline_json() -> None:
    contract = {
        "python_executable": "/usr/bin/python3",
        "runner": "scripts/native_full_baseline_eval_runner.py",
        "root": "/remote/eval",
        "runner_sha256": "a" * 64,
        "contract_sha256": "b" * 64,
        "energy_workload": {"available": True, "kind": "hailo_full_hotloop"},
    }
    argv = _full_runtime_argv(
        contract, duration_s=10.0, frames=1,
        remote_tool_dir="/remote/tool", preflight_nonce="nonce",
        authoritative_root="/remote/eval",
        preflight_attestation="/remote/preflight.json",
        preflight_max_age_s=300.0,
        remote_contract_file="/remote/contract.json",
    )
    assert "--energy-command-contract-file" in argv
    assert "/remote/contract.json" in argv
    assert "--energy-command-contract-json" not in argv

    split_argv = _split_preflight_argv(
        {
            "python_executable": "/usr/bin/python3",
            "contract_sha256": "c" * 64,
        },
        remote_tool_dir="/remote/tool", preflight_nonce="nonce",
        preflight_attestation="/remote/preflight.json", max_age_s=300.0,
        remote_contract_file="/remote/split-contract.json",
    )
    assert "--contract-json" in split_argv
    assert "/remote/split-contract.json" in split_argv
    assert "--contract-json-payload" not in split_argv

    source = (ROOT / "scripts/native_producer_energy_plan.py").read_text(
        encoding="utf-8",
    )
    assert "ssh_stdin_to_hash_verified_remote_file" in source
    assert "sha256sum" in source
    assert "stdin_path=local_contract_file" in source
    assert 'cat > "$tmp_contract"' in source


def test_offline_postprocess_audit_accepts_directory_and_zip(
    tmp_path: Path,
) -> None:
    outputs = _yolov7_outputs()
    contract, result = _frozen_evidence()
    dump = tmp_path / "pack" / "native_full_outputs" / "yolov7"
    dump.mkdir(parents=True)
    entries = []
    for index, (name, array) in enumerate(outputs.items()):
        filename = f"output_{index:02d}.bin"
        payload = np.ascontiguousarray(array).tobytes()
        (dump / filename).write_bytes(payload)
        entries.append({
            "name": name,
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    (dump / "native_full_input_manifest.json").write_text(json.dumps({
        "runtime_input_shape": [1, 3, 640, 640],
    }), encoding="utf-8")
    (dump / "native_full_outputs_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/runner-output-dump",
        "task": "detection",
        "model": "yolov7_paper",
        "backend": "native_full_hailo8",
        "contract_family": "raw_head",
        "outputs": entries,
        "frozen_host_postprocess_contract": contract,
        "frozen_host_postprocess_result": result,
    }), encoding="utf-8")

    directory_report = audit_pack(tmp_path / "pack")
    assert directory_report["ok"] is True
    assert directory_report["technical_pass_count"] == 1
    assert directory_report["rows"][0]["status"] == "passed"

    archive = tmp_path / "debug_pack.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in (tmp_path / "pack").rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(tmp_path / "pack"))
    zip_report = audit_pack(archive)
    assert zip_report["ok"] is True
    assert zip_report["rows"][0]["detections_sha256"] == (
        directory_report["rows"][0]["detections_sha256"]
    )


def test_offline_replay_rebinds_legacy_yolov7_only_as_diagnostic(
    tmp_path: Path,
) -> None:
    contract = json.loads((
        ROOT / "tests/fixtures/v270c/run39_native_repairs.json"
    ).read_text(encoding="utf-8"))["deepx_yolov7_prepared_feed"][
        "frozen_host_postprocess_contract"
    ]
    archived = verify_frozen_postprocess_contract(contract)
    assert archived["schema_version"] == 1
    assert archived["legacy_decoder_contract_unbound"] is True
    outputs = {
        "model_outputs": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "output_1": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "output_2": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }
    dump = tmp_path / "legacy_pack" / "native_full_outputs"
    dump.mkdir(parents=True)
    entries = []
    for index, (name, array) in enumerate(outputs.items()):
        filename = f"output_{index:02d}.bin"
        payload = np.ascontiguousarray(array).tobytes()
        (dump / filename).write_bytes(payload)
        entries.append({
            "name": name,
            "file": filename,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    (dump / "native_full_outputs_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/runner-output-dump",
        "task": "detection",
        "model": "yolov7_paper",
        "backend": "native_full_hailo8",
        "contract_family": "raw_head",
        "outputs": entries,
        "frozen_host_postprocess_contract": contract,
    }), encoding="utf-8")

    report = audit_pack(tmp_path / "legacy_pack")
    row = report["rows"][0]

    assert report["ok"] is False
    assert report["technical_pass_count"] == 0
    assert row["technical_ok"] is False
    assert row["claim_evidence"] is False
    assert row["status"] == "technical_decode_failed"
    assert row["error"].endswith(
        "offline_yolov7_replay_exact_model_sha256_required"
    )
    assert (
        "archived_frozen_contract_activation_strategy_unbound"
        in row["warnings"]
    )


def test_modified_remote_script_mirrors_are_byte_identical() -> None:
    for name in (
        "native_full_baseline_eval_runner.py",
        "native_producer_energy_plan.py",
        "native_producer_final_report.py",
        "native_producer_validate_visualize.py",
        "smoke_hailo10_hef_runner.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT / "onnx_splitpoint_tool/resources/remote_scripts" / name
        ).read_bytes()
