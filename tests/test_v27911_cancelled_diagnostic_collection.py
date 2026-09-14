from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import zipfile
from pathlib import Path

from onnx_splitpoint_tool.benchmark.remote_run import (
    _rowless_full_only_quality_run_statuses,
)
from onnx_splitpoint_tool.build_scheduler import BuildScheduler, BuildTaskSpec
from onnx_splitpoint_tool.filesystem_admission import FilesystemWriteInspection
from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool.workflow.zip_utils import portable_zip_datetime


ROOT = Path(__file__).resolve().parents[1]


def _sha(value: dict) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mixed_quality_report() -> dict:
    identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": "tensorrt_at_hailo8_full",
        "eval_run_id": "eval-1",
        "model_id": "resnet50",
        "setup_id": "orin_nx_hailo8_01",
        "source_run_id": "native_full_tensorrt",
        "backend": "tensorrt",
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    request = {
        "record_count": 500,
        "quality_canary_id": "tensorrt_at_hailo8_full",
        "eval_run_id": "eval-1",
        "performance_claims_emitted": False,
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": identity,
        "full_only_plan_identity_sha256": _sha(identity),
    }
    return {
        "schema": (
            "onnx-splitpoint/native-full-tensorrt-quality-evidence-report"
        ),
        "schema_version": 1,
        "quality_evidence_only": True,
        "performance_claims_emitted": False,
        "eval_run_id": "eval-1",
        "model_id": "resnet50",
        "setup_id": "orin_nx_hailo8_01",
        "source_run_id": "native_full_tensorrt",
        "backend": "native_tensorrt",
        "variant": "full",
        "execution_role": "full_quality_only",
        "record_count": 500,
        "quality_input_request": request,
    }


def test_mixed_suite_quality_only_terminal_is_not_failed() -> None:
    plan = {
        "runs": [
            {
                "id": "ort_tensorrt",
                "type": "onnxruntime",
                "backend": "tensorrt",
                "provider": "tensorrt",
                "stage1": {"provider": "tensorrt"},
                "stage2": {"provider": "tensorrt"},
            },
            {"id": "hailo8", "type": "hailo", "hw_arch": "hailo8"},
            {"id": "hailo8_to_trt", "type": "matrix"},
        ]
    }
    report = _mixed_quality_report()
    suite_status = {
        "any_rows": True,
        "any_quality_evidence": True,
        "quality_evidence_report_count": 1,
        "quality_evidence_reports": [report],
        "performance_claims_emitted": True,
        "failed_runs": [],
        "total_runs": 3,
    }
    result = _rowless_full_only_quality_run_statuses(
        plan=plan,
        suite_status=suite_status,
        final_status="ok",
        benchmark_result_count=2,
        expected_eval_run_id="eval-1",
        expected_model_id="resnet50",
        expected_setup_id="orin_nx_hailo8_01",
        explicit_quality_only_run_ids=["ort_tensorrt"],
        expected_endpoint_id="tensorrt_at_hailo8_full",
    )
    assert result == {"ort_tensorrt": "quality_evidence_only_complete"}

    report["quality_input_request"]["full_only_plan_identity_sha256"] = "0" * 64
    assert _rowless_full_only_quality_run_statuses(
        plan=plan,
        suite_status=suite_status,
        final_status="ok",
        benchmark_result_count=2,
        expected_eval_run_id="eval-1",
        expected_model_id="resnet50",
        expected_setup_id="orin_nx_hailo8_01",
        explicit_quality_only_run_ids=["ort_tensorrt"],
        expected_endpoint_id="tensorrt_at_hailo8_full",
    ) == {}


def test_local_dfc_workspace_preflight_checks_bytes_and_inodes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    inspection = FilesystemWriteInspection(
        requested_path=tmp_path,
        probe_path=tmp_path,
        writable=True,
        read_only_mount=False,
        permission_bits_allow=True,
        os_access_allow=True,
        free_bytes=50_000_000_000,
        free_inodes=40_000,
        reason="ok",
    )
    monkeypatch.setattr(hailo_backend, "inspect_write_target", lambda _path: inspection)
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_DFC_WORKSPACE_RESERVE_BYTES", "0")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_DFC_MIN_FREE_BYTES", "0")
    monkeypatch.setenv("ONNX_SPLITPOINT_HAILO_DFC_MIN_FREE_INODES", "50000")
    result = hailo_backend.hailo_dfc_workspace_preflight(
        tmp_path,
        calibration_count=500,
        input_shapes=[[640, 640, 3]],
    )
    assert result["status"] == "failed"
    assert result["calculation"]["required_free_bytes"] == 58_982_400_000
    assert any(value.startswith("free_bytes=") for value in result["problems"])
    assert any(value.startswith("free_inodes=") for value in result["problems"])


def test_mapping_timeout_and_enospc_are_machine_classified() -> None:
    mapping = hailo_backend._classify_hailo_failure_text(
        "Watchdog expired after 1h 0m 0s\nMapping Failed (Timeout, allocation time: 1h)"
    )
    assert mapping["failure_kind"] == "hailo_dfc_mapping_timeout"
    assert mapping["timed_out"] is True
    assert mapping["timeout_kind"] == "hailo_dfc_mapping_watchdog"

    storage = hailo_backend._hef_result_from_payload(
        {"ok": False, "error": "[Errno 28] No space left on device", "failure_kind": None},
        elapsed_default=1.0,
        hw_arch="hailo10h",
        net_name="part1",
        backend_default="venv",
    )
    assert storage.failure_kind == "local_dfc_workspace_exhausted"
    assert storage.timed_out is False


def test_scheduler_separates_transport_from_artifact_result() -> None:
    logs: list[str] = []
    with BuildScheduler(
        max_workers=1, cpu_tokens=1, log=logs.append,
    ) as scheduler:
        future = scheduler.submit(
            BuildTaskSpec("hailo-case", "hailo8"),
            lambda: {"ok": False, "failure_kind": "mapping_failed"},
        )
        assert future.result()["ok"] is False
        event = scheduler.events()[0]
    assert event["transport_status"] == "success"
    assert event["invocation_status"] == "returned"
    assert event["artifact_status"] == "failed"
    assert event["semantic_status"] == "failed"
    assert event["status"] == "failed"
    assert "transport=success" in logs[-1]
    assert "artifact=failed" in logs[-1]


def _collector_module():
    path = ROOT / "scripts" / "collect_cancelled_diagnostic.py"
    spec = importlib.util.spec_from_file_location(
        "v27911_collect_cancelled_diagnostic", path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cancel_collector_prioritizes_control_plane_and_clamps_zip_time(
    tmp_path: Path,
) -> None:
    run = tmp_path / "interrupted_run"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps({
            "schema": "onnx-splitpoint/evaluation-run-manifest",
            "schema_version": 1,
            "run_id": run.name,
            "status": "running",
        }),
        encoding="utf-8",
    )
    (run / "profile.yaml").write_text("model_suite: {primary: []}\n", encoding="utf-8")
    (run / "evaluation_workflow.log").write_text("power loss\n", encoding="utf-8")
    priority_payloads = {
        "jobs/remote_process_leases/session/lease.remote-lease.json": "{}\n",
        "models/resnet50/stages/run_benchmarks/stage_result.json": (
            '{"status":"ok"}\n'
        ),
        "models/resnet50/benchmark_results/remote_diagnostics/"
        "orin_nx_hailo8_01/run_status.json": '{"status":"ok"}\n',
        "quality_management/central_quality_queue_status.json": (
            '{"status":"partial"}\n'
        ),
    }
    for relative, payload in priority_payloads.items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        os.utime(path, (0, 0))
    bulky = run / "models" / "resnet50" / "validation" / "raw_rows.json"
    bulky.parent.mkdir(parents=True, exist_ok=True)
    bulky.write_bytes(b"x" * 4096)

    output = tmp_path / "cancelled.zip"
    result = _collector_module().collect_cancelled_diagnostic(
        run, output, max_small_file_bytes=64,
    )
    assert result["priority_complete"] is True
    assert result["zip_timestamp_contract"] == "clamped_1980_2107"
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert set(priority_payloads) <= names
        assert "models/resnet50/validation/raw_rows.json" not in names
        for relative in priority_payloads:
            assert archive.getinfo(relative).date_time[0] == 1980
        manifest = json.loads(archive.read("debug_pack_manifest.json"))
        assert manifest["cancelled_diagnostic_priority"]["complete"] is True


def test_zip_timestamp_clamp_covers_both_bounds_and_non_finite_values() -> None:
    assert portable_zip_datetime(-1) == (1980, 1, 1, 0, 0, 0)
    assert portable_zip_datetime(float("-inf")) == (1980, 1, 1, 0, 0, 0)
    assert portable_zip_datetime(10**30) == (2107, 12, 31, 23, 59, 58)
    assert portable_zip_datetime(float("inf")) == (
        2107, 12, 31, 23, 59, 58,
    )
    assert 1980 <= portable_zip_datetime(float("nan"))[0] <= 2107
