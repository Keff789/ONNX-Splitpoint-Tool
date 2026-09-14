from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.results import (
    expand_normalized_benchmark_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_gate():
    path = ROOT / "scripts" / "verify_v27910_yolo11_r8b_gate.py"
    spec = importlib.util.spec_from_file_location("v27910_backend_gate", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _completion_row() -> tuple[dict, dict]:
    completed_payload = {
        "schema": "onnx-splitpoint/frozen-completed-detection-result-artifact",
        "schema_version": 1,
        "record_schema": "xyxy_score_class_id_v1",
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": "score_desc_class_id_asc_xyxy_lexicographic_v1",
        "detections": [{
            "class_id": 0,
            "score": 0.9,
            "x1": 1.0,
            "y1": 2.0,
            "x2": 10.0,
            "y2": 20.0,
        }],
    }
    completed_sha = gate._canonical_sha(completed_payload)
    command = {
        "schema": "onnx-splitpoint/native-full-command-contract",
        "schema_version": 1,
        "runner": "scripts/native_full_baseline_eval_runner.py",
        "runner_sha256": gate.base._file_sha(
            ROOT / "scripts/native_full_baseline_eval_runner.py"
        ),
        "backend": "native_full_hailo10h",
        "backend_arg": "hailo10h",
        "model": "yolo11l",
        "case": "full",
        "setup_id": "orin_nx_hailo10_01",
        "comparison_backend": "hailo10h",
        "runtime_options": {
            "frames": 32,
            "warmup": 4,
            "performance_repetitions": 1,
            "inflight": 1,
            "dump_outputs": True,
        },
        "artifacts": {
            "performance_report": {
                "path": "/remote/native_full_hailo_report.json",
                "sha256": "5" * 64,
            },
        },
        "complete": True,
    }
    command_sha = gate._canonical_sha(command)
    command["contract_sha256"] = command_sha
    row = {
        "frames": 32,
        "completed_frames": 32,
        "completed_work_units": 32,
        "completed_work_units_source": (
            "dx_engine_prepared_feed_frozen_completion_timed_loop"
        ),
        "completed_work_units_status": "exact_runtime_counter",
        "fps_makespan": 91.5,
        "latency_mean_ms": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "latency_semantics": "not_measured_async_or_streaming_throughput",
        "completion_interval_mean_ms": 1000.0 / 91.5,
        "measurement_concurrency": 1,
        "configured_inflight": 1,
        "e2e_scope": "full_task_pipeline",
        "postprocess_included": True,
        "postprocess_completion_verified": True,
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "completed_task_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "endpoint_contract_hash": "a" * 64,
            "output_endpoint_id": "decoded_nms:" + "a" * 64,
        },
        "completed_task_endpoint_contract_hash": "a" * 64,
        "completed_task_output_endpoint_id": "decoded_nms:" + "a" * 64,
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": completed_payload,
        "completed_task_result_artifact_sha256": completed_sha,
        "completed_task_result_artifact_file_sha256": completed_sha,
        "completed_task_result_artifact_verification_status": "verified_exact",
        "normalization_frozen": False,
        "frozen_host_postprocess_result": {
            "task": "detection",
            "contract_family": "decoded_nms",
            "detection_count": 1,
            "detections": completed_payload["detections"],
            "completed_result_artifact": completed_payload,
            "completed_result_artifact_sha256": completed_sha,
        },
        "full_command_contract": command,
        "full_command_contract_sha256": command_sha,
    }
    indexed = {
        "reports/native/h10.completed.json": {
            "relative_path": "reports/native/h10.completed.json",
            "sha256": completed_sha,
            "size_bytes": 1,
        },
    }
    return row, indexed


def test_backend_bound_specs_are_unique_and_exact() -> None:
    assert [spec.key for spec in gate.NATIVE_SPECS] == [
        "hailo8_full", "hailo10h_full", "deepx_full",
    ]
    assert [spec.key for spec in gate.COMPOSED_SPECS] == [
        "hailo10h_b067_composed", "deepx_b067_composed",
    ]
    assert len({spec.key for spec in gate.COMPOSED_SPECS}) == 2
    assert gate.COMPOSED_SPECS[0].run_id == "hailo10_to_tensorrt"
    assert gate.COMPOSED_SPECS[0].filename == (
        "benchmark_results_hailo10_to_tensorrt_auto.json"
    )


def test_full_projection_seals_b067_source_case_and_exact_row_hash(
    tmp_path: Path,
) -> None:
    raw = {
        "case_id": "b067",
        "run_id": "hailo10_to_tensorrt",
        "variant": "composed",
        "primary_variant": "composed",
        "provider": "tensorrt",
        "full_provider": "hailo10h",
        "stage1_provider": "hailo10h",
        "stage2_provider": "tensorrt",
        "runtime_ok": True,
        "returncode": 0,
        "full_mean_ms": 20.0,
        "measured_variants": ["full", "composed"],
        "variant_status": {"full": "ok", "composed": "ok"},
        "timings": {
            "full": {"mean_ms": 20.0},
            "composed": {"mean_ms": 12.0},
        },
    }
    rows = expand_normalized_benchmark_rows(
        raw, model_id="yolo11l", source_path=tmp_path / "result.json",
    )
    primary = next(row for row in rows if row["case_id"] == "b067")
    projected = [
        row for row in rows if "source_case_row_sha256" in row
    ]
    assert len(projected) == 1
    full = projected[0]
    assert full["case_id"] == "full"
    assert full["source_case_id"] == "b067"
    assert full["source_case_row_sha256"] == primary["source_row_sha256"]
    assert full["source_case_row_sha256"] == gate._canonical_sha(raw)


def test_composed_raw_row_binds_to_unique_normalized_source_digest(
    tmp_path: Path,
) -> None:
    spec = gate.COMPOSED_SPECS[0]
    raw = {
        "case_id": "b067",
        "run_id": spec.run_id,
        "provider": "tensorrt",
        "stage1_provider": "hailo10h",
        "stage2_provider": "tensorrt",
        "throughput": {
            "mode": "measured_streaming",
            "fps_makespan": 51.0,
        },
    }
    assert "source_row_sha256" not in raw
    source_sha = gate._canonical_sha(raw)
    normalized_path = (
        tmp_path / "models/yolo11l/benchmark_results/normalized_results.json"
    )
    normalized_path.parent.mkdir(parents=True)
    normalized_path.write_text(json.dumps({"results": [{
        "case_id": "b067",
        "run_id": spec.run_id,
        "source_row_sha256": source_sha,
    }]}), encoding="utf-8")
    bound_sha, bound_path, bound_row = gate._bind_normalized_composed_source(
        tmp_path, spec, raw,
    )
    assert bound_sha == source_sha
    assert bound_path == normalized_path
    assert bound_row["source_row_sha256"] == source_sha

    normalized_path.write_text(json.dumps({"results": [{
        **bound_row, "source_row_sha256": "0" * 64,
    }]}), encoding="utf-8")
    with pytest.raises(gate.GateError, match="normalized_source_binding"):
        gate._bind_normalized_composed_source(tmp_path, spec, raw)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("latency_mean_ms", 10.0, "native_reciprocal_latency_forbidden"),
        ("latency_p50_ms", 10.0, "native_p50_latency_forbidden"),
        ("latency_p95_ms", 11.0, "native_p95_latency_forbidden"),
        ("latency_semantics", "request_latency", "native_latency_semantics"),
        ("measurement_concurrency", 2, "native_measurement_concurrency"),
        ("configured_inflight", 8, "native_configured_inflight"),
        ("completed_work_units", 31, "native_completed_units"),
        ("completed_work_units_status", "estimated", "native_completion_counter"),
        ("postprocess_completion_verified", False, "native_postprocess_unverified"),
        ("completed_task_endpoint_attested", False, "native_endpoint_unattested"),
        ("full_backend_throughput_source", "full_latency_fps", "native_generic_full_latency_source_forbidden"),
    ],
)
def test_hailo_native_completion_rejects_each_generic_or_unbound_mutation(
    field: str, value: object, reason: str,
) -> None:
    row, indexed = _completion_row()
    row[field] = value
    with pytest.raises(gate.GateError, match=reason):
        gate._validate_completion(
            row, gate.NATIVE_SPECS[1], indexed=indexed,
        )


def test_hailo_native_completion_binds_command_and_completed_bytes() -> None:
    row, indexed = _completion_row()
    evidence = gate._validate_completion(
        row, gate.NATIVE_SPECS[1], indexed=indexed,
    )
    assert evidence["completed_result_sha256"] in {
        item["sha256"] for item in indexed.values()
    }
    assert evidence["full_command_contract_sha256"] == (
        row["full_command_contract_sha256"]
    )

    row["completed_task_result_artifact"]["schema"] = (
        "onnx-splitpoint/completed-task-result"
    )
    with pytest.raises(gate.GateError, match="artifact.*schema"):
        gate._validate_completion(
            row, gate.NATIVE_SPECS[1], indexed=indexed,
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("runner_sha", "runner_bytes"),
        ("backend_arg", "backend_arg"),
        ("comparison_backend", "comparison"),
        ("frames", "option.*frames"),
        ("warmup", "option.*warmup"),
        ("repetitions", "option.*performance_repetitions"),
        ("inflight", "option.*inflight"),
        ("dump_outputs", "option.*dump_outputs"),
        ("performance_report", "report_sha"),
    ],
)
def test_self_rehashed_wrong_runner_contract_is_still_rejected(
    mutation: str, reason: str,
) -> None:
    row, indexed = _completion_row()
    command = dict(row["full_command_contract"])
    command.pop("contract_sha256")
    command["runtime_options"] = dict(command["runtime_options"])
    command["artifacts"] = {
        key: dict(value) for key, value in command["artifacts"].items()
    }
    if mutation == "runner_sha":
        command["runner_sha256"] = "6" * 64
    elif mutation == "backend_arg":
        command["backend_arg"] = "generic"
    elif mutation == "comparison_backend":
        command["comparison_backend"] = "hailo8"
    elif mutation == "frames":
        command["runtime_options"]["frames"] = 31
    elif mutation == "warmup":
        command["runtime_options"]["warmup"] = 3
    elif mutation == "repetitions":
        command["runtime_options"]["performance_repetitions"] = 2
    elif mutation == "inflight":
        command["runtime_options"]["inflight"] = 8
    elif mutation == "dump_outputs":
        command["runtime_options"]["dump_outputs"] = False
    else:
        command["artifacts"]["performance_report"]["sha256"] = "not-a-sha"
    command_sha = gate._canonical_sha(command)
    command["contract_sha256"] = command_sha
    row["full_command_contract"] = command
    row["full_command_contract_sha256"] = command_sha
    with pytest.raises(gate.GateError, match=reason):
        gate._validate_completion(
            row, gate.NATIVE_SPECS[1], indexed=indexed,
        )


def test_deepx_completion_allows_measured_latency_without_interval() -> None:
    row, indexed = _completion_row()
    row.update({
        "latency_mean_ms": 14.0,
        "latency_p50_ms": 13.5,
        "latency_p95_ms": 15.2,
        "latency_semantics": "measured_prepared_feed_request_latency",
    })
    row.pop("completion_interval_mean_ms")
    row.pop("configured_inflight")
    command = dict(row["full_command_contract"])
    command.pop("contract_sha256")
    command.update({
        "backend": "native_full_deepx",
        "backend_arg": "deepx",
        "setup_id": "orin_nx_deepx_m1_01",
        "comparison_backend": "deepx",
    })
    command_sha = gate._canonical_sha(command)
    command["contract_sha256"] = command_sha
    row["full_command_contract"] = command
    row["full_command_contract_sha256"] = command_sha
    evidence = gate._validate_completion(
        row, gate.NATIVE_SPECS[2], indexed=indexed,
    )
    assert evidence["completed_result_sha256"]


def _deepx_runtime_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    row, _indexed = _completion_row()
    completed = row["completed_task_result_artifact"]
    completed_sha = row["completed_task_result_artifact_file_sha256"]
    attestation = row["completed_task_endpoint_attestation"]
    prepared = {
        "enabled": True,
        "status": "ok",
        "benchmark_kind": "prepared_feed_dx_engine",
        "run_count": 32,
        "requested_frames": 32,
        "completed_frames": 32,
        "completed_work_units": 32,
        "completed_work_units_source": (
            "dx_engine_prepared_feed_frozen_completion_timed_loop"
        ),
        "completed_work_units_status": "exact_runtime_counter",
        "warmup_count": 4,
        "makespan_s": 1.6,
        "fps_makespan": 20.0,
        "completed_task_endpoint_attestation": attestation,
        "completed_task_result_artifact": completed,
        "completed_task_result_artifact_file_sha256": completed_sha,
    }
    outer = {
        "case_id": "full",
        "variant": "full",
        "run_id": "deepx_m1_full",
        "provider": "deepx_m1",
        "backend": "deepx_m1",
        "runtime_ok": True,
        "performance_benchmark_source": "dx_engine_prepared_feed",
        "completed_frames": 32,
        "completed_work_units": 32,
        "completed_work_units_source": (
            "dx_engine_prepared_feed_frozen_completion_timed_loop"
        ),
        "completed_work_units_status": "exact_runtime_counter",
        "fps_makespan": 20.0,
        "measured_makespan_s": 1.6,
        "deepx_prepared_feed_benchmark": prepared,
    }
    relative = Path(
        "models/yolo11l/benchmark_results/"
        "benchmark_results_deepx_m1_full_auto.json"
    )
    result_path = tmp_path / relative
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps([outer], sort_keys=True), encoding="utf-8",
    )
    summary_prepared = dict(prepared)
    summary_prepared["performance_benchmark_source"] = (
        "dx_engine_prepared_feed"
    )
    summary = {
        "result_source": result_path.name,
        "fps_makespan": 20.0,
        "measured_makespan_s": 1.6,
        "completed_frames": 32,
        "completed_work_units": 32,
        "completed_work_units_source": (
            "dx_engine_prepared_feed_frozen_completion_timed_loop"
        ),
        "completed_work_units_status": "exact_runtime_counter",
        "completed_task_endpoint_attestation": attestation,
        "completed_task_endpoint_contract_hash": (
            row["completed_task_endpoint_contract_hash"]
        ),
        "completed_task_output_endpoint_id": (
            row["completed_task_output_endpoint_id"]
        ),
        "completed_task_result_artifact": completed,
        "completed_task_result_artifact_file_sha256": completed_sha,
        "deepx_prepared_feed_benchmark": summary_prepared,
    }
    indexed = {
        relative.as_posix(): {
            "relative_path": relative.as_posix(),
            "sha256": gate.base._file_sha(result_path),
            "size_bytes": result_path.stat().st_size,
        },
    }
    return result_path, summary, indexed


def test_deepx_runtime_binds_actual_nested_prepared_feed_result(
    tmp_path: Path,
) -> None:
    result_path, summary, indexed = _deepx_runtime_fixture(tmp_path)
    identity = gate._deepx_native_runtime_result(
        tmp_path, indexed, spec=gate.NATIVE_SPECS[2], row=summary,
    )
    assert identity["sha256"] == gate.base._file_sha(result_path)

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload[0]["deepx_prepared_feed_benchmark"][
        "fps_makespan"
    ] = 21.0
    result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    indexed[next(iter(indexed))]["sha256"] = gate.base._file_sha(result_path)
    with pytest.raises(gate.GateError, match="prepared_summary_mismatch"):
        gate._deepx_native_runtime_result(
            tmp_path, indexed, spec=gate.NATIVE_SPECS[2], row=summary,
        )


def test_deepx_runtime_rejects_top_level_only_completion_evidence(
    tmp_path: Path,
) -> None:
    result_path, summary, indexed = _deepx_runtime_fixture(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    outer = payload[0]
    prepared = outer.pop("deepx_prepared_feed_benchmark")
    outer.update(prepared)
    result_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    indexed[next(iter(indexed))]["sha256"] = gate.base._file_sha(result_path)
    with pytest.raises(gate.GateError, match="result_prepared_feed"):
        gate._deepx_native_runtime_result(
            tmp_path, indexed, spec=gate.NATIVE_SPECS[2], row=summary,
        )


def test_artifact_index_requires_exact_sealed_schema_v2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate.base, "_json",
        lambda *_args, **_kwargs: {
            "schema": "onnx-splitpoint/artifact-index",
            "schema_version": 3,
        },
    )
    with pytest.raises(gate.GateError, match="exact_schema_v2"):
        gate._validate_artifact_index(Path("/unused"), {})


@pytest.mark.parametrize("hailo10_token", ["hailo10", "hailo10h"])
def test_run_identity_accepts_only_the_two_exact_hailo10_aliases(
    monkeypatch: pytest.MonkeyPatch, hailo10_token: str,
) -> None:
    matrix = {
        "hardware_targets": [
            {"id": "orin_nx_hailo8_01", "accelerator": "hailo8"},
            {"id": "orin_nx_hailo10_01", "accelerator": hailo10_token},
            {"id": "orin_nx_deepx_m1_01", "accelerator": "deepx_m1"},
        ],
    }
    monkeypatch.setattr(gate.base, "_json", lambda *_args, **_kwargs: matrix)

    def inherited(_run: Path) -> dict:
        assert gate.base.SETUPS["orin_nx_hailo10_01"] == hailo10_token
        return {"current_session_id": "session"}

    monkeypatch.setattr(gate.base, "_validate_run_identity", inherited)
    assert gate._validate_run_identity_alias_aware(Path("/unused")) == {
        "current_session_id": "session",
    }


def test_run_identity_rejects_unapproved_hailo10_like_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = {
        "hardware_targets": [
            {"id": "orin_nx_hailo8_01", "accelerator": "hailo8"},
            {"id": "orin_nx_hailo10_01", "accelerator": "hailo10p"},
            {"id": "orin_nx_deepx_m1_01", "accelerator": "deepx_m1"},
        ],
    }
    monkeypatch.setattr(gate.base, "_json", lambda *_args, **_kwargs: matrix)
    with pytest.raises(gate.GateError, match="hardware_provider_mismatch"):
        gate._validate_run_identity_alias_aware(Path("/unused"))
