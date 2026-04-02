from __future__ import annotations

import json
import sys
import types
from pathlib import Path


if "onnx" not in sys.modules:
    fake_onnx = types.ModuleType("onnx")
    fake_onnx.AttributeProto = type("AttributeProto", (), {})
    fake_onnx.helper = types.SimpleNamespace()
    sys.modules["onnx"] = fake_onnx

from onnx_splitpoint_tool.benchmark.remote_run import (
    RemoteBenchmarkArgs,
    _finalize_run_status,
    apply_remote_timeout_hint,
    estimate_remote_timeout_hint,
)
from onnx_splitpoint_tool.hailo_backend import (
    _StreamedSubprocessResult,
    _build_subprocess_detail_bundle,
    _extract_hailo_process_summary,
)


def test_remote_timeout_hint_scales_with_cases_runs_and_repeats(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    for idx in (1, 2, 3):
        case_dir = suite / f"b{idx:03d}"
        case_dir.mkdir()
        (case_dir / "split_manifest.json").write_text("{}\n", encoding="utf-8")

    (suite / "benchmark_plan.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"id": "ort_cpu", "type": "onnxruntime", "provider": "cpu"},
                    {
                        "id": "hailo8",
                        "type": "hailo",
                        "stage1": {"type": "hailo", "hw_arch": "hailo8"},
                        "stage2": {"type": "hailo", "hw_arch": "hailo8"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    args = RemoteBenchmarkArgs(repeats=2, warmup=10, iters=100, timeout_s=7200)
    hint = estimate_remote_timeout_hint(suite, args)

    assert hint["case_count"] == 3
    assert hint["planned_run_count"] == 2
    assert hint["effective_runs"] == 200
    assert hint["invocations_per_case_run"] == 4 * (10 + 200)
    assert hint["total_phase_invocations"] == 3 * 2 * 4 * (10 + 200)
    assert hint["plan_uses_hailo"] is True
    assert hint["recommended_timeout_s"] > hint["lower_bound_s"] > 0



def test_apply_remote_timeout_hint_auto_raises_legacy_default_only() -> None:
    hint = {"recommended_timeout_s": 18000}

    auto = apply_remote_timeout_hint(7200, hint)
    assert auto["effective_timeout_s"] == 18000
    assert auto["auto_raised"] is True
    assert auto["warn_too_low"] is False

    explicit = apply_remote_timeout_hint(3600, hint)
    assert explicit["effective_timeout_s"] == 3600
    assert explicit["auto_raised"] is False
    assert explicit["warn_too_low"] is True



def test_hailo_process_summary_extracts_partition_timing_and_warnings() -> None:
    stdout = """
[info] Single context flow failed: Recoverable single context error
[info] Using Multi-context flow
[info] Found valid partition to 4 contexts, Performance improved by 12.1%
[info] Partitioner finished after 327 iterations, Time it took: 26m 8s 215ms
[info] Successful Mapping (allocation time: 32m 55s)
[info] Successful Compilation (compilation time: 1m 4s)
[info] The calibration set seems to not be normalized, because the values range is [(0.0, 1.0)]
[info] \tyolov7_full/output_layer1 SNR:\t28.08 dB
[info] \tyolov7_full/output_layer2 SNR:\t26.70 dB
""".strip()

    summary = _extract_hailo_process_summary(
        stdout,
        "",
        stage_history=[
            {"stage": "translation", "t_s": 0.5},
            {"stage": "statistics_collector", "t_s": 4.0},
            {"stage": "bias_correction", "t_s": 35.0},
            {"stage": "partition_search", "t_s": 120.0},
            {"stage": "allocation", "t_s": 300.0},
            {"stage": "compile", "t_s": 400.0},
        ],
        elapsed_s=480.0,
        last_stage="compile",
    )

    assert summary["context_count"] == 4
    assert summary["partition_iterations"] == 327
    assert summary["partition_time_s"] == 1568.215
    assert summary["allocation_time_s"] == 1975.0
    assert summary["compilation_time_s"] == 64.0
    assert summary["snr_db"]["yolov7_full/output_layer1"] == 28.08
    assert summary["detected"]["normalization_warning"] is True
    assert summary["detected"]["single_context_failed"] is True
    assert summary["detected"]["multi_context_used"] is True
    assert summary["stage_history"][0]["stage"] == "translation"
    assert summary["stage_durations_s"]["compile"] == 80.0



def test_build_subprocess_detail_bundle_keeps_process_summary_without_system_snapshot() -> None:
    run = _StreamedSubprocessResult(
        returncode=124,
        stdout="[info] Using Multi-context flow",
        stderr="",
        timed_out=True,
        timeout_kind="hard",
        last_stage="allocation",
        stage_history=[{"stage": "allocation", "t_s": 1.5}],
        elapsed_s=5.0,
    )
    details = _build_subprocess_detail_bundle(run, run.stdout, run.stderr, include_system_snapshot=False)

    assert isinstance(details, dict)
    assert "process_summary" in details
    assert details["process_summary"]["last_stage"] == "allocation"
    assert "system_snapshot" not in details



def test_finalize_run_status_can_persist_progress_metadata(tmp_path: Path) -> None:
    started = "2026-03-21T10:00:00Z"
    ended = "2026-03-21T12:00:00Z"
    _finalize_run_status(
        tmp_path,
        status="failed",
        started_at=started,
        ended_at=ended,
        remote_rc=124,
        fail_message="Remote benchmark timed out",
        stdout_tail=["line-a"],
        stderr_tail=["line-b"],
        extra_fail_reason={
            "last_suite_progress": {"run_id": "ort_cpu", "case_id": "b080", "index": 7, "count": 23},
            "timeout_estimate": {"recommended_timeout_s": 18000},
        },
    )

    payload = json.loads((tmp_path / "run_status.json").read_text(encoding="utf-8"))
    fail_reason = payload["fail_reason"]
    assert fail_reason["last_suite_progress"]["case_id"] == "b080"
    assert fail_reason["timeout_estimate"]["recommended_timeout_s"] == 18000
