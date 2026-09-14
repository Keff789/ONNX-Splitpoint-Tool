from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.build_scheduler import semantic_result_status
from onnx_splitpoint_tool.hailo_attempt_receipts import (
    begin_hailo_attempt,
    finalize_hailo_attempt,
)
from onnx_splitpoint_tool.workflow.phase_eta import PhaseEtaEstimator


def test_returned_failure_is_not_scheduler_ok() -> None:
    assert semantic_result_status({"ok": False, "error": "compile failed"}) == "failed"
    assert semantic_result_status({"timed_out": True}) == "timeout"
    assert semantic_result_status({"unsupported_reason": "op"}) == "unsupported"
    assert semantic_result_status({"ok": True}) == "success"


def test_phase_eta_has_warmup_and_range_not_false_precision() -> None:
    eta = PhaseEtaEstimator(phase="quality", warmup_completions=3, parallelism=4)
    eta.set_baseline(342, now=0.0)
    for idx, duration in enumerate((100.0, 120.0), start=1):
        eta.observe_completion("detection", now=float(idx), duration_s=duration)
    assert eta.estimate(remaining_by_cohort={"detection": 10})["display"] == "ETA=UNAVAILABLE"
    eta.observe_completion("detection", now=3.0, duration_s=110.0)
    estimate = eta.estimate(remaining_by_cohort={"detection": 10})
    assert estimate["status"] == "available"
    assert estimate["lower_s"] < estimate["upper_s"]
    assert estimate["parallelism"] == 4
    assert estimate["display"].startswith("ETA=") and "-" in estimate["display"]


def test_hailo_attempt_receipts_are_immutable_and_terminal_selects_failure(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    first = begin_hailo_attempt(outdir=tmp_path, bound={
        "onnx_path": str(model), "hw_arch": "hailo8", "endpoint": "decoded_full",
    })
    first_path = finalize_hailo_attempt(attempt=first, result={"ok": False, "error": "parser"})
    second = begin_hailo_attempt(outdir=tmp_path, bound={
        "onnx_path": str(model), "hw_arch": "hailo8", "endpoint": "raw_head_fallback",
    })
    second_path = finalize_hailo_attempt(attempt=second, result={
        "ok": False, "timed_out": True, "timeout_kind": "hard_timeout",
        "last_stage": "compile_prep", "error": "timeout",
    })
    assert first_path.is_file() and second_path.is_file() and first_path != second_path
    terminal = json.loads((tmp_path / "hailo_attempt_receipts" / "terminal_attempt.json").read_text())
    assert terminal["semantic_status"] == "timeout"
    assert terminal["endpoint"] == "raw_head_fallback"
    assert terminal["last_active_stage"] == "compile_prep"
    assert terminal["immutable_receipt"] == str(second_path)


def test_hailo_service_binding_retains_all_attempts_and_selects_actual_last(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.workflow.hailo_remote_binding import (
        _attempt_for,
        _collect_hailo_build_attempts,
    )

    full = tmp_path / "hailo" / "hailo8" / "full"
    receipts = full / "hailo_attempt_receipts"
    receipts.mkdir(parents=True)
    parser = {
        "schema": "onnx-splitpoint/hailo-build-attempt-receipt",
        "schema_version": 1,
        "attempt_id": "decoded",
        "hw_arch": "hailo8",
        "endpoint": "decoded_full",
        "started_at_epoch_s": 1.0,
        "ended_at_epoch_s": 2.0,
        "invocation_status": "returned",
        "semantic_status": "failed",
        "failure_kind": "parser_error",
        "compiler_phase": "parse",
        "end_nodes": [],
        "source_onnx_sha256": "a" * 64,
    }
    timeout = {
        "schema": "onnx-splitpoint/hailo-build-attempt-receipt",
        "schema_version": 1,
        "attempt_id": "raw",
        "hw_arch": "hailo8",
        "endpoint": "raw_head_fallback",
        "started_at_epoch_s": 3.0,
        "ended_at_epoch_s": 4.0,
        "invocation_status": "returned",
        "semantic_status": "timeout",
        "timed_out": True,
        "timeout_kind": "hard_timeout",
        "compiler_phase": "compile_prep",
        "end_nodes": ["head0", "head1", "head2"],
        "source_onnx_sha256": "a" * 64,
    }
    (receipts / "attempt_decoded.json").write_text(json.dumps(parser))
    (receipts / "attempt_raw.json").write_text(json.dumps(timeout))
    # The legacy single-result file must not double-count the latest attempt.
    (full / "hailo_hef_build_result.json").write_text(json.dumps({
        "ok": False, "timed_out": True, "last_stage": "compile_prep",
    }))

    attempts = _collect_hailo_build_attempts(tmp_path, tmp_path)
    assert len(attempts) == 2
    assert [row["endpoint"] for row in attempts] == [
        "decoded_full", "raw_head_fallback",
    ]
    selected = _attempt_for(
        attempts, backend="hailo8", variant="full", case_id="full",
    )
    assert selected["attempt_id"] == "raw"
    assert selected["status"] == "attempted_timeout"
    assert selected["compiler_phase"] == "compile_prep"
    assert selected["end_nodes"] == ["head0", "head1", "head2"]


def test_hailo_hard_timeout_can_be_explicitly_disabled(monkeypatch) -> None:
    from onnx_splitpoint_tool.hailo_backend import _resolve_hef_timeout_policy
    from onnx_splitpoint_tool.runners.backends.hailo_backend import _timeout_seconds

    monkeypatch.delenv("ONNX_SPLITPOINT_HAILO_HEF_TIMEOUT_S", raising=False)
    monkeypatch.delenv("OSP_HAILO_HARD_TIMEOUT_S", raising=False)
    assert _resolve_hef_timeout_policy(0)[0] == 0
    assert _timeout_seconds("off", 9000) == 0
    assert _timeout_seconds("none", 9000) == 0
    assert _timeout_seconds(9000, 1800) == 9000


def test_hailo_attempt_start_receipt_and_out_of_order_terminal_selection(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    first = begin_hailo_attempt(outdir=tmp_path, bound={
        "onnx_path": str(model), "endpoint": "decoded_full",
    })
    second = begin_hailo_attempt(outdir=tmp_path, bound={
        "onnx_path": str(model), "endpoint": "raw_head_fallback",
    })
    receipts = tmp_path / "hailo_attempt_receipts"
    first_started = receipts / f"attempt_{first['attempt_id']}.started.json"
    second_started = receipts / f"attempt_{second['attempt_id']}.started.json"
    assert first_started.is_file() and second_started.is_file()
    before = first_started.read_bytes()
    # The attempt that actually finishes last must become terminal, regardless
    # of which one started first.
    finalize_hailo_attempt(attempt=second, result={"ok": True})
    first_path = finalize_hailo_attempt(attempt=first, result={
        "ok": False, "failure_kind": "parser_error", "last_stage": "parse",
    })
    terminal = json.loads((receipts / "terminal_attempt.json").read_text())
    assert terminal["attempt_id"] == first["attempt_id"]
    assert terminal["semantic_status"] == "failed"
    assert terminal["immutable_receipt"] == str(first_path)
    assert first_started.read_bytes() == before
    assert json.loads(first_started.read_text())["terminal"] is False


def test_hailo_service_collector_ignores_start_and_heartbeat_receipts(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.workflow.hailo_remote_binding import _collect_hailo_build_attempts

    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    attempt = begin_hailo_attempt(outdir=tmp_path, bound={
        "onnx_path": str(model), "hw_arch": "hailo8", "endpoint": "decoded_full",
    })
    from onnx_splitpoint_tool.hailo_attempt_receipts import HailoAttempt
    HailoAttempt(dict(attempt)).heartbeat(compiler_phase="parse", detail="active")
    finalize_hailo_attempt(attempt=attempt, result={"ok": True})
    rows = _collect_hailo_build_attempts(tmp_path, tmp_path)
    assert len(rows) == 1
    assert rows[0]["semantic_status"] == "success"


def test_hailo_full_timeout_explicit_flag_is_wired_through_generation() -> None:
    from dataclasses import fields
    from onnx_splitpoint_tool.benchmark.services import (
        BenchmarkGenerationExecutionConfig, BenchmarkGenerationOrchestrationConfig,
    )

    assert "hailo_full_timeout_explicit" in {field.name for field in fields(BenchmarkGenerationExecutionConfig)}
    assert "hailo_full_timeout_explicit" in {field.name for field in fields(BenchmarkGenerationOrchestrationConfig)}
    root = Path(__file__).resolve().parents[1]
    legacy = (root / "onnx_splitpoint_tool/workflow/legacy_benchmarkset_binding.py").read_text(encoding="utf-8")
    services = (root / "onnx_splitpoint_tool/benchmark/services.py").read_text(encoding="utf-8")
    assert 'hailo_full_timeout_explicit = "cold_build_timeout_s" in hailo_build_cfg' in legacy
    assert 'if bool(getattr(cfg, \"hailo_full_timeout_explicit\", False))' in services
