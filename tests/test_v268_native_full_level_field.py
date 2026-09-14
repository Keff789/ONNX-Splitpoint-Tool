from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_trtexec_iteration_mode_is_explicit_and_export_times_is_exact(tmp_path: Path) -> None:
    mod = _load("v268_native_trt", "scripts/native_trt_from_benchmarkset.py")
    cmd = mod._engine_run_cmd("trtexec", tmp_path / "m.engine", "", 7, 0, 0, [])
    assert "--iterations=7" in cmd
    assert "--duration=0" in cmd
    assert "--warmUp=0" in cmd

    evidence_path = tmp_path / "times.json"
    evidence_path.write_text(json.dumps([
        {"latencyMs": 1.0}, {"latencyMs": 2.0}, {"latencyMs": 9.0},
    ]), encoding="utf-8")
    evidence = mod._trtexec_export_times_evidence(evidence_path)
    assert evidence["status"] == "ok"
    assert evidence["completed_work_units"] == 3
    assert evidence["latency_mean_ms"] == 4.0
    assert evidence["latency_p50_ms"] == 2.0


def test_full_trt_forwards_exact_frame_warmup_and_reuses_engine(monkeypatch, tmp_path: Path) -> None:
    mod = _load("v268_full_trt", "scripts/native_full_baseline_eval_runner.py")
    captured: list[list[str]] = []
    engine = tmp_path / "native_trt" / "full" / "fp16" / "full_fp16.engine"
    producer_sha = "a" * 64
    producer = {
        "producer_identity_sha256": producer_sha,
        "setup_id": "setup-a", "task": "classification",
        "source_onnx": {"sha256": "b" * 64},
        "build_onnx": {"sha256": "b" * 64},
        "engine": {"path": str(engine), "sha256": "c" * 64},
        "trtexec": {"sha256": "d" * 64},
        "engine_build_receipt": {
            "sha256": "e" * 64,
            "receipt": {"receipt_sha256": "f" * 64},
        },
        "engine_build_receipt_file_sha256": "1" * 64,
        "endpoint_contract_hash": "2" * 64,
        "endpoint_contract_complete": True,
    }
    monkeypatch.setattr(
        mod, "_quality_first_trt_producer_identity",
        lambda *_args, **_kwargs: (producer, "quality_first_identity_verified_exact", tmp_path / "producer.json"),
    )
    monkeypatch.setattr(
        mod, "_trt_quality_identity_cli_args",
        lambda *_args, **_kwargs: ["--explicit-full-engine", str(engine)],
    )

    def fake_run(cmd, **_kwargs):
        captured.append(list(cmd))
        out = tmp_path / "native_trt" / "full" / "fp16"
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_trtexec.log").write_text("Throughput: 42 qps\n", encoding="utf-8")
        (out / "native_trt_meta.json").write_text(json.dumps({
            "build_ok": True, "run_ok": True,
            "completed_work_units": 100,
            "completed_work_units_source": "trtexec_export_times",
            "completed_work_units_status": "exact_runtime_counter",
            "warmup_policy": "separate_exact_iteration_invocation",
            "warmup_iterations_requested": 10,
            "latency_mean_ms": 3.0, "latency_p50_ms": 2.8, "latency_p95_ms": 4.2,
            "quality_first_identity": {
                "quality_first_producer_identity_sha256": producer_sha,
                "paths": {"engine": str(engine)},
                "hashes": {"engine": producer["engine"]["sha256"]},
            },
        }), encoding="utf-8")
        return {"rc": 0, "returncode": 0, "timed_out": False}

    monkeypatch.setattr(mod, "_run", fake_run)
    ns = SimpleNamespace(
        engine_python_selected=sys.executable, trt_precision="fp16", frames=100,
        warmup=10, workspace_mb=1024, no_shapes=False, timeout=30,
        out_dir=str(tmp_path), duration_s=0.0, full_repetition_index=2,
    )
    row = mod._native_trt_full(tmp_path / "benchmark_set", "m", ns)
    command = captured[0]
    assert command[command.index("--warmup-iterations") + 1] == "10"
    assert command[command.index("--warmup-ms") + 1] == "0"
    assert command[command.index("--duration-s") + 1] == "0"
    assert "--no-build" in command
    assert row["ok"] is True
    assert row["completed_work_units"] == 100
    assert row["latency_p95_ms"] == 4.2


def test_full_repetitions_use_median_and_not_fastest_result() -> None:
    mod = _load("v268_full_repeats", "scripts/native_full_baseline_eval_runner.py")
    rows = [
        {"ok": True, "fps_makespan": fps, "latency_mean_ms": latency,
         "backend": "native_full_deepx", "model": "m", "case": "full",
         "execution_precision": "int8", "repetition_index": index,
         "run_id": "deepx_m1_full",
         "prepared_feed_contract_version": "deepx-sealed-runtime-input-v3",
         "prepared_input_sha256": "a" * 64,
         "prepared_input_bytes": 12,
         "prepared_input_name": "images",
         "prepared_input_shape": [2, 2, 3],
         "prepared_input_dtype": "uint8",
         "prepared_input_layout": "HWC",
         "prepared_input_source_image_id": "image.jpg",
         "prepared_input_source_image_sha256": "b" * 64,
         "runtime_preprocessing_sha256": "c" * 64,
         "runtime_numeric_input_sha256": "d" * 64}
        for index, (fps, latency) in enumerate(((10.0, 100.0), (20.0, 50.0), (90.0, 11.0)), 1)
    ]
    result = mod._aggregate_full_repetitions(rows, requested=3)
    assert result["ok"] is True
    assert result["fps_makespan"] == 20.0
    assert result["fps_makespan"] != 90.0
    assert result["latency_mean_ms"] == 50.0
    assert result["repetition_count_valid"] == 3
    assert result["fps_ci95_low"] is not None


def test_failed_full_repetition_preserves_concrete_child_diagnostics() -> None:
    mod = _load(
        "v27513_full_repeat_failure",
        "scripts/native_full_baseline_eval_runner.py",
    )
    failed = {
        "ok": False,
        "status": "failed",
        "failure_reason": "prepared_input_manifest_mismatch",
        "status_detail": "sealed image sha256 differs",
        "error": "ValueError: image identity mismatch",
        "returncode": 4,
        "timed_out": False,
        "stdout_tail": "child stdout",
        "stderr_tail": "child stderr",
        "report": "/tmp/native-full-report.json",
        "backend": "native_full_deepx",
        "model": "m",
        "case": "full",
        "execution_precision": "int8",
        "repetition_index": 1,
    }

    result = mod._aggregate_full_repetitions([failed], requested=3)

    assert result["ok"] is False
    assert result["failure_reason"] == "native_full_repetition_set_incomplete"
    assert result["primary_repetition_failure_reason"] == (
        "prepared_input_manifest_mismatch"
    )
    assert result["primary_repetition_status_detail"] == (
        "sealed image sha256 differs"
    )
    assert result["primary_repetition_error"].startswith("ValueError")
    concrete = result["primary_repetition_failure"]
    assert concrete["returncode"] == 4
    assert concrete["stderr_tail"] == "child stderr"
    assert result["repetition_records"][0]["status_detail"] == (
        "sealed image sha256 differs"
    )
    assert result["repetition_records"][0]["error"].startswith("ValueError")


def test_full_repetition_does_not_invent_missing_execution_receipt() -> None:
    mod = _load(
        "v27513_full_repeat_missing_receipt",
        "scripts/native_full_baseline_eval_runner.py",
    )
    failed = {
        "ok": False,
        "status": "failed",
        "failure_reason": "child_receipt_incomplete",
        "backend": "native_full_deepx",
        "model": "m",
        "case": "full",
        "execution_precision": "int8",
        "repetition_index": 1,
    }

    result = mod._aggregate_full_repetitions([failed], requested=1)

    assert result["primary_repetition_failure"]["returncode"] is None
    assert result["primary_repetition_failure"]["timed_out"] is None
    assert result["repetition_records"][0]["returncode"] is None
    assert result["repetition_records"][0]["timed_out"] is None


def test_deepx_template_has_outer_makespan_and_separate_latency_statistics() -> None:
    text = (ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt").read_text(encoding="utf-8")
    for token in (
        "makespan_start = perf()", "makespan_end = perf()",
        '"fps_makespan": fps_makespan', '"latency_semantics": "synchronous_request_end_to_end"',
        '"latency_p50_ms": prepared_bench.get("p50_ms")',
        '"throughput_primary_source": ("dx_engine_prepared_feed_outer_makespan"',
    ):
        assert token in text


def test_deepx_full_does_not_cap_frames_or_warmup() -> None:
    text = (ROOT / "scripts/native_full_baseline_eval_runner.py").read_text(encoding="utf-8")
    assert "max(1, int(ns.frames))" in text
    assert 'str(max(0, int(ns.warmup)))' in text
    assert "min(ns.warmup, 32)" not in text


def test_final_report_aggregates_repeats_and_fails_closed_on_endpoint_quality() -> None:
    mod = _load("v268_final_report", "scripts/native_producer_final_report.py")
    endpoint_hash = "a" * 64
    base = {
        "backend": "native_full_tensorrt", "producer_impl": "trt", "model": "m",
        "case": "full", "execution_mode": "native_full_baseline", "precision": "p",
        "setup_id": "s", "comparison_backend": "vendor", "execution_precision": "fp16",
        "task": "classification", "stage": "classification_logits",
        "contract_family": "classification_logits",
        "output_format": "classification_logits", "ok": True,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "full_command_contract_sha256": "d" * 64,
        "output_endpoint_attestation": {
            "attested": True, "status": "passed",
            "task": "classification",
            "stage": "classification_logits",
            "endpoint": "classification_logits",
            "endpoint_contract_hash": endpoint_hash,
        },
    }
    aggregated = mod._dedupe([
        {**base, "fps_makespan": fps, "report": f"r{index}"}
        for index, fps in enumerate((10.0, 20.0, 90.0), 1)
    ])
    assert len(aggregated) == 1
    assert aggregated[0]["fps_makespan"] == 20.0

    vendor = {
        **base, "backend": "native_full_hailo10h", "producer_impl": "h10",
        "fps_makespan": 30.0, "claim_ok": True,
        "semantic_ok": True, "contract_consistent": True,
    }
    trt = {
        **base, "fps_makespan": 40.0, "claim_ok": True,
        "semantic_ok": True, "contract_consistent": True,
    }
    for row in (vendor, trt):
        contract = {
            "schema": "onnx-splitpoint/native-full-command-contract",
            "schema_version": 1, "complete": True,
            "backend": row["backend"], "model": row["model"],
            "case": row["case"], "setup_id": row["setup_id"],
            "comparison_backend": row["comparison_backend"],
            "runner_sha256": "c" * 64, "input_image_sha256": "d" * 64,
            "artifacts": {"runtime_input_tensor": {"sha256": "e" * 64}},
            "runtime_options": {}, "energy_workload": {"available": True},
        }
        contract["contract_sha256"] = hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        row["full_command_contract"] = contract
        row["full_command_contract_sha256"] = contract["contract_sha256"]
    for row, samples, prefix in (
        (vendor, [29.0, 30.0, 31.0], "vendor"),
        (trt, [39.0, 40.0, 41.0], "trt"),
    ):
        row.update({
            "repetition_count_requested": 3,
            "repetition_count_attempted": 3,
            "repetition_count_valid": 3,
            "repetition_status": "complete",
            "repetition_aggregation": "median_with_deterministic_percentile_bootstrap_ci95",
            "repetition_runtime_scope": "fresh_runtime_per_repetition",
            "repetition_independence_verified": True,
            "repetition_records": [
                {"repetition_index": index, "runtime_instance_id": f"{prefix}-{index}", "ok": True, "status": "ok", "fps_makespan": fps, "completed_work_units": 100, "workload_contract_sha256": "c" * 64}
                for index, fps in enumerate(samples, 1)
            ],
            "fps_repetition_samples": samples,
            "fps_median": samples[1],
            "fps_ci95_low": samples[0],
            "fps_ci95_high": samples[-1],
        })
    quality_rows = []
    for row in (vendor, trt):
            quality_rows.append({
                **{key: row.get(key, "") for key in mod._QUALITY_IDENTITY_FIELDS},
                "full_command_contract_sha256": row["full_command_contract_sha256"],
                "source_request_sha256": "1" * 64,
                "model_sha256": "2" * 64,
                "validation_dataset_sha256": "3" * 64,
                "validation_dataset_image_ids_sha256": "4" * 64,
                "validation_dataset_ground_truth_sha256": "5" * 64,
                "accuracy_gate_policy_sha256": "6" * 64,
                "task_quality_policy_sha256": "6" * 64,
                "runtime_quality_gate_policy_sha256": "6" * 64,
                "quality_contract_sha256": "7" * 64,
                "preprocessing_contract_sha256": "8" * 64,
                "decoder_contract_sha256": "9" * 64,
                "nms_contract_sha256": "b" * 64,
                "central_quality_evidence_verified": True,
                "precision_quality_verified": True,
                "ok": True, "claim_ok": True, "status": "claim_ok",
            "gate_status": "eligible", "contract_consistent": True, "task_valid": True,
            "accuracy_gate_pass": True, "eligible_for_ranking": True,
            "accuracy_gate_policy_match": True,
        })
    quality_summary = {
        "schema": "onnx-splitpoint/native-producer-validation-summary",
        "schema_version": 5, "status": "complete",
        "row_count": len(quality_rows), "rows": quality_rows,
    }
    verified, _ = mod._attach_quality_evidence([vendor, trt], quality_summary)
    gated = mod._apply_comparison_claim_gates(verified)
    assert all(row["output_endpoint_match"] for row in gated)
    # Legacy v5 evidence remains readable, but a Native-Full TensorRT row has
    # no setup-local Quality-FIRST producer identity and is therefore no longer
    # scientifically claimable.  The vendor row does not receive this
    # TensorRT-specific exclusion (its deliberately minimal legacy command
    # fixture can still fail other modern claim gates).
    assert next(
        row for row in gated if row["backend"] == "native_full_hailo10h"
    ).get("quality_first_binding_status") in {None, ""}
    legacy_trt = next(
        row for row in gated if row["backend"] == "native_full_tensorrt"
    )
    assert legacy_trt["performance_claim_eligible"] is False
    assert legacy_trt["quality_first_binding_status"] == "legacy_unbound_nonclaimable"
    assert "precision_variant_quality_not_verified" in legacy_trt[
        "performance_claim_exclusion_reasons"
    ]

    mismatched_rows = [vendor, {
        **trt, "stage": "raw_head", "contract_family": "raw_head",
        "output_format": "raw_heads", "endpoint_contract_hash": "b" * 64,
        "output_endpoint_attestation": {
            "attested": True, "status": "passed",
            "stage": "raw_head", "endpoint": "raw_head",
            "endpoint_contract_hash": "b" * 64,
        },
    }]
    mismatched_rows, _ = mod._attach_quality_evidence(mismatched_rows, quality_summary)
    mismatched = mod._apply_comparison_claim_gates(mismatched_rows)
    assert not any(row["performance_claim_eligible"] for row in mismatched)
    assert all("output_endpoint_not_common_across_backends" in row["performance_claim_exclusion_reasons"] for row in mismatched)


def test_final_report_ingests_split_agent_repeat_aliases_without_losing_raw_evidence() -> None:
    mod = _load("v268_final_report_aliases", "scripts/native_producer_final_report.py")
    evidence = [
        {"repetition_index": index, "ok": True, "fps_makespan": fps}
        for index, fps in enumerate((10.0, 20.0, 30.0), 1)
    ]
    fields = mod._repeat_fields({
        "repetition_evidence": evidence,
        "repetitions_requested": 3,
        "repetitions_completed": 3,
        "fps_makespan_median": 20.0,
        "fps_makespan_ci95_low": 10.0,
        "fps_makespan_ci95_high": 30.0,
    })
    assert fields["repetition_records"] == evidence
    assert fields["performance_repetitions"] == evidence
    assert fields["fps_repetition_samples"] == [10.0, 20.0, 30.0]
    assert fields["fps_median"] == 20.0
    assert fields["fps_ci95_low"] == 10.0
    assert fields["repetition_count_requested"] == 3
    assert fields["repetition_count_valid"] == 3


def test_hailo_throughput_is_not_reported_as_latency() -> None:
    text = (ROOT / "scripts/native_full_baseline_eval_runner.py").read_text(encoding="utf-8")
    assert '"latency_mean_ms": None' in text
    assert '"completion_interval_mean_ms": (1000.0 / fps) if fps else None' in text
    assert '"latency_semantics": "not_measured_async_or_streaming_throughput"' in text
