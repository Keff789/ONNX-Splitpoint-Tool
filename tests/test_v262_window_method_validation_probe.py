from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from onnx_splitpoint_tool.energy.config import EnergyDefaults
from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract
from onnx_splitpoint_tool.window_method_validation_probe import (
    MIN_DECISION_REPEATS,
    _repeat_evidence,
    main as probe_main,
)
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _summary(path: Path, rows: list[dict]) -> Path:
    return _write_json(path, {"rows": rows, "row_count": len(rows)})


def _sealed_hailo8_contract() -> dict:
    return seal_native_command_contract({
        "complete": True,
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "precision": "float32_layout_fp16",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "runner": "scripts/native_hailo_trt_fifo_from_benchmarkset.py",
        "runner_sha256": "a" * 64,
        "python_executable": "/home/nx/venv/bin/python",
        "interpreter_identity": {
            "executable": "/home/nx/venv/bin/python",
            "resolved_executable": "/home/nx/venv/bin/python3.8",
            "executable_sha256": "1" * 64,
        },
        "benchmark_set": "/home/nx/native_fifo_evalsets/complete_set_20260626_194014/resnet50/benchmark_set",
        "input_image": "/home/nx/dataset/resnet50.jpg",
        "input_image_sha256": "b" * 64,
        "artifacts": {
            "python_executable": {"path": "/home/nx/venv/bin/python", "sha256": "1" * 64},
            "hef": {"path": "/home/nx/part1.hef", "sha256": "2" * 64},
            "engine": {"path": "/home/nx/part2.engine", "sha256": "3" * 64},
            "native_executable": {"path": "/home/nx/native_fifo", "sha256": "4" * 64},
            "generated_cpp": {"path": "/home/nx/main.cpp", "sha256": "5" * 64},
            "cmake": {"path": "/home/nx/CMakeLists.txt", "sha256": "6" * 64},
            "prepared_input": {"path": "/home/nx/prepared.rgb", "sha256": "7" * 64},
        },
        "runtime_options": {
            "frames": 100, "duration_s": 0.0, "warmup": 10, "queue_depth": 3,
            "hailo_format": "uint8", "letterbox_pad_value": 0,
            "copy_outputs": True, "dump_outputs": False, "dump_boundary": False,
            "device_id": "", "build": False,
            "energy_prepared_feed_capable": True,
            "prepared_input_bound": True,
            "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 0,
            "letterbox_pad_value_effective": 0,
        },
        "prepared_input_contract": {
            "format": "raw_rgb_uint8", "shape": [224, 224, 3],
            "dtype": "uint8", "layout": "HWC", "task": "classification",
            "preprocess_mode_requested": "auto",
            "preprocess_mode_effective": "resize",
            "letterbox_pad_value_requested": 0,
            "letterbox_pad_value_effective": 0,
            "letterbox_pad_value": 0,
            "pad_value_effective": 0,
        },
        "boundary_contract": {
            "boundary_layout_requested": "as_input",
            "boundary_layout_effective": "as_input",
        },
    })


def test_probe_defaults_are_explicit_and_decision_capable() -> None:
    defaults = EnergyDefaults()
    assert defaults.window_method_validation_probe_enabled is True
    assert defaults.window_method_validation_probe_repeats == MIN_DECISION_REPEATS == 3
    assert defaults.window_method_validation_probe_include_raw_parquet is True
    assert defaults.window_method_validation_probe_strict is True


def test_screening_plan_selects_exactly_one_successful_native_target(tmp_path: Path) -> None:
    summary = _summary(tmp_path / "summary.json", [
        {
            "ok": True, "backend": "native_full_tensorrt", "model": "resnet50",
            "case": "full", "precision": "fp16", "comparison_backend": "hailo8",
            "setup_id": "orin_nx_hailo8_01", "fps_makespan": 100.0,
        },
        {
            "ok": True, "backend": "hailo8_to_trt", "model": "resnet50",
            "case": "b052", "precision": "float32_layout_fp16",
            "setup_id": "orin_nx_hailo8_01", "comparison_backend": "hailo8",
            "fps_makespan": 80.0, "native_command_contract": _sealed_hailo8_contract(),
        },
    ])
    out = tmp_path / "probe" / "plan"
    command = [
        sys.executable, str(ROOT / "scripts" / "native_producer_energy_plan.py"),
        "--summary", str(summary), "--out-dir", str(out),
        "--screening-window-probe", "--allow-unpaired",
        "--hailo8-ssh", "nx@example.invalid", "--duration-s", "1",
        "--runs", "3",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads((out / "native_producer_energy_plan.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "onnx-splitpoint/window-method-validation-probe-plan"
    assert payload["pairing_policy"] == "screening_probe_one_successful_native_target_no_pairing"
    assert payload["paired_only"] is False
    assert payload["diagnostic_only"] is True
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["backend"] == "hailo8_to_trt"
    assert row["screening_only"] is True
    assert row["claim_ok"] is False
    assert row["eligible_for_energy_results_import"] is False
    assert "--compare-legacy-window" in row["measure_command"]
    assert "--runs 3" in row["measure_command"]
    assert str((out.parent / "measurement").resolve()) in row["measure_command"]


def test_screening_plan_without_sealed_runtime_contract_fails_closed(tmp_path: Path) -> None:
    summary = _summary(tmp_path / "summary.json", [{
        "ok": True, "backend": "hailo8_to_trt", "model": "resnet50",
        "case": "b052", "precision": "float32_layout_fp16",
        "setup_id": "orin_nx_hailo8_01", "comparison_backend": "hailo8",
        "fps_makespan": 80.0,
    }])
    out = tmp_path / "probe" / "plan"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "native_producer_energy_plan.py"),
        "--summary", str(summary), "--out-dir", str(out),
        "--screening-window-probe", "--allow-unpaired",
        "--hailo8-ssh", "nx@example.invalid", "--duration-s", "1",
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads((out / "native_producer_energy_plan.json").read_text(encoding="utf-8"))
    assert payload["rows"] == []
    assert payload["semantically_admitted_rows"] == 0
    assert payload["excluded_rows"][0]["reason"] == "successful_command_contract_missing_or_invalid"
    assert payload["excluded_rows"][0]["command_contract_status"] == "native_command_contract_missing"


def test_requested_probe_without_runtime_target_is_blocked_and_nonzero(tmp_path: Path) -> None:
    summary = _summary(tmp_path / "summary.json", [])
    out = tmp_path / "probe"
    rc = probe_main([
        "--summary", str(summary), "--out-dir", str(out),
        "--strict", "--repeats", "3",
    ])
    payload = json.loads((out / "window_method_validation_probe.json").read_text(encoding="utf-8"))
    assert rc != 0
    assert payload["ok"] is False
    assert payload["complete"] is False
    assert payload["status"] == "blocked_no_successful_representative_native_target"
    assert payload["started_repeat_count"] == 0
    assert payload["eligible_for_energy_results_import"] is False
    artifact_index = json.loads((out / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["schema"] == "onnx-splitpoint/window-method-validation-probe-artifact-index"
    assert all(row.get("sha256") for row in artifact_index["files"])


def test_requested_native_energy_without_pairs_is_not_reported_complete(tmp_path: Path) -> None:
    summary = _summary(tmp_path / "summary.json", [])
    out = tmp_path / "native_energy"
    command = [
        sys.executable, str(ROOT / "scripts" / "run_native_producer_energy_from_summary.py"),
        "--summary", str(summary), "--out-dir", str(out),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    payload = json.loads((out / "native_producer_energy_results.json").read_text(encoding="utf-8"))
    assert result.returncode != 0
    assert payload["ok"] is False
    assert payload["complete"] is False
    assert payload["status"] == "blocked_no_runtime_constructible_rows"
    assert payload["started_measurement_count"] == 0


def test_repeat_evidence_requires_same_hash_verified_trace(tmp_path: Path) -> None:
    measurement = tmp_path / "measurement"
    run = measurement / "run_000"
    storage = run / "collector_storage"
    storage.mkdir(parents=True)
    trace = storage / "fast_firmware.parquet"
    trace.write_bytes(b"PAR1" + b"parquet-test-trace" + b"PAR1")
    import hashlib
    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    _write_json(run / "window_method_comparison.json", {
        "status": "ok",
        "same_raw_trace_verified": True,
        "raw_trace": {
            "trace_path": str(trace),
            "request_sha256": digest,
            "sha256_before_legacy_postprocess": digest,
            "sha256_after_legacy_postprocess": digest,
        },
        "legacy_minus_command_window": {"energy_j": {"relative_percent": 0.1}},
    })
    evidence = _repeat_evidence(measurement, {"runs": [{"run_index": 0, "started_at": 1, "collector_rc": 0}]})
    assert len(evidence) == 1
    assert evidence[0]["same_raw_trace_verified"] is True
    assert evidence[0]["trace_sha256"] == digest
    assert evidence[0]["raw_parquet_present"] is True
    assert evidence[0]["raw_parquet"][0]["sha256"] == digest


def test_repeat_evidence_rejects_multiple_or_unrelated_parquet_traces(tmp_path: Path) -> None:
    import hashlib

    measurement = tmp_path / "measurement"
    run = measurement / "run_000"
    storage = run / "collector_storage"
    storage.mkdir(parents=True)
    trace = storage / "requested.parquet"
    trace.write_bytes(b"PAR1" + b"requested" + b"PAR1")
    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    comparison = {
        "status": "ok",
        "same_raw_trace_verified": True,
        "raw_trace": {
            "trace_path": str(trace),
            "request_sha256": digest,
            "sha256_before_legacy_postprocess": digest,
            "sha256_after_legacy_postprocess": digest,
        },
    }
    _write_json(run / "window_method_comparison.json", comparison)
    aggregate = {"runs": [{"run_index": 0, "started_at": 1, "collector_rc": 0}]}

    extra = storage / "stale.parquet"
    extra.write_bytes(b"stale")
    evidence = _repeat_evidence(measurement, aggregate)[0]
    assert evidence["raw_parquet_count"] == 2
    assert evidence["raw_parquet_exactly_one"] is False
    assert evidence["raw_parquet_present"] is False
    assert evidence["same_raw_trace_verified"] is False

    extra.unlink()
    comparison["raw_trace"]["trace_path"] = str(trace)
    comparison["raw_trace"]["sha256_after_legacy_postprocess"] = ""
    _write_json(run / "window_method_comparison.json", comparison)
    evidence = _repeat_evidence(measurement, aggregate)[0]
    assert evidence["raw_parquet_matches_comparison_trace_path"] is True
    assert evidence["request_before_after_hash_chain_complete"] is False
    assert evidence["same_raw_trace_verified"] is False

    outside = run / "unrelated.parquet"
    outside.write_bytes(trace.read_bytes())
    comparison["raw_trace"]["trace_path"] = str(outside)
    _write_json(run / "window_method_comparison.json", comparison)
    evidence = _repeat_evidence(measurement, aggregate)[0]
    assert evidence["raw_parquet_count"] == 1
    assert evidence["raw_parquet_matches_comparison_trace_path"] is False
    assert evidence["raw_parquet_present"] is False
    assert evidence["same_raw_trace_verified"] is False


def test_probe_invocations_use_fresh_attempt_directories(tmp_path: Path) -> None:
    summary = _summary(tmp_path / "summary.json", [])
    out = tmp_path / "probe"
    assert probe_main(["--summary", str(summary), "--out-dir", str(out), "--strict"]) != 0
    first = json.loads((out / "window_method_validation_probe.json").read_text(encoding="utf-8"))
    assert probe_main(["--summary", str(summary), "--out-dir", str(out), "--strict"]) != 0
    second = json.loads((out / "window_method_validation_probe.json").read_text(encoding="utf-8"))

    assert first["attempt_id"] != second["attempt_id"]
    assert Path(first["attempt_directory"]).is_dir()
    assert Path(second["attempt_directory"]).is_dir()
    assert second["stale_measurement_reuse_allowed"] is False


def test_profile_resolution_materializes_probe_in_profile_snapshot(tmp_path: Path) -> None:
    options = WorkflowOptions(profile="unused", out=str(tmp_path))
    options.window_method_validation_probe_enabled = True
    options.window_method_validation_probe_repeats = 2
    options.window_method_validation_probe_include_raw_parquet = False
    options.window_method_validation_probe_strict = True
    runner = EvaluationWorkflowRunner(options)
    runner.profile_payload = {
        "native_producers": {"enabled": True, "energy": {"enabled": True, "mode": "measure"}}
    }
    runner._materialize_window_method_probe_profile()
    probe = runner.profile_payload["native_producers"]["energy"]["window_method_validation_probe"]
    assert probe["enabled"] is True
    assert probe["repeats"] == 2
    assert probe["decision_capable_repeat_count"] is False
    assert probe["include_raw_parquet"] is False
    assert probe["eligible_for_energy_results_import"] is False


def test_debug_pack_includes_probe_parquet_only_when_resolved_enabled(tmp_path: Path) -> None:
    run = tmp_path / "evaluation_run"
    probe_root = run / "reports" / "window_method_validation_probe"
    trace = probe_root / "measurement" / "run_000" / "collector_storage" / "fast_firmware.parquet"
    trace.parent.mkdir(parents=True)
    trace.write_bytes(b"raw-probe-trace")
    comparison = _write_json(
        probe_root / "measurement" / "run_000" / "window_method_comparison.json",
        {"status": "ok"},
    )
    _write_json(probe_root / "window_method_validation_probe.json", {
        "status": "screening_complete_non_decision_capable_too_few_repeats",
        "raw_parquet_debug_pack_requested": True,
    })
    _write_json(run / "reports" / "native_producer_stage.json", {
        "window_method_validation_probe": {
            "resolved_config": {"include_raw_parquet": True}
        }
    })
    _write_json(run / "run_manifest.json", {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "status": "partial",
    })
    out = tmp_path / "debug.zip"
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "create_evaluation_debug_pack.py"),
        "--eval-run-dir", str(run), "--out", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        assert trace.relative_to(run).as_posix() in names
        assert comparison.relative_to(run).as_posix() in names
        manifest = json.loads(archive.read("debug_pack_manifest.json"))
    block = manifest["window_method_validation_probe"]
    assert block["include_raw_parquet_resolved"] is True
    assert block["raw_parquet_count"] == 1
    assert block["all_probe_members_sha256_recorded"] is True
