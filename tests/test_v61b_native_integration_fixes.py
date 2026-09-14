from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from onnx_splitpoint_tool.native_energy_reporting import (
    build_native_energy_pairs,
    collect_native_energy,
    scientific_energy_rows,
)
from onnx_splitpoint_tool.native_progress import NativeProgressJournal, run_streaming
from onnx_splitpoint_tool.workflow.runner import (
    _native_expected_full_rows_v61b,
    _native_expected_matrix_status_v60y,
    _native_full_backends_by_producer_v61b,
)


ROOT = Path(__file__).resolve().parents[1]


def test_producer_local_full_matrix_is_not_flattened() -> None:
    cfg = {
        "enabled": True,
        "backends": ["hailo8", "hailo10h", "deepx", "tensorrt"],
        "backends_by_producer": {
            "hailo8": ["hailo8", "tensorrt", "deepx"],
            "hailo10h": ["hailo10h", "tensorrt", "hailo8"],
            "deepx": ["deepx", "tensorrt", "hailo10h"],
        },
    }
    resolved = _native_full_backends_by_producer_v61b(cfg, ["hailo8", "hailo10h", "deepx"])
    assert resolved == {
        "hailo8": ["hailo8", "tensorrt"],
        "hailo10h": ["hailo10h", "tensorrt"],
        "deepx": ["deepx", "tensorrt"],
    }


def test_expected_full_matrix_is_setup_local_and_has_twelve_rows() -> None:
    mapping = {
        "hailo8": ["hailo8", "tensorrt"],
        "hailo10h": ["hailo10h", "tensorrt"],
        "deepx": ["deepx", "tensorrt"],
    }
    setup_ids = {"hailo8": "h8", "hailo10h": "h10", "deepx": "dx"}
    rows = _native_expected_full_rows_v61b(["resnet50", "yolo26s"], mapping, setup_ids)
    assert len(rows) == 12
    assert {(r["setup_id"], r["backend"]) for r in rows} == {
        ("h8", "native_full_hailo8"), ("h8", "native_full_tensorrt"),
        ("h10", "native_full_hailo10h"), ("h10", "native_full_tensorrt"),
        ("dx", "native_full_deepx"), ("dx", "native_full_tensorrt"),
    }


def test_expected_matrix_does_not_reuse_trt_full_from_another_setup() -> None:
    expected = [
        {"backend_key": "hailo8", "backend": "native_full_tensorrt", "model": "resnet50", "case": "full", "setup_id": "h8", "execution_mode": "native_full_baseline"},
        {"backend_key": "deepx", "backend": "native_full_tensorrt", "model": "resnet50", "case": "full", "setup_id": "dx", "execution_mode": "native_full_baseline"},
    ]
    actual = [
        {"backend": "native_full_tensorrt", "model": "resnet50", "case": "full", "setup_id": "h8", "ok": True},
    ]
    status = _native_expected_matrix_status_v60y(expected, actual, [{"backend": "hailo8", "ok": True}, {"backend": "deepx", "ok": True}])
    assert status["present_expected_row_count"] == 1
    assert status["missing_expected_row_count"] == 1
    assert status["missing_expected_rows"][0]["setup_id"] == "dx"


def test_energy_helper_runs_from_foreign_cwd_and_rejects_legacy_unbound_rows(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    validation = tmp_path / "validation.json"
    out = tmp_path / "out"
    summary.write_text(json.dumps({"rows": [
        {"backend": "hailo8_to_trt", "model": "resnet50", "case": "b052", "precision": "fp16", "setup_id": "h8", "ok": True, "fps_makespan": 100},
            {"backend": "native_full_tensorrt", "model": "resnet50", "case": "full", "precision": "fp16", "setup_id": "h8", "comparison_backend": "hailo8", "execution_mode": "native_full_baseline", "ok": True, "fps_makespan": 150},
            {"backend": "native_full_tensorrt", "model": "resnet50", "case": "full", "precision": "fp16", "setup_id": "dx", "comparison_backend": "deepx", "execution_mode": "native_full_baseline", "ok": True, "fps_makespan": 145},
    ]}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": [
        {"backend": "hailo8_to_trt", "model": "resnet50", "case": "b052", "precision": "fp16", "task": "classification", "top1_match": True, "contract_consistent": True, "claim_ok": True, "semantic_ok": True},
    ]}), encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cp = subprocess.run([
        sys.executable, "-u", str(ROOT / "scripts" / "run_native_producer_energy_from_summary.py"),
        "--summary", str(summary), "--validation-summary", str(validation), "--out-dir", str(out),
        "--hailo8-ssh", "h8-host", "--deepx-ssh", "dx-host", "--dry-run", "--allow-unpaired", "--duration-s", "1", "--timeout", "10",
    ], cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert cp.returncode == 3, cp.stderr + cp.stdout
    plan = json.loads((out / "plan" / "native_producer_energy_plan.json").read_text(encoding="utf-8"))
    assert plan["rows"] == []
    assert len(plan["excluded_rows"]) == 3
    assert all(
        row.get("reason") == "successful_command_contract_missing_or_invalid"
        for row in plan["excluded_rows"]
    )


def test_parent_streaming_emits_heartbeat_for_silent_child(tmp_path: Path) -> None:
    lines: list[str] = []
    journal = NativeProgressJournal.from_dir(tmp_path)
    cp = run_streaming(
        [sys.executable, "-u", "-c", "import time; time.sleep(0.7); print('done', flush=True)"],
        timeout=5, label="silent", heartbeat_s=0.15, journal=journal, line_callback=lines.append,
    )
    assert cp.returncode == 0
    assert any("HEARTBEAT" in line for line in lines)
    events = [json.loads(line) for line in (tmp_path / "native_progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event.get("event") == "HEARTBEAT" for event in events)


def test_energy_stdout_tail_is_ingested_into_scientific_rows(tmp_path: Path) -> None:
    result_dir = tmp_path / "reports" / "native_energy_measurements"
    result_dir.mkdir(parents=True)
    energy_json = {
        "rows": [
            {
                "row": {"backend": "hailo8_to_trt", "model": "resnet50", "case": "b052", "precision": "fp16", "setup_id": "h8", "fps": 100, "duration_s": 2.0, "semantic_gate": "pass", "claim_ok": True, "contract_consistent": True, "task": "classification", "direction": "hailo8_to_trt", "contract_hash": "c", "preprocessing_hash": "p", "model_sha256": "m", "validation_input_or_image_sha256": "image", "prepared_feed_task": "classification", "prepared_feed_preprocess_mode": "resize", "prepared_feed_letterbox_pad_value": "0", "prepared_feed_source_image_sha256": "image"},
                "ok": True,
                "run": {"rc": 0, "stdout_tail": 'prefix "energy_total_j": 20.0, "active_duration_s": 2.0, "avg_power_w": 10.0, "energy_work_units_used": 200, "energy_per_work_unit_j": 0.1, "postprocess_status": "ok", "energy_physical_scope": "MB", "energy_window_effective": "command_window", "energy_efficiency_claim_eligible": true, "final_energy_gate_status": "pass", "energy_work_units_source": "runtime_completed_work_units", "runtime_completed_work_unit_run_count": 1, "valid_postprocessed_runs": 1, "energy_primary_metric": "raw_input_energy", "energy_raw_primary": true suffix'},
            },
            {
                "row": {"backend": "native_full_hailo8", "model": "resnet50", "case": "full", "precision": "fp16", "setup_id": "h8", "comparison_backend": "hailo8", "fps": 80, "duration_s": 2.0, "claim_ok": True, "contract_consistent": True, "task": "classification", "direction": "hailo8_to_trt", "contract_hash": "c", "preprocessing_hash": "p", "model_sha256": "m", "validation_input_or_image_sha256": "image", "prepared_feed_task": "classification", "prepared_feed_preprocess_mode": "resize", "prepared_feed_letterbox_pad_value": "0", "prepared_feed_source_image_sha256": "image"},
                "ok": True,
                "run": {"rc": 0, "stdout_tail": '"energy_total_j": 24.0, "active_duration_s": 2.0, "avg_power_w": 12.0, "energy_work_units_used": 160, "energy_per_work_unit_j": 0.15, "postprocess_status": "ok", "energy_physical_scope": "MB", "energy_window_effective": "command_window", "energy_efficiency_claim_eligible": true, "final_energy_gate_status": "pass", "energy_work_units_source": "runtime_completed_work_units", "runtime_completed_work_unit_run_count": 1, "valid_postprocessed_runs": 1, "energy_primary_metric": "raw_input_energy", "energy_raw_primary": true'},
            },
            {
                "row": {"backend": "native_full_tensorrt", "model": "resnet50", "case": "full", "precision": "fp16", "setup_id": "h8", "comparison_backend": "hailo8", "fps": 150, "duration_s": 2.0, "claim_ok": True, "contract_consistent": True, "task": "classification", "direction": "hailo8_to_trt", "contract_hash": "c", "preprocessing_hash": "p", "model_sha256": "m", "validation_input_or_image_sha256": "image", "prepared_feed_task": "classification", "prepared_feed_preprocess_mode": "resize", "prepared_feed_letterbox_pad_value": "0", "prepared_feed_source_image_sha256": "image"},
                "ok": True,
                "run": {"rc": 0, "stdout_tail": '"energy_total_j": 18.0, "active_duration_s": 2.0, "avg_power_w": 9.0, "energy_work_units_used": 300, "energy_per_work_unit_j": 0.06, "postprocess_status": "ok", "energy_physical_scope": "MB", "energy_window_effective": "command_window", "energy_efficiency_claim_eligible": true, "final_energy_gate_status": "pass", "energy_work_units_source": "runtime_completed_work_units", "runtime_completed_work_unit_run_count": 1, "valid_postprocessed_runs": 1, "energy_primary_metric": "raw_input_energy", "energy_raw_primary": true'},
            },
        ]
    }
    for index, item in enumerate(energy_json["rows"], start=1):
        row = item["row"]
        row.update({
            "source_request_sha256": str(index) * 64,
            "model_sha256": "4" * 64,
            "validation_dataset_sha256": "5" * 64,
            "validation_dataset_image_ids_sha256": "6" * 64,
            "validation_dataset_ground_truth_sha256": "7" * 64,
            "accuracy_gate_policy_sha256": "8" * 64,
            "task_quality_policy_sha256": "8" * 64,
            "runtime_quality_gate_policy_sha256": "8" * 64,
            "quality_contract_sha256": "9" * 64,
            "preprocessing_contract_sha256": "a" * 64,
            "validation_input_or_image_sha256": "b" * 64,
            "prepared_feed_source_image_sha256": "b" * 64,
                "central_quality_evidence_verified": True,
                "precision_quality_verified": True,
                "precision_quality_binding_verified": True,
                "task_quality_observation_valid": True,
                "accuracy_gate_pass": True,
                "quality_claim_result_verified": True,
            })
    (result_dir / "native_producer_energy_results.json").write_text(json.dumps(energy_json), encoding="utf-8")
    rows = collect_native_energy(tmp_path)
    assert len(rows) == 3
    assert rows[0]["energy_per_work_j"] == 0.1
    assert len(scientific_energy_rows(tmp_path)) == 3
    pairs = build_native_energy_pairs(rows)
    assert len(pairs) == 2
    vendor_pair = next(p for p in pairs if p["baseline_kind"] == "vendor_full")
    tensorrt_pair = next(p for p in pairs if p["baseline_kind"] == "tensorrt_full")
    assert vendor_pair["comparable"] is True
    assert tensorrt_pair["comparable"] is False
    assert "baseline_required_host_normalization_unavailable" in tensorrt_pair["comparison_reasons"]


def test_runner_source_uses_streaming_and_producer_specific_full_map() -> None:
    text = (ROOT / "onnx_splitpoint_tool" / "workflow" / "runner.py").read_text(encoding="utf-8")
    assert 'values = list(native_full_by_producer.get(producer, []))' in text
    assert 'values = [value for value in values if value != "tensorrt"]' in text
    assert 'label=f"full:{b}:{\',\'.join(fb_list)}"' in text
    assert 'label="energy:measure"' in text
    assert 'sys.executable, "-u", str(scripts_root / "run_native_producer_energy_from_summary.py")' in text
