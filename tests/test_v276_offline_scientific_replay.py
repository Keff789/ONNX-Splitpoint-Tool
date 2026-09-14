from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.workflow.analysis_pack import (
    CANONICAL_RESULT_FILES,
    CANONICAL_TABLE_FILES,
)
from onnx_splitpoint_tool.workflow.scientific_replay import (
    replay_scientific_reports,
)


ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = ROOT / "scripts" / "replay_scientific_reports.py"


def _write_evalrun(run_dir: Path) -> None:
    reports = run_dir / "reports"
    model_results = (
        run_dir / "models" / "resnet50" / "benchmark_results"
    )
    model_results.mkdir(parents=True)
    model_analysis = run_dir / "models" / "resnet50" / "analysis"
    model_analysis.mkdir(parents=True)
    benchmark_set = run_dir / "models" / "resnet50" / "benchmark_set"
    benchmark_set.mkdir(parents=True)
    reports.mkdir(parents=True)
    quality_management = run_dir / "quality_management"
    quality_management.mkdir(parents=True)
    profile = {
        "name": "offline-replay-fixture",
        "campaign": {"mode": "development"},
        "model_suite": {
            "primary": [{
                "id": "resnet50",
                "task": "classification",
                "evaluation_role": "development",
            }],
            "reserve": [],
        },
        "quality_gate": {
            "dataset_tier": "screening",
            "frozen_before_final_campaign": False,
        },
    }
    (run_dir / "profile.yaml").write_text(
        yaml.safe_dump(profile), encoding="utf-8",
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": run_dir.name, "status": "ok"}),
        encoding="utf-8",
    )
    (reports / "run_status_summary.json").write_text(
        json.dumps({"technical_status": "ok"}), encoding="utf-8",
    )
    (run_dir / "effective_execution_plan.json").write_text(
        json.dumps({
            "models": ["resnet50"],
            "effective_generic_run_ids": ["hailo8_to_trt"],
            "expected_generic_result_rows_total": 1,
        }),
        encoding="utf-8",
    )
    (model_analysis / "final_candidate_plan.json").write_text(
        json.dumps({"selected_candidates": [{"case_id": "b001"}]}),
        encoding="utf-8",
    )
    (benchmark_set / "generation_decisions.json").write_text(
        json.dumps({
            "accepted_cases": [{
                "case_id": "b001",
                "generation_status": "accepted_split_exported",
            }],
            "rejected_cases": [],
        }),
        encoding="utf-8",
    )
    central_results = []
    for variant, backend, case_id, candidate in (
        ("full", "hailo8", "full", 0.79),
        ("composed", "hailo8_to_trt", "b001", 0.78),
    ):
        central_results.append({
            "model_id": "resnet50",
            "case_id": case_id,
            "variant": variant,
            "backend": backend,
            "source_run_id": backend,
            "setup_id": "orin_nx_hailo8_01",
            "task": "classification",
            "technical_status": "completed",
            "decision": "pass",
            "reference_identity": "reference-resnet50",
            "validation_dataset_sha256": "a" * 64,
            "preprocessing_contract_sha256": "b" * 64,
            "task_quality_policy_sha256": "c" * 64,
            "primary": {
                "metric": "top1_accuracy",
                "candidate": candidate,
                "reference": 0.80,
                "delta": candidate - 0.80,
                "decision": "pass",
            },
            "guardrails": {
                "top5_accuracy": {
                    "metric": "top5_accuracy",
                    "candidate": 0.94,
                    "reference": 0.95,
                    "delta": -0.01,
                    "decision": "pass",
                },
            },
            "metric_gate_config": {
                "primary_metric": "top1_accuracy",
                "guardrails": {"top5_accuracy_margin": 0.01},
            },
        })
    (quality_management / "central_quality_summary.json").write_text(
        json.dumps({
            "status": "ok",
            "request_count": 2,
            "results": central_results,
        }),
        encoding="utf-8",
    )
    source_row = {
        "model_id": "resnet50",
        "case_id": "b001",
        "backend": "hailo8_to_tensorrt",
        "direction": "hailo8_to_tensorrt",
        "variant": "composed",
        "task": "classification",
        "buildable": True,
        "runtime_executable": True,
        "contract_consistent": True,
        "pipeline_cycle_selected_ms": 10.0,
        "throughput_primary_fps": 100.0,
    }
    (model_results / "normalized_results.json").write_text(
        json.dumps({"results": [source_row]}), encoding="utf-8",
    )

    # These source presentation files prove that external replay does not run
    # cleanup or compatibility writers against the original EvaluationRun.
    scientific = reports / "scientific"
    scientific.mkdir()
    (scientific / "source-sentinel.txt").write_text(
        "original scientific output\n", encoding="utf-8",
    )
    (reports / "summary.csv").write_text(
        "original compatibility output\n", encoding="utf-8",
    )


def _snapshot_files(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_offline_replay_writes_separate_output_without_mutating_source(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "evaluation_run"
    output_dir = tmp_path / "reports_reprojected_v276"
    _write_evalrun(run_dir)
    before = _snapshot_files(run_dir)

    result = replay_scientific_reports(
        run_dir,
        output_dir,
        tool_version="test",
        workflow_version="test-replay",
    )

    assert result["status"] == "ok"
    assert result["evidence_completeness_status"] == "complete"
    assert result["measurement_completeness_status"] == "complete"
    assert result["source_mutated"] is False
    assert result["execution_mode"] == "offline_read_only_report_replay"
    assert Path(result["output_dir"]) == output_dir
    assert (output_dir / "scientific_report.json").is_file()
    scientific = json.loads(
        (output_dir / "scientific_report.json").read_text(encoding="utf-8")
    )
    assert "cross_runner_identity_diagnostics" in scientific
    assert "identity_exclusions" in scientific[
        "cross_runner_identity_diagnostics"
    ]
    assert (output_dir / "performance_observations.csv").is_file()
    assert (output_dir / "performance_observations.json").is_file()
    assert (output_dir / "endpoint_lifecycle_ledger.csv").is_file()
    assert (output_dir / "endpoint_lifecycle_ledger.json").is_file()
    lifecycle = json.loads(
        (output_dir / "endpoint_lifecycle_summary.json").read_text(
            encoding="utf-8",
        )
    )
    assert lifecycle["planned"] == 1
    assert lifecycle["measured"] == 1
    assert lifecycle["terminal"] == 1
    assert len(json.loads(
        (output_dir / "task_quality_reference_comparison.json").read_text(
            encoding="utf-8",
        )
    )) == 2
    assert len(json.loads(
        (output_dir / "task_quality_loss_decomposition.json").read_text(
            encoding="utf-8",
        )
    )) == 1
    manifest_paths = {
        row["path"]
        for row in json.loads(
            (output_dir / "report_manifest.json").read_text(encoding="utf-8")
        )["artifacts"]
    }
    assert "task_quality_reference_comparison.json" in manifest_paths
    assert "task_quality_loss_decomposition.json" in manifest_paths
    assert (
        output_dir / "thesis_tables" / "performance_observations.tex"
    ).is_file()
    assert _snapshot_files(run_dir) == before


def test_offline_replay_rejects_source_local_and_existing_destinations(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "evaluation_run"
    _write_evalrun(run_dir)

    with pytest.raises(RuntimeError, match="outside the read-only source"):
        replay_scientific_reports(
            run_dir, run_dir / "reports_reprojected_v276",
        )

    existing = tmp_path / "already_there"
    existing.mkdir()
    (existing / "sentinel.txt").write_text("keep\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="destination already exists"):
        replay_scientific_reports(run_dir, existing)
    assert (existing / "sentinel.txt").read_text(encoding="utf-8") == "keep\n"


def test_offline_replay_cli_publishes_the_separate_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "evaluation_run"
    output_dir = tmp_path / "cli_reprojected"
    _write_evalrun(run_dir)
    before = _snapshot_files(run_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPLAY_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["ok"] is True
    assert status["measurement_completeness_status"] == "complete"
    assert status["source_mutated"] is False
    assert Path(status["output_dir"]) == output_dir
    assert (output_dir / "scientific_report.json").is_file()
    assert _snapshot_files(run_dir) == before


def test_offline_replay_binds_separate_native_source_read_only(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "generic_run"
    native_dir = tmp_path / "native_supplement"
    output_dir = tmp_path / "bound_reprojected"
    _write_evalrun(run_dir)
    (native_dir / "reports").mkdir(parents=True)
    (native_dir / "reports" / "native-sentinel.txt").write_text(
        "native source remains untouched\n", encoding="utf-8",
    )
    generic_before = _snapshot_files(run_dir)
    native_before = _snapshot_files(native_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPLAY_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--native-run-dir",
            str(native_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["native_source_is_separate"] is True
    assert Path(status["native_source_run_dir"]) == native_dir
    report = json.loads(
        (output_dir / "scientific_report.json").read_text(encoding="utf-8")
    )
    assert report["native_source_binding"] == {
        "schema": "onnx-splitpoint/native-replay-source-binding",
        "schema_version": 1,
        "generic_run_dir": str(run_dir),
        "native_run_dir": str(native_dir),
        "native_source_is_separate": True,
        "native_run_id": native_dir.name,
        "binding_status": "explicit",
    }
    assert _snapshot_files(run_dir) == generic_before
    assert _snapshot_files(native_dir) == native_before


def test_analysis_pack_includes_complete_performance_observation_surface() -> None:
    assert "performance_observations.csv" in CANONICAL_RESULT_FILES
    assert "performance_observations.json" in CANONICAL_RESULT_FILES
    assert "performance_observations.tex" in CANONICAL_TABLE_FILES
    assert "performance_cohorts.csv" in CANONICAL_RESULT_FILES
    assert "endpoint_lifecycle_ledger.json" in CANONICAL_RESULT_FILES
    assert "ranking_method_cohort_sensitivity.csv" in CANONICAL_RESULT_FILES
    assert "ranking_method_cohort_sensitivity.json" in CANONICAL_RESULT_FILES
    assert "ranking_method_cohort_sensitivity.tex" in CANONICAL_TABLE_FILES
