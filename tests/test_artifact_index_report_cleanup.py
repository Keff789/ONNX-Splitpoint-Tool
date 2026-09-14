from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.artifacts import file_record
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
import onnx_splitpoint_tool.workflow.runner as runner_module


def _write(path: Path, text: str = "data\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _runner(run_dir: Path) -> EvaluationWorkflowRunner:
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = "report-cleanup-fixture"
    runner.session_id = "1" * 32
    runner.artifact_index_path = run_dir / "artifact_index.json"
    runner.artifact_index = {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 2,
        "run_id": runner.run_id,
        "artifacts": [],
    }
    runner.outputs = {}
    runner.report_paths = []
    runner.manifest = {"models": {}}
    runner.profile_payload = {}
    runner.profile_id = "fixture"
    return runner


def _register(runner: EvaluationWorkflowRunner, *paths: Path) -> None:
    runner.artifact_index["artifacts"] = [
        file_record(
            path,
            root=runner.run_dir,
            kind="stage_artifact",
            producer_stage="aggregate_results",
        )
        for path in paths
    ]
    runner._save_artifact_index()


def test_cleanup_retires_only_owned_missing_presentation_artifacts(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path / "run")
    prediction = _write(
        runner.run_dir / "reports" / "prediction_vs_benchmark.csv"
    )
    validation = _write(
        runner.run_dir / "reports" / "validation_summary.csv"
    )
    old_scientific = _write(
        runner.run_dir / "reports" / "scientific" / "old_report.csv"
    )
    native_energy = _write(
        runner.run_dir / "reports" / "native_energy_preflight.json"
    )
    unrelated = _write(runner.run_dir / "models" / "m" / "result.json")
    _register(
        runner, prediction, validation, old_scientific, native_energy,
        unrelated,
    )

    prediction.unlink()
    validation.unlink()
    old_scientific.unlink()
    native_energy.unlink()
    unrelated.unlink()

    retired = runner._reconcile_canonical_report_cleanup()

    assert retired == [
        "reports/prediction_vs_benchmark.csv",
        "reports/scientific/old_report.csv",
        "reports/validation_summary.csv",
    ]
    indexed = {
        row["path"] for row in runner.artifact_index["artifacts"]
    }
    assert indexed == {
        "models/m/result.json",
        "reports/native_energy_preflight.json",
    }
    with pytest.raises(
        RuntimeError,
        match="artifact_index_registered_file_missing:models/m/result.json",
    ):
        runner._finalize_artifact_index(status="ok")


def test_reporter_must_not_advertise_a_missing_replacement(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path / "run")
    with pytest.raises(
        RuntimeError,
        match="canonical_report_artifact_missing_or_external:missing",
    ):
        runner._validated_canonical_report_artifacts({
            "artifacts": {
                "missing": runner.run_dir / "reports" / "scientific"
                / "missing.json",
            },
        })


def test_generate_report_reconciles_deleted_aggregate_csv_before_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path / "run")
    prediction = _write(
        runner.run_dir / "reports" / "prediction_vs_benchmark.csv",
        "model_id,status\nm,ok\n",
    )
    _register(runner, prediction)
    runner.report_paths = [prediction]
    runner.outputs = {"prediction_vs_benchmark_csv": str(prediction)}

    def fake_scientific_reports(run_dir: Path, **_kwargs: object) -> dict:
        prediction.unlink()
        scientific = _write(
            Path(run_dir) / "reports" / "scientific" / "scientific_report.md",
            "# Canonical report\n",
        )
        return {
            "artifacts": {"scientific_report_md": scientific},
            "model_count": 0,
            "scientific_row_count": 0,
            "ranking_method_comparison_rows": 0,
            "figure_count": 0,
        }

    monkeypatch.setattr(
        runner_module, "build_scientific_reports", fake_scientific_reports
    )

    artifacts, _metrics, _message, status = runner._stage_generate_report()
    assert status == "ok"
    runner._register_artifacts(
        list(artifacts.values()),
        kind="stage_artifact",
        producer_stage="generate_report",
    )

    assert not prediction.exists()
    assert "reports/prediction_vs_benchmark.csv" not in {
        row["path"] for row in runner.artifact_index["artifacts"]
    }
    bundle = json.loads(
        (runner.run_dir / "reports" / "results_bundle_manifest.json")
        .read_text(encoding="utf-8")
    )
    assert (
        bundle["retired_outputs_not_expected"][
            "prediction_vs_benchmark_csv"
        ]
        == "reports/prediction_vs_benchmark.csv"
    )

    runner._finalize_artifact_index(status="ok")
    terminal = json.loads(
        runner.artifact_index_path.read_text(encoding="utf-8")
    )
    assert terminal["terminal_closure"]["status"] == "pass"
