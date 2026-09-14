from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.artifacts import file_record, sha256_file
from onnx_splitpoint_tool.workflow.deepx_build_binding import (
    _mirror_deepx_cache_receipt,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _runner(run_dir: Path) -> EvaluationWorkflowRunner:
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = EvaluationWorkflowRunner.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = run_dir.name
    runner.session_id = "7" * 32
    runner.artifact_index_path = run_dir / "artifact_index.json"
    runner.artifact_index = {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 1,
        "run_id": run_dir.name,
        "artifacts": [],
    }
    runner.outputs = {}
    runner.report_paths = []
    return runner


def test_deepx_external_cache_receipt_is_mirrored_byte_exact_locally(
    tmp_path: Path,
) -> None:
    external = tmp_path / "cache" / "build_manifest.json"
    external.parent.mkdir()
    external.write_bytes(b'{"schema":"deepx-cache-receipt","value":7}\n')
    local_dir = tmp_path / "run" / "models" / "yolo11l" / "deepx" / "full"

    mirrored = _mirror_deepx_cache_receipt(external, local_dir=local_dir)

    assert mirrored == local_dir / "deepx_cache_receipt.json"
    assert mirrored.read_bytes() == external.read_bytes()
    assert mirrored.resolve().is_relative_to((tmp_path / "run").resolve())
    assert mirrored.resolve() != external.resolve()


def test_artifact_registration_is_one_current_record_per_logical_path(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path / "run")
    first = runner.run_dir / "reports" / "z.json"
    second = runner.run_dir / "reports" / "a.json"
    _write_json(first, {"generation": 1})
    _write_json(second, {"stable": True})

    runner._register_artifacts(
        [first, second], kind="stage_artifact", producer_stage="first", model_id="old"
    )
    _write_json(first, {"generation": 2})
    runner._register_artifacts(
        [first], kind="report", producer_stage="latest", model_id="current"
    )

    rows = runner.artifact_index["artifacts"]
    assert [row["path"] for row in rows] == ["reports/a.json", "reports/z.json"]
    assert len(rows) == 2
    current = rows[1]
    assert current["kind"] == "report"
    assert current["producer_stage"] == "latest"
    assert current["model_id"] == "current"
    assert current["sha256"] == sha256_file(first)
    assert current["size_bytes"] == first.stat().st_size


def test_artifact_registration_rejects_external_file(tmp_path: Path) -> None:
    runner = _runner(tmp_path / "run")
    external = tmp_path / "cache" / "receipt.json"
    _write_json(external, {"status": "verified"})

    with pytest.raises(RuntimeError, match="artifact_registration_external_path"):
        runner._register_artifacts(
            [external], kind="cache_receipt", producer_stage="deepx_build"
        )

    assert runner.artifact_index["artifacts"] == []


def test_terminal_finalization_collapses_history_last_metadata_wins(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path / "run")
    report = runner.run_dir / "reports" / "summary.json"
    _write_json(report, {"status": "ok"})
    old = file_record(
        report, root=runner.run_dir, kind="stage_artifact", producer_stage="old"
    )
    latest = file_record(
        report, root=runner.run_dir, kind="report", producer_stage="latest"
    )
    runner.artifact_index["artifacts"] = [old, latest]

    runner._finalize_artifact_index(status="ok")

    payload = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    rows = [row for row in payload["artifacts"] if row["path"] == "reports/summary.json"]
    assert len(rows) == 1
    assert rows[0]["kind"] == "report"
    assert rows[0]["producer_stage"] == "latest"
    assert payload["terminal_closure"]["status"] == "pass"


def test_internal_terminal_verifier_rejects_duplicate_committed_path(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path / "run")
    report = runner.run_dir / "reports" / "summary.json"
    _write_json(report, {"status": "ok"})
    runner._finalize_artifact_index(status="ok")
    payload = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    row = next(
        row for row in payload["artifacts"] if row["path"] == "reports/summary.json"
    )
    payload["artifacts"].append(dict(row))
    _write_json(runner.artifact_index_path, payload)

    verification = runner._verify_terminal_artifact_index(
        required_paths=runner._terminal_evidence_candidates()
    )

    assert verification["ok"] is False
    assert {item["error"] for item in verification["errors"]} >= {
        "duplicate_path",
        "terminal_closure_artifact_record_count_mismatch",
        "closure_report_artifact_record_count_mismatch",
    }


@pytest.mark.parametrize("logical", ["/tmp/external.json", "../external.json"])
def test_terminal_finalization_rejects_external_or_escaping_record(
    tmp_path: Path,
    logical: str,
) -> None:
    runner = _runner(tmp_path / "run")
    runner.artifact_index["artifacts"] = [{"path": logical}]

    with pytest.raises(RuntimeError, match="artifact_index_"):
        runner._finalize_artifact_index(status="ok")


def test_terminal_finalization_and_verifier_reject_unknown_schema_version(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path / "run")
    runner.artifact_index["schema_version"] = 3
    with pytest.raises(RuntimeError, match="artifact_index_schema_version_unsupported"):
        runner._finalize_artifact_index(status="ok")

    report = runner.run_dir / "reports" / "summary.json"
    _write_json(report, {"status": "ok"})
    runner.artifact_index["schema_version"] = 1
    runner._finalize_artifact_index(status="ok")
    payload = json.loads(runner.artifact_index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    payload["schema_version"] = 3
    _write_json(runner.artifact_index_path, payload)
    verification = runner._verify_terminal_artifact_index(
        required_paths=runner._terminal_evidence_candidates()
    )
    assert verification["ok"] is False
    assert "index_schema_version_invalid" in {
        item["error"] for item in verification["errors"]
    }
