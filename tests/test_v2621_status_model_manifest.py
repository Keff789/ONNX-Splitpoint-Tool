from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _runner(tmp_path: Path, *, no_model_hash: bool = False) -> EvaluationWorkflowRunner:
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="unused", out=str(tmp_path), no_model_hash=no_model_hash)
    )
    runner.run_id = "v2621_regression"
    runner.profile_id = "test-profile"
    runner.run_dir = tmp_path / runner.run_id
    runner.run_dir.mkdir(parents=True, exist_ok=True)
    runner.manifest = {"models": {}}
    runner.outputs = {}
    runner.report_paths = []
    runner.warnings = []
    runner.run_log_path = runner.run_dir / "evaluation_workflow.log"
    runner.parent_log_path = None
    return runner


@pytest.mark.parametrize("no_model_hash", [False, True])
def test_model_manifest_mirrors_top_level_fingerprint(
    tmp_path: Path, no_model_hash: bool,
) -> None:
    runner = _runner(tmp_path, no_model_hash=no_model_hash)
    model = tmp_path / "model.onnx"
    payload_bytes = b"v2621-model-bytes"
    model.write_bytes(payload_bytes)

    paths, _metrics, _message, status = runner._stage_resolve_model(
        "resnet50", {"resolved_path": str(model), "task": "classification"}
    )
    assert status == "ok"
    payload = json.loads(paths["model_manifest_json"].read_text(encoding="utf-8"))
    expected_sha = (
        "" if no_model_hash
        else "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
    )
    assert payload["model_sha256"] == payload["file"]["sha256"] == expected_sha
    assert payload["model_size_bytes"] == payload["file"]["size_bytes"] == len(payload_bytes)
    recorded = runner.manifest["models"]["resnet50"]
    assert recorded["model_sha256"] == expected_sha
    assert recorded["model_size_bytes"] == len(payload_bytes)


def test_unresolved_model_has_explicit_empty_top_level_identity(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    paths, _metrics, _message, status = runner._stage_resolve_model("missing", {})
    assert status == "partial"
    payload = json.loads(paths["model_manifest_json"].read_text(encoding="utf-8"))
    assert payload["model_sha256"] == payload["file"]["sha256"] == ""
    assert payload["model_size_bytes"] is None
    assert payload["file"]["size_bytes"] is None
    assert runner.manifest["models"]["missing"]["model_size_bytes"] is None


def test_unrelated_log_tokens_do_not_invent_missing_full_onnx_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path)
    runner.run_log_path.write_text(
        "FileNotFoundError: workload_timing.txt\n"
        "normal build input model.onnx\n"
        "valid --onnx=../models/yolo26s.onnx\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_derive_final_status",
        lambda: ("ok", {"stage_status_counts": {}, "blocking_reasons": [], "non_blocking_reasons": []}),
    )
    monkeypatch.setattr(runner, "_register_artifacts", lambda *_args, **_kwargs: None)
    runner._write_run_status_summary("ok")
    payload = json.loads(
        (runner.run_dir / "reports" / "run_status_summary.json").read_text(encoding="utf-8")
    )
    reasons = [str(row.get("reason") or "") for row in payload["blocking_reasons"]]
    assert not any("suite-root full ONNX was missing" in reason for reason in reasons)
