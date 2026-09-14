from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _runner(tmp_path: Path, *, no_model_hash: bool) -> EvaluationWorkflowRunner:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.options = SimpleNamespace(no_model_hash=no_model_hash)
    runner.run_dir = tmp_path / "run"
    runner.manifest = {}
    runner._task_for = lambda _row: "detection"  # type: ignore[method-assign]
    runner._validation_preset = lambda _row: "coco500"  # type: ignore[method-assign]
    return runner


def _row(model: Path, declared: str | None) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "yolov7_paper",
        "resolved_path": str(model),
        "family": "yolov7",
    }
    if declared is not None:
        row["model_sha256"] = declared
    return row


def _manifest(artifacts: dict[str, Path]) -> dict[str, object]:
    return json.loads(artifacts["model_manifest_json"].read_text(encoding="utf-8"))


def test_declared_model_sha256_is_verified_even_with_no_model_hash(
    tmp_path: Path,
) -> None:
    model = tmp_path / "yolov7_paper.onnx"
    model.write_bytes(b"exact-model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    runner = _runner(tmp_path, no_model_hash=True)

    artifacts, metrics, message, status = runner._stage_resolve_model(
        "yolov7_paper", _row(model, "sha256:" + digest.upper()),
    )

    payload = _manifest(dict(artifacts))
    assert status == "ok"
    assert metrics["model_identity_status"] == "verified"
    assert metrics["model_identity_error_code"] == ""
    assert payload["model_sha256"] == "sha256:" + digest
    assert payload["declared_model_sha256"] == digest
    assert payload["observed_model_sha256"] == digest
    assert payload["model_identity_status"] == "verified"
    assert "verified" in message


def test_declared_model_sha256_mismatch_fails_resolve_model(tmp_path: Path) -> None:
    model = tmp_path / "yolov7_paper.onnx"
    model.write_bytes(b"wrong-export")
    observed = hashlib.sha256(model.read_bytes()).hexdigest()
    expected = "7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d"
    runner = _runner(tmp_path, no_model_hash=True)

    artifacts, metrics, message, status = runner._stage_resolve_model(
        "yolov7_paper", _row(model, expected),
    )

    payload = _manifest(dict(artifacts))
    assert status == "failed"
    assert metrics["model_identity_error_code"] == "declared_model_sha256_mismatch"
    assert payload["declared_model_sha256"] == expected
    assert payload["model_sha256"] == "sha256:" + observed
    assert payload["observed_model_sha256"] == observed
    assert payload["model_identity_status"] == "failed"
    assert f"expected={expected}" in message
    assert f"observed={observed}" in message


def test_invalid_declared_model_sha256_fails_closed(tmp_path: Path) -> None:
    model = tmp_path / "yolov7_paper.onnx"
    model.write_bytes(b"model")
    runner = _runner(tmp_path, no_model_hash=False)

    _artifacts, metrics, message, status = runner._stage_resolve_model(
        "yolov7_paper", _row(model, "sha256:not-a-digest"),
    )

    assert status == "failed"
    assert metrics["model_identity_error_code"] == "declared_model_sha256_invalid"
    assert message.startswith("declared_model_sha256_invalid:")


def test_missing_pinned_model_fails_instead_of_partial(tmp_path: Path) -> None:
    model = tmp_path / "missing.onnx"
    runner = _runner(tmp_path, no_model_hash=True)
    expected = "7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d"

    _artifacts, metrics, message, status = runner._stage_resolve_model(
        "yolov7_paper", _row(model, expected),
    )

    assert status == "failed"
    assert metrics["model_identity_error_code"] == (
        "declared_model_sha256_unverifiable_model_missing"
    )
    assert "observed=<unavailable>" in message


def test_legacy_unpinned_no_hash_behavior_is_unchanged(tmp_path: Path) -> None:
    model = tmp_path / "legacy.onnx"
    model.write_bytes(b"legacy")
    runner = _runner(tmp_path, no_model_hash=True)

    artifacts, metrics, message, status = runner._stage_resolve_model(
        "legacy_model", _row(model, None),
    )

    payload = _manifest(dict(artifacts))
    assert status == "ok"
    assert metrics["model_identity_status"] == "not_declared"
    assert metrics["model_identity_error_code"] == ""
    assert payload["model_sha256"] == ""
    assert payload["declared_model_sha256"] == ""
    assert message == "Model path resolved."
