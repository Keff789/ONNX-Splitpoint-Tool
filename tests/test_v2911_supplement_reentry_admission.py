"""Resume publication safety and completed work before aggregate reporting."""
import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool import hailo_backend, hailo_cache_bundle
from onnx_splitpoint_tool.workflow.runner import WorkflowRunTargetError
from onnx_splitpoint_tool.workflow.supplement import build_supplement_plan, completed_quality
from tests.test_v27924_terminal_hailo_aliases import published
from tests.test_v2911_supplement_workflow import archive, write


def test_resume_admits_published_source_aliases_without_reading_hefs(published, monkeypatch):
    runner, alias, hef = published
    before = [(path, path.lstat()) for path in alias.parent.rglob("*")]
    monkeypatch.setattr(hailo_backend, "_load_valid_hailo_receipt",
                        lambda *a, **kw: pytest.fail("path admission must not rehash archived HEFs"))
    runner._validate_resume_target_paths()
    for path, stat in before:
        assert path.lstat() == stat


@pytest.mark.parametrize("damage", ["external", "pointer", "member", "missing_receipt",
                                    "control", "output", "broken_root", "broken_control"])
def test_resume_rejects_escape_and_mutable_links(published, tmp_path, damage):
    runner, alias, hef = published
    if damage == "external":
        alias.unlink(); alias.symlink_to(hef)
    elif damage == "pointer":
        pointer = alias.parent / hailo_cache_bundle.POINTER_NAME
        pointer.unlink(); pointer.symlink_to(hef.parent)
    elif damage == "member":
        hef.unlink(); hef.symlink_to(tmp_path / "outside")
    elif damage == "missing_receipt":
        (hef.parent / hailo_cache_bundle.RECEIPT_NAME).unlink()
    elif damage in {"control", "broken_control"}:
        (runner.run_dir / "run_manifest.json").symlink_to(
            hef if damage == "control" else tmp_path / "missing")
    elif damage == "output":
        output = runner.run_dir / "models/mobilenet_v3_large/benchmark_results/compiled.hef"
        output.parent.mkdir(); output.symlink_to(alias)
    else:
        (runner.run_dir / "jobs").symlink_to(tmp_path / "missing")
    with pytest.raises(WorkflowRunTargetError, match="symlink/non-regular"):
        runner._validate_resume_target_paths()


def _new_normalized(archive):
    plan = build_supplement_plan(archive)
    row = copy.deepcopy(plan["generic_rows"][1])
    raw = write(archive / "models/yolo26m/benchmark_results/attempts/attempt/new_result.json", {"fps": 25})
    row.update(source_path=str(raw), source_row_sha256="a" * 64,
               logical_measurement_id="b" * 64, logical_measurement_primary=True,
               required_scope_binding_status="exact_physical_match", tool_version="2.91.1")
    normalized = {"schema": "onnx-splitpoint/normalized-benchmark-results", "model_id": "yolo26m",
                  "evaluation_run_id": archive.name, "results": [row]}
    path = archive / "models/yolo26m/benchmark_results/normalized_results.json"
    write(path, normalized)
    return path, normalized


def test_second_resume_preserves_new_normalized_row_before_scientific_report(archive):
    report = archive / "reports/scientific/performance_results.json"
    original = report.read_bytes()
    path, normalized = _new_normalized(archive)
    plan = build_supplement_plan(archive)
    assert plan["counts"]["generic"] == {"preserve": 2, "negative_completed": 1}
    assert report.read_bytes() == original
    assert json.loads(path.read_text()) == normalized


@pytest.mark.parametrize("damage", ["scope", "source", "duplicate", "run"])
def test_second_resume_rejects_unbound_normalized_observation(archive, damage):
    path, payload = _new_normalized(archive)
    row = payload["results"][0]
    if damage == "scope":
        row["logical_identity_sha256"] = "other"
    elif damage == "source":
        row["source_path"] = str(archive.parent / "outside")
    elif damage == "duplicate":
        payload["results"].append(copy.deepcopy(row))
    else:
        payload["evaluation_run_id"] = "other"
    write(path, payload)
    with pytest.raises(ValueError, match="supplement_normalized_results_"):
        build_supplement_plan(archive)


def _local_quality(archive, *, include_summary=False):
    summary = archive / "quality_management/central_quality_summary.json"
    row = json.loads(summary.read_text())["results"][0]
    request = archive / row["source_request"]
    local = request.with_name(request.stem.replace("_request", "") + "_central_result.json")
    write(local, row)
    if not include_summary:
        write(summary, {"results": []})
    return local, row


@pytest.mark.parametrize("include_summary", [False, True])
def test_completed_request_local_result_survives_aggregate_report_interruption(archive, include_summary):
    local, expected = _local_quality(archive, include_summary=include_summary)
    results, requests = completed_quality(archive)
    assert results == [expected]
    assert len(requests) == 1
    assert results[0]["decision"] == "fail"
    assert results[0]["tool_version"] == "2.91.0"


@pytest.mark.parametrize("damage", ["digest", "request_path", "identity", "decision", "symlink"])
def test_local_quality_reentry_rejects_conflicting_or_redirected_evidence(archive, damage):
    local, row = _local_quality(archive, include_summary=True)
    if damage == "digest":
        row["source_request_sha256"] = "f" * 64
    elif damage == "request_path":
        row["source_request"] = "models/yolo26m/another_request.json"
    elif damage == "identity":
        row["model_id"] = "other"
    elif damage == "decision":
        row["decision"] = "pass"
    else:
        local.unlink(); local.symlink_to(archive / "quality_management/central_quality_summary.json")
    if damage != "symlink":
        write(local, row)
    with pytest.raises(ValueError, match="supplement_"):
        completed_quality(archive)
